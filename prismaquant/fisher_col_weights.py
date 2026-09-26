"""Per-column KL-Fisher weights for codeword/scale search (nvfp4-cb exp 4).

The GGUF/IQ and nvfp4-cb encoders weight their weighted-LS codeword and scale
search by a per-input-column importance vector (``col_weights`` / ``qw`` slot;
gguf_formats._imatrix_weights, l.180). llama.cpp fills that slot with the
**imatrix** ``E[x^2]`` (mean-squared activation per column) — the only signal it
can compute with no backward pass. This module fills it instead with a
**KL-Fisher per-column energy**::

    fisher_col[j] = mean_probes  Σ_out  ( ∂ probe / ∂W )[:, j] ^2

the same Rademacher-probe weight-grad energy ``aura_cost`` already reduces to the
scalar ``h_trace``, reduced over output rows to a length-``in_features`` vector
instead of fully summed (so ``fisher_col.sum() == h_trace``). This targets the
columns whose error most moves the model's output KL — a Gauss-Newton Fisher
quantity — which ``E[x^2]`` only proxies.

Two compositions are exposed for the exp-4 A/B (both land in the same
``col_weights`` slot and are further composed by the encoder's own
``qw · sqrt(σ²_w + w²)`` weighting):

  * ``fisher_only(v)``   — the Fisher vector alone (normalized). The grad energy
    already carries one factor of the column's activation energy
    (∂L/∂W_ij = grad_out_i · x_j), so this is the composition that does **not**
    re-multiply by ``E[x^2]``.
  * ``fisher_x_act(v, sigma2, xbar2)`` — the llama.cpp-style composition with the
    Fisher vector standing in for the imatrix term:
    ``v · sqrt(sigma2 + xbar2)`` (mirrors ``_imatrix_weights``). This *does*
    fold in the activation imatrix ``xbar2 = E[x^2]`` again; it therefore
    double-counts activation energy on purpose — the A/B decides whether that
    extra activation weighting helps or hurts. Do not treat it as the default.

Reuses ``aura_cost.compute_aura_cost(collect_col_energy=True)`` for the harvest;
no new backward machinery.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Sequence

import torch


def _log(msg: str) -> None:
    print(f"[fisher-col {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_commit() -> str | None:
    from prismaquant.aura_cost import _git_head_commit

    return _git_head_commit()


def _dataset_sha256(dataset: str | None) -> str | None:
    """Content hash of the calibration source when it is a readable file,
    else a hash of the identifier string (HF dataset id / literal)."""
    if not dataset:
        return None
    p = Path(dataset)
    try:
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
    except Exception:
        pass
    return hashlib.sha256(str(dataset).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Compositions (the two exp-4 sub-arms). Both take a per-column Fisher vector
# (shape (in_features,) or any shape broadcastable to the weight, e.g.
# (N, 1, in) for a stacked/expert batch) and return a vector of the same shape
# that drops straight into the encoder's ``col_weights`` / ``qw`` slot
# (gguf_formats._qw_blocks). Normalization is over the last (input) axis.
# ---------------------------------------------------------------------------

def _normalize(v: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Scale each row's importance vector to mean 1 over the input axis.

    Weighted-LS codeword/scale search is invariant to a global rescaling of the
    weights, so only the *relative* per-column profile matters; mean-1 puts
    Fisher and imatrix vectors on the same comparable scale. Dead/negative
    entries are floored to a tiny positive value so the encoder always receives
    a strictly-positive, finite ``col_weights`` (its own ``_guard_dead_subblocks``
    handles the residual zero case, but downstream consumers expect positivity).
    """
    v = v.to(torch.float32).clamp_min(0.0)
    mean = v.mean(dim=-1, keepdim=True).clamp_min(eps)
    out = v / mean
    return out.clamp_min(eps)


def fisher_only(v: torch.Tensor) -> torch.Tensor:
    """Fisher per-column vector alone (normalized). No re-multiplication by the
    activation imatrix — the grad energy already carries one factor of it."""
    return _normalize(v)


def fisher_x_act(
    v: torch.Tensor,
    sigma2: torch.Tensor | float,
    xbar2: torch.Tensor,
) -> torch.Tensor:
    """llama.cpp-style composition with Fisher replacing the imatrix term.

    Mirrors ``gguf_formats._imatrix_weights`` (``qw · sqrt(σ² + x²)``) in the
    activation domain: ``v · sqrt(sigma2 + xbar2)`` with ``xbar2 = E[x^2]`` the
    per-column imatrix and ``sigma2`` a mean-square floor (scalar or per-column,
    the ``sigma2_factor · mean`` guard that keeps never-activated columns from
    collapsing). This folds activation energy back in on top of the Fisher
    weight, so it double-counts activation *by design*; it is one A/B arm, not
    the default. Returns a normalized same-shape vector for the ``qw`` slot.
    """
    v = v.to(torch.float32)
    xbar2 = xbar2.to(torch.float32)
    if not torch.is_tensor(sigma2):
        sigma2 = torch.as_tensor(float(sigma2), dtype=torch.float32,
                                 device=v.device)
    else:
        sigma2 = sigma2.to(torch.float32)
    return _normalize(v * (sigma2 + xbar2).clamp_min(0.0).sqrt())


# Named dispatch so callers/tests can select an arm by string.
COMPOSITIONS = {
    "fisher_only": fisher_only,
    "fisher_x_act": fisher_x_act,
}


# ---------------------------------------------------------------------------
# Harvest driver
# ---------------------------------------------------------------------------

def compute_fisher_col_weights(
    model_path: str,
    dataset_path: str | None,
    *,
    nprobes: int = 32,
    nsamples: int = 8,
    seqlen: int = 128,
    seed: int = 42,
    device: str = "cuda",
    formats: Sequence[str] = ("NVFP4",),
) -> dict[str, torch.Tensor]:
    """Run the (extended) aura_cost probe and return per-Linear Fisher vectors.

    Returns ``{qname: fisher_col}`` where ``fisher_col`` is a fp32 CPU tensor of
    length ``in_features`` (mean KL-Fisher weight-grad energy per input column).
    GPU-first: the probe backward runs resident on ``device``. ``formats`` only
    determines the (unused-here) dW rendering the shared harvest also does; the
    column energy is independent of it.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.aura_cost import compute_aura_cost
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.calibration_data import load_wikitext_calibration_windowed

    staged, _cleanup = stage_multimodal(model_path)
    local_only = Path(staged).exists()
    _log(f"loading {model_path} (staged={staged}) dtype=float32")
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    load_kwargs = dict(
        dtype=torch.float32, trust_remote_code=True,
        local_files_only=local_only, attn_implementation="eager",
    )
    if device.startswith("cuda"):
        load_kwargs["device_map"] = device
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(device)
    model.eval()

    if dataset_path:
        from prismaquant.sensitivity_probe import load_calibration
        calib = load_calibration(
            tok, dataset_path, nsamples, seqlen, calib_seed=seed,
        ).to(device)
    else:
        calib = load_wikitext_calibration_windowed(
            tok, nsamples, seqlen, split="train", seed=seed,
        ).to(device)

    payload = compute_aura_cost(
        model, calib, list(formats),
        n_probes=nprobes, seed_base=seed, min_free_gib=0.0,
        collect_col_energy=True,
        # Column energy covers only 2D nn.Linear; if the model packs MoE experts
        # they are simply absent from this vector (nothing to weight there yet).
        allow_packed_expert_omission=True,
    )
    out: dict[str, torch.Tensor] = {}
    for name, st in payload["stats"].items():
        vec = st.get("fisher_col")
        if vec is not None:
            out[name] = vec.float().cpu()
    _log(f"harvested Fisher col-weights for {len(out)} Linears "
         f"(nprobes={nprobes}, nsamples={nsamples}, seqlen={seqlen}, seed={seed})")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-column KL-Fisher col_weights (nvfp4-cb exp 4)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default=None,
                   help="Calibration source (HF id / .jsonl / .txt). Default: "
                        "WikiText train windows.")
    p.add_argument("--nprobes", type=int, default=32)
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", default="NVFP4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True,
                   help="Output pkl: {'col_weights': {qname: fp32 tensor}, "
                        "'provenance': {...}}")
    args = p.parse_args(argv)

    col_weights = compute_fisher_col_weights(
        args.model, args.dataset,
        nprobes=args.nprobes, nsamples=args.nsamples, seqlen=args.seqlen,
        seed=args.seed, device=args.device,
        formats=[f.strip() for f in args.formats.split(",") if f.strip()],
    )
    payload = {
        "col_weights": col_weights,
        "provenance": {
            "model": str(args.model),
            "dataset": str(args.dataset or "wikitext:train"),
            "dataset_sha256": _dataset_sha256(args.dataset),
            "nprobes": int(args.nprobes),
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "seed": int(args.seed),
            "formats": [f.strip() for f in args.formats.split(",") if f.strip()],
            "git_commit": _git_commit(),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(payload, fh)
    _log(f"wrote {out}: {len(col_weights)} Linears")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
