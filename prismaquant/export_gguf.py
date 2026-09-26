"""Materialize a PrismaQuant recipe as a GGUF checkpoint (llama.cpp lane).

Two-step container strategy: llama.cpp's own ``convert_hf_to_gguf.py
--outtype bf16`` produces the *skeleton* — a full-precision GGUF whose
metadata, tokenizer embedding, tensor naming, and expert stacking are
guaranteed llama.cpp-correct for every architecture the converter knows.
This exporter then rewrites the skeleton: it copies all key/value metadata
verbatim and requantizes each weight tensor to the format the allocator
assigned, using the packers in :mod:`prismaquant.gguf_formats` (whose math
is bit-identical to the registry emulation the cost measurement used).

The result serves in llama.cpp natively and in vLLM via the GGUF path
(in-tree <= 0.19, vllm-gguf-plugin on current vLLM).

Usage:
    python convert_hf_to_gguf.py <hf_model_dir> --outtype bf16 \
        --outfile skeleton.gguf
    python -m prismaquant.export_gguf \
        --skeleton skeleton.gguf \
        --layer-config WORK_DIR/artifacts/layer_config.json \
        --out model-prismaquant.gguf

Provenance (git commit, assignment hash, per-tensor format map) is baked
into the output KV metadata under the ``prismaquant.*`` namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as QT

from prismaquant.gguf_formats import GGUF_BLOCK_BYTES, gguf_pack
from prismaquant.layer_config import load_assignment
from prismaquant.moe_imatrix import build_imatrix_from_act_cache
from prismaquant.export_output_safety import (
    prepare_fresh_export_file,
    transactional_file_output,
)
from prismaquant.footprint import (
    enforce_whole_artifact_budget,
    whole_artifact_budget_from_assignment_payload,
)

# Skeleton tensor types we are willing to treat as a full-precision source.
_SOURCE_TYPES = {QT.F32, QT.F16, QT.BF16}

# GGUF field names that the writer emits itself; never copied from the
# skeleton reader.
_VIRTUAL_KEYS = {
    "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
    "general.alignment",
    # Rewritten below: the skeleton's file_type (BF16) would mislabel the
    # mixed-precision output.
    "general.file_type",
}


def _git_commit() -> str:
    from prismaquant.aura_cost import _git_commit as _aura_git_commit

    return _aura_git_commit() or "unknown"


def _reader_tensor_to_torch(tensor: "gguf.ReaderTensor") -> torch.Tensor:
    """Skeleton tensor -> float32 torch tensor with torch-order shape."""
    shape = tuple(int(d) for d in reversed(tensor.shape.tolist()))
    if tensor.tensor_type == QT.BF16:
        raw = np.ascontiguousarray(tensor.data).view(np.uint16).reshape(shape)
        t = torch.from_numpy(raw.astype(np.uint16)).to(torch.uint16)
        return t.view(torch.bfloat16).to(torch.float32)
    data = gguf.quants.dequantize(
        np.ascontiguousarray(tensor.data), tensor.tensor_type
    )
    return torch.from_numpy(np.ascontiguousarray(data)).reshape(shape).to(
        torch.float32
    )


def _copy_metadata(reader: "gguf.GGUFReader", writer: "gguf.GGUFWriter") -> str:
    """Copy every KV field from skeleton to output. Returns the arch."""
    arch = None
    for field in reader.fields.values():
        if field.name in _VIRTUAL_KEYS:
            continue
        if field.name == "general.architecture":
            arch = str(field.contents())
            continue  # the writer wrote it at construction time
        value = field.contents()
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(field.name, value, vtype,
                                 sub_type=field.types[-1])
        else:
            writer.add_key_value(field.name, value, vtype)
    if arch is None:
        raise ValueError("skeleton GGUF has no general.architecture")
    return arch


def _tensor_name_map(arch_name: str, n_layers: int):
    """gguf-py's per-arch tensor name map (the same table
    convert_hf_to_gguf used to write the skeleton)."""
    arch = None
    for key, value in gguf.MODEL_ARCH_NAMES.items():
        if value == arch_name:
            arch = key
            break
    if arch is None:
        raise ValueError(f"unknown GGUF architecture: {arch_name}")
    return gguf.get_tensor_name_map(arch, n_layers)


def _map_assignment_to_gguf(
    arch_name: str, n_layers: int, assignment: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    """HF-qname assignment -> {gguf tensor name: format}.

    Maps forward from the assignment's HF module qnames. Returns the
    gguf-name map and the set of assignment entries that did not map
    (a naming bug upstream).
    """
    name_map = _tensor_name_map(arch_name, n_layers)
    gguf_formats: dict[str, str] = {}
    unmatched: set[str] = set()
    for hf_qname, fmt in assignment.items():
        gguf_name = name_map.get_name(hf_qname)
        if gguf_name is None:
            unmatched.add(hf_qname)
            continue
        key = gguf_name + ".weight"
        prev = gguf_formats.get(key)
        if prev is not None and prev != fmt:
            # Many-to-one name mapping (a converter that fuses HF Linears
            # into one GGUF tensor) with conflicting formats: silent
            # last-write-wins would ship bytes that do not match the
            # allocation — the GGUF analog of the fused-coherence bug.
            raise ValueError(
                f"{key}: assignment maps multiple HF Linears onto one GGUF "
                f"tensor with conflicting formats ({prev} vs {fmt} from "
                f"{hf_qname}); promote the fused group to one format"
            )
        gguf_formats[key] = fmt
    return gguf_formats, unmatched


def _resolve_token_embedding_format(explicit: str | None,
                                    tied: bool) -> str | None:
    """Tied-embedding guard: when the skeleton carries no output tensor,
    ``token_embd`` IS the output head, and an aggressive embedding format
    silently ships a low-bit head (measured on Qwen3-4B: Q2_K embd cost
    ~4pp top-1 agreement at matched bytes). Default to Q6_K for tied
    models — the llama.cpp preset convention — until the allocator owns
    this decision; an explicit flag always wins."""
    if explicit is not None:
        low = explicit.strip()
        if tied and low in {"Q2_K", "Q3_K"}:
            print(f"[export-gguf] WARNING: tied embeddings — "
                  f"token_embd IS the output head; {low} gives a "
                  f"~{'2.6' if low == 'Q2_K' else '3.4'}-bit head")
        return low
    if tied:
        print("[export-gguf] tied embeddings: defaulting token_embd to "
              "Q6_K (it doubles as the output head; override with "
              "--token-embedding-format)")
        return "Q6_K"
    return None


def _load_act_inputs(act_dir: str | Path,
                     hf_qname: str | None) -> torch.Tensor | None:
    """Full fp32 calibration activations for one Linear, or None."""
    if hf_qname is None:
        return None
    p = Path(act_dir) / (hf_qname.replace(".", "__") + ".pt")
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    inputs = blob.get("inputs") if isinstance(blob, dict) else None
    if inputs is None or inputs.ndim != 2:
        return None
    return inputs.float()


@transactional_file_output(
    source_parameter="skeleton_path",
    output_parameter="out_path",
    where="export_gguf",
)
def export_gguf(
    skeleton_path: str | Path,
    layer_config_path: str | Path,
    out_path: str | Path,
    default_format: str | None = None,
    token_embedding_format: str | None = None,
    output_format: str | None = None,
    imatrix: dict[str, torch.Tensor] | None = None,
    device: str | None = None,
    allow_imatrix_gaps: bool = False,
    gptq_act_dir: str | Path | None = None,
    allow_research_cost_selection: bool = False,
) -> dict[str, int]:
    out_path = prepare_fresh_export_file(
        skeleton_path,
        out_path,
        where="export_gguf",
    )
    if device is None:
        # GPU-first: the weighted scale search is the export hot path.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assignment = load_assignment(layer_config_path)
    recipe_payload = json.loads(Path(layer_config_path).read_text())
    from prismaquant.research_cost_acceptance import (
        enforce_research_export_acknowledgement,
    )
    research_cost_selection = enforce_research_export_acknowledgement(
        recipe_payload,
        acknowledged=allow_research_cost_selection,
        where="export_gguf",
    )
    whole_artifact_budget_from_assignment_payload(
        recipe_payload,
        where="export_gguf preflight",
        assignment=assignment,
    )
    illegal = sorted({
        fmt for fmt in assignment.values()
        if fmt not in GGUF_BLOCK_BYTES and fmt != "BF16"
    })
    if illegal:
        # Fail fast: silently shipping a non-GGUF assignment at skeleton
        # precision would be the coerce-to-BF16 landmine all over again.
        raise ValueError(
            f"assignment contains formats the GGUF container cannot carry: "
            f"{illegal} — allocate with --target-profile gguf"
        )
    reader = gguf.GGUFReader(str(skeleton_path))

    arch_field = reader.fields["general.architecture"]
    arch_name = str(arch_field.contents())
    n_layers = int(reader.fields[f"{arch_name}.block_count"].contents())

    writer = gguf.GGUFWriter(str(out_path), arch_name)
    _copy_metadata(reader, writer)

    gguf_fmt_map, unmatched_assignment = _map_assignment_to_gguf(
        arch_name, n_layers, assignment
    )
    if unmatched_assignment:
        # Fail fast: an assignment entry that maps to no gguf tensor name is
        # a naming bug that would otherwise silently ship the wrong bytes.
        raise ValueError(
            f"{len(unmatched_assignment)} assignment entries have no GGUF "
            f"name mapping, e.g. {sorted(unmatched_assignment)[:8]}"
        )
    seen_gguf_names: set[str] = set()

    counts: Counter[str] = Counter()
    tensor_formats: dict[str, str] = {}
    imatrix_fallback_names: list[str] = []

    # Embedding / output-head policy: these sit outside the allocator's
    # body budget (bpp is reported over quantizable Linears only), but the
    # llama.cpp ecosystem quantizes them and size comparisons must match.
    tensor_names = {t.name for t in reader.tensors}
    tied = "output.weight" not in tensor_names
    token_embedding_format = _resolve_token_embedding_format(
        token_embedding_format, tied,
    )
    if token_embedding_format is not None:
        gguf_fmt_map.setdefault("token_embd.weight", token_embedding_format)
    if output_format is not None:
        gguf_fmt_map.setdefault("output.weight", output_format)

    hf_by_gguf: dict[str, str] = {}
    if gptq_act_dir is not None:
        nm = _tensor_name_map(arch_name, n_layers)
        for hf_qname in assignment:
            gname = nm.get_name(hf_qname)
            if gname is not None:
                hf_by_gguf[gname + ".weight"] = hf_qname

    if imatrix is not None and not imatrix:
        raise ValueError(
            "imatrix requested but empty — the act-cache dir is missing, "
            "cleaned, or has an unexpected blob schema; exporting unweighted "
            "bytes would silently diverge from the imatrix-weighted cost "
            "measurement"
        )
    imatrix_by_gguf: dict[str, torch.Tensor] = {}
    if imatrix:
        nm = _tensor_name_map(arch_name, n_layers)
        for hf_qname, qw in imatrix.items():
            gname = nm.get_name(hf_qname)
            if gname is not None:
                imatrix_by_gguf[gname + ".weight"] = qw

    for tensor in reader.tensors:
        fmt = gguf_fmt_map.get(tensor.name)
        if fmt is not None:
            seen_gguf_names.add(tensor.name)
        wants_quant = fmt is not None and fmt in GGUF_BLOCK_BYTES
        if fmt is None and default_format is not None:
            # Opt-in fallback for 2-D weights the allocator did not cover.
            if tensor.name.endswith(".weight") and len(tensor.shape) >= 2:
                fmt = default_format
                wants_quant = fmt in GGUF_BLOCK_BYTES

        if (
            wants_quant
            and tensor.tensor_type in _SOURCE_TYPES
            and len(tensor.shape) >= 2
            and int(tensor.shape[0]) % GGUF_BLOCK_BYTES[fmt][0] == 0
            # GGUF shape order is reversed: shape[0] is the input dim.
        ):
            w = _reader_tensor_to_torch(tensor).to(device)
            qw = imatrix_by_gguf.get(tensor.name)
            if qw is not None and qw.numel() != w.shape[-1]:
                raise ValueError(
                    f"{tensor.name}: imatrix vector has {qw.numel()} columns "
                    f"but the tensor has {w.shape[-1]} — the act cache does "
                    f"not describe this checkpoint"
                )
            if imatrix:
                if qw is not None:
                    counts["imatrix_weighted"] += 1
                else:
                    counts["imatrix_fallback"] += 1
                    imatrix_fallback_names.append(tensor.name)
            acts = None
            if gptq_act_dir is not None and w.ndim == 2:
                from prismaquant.gguf_gptq import GPTQ_SUPPORTED
                if fmt in GPTQ_SUPPORTED:
                    acts = _load_act_inputs(gptq_act_dir,
                                            hf_by_gguf.get(tensor.name))
            if acts is not None:
                # GPTQ under the frozen two-tier scales: same fields
                # contract, only q is re-decided with OBS propagation.
                from prismaquant.gguf_formats import gguf_pack_fields
                from prismaquant.gguf_gptq import gptq_fields

                fields = gptq_fields(
                    w, fmt, acts.to(w.device), col_weights=qw,
                )
                packed = gguf_pack_fields(fields, fmt, tuple(w.shape))
                counts["gptq"] += 1
            else:
                packed = gguf_pack(w, fmt, col_weights=qw)
            # No raw_shape: for quantized dtypes gguf-py derives the logical
            # shape from the packed byte shape (quant_shape_from_byte_shape).
            writer.add_tensor(tensor.name, packed, raw_dtype=getattr(QT, fmt))
            counts[fmt] += 1
            tensor_formats[tensor.name] = fmt
        elif wants_quant and fmt is not None and tensor.name not in gguf_fmt_map:
            # --default-format blanket tensor failing a precondition:
            # soft-skip is the documented semantics, but leave a trace so
            # a silently-larger artifact is explainable from the printout.
            counts[f"default_skip({fmt}->{tensor.tensor_type.name})"] += 1
            data = np.ascontiguousarray(tensor.data)
            writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
            counts[tensor.tensor_type.name] += 1
            tensor_formats[tensor.name] = tensor.tensor_type.name
        elif wants_quant and tensor.name in gguf_fmt_map:
            # An explicitly assigned tensor that fails a quantize
            # precondition means the skeleton disagrees with what the
            # allocator's legality gate checked — never ship it silently at
            # skeleton precision (the artifact would exceed the allocated
            # budget and the measured cost would not describe the bytes).
            raise ValueError(
                f"{tensor.name}: assigned {fmt} but cannot quantize "
                f"(source type {tensor.tensor_type.name}, shape "
                f"{tuple(int(d) for d in tensor.shape)}, block "
                f"{GGUF_BLOCK_BYTES[fmt][0]})"
            )
        else:
            # Verbatim copy: reader data is already in the writer's expected
            # layout for every tensor type (typed+logical for f32/f16,
            # byte-shaped for bf16/quantized).
            data = np.ascontiguousarray(tensor.data)
            writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
            counts[tensor.tensor_type.name] += 1
            tensor_formats[tensor.name] = tensor.tensor_type.name

    missing = set(gguf_fmt_map) - seen_gguf_names
    # The tied-embeddings case: a skeleton may carry no output.weight.
    missing.discard("output.weight")
    if missing:
        raise ValueError(
            f"{len(missing)} assignment entries matched no skeleton "
            f"tensor, e.g. {sorted(missing)[:8]}"
        )

    digest = hashlib.sha256(
        json.dumps(dict(sorted(assignment.items())),
                   separators=(",", ":")).encode()
    ).hexdigest()
    if imatrix and counts.get("imatrix_weighted", 0) == 0:
        raise ValueError(
            "imatrix was provided but weighted zero tensors — act-cache "
            "naming has drifted from the gguf name map; shipping unweighted "
            "bytes against an imatrix-weighted cost would be a silent "
            "rendering confound"
        )
    # Strict coverage: a PARTIALLY drifted act cache (one shard's flush
    # died, disk-hygiene removed a subset of act/*.pt) must not ship a
    # mixed weighted/unweighted artifact with exit 0. token_embd has no
    # activation blob by construction (row lookups, not a matmul input).
    unexpected_gaps = [
        n for n in imatrix_fallback_names if n != "token_embd.weight"
    ]
    if imatrix and unexpected_gaps and not allow_imatrix_gaps:
        raise ValueError(
            f"{len(unexpected_gaps)} quantized tensors have no imatrix "
            f"entry (e.g. {unexpected_gaps[:5]}) — the act cache is "
            f"partially missing or misnamed; these tensors would ship "
            f"unweighted against an imatrix-weighted cost. Pass "
            f"--allow-imatrix-gaps to override deliberately."
        )

    writer.add_file_type(gguf.LlamaFileType.GUESSED)
    writer.add_key_value("prismaquant.git_commit", _git_commit(),
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("prismaquant.assignment_sha256", digest,
                         gguf.GGUFValueType.STRING)
    if research_cost_selection is not None:
        writer.add_key_value(
            "prismaquant.research_cost_selection",
            json.dumps(research_cost_selection, sort_keys=True),
            gguf.GGUFValueType.STRING,
        )
    writer.add_key_value("prismaquant.tensor_formats",
                         json.dumps(tensor_formats, sort_keys=True),
                         gguf.GGUFValueType.STRING)
    # Calibration provenance: the imatrix is a deterministic function of the
    # calibration activations, so its digest identifies the calibration; the
    # weighted/fallback counts make a silent-unweighted export detectable
    # after the fact (house rule: an irreproducible number is quarantined).
    imatrix_digest = ""
    if imatrix:
        h = hashlib.sha256()
        for name in sorted(imatrix):
            h.update(name.encode())
            h.update(imatrix[name].to(torch.float32).cpu().numpy().tobytes())
        imatrix_digest = h.hexdigest()
    writer.add_key_value("prismaquant.imatrix", bool(imatrix),
                         gguf.GGUFValueType.BOOL)
    writer.add_key_value("prismaquant.imatrix_sha256", imatrix_digest,
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("prismaquant.imatrix_weighted",
                         int(counts.get("imatrix_weighted", 0)),
                         gguf.GGUFValueType.UINT32)
    writer.add_key_value("prismaquant.imatrix_fallback",
                         int(counts.get("imatrix_fallback", 0)),
                         gguf.GGUFValueType.UINT32)
    writer.add_key_value("prismaquant.gptq", gptq_act_dir is not None,
                         gguf.GGUFValueType.BOOL)
    writer.add_key_value("prismaquant.token_embedding_format",
                         token_embedding_format or "",
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("prismaquant.output_format", output_format or "",
                         gguf.GGUFValueType.STRING)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    budget_attestation = enforce_whole_artifact_budget(
        out_path,
        recipe_payload,
        where="export_gguf",
        assignment=assignment,
    )
    if budget_attestation is not None:
        print(
            "whole-artifact budget passed: "
            f"{budget_attestation['actual_bytes']}B <= "
            f"{budget_attestation['budget_bytes']}B"
        )
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton", required=True,
                    help="bf16/f16 GGUF produced by convert_hf_to_gguf.py")
    ap.add_argument("--layer-config", required=True)
    ap.add_argument(
        "--allow-research-cost-selection",
        action="store_true",
        help="explicitly acknowledge export of a research-stamped assembled "
             "cost selection; recorded in GGUF metadata",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--default-format", default=None,
        help="optional GGUF format for 2-D weights absent from the "
        "assignment (e.g. Q6_K); default: keep skeleton precision",
    )
    ap.add_argument("--token-embedding-format", default=None,
                    help="quantize token_embd.weight (e.g. Q2_K)")
    ap.add_argument("--output-format", default=None,
                    help="quantize output.weight / lm_head (e.g. Q6_K)")
    ap.add_argument("--device", default=None,
                    help="quantization device (default: cuda if available)")
    ap.add_argument("--allow-imatrix-gaps", action="store_true",
                    help="permit quantized tensors without an imatrix entry "
                    "(default: hard-fail — mixed weighted/unweighted bytes "
                    "diverge from the measured cost)")
    ap.add_argument("--gptq", action="store_true",
                    help="RESEARCH: GPTQ OBS rounding under the frozen "
                    "two-tier scales (needs --imatrix-from-act-cache for "
                    "activations). Default OFF: the cost stage does not "
                    "score GPTQ renders yet, so a GPTQ export diverges "
                    "from the measured cost — A/B only, hold the "
                    "allocation fixed.")
    ap.add_argument(
        "--imatrix-from-act-cache", default=None,
        help="activation-cache dir; builds per-column importance "
        "(mean squared activation) and biases k-quant scale selection "
        "with llama.cpp imatrix semantics",
    )
    args = ap.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("export_gguf")
    imatrix = None
    if args.imatrix_from_act_cache:
        imatrix = build_imatrix_from_act_cache(args.imatrix_from_act_cache)
        print(f"imatrix: {len(imatrix)} Linears from act cache")
    if args.gptq and not args.imatrix_from_act_cache:
        ap.error("--gptq requires --imatrix-from-act-cache (activations)")
    counts = export_gguf(
        args.skeleton, args.layer_config, args.out,
        default_format=args.default_format,
        token_embedding_format=args.token_embedding_format,
        output_format=args.output_format,
        imatrix=imatrix,
        device=args.device,
        allow_imatrix_gaps=args.allow_imatrix_gaps,
        gptq_act_dir=(args.imatrix_from_act_cache if args.gptq else None),
        allow_research_cost_selection=args.allow_research_cost_selection,
    )
    size = Path(args.out).stat().st_size / 1e9
    print(f"wrote {args.out} ({size:.2f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
