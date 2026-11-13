"""incremental_probe.py — PrismaQuant probe: one streamed forward, no gradient.

The model is always loaded through the layer-streaming primitives in
`layer_streaming`: the head (embed + norm + lm_head + rotary) stays resident
and the decoder layers stream in on demand. One no-grad forward over the full
calibration batch writes the act/ rows (#3447) of every probed body Linear and
packed-experts module, of the lm_head input and of the MTP Linears, and the
router statistics. probe.pkl keeps the per-Linear shape rows, the routing and
the calibration identity that the cost stages and the allocator read. The
probe computes no Fisher (#3448): AURA prices from its own KL adjoint.
"""
from __future__ import annotations

from . import probe_inventory
from . import probe_statistics
from .fla_config_pin import fla_config_identity, pin_fla_configs

import argparse
import dataclasses
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

from . import io_spans
from prismaquant.incremental_shards import build_layer_shard_regexes

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from .layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)
from .perturbed_x_cache import calibration_data_hash
from prismaquant.sensitivity_probe import (
    RouterTracker,
    discover_moe_structure,
    discover_moe_routers,
    grouped_linear_groups,
    read_top_k,
    stage_text_only,
)
from prismaquant.calibration_data import load_calibration

from .streaming_model import (
    StreamingContext,
    _build_streaming_context,
)


# v26: central runtime-flag helper. Each named env var defaults to the
# given value (typically True for performance flags whose math is
# equivalent to the legacy path). Set the env var to "0" to disable.
# This replaces the proliferating `os.environ.get(NAME) == "1"`
# pattern that left every perf flag opt-in indefinitely.
def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False", "FALSE", "no", "NO")


# ---- Per-channel Fisher marginals -------------------------------------
# The per-element diagonal H[o,i] = Σ_t g[t,o]²·x[t,i]² is unstorable at
# scale (47k Linears × 17 MB ≈ 800 GB — the reason only two scalars
# survive today). Its two MARGINALS are storable, and they are what a
# per-channel sensitivity contract actually needs:
#
#   fisher_row[o] = Σ_i H[o,i]      fisher_col[i] = Σ_o H[o,i]
#
# plus the two pure factors (g_sq_sum, act_sq_sum) that let a consumer
# separate "this output channel has hot gradients" from "this input
# channel has hot activations", and act_absmax for clipping decisions.
#
# The row/col marginals are read off the `chunk_h` every accumulation
# site already materializes, so they cost no extra matmul AND satisfy
# sum(fisher_row) == sum(fisher_col) == chunk_h.sum() by construction —
# that identity is the wiring check (tests/test_probe_marginals.py).


def _read_proc_status_kb(*keys: str) -> dict[str, int]:
    """Read /proc/self/status for the given keys (e.g. 'VmHWM', 'VmRSS').
    Returns a dict of key -> kilobytes. Missing keys map to 0."""
    try:
        status = io_spans.read_proc_status()
    except Exception:
        status = {}
    return {k: status.get(k, 0) // 1024 for k in keys}


# ---------------------------------------------------------------------------
# Shard regex builders (unchanged public API)
# ---------------------------------------------------------------------------


def _detect_profile_for_shards(model_path: str):
    """The checkpoint's profile, or `DefaultProfile` for an unknown one.

    `DeadVendoredOverrideError` is not that case and is re-raised (#201): the
    architecture IS known, its vendored modelling path is dead, and answering
    `DefaultProfile` here would shard and probe the model against upstream
    modelling code under a profile that promises the vendored copy.
    """
    from .model_profiles.registry import DeadVendoredOverrideError, detect_profile

    try:
        return detect_profile(model_path)
    except DeadVendoredOverrideError:
        raise
    except Exception:
        from .model_profiles.default import DefaultProfile

        return DefaultProfile()


def resolve_linear_exclude(model_path: str) -> str:
    """The probe's Linear exclusion: the router baseline OR'd with any
    profile-declared extra (`ModelProfile.probe_linear_exclude_extra`),
    for live Linears outside the serving contract's quantizable set.
    All meta stamps and hook installs must use this one resolver so
    shard-reuse keys stay consistent."""
    profile = _detect_profile_for_shards(model_path)
    return probe_inventory.linear_exclusion_for_profile(profile)


def build_extended_shard_regexes(
    model_path: str,
    layers_per_shard: int,
    *,
    include_body: bool = True,
    include_mtp: bool = True,
    include_visual: bool = True,
    include_lm_head: bool = True,
) -> list[str]:
    """Extended shard list covering the profile-declared probe regions:

      - body transformer layers
      - optional MTP block(s)
      - optional visual/audio tower layers
      - optional lm_head
    """
    profile = _detect_profile_for_shards(model_path)
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    body_prefix = profile.body_layer_prefix()
    mtp_prefix = profile.mtp_layer_prefix()
    visual_key = profile.visual_config_key()
    visual_prefix = profile.visual_layer_prefix()
    lm_head_name = profile.lm_head_name()

    regexes: list[str] = []

    if include_body:
        n_body = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers", 0)))
        regexes.extend(build_layer_shard_regexes(
            n_body, layers_per_shard, layer_prefix=body_prefix))

    if include_mtp:
        n_mtp_config = int(profile.mtp_layer_count(cfg) or 0)
        n_mtp_actual = _count_mtp_layers_from_safetensors(
            model_path,
            layer_prefix=mtp_prefix,
        )
        # Empirical safetensors count is ground truth: a config may
        # declare MTP layers (inherited from a base) when the finetune
        # actually stripped the weights. Conversely, local Qwen3.5/3.6
        # exports can carry `mtp.*` weights even when the text config omits
        # the count. Use actual safetensors as the fallback, and cap declared
        # counts to actual when both are present.
        if n_mtp_actual > 0:
            n_mtp = min(n_mtp_config, n_mtp_actual) if n_mtp_config > 0 else n_mtp_actual
        else:
            n_mtp = 0
        if n_mtp_config > 0 and n_mtp_actual == 0:
            print(f"[shard-schedule] config declares "
                  f"{n_mtp_config} MTP layer(s) but safetensors index "
                  f"has no `{mtp_prefix}.*` keys; skipping MTP shards "
                  f"(common on finetunes that strip MTP)",
                  flush=True)
        if n_mtp > 0:
            mtp_regexes = build_layer_shard_regexes(
                n_mtp, layers_per_shard, layer_prefix=mtp_prefix)
            if mtp_regexes and profile.mtp_extra_linear_names():
                extra = "|".join(
                    re.escape(name) for name in profile.mtp_extra_linear_names()
                )
                mtp_regexes[0] = rf"(?:{extra}|{mtp_regexes[0]})"
            regexes.extend(mtp_regexes)

    if include_visual and visual_key and visual_prefix:
        regexes.extend(profile.visual_shard_regexes(cfg, layers_per_shard))

    if include_lm_head:
        regexes.append(rf"^{re.escape(lm_head_name)}$")

    return regexes


def _count_mtp_layers_from_safetensors(
    model_path: str,
    *,
    layer_prefix: str = "mtp.layers",
) -> int:
    """Fallback for when the config doesn't carry an MTP layer count:
    scan the source safetensors index and count `<layer_prefix>.<N>.` paths."""
    src = Path(model_path)
    layer_re = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.")
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        try:
            from safetensors.torch import safe_open
            mtp_indices: set[int] = set()
            for f in os.listdir(src):
                if not f.endswith(".safetensors"):
                    continue
                with safe_open(str(src / f), framework="pt") as sf:
                    for k in sf.keys():
                        m = layer_re.match(k)
                        if m:
                            mtp_indices.add(int(m.group(1)))
            return max(mtp_indices) + 1 if mtp_indices else 0
        except Exception:
            return 0
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_indices = set()
    for k in wm:
        m = layer_re.match(k)
        if m:
            mtp_indices.add(int(m.group(1)))
    return max(mtp_indices) + 1 if mtp_indices else 0


# ---------------------------------------------------------------------------
# Predeclared shard schedule (v20 step 1)
#
# A ShardSchedule is the full, statically-known list of shards that phase-3
# will process for a chunk. Each entry pairs the linear-include regex (the
# only thing the runners themselves consume) with kind + the layer indices
# in scope, so policy code (cache mark_done, instrumentation, allocator)
# can answer "what layers are in shard S?" without re-parsing regexes.
#
# This unblocks v20 steps 2-5: mark_done events fall out of
# `layers_done_after(shard_idx)`, value-aware retention can preload the
# layers reused across all shards, etc.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ShardEntry:
    shard_idx: int
    linear_include: str
    kind: str  # "body", "mtp", "visual", "lm_head"
    layer_indices: frozenset[int]
    layer_prefix: str | None  # None for lm_head and non-block visual regions.


@dataclasses.dataclass(frozen=True)
class ShardSchedule:
    entries: tuple[ShardEntry, ...]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> ShardEntry:
        return self.entries[i]

    def regexes(self) -> list[str]:
        return [e.linear_include for e in self.entries]

    def body_layer_indices(
        self,
        layer_prefix: str | None = None,
    ) -> frozenset[int]:
        out: set[int] = set()
        for e in self.entries:
            if e.kind != "body":
                continue
            if layer_prefix is None or e.layer_prefix == layer_prefix:
                out |= e.layer_indices
        return frozenset(out)

    def layers_done_after(self, shard_idx: int,
                          layer_prefix: str | None = None) -> frozenset[int]:
        """Layer indices in shard_idx's scope that no later shard touches.

        For the canonical body-shard layout (contiguous, disjoint ranges
        per shard), this is exactly the in-scope layers of shard_idx.
        For unified-sweep (one body shard) it returns the full body
        layer set after the only shard. The cache uses this signal to
        evict layers we've provably stopped tracking stats for."""
        if shard_idx >= len(self.entries):
            return frozenset()
        cur = self.entries[shard_idx]
        if layer_prefix is None:
            layer_prefix = cur.layer_prefix
        if cur.layer_prefix != layer_prefix:
            return frozenset()
        future: set[int] = set()
        for e in self.entries[shard_idx + 1:]:
            if e.layer_prefix == layer_prefix:
                future |= e.layer_indices
        return cur.layer_indices - future


def _build_body_shard_entries(num_layers: int, layers_per_shard: int,
                              layer_prefix: str,
                              kind: str,
                              start_idx: int) -> list[ShardEntry]:
    """Mirror of build_layer_shard_regexes but emits ShardEntry list."""
    entries: list[ShardEntry] = []
    sidx = start_idx
    for start in range(0, num_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        entries.append(ShardEntry(
            shard_idx=sidx,
            linear_include=body,
            kind=kind,
            layer_indices=frozenset(range(start, end)),
            layer_prefix=layer_prefix,
        ))
        sidx += 1
    return entries


def _build_visual_shard_entries(
    profile, cfg: dict, layers_per_shard: int, start_idx: int,
) -> list[ShardEntry]:
    """Adapt profile-declared visual regions to production shard entries."""
    regexes = profile.visual_shard_regexes(cfg, layers_per_shard)
    if not regexes:
        return []
    visual_key = profile.visual_config_key()
    visual_prefix = profile.visual_layer_prefix()
    vis_cfg = cfg.get(visual_key, {})
    n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
    entries = _build_body_shard_entries(
        n_vis, max(layers_per_shard, 4), visual_prefix, "visual", start_idx,
    )
    n_block_shards = len(entries)
    if [entry.linear_include for entry in entries] != regexes[:n_block_shards]:
        raise ValueError("profile visual scope disagrees with visual block shard layout")
    entries.extend(
        ShardEntry(
            shard_idx=start_idx + n_block_shards + offset,
            linear_include=regex,
            kind="visual",
            layer_indices=frozenset(),
            layer_prefix=None,
        )
        for offset, regex in enumerate(regexes[n_block_shards:])
    )
    return entries


def build_shard_schedule(
    *,
    model_path: str,
    num_body_layers: int,
    body_layers_per_shard: int,
    body_layer_range: tuple[int, int],
    include_mtp: bool,
    include_visual: bool,
    include_lm_head: bool,
    unified_body_sweep: bool,
) -> ShardSchedule:
    """Single source of truth for the shard list.

    body_layer_range = (first_layer, last_layer_exclusive) — slices the
    body shard list to this range (default (0, num_body_layers))."""
    profile = _detect_profile_for_shards(model_path)
    body_prefix = profile.body_layer_prefix()
    mtp_prefix = profile.mtp_layer_prefix()
    lm_head_name = profile.lm_head_name()
    sidx = 0

    # Body shards (mirror old slice semantics).
    body_entries_full = _build_body_shard_entries(
        num_body_layers, body_layers_per_shard, body_prefix, "body", sidx)
    first = body_layer_range[0] // body_layers_per_shard
    last = (body_layer_range[1] + body_layers_per_shard - 1) // body_layers_per_shard
    body_entries = body_entries_full[first:last]
    # Renumber after slice so shard_idx is contiguous from 0.
    body_entries = [
        dataclasses.replace(e, shard_idx=sidx + i)
        for i, e in enumerate(body_entries)
    ]
    sidx += len(body_entries)

    if unified_body_sweep and body_entries:
        union = "(?:" + "|".join(
            f"(?:{e.linear_include})" for e in body_entries) + ")"
        union_layers = frozenset().union(
            *(e.layer_indices for e in body_entries))
        body_entries = [ShardEntry(
            shard_idx=0,
            linear_include=union,
            kind="body",
            layer_indices=union_layers,
            layer_prefix=body_prefix,
        )]
        sidx = 1

    extras: list[ShardEntry] = []
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    if include_mtp:
        n_mtp_config = int(profile.mtp_layer_count(cfg) or 0)
        n_mtp_actual = _count_mtp_layers_from_safetensors(
            model_path,
            layer_prefix=mtp_prefix,
        )
        if n_mtp_actual > 0:
            n_mtp = min(n_mtp_config, n_mtp_actual) if n_mtp_config > 0 else n_mtp_actual
        else:
            n_mtp = 0
        if n_mtp_config > 0 and n_mtp_actual == 0:
            print(f"[shard-schedule] config declares "
                  f"{n_mtp_config} MTP layer(s) but safetensors index "
                  f"has no `{mtp_prefix}.*` keys; skipping MTP shards "
                  f"(common on finetunes that strip MTP)",
                  flush=True)
        if n_mtp > 0:
            mtp_entries = _build_body_shard_entries(
                n_mtp, body_layers_per_shard, mtp_prefix, "mtp", sidx)
            if mtp_entries and profile.mtp_extra_linear_names():
                extra = "|".join(
                    re.escape(name) for name in profile.mtp_extra_linear_names()
                )
                mtp_entries[0] = dataclasses.replace(
                    mtp_entries[0],
                    linear_include=rf"(?:{extra}|{mtp_entries[0].linear_include})",
                )
            extras.extend(mtp_entries)
            sidx += len(mtp_entries)

    if include_visual:
        vis_entries = _build_visual_shard_entries(
            profile, cfg, body_layers_per_shard, sidx,
        )
        extras.extend(vis_entries)
        sidx += len(vis_entries)

    if include_lm_head:
        # A tied head (`tie_word_embeddings` declared AND no head tensor
        # in the index) is an alias of the input embedding: same storage,
        # no source bytes of its own. It is structurally passthrough-only
        # — re-encoding it would re-encode the non-quantizable embedding
        # — so it gets no Fisher row and no cost row. Same shape as the
        # MTP skip above: config declares it, the index does not have it.
        from .tied_embeddings import lm_head_is_tied_alias
        if lm_head_is_tied_alias(model_path, profile=profile):
            print(f"[shard-schedule] `{lm_head_name}` is a tied alias of the "
                  "input embedding (config declares tie_word_embeddings and "
                  "the safetensors index has no head tensor); skipping the "
                  "lm_head shard — a tied head shares storage with the "
                  "non-quantizable embedding and is never quantized",
                  flush=True)
        else:
            extras.append(ShardEntry(
                shard_idx=sidx,
                linear_include=rf"^{re.escape(lm_head_name)}$",
                kind="lm_head",
                layer_indices=frozenset(),
                layer_prefix=None,
            ))
            sidx += 1

    return ShardSchedule(entries=tuple(body_entries + extras))


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged)
# ---------------------------------------------------------------------------
def load_num_hidden_layers(model_path: str) -> int:
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Could not infer num_hidden_layers from {cfg_path}")
    return n


def _resident_linear_fqns(model: nn.Module, layers_prefix: str,
                          num_layers: int) -> list[str]:
    """All nn.Linear fqns NOT under a decoder-layer prefix (lm_head,
    root-level projections). These are resident during streaming."""
    resident: list[str] = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(n.startswith(f"{layers_prefix}{L}.") for L in range(num_layers)):
            continue
        resident.append(n)
    return resident


class _ActRowCache:
    """The act/ rows of one capture: keyed row selection and the files."""

    def __init__(self, cache_dir: str, rows_limit: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rows_limit = max(1, int(rows_limit))
        self.snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.indices: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.offsets: dict[str, int] = defaultdict(int)
        self._pending: list = []
        self._pool = None
        # PRISMAQUANT_ACT_CACHE_ASYNC=0 keeps per-Linear writes synchronous.
        if _env_flag("PRISMAQUANT_ACT_CACHE_ASYNC", default=True):
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(
                max_workers=int(os.environ.get("PRISMAQUANT_ACT_CACHE_WORKERS", "4")),
                thread_name_prefix="act-save",
            )

    def keep(self, name: str, x: torch.Tensor) -> None:
        """Number each row of the call globally, then keep the keyed top-K."""
        flat = x.detach().reshape(-1, x.size(-1))
        base = self.offsets[name]
        self.offsets[name] += int(flat.size(0))
        rows = torch.arange(base, base + flat.size(0), device=flat.device)
        probe_statistics.keep_act_rows(
            self.snaps, self.indices, name, flat, rows, self.rows_limit)

    def flush(self, snaps_by_name=None, indices_by_name=None) -> None:
        """Write and drop each kept name: one host copy per Linear."""
        snaps_by_name = self.snaps if snaps_by_name is None else snaps_by_name
        indices_by_name = self.indices if indices_by_name is None else indices_by_name
        # #43: PRISMAQUANT_ACT_CACHE_FP32 keeps activations at FP32
        # for better Hessian numerical stability in the cost step.
        cache_dtype = (torch.float32
                       if os.environ.get("PRISMAQUANT_ACT_CACHE_FP32", "1") != "0"
                       else torch.bfloat16)
        for name in list(snaps_by_name.keys()):
            snaps = snaps_by_name.pop(name)
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to("cpu", dtype=cache_dtype).contiguous()
            payload = {"inputs": X, "name": name,
                       "row_selection": probe_statistics.ACT_ROW_SELECTION}
            index_parts = indices_by_name.pop(name, [])
            if index_parts:
                row_indices = torch.cat(index_parts, dim=0).to(
                    "cpu", dtype=torch.long).contiguous()
                if row_indices.numel() == X.shape[0]:
                    payload["row_indices"] = row_indices
            target = self.cache_dir / (re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt")
            if self._pool is not None:
                self._pending.append(self._pool.submit(torch.save, payload, target))
            else:
                torch.save(payload, target)

    def drain(self) -> None:
        """Block until every background write has completed, so a later stage
        sees a fully written act/ directory."""
        for fut in self._pending:
            fut.result()
        self._pending.clear()


def _load_mtp_source_for_probe(
    mtp_profile,
    inner_mtp: nn.Module,
    model_path: str,
) -> dict[str, torch.Tensor]:
    """Admit complete MTP source weights, or preserve the empty-source skip."""
    from .model_profiles.base import load_complete_mtp_state_dict

    raw = mtp_profile.read_mtp_source_state_dict(model_path)
    if not raw:
        return {}
    load_complete_mtp_state_dict(
        mtp_profile, inner_mtp, raw, where="incremental MTP probe",
    )
    return raw


def _w_stats(weight: torch.Tensor) -> tuple[float, float]:
    """(max |w|, sum w^2) of one weight, read off the device in one sync."""
    w = weight.detach()
    pair = torch.stack([w.abs().max(), w.pow(2).sum()]).float().cpu().tolist()
    return float(pair[0]), float(pair[1])


def _linear_row(mod: nn.Linear, expert=None) -> dict:
    """The probe row of one Linear: its shape, weight scale and routing."""
    w_max_abs, w_norm_sq = _w_stats(mod.weight)
    router_path, expert_id = expert or (None, None)
    return {"w_max_abs": w_max_abs, "w_norm_sq": w_norm_sq,
            "n_params": int(mod.weight.numel()),
            "in_features": int(mod.in_features), "out_features": int(mod.out_features),
            "n_tokens_seen": 0, "route_prob": None,
            "router_path": router_path, "expert_id": expert_id}


def _probe_unit_rows(model: nn.Module, profile, unit, expert_info: dict) -> dict[str, dict]:
    """Probe rows of one inventory unit. A packed-experts module gives one row
    per 3D parameter, keyed `<module>.<param>`, as `[experts, out, in]`."""
    mod = model.get_submodule(unit.module_name)
    if unit.kind != "packed_experts":
        row = _linear_row(mod, expert_info.get(unit.module_name))
        if unit.kind == "grouped_linear":
            # `num_groups`, not `num_experts`: downstream reads the latter as a
            # packed expert stack.
            row["num_groups"] = int(grouped_linear_groups(mod, profile))
        return {unit.module_name: row}
    rows = {}
    for full_name in unit.parameters:
        p = model.get_parameter(full_name)
        w_max_abs, w_norm_sq = _w_stats(p)
        rows[full_name] = {
            "w_max_abs": w_max_abs, "w_norm_sq": w_norm_sq,
            "n_params": int(p.numel()), "in_features": int(p.shape[2]),
            "out_features": int(p.shape[1]), "num_experts": int(p.shape[0]),
            "n_tokens_seen": 0, "route_prob": None, "router_path": None,
            "expert_id": None, "_packed_experts_module": unit.module_name,
            "_packed_param": full_name.rsplit(".", 1)[-1]}
    return rows


def _input_hook(act_rows: _ActRowCache, name: str):
    return lambda _m, inp, _out: act_rows.keep(name, inp[0] if isinstance(inp, tuple) else inp)


def _probe_forward(ctx: StreamingContext, *, calib: torch.Tensor, scope: str,
                   exclude: str, act_rows: _ActRowCache, prefetch_lookahead: int):
    """One no-grad streamed forward of the full calibration batch.

    Writes the act/ rows of every probed body Linear and packed-experts module
    that `scope` matches, and of the lm_head input; records the router
    statistics. Returns (stats, routing, embeds_cpu, final_hidden_cpu).
    """
    from .model_profiles import profile_from_model
    device, dtype, model, base_model = ctx.device, ctx.dtype, ctx.model, ctx.base_model
    profile, base_profile = profile_from_model(model), profile_from_model(base_model)
    layers_prefix = ctx.layers_prefix
    tokens, batch = calib.size(-1), calib.size(0)
    ids = calib.to(device)
    position_ids = torch.arange(tokens, device=device).unsqueeze(0)
    expert_info = discover_moe_structure(model, profile=base_profile)
    tracker = RouterTracker(model, sorted(discover_moe_routers(model, profile=base_profile)),
                            top_k=read_top_k(model))
    scope_re = re.compile(scope)
    units = [unit for unit in probe_inventory.build_probe_inventory(
                 model, profile, linear_exclude=exclude).priced_units
             if unit.module_name.startswith(layers_prefix) and scope_re.search(unit.module_name)]
    stats: dict[str, dict] = {}
    t_phase = time.time()
    with torch.no_grad():
        hidden = base_model.embed_tokens(ids).to(dtype)
        embeds_cpu = hidden.detach().cpu()
        position_embeddings = _compute_position_embeddings(
            base_model, hidden, position_ids, base_profile)
        causal_mask = _compute_attention_mask(base_model, hidden, position_ids)
        hidden = base_profile.expand_hidden_for_layers(hidden, base_model)
        # Per-pass shared forward state (e.g. Gemma4 cross-layer KV sharing).
        pass_state = base_profile.new_forward_pass_state()
        print(f"[incremental] forward N={batch} T={tokens} hidden={tuple(hidden.shape)}",
              flush=True)
        for d in range(prefetch_lookahead):
            ctx.schedule_prefetch(d)
        try:
            for L in range(ctx.num_layers):
                load_t0 = time.time()
                src = ctx.install(L)
                ctx.schedule_prefetch(L + prefetch_lookahead)
                load_s = time.time() - load_t0
                here = [u for u in units if u.module_name.startswith(f"{layers_prefix}{L}.")]
                hooks = [model.get_submodule(u.module_name).register_forward_hook(
                    _input_hook(act_rows, u.module_name)) for u in here]
                for unit in here:
                    stats.update(_probe_unit_rows(model, profile, unit, expert_info))
                fwd_t0 = time.time()
                hidden = _call_layer(
                    ctx.layers[L], hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    **base_profile.extra_layer_kwargs(input_ids=ids),
                    pass_state=pass_state,
                )
                fwd_s = time.time() - fwd_t0
                for handle in hooks:
                    handle.remove()
                act_rows.flush()
                ctx.unload(L)
                if L % 8 == 0 or L == ctx.num_layers - 1:
                    print(f"[incremental] fwd L{L:02d}  src={src}  "
                          f"load={load_s:.2f}s  fwd={fwd_s:.2f}s", flush=True)
        finally:
            tracker.remove_hooks()
        # Profile-driven hidden-state collapse: DSv4 folds its multi-stream
        # hidden back to `[B, T, H]` before the final norm.
        final = _get_final_norm(base_model)(
            base_profile.collapse_hidden_after_layers(hidden, base_model))
    for name, entry in stats.items():
        entry["n_tokens_seen"] = int(act_rows.offsets.get(
            entry.get("_packed_experts_module", name), 0))
        if entry["router_path"]:
            entry["route_prob"] = tracker.prob(entry["router_path"], entry["expert_id"])
    lm_names = [n for n in _resident_linear_fqns(model, layers_prefix, ctx.num_layers)
                if scope_re.search(n) and not re.search(exclude, n)]
    # The lm_head input of each scored position: row = sample * T + token.
    # The last position predicts nothing and is not kept.
    scored = final[:, :-1, :]
    rows = (torch.arange(batch, device=device)[:, None] * tokens
            + torch.arange(tokens - 1, device=device)[None, :]).reshape(-1)
    for name in lm_names:
        stats[name] = _linear_row(model.get_submodule(name))
        stats[name]["n_tokens_seen"] = int(rows.numel())
        probe_statistics.keep_act_rows(act_rows.snaps, act_rows.indices, name,
                                       scored.reshape(-1, scored.size(-1)), rows, 256)
    act_rows.flush()
    print(f"[incremental] forward: {time.time() - t_phase:.1f}s  "
          f"{ctx.layer_cache.summary()}", flush=True)
    routing = {"router_counts": tracker.counts,
               "router_totals": dict(tracker.total_tokens),
               "router_active_counts": tracker.active_counts,
               "expert_route_stats": tracker.route_stats,
               "expert_info": {k: v for k, v in expert_info.items() if k in stats}}
    return stats, routing, embeds_cpu, final.detach().cpu()


def _probe_mtp(ctx: StreamingContext, *, calib: torch.Tensor,
               act_rows: _ActRowCache, embeds_cpu: torch.Tensor,
               final_cpu: torch.Tensor, model_path: str) -> tuple[dict, dict]:
    """A forward-only pass of the profile's MTP module: its probe rows and the
    act/ rows of its Linears. Returns (stats, expert_info)."""
    from .model_profiles import profile_from_model
    from transformers.masking_utils import create_causal_mask
    device, dtype, model, base_model = ctx.device, ctx.dtype, ctx.model, ctx.base_model
    mtp_profile = profile_from_model(model)
    inner_mtp = mtp_profile.build_mtp_module(model.config)
    if inner_mtp is None:
        raise RuntimeError(
            f"profile '{mtp_profile.name}' declares has_mtp() but build_mtp_module() "
            "returned None; implement build_mtp_module() or set has_mtp() -> False")
    # The parent named `mtp` makes the qualified names the allocator's recipe names.
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device, dtype=dtype).eval()
    if not _load_mtp_source_for_probe(mtp_profile, inner_mtp, model_path):
        print("[incremental/mtp] no MTP weights in source", flush=True)
        return {}, {}
    tracked = [n for n, m in mtp_wrapper.named_modules()
               if isinstance(m, nn.Linear) and not re.search(r"mlp\.gate$", n)]
    expert_info = {k: v for k, v in discover_moe_structure(
        mtp_wrapper, profile=mtp_profile).items() if k in tracked}
    stats = {n: _linear_row(mtp_wrapper.get_submodule(n), expert_info.get(n)) for n in tracked}
    hooks = [mtp_wrapper.get_submodule(n).register_forward_hook(_input_hook(act_rows, n))
             for n in tracked]
    try:
        with torch.no_grad():
            for i in range(calib.size(0)):
                # MTP(embed_{t+1}, body_hidden_t) predicts token t+2.
                embed = embeds_cpu[i:i + 1].to(device, dtype=dtype)[:, 1:-1, :].contiguous()
                body = final_cpu[i:i + 1].to(device, dtype=dtype)[:, :-2, :].contiguous()
                B, T2, _ = embed.shape
                pos = torch.arange(T2, device=device).view(1, T2).expand(B, T2)
                inner_mtp(
                    inputs_embeds=embed, body_hidden_states=body,
                    position_embeddings=base_model.rotary_emb(
                        embed, pos.view(1, B, T2).expand(3, B, T2)),
                    causal_mask=create_causal_mask(
                        config=model.config, inputs_embeds=embed, attention_mask=None,
                        past_key_values=None, position_ids=pos),
                    position_ids=pos)
    finally:
        for handle in hooks:
            handle.remove()
    act_rows.flush()
    for name, entry in stats.items():
        entry["n_tokens_seen"] = int(act_rows.offsets.get(name, 0))
    print(f"[incremental/mtp] {len(tracked)} MTP Linears", flush=True)
    del mtp_wrapper, inner_mtp
    return stats, expert_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
#: The retired worker mode refuses at the CLI boundary before model reads.
#: Its complete execution snapshot is in archive/sample_parallel_probe_2026-10-15/.
SAMPLE_PARALLEL_RETIRED = (
    "sample-parallel probe is unavailable since 2026-09-02: its run contract "
    "and per-worker source census were built on the strict-Ada FP8-CB "
    "artifact census, archived with the Gridbook lane "
    "(archive/gridbook_lane_2026-09-02/). Run without "
    "--global-calibration-tensor/--sample-partition-index. Execution snapshot: "
    "archive/sample_parallel_probe_2026-10-15/."
)
SAMPLE_PARALLEL_RETIRED_HELP = (
    "Retired sample-parallel mode; see archive/sample_parallel_probe_2026-10-15/."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=0,
                    help="Calibration sample count. 0 (the default) uses every "
                         "line of the --dataset jsonl.")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--calib-seed", type=int, default=42,
                    help="Seed for the calibration text-subset shuffle and the "
                         "within-text window start. Default 42 reproduces "
                         "historical behavior.")
    ap.add_argument("--global-calibration-tensor", default=None,
                    help=SAMPLE_PARALLEL_RETIRED_HELP)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="Holds the streaming offload directory.")
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Probe the profile-declared MTP layers.")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Schedule the profile-declared visual blocks; text-only "
                         "staging gives them no rows.")
    ap.add_argument("--no-include-visual", action="store_false", dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Probe the profile-declared LM head.")
    ap.add_argument("--no-include-lm-head", action="store_false", dest="include_lm_head")
    ap.add_argument("--prefetch-lookahead",
                    default=os.environ.get("PREFETCH_LOOKAHEAD", "auto"),
                    help="Number of layers to queue ahead in the disk "
                         "prefetch pool, or 'auto' to bound lookahead by "
                         "the layer-cache budget and estimated layer size.")
    ap.add_argument("--prefetch-workers",
                    default=os.environ.get("PREFETCH_WORKERS", "auto"),
                    help="Number of concurrent layer prefetch workers, or "
                         "'auto' to derive from cache budget and layer size.")
    ap.add_argument("--prefetch-min-available-gb",
                    default=os.environ.get("PREFETCH_MIN_AVAILABLE_GB", "auto"),
                    help="Pause scheduling new prefetches below this "
                         "available-memory floor, or 'auto' for two "
                         "estimated layers with an 8 GiB minimum.")
    ap.add_argument("--activation-rows-limit", type=int,
                    default=int(os.environ.get("ACTIVATION_ROWS_LIMIT", "256")),
                    help="Maximum sampled activation rows to keep per Linear "
                         "for the cost stage. Lower values are useful for "
                         "debug runs on very wide MoE checkpoints.")
    args = ap.parse_args()
    pin_fla_configs()
    if args.global_calibration_tensor is not None:
        raise SystemExit(SAMPLE_PARALLEL_RETIRED)

    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("incremental_probe", args.device)

    n_layers = load_num_hidden_layers(args.model)
    schedule = build_shard_schedule(
        model_path=args.model, num_body_layers=n_layers, body_layers_per_shard=1,
        body_layer_range=(0, n_layers), include_mtp=args.include_mtp,
        include_visual=args.include_visual, include_lm_head=args.include_lm_head,
        unified_body_sweep=True,
    )
    scope = "(?:" + "|".join(f"(?:{r})" for r in schedule.regexes()) + ")"
    linear_exclude = resolve_linear_exclude(args.model)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    staged = stage_text_only(args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    ctx = _build_streaming_context(
        args.model, device=device, dtype=dtype,
        offload_folder=str(work_dir / "streaming_offload"), layers_per_shard=1,
        prefetch_workers=args.prefetch_workers,
        prefetch_min_available_gb=args.prefetch_min_available_gb,
        log_prefix="[incremental]",
    )
    nsamples = args.nsamples
    if nsamples == 0 and Path(args.dataset).is_file():
        with open(args.dataset) as f:
            nsamples = sum(1 for _ in f)
    nsamples = nsamples or 4  # non-jsonl datasets
    calib = load_calibration(tokenizer, args.dataset, nsamples, args.seqlen,
                             calib_seed=int(args.calib_seed))
    print(f"[incremental] calibration ready: {tuple(calib.shape)}", flush=True)
    raw = str(args.prefetch_lookahead).strip().lower()
    lookahead = (ctx.suggest_prefetch_lookahead() if raw in ("", "auto")
                 else max(1, int(raw)))
    act_rows = _ActRowCache(args.activation_cache_dir, args.activation_rows_limit)
    try:
        stats, payload, embeds_cpu, final_cpu = _probe_forward(
            ctx, calib=calib, scope=scope, exclude=linear_exclude,
            act_rows=act_rows, prefetch_lookahead=lookahead)
        if any(entry.kind == "mtp" for entry in schedule):
            mtp_stats, mtp_experts = _probe_mtp(
                ctx, calib=calib, act_rows=act_rows, embeds_cpu=embeds_cpu,
                final_cpu=final_cpu, model_path=args.model)
            stats.update(mtp_stats)
            payload["expert_info"].update(mtp_experts)
        act_rows.drain()
    finally:
        ctx.shutdown()
    # Body-coverage gate: a probe missing whole layers would silently drop
    # them in every later stage.
    covered = {int(m.group(1)) for name in stats
               for m in [re.match(re.escape(ctx.layers_prefix) + r"(\d+)\.", name)] if m}
    missing = sorted(set(schedule.body_layer_indices()) - covered)
    if missing:
        raise SystemExit(f"[incremental] FATAL: probe has no rows for body layers {missing}")
    calib_hash = calibration_data_hash(calib)
    meta = {
        "model": args.model, "dataset": args.dataset, "nsamples": int(calib.size(0)),
        "seqlen": int(args.seqlen), "calib_hash": calib_hash, "calib_hashes": [calib_hash],
        "dtype": args.dtype, "device_map": "streaming-layerwise",
        "top_k": read_top_k(ctx.model, default=2),
        "activation_cache_dir": args.activation_cache_dir,
        "activation_rows_limit": int(args.activation_rows_limit),
        "linear_include": scope, "linear_exclude": linear_exclude,
        "fla_configs": fla_config_identity(),
    }
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes
    atomic_write_bytes(Path(args.output), pickle.dumps(
        {"stats": stats, **payload, "meta": meta}, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[incremental] wrote probe to {args.output} ({len(stats)} rows)", flush=True)


if __name__ == "__main__":
    main()
