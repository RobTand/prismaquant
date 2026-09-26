"""AQUA-AURA: price the ACTIVATION side and merge it into a weight cost.

WHY THIS IS A SEPARATE STAGE
----------------------------
The allocator's cost was weight-only. Choosing NVFP4 does not just round the
weights to 4 bits -- on a lane that serves it fused, it is W4A4 and commits the
layer's ACTIVATIONS to 4 bits at serve time. FP8 commits them to 8. BF16 leaves
them alone. A weight-only surrogate is structurally blind to that difference
(NVFP4 and NVFP4A16 render weights bit-identically), so on such a lane the DP
was buying 4-bit formats at a discount to their true cost.

"ON A LANE THAT SERVES IT FUSED" IS LOAD-BEARING AND WAS ONCE MISSING HERE.
Whether activations are quantized is a property of the RUNTIME, not of the
format, and one lane can answer differently per FAMILY. Gridbook's CB runtime:

  * ``FP8_CB_*`` is genuinely W8A8 -- ``linear.py`` feeds quantized ``xq`` with
    per-token dynamic scales into ``native_cutlass_scaled_mm``, and ``moe.py``
    declares ``_FP8_GROUPED_CONTRACT = "fp8_per_token_dynamic"``. Real A-side.
  * ``NVFP4_CB_*`` is NOT -- it decodes to BF16 and runs a BF16 GEMM ("the
    exact native BF16 bridge") unless a PROCESS-GLOBAL env selector picks a
    fused mode, and every CB gate and gold serve leaves those unset. A-side
    exactly zero.

Charging the NVFP4 family a phantom A4 is not conservative, it is wrong in a
direction that costs bytes: it made FP8 look relatively cheap and the DSv4-Flash
92 GB body bought it, going from 96.8% ``nvfp4_cb`` (K16 bulk) to 25.4%
``FP8_CB`` with the bulk rung crushed K16 -> K12 -- paying weight bits to escape
a cost it never incurs. Keeping the REAL FP8 term is what stops the correction
from overshooting the other way. ``activation_dloss_table`` therefore REQUIRES
the lane's ``served_activation_quantization.executes`` (glob patterns over
format names) and refuses to guess it.

This stage uses a diagonal approximation that treats the activation term
separately from the weight term:

  * It needs NO render. ``activation_dloss`` reads the DENSE weight (as
    ``W[o,j]^2``), the card's ``g_sq_sum``, and the format's activation grid.
    The render basis never enters, so the number is identical whether the
    W-side was rendered with RTN or with the full GPTQ+JSO production recipe.
  * It therefore costs one streaming pass over the checkpoint and a row-blocked
    ``W^2 @ var`` per unit -- minutes, not the hours a render costs -- and it can
    be recomputed against any existing cost artifact without rebuilding it.

This computational separation does not establish that the actual errors are
independent. The approximation omits weight/activation interference and
channel correlations; its sensitivity card is not the KL/Gauss-Newton
adjoint used by AURA. Joint downstream pricing must project the full served
output residual before squaring (tracked in PrismaQuant #237).

That separate accounting is also the reason the term matters MORE than its size on an
RTN basis suggests: GPTQ and JSO shrink the W-side substantially and do nothing
whatever to the A-side. Measured on Qwen3.8-27B, production rendering cut
NVFP4's median W-side to 0.13x its RTN value while the A-side was unchanged, so
on the shipping render the activation term is several times the weight term for
the median Linear.

WHAT IT DOES NOT DO
-------------------
It does not choose formats and it does not rewrite an allocation. It writes one
number per (unit, format) into the cost rows and lets the DP do what it already
does. Hand-promoting the units it flags would be the post-allocator rewrite the
platform vetoes (principle 1): if the allocator picks something bad, the cost
model is what is wrong.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import pickle
import time

import numpy as np

from .allocator_candidates import (
    ACT_DLOSS_KEY, JointCellCoordinateError, cost_entry_is_joint_aura,
    joint_row_binds_cell)
from .schemas import refuse_retired_codebook_format

#: Return the CUDA pool to the OS once it has reserved this much. On GB10's
#: UNIFIED memory a reserved CUDA block IS host RAM, so it competes with the
#: numpy side rather than living in a separate budget.
CUDA_RESERVED_DRAIN_GIB = 8.0


def log(m: str) -> None:
    print(f"[aqua-cost] {m}", flush=True)


def build_weight_resolver(weight_map: dict, profile=None) -> dict:
    """Map a card unit name to its checkpoint tensor key.

    These differ, and not cosmetically. A card's unit names come from the module
    tree the probe walked (``model.layers.7.mlp.gate_proj``), while a multimodal
    checkpoint nests the text tower one level deeper
    (``model.language_model.layers.7.mlp.gate_proj.weight``). Matching naively
    resolves a handful of units and silently prices almost nothing, which reads
    as "the menu is unavailable" rather than as a name mismatch.

    So index every ``.weight`` key under its own base name AND a de-nested
    alias. The alias is registered with ``setdefault`` so a real ``model.layers``
    key stays authoritative over one.

    THAT IS NOT ENOUGH IN GENERAL, and the de-nesting alias above is the tell:
    it is one architecture's rename hardcoded here. Some checkpoints rename the
    LEAVES too, not just the path. DSv4-Flash stores
    ``model.layers.0.mlp.experts.0.down_proj`` as
    ``layers.0.ffn.experts.0.w2.weight`` -- no ``model.`` prefix, ``mlp``
    renamed to ``ffn``, and the Mixtral ``w1/w2/w3`` leaf convention. No amount
    of path aliasing recovers that, and the failure is silent: resolution drops
    to 0/33325, every unit becomes unpriced, and because ``cost_entry_act_dloss``
    defaults to 0.0 the DP cannot distinguish "unmeasured" from "free".

    So when a ``profile`` is supplied, invert its own
    ``checkpoint_to_live_name`` over the weight map. That mapping is the
    architecture's declared, tested source-of-truth -- the same one the probe,
    the cost stage and the exporter resolve names through -- rather than
    another rename guessed here (principle 2: no heuristics when an explicit
    exists). Note the *forward* direction is not usable for this:
    ``source_tensor_name`` rewrites the path but not the leaf, so it yields
    ``layers.0.ffn.experts.0.down_proj``, which is not a key.
    """
    # A packed routed-expert parameter is a bare 3-D tensor with NO ``.weight``
    # suffix -- ``...mlp.experts.gate_up_proj``, not ``....gate_up_proj.weight``
    # -- because it is an ``nn.Parameter`` on the experts module rather than an
    # ``nn.Linear``. Indexing only ``.weight`` keys therefore drops every expert
    # tensor, which on an A3B is 94% of the parameters. They resolved to nothing
    # and were reported as "units without an act price" -- an accurate count of
    # a silent hole.
    def _bases(key: str):
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        yield base
        if ".language_model." in base:
            yield base.replace("model.language_model.", "model.")

    idx: dict[str, str] = {}
    for key in weight_map:
        own = next(iter(_bases(key)))
        idx[own] = key
    for key in weight_map:
        for alias in list(_bases(key))[1:]:
            idx.setdefault(alias, key)
    if profile is not None:
        mapped = 0
        for key in weight_map:
            if not key.endswith(".weight"):
                continue
            try:
                live = profile.checkpoint_to_live_name(key)
            except Exception:
                continue
            if live and live.endswith(".weight"):
                # setdefault: a key that already indexes under its own name
                # stays authoritative; this only ADDS reachable aliases.
                if idx.setdefault(live[: -len(".weight")], key) is key:
                    mapped += 1
        log(f"profile {type(profile).__name__} contributed {mapped} "
            f"checkpoint-name aliases")
    return idx


#: Dtypes a checkpoint tensor may have and simply BE the dense weight. Anything
#: else is an encoding that must be decoded (or refused), never cast.
_DENSE_FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")


def materialize_source_weight(name: str, weight, scale, fp8_map):
    """Dense fp32 ``(out, in)`` weight from whatever the checkpoint stores.

    ``activation_dloss`` needs the SOURCE weight as a dense float matrix, but a
    native-quantized checkpoint does not store one: DSv4-Flash keeps its routed
    experts as MXFP4 nibble-packs (int8 ``(out, in/2)`` + per-32 E8M0 scales)
    and everything else as block-FP8 (e4m3 + a 128x128 E8M0 scale grid).
    Reading those bytes as a weight is not approximately right, it is garbage
    -- the packed expert has half the columns and fp8 codes cast to float land
    in the code range, not the value range.

    DISPATCH IS DECLARATION-DRIVEN, NEVER SHAPE-INFERRED. ``fp8_map`` is the
    streaming loader's own ``_build_fp8_scale_inv_map`` product -- the same map
    the probe, the cost stage and the exporter loaded this model through -- so
    membership (and ``mxfp4_names``, and the dequant ``block``) come from the
    checkpoint's declarations, and this function's output agrees with the
    weights the loader installs. Decoders are REUSED, not re-derived:

      * MXFP4: ``mxfp4_widen.dequantize_mxfp4_source`` (documented mirror of
        ``layer_streaming._read_layer_to_device`` step 3b -- same LUT, same
        low-nibble-first order, same 0xFF NaN rule), in fp32. The loader's
        installed tensor is this value downcast to bf16.
      * block-FP8: ``layer_streaming._dequant_fp8_block_weight`` with the
        map's declared block. Torch decodes the ``float8_e8m0fnu`` scale in
        ``.to(bfloat16)`` (0xFF -> NaN included), and the aligned path is the
        same bf16 tile multiply as the loader's batched step 3, so the bf16
        result is bit-identical to the loader install; the fp32 upcast is
        lossless. Note the ``.scale`` sibling is a MULTIPLIER (the loader
        multiplies by it), despite the legacy ``scale_inv`` spelling.

    Every mismatch RAISES -- a unit this stage cannot materialize must never
    be silently skipped, because an unpriced A-side reads as 0.0 (= free) to
    the DP, which is the exact mispricing the stage exists to remove. And the
    encodings are uniform per tensor class, so the first bad unit means a
    systematic bug: dying on it costs seconds, not a wrong 33k-unit artifact.
    """
    import torch

    from .layer_streaming import (_FLOAT8_DTYPES, _check_mxfp4_packed_grid,
                                  _dequant_fp8_block_weight,
                                  _fp8_dequant_block)
    from .mxfp4_widen import dequantize_mxfp4_source

    entry = fp8_map.get(name + ".weight") if fp8_map else None
    is_dense_float = str(weight.dtype).split(".")[-1] in _DENSE_FLOAT_DTYPES
    if entry is None:
        if is_dense_float:
            return weight.to(torch.float32)
        raise RuntimeError(
            f"{name}: source tensor has dtype {weight.dtype} but no scale "
            f"entry in the streaming loader's fp8/mxfp4 map "
            f"(_build_fp8_scale_inv_map). Casting these bytes to float would "
            f"install code-range values, not weights -- the historical "
            f"fp8-range bug. Check the checkpoint's scale-sibling naming "
            f"against the model profile's fp8_scale_pairs.")
    if scale is None:
        raise RuntimeError(
            f"{name}: the scale map declares a sibling for this weight but "
            f"none was supplied to materialize_source_weight")
    if is_dense_float:
        raise RuntimeError(
            f"{name}: source tensor is already a dense float "
            f"({weight.dtype}) yet the scale map has an entry for it; "
            f"applying a block scale to a dense weight would corrupt it. "
            f"The map and the checkpoint disagree -- refusing to guess.")
    if name + ".weight" in getattr(fp8_map, "mxfp4_names", frozenset()):
        _check_mxfp4_packed_grid(name + ".weight", weight, scale)
        return dequantize_mxfp4_source(weight, scale, dtype=torch.float32)
    if weight.dtype not in _FLOAT8_DTYPES:
        raise RuntimeError(
            f"{name}: mapped as block-FP8 but the tensor dtype is "
            f"{weight.dtype}, not a float8 wire; the declaration and the "
            f"bytes disagree.")
    return _dequant_fp8_block_weight(
        weight, scale, block=_fp8_dequant_block(fp8_map),
        name=name).to(torch.float32)


def cached_act_path(act_dir: str, name: str) -> str:
    """Where the probe parked this Linear's real input rows."""
    return os.path.join(act_dir, name.replace(".", "__") + ".pt")


def measured_act_var(spec, x_cpu, device: str):
    """Per-input-channel error variance on the layer's REAL activations.

    The synthetic path this replaces samples independent per-channel Gaussians.
    That reproduces every channel's marginal exactly and destroys the joint --
    and the joint is what an NVFP4 block scale is a function of, since 16
    consecutive channels share one FP8 scale set by the largest magnitude among
    them in that token. Real rows carry the co-occurrence; a Gaussian batch
    cannot.

    LANDMINE: the cached tensors are CPU-resident. Without the explicit
    ``.to(device)`` the quantizer runs on CPU at full numerical fidelity and no
    speed, which reads as "slow GPU" rather than "wrong device".
    """
    import torch
    x = x_cpu.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        xq = spec.activation_quantize_dequantize(x)
        per_channel = ((x.float() - xq.float()) ** 2).mean(dim=0)
    return per_channel.double().cpu().numpy()


def resolve_executed_activation_formats(*, lane_id: str | None,
                                        executes_all: bool = False):
    """The formats whose activation grid the SERVING LANE executes.

    One resolver so every A-side caller reaches the same authority. Returns
    either the string ``"all"`` or a ``frozenset`` suitable for
    ``activation_dloss_table(executed_activation_formats=...)``.

    Refuses rather than defaulting: "which formats does the runtime quantize
    activations for" has no safe default, and the unsafe one (trust the format
    registry) is the mispricing this exists to prevent.
    """
    if executes_all and lane_id:
        raise SystemExit(
            "REFUSE: --serving-lane and --lane-executes-all-activation-grids "
            "are mutually exclusive; the lane spec is the authority when a "
            "lane is named.")
    if executes_all:
        log("serving lane executes EVERY format's activation grid (asserted); "
            "pricing the full A-side")
        return "all"
    if not lane_id:
        raise SystemExit(
            "REFUSE: name the serving lane (--serving-lane) so the A-side is "
            "priced against the activation contract the runtime actually "
            "executes, or assert "
            "--lane-executes-all-activation-grids. There is no default: "
            "assuming the format registry's W4A4 claim is what priced a full "
            "A-side onto the BF16-bridge CB lane that Gridbook served until "
            "2026-09-02 (archive/gridbook_lane_2026-09-02/). The lane is "
            "retired; the failure mode is not, because it is a property of "
            "trusting the registry over the lane.")
    from .lane_spec import load_lane_spec
    spec = load_lane_spec(lane_id)
    contract = spec.served_activation_quantization
    if contract is None:
        raise SystemExit(
            f"REFUSE: lane {lane_id!r} does not declare "
            f"`served_activation_quantization`, so which formats it executes "
            f"activation quantization for is unknown. Declare it on "
            f"prismaquant/lane_specs/{lane_id}.json -- an empty `executes` "
            f"list is a valid and common answer.")
    log(f"serving lane {lane_id!r} executes activation quantization for: "
        f"{sorted(contract.executes) or '(nothing)'}")
    if contract.rationale:
        log(f"  rationale: {contract.rationale}")
    return frozenset(contract.executes)


def required_activation_formats(formats, *, shape, device, executes_all,
                                patterns):
    """``(required, non_act, not_executed, unbuildable)`` for one unit shape.

    Whether this stage OWES an A-side price for a format is a conjunction, and
    it is the same conjunction the dense path has always enforced:

      * the format quantizes activations -- ``descriptor.quantizes_activations``,
        a dtype-level fact, not the registry's claim about what any runtime
        does; AND
      * the named SERVING LANE executes that format's activation grid.

    Factored out because this stage now has THREE weight sources -- one key, a
    packed 3-D tensor, and one 2-D tensor per routed expert -- and a second copy
    of this rule is how one of them quietly starts pricing a different lane.

    ``required`` is ``[(fmt, plugin)]``. ``non_act`` and ``not_executed`` name
    the formats that are correctly FREE, each for its own reason (leaves
    activations alone / that lane never runs this grid); folding the two sets
    together would make them indistinguishable in the report. ``unbuildable`` is
    ``[(fmt, error)]`` -- a plugin that cannot be built at this shape is a HOLE,
    never a free price.
    """
    from .format_cost_registry import RegistryFormatPlugin

    non_act: set[str] = set()
    not_executed: set[str] = set()
    unbuildable: list[tuple[str, str]] = []
    required: list[tuple[str, object]] = []
    for fmt in formats:
        # A stale cost.pkl can still name a retired codebook rung (archived
        # 2026-09-25, #1304). It refuses here: the handler below would record
        # it as an unbuildable hole and let the run finish.
        refuse_retired_codebook_format(fmt)
        try:
            plugin = RegistryFormatPlugin.build(fmt, shape=shape, device=device)
        except Exception as exc:
            unbuildable.append((fmt, str(exc)))
            continue
        if not plugin.descriptor.quantizes_activations:
            non_act.add(fmt)
            continue
        if not executes_all and not any(
                fnmatch.fnmatchcase(fmt, pat) for pat in patterns):
            not_executed.add(fmt)
            continue
        required.append((fmt, plugin))
    return required, non_act, not_executed, unbuildable


def _close_safetensor_handle(handle) -> None:
    """Release one open shard's mmap.

    ``safetensors.safe_open`` exposes no ``close()``; its context-manager exit
    is the only way to drop the file. Resident mapped pages are the resource
    being bounded here, so a missing exit must RAISE rather than silently leave
    every shard of the checkpoint resident.
    """
    exit_ = getattr(handle, "__exit__", None)
    if exit_ is None:
        raise RuntimeError(
            "the installed safetensors reader exposes no way to close a shard "
            "handle; the per-expert bridge cannot bound resident mmap without "
            "it")
    exit_(None, None, None)


class _CheckpointShardHandles:
    """Bounded LRU of open shards, for key-major (not shard-major) reads.

    The dense loop reads one shard at a time by construction, which is what
    bounds resident mmap to that shard (measured 7.1 -> 48.2 GiB on this model
    before it did). The per-expert bridge cannot be shard-major -- one packed
    unit's keys are spread across the experts and the reduction is expert-major
    -- so the SAME bound is kept by the same means: a closed handle releases its
    faulted pages. ``capacity`` is explicit instead of an unbounded dict holding
    every shard in the checkpoint at once.
    """

    def __init__(self, model_path: str, weight_map: dict, fp8_map=None,
                 capacity: int = 2):
        self._model_path = model_path
        self._weight_map = weight_map
        self._fp8_map = fp8_map
        self._capacity = max(1, int(capacity))
        self._open: collections.OrderedDict = collections.OrderedDict()
        self._keys: dict[str, frozenset] = {}

    def _handle(self, shard: str):
        from safetensors import safe_open

        handle = self._open.get(shard)
        if handle is None:
            handle = safe_open(os.path.join(self._model_path, shard),
                               framework="pt", device="cpu")
            self._open[shard] = handle
            if self._fp8_map:
                self._keys[shard] = frozenset(handle.keys())
            while len(self._open) > self._capacity:
                evicted_shard, evicted = self._open.popitem(last=False)
                _close_safetensor_handle(evicted)
                self._keys.pop(evicted_shard, None)
        self._open.move_to_end(shard)
        return handle

    def __call__(self, key: str):
        """The dense source weight at one checkpoint ``key``, materialized.

        Reuses :func:`materialize_source_weight` -- the same declaration-driven
        dispatch the dense path uses -- so a per-expert tensor out of a
        quantized source is decoded by the streaming loader's own decoder rather
        than cast from its wire bytes.
        """
        weight_shard = self._weight_map[key]
        handle = self._handle(weight_shard)
        weight = handle.get_tensor(key)
        scale = None
        entry = self._fp8_map.get(key) if self._fp8_map else None
        if entry is not None:
            scale_shard, scale_key = entry
            scale_handle = self._handle(scale_shard)
            scale = scale_handle.get_tensor(scale_key)
        return materialize_source_weight(key[: -len(".weight")], weight, scale,
                                         self._fp8_map)

    def close(self) -> None:
        for handle in self._open.values():
            _close_safetensor_handle(handle)
        self._open.clear()
        self._keys.clear()


# ---------------------------------------------------------------------------
# Per-expert checkpoint layouts
# ---------------------------------------------------------------------------
# ``build_weight_resolver`` maps one card unit to ONE checkpoint key, which is
# right for a packed routed-expert parameter stored as a bare 3-D tensor
# (``...mlp.experts.gate_up_proj``).  GLM-5.3-Flash does not store it that way:
# every expert is its own 2-D ``nn.Linear`` weight
# (``...mlp.experts.{e}.gate_proj.weight``), and the fused gate/up pair is two
# tensors, not one.  So the 84 packed units of a 45-layer GLM body resolve to
# nothing -- 97% of the parameters -- and because ``cost_entry_act_dloss``
# defaults to 0.0 the DP cannot tell "unmeasured" from "free".  On a lane whose
# own attested contract says NVFP4 is "Real A4 on BOTH the dense and the
# packed-expert route", that is the DSv4 mispricing with the sign flipped: it
# makes the W4A4 rung look free exactly where it is not.
#
# The A-side math for packed units already exists and is tested
# (``_activation_dloss_packed``); only the checkpoint layout does not reach it.
# What follows is the bridge, and it STREAMS rather than stacking: the packed
# sum
#
#     dLoss ~= 0.5 / T_global * sum_e sum_o g_sq[e,o] * sum_j W[e,o,j]^2 var[e,j]
#
# is separable over ``e``, so a [288, 4096, 4096] gate_up never has to exist --
# it would be 19 GiB in float32, per format.  Each expert is promoted to
# float32 one at a time and reduced with ``_weighted_row_sum``, the same kernel
# and the same float64 accumulation the packed path uses, so this is the
# production quantity computed in a different order, not a second estimator.

def per_expert_weight_keys(unit_name: str, weight_map: dict, *,
                           n_experts: int) -> list[list[str]] | None:
    """Checkpoint keys for one packed unit, as ``[expert][sibling]``.

    Returns ``None`` when the layout is not per-expert, so a caller can fall
    back to the single-key resolver without a special case.
    """
    if ".mlp.experts." not in unit_name:
        return None
    stem, _, leaf = unit_name.rpartition(".mlp.experts.")
    siblings = {"gate_up_proj": ("gate_proj", "up_proj"),
                "down_proj": ("down_proj",)}.get(leaf)
    if siblings is None:
        return None
    keys: list[list[str]] = []
    for expert in range(n_experts):
        row = [f"{stem}.mlp.experts.{expert}.{s}.weight" for s in siblings]
        if any(k not in weight_map for k in row):
            return None
        keys.append(row)
    return keys


def packed_act_dloss_per_expert(unit, keys: list[list[str]], model_path: str,
                                act_var, *, gain: float = 1.0,
                                handles=None) -> float:
    """``_activation_dloss_packed`` over a per-expert checkpoint, streamed.

    ``keys[e]`` are the sibling tensors of expert ``e``, concatenated along the
    output axis in the order vLLM fuses them (gate then up) -- the same order
    ``expert_g_sq_sum``'s rows are indexed in, which is why the concatenation
    may not be reordered.
    """
    import numpy as np
    import torch

    from .format_cost_protocol import _row_chunk, _weighted_row_sum

    g_all = np.asarray(unit.expert_g_sq_sum, dtype=np.float64)
    var = np.asarray(act_var, dtype=np.float64)
    n_e = int(g_all.shape[0])
    if len(keys) != n_e:
        raise ValueError(f"{unit.topology.name}: {len(keys)} experts in the "
                         f"checkpoint, {n_e} in the card")
    if var.shape == (unit.in_features,):
        var = np.broadcast_to(var, (n_e, unit.in_features))
    elif var.shape != (n_e, unit.in_features):
        raise ValueError(f"{unit.topology.name}: packed act_var shape "
                         f"{var.shape}, expected {(n_e, unit.in_features)} "
                         f"or {(unit.in_features,)}")

    rows_per_chunk = _row_chunk(unit.in_features)
    total = 0.0
    for e, row in enumerate(keys):
        parts = [handles(k) for k in row]
        w_e = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if tuple(w_e.shape) != (unit.out_features, unit.in_features):
            raise RuntimeError(
                f"{unit.topology.name}: expert {e} materialized "
                f"{tuple(w_e.shape)}, expected "
                f"{(unit.out_features, unit.in_features)}")
        w_e = w_e.to(torch.float32)
        g_e, v_e = g_all[e], var[e]
        for lo in range(0, unit.out_features, rows_per_chunk):
            hi = min(lo + rows_per_chunk, unit.out_features)
            total += _weighted_row_sum(w_e[lo:hi], v_e, g_e[lo:hi])
        del parts, w_e
    return 0.5 * (total / max(1, unit.n_tokens)) * float(gain)


def activation_dloss_table(card, model_path: str, formats: list[str], *,
                           device: str = "cpu", names=None,
                           act_dir: str | None = None,
                           profile=None,
                           executed_activation_formats=None,
                           already_priced_cells=None,
                           formats_by_name=None,
                           ) -> tuple[dict, dict, dict]:
    """``{unit: {format: act_dloss}}`` plus a report of what could not be priced.

    A format that does not quantize activations is simply absent from a unit's
    inner dict -- that is not a hole, it is the correct answer (BF16 costs
    nothing on the A-side). A format that DOES quantize activations but could
    not be priced is recorded in ``holes``, because an unpriced A-side read as
    zero is the exact mispricing this stage exists to remove.

    ``executed_activation_formats`` is REQUIRED and is the set of formats whose
    activation grid the SERVING LANE actually executes -- normally
    ``LaneSpec.served_activation_quantization.executes``. The format registry
    can only say whether a format *is* W4A4; it cannot know whether the runtime
    that will serve this artifact runs the fused kernel or decodes to BF16 and
    runs a BF16 GEMM. Those are different questions and this stage needs the
    second one. Passing ``"all"`` asserts that the lane executes every format's
    activation grid, and is the correct answer for a plain fused-W4A4 lane.

    It has no default on purpose. Defaulting to the registry's claim is what
    priced a full A-side onto the nvfp4_cb lane -- which serves on gridbook's
    exact BF16 bridge and quantizes no activations at all -- and cost the
    DSv4-Flash 92 GB body the majority of its codebook rung (K16 -> K12) buying
    FP8 promotions to escape a cost of zero.

    ``already_priced_cells`` names the ``(unit, format)`` cells whose cost is
    ALREADY in the artifact under a currency that carries its own activation
    term -- a joint AURA row's signed residual. Nothing is built, read or
    priced for them: computing an A-side there is work whose result the merge
    discards (and would be refused), and excluding them before the shard read is
    what makes an all-joint artifact cost nothing rather than a full pass. This
    is a per-CELL selection, so a mixed artifact prices exactly the legacy cells
    and leaves the joint ones alone.

    ``formats_by_name`` is the other half of that selection and the reason the
    first version of it was wrong: the REQUESTED formats are not global. A
    format the artifact does not carry for a unit is not a cell of that unit --
    nothing could ever be merged there -- so asking the global list "minus the
    joint ones" manufactures cells that do not exist, prices them, and reports
    the unit as needing work it does not need. The caller that owns the artifact
    supplies each unit's own format list; a unit absent from the mapping has no
    requested cell at all and is not counted as fulfilled by anything.
    """
    if executed_activation_formats is None:
        raise SystemExit(
            "REFUSE: executed_activation_formats is required. The A-side price "
            "depends on what the SERVING LANE executes, not on what the format "
            "registry declares the format to be -- gridbook's CB lane "
            "served NVFP4_CB weights through the exact BF16 bridge and "
            "quantized no activations, so its correct A-side was exactly "
            "zero (that lane retired 2026-09-02, "
            "archive/gridbook_lane_2026-09-02/; the example is kept because "
            "it is the measurement that made this argument mandatory). Pass "
            "the "
            "lane's `served_activation_quantization.executes` (an empty set is "
            "a valid, common answer), or the string \"all\" for a lane that "
            "genuinely serves every format's activation grid fused.")
    executes_all = executed_activation_formats == "all"
    # GLOB PATTERNS, not names: the answer is per FAMILY and rungs are
    # open-ended. `FP8_CB_*` must keep covering a rung added tomorrow, or the
    # A-side silently vanishes for it -- the same silent default this argument
    # exists to remove.
    patterns = (("*",) if executes_all
                else tuple(executed_activation_formats))
    if not executes_all and not patterns:
        raise SystemExit(
            "REFUSE: this lane executes NO format's activation quantization, "
            "so every A-side price is exactly zero and merging one would only "
            "overcharge the DP. Allocate from the weight-only cost instead of "
            "building an A-side that the served artifact never pays. "
            "(If a fused activation mode is being enabled for this artifact, "
            "declare the formats it executes on the lane spec first.)")
    import torch
    from safetensors import safe_open

    from .format_cost_protocol import (
        price_activation_only, resolve_act_quant_variance,
    )

    with open(os.path.join(model_path, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    resolver = build_weight_resolver(weight_map, profile=profile)

    wanted = list(names) if names is not None else [u.topology.name
                                                   for u in card.units()]
    already = frozenset(already_priced_cells or ())
    resolvable = [n for n in wanted if n in resolver]
    # A packed routed-expert unit whose checkpoint stores one 2-D tensor PER
    # expert has no single key for ``build_weight_resolver`` to return, so it
    # lands outside ``resolvable``. Before this classification existed it landed
    # outside the REPORT too -- no price, no hole -- and ``cost_entry_act_dloss``
    # reads that absence as 0.0, i.e. as a free activation exactly where the
    # served lane is W4A4. So classify, never drop:
    #
    #   * per-expert layout -> priced below by the streaming bridge;
    #   * anything else     -> a HOLE for every format this lane executes,
    #                          reported per format like any other unpriced unit.
    #
    # Missing evidence is never a zero HERE, but a hole is a REPORT and not a
    # refusal: the refusal below stays all-or-nothing (no coverage threshold is
    # invented), and a positive weight-only row with no A-side is still admitted
    # downstream -- `cost_entry_prices_unmeasured_activation_at_zero` only
    # removes rows priced at exactly 0.0. Requiring coverage for a campaign is
    # an explicit policy decision, recorded in docs/ARCHITECTURE.md rather than
    # improvised here.
    units: dict[str, object] = {}
    per_expert: dict[str, list[list[str]]] = {}
    unresolved: list[str] = []
    needed_formats: dict[str, list[str]] = {}
    already_priced: list[str] = []
    no_cells: list[str] = []
    for name in wanted:
        # The REQUESTED cells of THIS unit. With `formats_by_name` supplied by
        # the caller that owns the cost artifact, a format the artifact does not
        # carry for this unit is not a cell: pricing it would be a price that
        # nothing can consume. Without the mapping (the library call), the global
        # list is what the caller asked for.
        unit_formats = (list(formats_by_name.get(name, ()))
                        if formats_by_name is not None else list(formats))
        if not unit_formats:
            no_cells.append(name)
            continue
        # (unit, format) cells the caller says already carry a price with their
        # OWN activation term -- a joint AURA row's signed residual. They are
        # dropped BEFORE anything is read or built: pricing one would be work
        # whose result is discarded, and merging one is refused. This is the
        # only place the selection happens, so a mixed artifact prices exactly
        # the legacy cells and nothing else.
        needed = [f for f in unit_formats if (name, f) not in already]
        if not needed:
            already_priced.append(name)
            continue
        needed_formats[name] = needed
        if name in resolver:
            continue
        try:
            unit = card[name]
        except (KeyError, TypeError, IndexError, AttributeError):
            unit = None
        units[name] = unit
        n_experts = getattr(unit, "n_experts", None)
        keys = (per_expert_weight_keys(name, weight_map,
                                       n_experts=int(n_experts))
                if isinstance(n_experts, int) and n_experts > 0 else None)
        if keys is None:
            unresolved.append(name)
        else:
            per_expert[name] = keys
    resolvable = [n for n in resolvable if n in needed_formats]
    log(f"weight-key resolution: {len(resolvable)}/{len(wanted)} card units "
        f"found in the checkpoint, {len(per_expert)} reachable only through "
        f"the per-expert layout, {len(unresolved)} unresolved, "
        f"{len(already_priced)} already carrying a joint A-side")
    # Refuse rather than write a no-op. "Nothing resolved" is never a valid
    # outcome for this stage, and the artifact it would otherwise produce is
    # indistinguishable from a real one -- same units, same formats, an A-side
    # that is absent and therefore read as 0.0 (free) by the DP. Only the
    # unambiguous case is a refusal; no coverage threshold is invented here,
    # because any such number would be a heuristic (principle 2). Partial
    # coverage is already reported per-format through `holes`.
    if (wanted and not resolvable and not per_expert and not already_priced
            and not no_cells):
        raise SystemExit(
            f"REFUSE: 0 of {len(wanted)} card units resolve to a checkpoint "
            f"tensor, so there is nothing to price. This is a NAME-SPACE "
            f"mismatch, not an empty menu: card units come from the module "
            f"tree the probe walked, while the checkpoint may rename both the "
            f"path and the leaf. Pass a model profile (--model-path must be a "
            f"directory `detect_profile` recognises) so the architecture's own "
            f"`checkpoint_to_live_name` can supply the aliases. Sample card "
            f"name {wanted[0]!r}; sample checkpoint key "
            f"{next(iter(weight_map))!r}."
        )

    # The streaming loader's own scale map -- fp8/mxfp4 declarations, scale
    # sibling keys, and the checkpoint-declared dequant block. Empty for a
    # dense checkpoint, in which case every path below is byte-identical to
    # the pre-quantized-source behaviour (plain get_tensor + fp32 cast).
    from .layer_streaming import _build_fp8_scale_inv_map
    fp8_map = _build_fp8_scale_inv_map(model_path)
    if fp8_map:
        log(f"quantized-source checkpoint: {len(fp8_map)} scale-mapped "
            f"weights ({len(fp8_map.mxfp4_names)} declared MXFP4, dequant "
            f"block {fp8_map.block}); units will be materialized through "
            f"the streaming loader's decoders")

    # SHARD-AT-A-TIME. ``safe_open(device="cpu")`` mmaps the shard and
    # ``get_tensor`` faults its pages in; while the handle lives those pages stay
    # RESIDENT. Holding every handle grows RSS by the full bf16 size of every
    # tensor touched -- measured at 7.1 -> 48.2 GiB over 500 units on this model,
    # which is the body's weights almost exactly. Grouping by shard and closing
    # each handle bounds resident mmap to ONE shard. Order is irrelevant: each
    # unit is priced independently.
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name in resolvable:
        by_shard[weight_map[resolver[name]]].append(name)
    log(f"pricing the A-side of {len(resolvable)} units across "
        f"{len(by_shard)} shards, one shard resident at a time")

    table: dict[str, dict[str, float]] = {}
    holes: dict[str, list[str]] = collections.defaultdict(list)
    non_act: set[str] = set()
    not_executed: set[str] = set()
    # The same two answers, kept PER (unit, format) because that is the
    # granularity the coverage requirement is stated at: a format can be built
    # (and be a passthrough, or a grid this lane never runs) at one unit's
    # shape and UNBUILDABLE at another's, and the union of the names would
    # exempt the second unit's cell from the gate while the stage itself calls
    # it a hole. ``non_act``/``not_executed`` stay as the readable summary of
    # what was seen; this map is what an exemption is read from.
    exempt_cells: dict[str, dict[str, str]] = {}

    def _record_exempt(name: str, na, ne_) -> None:
        for fmt in na:
            exempt_cells.setdefault(name, {})[fmt] = "activation_identity"
        for fmt in ne_:
            exempt_cells.setdefault(name, {})[fmt] = "not_executed_by_lane"

    t0 = time.time()
    done = 0
    var_source = collections.Counter()
    for shard in sorted(by_shard):
        with safe_open(os.path.join(model_path, shard),
                       framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys()) if fp8_map else frozenset()
            for name in by_shard[shard]:
                unit = card[name]
                raw = handle.get_tensor(resolver[name])
                scale_t = None
                entry = fp8_map.get(name + ".weight") if fp8_map else None
                if entry is not None:
                    scale_shard, scale_key = entry
                    if scale_key in shard_keys:
                        scale_t = handle.get_tensor(scale_key)
                    else:
                        # Scale sibling serialized in a different shard than
                        # its weight -- rare, but the map records the shard
                        # so honour it rather than assume co-location.
                        with safe_open(scale_shard, framework="pt",
                                       device="cpu") as sh:
                            scale_t = sh.get_tensor(scale_key)
                w_t = materialize_source_weight(name, raw, scale_t, fp8_map)
                # A packed routed-expert unit is [E, out, in]: ONE decision
                # unit, E matrices. `out_features`/`in_features` describe one
                # expert's slice (that is what the probe recorded), so the
                # expected rank depends on whether the card carries per-expert
                # statistics for it.
                n_experts = unit.n_experts
                want = ((unit.out_features, unit.in_features)
                        if n_experts is None else
                        (n_experts, unit.out_features, unit.in_features))
                if tuple(w_t.shape) != want:
                    raise RuntimeError(
                        f"{name}: materialized weight has shape "
                        f"{tuple(w_t.shape)}, expected {want} from the "
                        f"probe stats (source dtype {raw.dtype}, "
                        f"{'scale-mapped' if entry else 'dense'}); refusing "
                        f"to price a wrong-shaped W")
                # Uploaded ONCE per unit and reused by every candidate format:
                # the A-side reduction is GPU work (principle 7), and pricing a
                # 3-format menu off a host array would either copy the tensor
                # three times or do 70 G float64 multiply-adds on the CPU. On a
                # packed expert tensor that is 2 GiB of copy per format.
                w_np = w_t.to(device=device, dtype=torch.float32) \
                    if torch.cuda.is_available() and device != "cpu" \
                    else w_t.to(torch.float32).numpy()
                del raw, scale_t, w_t
                # Real input rows if the probe cached them for this Linear.
                # Loaded once per unit and reused across formats -- the tensor
                # is the same batch, only the quantizer differs.
                x_cpu = None
                # Packed units deliberately do NOT take the measured path.
                # The probe caches the experts MODULE's input -- every token
                # before routing -- so a measured variance from it is one
                # population figure shared by all E experts, which is exactly
                # the routing structure this unit needs to keep. The per-expert
                # synthetic fit (source 2) is used instead: it loses the
                # cross-channel joint but keeps the per-expert distribution,
                # and routing is the larger effect on a 256-expert layer.
                if act_dir and n_experts is None:
                    p = cached_act_path(act_dir, name)
                    if os.path.exists(p):
                        blob_ = torch.load(p, map_location="cpu",
                                           weights_only=False)
                        cand = blob_.get("inputs")
                        # A shape mismatch means the cache is from a different
                        # model/shape; silently pricing on it would be worse
                        # than falling back, so require the match.
                        if (cand is not None
                                and cand.ndim == 2
                                and cand.shape[1] == unit.in_features):
                            x_cpu = cand
                row: dict[str, float] = {}
                # Which formats this lane OWES a price for, in one place shared
                # with the per-expert bridge below -- "leaves activations alone"
                # and "this lane does not execute that grid" are both correctly
                # free, but they are different answers and are reported apart.
                required, na, ne_, unbuildable = required_activation_formats(
                    needed_formats[name],
                    shape=(unit.out_features, unit.in_features),
                    device=device, executes_all=executes_all, patterns=patterns)
                non_act |= na
                not_executed |= ne_
                _record_exempt(name, na, ne_)
                for fmt, message in unbuildable:
                    holes[fmt].append(f"{name}: unbuildable ({message})")
                for fmt, plugin in required:
                    v = None
                    if x_cpu is not None:
                        v = measured_act_var(plugin.spec, x_cpu, device)
                    if v is not None:
                        var_source["measured"] += 1
                    elif n_experts is None:
                        var_source["modelled"] += 1
                    else:
                        var_source["modelled_per_expert"] += 1
                    a = price_activation_only(unit, w_np, plugin, act_var=v)
                    if a is None:
                        holes[fmt].append(name)
                    else:
                        row[fmt] = float(a)
                    del plugin
                if row:
                    table[name] = row
                del w_np, x_cpu
                done += 1
                if torch.cuda.is_available() and (
                        torch.cuda.memory_reserved() / (1 << 30)
                        >= CUDA_RESERVED_DRAIN_GIB):
                    torch.cuda.empty_cache()
                if done % 100 == 0:
                    log(f"  priced {done}/{len(resolvable)} "
                        f"({time.time() - t0:.0f}s)")
    if per_expert:
        # STREAMED, one expert at a time, through the SAME variance resolver and
        # the same float64 row kernel the packed 3-D path uses -- the
        # equivalence is pinned by tests/test_aqua_per_expert_checkpoint.py.
        # Nothing here builds the [E, M, N] tensor: for GLM-5.3-Flash that is
        # 19 GiB per format, and it is the reason the bridge exists.
        log(f"pricing {len(per_expert)} per-expert packed units "
            f"(streamed, one expert at a time)")
        handle_cache = _CheckpointShardHandles(model_path, weight_map, fp8_map)
        try:
            for name in sorted(per_expert):
                unit = units[name]
                keys = per_expert[name]
                row = {}
                required, na, ne_, unbuildable = required_activation_formats(
                    needed_formats[name],
                    shape=(unit.out_features, unit.in_features),
                    device=device, executes_all=executes_all, patterns=patterns)
                non_act |= na
                not_executed |= ne_
                _record_exempt(name, na, ne_)
                for fmt, message in unbuildable:
                    holes[fmt].append(f"{name}: unbuildable ({message})")
                for fmt, plugin in required:
                    var = resolve_act_quant_variance(unit, plugin)
                    if var is None:
                        # Same answer as the dense path's `price_activation_only`
                        # returning None: a HOLE, never a free activation.
                        holes[fmt].append(name)
                        del plugin
                        continue
                    var_source["modelled_per_expert"] += 1
                    try:
                        a = packed_act_dloss_per_expert(
                            unit, keys, model_path, var, handles=handle_cache)
                    except Exception as exc:
                        # A layout/roster/shape disagreement is a hole on THIS
                        # unit, named, rather than a price invented for it.
                        holes[fmt].append(f"{name}: {exc}")
                        del plugin
                        continue
                    row[fmt] = float(a)
                    del plugin
                if row:
                    table[name] = row
                done += 1
                if done % 100 == 0:
                    log(f"  priced {done}/{len(wanted)} "
                        f"({time.time() - t0:.0f}s)")
        finally:
            handle_cache.close()
    if unresolved:
        # Reported, never silently dropped. These units cannot be priced at all
        # -- no single key and no per-expert roster -- so every format this lane
        # executes is a HOLE for each of them. A unit whose card entry is
        # missing entirely cannot even be classified per format, so it is named
        # once here; either way the artifact says "unpriced", not "free".
        log(f"unresolved units (no single checkpoint key, no per-expert "
            f"roster): {len(unresolved)}")
        for name in unresolved:
            unit = units.get(name)
            if unit is None:
                for fmt in formats:
                    holes[fmt].append(f"{name}: no card entry to classify")
                continue
            required, na, ne_, unbuildable = required_activation_formats(
                needed_formats[name],
                shape=(unit.out_features, unit.in_features),
                device=device, executes_all=executes_all, patterns=patterns)
            non_act |= na
            not_executed |= ne_
            _record_exempt(name, na, ne_)
            for fmt, message in unbuildable:
                holes[fmt].append(f"{name}: unbuildable ({message})")
            for fmt, plugin in required:
                holes[fmt].append(name)
                del plugin
    log(f"A-side priced for {len(table)} units in {time.time() - t0:.0f}s")
    if var_source:
        log(f"act_var source: {dict(var_source)} (measured = real cached "
            f"activations; modelled = per-channel Gaussian fit; "
            f"modelled_per_expert = the same fit, one per routed expert)")
    if non_act:
        log(f"formats that leave activations alone (correctly unpriced): "
            f"{sorted(non_act)}")
    if not_executed:
        log(f"formats that quantize activations but THIS LANE DOES NOT EXECUTE "
            f"(correctly unpriced -- served A-side is exactly zero): "
            f"{sorted(not_executed)}")
    for fmt, names_ in sorted(holes.items()):
        log(f"HOLE: {fmt} quantizes activations but {len(names_)} units could "
            f"not be priced; those rows keep a weight-only cost. "
            f"e.g. {names_[:3]}")
    return (table, {k: v for k, v in holes.items()},
            {"act_var_source": dict(var_source),
             "per_expert_units_priced": len(per_expert),
             "units_unresolved": len(unresolved),
             "units_without_requested_cells": len(no_cells),
             "units_fully_joint_priced": len(already_priced),
             # The two ways a requested cell is free of an A-side BY CONTRACT,
             # reported apart because they are different answers -- and because
             # the campaign-coverage gate below has to subtract them from the
             # unfulfilled set rather than call BF16 a hole. Descriptive only:
             # these are the names seen free SOMEWHERE, and a name is not an
             # exemption. The gate reads ``coverage_exempt_cells``, which is per
             # (unit, format), so a format that is a passthrough at one unit's
             # shape and unbuildable at another's still keeps the second unit's
             # cell a hole. A format the registry cannot build for a shape
             # appears in neither.
             "activation_identity_formats": sorted(non_act),
             "not_executed_formats": sorted(not_executed),
             "coverage_exempt_cells": {name: dict(sorted(by_format.items()))
                                       for name, by_format in sorted(exempt_cells.items())}})


def merge_act_dloss(costs: dict, table: dict) -> dict:
    """Write ``act_dloss`` into the cost rows. Returns a merge report.

    Mutates ``costs`` in place. Rows with no priced A-side are left untouched
    rather than set to 0.0, so ``cost_entry_act_dloss``'s default and a genuine
    measured zero stay distinguishable in the artifact.

    A JOINT AURA row is skipped, not written. Its one signed residual already
    contains the weight, activation and mixed terms under a single downstream
    Fisher, and ``validate_joint_aura_entry`` REFUSES any row carrying
    ``act_dloss`` -- correctly, because that would apply the activation term a
    second time. The allocator never adds ``cost_entry_act_dloss`` to a joint row
    (``cost_entry_predicted_dloss`` returns before that branch), so this stage
    must not stamp one on: the write would not double-count, it would INVALIDATE
    the row and stop the allocation.

    A unit whose every cell is a joint row therefore ends with no ``act_dloss``
    written and no table entry at all, which is a COVERED unit, not an unpriced
    one: it is counted in ``joint_rows_skipped`` and left out of
    ``units_without_act_price``. An all-joint artifact is a fulfilled artifact.
    """
    merged = 0
    unit_hits = 0
    missing_units = []
    joint_rows_skipped = 0
    for name, entry in costs.items():
        row = table.get(name) or {}
        if row:
            unit_hits += 1
        # Walk the ARTIFACT's rows, not only the ones this stage priced. A unit
        # whose every cell was already joint-priced has no table entry at all,
        # and a loop that started from `table` would call it "without an act
        # price" -- losing both the skip count and the fact that the unit IS
        # covered. Whether a row already carries the activation term is a
        # property of the artifact, so it is read from the artifact.
        joint_here = 0
        if isinstance(entry, dict):
            for fmt, cost_row in entry.items():
                if not isinstance(cost_row, dict):
                    continue
                if cost_entry_is_joint_aura(cost_row):
                    joint_rows_skipped += 1
                    joint_here += 1
                    continue
                if fmt in row:
                    cost_row[ACT_DLOSS_KEY] = float(row[fmt])
                    merged += 1
        if not row and not joint_here:
            missing_units.append(name)
    return {"units_in_cost": len(costs), "units_merged": unit_hits,
            "entries_merged": merged,
            "joint_rows_skipped": joint_rows_skipped,
            "units_without_act_price": len(missing_units),
            "examples_without_act_price": missing_units[:5]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", required=True, help="sensitivity card .npz")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cost-in", required=True, help="weight-only cost pkl")
    ap.add_argument("--cost-out", required=True,
                    help="written; --cost-in is left untouched so the "
                         "weight-only allocation stays reproducible as an arm")
    ap.add_argument("--formats", default=None,
                    help="default: every format present in the cost artifact")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--serving-lane", default=None,
        help="lane id whose served_activation_quantization declares which "
             "formats' activation grid the runtime actually EXECUTES. The "
             "A-side is a property of the runtime, not of the format: a lane "
             "that decodes CB weights to BF16 and runs a BF16 GEMM pays no "
             "activation cost at all, and pricing one there makes the DP buy "
             "weight bits to escape zero. Required unless "
             "--lane-executes-all-activation-grids is given.")
    ap.add_argument(
        "--lane-executes-all-activation-grids", action="store_true",
        help="assert that the serving lane executes EVERY format's activation "
             "grid fused (the correct answer for a plain W4A4 lane). Mutually "
             "exclusive with --serving-lane.")
    ap.add_argument("--act-dir", default=None,
                    help="directory of cached real activations (the probe's "
                         "act/ dir). When given, act_var is MEASURED on each "
                         "Linear's real input rows instead of modelled from a "
                         "per-channel Gaussian fit; units with no cached rows "
                         "fall back to the model and are counted separately.")
    ap.add_argument(
        "--require-complete-coverage", action="store_true",
        help="refuse unless EVERY requested (unit, format) cell this lane's "
             "activation contract OWES a price has one: a joint AURA row, or an "
             "A-side computed and merged here. A cell whose format leaves "
             "activations alone AT THAT UNIT'S SHAPE (BF16 and the other "
             "passthroughs), or whose grid this lane never executes, is free by "
             "contract and counts as covered -- per cell, not per format name, "
             "because the same name can be a passthrough at one shape and "
             "unbuildable at another. Off by default: partial coverage is a "
             "hole set a "
             "research arm may deliberately carry, and the campaign "
             "requirement is the campaign's to declare, not this stage's "
             "default.")
    args = ap.parse_args()

    from .sensitivity_card import SensitivityCard

    card = SensitivityCard.from_npz(args.card)
    card.validate()
    fingerprint = card.provenance.fingerprint()
    log(f"card: {len(card)} units, fingerprint {fingerprint}")

    with open(args.cost_in, "rb") as fh:
        blob = pickle.load(fh)
    costs = blob["costs"]
    formats = ([f.strip() for f in args.formats.split(",") if f.strip()]
               if args.formats
               else sorted({f for r in costs.values() for f in r}))
    log(f"cost artifact: {len(costs)} units, formats {formats}")

    # Validate the joint rows ONCE, here, and hand the stage the exact CELLS
    # they already price. `joint_row_binds_cell` is `cost_entry_is_joint_aura`
    # plus the one comparison the internal check cannot make: the row's own
    # operator coordinate against the key it was found under. Both halves are
    # needed here for the same reason. The internal check RAISES on a row that
    # claims a joint currency without carrying one, so malformed joint evidence
    # stops the run before an A-side is computed for a cell that should not get
    # one; the coordinate check is what keeps a *valid* row produced for another
    # Linear from being read as this cell's A-side, which would leave the cell
    # weight-only while the coverage set below said it was covered. The
    # selection is per (unit, format), so a mixed artifact prices exactly its
    # legacy cells.
    already: set[tuple[str, str]] = set()
    for name, entry in costs.items():
        if not isinstance(entry, dict):
            continue
        for fmt, row in entry.items():
            if not isinstance(row, dict):
                continue
            try:
                binds = joint_row_binds_cell(
                    row, name, fmt, where=f"{args.cost_in}: {name}@{fmt}")
            except JointCellCoordinateError as error:
                raise SystemExit(f"REFUSE: {error}") from error
            if binds:
                already.add((name, fmt))
    if already:
        log(f"joint AURA coverage: {len(already)} (unit, format) cells already "
            f"carry their own activation term; no A-side is computed for them")

    # The cells this run was actually ASKED to price: the requested formats that
    # the artifact carries for that unit. A format the artifact does not carry is
    # not a cell -- nothing could be merged there -- and, critically, a joint row
    # for some OTHER cell does not fulfil it. `requested` is the denominator the
    # acceptance below is stated against, so an unrelated joint rung can no
    # longer licence a run that priced nothing.
    formats_by_name: dict[str, list[str]] = {}
    requested: set[tuple[str, str]] = set()
    for name, entry in costs.items():
        if not isinstance(entry, dict):
            continue
        present = [f for f in formats if isinstance(entry.get(f), dict)]
        if present:
            formats_by_name[name] = present
            requested.update((name, f) for f in present)
    log(f"requested cells: {len(requested)} (unit, format) pairs over "
        f"{len(formats_by_name)} units")

    # The architecture's declared name mapping. Optional by design: a model
    # whose checkpoint names match its module tree needs none, and a path that
    # no profile claims must not become a hard failure for those models. When
    # one IS detected it supplies the checkpoint aliases the generic index
    # cannot derive (see build_weight_resolver).
    #
    # "No profile claims this path" is the tolerated case. A dead vendored
    # override is not it (#202): the architecture IS claimed, and dropping to
    # generic name matching would price AQUA activation cost through an index
    # built without the aliases that profile exists to supply. The import is
    # hoisted out of the `try` so the handler below can always be evaluated.
    profile = None
    from .model_profiles.registry import (
        DeadVendoredOverrideError,
        detect_profile,
    )
    try:
        profile = detect_profile(args.model_path)
    except DeadVendoredOverrideError:
        raise
    except Exception as exc:
        log(f"no model profile for {args.model_path} ({exc}); "
            f"falling back to generic name matching")
    if profile is not None:
        log(f"model profile: {type(profile).__name__}")

    executed = resolve_executed_activation_formats(
        lane_id=args.serving_lane,
        executes_all=args.lane_executes_all_activation_grids,
    )
    table, holes, meta = activation_dloss_table(
        card, args.model_path, formats, device=args.device,
        names=[n for n in formats_by_name], act_dir=args.act_dir,
        profile=profile,
        executed_activation_formats=executed,
        already_priced_cells=already,
        formats_by_name=formats_by_name)
    report = merge_act_dloss(costs, table)
    log(f"merge: {report}")
    # Acceptance is stated against the REQUESTED CELLS, not against whatever
    # else the artifact happens to contain. `priced` is what this run actually
    # added; `joint_requested` is the requested subset a joint row already
    # covers; anything left has no A-side in the output.
    priced_cells = {(name, fmt) for name, row in table.items()
                    for fmt in row if fmt in costs.get(name, {})}
    joint_requested = requested & already
    unfulfilled = requested - priced_cells - already
    if len(priced_cells) != report["entries_merged"]:
        raise SystemExit(
            f"REFUSE: internal accounting disagreement -- {len(priced_cells)} "
            f"cells were priced but {report['entries_merged']} were merged; "
            f"the merge is not writing what this stage computed.")
    if unfulfilled:
        # Reported, NOT refused: partial coverage is a hole set, and refusing it
        # would invent a coverage threshold (see #655 for the campaign-level
        # requirement this deliberately does not decide here).
        log(f"coverage: {len(unfulfilled)} of {len(requested)} requested cells "
            f"have no A-side in this output; they keep a weight-only cost")
    # A requested cell is free BY CONTRACT when the format quantizes no
    # activations at THIS unit's shape, or its grid is one THIS lane never
    # executes. Both answers are recorded per (unit, format) by the stage that
    # built the plugin, and the subtraction is per cell for that reason: a
    # format can be a passthrough at one unit's shape and UNBUILDABLE at
    # another's. A name-level union would exempt the unbuildable cell from the
    # gate while the stage itself reported it as a hole -- the gate would pass
    # on exactly the artifact it exists to refuse.
    exempt_cells = {(name, fmt)
                    for name, by_format in (meta.get("coverage_exempt_cells") or {}).items()
                    for fmt in by_format}
    uncovered = {cell for cell in unfulfilled if cell not in exempt_cells}
    if args.require_complete_coverage and uncovered:
        # The campaign requirement (#655): every cell the campaign asked for
        # carries its own activation term. A positive WEIGHT-ONLY surrogate is
        # the case this must not accept -- it is tradeable and biased, but it is
        # not an A-side price, and an unmeasured A-side on an
        # activation-quantizing cell is what silently buys 4-bit.
        examples = ", ".join(f"{name}@{fmt}" for name, fmt in sorted(uncovered)[:5])
        # Bounded, like `examples`: a full campaign has tens of thousands of
        # exempt cells, and naming every one of them in a refusal floods the
        # log the operator has to read. The count is the contract; the sample
        # is what makes it checkable.
        exempt_sample = ", ".join(sorted(
            f"{name}@{fmt}" for name, fmt in exempt_cells)[:5])
        raise SystemExit(
            f"REFUSE: --require-complete-coverage: {len(uncovered)} of "
            f"{len(requested)} requested (unit, format) cells that this lane "
            f"OWES an activation-side price have none in --cost-out, so their "
            f"cost would be weight-only (examples: {examples}). "
            f"{len(exempt_cells)} cell(s) are free by contract AT THEIR OWN "
            f"SHAPE and therefore excluded"
            + (f" (e.g. {exempt_sample})" if exempt_sample else "") +
            ". Price them, joint-price them, or drop the requirement for this "
            "arm.")
    # The silent no-op this refusal exists for: nothing was priced AND at least
    # one requested cell is unaccounted for. An all-joint artifact is the other
    # case -- every requested cell covered, nothing computed, nothing to add.
    if not priced_cells and (unfulfilled or not requested):
        raise SystemExit(
            f"REFUSE: the merge wrote 0 entries and "
            f"{len(unfulfilled)} of {len(requested)} requested (unit, format) "
            f"cells have no A-side, so --cost-out would be a byte-equivalent "
            f"copy of --cost-in carrying the AQUA name. Most likely the card "
            f"has no `g_sq_sum` (a scalar-only card built from a probe "
            f"predating marginal emission); `activation_dloss` returns None "
            f"for every unit in that case. A joint AURA row for a DIFFERENT "
            f"cell does not fulfil a requested legacy cell."
        )
    if not priced_cells:
        log(f"nothing to add: all {len(joint_requested)} requested cells "
            f"already carry a priced activation term, so no A-side was "
            f"computed; provenance records the joint coverage")

    prov = dict(blob.get("provenance") or {})
    prov["aqua_activation_cost"] = {
        "card_fingerprint": fingerprint,
        "card_path": os.path.abspath(args.card),
        "formats_priced": formats,
        "holes": {k: len(v) for k, v in holes.items()},
        "merge_report": report,
        "act_dir": os.path.abspath(args.act_dir) if args.act_dir else None,
        "requested_cells": len(requested),
        "priced_cells": len(priced_cells),
        "cells_without_act_price": len(unfulfilled),
        # The same count with the cells that are free by contract subtracted --
        # what the campaign-coverage gate reads, and the number an operator
        # compares against the campaign's requirement.
        "required_cells_without_act_price": len(uncovered),
        "require_complete_coverage": bool(args.require_complete_coverage),
        "joint_cells_already_priced": [
            list(cell) for cell in sorted(joint_requested)],
        **meta,
    }
    blob["provenance"] = prov
    with open(args.cost_out, "wb") as fh:
        pickle.dump(blob, fh)
    log(f"wrote {args.cost_out}")

    # A one-line readout of what the DP will now see differently. Not a result --
    # the served KL A/B is -- but enough to catch a merge that did nothing.
    ratios = []
    for name, entry in costs.items():
        n = entry.get("NVFP4")
        if isinstance(n, dict) and ACT_DLOSS_KEY in n and n.get(
                "predicted_dloss", 0.0) > 0:
            ratios.append(n[ACT_DLOSS_KEY] / n["predicted_dloss"])
    if ratios:
        r = np.array(ratios)
        log(f"NVFP4 A-side / W-side over {len(r)} units: "
            f"p10={np.percentile(r, 10):.2f} med={np.median(r):.2f} "
            f"p90={np.percentile(r, 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
