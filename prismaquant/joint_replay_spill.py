"""Stage B one-pass replay: spill each probe's X/G once, replay windows from it.

The windowed replay (``joint_statistics_replay.observe_and_project_retained_
windows``) runs the whole layer forward and backward once per (retained
window, probe), because each window's statistics lease hooks only that
window's Linears. On GLM-5.3-Flash that is about 15 layer passes per probe.

This module runs ONE forward and backward per probe with a spill observer on
every pending target Linear. At each backward invocation it records the
Linear's bf16 input ``x`` and its selected bf16 output gradient to a
local-NVMe scratch file. Each retained window then feeds its own fresh
``JointOperatorStatisticsLease`` from the spill through the lease's one
statistics arithmetic (``_observe_invocation``). The records are
bit-identical to the windowed path's because:

* every statistics key is an independent FP32 accumulator of one Linear, so
  only each Linear's own ``add_`` order matters, and the replay feeds each
  Linear's invocations in firing order;
* the spilled operands are the exact bf16 tensors the live hook reads at fire
  time, rebuilt with the same shape and strides and, on CUDA, the same
  address residue modulo 512 bytes, so ``.float()``, the activation QDQ and
  the GEMMs see identical operands (see ``StageBReplaySpill._same_layout``
  for why a host residue is not held);
* a non-dense selected gradient is stored contiguous, which is what the live
  path's ``reshape(...).float()`` makes of it anyway. A non-dense ``x`` is
  refused, because it reaches the activation QDQ as is.

The spill is laid out per Linear (``_Window``): one input stream per Linear
that first read a tensor, and one gradient stream per Linear and probe, so a
per-operator GEMM over a Linear's rows reads two ordered streams.

Only probe 0's inputs are written: every probe's forward is the same, so
every probe's ``x`` must equal probe 0's, and the replay reads probe 0's.
Each input is digested on its own device when the hook fires
(``_InputDigest``), and a later probe's input that differs from probe 0's
fails the capture. A later probe's input is never copied to the host.

At the default replay regime the arithmetic identity and the resource
policy are unchanged; the replay mode is recorded only in the quantum
counters. Without the spill environment the windowed path runs unchanged and
is the bitwise reference. A non-default regime (``joint_replay_regime``:
several stored batches per capture pass, or one GEMM per operator over row
chunks) changes the arithmetic and is stamped in the statistics identity.

The scratch is declared like the #956 cotangent sink: an environment root and
a byte ceiling, forwarded by the campaign container launcher through an
identity bind. The layer's spill bytes are bounded from geometry before any
GPU work and the whole bound is allocated up front; see
``perturbed_x_cache.StageBSpillScratch``. The ceiling is not a spec literal:
the record builder seals the layer's full-roster geometry and its reservation
(:func:`seal_spill_bound`), the dispatcher sets the row's ceiling, and so
PrismaBuild's ``spool_gb`` charge, from that reservation, and the quantum
recomputes the geometry and refuses a record whose seal differs
(:func:`require_sealed_spill_bound`). All spill I/O is direct
(``O_DIRECT``, PQ #1060): the arena, the file and the read buffer share one
slot layout on the file's direct-I/O grid (``_slot``), so each tensor
writes from the pinned arena and reads back into the pinned read buffer
with no page cache in between, in calls that bound what is in flight.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import bisect
import os
import queue
import threading
import time

import torch

from .dev_mode import dev_mode_enabled, seal_check
from .joint_aura import (
    JointOperatorStatisticsLease,
    SignedJointProjectionLease,
    select_invocation_gradient,
)
from .joint_replay_regime import OPERATOR_GEMM, PER_INVOCATION, normalize_replay_regime
from .routed_experts import PackedExpertProjection, ProfileRoutedExpertClassifier

SPILL_ENV = ("PRISMAQUANT_STAGE_B_SPILL_ROOT", "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")
REPLAY_WINDOWED = "windowed"
REPLAY_SPILL = "one_pass_spill"
SPILL_DTYPES = (torch.bfloat16, torch.float16)
#: Replayed operands keep the live tensor's address residue modulo this many
#: bytes: the CUDA caching allocator's block alignment, so no kernel variant
#: chosen on pointer alignment can differ between the two paths.
ADDRESS_ALIGNMENT = 512
#: Write arenas and read buffers are at least this large (or one largest
#: tensor pair), and never larger than the layer needs.
ARENA_BYTES = 256 << 20
READ_BYTES = 64 << 20
ARENA_COUNT = 3
#: Read buffers a replay phase reserves: the chunk it consumes and the one it
#: asks for next. The IO engine reads further ahead within its budget's live
#: headroom (PQ #1348), not within this count.
READ_BUFFER_COUNT = 2
#: Direct I/O (PQ #1060): the most bytes one write call and one read call
#: put in flight. The NVMe takes at most 128 KiB per request, so throughput
#: is queue depth times 128 KiB over the latency, and Netdata's disk backlog
#: integrates that queue. Measured on lina from pinned memory (PB
#: f6733604db33, b55e4305076c): one 1 MiB write in flight ran 4.8 GB/s at
#: an average queue of 4 and 0.11 ms await, against 5.9 GB/s at a queue of
#: 811 and 18 ms for a whole 256 MiB arena at once. Replay reads go through
#: the IO engine (PQ #1348): each read chunk is one entry of one stream, and
#: the engine's pool sets how many chunks are read at once from the measured
#: rates. See docs/ARCHITECTURE.md.
WRITE_CALL_BYTES = 1 << 20
READ_CALL_BYTES = 1 << 20
#: A writer thread by default; tests run the same writes inline.
DEFAULT_THREADS = True
#: The sealed spill bound's schema (``executable_readset.spill_bound``).
SPILL_BOUND_SCHEMA = "prismaquant.stage_b_spill_bound.v1"
#: The direct-I/O grid a sealed spill bound is sized on unless the builder is
#: told otherwise: 4 KiB, the page size and the ext4 block of the spill roots
#: the campaign declares (the grid lina's root reported in PB 6f4f058751e6).
#: It is the one input to the reservation that the plan does not carry. The
#: scratch refuses a coarser live grid before it allocates the file
#: (``StageBSpillScratch(max_block=...)``).
SPILL_SEAL_BLOCK_BYTES = 4096
#: The quantum measures in the streamed source's dtype, which is bfloat16
#: (``joint_cost_quantum.build_quantum_source_runner``).
SPILL_SEAL_DTYPE = "bfloat16"
_TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_top_k",
               "num_active_experts")


def stage_b_spill_config(environ=None):
    """``(root, max_bytes)`` when the spill is declared, else ``None``."""
    environ = os.environ if environ is None else environ
    root, ceiling = (environ.get(name) for name in SPILL_ENV)
    if root is None and ceiling is None:
        return None
    if not root or not ceiling or not ceiling.isdecimal() or int(ceiling) <= 0:
        raise ValueError(
            "Stage B spill requires both PRISMAQUANT_STAGE_B_SPILL_ROOT and a "
            "positive PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")
    return root, int(ceiling)


def experts_per_token(model):
    """The declared routed top-k, or ``None`` when the config states none."""
    config = getattr(model, "config", None)
    # A config.json read as a mapping (the record builder, before any model
    # exists) nests its text config the same way the live config does.
    text_config = (config.get("text_config") if isinstance(config, Mapping)
                   else getattr(config, "text_config", None))
    for candidate in (config, text_config):
        if candidate is None:
            continue
        for key in _TOP_K_KEYS:
            value = (candidate.get(key) if isinstance(candidate, Mapping)
                     else getattr(candidate, key, None))
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


class ReplayRegimeInadmissible(RuntimeError):
    """A non-default replay regime this quantum cannot run as declared."""


def require_row_local_activation_qdq(modules, specs_by_qname, activation_max_abs, *,
                                     device, dtype, rows=8):
    """Refuse a replay regime unless every activation QDQ is row-local.

    A batched capture (``capture_batch`` > 1) hands the QDQ B samples' rows
    at once, and ``operator_gemm`` hands it row chunks that cut across
    invocations. Both measure the same thing only if the QDQ of a block of
    rows is the rows' own QDQs stacked: no scale may be shared across rows.
    This checks exactly that, bit for bit, on this device and dtype, through
    the function the statistics lease calls (``perturbed_x_cache.
    _activation_qdq``), for one Linear of each (format, input width) in the
    roster. Row ``r`` is scaled by ``2**(r - rows // 2)`` so that a tensor-wide
    scale would show. Returns the checked ``(format, input width)`` pairs.
    """
    from .perturbed_x_cache import _activation_qdq

    generator = torch.Generator(device="cpu").manual_seed(994)
    checked = {}
    for name in sorted(modules):
        width = int(modules[name].weight.shape[1])
        for fmt, spec in sorted(specs_by_qname[name].items()):
            if not spec.act_quant_changes_input or (fmt, width) in checked:
                continue
            scale = torch.pow(2.0, torch.arange(rows, dtype=torch.float32) - rows // 2)
            block = (torch.randn(rows, width, generator=generator) * scale[:, None]).to(
                device=device, dtype=dtype)
            with torch.no_grad():
                whole = _activation_qdq(block.reshape(2, rows // 2, width), spec,
                                        activation_max_abs, name).reshape(rows, width)
                alone = torch.cat([_activation_qdq(block[row:row + 1], spec,
                                                   activation_max_abs, name)
                                   for row in range(rows)])
            if not torch.equal(whole, alone):
                raise ReplayRegimeInadmissible(
                    f"{fmt}'s activation QDQ is not row-local at input width {width} "
                    f"({name}): a batched or row-chunked replay would change what it "
                    "measures")
            checked[(fmt, width)] = name
    return sorted(checked)


@dataclass(frozen=True)
class SpillTarget:
    """What the spill geometry reads of one target: its shape and its role.

    ``packed`` is ``(module_qname, param_name, projection_name, expert_id)``
    for a routed expert's view of a packed parameter, and ``None`` for a
    target that spills every token. The quantum takes it from the live
    module (:func:`spill_target`); the record builder takes it from the
    verified render shape and the profile (:func:`sealed_spill_targets`).
    """

    out_features: int
    in_features: int
    packed: tuple | None = None


def spill_target(module):
    """The :class:`SpillTarget` of a live target (or of a sealed one)."""
    if isinstance(module, SpillTarget):
        return module
    out_features, in_features = (int(size) for size in module.weight.shape)
    if isinstance(module, PackedExpertProjection):
        return SpillTarget(out_features, in_features,
                           (module.module_qname, module.param_name,
                            module.projection_name, int(module.expert_id)))
    return SpillTarget(out_features, in_features)


def sealed_spill_targets(shapes, profile):
    """Spill targets from verified render shapes, before any model exists.

    A name the profile classifies as a routed expert is a view of its
    packed parameter, keyed as ``profile_declared_packed_expert_projections``
    keys the live view: the qname is ``{module}.{expert}.{projection}`` and
    the parameter is the projection's declared packed parent. Every other
    name spills every token. The quantum recomputes the geometry from its
    live modules and refuses a record whose seal differs
    (:func:`require_sealed_spill_bound`), so a model that loads its experts
    unpacked refuses rather than overrunning the sealed ceiling.
    """
    classifier = ProfileRoutedExpertClassifier(profile)
    targets = {}
    for name, shape in sorted(shapes.items()):
        if (not isinstance(name, str) or not name or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape)):
            raise ValueError("a sealed spill target needs a named positive 2-D shape")
        out_features, in_features = shape
        match = classifier.classify(name)
        if match is None:
            targets[name] = SpillTarget(out_features, in_features)
            continue
        parts = name.rsplit(".", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not match.regex_declared:
            raise ValueError(f"routed expert {name} has no declared per-expert view")
        parent = profile.packed_expert_parent_for_projection(match.projection_name)
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"routed expert {name} names no packed parent parameter")
        targets[name] = SpillTarget(out_features, in_features,
                                    (parts[0], parent, parts[2], int(parts[1])))
    return targets


@dataclass(frozen=True)
class SpillGeometry:
    """Upper bound on one layer's spill, from shapes and token counts only."""

    element_size: int
    experts_per_token: int | None
    n_probes: int
    tokens: int
    max_batch_tokens: int
    window_x_bytes: tuple
    window_g_bytes: tuple
    x_bytes: int
    g_bytes_per_probe: int
    total_bytes: int
    batch_x_bytes: int
    batch_bytes: int
    largest_tensor_bytes: int
    max_parts: int

    def as_dict(self):
        return {
            "element_size": self.element_size,
            "experts_per_token": self.experts_per_token,
            "n_probes": self.n_probes, "tokens": self.tokens,
            "max_batch_tokens": self.max_batch_tokens,
            "x_bytes": self.x_bytes, "g_bytes_per_probe": self.g_bytes_per_probe,
            "total_bytes": self.total_bytes, "batch_x_bytes": self.batch_x_bytes,
            "batch_bytes": self.batch_bytes,
            "largest_tensor_bytes": self.largest_tensor_bytes,
            "max_parts": self.max_parts,
            "windows": len(self.window_x_bytes),
        }


def spill_geometry(linears, window_names, *, pending, batch_tokens, n_probes,
                   element_size, experts_per_token):
    """Bound the spill of ``pending`` targets over ``window_names``.

    One invocation per target per sample is assumed; the scratch refuses at
    the first byte past this bound, so a model that breaks the assumption
    fails closed instead of overrunning the disk.

    * A dense target spills ``x`` and its gradient for every token.
    * A token visits each routed expert at most once and at most ``top_k``
      experts. A packed projection sits in one window, so its gradient rows
      over its ``m`` pending experts, across all windows, are at most
      ``tokens * min(m, top_k)``. Without a declared top-k the bound is
      ``tokens * m``, looser but still exact.
    * An expert's ``x`` is written once per window that holds any projection
      of its packed parameter (gate and up read the same rows), ``c`` times
      in all. A token then writes at most the ``top_k`` largest ``c`` of the
      experts it visits, so the parameter's input rows are at most ``tokens``
      times that sum. Bounding each window by ``min(m, top_k)`` instead would
      charge ``top_k`` rows per token to every window a parameter spans.

    ``window_x_bytes`` and ``window_g_bytes`` bound each window alone; the
    totals bound the layer and are what the scratch reserves.

    ``max_parts`` bounds the tensors written, which the scratch pads to its
    direct-I/O grid: per sample, one gradient per pending target and probe,
    and for probe 0 at most one input per pending target (an input is
    written once per window by the Linear that first reads it).
    """
    if type(n_probes) is not int or n_probes <= 0:
        raise ValueError("Stage B spill geometry requires a positive probe count")
    batch_tokens = [int(tokens) for tokens in batch_tokens]
    if not batch_tokens or any(tokens <= 0 for tokens in batch_tokens):
        raise ValueError("Stage B spill geometry requires positive batch token counts")

    def rows(experts):
        return experts if experts_per_token is None else min(experts, experts_per_token)

    def largest(counts):
        ordered = sorted(counts, reverse=True)
        return sum(ordered if experts_per_token is None else ordered[:experts_per_token])

    per_token_x, per_token_g, widest, targets = [], [], 0, 0
    dense_x = dense_g = 0
    layer_in, layer_out = {}, {}
    for names in window_names:
        x_width = g_width = 0
        packed_in, packed_out = {}, {}
        for name in names:
            if name not in pending:
                continue
            targets += 1
            target = spill_target(linears[name])
            out_features, in_features = target.out_features, target.in_features
            widest = max(widest, out_features, in_features)
            if target.packed is not None:
                module_qname, param_name, projection_name, expert_id = target.packed
                key = (module_qname, param_name)
                width, experts = packed_in.setdefault(key, [in_features, set()])
                if width != in_features:
                    raise RuntimeError(f"packed parameter input width differs for {name}")
                experts.add(expert_id)
                role = (key, projection_name)
                width, experts = packed_out.setdefault(role, [out_features, set()])
                if width != out_features:
                    raise RuntimeError(f"packed projection output width differs for {name}")
                experts.add(expert_id)
            else:
                x_width += in_features
                g_width += out_features
        dense_x += x_width
        dense_g += g_width
        for key, (width, experts) in packed_in.items():
            layer_width, windows = layer_in.setdefault(key, [width, {}])
            if layer_width != width:
                raise RuntimeError(f"packed parameter input width differs for {key}")
            for expert in experts:
                windows[expert] = windows.get(expert, 0) + 1
        for role, (width, experts) in packed_out.items():
            layer_width, seen = layer_out.setdefault(role, [width, set()])
            if layer_width != width or seen & experts:
                raise RuntimeError(f"packed projection {role} is split inconsistently "
                                   "across windows")
            seen |= experts
        x_width += sum(width * rows(len(experts)) for width, experts in packed_in.values())
        g_width += sum(width * rows(len(experts)) for width, experts in packed_out.values())
        per_token_x.append(x_width)
        per_token_g.append(g_width)
    layer_x = dense_x + sum(width * largest(windows.values())
                            for width, windows in layer_in.values())
    layer_g = dense_g + sum(width * rows(len(experts))
                            for width, experts in layer_out.values())
    tokens, widest_batch = sum(batch_tokens), max(batch_tokens)
    window_x = tuple(width * tokens * element_size for width in per_token_x)
    window_g = tuple(width * tokens * element_size for width in per_token_g)
    x_bytes = layer_x * tokens * element_size
    g_bytes = layer_g * tokens * element_size
    return SpillGeometry(
        element_size=int(element_size), experts_per_token=experts_per_token,
        n_probes=n_probes, tokens=tokens, max_batch_tokens=widest_batch,
        window_x_bytes=window_x, window_g_bytes=window_g,
        x_bytes=x_bytes, g_bytes_per_probe=g_bytes,
        total_bytes=x_bytes + n_probes * g_bytes,
        batch_x_bytes=layer_x * widest_batch * element_size,
        batch_bytes=(layer_x + layer_g) * widest_batch * element_size,
        largest_tensor_bytes=widest * widest_batch * element_size,
        max_parts=(n_probes + 1) * targets * len(batch_tokens))


def spill_capture_batch_tokens(n_samples, seqlen, *, probe_microbatch, capture_batch):
    """Tokens per capture group, grouped exactly as the quantum groups them.

    The quantum stores ``probe_microbatch`` rows per batch (all rows when it
    is 0) and captures ``capture_batch`` consecutive batches per pass, the
    last group ragged. Every row holds ``seqlen`` tokens: the calibration
    draw is refused unless it is exactly ``n_samples`` by ``seqlen``.
    """
    for label, value in (("sample count", n_samples), ("sequence length", seqlen),
                         ("capture batch", capture_batch)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Stage B spill {label} must be a positive integer")
    if type(probe_microbatch) is not int or probe_microbatch < 0:
        raise ValueError("Stage B spill probe microbatch must be a nonnegative integer")
    batch_rows = min(probe_microbatch or n_samples, n_samples)
    batches = [min(batch_rows, n_samples - start) * seqlen
               for start in range(0, n_samples, batch_rows)]
    return [sum(batches[start:start + capture_batch])
            for start in range(0, len(batches), capture_batch)]


def spill_reservation_bytes(total_bytes, max_parts, *, block):
    """The spill file's reservation for a geometry on a ``block`` grid.

    The one sizing rule (``StageBSpillScratch.reservation_bytes``) with the
    spill's own slot padding: each part sits fewer than ``ADDRESS_ALIGNMENT``
    bytes into a slot that ends on the next grid boundary. The scratch
    reserves this; the record builder seals it as the row's ceiling, which
    is also PrismaBuild's ``spool_gb`` charge.
    """
    from .perturbed_x_cache import StageBSpillScratch

    if type(block) is not int or block < ADDRESS_ALIGNMENT or block & (block - 1):
        raise ValueError(f"Stage B spill grid must be a power of two of at least "
                         f"{ADDRESS_ALIGNMENT} bytes, not {block!r}")
    return StageBSpillScratch.reservation_bytes(
        total_bytes, parts=max_parts, part_padding=ADDRESS_ALIGNMENT, block=block)


class SpillBoundRefused(RuntimeError):
    """A sealed spill bound that is malformed or does not match its quantum."""


_SPILL_BOUND_KEYS = frozenset(
    {"schema", "block", "capture_batch", "element_dtype", "geometry",
     "reservation_bytes"})


def seal_spill_bound(geometry, *, block, capture_batch, element_dtype):
    """The ``executable_readset.spill_bound`` block for one layer quantum.

    ``geometry`` is the layer's full-roster :class:`SpillGeometry`: a fresh
    run spills every target, and a resume spills a subset, which reserves
    no more. ``reservation_bytes`` is :func:`spill_reservation_bytes` on the
    ``block`` grid; the dispatcher sets the row's ceiling to it.
    """
    if not isinstance(geometry, SpillGeometry):
        raise ValueError("a spill bound seals a SpillGeometry")
    bound = {"schema": SPILL_BOUND_SCHEMA, "block": block,
             "capture_batch": capture_batch, "element_dtype": element_dtype,
             "geometry": geometry.as_dict(),
             "reservation_bytes": spill_reservation_bytes(
                 geometry.total_bytes, geometry.max_parts, block=block)}
    check_spill_bound(bound)
    return bound


def check_spill_bound(bound):
    """Validate a sealed spill bound and return its reservation in bytes.

    The reservation must be what :func:`spill_reservation_bytes` computes
    from the sealed geometry and grid: a bound whose ceiling was edited, or
    computed by another rule, refuses.
    """
    if not isinstance(bound, Mapping) or set(bound) != _SPILL_BOUND_KEYS:
        raise SpillBoundRefused(
            f"a spill bound carries exactly {sorted(_SPILL_BOUND_KEYS)}")
    if bound["schema"] != SPILL_BOUND_SCHEMA:
        raise SpillBoundRefused(f"a spill bound has a foreign schema {bound['schema']!r}")
    if type(bound["capture_batch"]) is not int or bound["capture_batch"] <= 0:
        raise SpillBoundRefused("a spill bound seals no positive capture batch")
    if bound["element_dtype"] not in {str(dtype).removeprefix("torch.")
                                      for dtype in SPILL_DTYPES}:
        raise SpillBoundRefused(
            f"a spill bound seals no 16-bit dtype, but {bound['element_dtype']!r}")
    geometry = bound["geometry"]
    if not isinstance(geometry, Mapping):
        raise SpillBoundRefused("a spill bound seals no geometry")
    for key in ("total_bytes", "max_parts"):
        if type(geometry.get(key)) is not int or geometry[key] < 0:
            raise SpillBoundRefused(f"a spill bound's geometry seals no {key}")
    try:
        reservation = spill_reservation_bytes(
            geometry["total_bytes"], geometry["max_parts"], block=bound["block"])
    except ValueError as exc:
        raise SpillBoundRefused(str(exc)) from exc
    if bound["reservation_bytes"] != reservation:
        raise SpillBoundRefused(
            f"a spill bound seals a {bound['reservation_bytes']!r}-byte reservation, "
            f"but its geometry needs {reservation} bytes on its "
            f"{bound['block']}-byte grid")
    return reservation


def require_sealed_spill_bound(bound, geometry, *, capture_batch, element_dtype,
                               ceiling):
    """Refuse a launch whose spill differs from what its record sealed.

    ``geometry`` is the quantum's own full-roster geometry from its live
    modules and calibration draw. It, the capture batch and the dtype must
    equal the sealed ones, and the declared ceiling must be the sealed
    reservation (the dispatcher sets it so). Returns the sealed grid, the
    coarsest the scratch may take (``StageBSpillScratch(max_block=...)``).
    """
    reservation = check_spill_bound(bound)
    if not isinstance(geometry, SpillGeometry):
        raise SpillBoundRefused("a spill bound is checked against a SpillGeometry")
    live = {"capture_batch": capture_batch, "element_dtype": element_dtype,
            "geometry": geometry.as_dict()}
    # The sealed spill identity is a run seal (PQ #1147): dev mode prints a
    # difference and runs on the live geometry.
    for key, value in live.items():
        seal_check(f"spill {key}", bound[key], value, where="Stage B spill bound",
                   refusal=SpillBoundRefused(
                       f"the record seals spill {key} {bound[key]!r}, but this quantum "
                       f"measures {value!r}; regenerate its executable readset"))
    if dev_mode_enabled():
        # The capacity half is a resource bound and refuses in both modes:
        # the scratch the live geometry needs must fit the admitted ceiling.
        # Certified mode checks it through the equalities around it.
        try:
            needed = spill_reservation_bytes(
                live["geometry"]["total_bytes"], live["geometry"]["max_parts"],
                block=bound["block"])
        except ValueError as exc:
            raise SpillBoundRefused(str(exc)) from exc
        if needed > ceiling:
            raise SpillBoundRefused(
                f"this quantum's spill needs {needed} bytes on its {bound['block']}-byte "
                f"grid, over the admitted ceiling of {ceiling} bytes")
    seal_check("spill ceiling", reservation, ceiling, where="Stage B spill bound",
               refusal=SpillBoundRefused(
                   f"the spill ceiling is {ceiling} bytes, but the record seals "
                   f"{reservation}; dispatch the row with tools/dispatch_joint_quanta.py, "
                   "which sets the ceiling from the record"))
    return bound["block"]


def _is_dense(tensor):
    """Non-overlapping and dense: its elements fill exactly ``numel`` slots."""
    if tensor.is_contiguous():
        return True
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in
                               zip(tensor.shape, tensor.stride()) if size != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _contiguous_strides(shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(int(size), 1)
    return tuple(reversed(strides))


def _layout(tensor, *, dense):
    if dense:
        return (tuple(tensor.shape), tuple(tensor.stride()),
                tensor.data_ptr() % ADDRESS_ALIGNMENT)
    return (tuple(tensor.shape), _contiguous_strides(tensor.shape), 0)


def _storage_order(tensor):
    """The tensor's elements in storage order, as one flat view (or copy)."""
    if _is_dense(tensor):
        return tensor.as_strided((tensor.numel(),), (1,))
    return tensor.contiguous().view(-1)


def _placed(cursor, residue):
    return cursor + (residue - cursor) % ADDRESS_ALIGNMENT


def _ceil(value, block):
    return value + (-value) % block


def _slot(cursor, residue, nbytes, block):
    """``(start, offset, end)`` of a slot on the direct-I/O grid.

    The slot starts on the first ``block`` boundary at or after ``cursor``;
    the tensor sits ``residue`` bytes in (its address residue modulo
    ``ADDRESS_ALIGNMENT``) and the slot ends on the next boundary. An empty
    tensor takes no slot. The arena, the file and the read buffer lay every
    tensor out by this one rule, so a slot's block-aligned envelope reads
    straight from the file into the buffer.
    """
    start = _ceil(cursor, block)
    if not nbytes:
        return start, start, start
    offset = start + residue
    return start, offset, _ceil(offset + nbytes, block)


def _empty_host_cache():
    """Hand the caching host allocator's idle pinned blocks back; the bytes freed."""
    stats = torch.cuda.memory.host_memory_stats
    before = stats().get("allocated_bytes.current", 0)
    torch._C._host_emptyCache()
    return max(0, before - stats().get("allocated_bytes.current", 0))


def _aligned_buffer(nbytes, block, pinned):
    """A uint8 tensor of ``nbytes`` whose address is a multiple of ``block``."""
    raw = torch.empty(nbytes + block, dtype=torch.uint8, pin_memory=pinned)
    return raw.narrow(0, (-raw.data_ptr()) % block, nbytes)


#: The input digest's modulus, the Mersenne prime 2^31 - 1.
_DIGEST_PRIME = (1 << 31) - 1
#: 16-bit words per segment. A segment sums 2^16 products of a word below 2^16
#: and a key below 2^31, so it stays below 2^63 in int64.
_DIGEST_SEGMENT_WORDS = 1 << 16
#: Segments per block: one block's int64 words and products are 24 MiB.
_DIGEST_BLOCK_SEGMENTS = 16
#: Segment keys held; an input of more than 2^32 words is refused.
_DIGEST_MAX_SEGMENTS = 1 << 16
_DIGEST_SEED = 1030


class _InputDigest:
    """Two independent digests of a tensor's bytes, computed on its device (#1030).

    The spill compares each later probe's input with probe 0's. The tensor's
    bytes in storage order are read as 16-bit words ``w``; word ``i`` of
    segment ``s`` is weighted by ``k[i] * r[s]``, both drawn once from a fixed
    seed below ``p = 2^31 - 1``, and the digest is the weighted sum mod
    ``p``, twice with independent keys. Every step is exact int64 arithmetic
    (see the bounds on the constants above), so the digest does not depend on
    reduction order or device. For inputs that differ, each digest is equal
    with probability at most ``2/p`` over the keys, so both are equal with
    probability below ``2^-59``. The spill's inputs are not adversarial; the
    check exists to catch a forward that is not deterministic.
    """

    def __init__(self, device):
        generator = torch.Generator().manual_seed(_DIGEST_SEED)
        self.word_keys = torch.randint(
            1, _DIGEST_PRIME, (2, _DIGEST_SEGMENT_WORDS), generator=generator,
            dtype=torch.int64).to(device)
        self.segment_keys = torch.randint(
            1, _DIGEST_PRIME, (2, _DIGEST_MAX_SEGMENTS), generator=generator,
            dtype=torch.int64).to(device)

    def __call__(self, tensor):
        """The ``(2,)`` int64 digest of ``tensor``'s bytes, on its device."""
        words = _storage_order(tensor.detach()).view(torch.int16)
        count = words.numel()
        full, tail = divmod(count, _DIGEST_SEGMENT_WORDS)
        if full + (1 if tail else 0) > _DIGEST_MAX_SEGMENTS:
            raise RuntimeError("Stage B spill input exceeds the digest's segment keys")
        total = torch.zeros(2, dtype=torch.int64, device=words.device)
        for start in range(0, full, _DIGEST_BLOCK_SEGMENTS):
            stop = min(full, start + _DIGEST_BLOCK_SEGMENTS)
            block = words[start * _DIGEST_SEGMENT_WORDS:stop * _DIGEST_SEGMENT_WORDS]
            block = block.view(stop - start, _DIGEST_SEGMENT_WORDS).to(torch.int64)
            block.bitwise_and_(0xFFFF)
            sums = (block.unsqueeze(0) * self.word_keys.unsqueeze(1)).sum(dim=2)
            sums.remainder_(_DIGEST_PRIME)
            sums.mul_(self.segment_keys[:, start:stop]).remainder_(_DIGEST_PRIME)
            total.add_(sums.sum(dim=1))
        if tail:
            block = words[full * _DIGEST_SEGMENT_WORDS:].to(torch.int64).bitwise_and_(0xFFFF)
            sums = (block.unsqueeze(0) * self.word_keys[:, :tail]).sum(dim=1)
            sums.remainder_(_DIGEST_PRIME)
            total.add_(sums.mul_(self.segment_keys[:, full]).remainder_(_DIGEST_PRIME))
        return total.remainder_(_DIGEST_PRIME)


class _Entry:
    __slots__ = ("logical", "nbytes", "layout", "digest")

    def __init__(self, logical, nbytes, layout):
        self.logical, self.nbytes, self.layout, self.digest = logical, nbytes, layout, None


class _Window:
    """One retained window's spilled streams and its probe-0 record order.

    Streams are per Linear. A Linear's output gradients are one stream per
    probe. An input stream belongs to the Linear that first read the tensor;
    a Linear that reads the same tensor afterwards (the up projection of an
    expert whose gate projection read it first) reads its owner's stream,
    and must do so for every one of its invocations. So each Linear's rows
    are one ordered input stream and one ordered gradient stream, which is
    what a per-operator GEMM over the Linear's rows reads.
    """

    def __init__(self, names):
        self.names = tuple(names)
        # owner -> input entries in first-use order
        self.entries: dict[str, list[_Entry]] = {}
        self.x_logical: dict[str, int] = {}
        self.x_runs: dict[str, list[tuple[int, int, int]]] = {}
        self.x_starts: dict[str, list[int]] = {}
        # Linear -> the input stream all of its records read
        self.x_source: dict[str, str] = {}
        # (name, owner, entry, g_logical, g_bytes, g_layout) in firing order
        self.records: list[tuple] = []
        self.g_logical: dict[str, int] = {}
        self.g_runs: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
        self.g_starts: dict[tuple[str, int], list[int]] = {}
        # (owner, entry) -> the last plan position that reads it
        self.last_ref: dict[tuple[str, int], int] = {}
        # [(owner, chunk)] in replay order
        self.plan: list[tuple] = []
        self.entry_cursor: dict[str, int] = {}
        self.record_cursor = 0
        self.dedupe: dict = {}
        # A later probe's (owner, entry, digest) per input, in first-use order;
        # compared with probe 0's once the capture ends.
        self.x_checks: list[tuple[str, int, torch.Tensor]] = []
        # operator_gemm: Linear -> (rows per probe, input width, output width)
        self.rows: dict[str, list[int]] = {}


class _Arena:
    __slots__ = ("buffer", "view", "used", "parts", "event", "probe")

    def __init__(self, nbytes, pinned, block):
        self.buffer = _aligned_buffer(nbytes, block, pinned)
        self.view = memoryview(self.buffer.numpy())
        self.used, self.parts, self.event, self.probe = 0, [], None, None


class _RowBlock:
    """One Linear's rows awaiting the operator GEMM (``operator_gemm``)."""

    __slots__ = ("x", "g", "filled", "calls")

    def __init__(self, rows, x_width, g_width, dtype, device):
        self.x = torch.empty((rows, x_width), dtype=dtype, device=device)
        self.g = torch.empty((rows, g_width), dtype=dtype, device=device)
        self.filled = self.calls = 0


class _SpillObserver(SignedJointProjectionLease):
    """The statistics lease's hooks and packed wrapper, recording instead.

    It reuses ``SignedJointProjectionLease``'s forward hooks and its packed
    ``F.linear``/``F.grouped_mm`` wrapper unchanged, and the gradient
    selection the statistics lease uses (``select_invocation_gradient``).
    """

    def __init__(self, session, modules, specs_by_qname, *, activation_max_abs,
                 projection_backend):
        self._session = session
        super().__init__(modules, specs_by_qname, {},
                         activation_max_abs=activation_max_abs,
                         projection_backend=projection_backend)
        self._sources = {name: JointOperatorStatisticsLease._source_fingerprint(module.weight)
                         for name, module in self.modules.items()}
        self._pending = 0
        self.observed = 0

    def _validate_delta_coverage(self, name, module):
        if name not in self.specs or not self.specs[name]:
            raise ValueError(f"Stage B spill missing spec coverage for {name}")

    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if not self.active:
            raise RuntimeError("Stage B spill forward outside active capture")
        if JointOperatorStatisticsLease._source_fingerprint(source_weight) != self._sources[name]:
            raise RuntimeError(f"Stage B spill source weight changed for {name}")
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError(f"Stage B spill Linear {name} needs Tensor input/output")
        held = [x.detach(), source_weight]
        fired = False

        def spill(gradient):
            nonlocal fired
            if not self.active or fired:
                raise RuntimeError("Stage B spill backward outside active capture")
            fired = True
            try:
                x_held, weight = held
                with torch.no_grad():
                    selected = select_invocation_gradient(
                        name, weight, x_held, gradient,
                        output_slice=output_slice, row_slice=row_slice)
                    self._session._record(name, x_held, selected)
                self._pending -= 1
                self.observed += 1
            except BaseException:
                self.active = False
                raise
            finally:
                held.clear()
            return gradient

        if output.requires_grad:
            self._pending += 1
            output.register_hook(spill)

    def end_batch(self):
        if self._pending:
            raise RuntimeError("Stage B spill has pending backward observations")
        self._session._end_batch()

    def finish_probe(self):
        raise RuntimeError("Stage B spill observer projects nothing")


class StageBReplaySpill:
    """One layer quantum's spill: capture per probe, then replay per window.

    ``window_names`` are the sealed retained windows in index order, already
    reduced to their pending targets. ``threads`` runs the NVMe writer on
    its own thread; ``False`` does the same writes inline. ``None`` takes
    ``DEFAULT_THREADS``. The replay reads through the IO engine either way
    (PQ #1348): one stream over every chunk of every pending window and
    probe, in replay order, read ahead within :meth:`bind_replay_budget`'s
    budget. ``max_block`` is the grid a sealed ceiling was sized on
    (:func:`require_sealed_spill_bound`).
    """

    def __init__(self, *, root, max_bytes, geometry, window_names, n_probes,
                 dtype, device, threads=None, accumulation=PER_INVOCATION,
                 chunk_rows=None, max_block=None):
        from .perturbed_x_cache import StageBSpillScratch

        regime = normalize_replay_regime({"accumulation": accumulation,
                                          "chunk_rows": chunk_rows})
        self.accumulation, self.chunk_rows = regime["accumulation"], regime["chunk_rows"]
        self._gemm_reserve = 0
        if dtype not in SPILL_DTYPES:
            raise RuntimeError(
                f"Stage B spill replays 16-bit measurement only, not {dtype}: "
                "an FP32 operand is used in place and its layout is not rebuilt")
        if not isinstance(geometry, SpillGeometry) or geometry.n_probes != n_probes:
            raise ValueError("Stage B spill geometry does not match the probe count")
        self.geometry = geometry
        self.n_probes = int(n_probes)
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # Tensors always name their index; compare against the same spelling.
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.element_size = torch.empty((), dtype=dtype).element_size()
        if self.element_size != geometry.element_size:
            raise ValueError("Stage B spill geometry element size differs")
        self._cuda = self.device.type == "cuda"
        self._digest = None
        self._threads = DEFAULT_THREADS if threads is None else bool(threads)
        self._windows = [_Window(names) for names in window_names]
        self._window_of = {}
        for index, window in enumerate(self._windows):
            for name in window.names:
                if name in self._window_of:
                    raise ValueError(f"Stage B spill window membership repeats {name}")
                self._window_of[name] = index
        # The file first: its direct-I/O grid, never finer than
        # ADDRESS_ALIGNMENT so a slot keeps its replay residue, sizes every
        # buffer below.
        self._scratch = StageBSpillScratch(
            directory=root, max_bytes=max_bytes, nbytes=geometry.total_bytes,
            parts=geometry.max_parts, part_padding=ADDRESS_ALIGNMENT,
            alignment=ADDRESS_ALIGNMENT, max_block=max_block)
        self._block = block = self._scratch.block
        # A tensor's slot: its residue ahead of it, the grid's tail after it.
        pair = 2 * (geometry.largest_tensor_bytes + ADDRESS_ALIGNMENT + block)
        pair += -pair % block
        per_probe = geometry.x_bytes + geometry.g_bytes_per_probe
        self.arena_bytes = _ceil(max(pair, min(ARENA_BYTES, per_probe + pair)), block)
        self.read_bytes = _ceil(max(pair, min(READ_BYTES, per_probe + pair)), block)
        self._replay_stream = None
        self._replay_budget = None
        # Reader threads add to the telemetry; the lock keeps the sums whole.
        self._telemetry_lock = threading.Lock()
        self._arenas: list[_Arena] = []
        self._free: queue.Queue | None = None
        self._pending: queue.Queue | None = None
        self._writer = None
        self._write_error = None
        self._arena = None
        self._probe = None
        self._captured = 0
        self._failed = False
        self.telemetry = {
            "bound_bytes": geometry.total_bytes, "ceiling_bytes": int(max_bytes),
            "x_bytes_written": 0, "g_bytes_written": 0, "bytes_read": 0,
            "runs_written": 0, "reads": 0, "x_streams": 0, "shared_input_linears": 0,
            "records_per_probe": [], "x_entries": 0, "x_digest_checks": 0,
            "capture_wall_s": [], "replay_wall_s": 0.0,
            "hook_wait_s": 0.0, "writer_busy_s": 0.0, "reader_wait_s": 0.0,
            "arena_bytes": self.arena_bytes,
            "arenas": ARENA_COUNT if self._threads else 1,
            "read_buffer_bytes": self.read_bytes,
            "read_buffers": READ_BUFFER_COUNT,
            "threads": self._threads,
            "direct_io_block": block, "reserved_bytes": self._scratch.capacity,
            "write_call_bytes": WRITE_CALL_BYTES, "read_call_bytes": READ_CALL_BYTES,
            "write_calls": 0, "file_bytes_written": 0, "read_calls": 0,
            "file_bytes_read": 0, "replay_stream": None,
            "replay_reclaims": 0, "replay_reclaimed_bytes": 0,
        }
        if self.accumulation == OPERATOR_GEMM:
            self.telemetry.update(accumulation=self.accumulation,
                                  chunk_rows=self.chunk_rows, row_chunks=0)

    # -- reservations the caller charges to its capture guard ---------------
    @property
    def capture_reserve_bytes(self):
        """Pinned arenas plus every target input held until its backward.

        Each arena is allocated one grid block over, to align its start.
        """
        return self.capture_reserve_host_bytes + self.capture_reserve_device_bytes

    @property
    def capture_reserve_host_bytes(self):
        """The pinned host arenas, until ``capture`` allocates them on entry."""
        return 0 if self._arenas else (
            (self.arena_bytes + self._block) * self.telemetry["arenas"])

    @property
    def capture_reserve_device_bytes(self):
        """Every target input the capture holds on the device until its backward."""
        return self.geometry.batch_x_bytes

    @property
    def replay_reserve_bytes(self):
        """Pinned read buffers plus the device staging a window keeps live.

        Under ``operator_gemm`` also one input stream's row blocks and one
        chunk's FP32 GEMM operands (bounded at the probe-0 capture).
        """
        return self.replay_reserve_host_bytes + self.replay_reserve_device_bytes

    @property
    def replay_reserve_host_bytes(self):
        """The pinned read buffers a replay phase allocates (``READ_BUFFER_COUNT``)."""
        return self.replay_chunk_bytes * READ_BUFFER_COUNT

    @property
    def replay_reserve_device_bytes(self):
        """The device staging, row blocks and GEMM operands a window keeps live."""
        return 2 * (self.read_bytes + ADDRESS_ALIGNMENT) + self._gemm_reserve

    # -- lifetime -------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        writer, self._writer = self._writer, None
        if writer is not None:
            self._pending.put(None)
            writer.join()
        self._arenas.clear()
        self._arena = None
        try:
            self.close_replay_stream()
        finally:
            for window in self._windows:
                window.dedupe.clear()
            self._scratch.close()

    # -- the replay's read stream (PQ #1348) ----------------------------------
    def bind_replay_budget(self, budget):
        """The IO engine budget the replay's stream reads ahead within.

        Unbound, the stream holds the phase's own ``READ_BUFFER_COUNT``
        buffers: the chunk it consumes and one chunk ahead.
        """
        if self._replay_stream is not None:
            raise RuntimeError("Stage B spill replay stream is already open")
        self._replay_budget = budget

    @property
    def replay_chunk_bytes(self):
        """Host bytes one read chunk's buffer holds: its IO engine charge."""
        return self.read_bytes + self._block

    def reclaim_replay(self, shortfall_bytes):
        """Drop chunks read ahead, farthest first; a capture guard's reclaimer.

        Returns the bytes freed: the dropped chunks' charge, or on CUDA what
        the caching host allocator hands back. A dropped buffer is pinned,
        and the allocator keeps a freed pinned block for reuse, so the guard's
        reading drops only once its idle blocks are emptied.
        """
        stream = self._replay_stream
        freed = 0 if stream is None else stream.reclaim(shortfall_bytes)
        if self._cuda:
            freed = max(freed, _empty_host_cache())
        if freed:
            self.telemetry["replay_reclaims"] += 1
            self.telemetry["replay_reclaimed_bytes"] += freed
        return freed

    def close_replay_stream(self):
        """Close the replay's stream and keep its counters in the telemetry.

        The per-group waits become a summary: the stream takes one group per
        read chunk, thousands in a quantum.
        """
        stream, self._replay_stream = self._replay_stream, None
        if stream is None:
            return
        try:
            stream.close()
        finally:
            counters = dict(stream.counters)
            taken = counters.pop("groups_taken", [])
            waits = [group["wait_s"] for group in taken]
            counters.update(groups_taken=len(taken),
                            max_group_wait_s=max(waits, default=0.0),
                            reread_chunks=sum(1 for group in taken
                                              if group["reread_entries"]))
            self.telemetry["replay_stream"] = counters
            if self._cuda:
                _empty_host_cache()

    def _open_replay_stream(self):
        """One stream over every chunk the replay reads, in replay order.

        A group is one chunk, keyed ``(window, probe, position)``: windows
        with pending targets in index order, each probe's chunks in plan order,
        which is the order ``replay`` takes them. A chunk is read only once
        its probe's capture has ended (``_chunk_ready``).
        """
        from .io_engine import FixedBudget, ReadEntry, read_stream

        held = self.replay_chunk_bytes
        entries = [
            ReadEntry(key=(index, probe, position), path=None, size=item[1][3],
                      limit=item[1][3], held_bytes=held, expected_sha256=None,
                      decoder=None, group=(index, probe, position),
                      reader=partial(self._read_chunk, window, probe, item))
            for index, window in enumerate(self._windows) if window.names
            for probe in range(self.n_probes)
            for position, item in enumerate(window.plan)]
        if not entries:
            return None
        budget = self._replay_budget or FixedBudget(
            buffer_bytes=held, headroom=self.replay_reserve_host_bytes)
        self._replay_stream = read_stream(entries, budget=budget, ready=self._chunk_ready)
        return self._replay_stream

    def _chunk_ready(self, group, cancel):
        """A chunk is on the file once its probe's capture has ended."""
        del cancel
        return group[1] < self._captured

    def _read_chunk(self, window, probe, item):
        """The engine's reader for one chunk: a new buffer, filled in plan order."""
        buffer = _aligned_buffer(self.read_bytes, self._block, self._cuda)
        self._fill(window, probe, item, memoryview(buffer.numpy()))
        return buffer, (item[1][3],)

    def _require_healthy(self):
        if self._failed:
            raise RuntimeError("Stage B spill failed earlier in this quantum")
        if self._write_error is not None:
            self._failed = True
            raise RuntimeError("Stage B spill writer failed") from self._write_error

    # -- capture --------------------------------------------------------------
    @contextmanager
    def capture(self, probe_index, modules, specs_by_qname, *, activation_max_abs,
                projection_backend):
        """Observe one probe's single forward/backward over every pending target."""
        self._require_healthy()
        if probe_index != self._captured or probe_index >= self.n_probes:
            raise RuntimeError("Stage B spill captures probes once, in index order")
        if set(modules) != set(self._window_of):
            raise RuntimeError("Stage B spill capture roster differs from its windows")
        self._probe = int(probe_index)
        for window in self._windows:
            window.entry_cursor = {}
            window.record_cursor = 0
            window.g_logical = {}
            window.dedupe.clear()
        self._start_arenas()
        started = time.time()
        records_before = self._records_seen = 0
        try:
            observer = _SpillObserver(self, modules, specs_by_qname,
                                      activation_max_abs=activation_max_abs,
                                      projection_backend=projection_backend)
            with observer:
                observer.begin_probe()
                yield observer
                if observer._pending:
                    raise RuntimeError("Stage B spill has pending backward observations")
                observer.active = False
            self._end_capture()
        except BaseException:
            self._failed = True
            raise
        self.telemetry["records_per_probe"].append(self._records_seen - records_before)
        self.telemetry["capture_wall_s"].append(time.time() - started)
        self._captured += 1

    def _start_arenas(self):
        if self._arenas:
            return
        count = ARENA_COUNT if self._threads else 1
        self._arenas = [_Arena(self.arena_bytes, self._cuda, self._block)
                        for _ in range(count)]
        self._free = queue.Queue()
        for arena in self._arenas[1:]:
            self._free.put(arena)
        self._arena = self._arenas[0]
        if self._threads:
            self._pending = queue.Queue()
            self._writer = threading.Thread(target=self._writer_loop,
                                            name="stage-b-spill-writer", daemon=True)
            self._writer.start()

    def _record(self, name, x, selected):
        """Runs inside a backward hook: stage one invocation's operands."""
        self._require_healthy()
        index = self._window_of.get(name)
        if index is None:
            raise RuntimeError(f"Stage B spill observed an undeclared target {name}")
        if x.dtype != self.dtype or selected.dtype != self.dtype:
            raise RuntimeError(f"Stage B spill operand dtype differs for {name}")
        if x.device != self.device or selected.device != self.device:
            raise RuntimeError(f"Stage B spill operand residency differs for {name}")
        if not _is_dense(x):
            raise RuntimeError(
                f"Stage B spill refuses a non-dense input for {name}: the "
                "activation QDQ reads it in place, so a copy is not the same operand")
        window = self._windows[index]
        key = (x.untyped_storage().data_ptr(), x.storage_offset(), tuple(x.shape),
               tuple(x.stride()), x._version)
        held = window.dedupe.get(key)
        if held is None:
            owner = name
            layout = _layout(x, dense=True)
            nbytes = x.numel() * self.element_size
            entries = window.entries.setdefault(owner, [])
            entry = window.entry_cursor.get(owner, 0)
            if self._probe == 0:
                logical = window.x_logical.get(owner, 0)
                entries.append(_Entry(logical, nbytes, layout))
                window.x_logical[owner] = logical + nbytes
            elif (entry >= len(entries)
                    or not self._same_layout(entries[entry].layout, layout)
                    or entries[entry].nbytes != nbytes):
                raise RuntimeError(
                    f"Stage B spill probe {self._probe} input layout differs from "
                    f"probe 0 for {name}")
            window.entry_cursor[owner] = entry + 1
            # The held reference keeps this storage from being reused inside
            # the sample, so an address key cannot name two tensors.
            window.dedupe[key] = (owner, entry, x)
            digest = self._input_digest(x)
            if self._probe == 0:
                entries[entry].digest = digest
                self._stage(index, ("x", owner), entry, entries[entry].logical, x,
                            entries[entry].layout[2])
            else:
                # Compared once the capture ends; never copied to the host.
                window.x_checks.append((owner, entry, digest))
        else:
            owner, entry = held[0], held[1]
        if window.x_source.setdefault(name, owner) != owner:
            raise RuntimeError(
                f"Stage B spill Linear {name} reads inputs first read by more than "
                "one Linear; its rows would not be one input stream")
        g_layout = _layout(selected, dense=_is_dense(selected))
        g_bytes = selected.numel() * self.element_size
        logical = window.g_logical.get(name, 0)
        if self._probe == 0:
            window.records.append((name, owner, entry, logical, g_bytes, g_layout))
            g_residue = g_layout[2]
        else:
            cursor = window.record_cursor
            if (cursor >= len(window.records)
                    or window.records[cursor][:4] != (name, owner, entry, logical)
                    or window.records[cursor][4] != g_bytes
                    or not self._same_layout(window.records[cursor][5], g_layout)):
                raise RuntimeError(
                    f"Stage B spill probe {self._probe} invocation order or gradient "
                    f"layout differs from probe 0 at {name}")
            # The replay places every probe's gradient at probe 0's residue.
            g_residue = window.records[cursor][5][2]
        window.record_cursor += 1
        window.g_logical[name] = logical + g_bytes
        self._records_seen += 1
        self._stage(index, ("g", name), window.record_cursor - 1, logical, selected,
                    g_residue)

    def _input_digest(self, x):
        """``x``'s digest, queued on its device behind the op that made it."""
        if self._digest is None:
            self._digest = _InputDigest(self.device)
        return self._digest(x)

    def _same_layout(self, recorded, observed):
        """Probe 0's layout is the one every probe is replayed at.

        Shape and strides must match on every device. The address residue
        must match on CUDA, where the caching allocator's 512-byte blocks
        make it a deterministic function of the view offset, so a mismatch
        means a different operand. A host allocation's residue is not: the
        windowed replay itself runs each window's pass at a different host
        address, and host kernels on these fresh or elementwise operands
        do not depend on it.
        """
        return recorded[:2] == observed[:2] and (not self._cuda or recorded[2] == observed[2])

    def _stage(self, window_index, stream, index, logical, tensor, residue):
        """Copy ``tensor`` into an arena slot as part of ``stream`` at ``logical``.

        The slot (``_slot``) puts the tensor at the address residue the
        replay will rebuild it at, so the slot writes to the file and later
        reads back into the read buffer as one block-aligned envelope.
        """
        nbytes = tensor.numel() * self.element_size
        arena = self._arena
        start, offset, end = _slot(arena.used, residue, nbytes, self._block)
        if end > len(arena.view):
            self._flush()
            arena = self._arena
            start, offset, end = _slot(0, residue, nbytes, self._block)
            if end > len(arena.view):
                raise RuntimeError("Stage B spill tensor exceeds its arena bound")
        if nbytes:
            destination = arena.buffer.narrow(0, offset, nbytes).view(self.dtype)
            destination.copy_(_storage_order(tensor), non_blocking=self._cuda)
        arena.parts.append((window_index, stream, index, logical, nbytes, offset, start, end))
        arena.used = end
        arena.probe = self._probe

    def _end_batch(self):
        for window in self._windows:
            window.dedupe.clear()

    def _flush(self):
        arena = self._arena
        if not arena.parts:
            return
        if self._cuda:
            arena.event = torch.cuda.Event()
            arena.event.record(torch.cuda.current_stream(self.device))
        if self._threads:
            self._pending.put(arena)
            started = time.time()
            self._arena = self._free.get()
            self.telemetry["hook_wait_s"] += time.time() - started
            self._require_healthy()
        else:
            self._write_arena(arena)
            self._reset(arena)

    def _drain(self):
        """Flush the current arena and wait until every arena is written."""
        self._flush()
        if self._threads:
            parked = [self._arena]
            started = time.time()
            while len(parked) < len(self._arenas):
                parked.append(self._free.get())
            self.telemetry["hook_wait_s"] += time.time() - started
            self._arena = parked[0]
            for arena in parked[1:]:
                self._free.put(arena)
        self._require_healthy()

    @staticmethod
    def _reset(arena):
        arena.used, arena.parts, arena.event, arena.probe = 0, [], None, None

    def _writer_loop(self):
        while True:
            arena = self._pending.get()
            if arena is None:
                return
            try:
                if self._write_error is None:
                    self._write_arena(arena)
            except BaseException as exc:  # surfaced on the capture thread
                self._write_error = exc
            finally:
                self._reset(arena)
                self._free.put(arena)

    def _write_arena(self, arena):
        """Write one arena: its (window, stream) groups back to back, slot by slot.

        The groups take one contiguous stretch of the file, written with
        direct I/O straight from the pinned arena in calls of at most
        ``WRITE_CALL_BYTES``. Each slot keeps its place in the grid, so a
        tensor's file offset has the residue it is replayed at. A stream's
        consecutive tensors that abut in the file share one run.
        """
        started = time.time()
        if arena.event is not None:
            arena.event.synchronize()
        probe, view = arena.probe, arena.view
        groups: dict[tuple, list] = {}
        for part in arena.parts:
            groups.setdefault((part[0], part[1]), []).append(part)
        placed, slots, size = [], [], 0
        for (window_index, (kind, stream)), parts in groups.items():
            window = self._windows[window_index]
            logical = parts[0][3]
            for part in parts:
                if part[3] != logical:
                    raise RuntimeError("Stage B spill stream is not contiguous")
                logical += part[4]
            if kind == "x":
                if probe != 0:
                    raise RuntimeError(
                        f"Stage B spill staged a probe {probe} input; only probe 0's "
                        "inputs are written")
                runs = window.x_runs.setdefault(stream, [])
            else:
                runs = window.g_runs.setdefault((stream, probe), [])
            if runs and runs[-1][0] + runs[-1][2] != parts[0][3]:
                raise RuntimeError("Stage B spill runs are out of stream order")
            if not runs and parts[0][3] != 0:
                raise RuntimeError("Stage B spill stream does not start at zero")
            for part in parts:
                placed.append((runs, part, size + part[5] - part[6]))
                size += part[7] - part[6]
                if part[7] > part[6]:
                    slots.append(view[part[6]:part[7]])
            key = "x_bytes_written" if kind == "x" else "g_bytes_written"
            self.telemetry[key] += logical - parts[0][3]
        base = self._scratch.allocate(size)
        for runs, part, relative in placed:
            logical, nbytes, file_offset = part[3], part[4], base + relative
            if not nbytes:
                continue  # An empty tensor has no slot and needs no run.
            if runs and runs[-1][1] + runs[-1][2] == file_offset:
                # Abuts the stream's last run in the file: extend it.
                runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + nbytes)
            else:
                runs.append((logical, file_offset, nbytes))
                self.telemetry["runs_written"] += 1
        if slots:
            self.telemetry["write_calls"] += self._scratch.write(
                base, slots, call_bytes=WRITE_CALL_BYTES)
        self.telemetry["file_bytes_written"] += size
        self.telemetry["writer_busy_s"] += time.time() - started

    def _end_capture(self):
        self._drain()
        probe = self._probe
        for index, window in enumerate(self._windows):
            if probe == 0:
                window.plan = self._plan(window)
                if self.accumulation == OPERATOR_GEMM:
                    self._gemm_reserve = max(self._gemm_reserve, self._row_bounds(window))
                window.x_starts = {owner: [run[0] for run in runs]
                                   for owner, runs in window.x_runs.items()}
                self.telemetry["x_entries"] += sum(len(e) for e in window.entries.values())
                self.telemetry["x_streams"] += len(window.entries)
                self.telemetry["shared_input_linears"] += sum(
                    1 for name, owner in window.x_source.items() if name != owner)
            elif (window.record_cursor != len(window.records)
                    or any(window.entry_cursor.get(owner, 0) != len(entries)
                           for owner, entries in window.entries.items())):
                raise RuntimeError(
                    f"Stage B spill probe {probe} observed a different invocation "
                    f"count than probe 0 in window {index}")
            else:
                self._check_inputs(index, window, probe)
            for name in {record[0] for record in window.records}:
                runs = window.g_runs.get((name, probe), [])
                window.g_starts[(name, probe)] = [run[0] for run in runs]
            if any(entry.digest is None for entries in window.entries.values()
                   for entry in entries):
                raise RuntimeError("Stage B spill input digest missing")
            window.dedupe.clear()
        if probe == self.n_probes - 1:
            # The last capture is done: release the pinned write arenas now.
            writer, self._writer = self._writer, None
            if writer is not None:
                self._pending.put(None)
                writer.join()
            self._arenas.clear()
            self._arena = None

    def _check_inputs(self, index, window, probe):
        """Fail unless every input of ``probe`` equals probe 0's, bit for bit.

        One device-to-host read per window: the per-input digests are
        stacked and compared on the device, and only the flags come back.
        """
        checks, window.x_checks = window.x_checks, []
        if not checks:
            return
        observed = torch.stack([digest for _owner, _entry, digest in checks])
        expected = torch.stack([window.entries[owner][entry].digest
                                for owner, entry, _digest in checks])
        differs = (observed != expected).any(dim=1).tolist()
        if any(differs):
            owner, entry, _digest = checks[differs.index(True)]
            raise RuntimeError(
                f"Stage B spill probe {probe} input differs from probe 0 "
                f"in window {index} ({owner}, entry {entry}); "
                "the replay would not be the windowed arithmetic")
        self.telemetry["x_digest_checks"] += len(checks)

    # -- replay ---------------------------------------------------------------
    def _row_bounds(self, window):
        """Record each Linear's rows; return the window's operator-GEMM bytes.

        One input stream's Linears hold row blocks at once (the plan replays
        a stream to its end before the next), and one chunk at a time is
        upcast: FP32 input and gradient rows, the QDQ output and its FP32
        difference.
        """
        es = self.element_size
        for name, _owner, _entry, _logical, _g_bytes, (shape, _, _) in window.records:
            rows = 1
            for size in shape[:-1]:
                rows *= int(size)
            counts = window.rows.setdefault(name, [0, 0, int(shape[-1])])
            counts[0] += rows
        for name, owner in window.x_source.items():
            window.rows[name][1] = int(window.entries[owner][0].layout[0][-1])
        blocks, upcast = {}, 0
        for name, (rows, x_width, g_width) in window.rows.items():
            held = min(rows, self.chunk_rows)
            blocks[window.x_source[name]] = (blocks.get(window.x_source[name], 0)
                                             + held * (x_width + g_width) * es)
            upcast = max(upcast, held * (x_width * (8 + es) + g_width * 4))
        return max(blocks.values(), default=0) + upcast

    def _plan(self, window):
        """Replay order: one input stream's records at a time, in read chunks.

        Each input stream's records (its owner's and those of any Linear that
        shares it) replay in firing order; every Linear belongs to exactly one
        stream, so every statistics key keeps its own order. A chunk fits one
        read buffer, where each tensor takes a slot by the arena's rule
        (``_slot``): the same residue, on the same direct-I/O grid. The
        buffer holds the chunk's inputs first, then each Linear's gradients
        together, in the order the file holds them, so a run of tensors that
        abut in the file lands as one read. Where a tensor sits in the buffer
        does not change what the replay computes: only its residue does.
        """
        block = self._block
        streams: dict[str, list[int]] = {}
        for position, record in enumerate(window.records):
            streams.setdefault(record[1], []).append(position)
        order = [name for name in window.names if name in streams]
        plan = []

        def close(owner, records, inputs, gradients):
            cursor, new, placed = 0, [], []
            for entry, nbytes, residue in inputs:
                _start, offset, cursor = _slot(cursor, residue, nbytes, block)
                new.append((entry, offset))
            by_name: dict[str, list] = {}
            for position, name, nbytes, residue in gradients:
                by_name.setdefault(name, []).append((position, nbytes, residue))
            for items in by_name.values():
                for position, nbytes, residue in items:
                    _start, offset, cursor = _slot(cursor, residue, nbytes, block)
                    placed.append((position, offset))
            plan.append((owner, (tuple(records), tuple(new), tuple(placed), cursor)))

        for owner in order:
            records, inputs, gradients, used, next_entry = [], [], [], 0, 0
            entries = window.entries[owner]
            for position in streams[owner]:
                name, _, entry, _, g_bytes, g_layout = window.records[position]
                need = 0
                if entry == next_entry:
                    x = (entry, entries[entry].nbytes, entries[entry].layout[2])
                    need += _slot(0, x[2], x[1], block)[2]
                elif entry > next_entry:
                    raise RuntimeError("Stage B spill entries are not in first-use order")
                need += _slot(0, g_layout[2], g_bytes, block)[2]
                if records and used + need > self.read_bytes:
                    close(owner, records, inputs, gradients)
                    records, inputs, gradients, used = [], [], [], 0
                if entry == next_entry:
                    inputs.append(x)
                    next_entry += 1
                gradients.append((position, name, g_bytes, g_layout[2]))
                used += need
                window.last_ref[(owner, entry)] = position
                records.append(position)
            if next_entry != len(entries):
                raise RuntimeError("Stage B spill input stream has an unread entry")
            if records:
                close(owner, records, inputs, gradients)
        if any(chunk[3] > self.read_bytes for _, chunk in plan):
            raise RuntimeError("Stage B spill record exceeds its read buffer")
        return plan

    @staticmethod
    def _physical(runs, starts, logical, nbytes):
        run = bisect.bisect_right(starts, logical) - 1
        if run < 0:
            raise RuntimeError("Stage B spill stream index is incomplete")
        start, file_offset, size = runs[run]
        if logical + nbytes > start + size:
            raise RuntimeError("Stage B spill stream index is incomplete")
        return file_offset + (logical - start)

    def _fill(self, window, probe, item, view):
        """Read one chunk's tensors into ``view`` at their planned offsets.

        Each tensor's slot envelope is read from the file straight into the
        buffer (the plan and the file share the slot rule, so the envelopes
        line up); envelopes that abut in both the file and the buffer are one
        read, cut at ``READ_CALL_BYTES``. The calls run in order on the IO
        engine's thread that reads this chunk; the engine reads chunks in
        parallel.
        """
        owner, (_records, new, gradients, _used) = item
        block = self._block
        pieces = []
        entries, x_runs, x_starts = (window.entries[owner], window.x_runs[owner],
                                     window.x_starts[owner])
        for entry, offset in new:
            record = entries[entry]
            if record.nbytes:
                pieces.append((self._physical(x_runs, x_starts, record.logical,
                                              record.nbytes), offset, record.nbytes))
        for position, offset in gradients:
            name, _, _, logical, nbytes, _ = window.records[position]
            if nbytes:
                pieces.append((self._physical(window.g_runs[(name, probe)],
                                              window.g_starts[(name, probe)], logical,
                                              nbytes), offset, nbytes))
        spans = []
        for file_offset, offset, nbytes in sorted(pieces):
            if file_offset % block != offset % block:
                raise RuntimeError("Stage B spill tensor is off its replay residue")
            lead = file_offset % block
            low, high = file_offset - lead, _ceil(file_offset + nbytes, block)
            at = offset - lead
            if spans and spans[-1][1] == low and spans[-1][2] + spans[-1][1] - spans[-1][0] == at:
                spans[-1][1] = max(spans[-1][1], high)
            elif spans and low < spans[-1][1]:
                raise RuntimeError("Stage B spill read envelopes overlap")
            else:
                spans.append([low, high, at])
        calls = []
        for low, high, at in spans:
            for cut in range(low, high, READ_CALL_BYTES):
                size = min(READ_CALL_BYTES, high - cut)
                calls.append((cut, at + (cut - low), size))
        read = self._scratch.read_into
        done = [read(cut, [view[at:at + size]]) for cut, at, size in calls]
        with self._telemetry_lock:
            self.telemetry["bytes_read"] += sum(nbytes for _, _, nbytes in pieces)
            self.telemetry["file_bytes_read"] += sum(done)
            self.telemetry["reads"] += len(spans)
            self.telemetry["read_calls"] += len(calls)

    def replay(self, window_index, probe_index, lease):
        """Feed ``lease`` this window's spilled invocations for one probe."""
        self._require_healthy()
        if probe_index >= self._captured:
            raise RuntimeError("Stage B spill replay precedes its probe's capture")
        window = self._windows[window_index]
        if set(lease.modules) != set(window.names):
            raise RuntimeError("Stage B spill replay lease roster differs from its window")
        started = time.time()
        live: dict[tuple[str, int], torch.Tensor] = {}
        es = self.element_size
        gemm = self.accumulation == OPERATOR_GEMM
        blocks: dict[str, _RowBlock] = {}
        stream = None
        reads = self._replay_stream
        if reads is None and window.plan:
            reads = self._open_replay_stream()
        try:
            for chunk, item in enumerate(window.plan):
                waited = time.time()
                delivered = reads.take((window_index, probe_index, chunk))
                self.telemetry["reader_wait_s"] += time.time() - waited
                host = delivered[0].value
                del delivered
                owner, (records, new, gradients, used) = item
                if gemm and owner != stream:
                    # The plan replays one input stream to its end first.
                    self._emit_blocks(blocks, lease)
                    stream = owner
                staging = torch.empty(used + 2 * ADDRESS_ALIGNMENT, dtype=torch.uint8,
                                      device=self.device)
                shift = (-staging.data_ptr()) % ADDRESS_ALIGNMENT
                # A pinned buffer's copy is recorded on its block, so the
                # caching host allocator reuses it only after the copy: the
                # stream may count it released once the copy is queued.
                staging.narrow(0, shift, used).copy_(host.narrow(0, 0, used),
                                                     non_blocking=self._cuda)
                del host
                reads.release()
                typed = staging.narrow(0, 0, (staging.numel() // es) * es).view(self.dtype)
                entries = window.entries[owner]
                for entry, offset in new:
                    shape, stride, _ = entries[entry].layout
                    live[(owner, entry)] = torch.as_strided(
                        typed, shape, stride, (shift + offset) // es)
                gradient_at = dict(gradients)
                for position in records:
                    name, _, entry, _, _, (shape, stride, _) = window.records[position]
                    gradient = torch.as_strided(typed, shape, stride,
                                                (shift + gradient_at[position]) // es)
                    if gemm:
                        self._append_rows(window, blocks, lease, name,
                                          live[(owner, entry)], gradient)
                    else:
                        lease._observe_invocation(name, lease.modules[name].weight,
                                                  live[(owner, entry)], gradient)
                    if window.last_ref[(owner, entry)] == position:
                        del live[(owner, entry)]
                del staging, typed
            if gemm:
                self._emit_blocks(blocks, lease)
            if live:
                raise RuntimeError("Stage B spill replay left an input unconsumed")
        except BaseException:
            self._failed = True
            raise
        finally:
            live.clear()
            blocks.clear()
        self.telemetry["replay_wall_s"] += time.time() - started

    def _append_rows(self, window, blocks, lease, name, x, gradient):
        """Copy one invocation's rows into ``name``'s block, emitting full chunks."""
        x2 = x.reshape(-1, x.shape[-1])
        g2 = gradient.reshape(-1, gradient.shape[-1])
        if x2.shape[0] != g2.shape[0]:
            raise RuntimeError(f"Stage B spill rows differ between input and gradient for {name}")
        block = blocks.get(name)
        if block is None:
            rows, x_width, g_width = window.rows[name]
            block = blocks[name] = _RowBlock(min(rows, self.chunk_rows), x_width, g_width,
                                             self.dtype, self.device)
        block.calls += 1
        start, total, capacity = 0, int(x2.shape[0]), int(block.x.shape[0])
        while start < total:
            take = min(capacity - block.filled, total - start)
            if take <= 0:
                raise RuntimeError(f"Stage B spill rows exceed their bound for {name}")
            block.x.narrow(0, block.filled, take).copy_(x2.narrow(0, start, take))
            block.g.narrow(0, block.filled, take).copy_(g2.narrow(0, start, take))
            block.filled += take
            start += take
            if block.filled == capacity:
                self._emit(lease, name, block)

    def _emit(self, lease, name, block):
        lease.observe_row_chunk(name, lease.modules[name].weight,
                                block.x.narrow(0, 0, block.filled),
                                block.g.narrow(0, 0, block.filled), calls=block.calls)
        block.filled = block.calls = 0
        self.telemetry["row_chunks"] += 1

    def _emit_blocks(self, blocks, lease):
        """Emit every partial chunk of the stream just replayed, then free it."""
        for name, block in blocks.items():
            if block.filled or block.calls:
                self._emit(lease, name, block)
        blocks.clear()
