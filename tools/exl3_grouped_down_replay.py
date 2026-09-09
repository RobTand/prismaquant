"""Repeat one grouped-down call on its own live arguments, and say what changed.

The EXL3 forward takes two values from one prompt.  Whole-tensor boundary
hashes place the earliest whole-batch difference at the first routed MoE layer,
inside ``Glm5NextMoE``.  Every *observed boundary tensor* it consumes is equal
across repeats; its output is not.  Observed is the load-bearing word: the
boundary hooks record the tensors passed at module boundaries, and a pointer
table's bytes being equal says the same device addresses were passed, not that
the weight bytes behind those addresses are equal.  Nothing here establishes
content equality of memory this process never reads.

This repeats the one call that produces that output, on the arguments that call
actually received, and reports what differs.  It answers exactly one question:
does ``exl3_fat_moe_down`` return the same bytes when invoked again on the same
live tensors?  The routing is among those tensors, so a difference cannot be
explained by a different expert selection -- selection is an input here, not a
variable.  Equally, nothing here infers route identity from boundary logits.

It does NOT answer whether the kernel is deterministic.  N identical replays
bound how often a difference appears at this shape and this occupancy, and
bound nothing else.  An all-identical result is not an exoneration and is
reported as what it is.  It also says nothing about gather or gate/up, which it
does not repeat, and it names no cause.

The served result is the first, real call.  The probe writes only into buffers
it allocates, and the original output is hashed before the replays and hashed
again after, so "untouched" is a measurement.  Hashes rather than
``torch.equal``, which reports two bit-identical NaNs as different and ``+0.0``
and ``-0.0`` as the same.

The ABI, read from ``exl3_fat_moe.cu:599-655`` rather than assumed:

    0 h2            half   [rows_cap, K]  contiguous CUDA
    1 down_ptrs     int64  [n_exp]        POINTER TABLE, not a weight body
    2 down_svh_ptrs int64  [n_exp]        POINTER TABLE, not a weight body
    3 out           float  [tokens, N]    the destination
    4 row_token     int64  [rows_cap]
    5 row_weight    half   [rows_cap]
    6 seg_expert    int32  [n_seg]
    7 seg_row0      int32  [n_seg]
    8 seg_rows      int32  [n_seg]
    9 num_segs      int32  [1]

All ten are small: the two pointer tables hold device addresses, so the weight
bytes they name are never passed to this entry point and are never hashed.
That is recorded on the result rather than left to be inferred.  ``rows_cap``
may exceed the used row count, so the tables can carry unused capacity; byte
equality of the whole passed tensor is still what is asked for.
"""

import hashlib
import json


REPLAY_SCHEMA = "prismaquant.exl3_grouped_down_replay/2"

# The boundary the whole-tensor comparison selected.  Named, not inferred from
# execution order: an ordering assumption would silently follow a change in
# which layers are routed.
DEFAULT_MODULE = "model.layers.3.mlp"
DEFAULT_REPEATS = 8
# The scorer's CONTEXT_LENGTH is the MoE input row count, so a whole prompt
# window arrives here as exactly that many rows.  (CONTEXT_LENGTH - 1 is the
# prediction count, which is a different number and not this one.)  A call with
# any other row count is a different request -- a warmup, a chunk, a decode
# step -- and must not consume the one shot.
DEFAULT_REQUIRED_ROWS = 2048

# Positional names, in ABI order.  A dict would not carry the order, and the
# order is what identifies an argument here.
ARGUMENT_NAMES = ("h2", "down_ptrs", "down_svh_ptrs", "out", "row_token",
                  "row_weight", "seg_expert", "seg_row0", "seg_rows",
                  "num_segs")

# Which arguments hold device addresses rather than values.  Their bytes are
# hashed like everything else -- the same addresses being passed is a real
# thing to check -- but this diagnostic does not read or hash what they point
# at, so it says nothing about those bytes.
POINTER_TABLES = ("down_ptrs", "down_svh_ptrs")

# Small integer tables get their values recorded outright, so a reader can
# compare two runs without re-deriving anything.
_VALUES_MAX_ELEMENTS = 4096


def _tensor_bytes(tensor):
    """The tensor's own bytes, contiguous, on host.

    A byte view rather than ``numpy()``, which raises on some of the dtypes
    this sees and which would hash a converted value rather than the stored
    bit pattern.  Bit patterns are what a bit-identity question is about.
    """
    import torch

    flat = tensor.detach().contiguous().cpu().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _hash(tensor):
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def describe(tensor, *, name):
    """Identity, full-byte hash, and what the hash does and does not cover."""
    record = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "contiguous": bool(tensor.is_contiguous()),
        "hashed": True,
        "sha256": _hash(tensor),
    }
    if name in POINTER_TABLES:
        # Said on the record, because a reader comparing two runs would
        # otherwise have to know the ABI to know what this hash covers.
        record["holds"] = "device addresses"
        record["pointed_to_bytes_hashed"] = False
    if not tensor.dtype.is_floating_point and tensor.numel() <= _VALUES_MAX_ELEMENTS:
        record["values"] = tensor.detach().cpu().tolist()
    return record


def _bf16_order(tensor):
    """A monotonic integer key over bfloat16, correct across the sign boundary.

    Reading the raw int16 bit pattern as a signed integer is wrong for
    negatives: the pattern grows as the value falls, and ``+0.0`` (0x0000) and
    ``-0.0`` (0x8000) read 32768 apart while being adjacent -- in fact equal.
    The usual total-order key fixes both: keep the pattern for non-negatives,
    and map a negative pattern ``u`` to ``0x8000 - u``.
    """
    import torch

    bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits >= 0x8000, 0x8000 - bits, bits)


def _finite(tensor):
    import torch

    finite = int(torch.isfinite(tensor).sum().item())
    total = int(tensor.numel())
    return {"finite_elements": finite, "elements": total,
            "all_finite": finite == total}


def compare(reference, candidate):
    """FP32 difference, and the BF16 difference that survives the cast.

    Both, because the cast at the end of the fused MoE is where most FP32
    ordering differences disappear.  How many survive it is the number that
    reaches the next layer; how many exist before it is the number the kernel
    produced.  Reporting one would hide which.

    Every count is reported beside a finiteness status, because a max delta
    over a tensor containing a NaN is not a magnitude and must not read as one.
    """
    import torch

    result = {
        "reference_sha256": _hash(reference),
        "candidate_sha256": _hash(candidate),
        "bitwise_identical": _hash(reference) == _hash(candidate),
        "reference_finite": _finite(reference),
        "candidate_finite": _finite(candidate),
    }

    differing = reference.ne(candidate)
    count = int(differing.sum().item())
    result["fp32_differing_elements"] = count
    result["fp32_elements"] = int(reference.numel())
    result["identical_fp32"] = count == 0
    delta = (reference - candidate).abs()
    finite_delta = delta[torch.isfinite(delta)]
    result["fp32_max_abs_delta"] = (
        float(finite_delta.max().item()) if finite_delta.numel() else 0.0)
    result["fp32_max_abs_delta_over_finite_only"] = bool(
        finite_delta.numel() != delta.numel())

    ref16 = reference.to(torch.bfloat16)
    can16 = candidate.to(torch.bfloat16)
    # Its own question, not inherited from FP32.  BF16 has FP32's exponent
    # range but 7 mantissa bits, so a finite FP32 just under the top of the
    # range rounds UP to infinity on this cast.  Reading FP32 finiteness as
    # BF16 finiteness would report a magnitude over an infinity.
    result["reference_bf16_finite"] = _finite(ref16)
    result["candidate_bf16_finite"] = _finite(can16)

    count16 = int(ref16.ne(can16).sum().item())
    result["bf16_differing_elements"] = count16
    result["identical_bf16"] = count16 == 0
    result["bf16_max_abs_delta"] = 0.0
    result["bf16_max_ulp_delta"] = 0
    both16_finite = torch.isfinite(ref16) & torch.isfinite(can16)
    result["bf16_max_over_finite_only"] = bool(
        int(both16_finite.sum().item()) != int(ref16.numel()))
    if count16 and bool(both16_finite.any().item()):
        delta16 = (ref16.float() - can16.float()).abs()[both16_finite]
        result["bf16_max_abs_delta"] = float(delta16.max().item())
        # A ULP distance is a count of representable steps, which infinities
        # and NaNs do not have.  Taken over the pairs where both are finite.
        ulp = (_bf16_order(ref16) - _bf16_order(can16)).abs()[both16_finite]
        result["bf16_max_ulp_delta"] = int(ulp.max().item())
    return result


class GroupedDownReplay:
    """One armed shot at one boundary, or nothing at all.

    Arming is deliberately narrow.  The capture state says whether a request is
    a scored one, so an engine warmup cannot consume the shot; the module hook
    says whether this call belongs to the named boundary, so call ordering is
    not assumed; and the row count says whether this is the whole prompt window
    rather than a chunk.  All three must hold.
    """

    def __init__(self, state, *, module_name=DEFAULT_MODULE,
                 repeats=DEFAULT_REPEATS, required_rows=DEFAULT_REQUIRED_ROWS):
        if int(repeats) < 2:
            raise ValueError("a replay of fewer than two calls compares nothing")
        if int(required_rows) < 1:
            raise ValueError("required_rows must be a positive row count")
        self.state = state
        self.module_name = str(module_name)
        self.repeats = int(repeats)
        self.required_rows = int(required_rows)
        self.inside = False
        self.fired = False
        self.record = None
        self.declined = []

    def armed(self):
        """True only inside a scored request.

        The owner's armed state is ``window_id is not None``: ``arm()`` sets it
        and ``finish()`` clears it.  ``PromptLogitsCapture`` has no ``armed``
        attribute, and reading one would leave this permanently disarmed
        against the real object while passing happily against a stand-in that
        invented the attribute.  The contract is the window id.
        """
        return getattr(self.state, "window_id", None) is not None

    def should_fire(self, out):
        if self.fired or not self.inside:
            return False
        if not self.armed():
            self.declined.append("not armed: request is not a scored window")
            return False
        if out.dim() != 2 or int(out.shape[0]) != self.required_rows:
            self.declined.append(
                f"row count {list(out.shape)} is not the {self.required_rows}-row window")
            return False
        return True

    def run(self, fn, args):
        """Replay ``fn`` on ``args``, into buffers this allocates.

        Storage, stated rather than implied.  Three output-shaped FP32 buffers
        are live at once: the retained clone of the original output, the
        retained clone of the first replay that later replays are compared
        against, and the current probe.  At ``[2048, 4096]`` FP32 that is about
        32 MB each, so about 96 MB resident for the length of the loop, plus
        two BF16 copies of half that transiently inside ``compare``.  Nothing
        is kept after the record is built.
        """
        import torch

        out = args[3]
        before = {name: describe(value, name=name)
                  for name, value in zip(ARGUMENT_NAMES, args)}

        # Retained, because "the original is unchanged" has to be checked
        # against something after the replays run.
        original = out.detach().clone()
        original_before = {
            "fp32_sha256": _hash(original),
            "bf16_sha256": _hash(original.to(torch.bfloat16)),
            "fp32_finite": _finite(original),
        }

        replays = []
        first = None
        for index in range(self.repeats):
            probe = torch.zeros_like(out)
            fn(args[0], args[1], args[2], probe, *args[4:])
            entry = {"index": index, "sha256": _hash(probe),
                     "finite": _finite(probe)}
            if first is None:
                first = probe.detach().clone()
            else:
                entry["vs_first_replay"] = compare(first, probe)
            entry["vs_original"] = compare(original, probe)
            replays.append(entry)
            del probe

        after = {name: describe(value, name=name)
                 for name, value in zip(ARGUMENT_NAMES, args)}
        original_after = {
            "fp32_sha256": _hash(out),
            "bf16_sha256": _hash(out.to(torch.bfloat16)),
            "fp32_finite": _finite(out),
        }
        # Hashes, not torch.equal: it calls two bit-identical NaNs different
        # and calls +0.0 and -0.0 the same, and this is a bit-identity check.
        unchanged = (original_before["fp32_sha256"] == original_after["fp32_sha256"])

        distinct = sorted({entry["sha256"] for entry in replays})

        self.record = {
            "schema": REPLAY_SCHEMA,
            "module": self.module_name,
            "rank": int(getattr(self.state, "rank", -1)),
            "world_size": int(getattr(self.state, "world_size", -1)),
            "window_id": getattr(self.state, "window_id", None),
            "repeats": self.repeats,
            "required_rows": self.required_rows,
            "argument_names": list(ARGUMENT_NAMES),
            "pointer_table_arguments": list(POINTER_TABLES),
            "pointed_to_weight_bytes_hashed": False,
            "original_out_before": original_before,
            "original_out_after": original_after,
            "original_out_unchanged_after_replays": unchanged,
            "inputs_before": before,
            "inputs_after": after,
            "inputs_stable": _inputs_stable(before, after),
            "replays": replays,
            "distinct_replay_digests": len(distinct),
            # The two conditions the interpretation names, evaluated here so a
            # reader does not have to assemble them from the entries.
            "same_input_inference_supported": bool(
                _inputs_stable(before, after)
                and all(entry["finite"]["all_finite"] for entry in replays)
                and original_before["fp32_finite"]["all_finite"]),
            "interpretation": (
                "distinct_replay_digests > 1 means this call returned different "
                "bytes across repeats.  Reading that as 'different output on the "
                "same arguments' requires inputs_stable to be true and the "
                "compared outputs to be finite; with inputs_stable false the "
                "arguments changed under the call and the digests say nothing "
                "about the kernel, and a non-finite result makes the magnitudes "
                "beside it meaningless.  A value of 1 bounds how often a "
                "difference appears at this shape and occupancy over this many "
                "repeats and bounds nothing else; it is not a determinism "
                "verdict, does not cover gather or gate/up, and names no cause.  "
                "down_ptrs and down_svh_ptrs hold device addresses: equal hashes "
                "mean the same addresses were passed, and this diagnostic does "
                "not read or hash the bytes they point at."),
        }
        self.fired = True
        return self.record


def _inputs_stable(before, after):
    """Whether every recorded argument still reads the same at the end.

    Full-byte hashes, so this is content equality of what was passed.  It is
    not content equality of what a pointer table points at: this diagnostic
    does not read those bytes, and ``pointed_to_bytes_hashed`` says so on each
    entry.
    """
    if set(before) != set(after):
        return False
    return all(before[name] == after[name] for name in before)


def install(model, ext, state, *, module_name=DEFAULT_MODULE,
            repeats=DEFAULT_REPEATS, required_rows=DEFAULT_REQUIRED_ROWS):
    """Wrap the extension entry point and mark the boundary that owns it.

    The wrap is on ``exl3_fat_moe_down`` itself rather than on its caller, so
    the replay sees the arguments the real call received instead of arguments
    rebuilt from the same inputs.  Rebuilding them would put the table
    construction inside the thing under test.
    """
    if getattr(model, "_tr3_grouped_down_replay", None) is not None:
        raise ValueError("candidate already has a grouped-down replay hook")
    replay = GroupedDownReplay(state, module_name=module_name, repeats=repeats,
                               required_rows=required_rows)
    target = None
    for name, module in model.named_modules():
        if name == module_name or name.endswith("." + module_name):
            target = module
            break
    if target is None:
        raise ValueError(f"replay boundary {module_name!r} is not a module of this model")

    def enter(_module, _args):
        replay.inside = True

    def leave(_module, _args, output):
        replay.inside = False
        return output

    handles = [target.register_forward_pre_hook(enter),
               target.register_forward_hook(leave)]

    original = ext.exl3_fat_moe_down

    def wrapped(*args):
        result = original(*args)
        if len(args) == len(ARGUMENT_NAMES) and replay.should_fire(args[3]):
            replay.run(original, args)
        return result

    ext.exl3_fat_moe_down = wrapped
    model._tr3_grouped_down_replay = replay
    model._tr3_grouped_down_handles = handles
    model._tr3_grouped_down_original = original
    return {"module": module_name, "repeats": repeats,
            "required_rows": required_rows, "schema": REPLAY_SCHEMA}


def uninstall(model, ext):
    """Put the entry point back, so a second install is not a second wrap."""
    original = getattr(model, "_tr3_grouped_down_original", None)
    if original is not None:
        ext.exl3_fat_moe_down = original
    for handle in getattr(model, "_tr3_grouped_down_handles", []) or []:
        handle.remove()
    record = getattr(model, "_tr3_grouped_down_replay", None)
    for attribute in ("_tr3_grouped_down_replay", "_tr3_grouped_down_handles",
                      "_tr3_grouped_down_original"):
        if hasattr(model, attribute):
            delattr(model, attribute)
    return None if record is None else record.record


def result_json(record):
    return json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
