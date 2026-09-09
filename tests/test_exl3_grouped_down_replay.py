"""The replay fires on one boundary, once, and reports what it saw.

Every kernel here is a fake, and deliberately so.  The real one runs on a GPU
this suite does not have, and the questions these tests ask are not about it:
whether the shot is spent on a warmup, whether a difference is reported when
one exists, whether the served output is checked rather than assumed.  A fake
that returns different bytes on the second call is the only way to see the
reporting path at all, since a real kernel cannot be made to differ on demand.

The fixture uses the real ABI, read from ``exl3_fat_moe.cu:599-655``: half
``h2``, int64 pointer tables, float32 ``out``, int64 ``row_token``, half
``row_weight``, int32 segment tables.  A fixture in convenient dtypes would
have hidden the two dtype-shaped defects this file now guards, and the pointer
tables are not weight bodies, which is why every one of the ten arguments is
hashed whole.

Where a test asserts that a check bites, it changes the driver -- the fake
kernel's behaviour, or the state object's shape -- rather than the recorded
result.  A fixture edited into the shape the check wants proves the fixture.
"""

import sys
from pathlib import Path

import pytest
import torch   # imported, not importorskip'd: a torchless runner must go red
               # rather than report a green suite of skips

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import exl3_grouped_down_replay as replay_mod  # noqa: E402


ROWS = 8
K = 32          # h2 columns; the kernel requires a multiple of 32
N = 256         # out columns; the kernel requires a multiple of 256
SEGS = 2


class _Capture:
    """The owner's contract, not a convenience.

    ``PromptLogitsCapture`` has no ``armed`` attribute.  Its armed state is
    ``window_id is not None``: ``arm()`` sets it, ``finish()`` clears it.  This
    stand-in carries exactly that, so a hook that reads anything else fails
    here the way it would fail against the real object.
    """

    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size
        self.window_id = None

    def arm(self, window_id):
        self.window_id = window_id

    def finish(self):
        self.window_id = None


class _Boundary(torch.nn.Module):
    """A module that calls the extension the way Glm5NextMoE does."""

    def __init__(self, ext, kernel_args):
        super().__init__()
        self.ext = ext
        self.kernel_args = kernel_args

    def forward(self, x):
        self.ext.exl3_fat_moe_down(*self.kernel_args)
        return x


class _Model(torch.nn.Module):
    def __init__(self, boundary):
        super().__init__()
        self.mlp = boundary

    def forward(self, x):
        return self.mlp(x)


class _Ext:
    """A stand-in extension module whose entry point can be swapped."""

    def __init__(self, kernel):
        self.exl3_fat_moe_down = kernel


def _args(rows=ROWS):
    """Ten positional arguments in ABI order and ABI dtypes."""
    return (
        torch.zeros(rows, K, dtype=torch.float16),                 # h2, half
        torch.tensor([0x7f00, 0x7f80], dtype=torch.int64),         # down_ptrs
        torch.tensor([0x7e00, 0x7e80], dtype=torch.int64),         # down_svh_ptrs
        torch.zeros(rows, N, dtype=torch.float32),                 # out, float
        torch.arange(rows, dtype=torch.int64),                     # row_token
        torch.ones(rows, dtype=torch.float16),                     # row_weight
        torch.zeros(SEGS, dtype=torch.int32),                      # seg_expert
        torch.zeros(SEGS, dtype=torch.int32),                      # seg_row0
        torch.full((SEGS,), rows // SEGS, dtype=torch.int32),      # seg_rows
        torch.tensor([SEGS], dtype=torch.int32),                   # num_segs
    )


def _deterministic(*args):
    args[3].add_(1.0)
    return None


def _make(kernel, args=None, state=None, armed=True, **kwargs):
    args = args if args is not None else _args()
    ext = _Ext(kernel)
    model = _Model(_Boundary(ext, args))
    state = state if state is not None else _Capture()
    if armed and state.window_id is None:
        state.arm("w0")
    replay_mod.install(model, ext, state, module_name="mlp", **kwargs)
    return model, ext, state, args


# --------------------------------------------------------------------------
# The armed contract.  Reading an attribute the owner does not have leaves the
# hook permanently disarmed against the real object.
# --------------------------------------------------------------------------

def test_a_scored_window_at_the_named_boundary_fires_once():
    model, ext, _, _ = _make(_deterministic, repeats=3, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record is not None
    assert record["module"] == "mlp"
    assert len(record["replays"]) == 3
    assert record["window_id"] == "w0"


def test_an_unarmed_capture_does_not_consume_the_shot():
    """A warmup runs the same code with no window armed.  If it could fire,
    the one shot would be spent before any scored request."""
    model, ext, _, _ = _make(_deterministic, armed=False, required_rows=ROWS)
    replay = model._tr3_grouped_down_replay
    model(torch.zeros(1))
    assert replay_mod.uninstall(model, ext) is None
    assert any("not armed" in reason for reason in replay.declined)


def test_a_finished_window_disarms_and_a_later_call_cannot_fire():
    model, ext, state, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    state.finish()
    model(torch.zeros(1))
    assert replay_mod.uninstall(model, ext) is None


def test_an_armed_attribute_is_not_the_contract():
    """The owner has no `armed` attribute.  A state that carries one but has
    no window must still be treated as unarmed, or the hook is reading a
    contract the real object does not implement."""

    class _Impostor:
        rank = 0
        world_size = 1
        armed = True
        window_id = None

    model, ext, _, _ = _make(_deterministic, state=_Impostor(), armed=False,
                             required_rows=ROWS)
    model(torch.zeros(1))
    assert replay_mod.uninstall(model, ext) is None


def test_a_wrong_row_count_does_not_consume_the_shot():
    """A chunked prefill or a decode step is a different request."""
    model, ext, _, _ = _make(_deterministic, required_rows=ROWS + 1)
    model(torch.zeros(1))
    assert replay_mod.uninstall(model, ext) is None


def test_a_call_outside_the_named_boundary_does_not_fire():
    """Other layers call the same entry point.  Only the named one counts."""
    args = _args()
    ext = _Ext(_deterministic)
    model = _Model(_Boundary(ext, args))
    state = _Capture()
    state.arm("w0")
    replay_mod.install(model, ext, state, module_name="mlp", required_rows=ROWS)
    # Call the extension directly, as another layer would: no hook has run, so
    # the replay is not inside its boundary.
    ext.exl3_fat_moe_down(*args)
    assert replay_mod.uninstall(model, ext) is None


def test_the_shot_is_spent_only_once_across_repeated_windows():
    model, ext, state, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    first = model._tr3_grouped_down_replay.record
    state.finish()
    state.arm("w1")
    model(torch.zeros(1))
    assert model._tr3_grouped_down_replay.record is first
    replay_mod.uninstall(model, ext)


def test_the_real_capture_arms_this_hook_when_it_is_available():
    """The stand-in above encodes the contract; this checks the contract was
    read off the real class and not off the stand-in."""
    scorer = Path("/home/rob/tmp/pq-tr3-repeatability/experiments/glm_tr3_full_vocab.py")
    if not scorer.is_file():
        pytest.skip(f"the scorer is not on this host: {scorer}")
    source = scorer.read_text()
    assert "self.window_id = None" in source
    assert "self.window_id, self.teacher = window_id, teacher" in source
    assert "def armed" not in source, "the owner grew an armed property; re-read it"


# --------------------------------------------------------------------------
# Coverage of the arguments.  All ten, whole, before and after.
# --------------------------------------------------------------------------

def test_every_one_of_the_ten_arguments_is_hashed_whole():
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert list(record["inputs_before"]) == list(replay_mod.ARGUMENT_NAMES)
    for name, entry in record["inputs_before"].items():
        assert entry["hashed"] is True, name
        assert len(entry["sha256"]) == 64, name


def test_the_pointer_tables_say_what_their_hash_does_not_cover():
    """Equal hashes mean the same device addresses were passed.  The bytes at
    those addresses are never read here, and a reader must not have to know
    the ABI to know that."""
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["pointed_to_weight_bytes_hashed"] is False
    for name in ("down_ptrs", "down_svh_ptrs"):
        entry = record["inputs_before"][name]
        assert entry["holds"] == "device addresses"
        assert entry["pointed_to_bytes_hashed"] is False
    assert "pointed_to_bytes_hashed" not in record["inputs_before"]["h2"]


def test_a_half_activation_is_hashed_rather_than_identified():
    """h2 is half, and an earlier size-threshold policy left it pointer-only,
    so input stability could not rule out content mutation."""
    entry = replay_mod.describe(torch.zeros(2048, 4096, dtype=torch.float16),
                                name="h2")
    assert entry["hashed"] is True
    assert entry["dtype"] == "torch.float16"


def test_a_mutated_argument_is_reported_whatever_its_dtype():
    for index, name in ((0, "h2"), (4, "row_token"), (6, "seg_expert")):
        def mutating(*args, _i=index):
            args[_i].add_(1)

        model, ext, _, _ = _make(mutating, repeats=2, required_rows=ROWS)
        model(torch.zeros(1))
        record = replay_mod.uninstall(model, ext)
        assert record["inputs_stable"] is False, name
        assert (record["inputs_before"][name]["sha256"]
                != record["inputs_after"][name]["sha256"]), name


def test_stable_inputs_are_reported_stable():
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["inputs_stable"] is True


def test_a_small_integer_table_carries_its_values():
    table = torch.tensor([3, 1, 4, 1], dtype=torch.int32)
    record = replay_mod.describe(table, name="seg_expert")
    assert record["values"] == [3, 1, 4, 1]


def test_the_hash_reads_the_stored_bits():
    a = torch.tensor([0.0], dtype=torch.float32)
    b = torch.tensor([-0.0], dtype=torch.float32)
    assert replay_mod.describe(a, name="x")["sha256"] != \
        replay_mod.describe(b, name="x")["sha256"]


# --------------------------------------------------------------------------
# Reporting: a difference must show, and an absence must not read as a verdict.
# --------------------------------------------------------------------------

def test_a_kernel_that_returns_the_same_bytes_reports_one_digest():
    model, ext, _, _ = _make(_deterministic, repeats=4, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["distinct_replay_digests"] == 1
    for entry in record["replays"][1:]:
        assert entry["vs_first_replay"]["bitwise_identical"] is True


def test_an_all_identical_result_is_not_reported_as_a_determinism_verdict():
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    text = replay_mod.uninstall(model, ext)["interpretation"]
    assert "not a determinism verdict" in text
    assert "names no cause" in text
    assert "bounds nothing else" in text
    assert "device addresses" in text
    assert "inputs_stable" in text


def test_a_clean_run_supports_the_same_input_reading():
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["same_input_inference_supported"] is True


def test_a_mutated_argument_withdraws_the_same_input_reading():
    """Differing digests under changing arguments say nothing about the
    kernel, so the record must not let that read as 'same inputs'."""
    def mutating(*args):
        args[6].add_(1)

    model, ext, _, _ = _make(mutating, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["inputs_stable"] is False
    assert record["same_input_inference_supported"] is False


def test_a_non_finite_result_withdraws_the_same_input_reading():
    """A magnitude beside a NaN is not a magnitude."""
    def nan_out(*args):
        args[3].fill_(float("nan"))

    model, ext, _, _ = _make(nan_out, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["inputs_stable"] is True
    assert record["same_input_inference_supported"] is False


def test_a_kernel_that_differs_is_reported_as_differing():
    calls = {"n": 0}

    def drifting(*args):
        calls["n"] += 1
        args[3].add_(float(calls["n"]))

    model, ext, _, _ = _make(drifting, repeats=3, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["distinct_replay_digests"] == 3
    later = record["replays"][1]["vs_first_replay"]
    assert later["bitwise_identical"] is False
    assert later["fp32_differing_elements"] == ROWS * N
    assert later["fp32_max_abs_delta"] > 0.0


def test_a_difference_too_small_to_survive_bf16_is_reported_as_both():
    """The number that reaches the next layer and the number the kernel
    produced are different numbers, and reporting one would hide which."""
    calls = {"n": 0}

    def tiny_drift(*args):
        calls["n"] += 1
        args[3].fill_(1.0)
        # Call 1 is the real one and call 2 is the first replay; the drift is
        # applied from the second replay on, so the two replays differ.
        if calls["n"] >= 3:
            args[3][0, 0] += 2.0 ** -20     # far below one bf16 step at 1.0

    model, ext, _, _ = _make(tiny_drift, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    second = replay_mod.uninstall(model, ext)["replays"][1]["vs_first_replay"]
    assert second["fp32_differing_elements"] == 1
    assert second["identical_fp32"] is False
    assert second["bf16_differing_elements"] == 0
    assert second["identical_bf16"] is True


# --------------------------------------------------------------------------
# The BF16 ULP metric, including across zero.
# --------------------------------------------------------------------------

def _next_bf16_up(value):
    """The next representable bfloat16 above ``value``, by bit pattern.

    Derived rather than written as a constant: bfloat16 keeps 7 mantissa bits,
    so the step at 1.0 is 2 ** -7, and a hand-written 2 ** -8 is half a step
    that rounds back to where it started.
    """
    bits = torch.tensor([value], dtype=torch.bfloat16).view(torch.int16)
    return (bits + 1).view(torch.bfloat16).float().item()


def _compare_scalars(a, b):
    return replay_mod.compare(torch.tensor([a], dtype=torch.float32),
                              torch.tensor([b], dtype=torch.float32))


def test_a_one_ulp_bf16_difference_is_counted_as_one_ulp():
    step = _next_bf16_up(1.0) - 1.0
    assert step == 2.0 ** -7, "bfloat16 keeps 7 mantissa bits"
    result = _compare_scalars(1.0, 1.0 + step)
    assert result["bf16_differing_elements"] == 1
    assert result["bf16_max_ulp_delta"] == 1
    assert result["bf16_max_abs_delta"] == pytest.approx(step)


def test_a_two_ulp_difference_is_counted_as_two():
    """A ULP count that always read 1 would pass the test above."""
    step = _next_bf16_up(1.0) - 1.0
    assert _compare_scalars(1.0, 1.0 + 2 * step)["bf16_max_ulp_delta"] == 2


def test_adjacent_values_either_side_of_zero_are_two_ulp_apart():
    """Reading the bit pattern as a signed int16 makes these 65534 apart: the
    pattern grows as a negative value falls, so the raw difference is nonsense
    across the sign boundary."""
    smallest = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16).float().item()
    result = _compare_scalars(-smallest, smallest)
    assert result["bf16_max_ulp_delta"] == 2


def test_one_step_below_zero_is_one_ulp_not_thirty_two_thousand():
    smallest = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16).float().item()
    assert _compare_scalars(0.0, -smallest)["bf16_max_ulp_delta"] == 1


def test_negative_values_one_step_apart_are_one_ulp_apart():
    """Signed-int16 ordering is reversed for negatives, so this is the case a
    raw subtraction gets wrong in the other direction."""
    step = _next_bf16_up(1.0) - 1.0
    assert _compare_scalars(-1.0, -(1.0 + step))["bf16_max_ulp_delta"] == 1


def test_signed_zeros_are_zero_ulp_apart_and_still_not_bit_identical():
    result = _compare_scalars(0.0, -0.0)
    assert result["bf16_max_ulp_delta"] == 0
    assert result["identical_fp32"] is True          # numerically equal
    assert result["bitwise_identical"] is False      # different stored bits


# --------------------------------------------------------------------------
# Finiteness, and the served output.
# --------------------------------------------------------------------------

def test_a_max_delta_over_a_tensor_holding_a_nan_says_so():
    """A max over NaN is not a magnitude and must not read as one."""
    reference = torch.zeros(4, dtype=torch.float32)
    candidate = torch.zeros(4, dtype=torch.float32)
    candidate[0] = float("nan")
    result = replay_mod.compare(reference, candidate)
    assert result["candidate_finite"]["all_finite"] is False
    assert result["candidate_finite"]["finite_elements"] == 3
    assert result["fp32_max_abs_delta_over_finite_only"] is True


def test_a_finite_fp32_that_overflows_the_bf16_cast_is_reported():
    """BF16 has FP32's exponent range but 7 mantissa bits, so a finite FP32
    just under the top of the range rounds UP to infinity here.  FP32
    finiteness does not answer the BF16 question."""
    big = torch.tensor([3.4e38], dtype=torch.float32)
    assert torch.isfinite(big).all()
    assert not torch.isfinite(big.to(torch.bfloat16)).all(), "no overflow to see"
    result = replay_mod.compare(torch.zeros(1, dtype=torch.float32), big)
    assert result["candidate_finite"]["all_finite"] is True
    assert result["candidate_bf16_finite"]["all_finite"] is False
    assert result["bf16_max_over_finite_only"] is True


def test_a_bf16_ulp_distance_is_not_taken_over_an_infinity():
    """A ULP distance counts representable steps, which infinities do not
    have.  Only the pairs where both sides are finite are counted."""
    step = _next_bf16_up(1.0) - 1.0
    reference = torch.tensor([1.0, 3.4e38], dtype=torch.float32)
    candidate = torch.tensor([1.0 + step, 0.0], dtype=torch.float32)
    result = replay_mod.compare(reference, candidate)
    assert result["bf16_max_ulp_delta"] == 1
    assert result["bf16_max_over_finite_only"] is True


def test_finite_comparisons_are_not_marked_finite_only():
    result = _compare_scalars(1.0, 2.0)
    assert result["bf16_max_over_finite_only"] is False
    assert result["reference_bf16_finite"]["all_finite"] is True


def test_two_bit_identical_nans_are_reported_identical():
    """torch.equal calls these different, which is why the check is a hash."""
    nan = torch.full((4,), float("nan"), dtype=torch.float32)
    assert replay_mod.compare(nan, nan.clone())["bitwise_identical"] is True
    assert torch.equal(nan, nan.clone()) is False


def test_the_served_output_is_confirmed_unchanged_by_hash():
    model, ext, _, args = _make(_deterministic, repeats=2, required_rows=ROWS)
    served_before = args[3].clone()
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["original_out_unchanged_after_replays"] is True
    assert (record["original_out_before"]["fp32_sha256"]
            == record["original_out_after"]["fp32_sha256"])
    assert record["original_out_before"]["bf16_sha256"]
    assert torch.equal(args[3], served_before + 1.0)


def test_a_probe_that_wrote_through_to_the_served_output_is_caught():
    """The check has to bite, so the driver is changed rather than the record:
    this kernel keeps a reference to the first output it was given and writes
    to it on every later call, which is what a write-through would look like."""
    seen = {}

    def leaky(*args):
        out = args[3]
        if "first" not in seen:
            seen["first"] = out
        else:
            seen["first"].add_(100.0)
        out.add_(1.0)

    model, ext, _, _ = _make(leaky, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["original_out_unchanged_after_replays"] is False
    assert (record["original_out_before"]["fp32_sha256"]
            != record["original_out_after"]["fp32_sha256"])


def test_a_served_output_changed_only_in_sign_of_zero_is_caught():
    """torch.equal would call this unchanged; the stored bits differ."""
    seen = {}

    def sign_flip(*args):
        out = args[3]
        if "first" not in seen:
            seen["first"] = out
            out.fill_(0.0)
        else:
            seen["first"].fill_(-0.0)

    model, ext, _, _ = _make(sign_flip, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    assert record["original_out_unchanged_after_replays"] is False


# --------------------------------------------------------------------------
# Install and uninstall.
# --------------------------------------------------------------------------

def test_uninstall_restores_the_entry_point():
    model, ext, _, _ = _make(_deterministic, required_rows=ROWS)
    assert ext.exl3_fat_moe_down is not _deterministic
    replay_mod.uninstall(model, ext)
    assert ext.exl3_fat_moe_down is _deterministic
    assert not hasattr(model, "_tr3_grouped_down_replay")


def test_a_second_install_is_refused_rather_than_wrapping_twice():
    model, ext, state, _ = _make(_deterministic, required_rows=ROWS)
    with pytest.raises(ValueError, match="already has"):
        replay_mod.install(model, ext, state, module_name="mlp")
    replay_mod.uninstall(model, ext)


def test_an_unknown_boundary_is_refused_at_install():
    ext = _Ext(_deterministic)
    model = _Model(_Boundary(ext, _args()))
    with pytest.raises(ValueError, match="not a module"):
        replay_mod.install(model, ext, _Capture(), module_name="layers.99.mlp")


def test_a_call_with_the_wrong_arity_is_left_alone():
    """The wrap is on a C entry point; a different arity is a different call."""
    model, ext, _, args = _make(_deterministic, required_rows=ROWS)
    ext.exl3_fat_moe_down(*args[:9])
    assert replay_mod.uninstall(model, ext) is None


def test_fewer_than_two_repeats_compares_nothing():
    with pytest.raises(ValueError, match="compares nothing"):
        replay_mod.GroupedDownReplay(_Capture(), repeats=1)


def test_the_record_serializes_and_refuses_a_non_finite_value():
    model, ext, _, _ = _make(_deterministic, repeats=2, required_rows=ROWS)
    model(torch.zeros(1))
    record = replay_mod.uninstall(model, ext)
    text = replay_mod.result_json(record)
    assert replay_mod.REPLAY_SCHEMA in text
    record["replays"][0]["vs_original"]["fp32_max_abs_delta"] = float("nan")
    with pytest.raises(ValueError):
        replay_mod.result_json(record)
