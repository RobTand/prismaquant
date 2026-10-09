"""Per-sequence/per-block signed attribution sidecar for joint AURA (#1962).

The authoritative whole-draw fields stay bitwise authoritative: the sidecar is
a separate descriptive arithmetic scope whose reconstruction residual is
reported, gated and never forced onto the price. These tests cover the
publisher/validator contract, the dense lease collector, the streamed
production collector and the checkpoint resume boundary. No model-quality,
estimator or serving claim is made; probe-sampling scope is unchanged.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant import joint_aura as joint

from test_joint_aura_assignment_diagnostics import UNIT_A, _row
from test_joint_aura_projection import _linear, _spec
from test_joint_aura_streamed import _fixture as _streamed_fixture
from test_joint_aura_streamed import _model_identity


LEAVEOUT_SCOPE = "descriptive_conditional_fixed_probes"


def _blocks(*specs):
    return [{"block_index": index, "first_sequence": first, "sequences": count}
            for index, (first, count) in enumerate(specs)]


def _comps(weight, activation, mixed):
    total = weight + activation + mixed
    return {"weight": weight, "activation": activation, "mixed": mixed,
            "total": total}


def _sidecar(blocks, components_per_probe, totals, *, gate_relative=1e-3,
             arithmetic_scope="test_per_invocation_contractions",
             sequence_length=4, selected_tokens_per_row=4, n_sequences=None,
             calibration_sha256="c" * 64, with_leaveout=None):
    return joint.sequence_attribution_sidecar(
        blocks=blocks, components_per_probe=components_per_probe,
        authoritative_totals=totals, gate_relative=gate_relative,
        arithmetic_scope=arithmetic_scope, sequence_length=sequence_length,
        selected_tokens_per_row=selected_tokens_per_row,
        n_sequences=n_sequences, calibration_sha256=calibration_sha256,
        with_leaveout=with_leaveout)


def _row_with_sidecar(sidecar, *, signed=(3.0, -1.0), authority_components=None):
    row = _row(UNIT_A, list(signed))
    if authority_components is None:
        authority_components = []
        for total, blocks in zip(signed, sidecar["components_per_probe"]):
            parts = {key: math.fsum(block[key] for block in blocks)
                     for key in ("weight", "activation", "mixed")}
            parts["weight"] += total - math.fsum(parts.values())
            authority_components.append({**parts, "total": total})
    row = joint.make_joint_aura_entry(
        operator_identity=row["joint_operator_identity"],
        probe_identity=row["probe_identity"],
        signed_components=copy.deepcopy(authority_components),
        sequence_attribution=sidecar,
    )
    return row


# ---- uncertainty_scope causal checks ---------------------------------------

def test_row_without_uncertainty_scope_stays_legacy_conditional():
    row = _row(UNIT_A, [2.0, -3.0])
    del row["uncertainty_scope"]
    assert joint.validate_joint_aura_entry(row)


@pytest.mark.parametrize("scope", [
    "calibration_draw_level", "sequence_sampling_generalization",
    "population", "draw_level_stderr", "probe_sampling_conditional_on_fixed_calibration ",
])
def test_foreign_uncertainty_scope_is_refused(scope):
    row = _row(UNIT_A, [2.0, -3.0])
    row["uncertainty_scope"] = scope
    with pytest.raises(ValueError, match="uncertainty_scope"):
        joint.validate_joint_aura_entry(row)


# ---- publisher oracle ------------------------------------------------------

def test_sidecar_exact_reconstruction_concentrates_and_reconciles():
    # Two probes, two single-sequence blocks. Reconstruction sums exactly to
    # the authoritative totals, so the residual is exactly zero and the
    # attribution identity closes on the published price.
    blocks = _blocks((0, 1), (1, 1))
    components = [
        [_comps(2.0, 0.5, 0.0), _comps(-1.0, 0.25, 0.0)],
        [_comps(1.0, -0.5, 0.0), _comps(0.5, 0.25, 0.0)],
    ]
    totals = [components[0][0]["total"] + components[0][1]["total"],
              components[1][0]["total"] + components[1][1]["total"]]
    sidecar = _sidecar(blocks, components, totals)
    row = _row_with_sidecar(sidecar, signed=tuple(totals))
    assert joint.validate_joint_aura_entry(row)
    published = row["sequence_attribution"]
    assert published["scope"] == "per_sequence"
    assert published["reconciliation"]["residual_per_probe"] == [0.0, 0.0]
    price = row["predicted_dloss"]
    total_attribution = math.fsum(published["attribution_per_block"])
    assert total_attribution == pytest.approx(price, rel=1e-12)
    # Same authoritative totals, opposite concentration: the attribution must
    # move the price onto the carrying block without moving the price itself.
    components_b = [
        [_comps(-1.0, 0.25, 0.0), _comps(2.0, 0.5, 0.0)],
        [_comps(0.5, 0.25, 0.0), _comps(1.0, -0.5, 0.0)],
    ]
    sidecar_b = _sidecar(blocks, components_b, totals)
    row_b = _row_with_sidecar(sidecar_b, signed=tuple(totals))
    assert row_b["predicted_dloss"] == price
    assert (row_b["sequence_attribution"]["attribution_per_block"]
            != published["attribution_per_block"])


def test_sidecar_residual_stays_separate_from_the_price():
    blocks = _blocks((0, 1), (1, 1))
    components = [
        [_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)],
        [_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)],
    ]
    totals = [3.0, 3.0]  # authority differs from the reconstructed 2.0
    sidecar = _sidecar(blocks, components, totals, gate_relative=0.5)
    row = _row_with_sidecar(sidecar, signed=tuple(totals))
    assert joint.validate_joint_aura_entry(row)
    published = row["sequence_attribution"]
    # residual = reconstruction minus authoritative total
    assert published["reconciliation"]["residual_per_probe"] == [-1.0, -1.0]
    assert published["reconciliation"]["norm_per_probe"] == [2.0, 2.0]
    # Attribution is computed against the authoritative totals, and the row
    # never claims the attribution sum equals the price when residual != 0.
    expected_c0 = 0.5 * ((1.0 * 3.0) + (1.0 * 3.0)) / 2
    assert published["attribution_per_block"][0] == pytest.approx(expected_c0)
    assert "equals_predicted_dloss" not in published
    assert "attribution_price" not in published


def test_sidecar_leaveout_prices_and_jackknife_standard_error():
    blocks = _blocks((0, 1), (1, 1), (2, 1))
    components = [
        [_comps(1.0, 0.0, 0.0), _comps(2.0, 0.0, 0.0), _comps(3.0, 0.0, 0.0)],
        [_comps(1.0, 0.0, 0.0), _comps(2.0, 0.0, 0.0), _comps(3.0, 0.0, 0.0)],
    ]
    totals = [6.0, 6.0]
    sidecar = _sidecar(blocks, components, totals)
    row = _row_with_sidecar(sidecar, signed=tuple(totals))
    published = row["sequence_attribution"]["leaveout"]
    assert published["uncertainty_scope"] == LEAVEOUT_SCOPE
    raw = [0.5 * (5.0 ** 2), 0.5 * (4.0 ** 2), 0.5 * (3.0 ** 2)]
    expected_prices = [1.5 * value for value in raw]  # N/(N-1) at k=1
    assert published["price_per_block"] == pytest.approx(expected_prices)
    assert published["delete_one_scale"] == pytest.approx(1.5)
    mean = sum(expected_prices) / 3
    expected_se = math.sqrt((2 / 3) * sum(
        (value - mean) ** 2 for value in expected_prices))
    assert published["jackknife_standard_error"] == pytest.approx(expected_se)
    assert published["assumption"].startswith("exchangeable")


def test_sidecar_equal_whole_block_leaveout_is_descriptive():
    blocks = _blocks((0, 2), (2, 2))
    components = [
        [_comps(1.0, 0.0, 0.0), _comps(3.0, 0.0, 0.0)],
        [_comps(1.0, 0.0, 0.0), _comps(3.0, 0.0, 0.0)],
    ]
    totals = [4.0, 4.0]
    sidecar = _sidecar(blocks, components, totals, n_sequences=4)
    row = _row_with_sidecar(sidecar, signed=tuple(totals))
    assert row["sequence_attribution"]["scope"] == "capture_batch_block"
    leaveout = row["sequence_attribution"]["leaveout"]
    assert leaveout["sequences_per_block"] == 2
    assert leaveout["delete_one_scale"] == pytest.approx(2.0)  # 4/(4-2)
    assert leaveout["price_per_block"] == pytest.approx([9.0, 1.0])
    assert leaveout["uncertainty_scope"] == LEAVEOUT_SCOPE


def test_sidecar_identical_blocks_allow_recomputed_zero_standard_error():
    blocks = _blocks((0, 1), (1, 1))
    components = [
        [_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)],
        [_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)],
    ]
    totals = [2.0, 2.0]
    sidecar = _sidecar(blocks, components, totals)
    row = _row_with_sidecar(sidecar, signed=tuple(totals))
    leaveout = row["sequence_attribution"]["leaveout"]
    assert leaveout["jackknife_standard_error"] == 0.0


def test_sidecar_single_coarse_block_publishes_no_leaveout():
    blocks = _blocks((0, 2))
    components = [[_comps(2.0, 0.0, 0.0)], [_comps(2.0, 0.0, 0.0)]]
    sidecar = _sidecar(blocks, components, [2.0, 2.0], n_sequences=2)
    row = _row_with_sidecar(sidecar, signed=(2.0, 2.0))
    assert row["sequence_attribution"]["scope"] == "capture_batch_block"
    assert "leaveout" not in row["sequence_attribution"]


def test_sidecar_cohort_summary_binds_whole_blocks_and_selected_tokens():
    blocks = _blocks((0, 1), (1, 1), (2, 1), (3, 1))
    components = [
        [_comps(float(i + 1), 0.0, 0.0) for i in range(4)],
        [_comps(float(i + 1), 0.0, 0.0) for i in range(4)],
    ]
    totals = [10.0, 10.0]
    sidecar = _sidecar(blocks, components, totals, sequence_length=4,
                       selected_tokens_per_row=4)
    cohort = joint.sequence_cohort_summary(
        sidecar, first_block_index=0, block_count=1)
    assert cohort["restricted_price"] == pytest.approx(0.5 * 1.0 ** 2)
    assert cohort["renormalized_scale"] == pytest.approx(4.0)
    assert cohort["renormalized_full_draw_estimate"] == pytest.approx(2.0)
    assert cohort["selected_tokens_per_row"] == 4
    with pytest.raises(ValueError):
        joint.sequence_cohort_summary(sidecar, first_block_index=3,
                                      block_count=2)


# ---- validator refusals ----------------------------------------------------

def _valid_components():
    return [
        [_comps(2.0, 0.5, 0.0), _comps(-1.0, 0.25, 0.0)],
        [_comps(1.0, -0.5, 0.0), _comps(0.5, 0.25, 0.0)],
    ]


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda s: s.update(schema="prismaquant.joint_aura.sequence_attribution.v2"),
                 id="schema"),
    pytest.param(lambda s: s["block_geometry"]["blocks"].pop(), id="missing-block"),
    pytest.param(lambda s: s["block_geometry"]["blocks"].__setitem__(
        1, {"block_index": 1, "first_sequence": 0, "sequences": 1}), id="overlapping"),
    pytest.param(lambda s: s["block_geometry"]["blocks"].__setitem__(
        0, {"block_index": 0, "first_sequence": 5, "sequences": 1}), id="unordered"),
    pytest.param(lambda s: s["block_geometry"]["blocks"][0].update(sequences=0),
                 id="empty-block"),
    pytest.param(lambda s: s["components_per_probe"][0].pop(), id="probe-mismatch"),
    pytest.param(lambda s: s["components_per_probe"][0][0].update(total=99.0),
                 id="component-sum"),
    pytest.param(lambda s: s.update(scope="per_sequence") or s["block_geometry"]["blocks"][0].update(sequences=2),
                 id="unproven-per-sequence"),
    pytest.param(lambda s: s["reconciliation"].update(residual_per_probe=[9.0, 0.0]),
                 id="wrong-residual"),
    pytest.param(lambda s: s["reconciliation"].update(norm_per_probe=[0.0, 0.0]),
                 id="wrong-norm"),
    pytest.param(lambda s: s["attribution_per_block"].__setitem__(0, [7.0, 7.0]),
                 id="wrong-attribution"),
    pytest.param(lambda s: s["block_geometry"].update(calibration_sha256="f" * 64),
                 id="cohort-sha"),
    pytest.param(lambda s: s.update(
        leaveout={"price_per_block": [0.0], "jackknife_standard_error": 0.0,
                  "uncertainty_scope": LEAVEOUT_SCOPE}), id="forged-leaveout"),
    pytest.param(lambda s: s["reconciliation"].update(gate_method="hidden"),
                 id="hidden-method"),
])
def test_malformed_sidecar_is_refused(mutation):
    sidecar = _sidecar(_blocks((0, 1), (1, 1)), _valid_components(),
                       [1.75, 1.25])
    mutation(sidecar)
    with pytest.raises(ValueError):
        joint.make_joint_aura_entry(
            operator_identity=_row(UNIT_A, [1.75, 1.25])["joint_operator_identity"],
            probe_identity=_row(UNIT_A, [1.75, 1.25])["probe_identity"],
            signed_components=[{"weight": 1.75, "activation": 0.0, "mixed": 0.0,
                                "total": 1.75},
                               {"weight": 1.25, "activation": 0.0, "mixed": 0.0,
                                "total": 1.25}],
            sequence_attribution=sidecar,
        )


def test_gate_beyond_reported_relative_bound_is_refused():
    blocks = _blocks((0, 1), (1, 1))
    components = [[_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)],
                  [_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)]]
    # Residual 1.0 against norm 2.0 exceeds the default 1e-3 relative gate.
    with pytest.raises(ValueError, match="gate"):
        _sidecar(blocks, components, [3.0, 1.0])


# ---- dense lease collector -------------------------------------------------

def test_dense_lease_block_sidecar_matches_per_invocation_oracle(monkeypatch):
    # This oracle explicitly clamps, so pin the matching emulation policy
    # rather than inheriting a campaign or worker process setting.
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "1")
    torch.manual_seed(23)
    weight = torch.randn(4, 8)
    delta = torch.randn_like(weight) * 0.1
    spec = _spec("f8", lambda x: torch.round(x * 2) / 2)
    module = _linear(weight)
    maximum = {"u": 1.0}
    invocations = []
    for _ in range(2):
        x = torch.randn(1, 3, 8)
        gradient = torch.randn(1, 3, 4)
        invocations.append((x, gradient))

    def oracle(x, gradient):
        x2 = x.reshape(-1, 8).float()
        g2 = gradient.reshape(-1, 4).float()
        # The shared emulation hook clips to the calibrated max_abs
        # before the spec's own dynamic quantizer.
        quantized = spec.activation_quantize_dequantize(x.clamp(-1.0, 1.0))
        dx = quantized.reshape_as(x2).float() - x2
        gw = g2.T @ x2
        ga = g2.T @ dx
        return {"weight": float((gw * delta).sum()),
                "activation": float((ga * weight.float()).sum()),
                "mixed": float((ga * delta).sum())}

    def run(attribution):
        with joint.SignedJointProjectionLease(
                {"u": module}, {"u": {"f8": spec}}, {("u", "f8"): delta},
                activation_max_abs=maximum, attribution=attribution) as lease:
            lease.begin_probe()
            per_block = []
            for index, (x, gradient) in enumerate(invocations):
                if attribution:
                    lease.note_block(index, first_sequence=index, sequences=1)
                module(x).backward(gradient)
                module.zero_grad(set_to_none=True)
                per_block.append(oracle(x, gradient))
            result = lease.finish_probe()
            sidecar = lease.finish_attribution() if attribution else None
        return result, sidecar, per_block

    off, _, oracle_blocks = run(False)
    on, sidecar, _ = run(True)
    # Authoritative totals are bitwise unchanged by the instrument flag.
    assert on == off
    assert sidecar["blocks"] == _blocks((0, 1), (1, 1))
    for index, expected in enumerate(oracle_blocks):
        published = sidecar["components"][("u", "f8")][index]
        for field, value in expected.items():
            assert published[field] == pytest.approx(value, rel=1e-5, abs=1e-7)


# ---- streamed production collector ----------------------------------------

_TWO_SEQ = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
_ATTR = {"candidates": "all"}


def test_streamed_production_rows_carry_sidecar_and_authoritative_bitwise():
    _model, _context, runner, cache = _streamed_fixture()
    off = aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"))
    _model, _context, runner, cache = _streamed_fixture()
    on = aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"),
        sequence_attribution=dict(_ATTR))
    for name, rows in on["costs"].items():
        for fmt, row in rows.items():
            assert joint.validate_joint_aura_entry(row)
            if fmt in aura._ZERO_COST_FORMATS:
                # A passthrough row's price is exact by rule; it publishes no
                # sidecar and claims none.
                assert "sequence_attribution" not in row
            else:
                sidecar = row["sequence_attribution"]
                assert sidecar["scope"] == "capture_batch_block"
                assert sidecar["block_geometry"]["n_sequences"] == 2
                assert len(sidecar["components_per_probe"]) == 3
            off_row = off["costs"][name][fmt]
            for field in ("signed_per_probe", "x2_per_probe",
                          "predicted_dloss", "predicted_dloss_stderr",
                          "signed_components_per_probe"):
                assert row[field] == off_row[field], (name, fmt, field)
            assert "sequence_attribution" not in off_row


def test_streamed_single_sequence_blocks_publish_per_sequence_scope():
    _model, _context, runner, cache = _streamed_fixture()
    payload = aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        probe_microbatch=1, min_free_gib=0, production_cache=cache,
        joint_activation=True, model_identity=_model_identity("joint-source"),
        sequence_attribution=dict(_ATTR))
    for name, rows in payload["costs"].items():
        for fmt, row in rows.items():
            assert joint.validate_joint_aura_entry(row)
            if fmt in aura._ZERO_COST_FORMATS:
                assert "sequence_attribution" not in row
                continue
            sidecar = row["sequence_attribution"]
            assert sidecar["scope"] == "per_sequence"
            assert sidecar["block_geometry"]["n_sequences"] == 2
            assert all(block["sequences"] == 1
                       for block in sidecar["block_geometry"]["blocks"])
            assert sidecar["leaveout"]["jackknife_standard_error"] >= 0.0


def test_streamed_attribution_config_must_be_well_formed():
    _model, _context, runner, cache = _streamed_fixture()
    with pytest.raises((ValueError, TypeError, KeyError)):
        aura.compute_aura_cost_streamed(
            runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
            min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-source"),
            sequence_attribution={"candidates": ["model.layers.0.missing@FP8_DYNAMIC"]})


# ---- checkpoint resume boundary -------------------------------------------

def test_checkpoint_resume_across_attribution_boundary_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _context, runner, cache = _streamed_fixture()
    aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"),
        sequence_attribution=dict(_ATTR), checkpoint_dir=tmp_path)
    _, context, runner, cache = _streamed_fixture()
    with pytest.raises((RuntimeError, ValueError), match="sequence_attribution"):
        aura.compute_aura_cost_streamed(
            runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
            min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-source"),
            checkpoint_dir=tmp_path, resume=True)
    assert context.install_calls == 0
    _, _context, runner, cache = _streamed_fixture()
    again = aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"),
        sequence_attribution=dict(_ATTR), checkpoint_dir=tmp_path, resume=True)
    for rows in again["costs"].values():
        for row in rows.values():
            assert joint.validate_joint_aura_entry(row)


def test_checkpoint_resume_attribution_on_plain_rows_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _context, runner, cache = _streamed_fixture()
    aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"), checkpoint_dir=tmp_path)
    _, _context, runner, cache = _streamed_fixture()
    with pytest.raises((RuntimeError, ValueError), match="sequence_attribution"):
        aura.compute_aura_cost_streamed(
            runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
            min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-source"),
            sequence_attribution=dict(_ATTR), checkpoint_dir=tmp_path,
            resume=True)

def test_projection_redistribution_cannot_hide_behind_matching_total():
    components = [[_comps(0.5, 0.5, 0.0), _comps(0.5, 0.5, 0.0)]] * 2
    sidecar = _sidecar(_blocks((0, 1), (1, 1)), components, [2.0, 2.0])
    with pytest.raises(ValueError, match="component"):
        _row_with_sidecar(sidecar, signed=(2.0, 2.0),
                          authority_components=[_comps(2.0, 0.0, 0.0)] * 2)


def test_out_of_draw_blocks_cannot_forge_complete_leaveout():
    components = [[_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)]] * 2
    with pytest.raises(ValueError, match="block|draw"):
        _sidecar(_blocks((0, 1), (3, 1)), components, [2.0, 2.0],
                 n_sequences=2)


def test_float_block_index_is_not_a_sequence_coordinate():
    blocks = _blocks((0, 1), (1, 1))
    blocks[0]["block_index"] = 0.0
    components = [[_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)]] * 2
    with pytest.raises(ValueError, match="block"):
        _sidecar(blocks, components, [2.0, 2.0])


def test_sidecar_geometry_must_match_priced_calibration_shape():
    components = [[_comps(1.0, 0.0, 0.0), _comps(1.0, 0.0, 0.0)]] * 2
    sidecar = _sidecar(_blocks((0, 1), (1, 1)), components, [2.0, 2.0])
    row = _row(UNIT_A, [2.0, 2.0])
    probe = {**row["probe_identity"], "calibration_shape": [3, 4],
             "token_scope": "all"}
    operator = {**row["joint_operator_identity"],
                "probe_identity_sha256": joint.identity_sha256(probe)}
    with pytest.raises(ValueError, match="calibration|geometry"):
        joint.make_joint_aura_entry(
            operator_identity=operator, probe_identity=probe,
            signed_components=row["signed_components_per_probe"],
            sequence_attribution=sidecar)

def test_spill_attribution_releases_candidate_lease_after_callback(monkeypatch):
    import gc
    import weakref
    from types import SimpleNamespace
    from prismaquant.joint_cost_quantum import SpillSequenceAttribution

    source = _linear(torch.ones(2, 2))
    spec = _spec("candidate", lambda x: x)
    references = []
    original = joint.JointBlockAttributionLease.__init__

    def capture_lease(self, *args, **kwargs):
        original(self, *args, **kwargs)
        references.append(weakref.ref(self))

    monkeypatch.setattr(joint.JointBlockAttributionLease, "__init__", capture_lease)

    def replay_records(window_index, probe_index, feed, charge):
        for block in range(2):
            feed("u", torch.ones(1, 2), torch.ones(1, 2), block)

    collector = SpillSequenceAttribution(
        config=joint.normalize_sequence_attribution({"candidates": "all"}),
        linears={"u": source}, formats_by_qname={"u": {"candidate": spec}},
        activation_maxima={}, projection_backend=None,
        spill=SimpleNamespace(replay_records=replay_records), guard=None,
        n_probes=2, n_samples=2, seqlen=1, probe_microbatch=1,
        capture_batch=1, token_scope="all", calibration_sha256="c" * 64)
    delta = torch.ones(2, 2)
    collector(key=("u", "candidate"), delta=delta, window_index=0, probe_index=0)
    gc.collect()
    assert references and references[-1]() is None, "resident candidate lease escaped callback"


def test_streamed_candidate_selector_does_not_instrument_other_rows():
    _model, _context, runner, cache = _streamed_fixture()
    target = next(name for name, fmt in cache.weights if fmt == "FP8_E4M3")
    payload = aura.compute_aura_cost_streamed(
        runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"),
        sequence_attribution={"candidates": [[target, "FP8_E4M3"]]})
    observed = [(name, fmt) for name, rows in payload["costs"].items()
                for fmt, row in rows.items() if "sequence_attribution" in row]
    assert observed == [(target, "FP8_E4M3")], "candidate selector widened the measured scope"


def test_streamed_candidate_selector_refuses_unknown_candidate():
    _model, context, runner, cache = _streamed_fixture()
    with pytest.raises(ValueError, match="candidate"):
        aura.compute_aura_cost_streamed(
            runner, _TWO_SEQ, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
            min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-source"),
            sequence_attribution={"candidates": [["absent", "FP8_E4M3"]]})
    assert context.install_calls == 0
