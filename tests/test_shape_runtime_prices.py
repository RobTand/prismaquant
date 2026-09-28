"""PACT shape-time price table (PQ #1583): parse, admit, derive, price.

The contract fixture is the INSTALLED pinned contract, read as published.
Since the pin moved to contract v42 (PQ #1274) it carries Tessera #685's fused
routed lanes itself: the routed E4M3 and BF16 cells name the fused launch
beside the compact one, and each lane publishes its ``requires`` predicate on
its own ``native_extensions`` row. The fixtures assert that instead of
grafting it. The lane decision itself is Tessera's
(``decide_lane_requirements``) reached through
``lane_eligibility.cell_lane_admits``; nothing here restates its rule.
"""
import copy
import hashlib
import json
import random
import statistics
from importlib.resources import as_file

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import shape_runtime_prices as srp
from prismaquant import tessera_runtime_contract as contract
from prismaquant.allocator_solver import Candidate
from prismaquant.measured_runtime_prices import RuntimePriceError, bootstrap_sum

SHA = "c" * 64
COMMIT = "e" * 40
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
E4M3, BF16, E2M1 = "TESSERA_E4M3_K1", "TESSERA_BF16_K1", "TESSERA_E2M1_K2"
COMPACT = {"symbol": "tessera.native_window_moe.NativeWindowMoE.__call__",
           "decoder": "native_window_moe_compact"}
FUSED = {"symbol": "tessera.routed_fused.FusedRoutedWindowMoE.__call__",
         "decoder": "native_routed_fused_window"}
DENSE_E4M3 = {"symbol": "tessera::window_gemm_dense", "decoder": "native_window_gemm"}
DENSE_BF16 = {"symbol": "tessera::window_gemm_dense", "decoder": "native_window_gemm_folded"}
SPAN2 = {"symbol": "tessera.kernel_a4.a4_span2_gemm", "decoder": "native_span2_gemm"}
SPAN2_GROUPED = {"symbol": "tessera.kernel_a4.a4_span2_grouped_gemm",
                 "decoder": "native_span2_grouped"}
ROUTED = "E288:w13=2048x4096:w2=4096x1024"
FUSED_LANE = {"decoder": "native_routed_fused_window",
              "requires": {"column_rates": [4], "window_bits": [14], "body": "window",
                           "plane": "channel", "release_overrides": False, "diagonals": False,
                           "rotation": ["none"], "grid_arities": [1]}}


FOLDED_COMPACT = dict(COMPACT, decoder="native_window_moe_compact_folded")
FOLDED_FUSED = dict(FUSED, decoder="native_routed_fused_window_folded")


def _payload():
    with as_file(contract.contract_path()) as path:
        payload = json.loads(path.read_bytes())
    # The pinned v42 contract publishes both fused lanes and names each
    # beside the compact launch in the routed cells; the fixtures below
    # depend on exactly that, so a re-pin that moves it fails here.
    lanes = {row["module_name_prefix"]: row.get("lane")
             for row in payload["native_extensions"]}
    assert lanes["tessera_routed_fused_e4m3"] == FUSED_LANE
    assert lanes["tessera_routed_fused_value"] == dict(
        FUSED_LANE, decoder="native_routed_fused_window_folded")
    want = {E4M3: [COMPACT, FUSED], BF16: [FOLDED_COMPACT, FOLDED_FUSED]}
    routed = [cell for cell in payload["lane_eligibility"]["cells"]
              if cell["structure"] == "routed_moe" and cell["family"] in want]
    assert routed and all(
        [dict(pair) for pair in cell["executes"]] == want[cell["family"]]
        for cell in routed), routed
    return payload


@pytest.fixture(scope="module")
def payload():
    return _payload()


@pytest.fixture(scope="module")
def eligibility(payload):
    return lane._parse_table(payload["lane_eligibility"], payload["formats"], "", COMMIT, SHA,
                             native_extensions=payload["native_extensions"])


@pytest.fixture(scope="module")
def formats(payload):
    return {row["family"]: row for row in payload["formats"]}


SCOPE = srp.ShapeTableScope(contract_sha256=SHA, tessera_commit=COMMIT, runtime_image_digest=IMAGE,
                            tensor_parallel=2, platform="sm_121", residency="resident",
                            execution_mode="eager")


def _measure(samples):
    return {"method": "cuda_events", "samples_ms": list(samples), "warmup_iterations": 10,
            "receipt_path": "bench.json", "receipt_sha256": "a" * 64}


def _row(structure, shape, family, rate, m, lane_launch, samples):
    return {"structure": structure, "rank_local_shape": shape, "family": family, "rate_q256": rate,
            "m": m, "kernel_lane": dict(lane_launch), "measurement": _measure(samples)}


def _doc(rows, pools=(), regimes=(2048,), **context):
    base = {"runtime_image_digest": IMAGE, "tessera_commit": COMMIT, "contract_sha256": SHA,
            "tensor_parallel": 2, "platform": "sm_121", "execution_mode": "eager",
            "residency": "resident", "batch_size": 1, "regimes": list(regimes)}
    base.update(context)
    return {"schema": srp.SCHEMA, "table_id": "fixture", "status": "proposal_data",
            "composition": "sequential_operator_sum", "claims": dict(srp.CLAIMS),
            "context": base, "rows": list(rows), "rate_pools": list(pools)}


#: The five GLM-5.3 serving shapes at TP2, as the after-#640 bench timed them.
GLM_ROWS = [
    ("routed_moe", ROUTED, BF16, 1024, FUSED | {"decoder": "native_routed_fused_window_folded"}, 20.9),
    ("routed_moe", ROUTED, E4M3, 1024, FUSED, 17.5),
    ("routed_moe", ROUTED, E2M1, 896, SPAN2_GROUPED, 75.5),
] + [
    ("dense", shape, family, rate, launch, t)
    for shape, times in (("12288x4096", (17.5, 18.5, 7.9)), ("4096x6144", (8.5, 9.2, 3.8)),
                         ("2048x4096", (3.16, 3.29, 1.45)), ("4096x1024", (1.42, 1.54, 0.59)))
    for (family, rate, launch), t in zip(((BF16, 1024, DENSE_BF16), (E4M3, 1024, DENSE_E4M3),
                                          (E2M1, 896, SPAN2)), times)
]


def _glm_table(eligibility, rows=GLM_ROWS, ms=(2048,)):
    doc = _doc([_row(s, shape, fam, rate, m, launch, (t, t * 1.01, t * 0.99))
                for s, shape, fam, rate, launch, t in rows for m in ms], regimes=ms)
    return srp.admit_shape_table(srp.parse_shape_table(doc), scope=SCOPE, eligibility=eligibility)


@pytest.fixture(scope="module")
def glm_eligibility(eligibility):
    # The after-#640 bench timed BF16 routed stacks on the folded fused lane,
    # which the pinned v42 contract publishes itself (asserted in _payload).
    return eligibility


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #

def test_a_table_parses_and_carries_its_fixed_claims():
    doc = _doc([_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED, (17.4, 17.5, 17.6)),
                _row("dense", "4096x1024", E4M3, 1024, 2048, DENSE_E4M3, (1.5, 1.6, 1.4))])
    table = srp.parse_shape_table(doc)
    identity = table.identity()
    assert identity["schema"] == "prismaquant.shape_runtime_prices.v1"
    assert identity["claims"] == {"time_claim": "operator_sum_proposal",
                                  "certifies_placement": False, "served_p95": "not_claimed"}
    assert identity["n_rows"] == 2 and not table.admitted
    found = table.lookup(srp.ShapeKey("routed_moe", ROUTED, E4M3, 1024, 2048))
    assert found.median_ms == 17.5 and found.kernel_lane.decoder == "native_routed_fused_window"
    assert table.lookup(srp.ShapeKey("routed_moe", ROUTED, E4M3, 896, 2048)) is None
    # The table round-trips to the same identity.
    assert srp.parse_shape_table(table.as_dict()).identity()["sha256"] == identity["sha256"]


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["claims"].update(certifies_placement=True), "claims"),
    (lambda d: d["context"].update(batch_size=2), "batch_size"),
    (lambda d: d["rows"][0]["measurement"].update(samples_ms=[1.0, 1.0]), "three"),
    (lambda d: d["rows"][0]["measurement"].update(method="synchronized_gpu_wall_clock"), "method"),
    (lambda d: d["rows"][0].update(rank_local_shape="4096x1024"), "routed shape"),
    (lambda d: d["rows"][0].update(m=512), "regime"),
    (lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])), "duplicate"),
    (lambda d: d["rows"][0].update(weight_identity="x"), "fields"),
])
def test_a_malformed_table_is_refused_by_name(mutate, needle):
    doc = _doc([_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED, (17.4, 17.5, 17.6))])
    mutate(doc)
    with pytest.raises(srp.ShapeRuntimeError, match=needle):
        srp.parse_shape_table(doc)


def test_load_verifies_every_receipt_digest(tmp_path):
    receipt = tmp_path / "bench.json"
    receipt.write_bytes(b"{}")
    doc = _doc([_row("dense", "4096x1024", E4M3, 1024, 2048, DENSE_E4M3, (1.5, 1.6, 1.4))])
    doc["rows"][0]["measurement"]["receipt_sha256"] = hashlib.sha256(b"{}").hexdigest()
    path = tmp_path / "table.json"
    path.write_text(json.dumps(doc))
    assert srp.load_shape_table(path).identity()["n_rows"] == 1
    receipt.write_bytes(b"{} ")
    with pytest.raises(srp.ShapeRuntimeError, match="SHA-256 mismatch"):
        srp.load_shape_table(path)


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field, value", [
    ("contract_sha256", "d" * 64), ("tessera_commit", "f" * 40),
    ("runtime_image_digest", IMAGE[:-1] + "0"), ("tensor_parallel", 1),
    ("platform", "gfx1151"), ("residency", "streamed"), ("execution_mode", "compiled"),
])
def test_admission_refuses_identity_drift(eligibility, field, value):
    doc = _doc([_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED, (17.4, 17.5, 17.6))],
               **{field: value})
    table = srp.parse_shape_table(doc)
    with pytest.raises(srp.ShapeRuntimeError, match=f"pinned scope on {field}"):
        srp.admit_shape_table(table, scope=SCOPE, eligibility=eligibility)


def test_admission_refuses_an_eligibility_table_that_is_not_the_pinned_contract(payload):
    other = lane._parse_table(payload["lane_eligibility"], payload["formats"], "", COMMIT,
                              "d" * 64, native_extensions=payload["native_extensions"])
    table = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED,
                                             (17.4, 17.5, 17.6))]))
    with pytest.raises(srp.ShapeRuntimeError, match="not the pinned contract"):
        srp.admit_shape_table(table, scope=SCOPE, eligibility=other)


def test_the_fused_r1024_row_admits_and_the_fused_r896_row_refuses(eligibility):
    admitted = srp.admit_shape_table(
        srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED,
                                         (17.4, 17.5, 17.6))])),
        scope=SCOPE, eligibility=eligibility)
    assert admitted.admitted
    assert admitted.admission["cell_by_key"] == {
        f"routed_moe|{ROUTED}|{E4M3}|R1024|M2048": "tessera_e4m3_k1_routed_moe_sm121_batch_resident"}
    assert admitted.admission["per_unit_weight_identity"] == "not_claimed"
    fused_896 = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E4M3, 896, 2048, FUSED,
                                                 (17.4, 17.5, 17.6))]))
    with pytest.raises(srp.ShapeRuntimeError, match="R896.*tessera_routed_fused_e4m3"):
        srp.admit_shape_table(fused_896, scope=SCOPE, eligibility=eligibility)
    # The same cell names the compact launch too, and the fused lane's
    # predicate is not asked about it: R896 on compact admits.
    compact_896 = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E4M3, 896, 2048, COMPACT,
                                                   (109.2, 109.3, 109.4))]))
    assert srp.admit_shape_table(compact_896, scope=SCOPE, eligibility=eligibility).admitted


def test_a_launch_the_cell_does_not_name_is_refused(eligibility):
    table = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E2M1, 896, 2048, FUSED,
                                             (17.4, 17.5, 17.6))]))
    with pytest.raises(srp.ShapeRuntimeError, match="not the row's launch"):
        srp.admit_shape_table(table, scope=SCOPE, eligibility=eligibility)
    uncovered = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E2M1, 1024, 2048,
                                                 SPAN2_GROUPED, (1.0, 1.1, 0.9))]))
    with pytest.raises(srp.ShapeRuntimeError, match="no pinned lane cell covers"):
        srp.admit_shape_table(uncovered, scope=SCOPE, eligibility=eligibility)


def test_m_one_is_the_decode_regime_and_larger_m_the_batch_regime(eligibility):
    assert srp.regime_for_m(1) == "decode" and srp.regime_for_m(2) == "batch"
    table = srp.admit_shape_table(srp.parse_shape_table(_doc(
        [_row("routed_moe", ROUTED, E4M3, 1024, m, FUSED, (1.0, 1.1, 0.9)) for m in (1, 2048)],
        regimes=(1, 2048))), scope=SCOPE, eligibility=eligibility)
    cells = table.admission["cell_by_key"]
    assert cells[f"routed_moe|{ROUTED}|{E4M3}|R1024|M1"].endswith("_decode_resident")
    assert cells[f"routed_moe|{ROUTED}|{E4M3}|R1024|M2048"].endswith("_batch_resident")


# --------------------------------------------------------------------------- #
# Rate pools
# --------------------------------------------------------------------------- #

def _pool(rates, launch=DENSE_BF16, shape="4096x1024", family=BF16):
    return {"structure": "dense", "rank_local_shape": shape, "family": family,
            "kernel_lane": dict(launch), "rates_q256": list(rates)}


def test_a_rate_pool_prices_its_rates_from_the_concatenated_samples(eligibility):
    rows = [_row("dense", "4096x1024", BF16, 1088, 2048, DENSE_BF16, (1.44, 1.45, 1.43)),
            _row("dense", "4096x1024", BF16, 832, 2048, DENSE_BF16, (1.41, 1.42, 1.40))]
    table = srp.admit_shape_table(srp.parse_shape_table(_doc(rows, [_pool((832, 1024, 1088))])),
                                  scope=SCOPE, eligibility=eligibility)
    assert table.admission["pooled_rates_admitted"] == 3
    found = table.lookup(srp.ShapeKey("dense", "4096x1024", BF16, 1024, 2048))
    assert found.source == "rate_pool"
    # Concatenated in the pool's ascending rate order.
    assert found.samples_ms == (1.41, 1.42, 1.40, 1.44, 1.45, 1.43)
    assert found.median_ms == statistics.median(found.samples_ms)
    assert found.pool["source_rates_q256"] == [832, 1088]
    assert found.pool["cross_rate_spread_measured"] is True
    single = srp.parse_shape_table(_doc(rows[:1], [_pool((1024, 1088))]))
    record = single.lookup(srp.ShapeKey("dense", "4096x1024", BF16, 1024, 2048)).pool
    assert record["cross_rate_spread_measured"] is False and record["cross_rate_spread_ms"] is None


def test_a_rate_pool_across_two_kernel_lanes_is_refused(eligibility):
    rows = [_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED, (17.4, 17.5, 17.6)),
            _row("routed_moe", ROUTED, E4M3, 896, 2048, COMPACT, (109.2, 109.3, 109.4))]
    pool = {"structure": "routed_moe", "rank_local_shape": ROUTED, "family": E4M3,
            "kernel_lane": FUSED, "rates_q256": [896, 1024]}
    with pytest.raises(srp.ShapeRuntimeError, match="share the pool's kernel lane"):
        srp.parse_shape_table(_doc(rows, [pool]))
    # A pool that would lend the fused lane to R896 with no R896 row is
    # refused at admission by the lane's own predicate.
    lent = srp.parse_shape_table(_doc(rows[:1], [pool]))
    with pytest.raises(srp.ShapeRuntimeError, match="pooled .*R896"):
        srp.admit_shape_table(lent, scope=SCOPE, eligibility=eligibility)


# --------------------------------------------------------------------------- #
# Unit derivation and pricing
# --------------------------------------------------------------------------- #

def test_served_operator_shapes_for_the_glm_units():
    experts = {f"L.mlp.experts.{e}.{role}": shape for e in range(288)
               for role, shape in (("gate_proj", (2048, 4096)), ("up_proj", (2048, 4096)),
                                   ("down_proj", (4096, 2048)))}
    assert srp.served_operator(experts, structure="routed_moe", tensor_parallel=2) == ROUTED
    assert srp.served_operator({"L.mlp.gate_proj": (12288, 4096), "L.mlp.up_proj": (12288, 4096)},
                               structure="dense", tensor_parallel=2) == "12288x4096"
    assert srp.served_operator({"L.mlp.down_proj": (4096, 12288)}, structure="dense",
                               tensor_parallel=2) == "4096x6144"
    with pytest.raises(srp.ShapeRuntimeError, match="no tensor-parallel cut rule"):
        srp.served_operator({"L.attn.q_proj": (4096, 4096)}, structure="dense", tensor_parallel=2)
    with pytest.raises(srp.ShapeRuntimeError, match="column-parallel"):
        srp.served_operator({"L.mlp.gate_proj": (8, 4), "L.mlp.down_proj": (4, 8)},
                            structure="dense", tensor_parallel=1)
    ragged = dict(experts)
    ragged["L.mlp.experts.7.down_proj"] = (4096, 1024)
    with pytest.raises(srp.ShapeRuntimeError):
        srp.served_operator(ragged, structure="routed_moe", tensor_parallel=2)


FMTS = {"T16": f"{BF16}_R1024", "T8": f"{E4M3}_R1024", "T4": f"{E2M1}_R896"}


def _glm_units():
    """132 serving units in the GLM-5.3 roster: 42 routed, 42+42 shared, 3+3 dense."""
    members, shapes, structure = {}, {}, {}
    for layer in range(3):
        members[f"L{layer}.dense.gate_up"] = {f"m.{layer}.mlp.gate_proj": (12288, 4096),
                                              f"m.{layer}.mlp.up_proj": (12288, 4096)}
        members[f"L{layer}.dense.down"] = {f"m.{layer}.mlp.down_proj": (4096, 12288)}
    for layer in range(3, 45):
        members[f"L{layer}.shared.gate_up"] = {f"m.{layer}.mlp.shared_experts.gate_proj": (2048, 4096),
                                               f"m.{layer}.mlp.shared_experts.up_proj": (2048, 4096)}
        members[f"L{layer}.shared.down"] = {f"m.{layer}.mlp.shared_experts.down_proj": (4096, 2048)}
        members[f"L{layer}.experts"] = {
            f"m.{layer}.mlp.experts.{e}.{role}": shape for e in range(288)
            for role, shape in (("gate_proj", (2048, 4096)), ("up_proj", (2048, 4096)),
                                ("down_proj", (4096, 2048)))}
    for unit, roster in members.items():
        for name, shape in roster.items():
            shapes[name] = shape
            structure[name] = "routed_moe" if ".mlp.experts." in name else "dense"
    return members, shapes, structure


def _candidates(members, fmts=FMTS.values()):
    candidates = {unit: [Candidate(fmt=f, bits_per_param=4.0, memory_bytes=1000 + i,
                                   predicted_dloss=0.1 * i) for i, f in enumerate(fmts)]
                  for unit in members}
    option_members = {(unit, c.fmt): {name: c.fmt for name in members[unit]}
                      for unit, options in candidates.items() for c in options}
    return candidates, option_members


def test_132_units_are_priced_from_the_glm_shape_rows(glm_eligibility, formats):
    table = _glm_table(glm_eligibility)
    members, shapes, structure = _glm_units()
    assert len(members) == 132
    candidates, option_members = _candidates(members)
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    assert len(pricing.resources) == 396 and not pricing.gaps
    assert pricing.operators["L10.experts"] == ROUTED
    assert pricing.operators["L10.shared.gate_up"] == "2048x4096"
    assert pricing.operators["L1.dense.down"] == "4096x6144"
    t8 = pricing.resources[("L10.experts", FMTS["T8"])]
    assert t8.prefill_ms == 17.5 and t8.decode_ms is None and t8.serialized_bytes == 1001
    routed_t4 = sum(pricing.resources[(u, FMTS["T4"])].prefill_ms for u in members if "experts" in u)
    assert routed_t4 == pytest.approx(42 * 75.5)


def test_an_unpriced_option_is_absent_not_zero(glm_eligibility, formats):
    rows = [row for row in GLM_ROWS if not (row[1] == "4096x1024" and row[2] == E2M1)]
    table = _glm_table(glm_eligibility, rows=rows)
    members, shapes, structure = _glm_units()
    candidates, option_members = _candidates(members)
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    missing = {(g["unit"], g["format"]) for g in pricing.gaps}
    assert missing == {(f"L{layer}.shared.down", FMTS["T4"]) for layer in range(3, 45)}
    assert all(g["kind"] == "no_time_row" for g in pricing.gaps)
    assert not missing & set(pricing.resources)
    kept = pricing.time_candidates(candidates)
    assert [c.fmt for c in kept["L10.shared.down"]] == [FMTS["T16"], FMTS["T8"]]
    assert len(kept["L10.experts"]) == 3
    assert pricing.gap_report()["by_kind"] == {"no_time_row": 42}
    # A unit whose every option is unpriced cannot be assigned at all.
    only_t4 = {unit: [c for c in options if c.fmt == FMTS["T4"]] for unit, options in candidates.items()}
    thin = srp.build_shape_runtime_resources(
        table, only_t4, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    with pytest.raises(srp.ShapeRuntimeError, match="42 unit.*no option"):
        thin.time_candidates(only_t4)


def test_a_mixed_rate_operator_and_a_passthrough_format_are_gaps(glm_eligibility, formats):
    table = _glm_table(glm_eligibility)
    members = {"L10.shared.gate_up": {"m.10.mlp.shared_experts.gate_proj": (2048, 4096),
                                      "m.10.mlp.shared_experts.up_proj": (2048, 4096)}}
    shapes = dict(members["L10.shared.gate_up"])
    structure = {name: "dense" for name in shapes}
    candidates = {"L10.shared.gate_up": [
        Candidate(fmt="GROUP", bits_per_param=4.0, memory_bytes=1, predicted_dloss=0.0),
        Candidate(fmt="BF16", bits_per_param=16.0, memory_bytes=2, predicted_dloss=0.0)]}
    option_members = {("L10.shared.gate_up", "GROUP"): {
        "m.10.mlp.shared_experts.gate_proj": f"{E4M3}_R1024",
        "m.10.mlp.shared_experts.up_proj": f"{E4M3}_R896"},
        ("L10.shared.gate_up", "BF16"): {name: "BF16" for name in shapes}}
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    assert {g["format"]: g["kind"] for g in pricing.gaps} == {
        "GROUP": "mixed_rate_operator", "BF16": "not_rate_addressed"}
    assert not pricing.resources


def test_decode_prices_from_the_m_one_rows_and_an_unmeasured_regime_refuses(glm_eligibility, formats):
    table = _glm_table(glm_eligibility, ms=(1, 2048))
    members, shapes, structure = _glm_units()
    candidates, option_members = _candidates(members)
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    assert pricing.resources[("L10.experts", FMTS["T8"])].decode_ms == 17.5
    with pytest.raises(srp.ShapeRuntimeError, match="not a regime the table measured"):
        srp.build_shape_runtime_resources(
            table, candidates, option_members=option_members, member_shapes=shapes,
            member_structure=structure, regime_m=512, published_formats=formats)
    unadmitted = srp.parse_shape_table(table.as_dict())
    with pytest.raises(srp.ShapeRuntimeError, match="after admit_shape_table"):
        srp.build_shape_runtime_resources(
            unadmitted, candidates, option_members=option_members, member_shapes=shapes,
            member_structure=structure, regime_m=2048, published_formats=formats)


# --------------------------------------------------------------------------- #
# Dispersion
# --------------------------------------------------------------------------- #

def test_bootstrap_sum_without_multiplicities_is_unchanged():
    rows = [(1.0, 2.0, 3.0, 4.0), (10.0, 11.0, 12.5)]
    rng = random.Random(7)
    expected = sorted(sum(statistics.median(rng.choices(list(s), k=len(s))) for s in rows)
                      for _ in range(200))
    result = bootstrap_sum(rows, draws=200, seed=7)
    assert result["p2.5"] == expected[5] and result["p97.5"] == expected[195]
    assert "multiplicities" not in result
    ones = bootstrap_sum(rows, draws=200, seed=7, multiplicities=[1, 1])
    assert (ones["p2.5"], ones["p50"], ones["p97.5"]) == (result["p2.5"], result["p50"], result["p97.5"])
    with pytest.raises(RuntimePriceError, match="multiplicities"):
        bootstrap_sum(rows, draws=10, seed=1, multiplicities=[1])


def test_one_shared_row_is_drawn_once_per_draw(glm_eligibility, formats):
    table = _glm_table(glm_eligibility)
    members, shapes, structure = _glm_units()
    candidates, option_members = _candidates(members)
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    routed = {unit: FMTS["T8"] for unit in members if unit.endswith("experts")}
    shared = pricing.operator_sum_bootstrap(routed, draws=400, seed=3)
    one = bootstrap_sum([pricing.prefill[("L10.experts", FMTS["T8"])].samples_ms], draws=400, seed=3)
    assert shared["distinct_measurements"] == 1 and shared["multiplicities"] == [42]
    assert shared["p97.5"] - shared["p2.5"] == pytest.approx(42 * (one["p97.5"] - one["p2.5"]))


def test_the_receipt_converter_is_a_named_stub(tmp_path):
    assert srp.main(["convert", "--out", str(tmp_path / "t.json"), "--table-id", "x",
                     "--receipts", str(tmp_path / "r.json")]) == 2
    with pytest.raises(srp.ShapeRuntimeError, match="tessera#688"):
        srp.consume_shape_time_panel([], table_id="x")
