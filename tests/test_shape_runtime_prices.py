"""PACT shape-time price table (PQ #1583): parse, admit, derive, price.

The contract fixture is the INSTALLED pinned contract, read as published.
Since the pin moved to contract v42 (PQ #1274) it carries Tessera #685's fused
routed lanes itself: the routed E4M3 and BF16 cells name the fused launch
beside the compact one, and each lane publishes its ``requires`` predicate on
its own ``native_extensions`` row. The fixtures assert that instead of
grafting it. Since the pin moved to contract v45 (PQ #1702) each fused lane
reads column rates 1 to 8 and gates its routed-expert launch to rates 1 to 6
(``column_rates_routed_moe``), so every rung the routed E4M3 cells list,
q832 to q1088, admits the fused launch; the refusal legs rewind the lanes to
v44's rate-4 predicate. The lane decision itself is Tessera's
(``decide_lane_requirements``) reached through
``lane_eligibility.cell_lane_admits``; nothing here restates its rule.
"""
import copy
import hashlib
import json
import random
import statistics
from importlib.resources import as_file
from pathlib import Path

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
              "requires": {"column_rates": [1, 2, 3, 4, 5, 6, 7, 8],
                           "column_rates_routed_moe": [1, 2, 3, 4, 5, 6],
                           "window_bits": [14], "body": "window",
                           "plane": "channel", "release_overrides": False, "diagonals": False,
                           "rotation": ["none"], "grid_arities": [1]}}
FUSED_LANE_PREFIXES = ("tessera_routed_fused_e4m3", "tessera_routed_fused_value")


FOLDED_COMPACT = dict(COMPACT, decoder="native_window_moe_compact_folded")
FOLDED_FUSED = dict(FUSED, decoder="native_routed_fused_window_folded")


def _payload():
    with as_file(contract.contract_path()) as path:
        payload = json.loads(path.read_bytes())
    # The pinned contract (v42 on, with v45's rate sets) publishes both fused
    # lanes and names each beside the compact launch in the routed cells; the
    # fixtures below depend on exactly that, so a re-pin that moves it fails
    # here.
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


def _v44_eligibility(payload):
    """The pinned table with both fused lanes rewound to v44's predicate
    (``column_rates`` [4] and no routed set), under the fixture's digest so
    the pinned-contract check still passes.  The pop has no default, so a pin
    whose fused lanes do not publish the routed set fails here."""
    moved = copy.deepcopy(payload)
    for row in moved["native_extensions"]:
        if row["module_name_prefix"] in FUSED_LANE_PREFIXES:
            requires = row["lane"]["requires"]
            requires.pop("column_rates_routed_moe")
            requires["column_rates"] = [4]
    return lane._parse_table(moved["lane_eligibility"], moved["formats"], "", COMMIT, SHA,
                             native_extensions=moved["native_extensions"])


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
    # which the pinned contract publishes itself (asserted in _payload).
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


def test_a_declared_but_unmeasured_regime_is_refused():
    doc = _doc([_row("dense", "4096x1024", E4M3, 1024, 2048, DENSE_E4M3,
                     (1.5, 1.6, 1.4))], regimes=(512, 2048))
    with pytest.raises(srp.ShapeRuntimeError, match="context declares regimes"):
        srp.parse_shape_table(doc)


def test_load_verifies_every_receipt_digest(tmp_path):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="digest")
    path = tmp_path / "table.json"
    srp.write_shape_table(table, path)
    assert srp.load_shape_table(path).identity()["n_rows"] == 1
    receipt = tmp_path / "obs" / "checker-receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(srp.ShapeRuntimeError, match="SHA-256"):
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


def test_fused_r1024_and_r896_rows_admit_and_a_v44_lane_refuses_r896(payload, eligibility):
    """Since contract v45 the fused routed lane reads the mixed-rate plans the
    routed cells list, so a fused R896 row admits beside the R1024 one. The
    lane's predicate still decides: rewound to v44's rate-4 predicate, the same
    lane refuses the fused R896 row, and R896 on the compact launch admits
    either way."""
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
    assert srp.admit_shape_table(fused_896, scope=SCOPE, eligibility=eligibility).admission[
        "cell_by_key"] == {
        f"routed_moe|{ROUTED}|{E4M3}|R896|M2048": "tessera_e4m3_k1_routed_moe_sm121_batch_resident"}
    v44 = _v44_eligibility(payload)
    with pytest.raises(srp.ShapeRuntimeError, match="R896.*tessera_routed_fused_e4m3"):
        srp.admit_shape_table(fused_896, scope=SCOPE, eligibility=v44)
    # The same cell names the compact launch too, and the fused lane's
    # predicate is not asked about it: R896 on compact admits under both.
    compact_896 = srp.parse_shape_table(_doc([_row("routed_moe", ROUTED, E4M3, 896, 2048, COMPACT,
                                                   (109.2, 109.3, 109.4))]))
    for table in (eligibility, v44):
        assert srp.admit_shape_table(compact_896, scope=SCOPE, eligibility=table).admitted


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


def test_a_rate_pool_across_two_kernel_lanes_is_refused(payload, eligibility):
    rows = [_row("routed_moe", ROUTED, E4M3, 1024, 2048, FUSED, (17.4, 17.5, 17.6)),
            _row("routed_moe", ROUTED, E4M3, 896, 2048, COMPACT, (109.2, 109.3, 109.4))]
    pool = {"structure": "routed_moe", "rank_local_shape": ROUTED, "family": E4M3,
            "kernel_lane": FUSED, "rates_q256": [896, 1024]}
    with pytest.raises(srp.ShapeRuntimeError, match="share the pool's kernel lane"):
        srp.parse_shape_table(_doc(rows, [pool]))
    # A pool that lends the fused lane to R896 with no R896 row is decided at
    # admission by the lane's own predicate. The v45 lane reads R896's plan, so
    # the pool admits both rates, and the pooled R896 carries R1024's samples
    # with ``cross_rate_spread_measured`` false: the pool, not a measurement,
    # prices it. Rewound to v44's rate-4 predicate, the lane refuses it.
    lent = srp.parse_shape_table(_doc(rows[:1], [pool]))
    admitted = srp.admit_shape_table(lent, scope=SCOPE, eligibility=eligibility)
    assert admitted.admission["pooled_rates_admitted"] == 2
    record = admitted.lookup(srp.ShapeKey("routed_moe", ROUTED, E4M3, 896, 2048)).pool
    assert record["cross_rate_spread_measured"] is False
    with pytest.raises(srp.ShapeRuntimeError, match="pooled .*R896"):
        srp.admit_shape_table(lent, scope=SCOPE, eligibility=_v44_eligibility(payload))


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


def test_the_converter_refuses_without_observations(tmp_path):
    assert srp.main(["convert", "--out", str(tmp_path / "t.json"), "--table-id", "x",
                     "--observations", str(tmp_path / "missing-observation.json")]) == 2
    with pytest.raises(srp.ShapeRuntimeError, match="at least one observation"):
        srp.consume_shape_time_observation([], table_id="x")


def test_the_kernel_lane_histogram_counts_units_per_priced_lane(glm_eligibility, formats):
    table = _glm_table(glm_eligibility)
    members, shapes, structure = _glm_units()
    candidates, option_members = _candidates(members)
    pricing = srp.build_shape_runtime_resources(
        table, candidates, option_members=option_members, member_shapes=shapes,
        member_structure=structure, regime_m=2048, published_formats=formats)
    routed = {unit: FMTS["T8"] for unit in members if unit.endswith("experts")}
    histogram = pricing.kernel_lane_histogram(routed)
    lane = pricing.prefill[("L10.experts", FMTS["T8"])].kernel_lane
    assert histogram == {f"{lane.symbol}/{lane.decoder}": len(routed)}
    assert sum(histogram.values()) == len(routed)
    with pytest.raises(srp.ShapeRuntimeError, match="no prefill time"):
        pricing.kernel_lane_histogram({"L10.experts": "not-a-format"})


# --------------------------------------------------------------------------- #
# Tessera observation consumer (tessera.shape_time_observation.v1)
# --------------------------------------------------------------------------- #

OBS_IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
             "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
OBS_COMMIT = "b40c93cb73745097e57a1ba4cf5b9eee166c759a"
OBS_CONTRACT = "0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa"
OBS_PRODUCER_TOOL = "a" * 64
OBS_REPLAY_TOOL = "b" * 64
OBS_ROUTE = "TESSERA_FP8"
OBS_FAMILY = "TESSERA_E4M3_K1"
OBS_LANE = ("tessera::fused_window_dense", "native_fused_window_dense")
_CHECKER_RESULTS = {}


@pytest.fixture(autouse=True)
def checker_sdk_fixture(tmp_path, monkeypatch):
    """Domain fixtures model the public SDK; actual PB acceptance is separate."""
    from prismaquant import staged_lease
    _CHECKER_RESULTS.clear()
    config = tmp_path / "checker-config.json"
    _obs_write(config, {"schema": srp.CHECKER_CONFIG_SCHEMA, "checkers": []})
    monkeypatch.setattr(srp, "CHECKER_CONFIG_PATH", config)

    class FixtureSDK:
        PoolQueue = staticmethod(lambda: None)

        @staticmethod
        def read_verified_action_result(queue, action_key, **selector):
            result = copy.deepcopy(_CHECKER_RESULTS[action_key])
            if (selector["attempt"] != result["attempt"]
                    or selector["published_unix"] != result["published_unix"]):
                raise ValueError("wrong explicit selector")
            return result

        @staticmethod
        def bind_standard_capture_command(request):
            expected = ["fixture-capture", *request["params"]["command"]]
            if request["task"]["argv"] != expected:
                raise ValueError("unbound wrapper")
            return expected

    monkeypatch.setattr(staged_lease, "client_sdk", lambda: FixtureSDK)


def _consume_observations(observations, **kwargs):
    return srp.consume_shape_time_observation(
        observations, checker_receipts=[path.parent / "checker-receipt.json" for path in observations],
        **kwargs)


def _obs_write(path, value, raw=False):
    path.write_bytes(value if raw else srp.DIRECT_ASCII_STRICT.encoded(value) + b"\n")
    import hashlib
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def observation_fixture(tmp_path, *, m=512, tp_degree=1, samples=(1.0, 2.0, 3.0, 4.0),
                        agent="obs", mutate=None, family=OBS_FAMILY, grid="E4M3",
                        route=OBS_ROUTE, rate_q256=896, kernel_lane=OBS_LANE,
                        runtime_image=OBS_IMAGE):
    """One internal-consistent bound observation, as Tessera#856 emits it."""
    import statistics
    scope = {"route": route, "grid": grid, "q256": rate_q256, "structure": "dense",
             "mode": "resident", "execution_mode": "eager", "regime": "batch",
             "tp_degree": tp_degree, "requested_platform": "sm_121",
             "shape": {"M": m, "N": 256, "K": 256}}
    runtime = {"image": runtime_image, "tessera_commit": OBS_COMMIT, "serving_source_sha256": "c" * 64,
               "contract_sha256": OBS_CONTRACT, "platform": "sm_121", "torch": "2.13.0+cu130",
               "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904",
               "serve_flags": {"TESSERA_SERVE_MODE": "resident"}, "residency": "resident",
               "execution_mode": "eager", "tp_rank": 0, "tp_degree": tp_degree,
               "package_root": "/mnt/shared/tessera-suite-envs/pq1934-pb95-tessera-b40-py312/"
                               "site-packages/tessera"}
    root = tmp_path / agent
    root.mkdir(parents=True)
    samples_doc = {"samples_ms": list(samples), "warmup_iterations": 5,
                   "interval_unix": [10.0, 11.0]}
    routes = {"records": [{"symbol": kernel_lane[0], "decoder": kernel_lane[1]} for _ in samples]}
    samples_b = _obs_write(root / "samples.json", samples_doc)
    routes_b = _obs_write(root / "routes.json", routes)
    contract_b = _obs_write(root / "contract.bin", b"raw b40 contract bytes", raw=True)
    wire_b = _obs_write(root / "wire.bin", b"fused wire bytes", raw=True)
    producer_doc = {"schema": "tessera.native_panel_producer_identity.v1", "commit": "8eb3c05174" + "0" * 30,
                    "commit_source": "sealed_checkout", "source_tree_sha256": "d" * 64,
                    "source_tree_members": 1, "tool_source_sha256": OBS_PRODUCER_TOOL}
    producer_b = _obs_write(root / "producer.json", producer_doc)
    request_doc = {"producer_identity": producer_b}
    request_b = _obs_write(root / "request.json", request_doc)
    runtime_b = _obs_write(root / "runtime.json", runtime)
    import gzip
    with (root / "trace.json.gz").open("wb") as handle:
        handle.write(gzip.compress(b'{"traceEvents":[{"cat":"kernel","name":"fixture","dur":1.0}]}'))
    trace_b = _obs_write(root / "trace.json.gz", (root / "trace.json.gz").read_bytes(), raw=True)
    evidence = {
        "runtime": runtime_b, "producer": producer_b, "contract": contract_b, "wire": wire_b,
        "preparation": _obs_write(root / "preparation.json",
                                  {"builder": "tessera.serving.lane.build_tessera_method",
                                   "wire_sha256": wire_b["sha256"], "roles": [],
                                   "shape": scope["shape"], "tp_rank": 0, "tp_degree": 1,
                                   "grid": grid, "native_packed_bytes": 1}),
        "samples": samples_b, "routes": routes_b, "trace": trace_b,
        "telemetry": _obs_write(root / "telemetry.json",
                                {"interval_unix": [10.0, 11.0], "fast_power_samples": [[10.5, 40.0]],
                                 "netdata": {"sparky": {}, "sparklina": {}}}),
        "native_binary": _obs_write(root / "fused.so", b"\x7fELF fixture", raw=True),
        "runtime_origins": _obs_write(root / "origins.json", {"package_root": runtime["package_root"]}),
    }
    q1, _mid, q3 = statistics.quantiles([float(v) for v in samples], n=4, method="inclusive")
    timing = {"method": "cuda_events", "n": len(samples), "median_ms": float(statistics.median(samples)),
              "p25_ms": float(q1), "p75_ms": float(q3), "iqr_ms": float(q3 - q1),
              "quartiles": "statistics.quantiles.inclusive"}
    preflight = {"result": _obs_write(root / "preflight-result.json",
                                      {"schema": "tessera.installed_contract_preflight.v1"}),
                 "phase": _obs_write(root / "preflight-phase.json",
                                     {"phase": "runtime-preflight", "returncode": 0,
                                      "command": ["env", "CUDA_VISIBLE_DEVICES=", "worker",
                                                  "--job-sha256", "e" * 64, "--preflight"]})}
    panel = {"schema": "tessera.shape_time_panel.v1", "status": "measured", "claims": dict(srp.CLAIMS),
             "runtime": runtime, "plan": {"gpu_executed": False, "rows": [{"id": "ffa460d8" * 8, "scope": scope}]},
             "rows": [{"scope_id": "ffa460d8" * 8, "cell_id": "dense-e4m3-sm121-batch-resident"}],
             "evidence": evidence, "preflight": preflight, "energy": {"status": "hold"}}
    panel_b = _obs_write(root / "panel.json", panel)
    replay_b = _obs_write(root / "replay-tool.json", {"tool": "replay"})
    command = ["env", "CUDA_VISIBLE_DEVICES=", "python", "--job-sha256", "e" * 64, "--preflight"]
    observation = {
        "schema": srp.SHAPE_TIME_OBSERVATION_SCHEMA, "status": "validated",
        "claims": dict(srp.CLAIMS), "gpu_executed": False, "panel": panel_b,
        "expected_panel_sha256": panel_b["sha256"], "request": request_b,
        "expected_runtime": runtime_b, "contract": contract_b, "evidence": evidence,
        "preflight": preflight, "producer": producer_b,
        "replay": {"source_tree_sha256": "f" * 64, "source_tree_members": 1,
                   "tool_source_sha256": OBS_REPLAY_TOOL, "tool": replay_b},
        "invocation": {"command": command, "phase": "runtime-preflight", "returncode": 0},
        "scope": scope, "scope_id": "ffa460d8" * 8,
        "cell_id": f"dense-{grid.lower()}-sm121-batch-resident", "kernel_lane": list(kernel_lane),
        "structure": "dense", "rank_local_shape": "256x256", "family": family,
        "payload": {"route": route, "grid": grid, "q256": rate_q256, "rows": 256, "columns": 256},
        "timing": timing,
        "sampling": {"method": "cuda_events", "sample_unit": "single_apply", "warmup_iterations": 5,
                     "n": len(samples), "samples_ms": list(samples), "interval_unix": [10.0, 11.0]},
        "operator_projection": {"batch_size": 1, "rows": m,
                                "reading": "one 2-D M-by-K operator apply; PQ may key this row at "
                                           "batch_size=1 for M prompt rows; not end-to-end serving evidence"},
        "energy_status": "hold"}
    emitted = copy.deepcopy(observation)
    if mutate is not None:
        mutate(observation)
    obs_b = _obs_write(root / "observation.json", observation)
    key = hashlib.sha256(str(root).encode()).hexdigest()
    source = {"id": "pbrun.checkout-snapshot", "bytes": 123, "sha256": "9" * 64}
    snapshot = {"schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v2",
                "input": source, "commit": "1" * 40, "parent": "2" * 40,
                "subdirectory": ".", "refs": {}}
    command = ["fixture-checker", str(root / "panel.json"), "--observation-out", obs_b["path"]]
    environment = {"variables": {"CUDA_VISIBLE_DEVICES": ""}}
    _CHECKER_RESULTS[key] = {
        "action_key": key, "published_unix": 10.0, "attempt": 1,
        "inputs": [source], "payload": json.dumps(emitted, sort_keys=True).encode() + b"\n",
        "request": {"params": {"command": command, "checkout_snapshot": snapshot, "cwd": "."},
                    "environment": environment,
                    "task": {"argv": ["fixture-capture", *command], "working_directory": "."}}}
    config = json.loads(srp.CHECKER_CONFIG_PATH.read_bytes())
    config["checkers"].append({"snapshot": snapshot, "command": command, "cwd": ".",
                               "working_directory": ".",
                               "environment": environment, "observation_output": obs_b["path"]})
    _obs_write(srp.CHECKER_CONFIG_PATH, config)
    _obs_write(root / "checker-receipt.json", {
        "schema": srp.CHECKER_RECEIPT_SCHEMA, "observation": obs_b,
        "selector": {"action_key": key, "published_unix": 10.0, "attempt": 1}})
    return obs_b


def _obs_scope(**overrides):
    base = {"contract_sha256": OBS_CONTRACT, "tessera_commit": OBS_COMMIT,
            "runtime_image_digest": OBS_IMAGE, "tensor_parallel": 1, "platform": "sm_121",
            "residency": "resident", "execution_mode": "eager"}
    base.update(overrides)
    return srp.ShapeTableScope(**base)


def test_observation_converts_to_one_proposal_row(tmp_path):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    assert table.context.tensor_parallel == 1 and table.context.regimes == (512,)
    assert len(table.rows) == 1 and table.rate_pools == ()
    row = table.rows[0]
    assert row.key == srp.ShapeKey("dense", "256x256", OBS_FAMILY, 896, 512)
    assert row.kernel_lane.as_pair() == OBS_LANE
    assert row.measurement.method == "cuda_events"
    assert row.measurement.samples_ms == (1.0, 2.0, 3.0, 4.0)
    assert row.measurement.median_ms == 2.5 and row.measurement.warmup_iterations == 5
    # The observation times one M; conversion does not invent a decode row.
    assert table.lookup(srp.ShapeKey("dense", "256x256", OBS_FAMILY, 896, 1)) is None


def test_observation_parses_the_already_bound_panel_samples_and_routes(tmp_path, monkeypatch):
    observation_fixture(tmp_path)
    reads = []
    original = srp.ArtifactReader.bytes

    def read(reader, binding, where, **kwargs):
        result = original(reader, binding, where, **kwargs)
        reads.append(result[0])
        return result

    monkeypatch.setattr(srp.ArtifactReader, "bytes", read)
    _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    for name in ("panel.json", "samples.json", "routes.json"):
        assert reads.count(tmp_path / "obs" / name) == 1


@pytest.mark.parametrize("raw, needle", [
    (b'{"schema":"first","schema":"second"}', "duplicate JSON key"),
    (b'{"value":NaN}', "nonfinite JSON constant"),
    (b'{"value":Infinity}', "nonfinite JSON constant"),
    (b'{"value":-Infinity}', "nonfinite JSON constant"),
])
def test_observation_reader_retains_strict_json_refusals(tmp_path, raw, needle):
    path = tmp_path / "observation.json"
    path.write_bytes(raw)
    with pytest.raises(srp.ShapeRuntimeError, match=needle):
        srp.read_shape_time_observation(path)


def test_observation_itself_refuses_a_tp2_scope(tmp_path):
    observation_fixture(tmp_path, tp_degree=2)
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations([tmp_path / "obs" / "observation.json"], table_id="tp2")


def test_observation_is_admitted_against_pq_pinned_scope(tmp_path, eligibility):
    from dataclasses import replace
    eligibility = replace(eligibility, contract_sha256=OBS_CONTRACT, runtime_commit=OBS_COMMIT)
    observation_fixture(tmp_path)
    table = _consume_observations(
        [tmp_path / "obs" / "observation.json"], table_id="pilot",
        expected_scope=_obs_scope(), eligibility=eligibility)
    assert table.admitted
    assert table.admission["rows_admitted"] == 1
    assert table.admission["cell_by_key"] == {
        table.rows[0].key.label(): table.admission["cell_by_key"][table.rows[0].key.label()]}
    assert list(table.admission["cell_by_key"]) == [table.rows[0].key.label()]


def test_a_tp2_expected_scope_refuses_the_tp1_observation(tmp_path, eligibility):
    observation_fixture(tmp_path)
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations(
            [tmp_path / "obs" / "observation.json"], table_id="pilot",
            expected_scope=_obs_scope(tensor_parallel=2), eligibility=eligibility)


def test_an_unbacked_kernel_lane_refuses_pq_admission(tmp_path, eligibility):
    def swap_lane(doc):
        doc["kernel_lane"] = ["tessera::window_gemm_dense", "native_window_gemm"]
    observation_fixture(tmp_path, mutate=swap_lane)
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations(
            [tmp_path / "obs" / "observation.json"], table_id="pilot",
            expected_scope=_obs_scope(), eligibility=eligibility)


@pytest.mark.parametrize("field,value", [
    ("cuda_events", "synchronized_gpu_wall_clock"), ("single_apply", "loop_average")])
def test_observation_refuses_a_non_event_or_loop_timing_method(tmp_path, field, value):
    def corrupt(doc):
        key = "method" if field == "cuda_events" else "sample_unit"
        doc["sampling"][key] = value
    observation_fixture(tmp_path, mutate=corrupt)
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations([tmp_path / "obs" / "observation.json"], table_id="x")


@pytest.mark.parametrize("mutation", ["samples", "summary", "lane", "family", "duplicate", "token"])
def test_observation_refuses_tampered_or_unbound_fields(tmp_path, mutation):
    def corrupt(doc):
        if mutation == "samples":
            doc["sampling"]["samples_ms"] = [1.0, 2.0, 99.0, 4.0]
        elif mutation == "summary":
            doc["timing"]["median_ms"] = 9.0
        elif mutation == "lane":
            doc["kernel_lane"] = ["x", "y"]
        elif mutation == "family":
            # Self-consistent with the panel's own claims: the observation
            # names a family its bound payload route does not resolve to.
            doc["family"] = "TESSERA_E2M1_K2"
        elif mutation == "token":
            # The replay may share the producer's SOURCE TREE; what it may not
            # do is be the sealed producer tool itself. Point the replay tool
            # at the producer's own bound bytes.
            doc["replay"]["tool"] = dict(doc["producer"])
    obs = [tmp_path / "obs" / "observation.json"]
    if mutation == "duplicate":
        observation_fixture(tmp_path)
        _consume_observations(obs, table_id="one")
        # Two observations with the same shape key refuse.
        with pytest.raises(srp.ShapeRuntimeError, match="duplicate shape key"):
            _consume_observations(obs * 2, table_id="two")
        return
    observation_fixture(tmp_path, mutate=corrupt)
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations(obs, table_id="x")


def test_convert_writes_out_atomically_and_refuses_on_a_bad_observation(tmp_path):
    observation_fixture(tmp_path)
    out = tmp_path / "table.json"
    rc = srp.main(["convert", "--out", str(out), "--table-id", "pilot",
                   "--checker-receipts", str(tmp_path / "obs" / "checker-receipt.json"),
                   "--observations", str(tmp_path / "obs" / "observation.json")])
    assert rc == 0 and out.exists()
    loaded = srp.load_shape_table(out)
    assert len(loaded.rows) == 1 and loaded.rows[0].measurement.samples_ms == (1.0, 2.0, 3.0, 4.0)
    # A refused conversion leaves no output and no partial file.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "observation.json").write_bytes(b"{not json")
    out2 = tmp_path / "should-not-exist.json"
    assert srp.main(["convert", "--out", str(out2), "--table-id", "bad",
                     "--checker-receipts", str(tmp_path / "obs" / "checker-receipt.json"),
                     "--observations", str(broken / "observation.json")]) == 2
    assert not out2.exists()


def test_load_refuses_a_row_whose_samples_were_edited_with_a_valid_receipt(tmp_path):
    observation_fixture(tmp_path)
    table_path = tmp_path / "table.json"
    srp.main(["convert", "--out", str(table_path), "--table-id", "pilot",
              "--checker-receipts", str(tmp_path / "obs" / "checker-receipt.json"),
              "--observations", str(tmp_path / "obs" / "observation.json")])
    document = json.loads(table_path.read_text())
    document["rows"][0]["measurement"]["samples_ms"] = [10.0, 20.0, 30.0, 40.0]
    table_path.write_text(srp.DIRECT_ASCII_STRICT.text(document))
    with pytest.raises(srp.ShapeRuntimeError, match="differ from the receipt's raw samples"):
        srp.load_shape_table(table_path)


def test_load_refuses_a_forged_panel_with_matching_hashes_and_no_checker(tmp_path):
    observation_fixture(tmp_path)
    observation = json.loads((tmp_path / "obs" / "observation.json").read_bytes())
    row = _row("dense", "256x256", OBS_FAMILY, 896, 512,
               dict(zip(("symbol", "decoder"), OBS_LANE)), (1.0, 2.0, 3.0, 4.0))
    row["measurement"].update(receipt_path=observation["panel"]["path"],
                              receipt_sha256=observation["panel"]["sha256"],
                              warmup_iterations=5)
    document = _doc([row], regimes=(512,), runtime_image_digest=OBS_IMAGE,
                    tessera_commit=OBS_COMMIT, contract_sha256=OBS_CONTRACT,
                    tensor_parallel=1)
    path = tmp_path / "forged-table.json"
    _obs_write(path, document)
    with pytest.raises(srp.ShapeRuntimeError, match="checker"):
        srp.load_shape_table(path)


@pytest.mark.parametrize("mutation", ["missing", "selector", "source", "commit", "subdirectory", "refs",
                                      "cwd", "working_directory", "command", "environment", "wrapper"])
def test_observation_requires_the_selected_reviewed_checker_at_convert_and_load(tmp_path, mutation):
    observation_fixture(tmp_path)
    observation = tmp_path / "obs" / "observation.json"
    table = _consume_observations([observation], table_id="pilot")
    table_path = tmp_path / "table.json"
    srp.write_shape_table(table, table_path)
    result = next(iter(_CHECKER_RESULTS.values()))
    if mutation == "missing":
        _CHECKER_RESULTS.clear()
    elif mutation == "selector":
        result["attempt"] = 2
    elif mutation == "source":
        result["request"]["params"]["checkout_snapshot"]["input"]["sha256"] = "8" * 64
    elif mutation == "commit":
        result["request"]["params"]["checkout_snapshot"]["commit"] = "3" * 40
    elif mutation == "subdirectory":
        result["request"]["params"]["checkout_snapshot"]["subdirectory"] = "archive"
    elif mutation == "refs":
        result["request"]["params"]["checkout_snapshot"]["refs"] = {"other": "3" * 40}
    elif mutation == "cwd":
        result["request"]["params"]["cwd"] = "archive"
    elif mutation == "working_directory":
        result["request"]["task"]["working_directory"] = "archive"
    elif mutation == "command":
        result["request"]["params"]["command"][0] = "forged-checker"
        result["request"]["task"]["argv"][1] = "forged-checker"
    elif mutation == "environment":
        result["request"]["environment"]["variables"]["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        result["request"]["task"]["argv"][0] = "unbound"
    with pytest.raises(srp.ShapeRuntimeError):
        _consume_observations([observation], table_id="forged")
    with pytest.raises(srp.ShapeRuntimeError):
        srp.load_shape_table(table_path)


def test_convert_refuses_forged_observation_without_a_real_checker(tmp_path):
    observation_fixture(tmp_path)
    _CHECKER_RESULTS.clear()
    out = tmp_path / "out.json"
    assert srp.main(["convert", "--out", str(out), "--table-id", "forged",
                     "--observations", str(tmp_path / "obs" / "observation.json"),
                     "--checker-receipts", str(tmp_path / "obs" / "checker-receipt.json")]) == 2
    assert not out.exists()


def test_checker_result_requires_exact_output_and_a_bounded_observation(tmp_path, monkeypatch):
    binding = observation_fixture(tmp_path)
    path = tmp_path / "obs" / "observation.json"
    result = next(iter(_CHECKER_RESULTS.values()))
    result["payload"] *= 2
    with pytest.raises(srp.ShapeRuntimeError, match="owned output"):
        _consume_observations([path], table_id="duplicate-output")
    monkeypatch.setattr(srp, "CHECKER_RESULT_MAX_BYTES", binding["bytes"] - 1)
    with pytest.raises(srp.ShapeRuntimeError, match="byte cap"):
        _consume_observations([path], table_id="oversized")


def test_load_refuses_a_rate_pool_that_lends_samples_to_an_unmeasured_rate(tmp_path):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    doc = table.as_dict()
    row = doc["rows"][0]
    doc["rate_pools"] = [{"structure": row["structure"],
                          "rank_local_shape": row["rank_local_shape"],
                          "family": row["family"], "kernel_lane": row["kernel_lane"],
                          "rates_q256": [896, 1024]}]
    path = tmp_path / "tampered-table.json"
    _obs_write(path, doc)
    with pytest.raises(srp.ShapeRuntimeError, match="rate_pools"):
        srp.load_shape_table(path)


def test_null_checker_observation_binding_refuses_cleanly(tmp_path):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    receipt_path = tmp_path / "obs" / "checker-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["observation"] = None
    binding = _obs_write(receipt_path, receipt)
    doc = table.as_dict()
    doc["rows"][0]["measurement"]["receipt_sha256"] = binding["sha256"]
    path = tmp_path / "tampered-table.json"
    _obs_write(path, doc)
    with pytest.raises(srp.ShapeRuntimeError, match="expected an object"):
        srp.load_shape_table(path)


@pytest.mark.parametrize("field", ["key", "lane", "context"])
def test_load_compares_the_table_to_the_checker_projection(tmp_path, field):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    doc = table.as_dict()
    if field == "key":
        doc["rows"][0]["family"] = "TESSERA_E2M1_K2"
    elif field == "lane":
        doc["rows"][0]["kernel_lane"]["decoder"] = "other"
    else:
        doc["context"]["tensor_parallel"] = 2
    path = tmp_path / "tampered-table.json"
    _obs_write(path, doc)
    with pytest.raises(srp.ShapeRuntimeError, match="checker observation"):
        srp.load_shape_table(path)


def test_expected_panel_digest_matches_authenticated_panel_at_convert_and_load(tmp_path, capsys):
    observation_fixture(tmp_path)
    observation_path = tmp_path / "obs" / "observation.json"
    table = _consume_observations([observation_path], table_id="pilot")
    document = json.loads(observation_path.read_bytes())
    document["expected_panel_sha256"] = "0" * 64
    observation_binding = _obs_write(observation_path, document)
    # Keep the selected source, wrapper and every other artifact binding valid.
    # The domain double now owns the same malformed observation bytes, so the
    # panel comparison, not output authentication, must refuse.
    next(iter(_CHECKER_RESULTS.values()))["payload"] = observation_path.read_bytes()
    receipt_path = tmp_path / "obs" / "checker-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["observation"] = observation_binding
    receipt_binding = _obs_write(receipt_path, receipt)
    table_document = table.as_dict()
    table_document["rows"][0]["measurement"]["receipt_sha256"] = receipt_binding["sha256"]
    table_path = tmp_path / "table.json"
    _obs_write(table_path, table_document)
    output = tmp_path / "refused-table.json"
    assert srp.main(["convert", "--out", str(output), "--table-id", "wrong-panel",
                    "--checker-receipts", str(receipt_path),
                    "--observations", str(observation_path)]) == 2
    assert "expected_panel_sha256" in capsys.readouterr().err
    assert not output.exists()
    with pytest.raises(srp.ShapeRuntimeError, match="expected_panel_sha256"):
        srp.load_shape_table(table_path)


@pytest.mark.parametrize("receipt", [{}, {"schema": "tessera.shape_time_receipt.v999"}, [], None])
@pytest.mark.parametrize("field", ["samples", "key", "lane", "context"])
def test_unknown_receipt_cannot_restore_digest_only_reload(tmp_path, receipt, field):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    document = table.as_dict()
    replacement = _obs_write(tmp_path / "unknown-receipt.json", receipt)
    document["rows"][0]["measurement"].update(
        receipt_path=replacement["path"], receipt_sha256=replacement["sha256"])
    if field == "samples":
        document["rows"][0]["measurement"]["samples_ms"] = [10.0, 20.0, 30.0, 40.0]
    elif field == "key":
        document["rows"][0]["family"] = "TESSERA_E2M1_K2"
    elif field == "lane":
        document["rows"][0]["kernel_lane"]["decoder"] = "other"
    else:
        document["context"]["tensor_parallel"] = 2
    path = tmp_path / "tampered-table.json"
    _obs_write(path, document)
    with pytest.raises(srp.ShapeRuntimeError, match="authenticated PB checker completion"):
        srp.load_shape_table(path)


def test_relative_checker_receipt_does_not_skip_the_row_comparison(tmp_path):
    observation_fixture(tmp_path)
    table = _consume_observations([tmp_path / "obs" / "observation.json"], table_id="pilot")
    doc = table.as_dict()
    doc["rows"][0]["measurement"]["receipt_path"] = "obs/checker-receipt.json"
    doc["rows"][0]["family"] = "TESSERA_E2M1_K2"
    path = tmp_path / "relative-proof-table.json"
    _obs_write(path, doc)
    with pytest.raises(srp.ShapeRuntimeError, match="checker observation"):
        srp.load_shape_table(path)


def test_relative_observation_and_checker_paths_convert(tmp_path, monkeypatch):
    observation_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    table = _consume_observations([Path("obs/observation.json")], table_id="relative")
    assert len(table.rows) == 1
