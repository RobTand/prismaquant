"""Explicit time-constrained PACT selection; CPU fixture prices only."""
import json
import itertools

import pytest

from prismaquant import prefill_frontier
from prismaquant.digests import bytes_sha256hex
from prismaquant.layer_config import LAYER_CONFIG_META_KEY, load_assignment

# test_pact_allocator_replay._fixture builds checker-bound rows, so this module needs the
# same autouse PB reader fixture its replay owner imports.
from test_shape_runtime_prices import checker_sdk_fixture  # noqa: F401


def _unsupported_fixture(tmp_path, monkeypatch, *, units=None):
    import test_pact_allocator_replay as owner

    monkeypatch.setattr(owner, "MENU", {
        owner.SLOW: (0.0, 1.0), owner.MID: (4.0, 1.0), owner.FAST: (5.0, 1.0),
    })
    monkeypatch.setattr(owner, "TIMES", {
        owner.SLOW: 12.0, owner.MID: 8.0, owner.FAST: 4.0,
    })
    case = owner._fixture(tmp_path, monkeypatch, units=units)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({unit: owner.MID for unit in case.units}))
    return owner, case, baseline


def test_actual_pact_constrained_entry_recovers_unsupported_optimum(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    # Existing hull correctly omits the unsupported middle option.
    _, hull = owner._hull(tmp_path, case.argv, "--bootstrap-draws", "20")
    assert [load_assignment(v["assignment_path"])[case.dense] for v in hull["vertices"]] == [
        owner.SLOW, owner.FAST,
    ]
    output = tmp_path / "constrained.json"
    assert prefill_frontier.main([
        "--output", str(output), "--bootstrap-draws", "20", "--", *case.argv,
        "--pact-selection-mode", "constrained", "--pact-baseline-assignment", str(baseline),
        "--pact-baseline-sha256", bytes_sha256hex(baseline.read_bytes()),
    ]) == 0
    document = json.loads(output.read_text())
    assert document["candidate_generator"] == "exact_runtime_frontier_constrained"
    selected = document["points"][0]
    assert load_assignment(selected["assignment_path"]) == {case.dense: owner.MID}
    # The shared signed-quadratic fixture records this binary64 cost, not 4 exactly.
    assert selected["predicted_dloss"] == 4.000000000000001
    assert selected["operator_sum_ms"] == 8.0
    assert "finding_probe" not in selected


def _constrained(case, baseline, output, *extra):
    assert prefill_frontier.main([
        "--output", str(output), "--bootstrap-draws", "20", "--", *case.argv,
        "--pact-selection-mode", "constrained", "--pact-baseline-assignment", str(baseline),
        "--pact-baseline-sha256", bytes_sha256hex(baseline.read_bytes()), *extra,
    ]) == 0
    return json.loads(output.read_text())


def test_unsupported_selected_choice_replays_without_affine_weights(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    frontier = tmp_path / "constrained.json"
    document = _constrained(case, baseline, frontier)
    output = tmp_path / "replayed.json"
    assert owner._replay(frontier, document["selected_assignment_sha256"], output) == 0
    assert load_assignment(output) == {case.dense: owner.MID}
    meta = json.loads(output.read_text())[LAYER_CONFIG_META_KEY]
    stamp = meta["prefill_frontier_replay"]
    assert "probe_weights" not in stamp
    assert stamp["candidate_generator"] == prefill_frontier.CONSTRAINED_GENERATOR
    assert stamp["baseline"]["file_sha256"] == bytes_sha256hex(baseline.read_bytes())
    assert stamp["constraints"]["max_prefill_ms"] == 8.0
    assert meta["research_only"] is True and meta["certifies_placement"] is False


def test_report_ceiling_does_not_change_the_baseline_constraint(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    document = _constrained(case, baseline, tmp_path / "constrained.json",
                            "--pact-time-ceiling-ms", "1")
    assert document["points"][0]["within_time_ceiling"] is False
    assert document["constraints"]["max_prefill_ms"] == 8.0
    assert load_assignment(document["points"][0]["assignment_path"])[case.dense] == owner.MID


def test_real_entry_matches_independent_exhaustive_tiny_optimum(tmp_path, monkeypatch):
    units = tuple(f"model.layers.{i}.mlp.down_proj" for i in range(3))
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch, units=units)
    document = _constrained(case, baseline, tmp_path / "constrained.json")
    losses = {owner.SLOW:0.0, owner.MID:4.000000000000001, owner.FAST:5.000000000000001}
    from prismaquant import format_registry
    sizes = {fmt:format_registry.get_format(fmt).memory_bytes_for_shape((256,256))
             for fmt in losses}
    options = []
    for formats in itertools.product(losses, repeat=3):
        byte_count = sum(sizes[fmt] for fmt in formats)
        milliseconds = sum(owner.TIMES[fmt] for fmt in formats)
        if byte_count <= document["constraints"]["max_memory_bytes"] and milliseconds <= 24:
            options.append((sum(losses[fmt] for fmt in formats), byte_count,
                            milliseconds, formats))
    expected = min(options)
    chosen = load_assignment(document["points"][0]["assignment_path"])
    assert tuple(chosen[unit] for unit in sorted(units)) == expected[3]
    assert document["points"][0]["predicted_dloss"] == expected[0]


def test_whole_artifact_bytes_bind_inside_search_and_replay(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    baseline.write_text(json.dumps({case.dense:owner.SLOW}))
    card = owner._whole_artifact_card(tmp_path, case)
    frontier = tmp_path / "constrained.json"
    document = _constrained(case, baseline, frontier,
                            "--target-disk-gb", card.disk_gb,
                            "--artifact-overhead-reserve-bytes", str(card.reserve))
    assert document["max_memory_bytes"] == card.card-card.reserve-card.floor-card.names
    assert load_assignment(document["points"][0]["assignment_path"])[case.dense] == owner.MID
    assert document["points"][0]["whole_artifact_upper_bound_bytes"] == (
        card.floor+card.names+card.unit[owner.MID]+card.reserve)
    output = tmp_path / "replayed.json"
    assert owner._replay(frontier, document["selected_assignment_sha256"], output) == 0
    budget = json.loads(output.read_text())[LAYER_CONFIG_META_KEY]["whole_artifact_budget"]
    assert budget["budget_bytes"] == card.card
    assert budget["selection_non_tensor_reserve_bytes"] == card.reserve


@pytest.mark.parametrize("mutation,diagnostic", [
    ("missing", "canonical roster"), ("extra", "canonical roster"),
    ("unpriced", "uniquely priced"), ("alias", "duplicate canonical"),
    ("serving_scope", "serving scope"), ("null_scope", "serving scope"),
    ("null_replay", "replay must be an object"), ("replay_m", "regime_m"),
    ("replay_table", "table_identity"), ("replay_tp", "tensor_parallel"),
    ("replay_tp_boolean", "tensor_parallel"), ("replay_scope", "scope"),
])
def test_baseline_population_and_declared_context_refuse(tmp_path, monkeypatch, mutation, diagnostic):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    payload = json.loads(baseline.read_text())
    if mutation == "missing":payload.clear()
    elif mutation == "extra":payload["foreign.unit"] = "BF16"
    elif mutation == "unpriced":payload[case.dense] = "NVFP4"
    elif mutation == "alias":payload[case.dense+".weight"] = owner.FAST
    elif mutation == "serving_scope":payload[LAYER_CONFIG_META_KEY] = {"tessera_serving_scope": {}}
    elif mutation == "null_scope":payload[LAYER_CONFIG_META_KEY] = {"tessera_serving_scope": None}
    elif mutation == "null_replay":payload[LAYER_CONFIG_META_KEY] = {"prefill_frontier_replay": None}
    else:
        key, value = {"replay_m": ("regime_m", owner.M+1),
                      "replay_table": ("table_identity", {}),
                      "replay_tp": ("tensor_parallel", 2),
                      "replay_tp_boolean": ("tensor_parallel", True),
                      "replay_scope": ("scope", {})}[mutation]
        payload[LAYER_CONFIG_META_KEY] = {"prefill_frontier_replay": {key:value}}
    baseline.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=diagnostic):
        _constrained(case, baseline, tmp_path / "refused.json")
    assert not (tmp_path / "refused.json").exists()


def test_baseline_file_digest_refuses_before_runtime_search(tmp_path, monkeypatch):
    _, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    from prismaquant import allocator_solver
    monkeypatch.setattr(allocator_solver, "solve_runtime_frontier",
                        lambda *a, **k: pytest.fail("baseline digest was not checked before solving"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prefill_frontier.main([
            "--output", str(tmp_path / "refused.json"), "--", *case.argv,
            "--pact-selection-mode", "constrained", "--pact-baseline-assignment", str(baseline),
            "--pact-baseline-sha256", "a"*64,
        ])


def test_missing_baseline_time_row_refuses(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    table = json.loads(case.table_path.read_text())
    table["rows"] = [row for row in table["rows"] if row["family"] != owner.MID.rsplit("_R",1)[0]]
    case.table_path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="uniquely priced"):
        _constrained(case, baseline, tmp_path / "refused.json")


@pytest.mark.parametrize("flag,diagnostic", [
    ("--pact-max-states", "max_states=1"),
    ("--pact-max-transitions", "max_transitions=1"),
])
def test_constrained_limits_refuse_without_partial_publication(tmp_path, monkeypatch, flag, diagnostic):
    _, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=diagnostic):
        _constrained(case, baseline, tmp_path / "refused.json", flag, "1")
    assert not (tmp_path / "refused.json").exists()
    assert not (tmp_path / "refused.json.assignments").exists()


@pytest.mark.parametrize("change", ["baseline_file", "bound", "generator", "assignment", "mode",
                                    "time", "loss", "bytes", "top_bound", "semantics", "whole_budget",
                                    "bytes_float", "point_boolean", "baseline_tp_boolean"])
def test_constrained_replay_binding_refuses_before_write(tmp_path, monkeypatch, change):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    frontier = tmp_path / "constrained.json"
    doc = _constrained(case, baseline, frontier)
    if change == "baseline_file":baseline.write_text(json.dumps({case.dense:owner.FAST}))
    elif change == "bound":doc["constraints"]["max_prefill_ms"] = 12.0
    elif change == "generator":doc["candidate_generator"] = "invented"
    elif change == "assignment":doc["selected_assignment_sha256"] = "a"*64
    elif change in ("time", "loss", "bytes"):
        key = {"time":"operator_sum_ms", "loss":"predicted_dloss", "bytes":"candidate_bytes"}[change]
        doc["points"][0][key] += 1
    elif change == "top_bound":doc["max_memory_bytes"] += 1
    elif change == "semantics":doc["numeric_semantics"] = "exact rational objective"
    elif change == "whole_budget":doc["whole_artifact_budget"] = {"budget_bytes":1}
    elif change == "bytes_float":
        doc["points"][0]["candidate_bytes"] = float(doc["points"][0]["candidate_bytes"])
    elif change == "point_boolean":doc["points"][0]["point"] = False
    elif change == "baseline_tp_boolean":doc["baseline"]["tensor_parallel"] = True
    else:
        i = doc["provenance"]["allocator_argv"].index("--pact-selection-mode")
        doc["provenance"]["allocator_argv"][i+1] = "hull"
    frontier.write_text(json.dumps(doc))
    output = tmp_path / "must-not-write.json"
    with pytest.raises(SystemExit) as refused:
        owner._replay(frontier, doc["points"][0]["assignment_sha256"], output)
    assert refused.value.code == 2
    assert not output.exists()


def test_existing_complete_layer_config_is_a_baseline_without_hull_membership(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    frontier = tmp_path / "first.json"
    doc = _constrained(case, baseline, frontier)
    recipe = tmp_path / "layer_config.json"
    assert owner._replay(frontier, doc["selected_assignment_sha256"], recipe) == 0
    second = _constrained(case, recipe, tmp_path / "second.json")
    assert second["baseline"]["file_sha256"] == bytes_sha256hex(recipe.read_bytes())
    assert second["baseline"]["assignment_sha256"] == doc["selected_assignment_sha256"]
    assert second["constraints"]["max_prefill_ms"] == 8.0


def test_declared_matching_baseline_replay_scope_is_accepted(tmp_path, monkeypatch):
    _, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    first = _constrained(case, baseline, tmp_path / "first.json")
    payload = json.loads(baseline.read_text())
    payload[LAYER_CONFIG_META_KEY] = {"prefill_frontier_replay": {"scope":first["scope"]}}
    baseline.write_text(json.dumps(payload))
    second = _constrained(case, baseline, tmp_path / "second.json")
    assert second["constraints"]["max_prefill_ms"] == 8.0


def test_infeasible_bytes_and_time_publish_no_assignment(tmp_path, monkeypatch):
    owner, case, baseline = _unsupported_fixture(tmp_path, monkeypatch)
    baseline.write_text(json.dumps({case.dense:owner.FAST}))
    with pytest.raises(ValueError, match="no feasible assignment"):
        _constrained(case, baseline, tmp_path / "refused.json", "--target-bits", "1")
    assert not (tmp_path / "refused.json").exists()
    assert not (tmp_path / "refused.json.assignments").exists()
