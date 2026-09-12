"""Gates for the quality-prefill manifest, envelopes and phase state machine.

Every test here is a refusal test or an identity test: the contract's job is to
refuse, and a schema that only accepts is not a gate.
"""

import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant.quality_prefill_contract import (
    COVERAGE_TYPES,
    ENVELOPE_SCHEMAS,
    ENVELOPE_VALIDATORS,
    JOINT_CURRENCY,
    LEGAL_TRANSITIONS,
    MANIFEST_SCHEMA,
    MANIFEST_SECTIONS,
    PHASE_DAG,
    PHASE_STATES,
    PRICE_TABLE_SCHEMA,
    RESOLVABLE_INPUTS,
    SCALAR_CURRENCY,
    PhaseState,
    QualityPrefillContractError,
    canonical_json_bytes,
    canonical_sha256,
    freeze_manifest,
    initial_phase_states,
    parse_manifest,
    seal_envelope,
    transition,
    unresolved_inputs,
    validate_manifest,
    validate_price_table,
)


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "experiments" / "tessera_quality_prefill.py"

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_COMMIT = "d" * 40


def _ref(name: str, digest: str = _A) -> dict:
    return {"path": f"/mnt/shared/qp/{name}.json", "sha256": digest}


def draft_manifest() -> dict:
    """A complete draft whose only gaps are the three unresolved inputs."""

    return {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": "glm53-quality-prefill-01",
        "source": {
            "model_content_sha256": _A,
            "config_sha256": _B,
            "tokenizer_sha256": _C,
            "profile": "glm53_flash",
            "tensor_map_sha256": _A,
            "source_closure_sha256": _B,
            "dependency_commits": [{"name": "prismaquant", "commit": _COMMIT}],
        },
        "producer_reader": {
            "producer_commit": _COMMIT,
            "producer_source_sha256": _A,
            "reader_pin": _ref("reader-pin"),
            "packaged_contract_sha256": _B,
            "recipe_resolver_version": "v1.2",
        },
        "data": {
            "calibration_id": "glm-canonical-512x512",
            "calibration_content_sha256": _A,
            "capture_sha256": _B,
            "census_sha256": _C,
            "hessian_sha256": _A,
            "source_execution_selectors": _ref("selectors"),
            "development_dataset": _ref("dev", _B),
            "confirmation_dataset": _ref("confirm", _C),
        },
        "population": {
            "inventory": _ref("inventory"),
            "eligibility_ledger": _ref("eligibility"),
            "pilot_roster": _ref("pilot"),
            "confirmation_roster": _ref("confirmation-roster"),
            "prior_exposure_ledger": _ref("exposure"),
        },
        "menu": {
            "legal_roster": _ref("legal"),
            "recipe_partitions": _ref("recipes"),
            "activation_route_map": _ref("routes"),
            "immutable_region_roster": _ref("immutable"),
            "control_assignments": _ref("controls"),
        },
        "screen": {
            "aqua_settings": _ref("aqua"),
            "transfer_forms": _ref("forms"),
            "audit_draws": _ref("draws"),
            "coverage_quotas": {name: 0 for name in COVERAGE_TYPES},
            "uncertainty_rule": {"name": "bootstrap", "version": "v1"},
            "refinement_policy": {
                "name": "declared_rounds",
                "version": "v1",
                "max_rounds": 1,
            },
        },
        "quality": {
            "joint_probe_count": 8,
            "joint_probe_seed": 0,
            "probe_identity_sha256": _A,
            "calibration_normalization": "global_token_count",
            "execution_microbatch": 1,
            "arithmetic": "fp32",
            "cache_settings": _ref("cache"),
            "source_residency": "resident",
        },
        "runtime": {
            "gpu_identity": {
                "name": "NVIDIA GB10",
                "uuid": "GPU-0000",
                "compute_capability": [12, 1],
            },
            "image_digest": "sha256:" + _A,
            "library_manifest_sha256": _B,
            "workload_manifests": {
                "unresolved": "workload feasibility not confirmed on the pinned engine"
            },
            "numerical_policy": "fp32_reference",
            "timing_policy": {
                "warmup_iterations": 3,
                "measured_iterations": 10,
                "clock": "cuda_event",
            },
            "resource_policy": {
                "partition_producer": "pq420_producer",
                "independent_recomputation": "pq420_consumer",
                "energy_attribution": "netdata_plus_nvml",
            },
        },
        "allocation": {
            "byte_budgets": {"unresolved": "Rob has not selected a deployment byte budget"},
            "prefill_budget_sweep": {"unresolved": "no prefill budget selected"},
            "device_constraints": _ref("device"),
            "solver_limits": {"max_seconds": 600, "max_states": 1000000},
            "tie_policy": "lowest_bytes",
        },
        "validation": {
            "baselines": _ref("baselines"),
            "development_selection_rule": {"name": "empirical_knee", "version": "v1"},
            "confirmation_protocol": {"name": "untouched_confirmation", "version": "v1"},
            "quality_thresholds": _ref("quality-thresholds"),
            "performance_thresholds": _ref("perf-thresholds"),
        },
        "execution": {
            "phases": {
                spec.phase: {
                    "task_roster": _ref("roster-" + spec.phase),
                    "dependencies": list(spec.dependencies),
                    "work_limit": 1,
                    "cost_limit": 1,
                }
                for spec in PHASE_DAG
            },
            "setup_cost_reference": _ref("setup-cost"),
            "pb_policy": {
                "batch_policy": "exact_cover",
                "progress_policy": "progress_v1",
                "priority": -10,
            },
            "retry_policy": {"max_attempts": 2},
            "withdrawal_policy": "release_after_cleanup",
        },
    }


def resolved_manifest() -> dict:
    """The same draft with the three open inputs supplied, for schema tests.

    These values exist only inside this test: the specification forbids the
    repository from carrying an invented production default.
    """

    manifest = draft_manifest()
    manifest["allocation"]["byte_budgets"] = [100_000_000_000]
    manifest["allocation"]["prefill_budget_sweep"] = [2048, 32768]
    manifest["runtime"]["workload_manifests"] = [
        {
            "workload_id": "prefill-2048",
            "prompt_tokens": 2048,
            "batch_size": 1,
            "concurrency": 1,
            "tensor_parallel": 1,
            "graph_mode": "eager",
            "chunked_prefill": False,
            "manifest_sha256": _A,
        }
    ]
    return manifest


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------


def test_a_frozen_manifest_round_trips_to_a_stable_digest():
    frozen = freeze_manifest(resolved_manifest())
    text = canonical_json_bytes(frozen).decode("utf-8")
    reparsed = parse_manifest(text, mode="frozen")
    first = canonical_sha256(reparsed)
    second = canonical_sha256(parse_manifest(canonical_json_bytes(reparsed).decode("utf-8")))
    assert first == second
    assert canonical_json_bytes(reparsed) == canonical_json_bytes(frozen)
    # The seal covers the body and never its own field.
    body = {k: v for k, v in frozen.items() if k != "identity_sha256"}
    assert canonical_sha256(body) == frozen["identity_sha256"]


def test_the_manifest_binds_exactly_the_eleven_named_sections():
    assert len(MANIFEST_SECTIONS) == 11
    frozen = freeze_manifest(resolved_manifest())
    assert set(frozen) == {"schema", "experiment_id", "identity_sha256", *MANIFEST_SECTIONS}


# ---------------------------------------------------------------------------
# 2. Refusals
# ---------------------------------------------------------------------------


def test_an_unknown_key_anywhere_is_a_refusal():
    manifest = resolved_manifest()
    manifest["quality"]["extra_knob"] = 1
    with pytest.raises(QualityPrefillContractError, match="extra="):
        validate_manifest(manifest, mode="draft")


def test_a_duplicate_json_key_is_a_refusal():
    text = json.dumps(freeze_manifest(resolved_manifest()))
    doubled = text.replace('"experiment_id"', '"experiment_id": "x", "experiment_id"', 1)
    with pytest.raises(QualityPrefillContractError, match="duplicate JSON member"):
        parse_manifest(doubled, mode="frozen")


def test_a_nonfinite_number_is_a_refusal_from_text_and_from_memory():
    text = json.dumps(freeze_manifest(resolved_manifest()))
    with pytest.raises(QualityPrefillContractError, match="non-JSON constant"):
        parse_manifest(text.replace('"max_seconds": 600', '"max_seconds": NaN'), mode="frozen")
    table = price_table(SCALAR_CURRENCY)
    table["rows"][0]["value"] = float("nan")
    with pytest.raises(QualityPrefillContractError, match="finite number"):
        validate_price_table(table, where="price_table")


def test_true_does_not_satisfy_an_integer_field():
    manifest = resolved_manifest()
    manifest["quality"]["joint_probe_count"] = True
    with pytest.raises(QualityPrefillContractError, match="joint_probe_count must be an integer"):
        validate_manifest(manifest, mode="draft")


def test_an_unknown_coverage_type_is_a_refusal():
    table = price_table(SCALAR_CURRENCY)
    table["rows"][0]["measurement_status"] = "probably_measured"
    with pytest.raises(QualityPrefillContractError, match="measurement_status must be one of"):
        validate_price_table(table, where="price_table")
    assert len(COVERAGE_TYPES) == 6


def price_table(currency: str) -> dict:
    return {
        "schema": PRICE_TABLE_SCHEMA,
        "currency": currency,
        "units": "dimensionless",
        "context_sha256": _A,
        "rows": [
            {
                "candidate_id": "cand-1",
                "observation_id": "obs-1",
                "currency": currency,
                "units": "dimensionless",
                "measurement_status": "measured",
                "value": 0.5,
                "provenance_sha256": _B,
            }
        ],
    }


def test_one_table_never_carries_both_currencies():
    table = price_table(JOINT_CURRENCY)
    table["rows"].append(
        {
            "candidate_id": "cand-2",
            "observation_id": "obs-2",
            "currency": SCALAR_CURRENCY,
            "units": "dimensionless",
            "measurement_status": "measured",
            "value": 0.25,
            "provenance_sha256": _C,
        }
    )
    with pytest.raises(QualityPrefillContractError, match="one table carries one currency"):
        validate_price_table(table, where="price_table")
    # Each currency alone is fine.
    validate_price_table(price_table(JOINT_CURRENCY), where="price_table")
    validate_price_table(price_table(SCALAR_CURRENCY), where="price_table")


def test_a_joint_row_cannot_be_a_screen():
    table = price_table(JOINT_CURRENCY)
    table["rows"][0]["measurement_status"] = "predicted_screen"
    with pytest.raises(QualityPrefillContractError, match="measurement_status must be"):
        validate_price_table(table, where="price_table")


def test_the_confirmation_dataset_cannot_be_the_development_dataset():
    manifest = resolved_manifest()
    manifest["data"]["confirmation_dataset"] = copy.deepcopy(
        manifest["data"]["development_dataset"]
    )
    with pytest.raises(QualityPrefillContractError, match="must be untouched"):
        validate_manifest(manifest, mode="draft")


def test_a_manifest_may_not_rewire_the_phase_dag():
    manifest = resolved_manifest()
    manifest["execution"]["phases"]["source_and_route_correctness"]["dependencies"] = []
    with pytest.raises(QualityPrefillContractError, match="dependencies must be"):
        validate_manifest(manifest, mode="draft")


# ---------------------------------------------------------------------------
# 3. Freeze names every unresolved value
# ---------------------------------------------------------------------------


def test_freeze_refuses_and_names_all_three_unresolved_inputs():
    with pytest.raises(QualityPrefillContractError) as excinfo:
        freeze_manifest(draft_manifest())
    message = str(excinfo.value)
    assert "cannot freeze" in message
    for path in ("allocation.byte_budgets", "allocation.prefill_budget_sweep",
                 "runtime.workload_manifests"):
        assert path in message, message
    assert "3 required value(s) are unresolved" in message


def test_unresolved_inputs_collects_all_of_them_rather_than_the_first():
    assert unresolved_inputs(draft_manifest()) == list(RESOLVABLE_INPUTS)
    assert unresolved_inputs(resolved_manifest()) == []


def test_a_frozen_manifest_cannot_carry_a_sentinel():
    frozen = freeze_manifest(resolved_manifest())
    frozen["allocation"]["byte_budgets"] = {"unresolved": "smuggled back in"}
    with pytest.raises(QualityPrefillContractError, match="cannot carry unresolved inputs"):
        validate_manifest(frozen, mode="frozen")


def test_a_draft_tolerates_sentinels_so_planning_can_proceed():
    draft = validate_manifest(draft_manifest(), mode="draft")
    assert "identity_sha256" not in draft


# ---------------------------------------------------------------------------
# 4. The phase state machine
# ---------------------------------------------------------------------------


def _attempt(state: str, target: str) -> bool:
    """Return True when the transition is accepted, False when it refuses."""

    current = PhaseState(
        phase=PHASE_DAG[0].phase, state=state, attempt=1, reason="prior"
    )
    try:
        transition(
            current,
            target,
            max_attempts=4,
            receipts=[{"task_id": "t-1"}],
            expected_task_ids=["t-1"],
            reason="named reason",
        )
    except QualityPrefillContractError:
        return False
    return True


def test_every_legal_transition_is_accepted_and_every_other_pair_refuses():
    legal = {
        (source, target)
        for source, targets in LEGAL_TRANSITIONS.items()
        for target in targets
    }
    assert set(LEGAL_TRANSITIONS) == set(PHASE_STATES)
    checked = 0
    for source, target in itertools.product(PHASE_STATES, PHASE_STATES):
        accepted = _attempt(source, target)
        assert accepted == ((source, target) in legal), f"{source} -> {target}"
        checked += 1
    assert checked == len(PHASE_STATES) ** 2 == 100
    # No self-loops, and the three terminal states really are terminal.
    assert not any(source == target for source, target in legal)
    for terminal in ("complete", "withdrawn", "requires_plan_revision"):
        assert LEGAL_TRANSITIONS[terminal] == frozenset()


def test_complete_needs_an_exact_cover_not_a_launch_message():
    current = PhaseState(phase=PHASE_DAG[0].phase, state="collecting", attempt=1, reason="")
    with pytest.raises(QualityPrefillContractError, match="requires its child receipts"):
        transition(current, "complete", max_attempts=2)
    with pytest.raises(QualityPrefillContractError, match="exact cover"):
        transition(
            current,
            "complete",
            max_attempts=2,
            receipts=[{"task_id": "t-1"}],
            expected_task_ids=["t-1", "t-2"],
        )
    done = transition(
        current,
        "complete",
        max_attempts=2,
        receipts=[{"task_id": "t-1"}, {"task_id": "t-2"}],
        expected_task_ids=["t-1", "t-2"],
    )
    assert done.state == "complete"


def test_a_retry_stays_inside_the_original_attempt_budget():
    current = PhaseState(phase=PHASE_DAG[0].phase, state="failed", attempt=2, reason="oom")
    with pytest.raises(QualityPrefillContractError, match="exceeds the attempt budget"):
        transition(current, "ready", max_attempts=2, reason="retry")
    assert transition(current, "ready", max_attempts=3, reason="retry").attempt == 3


def test_requires_plan_revision_must_name_what_is_missing():
    current = PhaseState(phase=PHASE_DAG[0].phase, state="planned", attempt=1, reason="")
    with pytest.raises(QualityPrefillContractError, match="requires a named reason"):
        transition(current, "requires_plan_revision", max_attempts=1)


def test_the_initial_states_cover_the_whole_dag():
    states = initial_phase_states()
    assert set(states) == {spec.phase for spec in PHASE_DAG}
    assert all(item.state == "planned" for item in states.values())


# ---------------------------------------------------------------------------
# 5. The driver
# ---------------------------------------------------------------------------


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DRIVER), *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_help_lists_all_seven_commands_and_exits_zero():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    for command in ("plan", "freeze", "submit", "status", "collect", "solve", "report"):
        assert command in result.stdout, command


def test_the_driver_plans_a_draft_and_refuses_to_freeze_it(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft_manifest()), encoding="utf-8")
    planned = _run("plan", "--manifest", str(path))
    assert planned.returncode == 0, planned.stderr
    payload = json.loads(planned.stdout)
    assert payload["freezable"] is False
    assert payload["unresolved_inputs"] == list(RESOLVABLE_INPUTS)
    assert payload["phase_count"] == len(PHASE_DAG)

    refused = _run("freeze", "--manifest", str(path), "--out", str(tmp_path / "frozen.json"))
    assert refused.returncode == 2
    for name in RESOLVABLE_INPUTS:
        assert name in refused.stderr
    assert not (tmp_path / "frozen.json").exists()


def test_the_driver_freezes_a_resolved_manifest_once(tmp_path):
    path = tmp_path / "resolved.json"
    out = tmp_path / "frozen.json"
    path.write_text(json.dumps(resolved_manifest()), encoding="utf-8")
    first = _run("freeze", "--manifest", str(path), "--out", str(out))
    assert first.returncode == 0, first.stderr
    identity = json.loads(first.stdout)["identity_sha256"]
    assert identity == freeze_manifest(resolved_manifest())["identity_sha256"]
    again = _run("freeze", "--manifest", str(path), "--out", str(out))
    assert again.returncode == 2
    assert "published once" in again.stderr


def test_submit_collect_and_solve_refuse_an_unfrozen_manifest(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft_manifest()), encoding="utf-8")
    for argv in (
        ("submit", "--manifest", str(path), "--phase", PHASE_DAG[0].phase),
        ("collect", "--manifest", str(path), "--phase", PHASE_DAG[0].phase,
         "--receipts", str(path)),
        ("solve", "--manifest", str(path), "--price-table", str(path)),
    ):
        result = _run(*argv)
        assert result.returncode == 2, argv
        assert "REFUSED" in result.stderr


def _frozen_with_roster(tmp_path, digest: str | None = None):
    """Freeze a manifest whose first phase names a roster file on disk."""

    frozen = tmp_path / "frozen.json"
    roster = tmp_path / "roster.json"
    body = '{"tasks": []}'
    roster.write_text(body, encoding="utf-8")
    manifest = resolved_manifest()
    phase = PHASE_DAG[0].phase
    manifest["execution"]["phases"][phase]["task_roster"] = {
        "path": "/" + str(roster).lstrip("/"),
        "sha256": digest or hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    frozen.write_text(json.dumps(freeze_manifest(manifest)), encoding="utf-8")
    return frozen, phase


def test_the_submit_seam_is_unimplemented_and_says_so(tmp_path):
    frozen, phase = _frozen_with_roster(tmp_path)
    result = _run("submit", "--manifest", str(frozen), "--phase", phase)
    assert result.returncode == 3
    assert "separate work package" in result.stderr


def test_submit_refuses_a_roster_whose_bytes_do_not_match_its_reference(tmp_path):
    """A path string is not an identity: the declared digest is checked."""

    frozen, phase = _frozen_with_roster(tmp_path, digest=_A)
    result = _run("submit", "--manifest", str(frozen), "--phase", phase)
    assert result.returncode == 2
    assert "REFUSED" in result.stderr
    assert "sha256" in result.stderr


# ---------------------------------------------------------------------------
# Evidence envelopes (section 3.2)
# ---------------------------------------------------------------------------


def envelope_body(kind: str) -> dict:
    """A minimal valid body for each of the seven envelopes."""

    if kind == "candidate":
        return {
            "candidate_id": "cand-1",
            "unit_map": _ref("unit-map"),
            "family": "tessera_trellis",
            "rate_q256": 1024,
            "recipe": _ref("recipe"),
            "activation_contract": "w4a16",
            "route_status": "backed",
            "source_sha256": _A,
            "footprint_bytes": 4096,
            "support_facts": _ref("support"),
        }
    if kind == "observation":
        return {
            "observation_id": "obs-1",
            "candidate_id": "cand-1",
            "phase_id": PHASE_DAG[0].phase,
            "task_id": "task-1",
            "context_sha256": _A,
            "inner_schema": "unit_kl_v1",
            "currency": SCALAR_CURRENCY,
            "units": "dimensionless",
            "measurement_status": "measured",
            "value": 0.25,
            "artifacts": [_ref("obs-1")],
            "action_key": "f" * 64,
            "attempt": 1,
            "receipt_sha256": _B,
            "validity": "valid",
            "unknown_reason": "",
        }
    if kind == "screen_decision":
        return {
            "decision_id": "dec-1",
            "rule": {"name": "declared_screen", "version": "v1"},
            "dispositions": [
                {
                    "candidate_id": "cand-1",
                    "disposition": "retained",
                    "measurement_status": "retained_pending_measurement",
                    "evidence_observation_ids": ["obs-1"],
                    "reason": "inside the declared screen band",
                    "uncertainty": 0.1,
                }
            ],
        }
    if kind == "transfer_seal":
        return {
            "seal_id": "seal-1",
            "form": {"name": "segment_transfer", "version": "v1"},
            "segment_id": "seg-1",
            "source_candidate_ids": ["cand-1"],
            "endpoint_candidate_ids": ["cand-2"],
            "predictions": [
                {
                    "candidate_id": "cand-3",
                    "predicted_value": 0.5,
                    "currency": SCALAR_CURRENCY,
                    "units": "dimensionless",
                    "measurement_status": "predicted_screen",
                }
            ],
            "prior_exposure_ledger": _ref("exposure"),
            "target_plan_sha256": _C,
        }
    if kind == "phase_result":
        return {
            "phase_id": PHASE_DAG[0].phase,
            "plan_identity_sha256": _A,
            "parent_phase_ids": list(PHASE_DAG[0].dependencies),
            "expected_task_ids": ["task-1"],
            "child_receipts": [
                {"task_id": "task-1", "action_key": "f" * 64, "receipt_sha256": _B}
            ],
            "merged_output_sha256": _C,
            "terminal_state": "complete",
            "coverage": {name: 0 for name in COVERAGE_TYPES},
            "refusal_reasons": [],
        }
    if kind == "assignment":
        return {
            "assignment_id": "assign-1",
            "member_recipes": _ref("members"),
            "serving_unit_recipes": _ref("serving-units"),
            "total_bytes": 1 << 30,
            "price_table": price_table(JOINT_CURRENCY),
            "context_sha256": _A,
            "resource_proposal": _ref("resources"),
            "eligibility": {
                "route_status": "backed",
                "target_platform": "sm121",
                "native": True,
            },
            "exported_artifact_sha256": _B,
        }
    if kind == "frontier_report":
        return {
            "report_id": "report-1",
            "proposed_points": [
                {
                    "point_id": "point-1",
                    "assignment_id": "assign-1",
                    "bytes": 1 << 30,
                    "prefill_budget": 4096,
                    "quality_value": 0.3,
                    "currency": SCALAR_CURRENCY,
                    "units": "dimensionless",
                    "measurement_status": "predicted_screen",
                }
            ],
            "served_points": [
                {
                    "point_id": "point-2",
                    "assignment_id": "assign-1",
                    "bytes": 1 << 30,
                    "prefill_budget": 4096,
                    "quality_value": 0.2,
                    "currency": JOINT_CURRENCY,
                    "units": "dimensionless",
                    "measurement_status": "measured",
                }
            ],
            "uncertainty": _ref("uncertainty"),
            "selected_neighbors": ["point-1"],
            "controls": ["control-1"],
            "omissions": ["the prefill runtime table is not yet acquired"],
            "evidence_links": [_ref("evidence")],
            "commands": ["experiments/tessera_quality_prefill.py report"],
        }
    raise AssertionError(f"no fixture for envelope {kind!r}")


@pytest.mark.parametrize("kind", sorted(ENVELOPE_VALIDATORS))
def test_every_envelope_seals_validates_and_hashes_stably(kind):
    sealed = seal_envelope(kind, envelope_body(kind))
    assert sealed["schema"] == ENVELOPE_SCHEMAS[kind]
    assert sealed["identity_sha256"] == canonical_sha256(
        {k: v for k, v in sealed.items() if k != "identity_sha256"}
    )
    revalidated = ENVELOPE_VALIDATORS[kind](
        json.loads(canonical_json_bytes(sealed).decode("utf-8"))
    )
    assert canonical_sha256(revalidated) == canonical_sha256(sealed)


@pytest.mark.parametrize("kind", sorted(ENVELOPE_VALIDATORS))
def test_every_envelope_refuses_an_unknown_key(kind):
    body = envelope_body(kind)
    body["surprise"] = 1
    with pytest.raises(QualityPrefillContractError):
        seal_envelope(kind, body)


def test_an_unknown_envelope_kind_is_refused():
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("price_table", {})


def test_a_joint_observation_may_not_be_a_screen():
    body = envelope_body("observation")
    body["currency"] = JOINT_CURRENCY
    body["measurement_status"] = "predicted_screen"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("observation", body)


def test_a_sealed_prediction_may_not_claim_a_measurement():
    body = envelope_body("transfer_seal")
    body["predictions"][0]["measurement_status"] = "measured"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("transfer_seal", body)


def test_a_seal_may_not_predict_its_own_endpoint():
    body = envelope_body("transfer_seal")
    body["predictions"][0]["candidate_id"] = "cand-2"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("transfer_seal", body)


def test_complete_needs_an_exact_cover_of_the_expected_tasks():
    body = envelope_body("phase_result")
    body["expected_task_ids"] = ["task-1", "task-2"]
    with pytest.raises(QualityPrefillContractError) as excinfo:
        seal_envelope("phase_result", body)
    assert "task-2" in str(excinfo.value)


def test_an_assignment_may_not_claim_native_on_an_unbacked_route():
    body = envelope_body("assignment")
    body["eligibility"]["route_status"] = "unbacked"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("assignment", body)


def test_a_point_is_never_both_proposed_and_served():
    body = envelope_body("frontier_report")
    body["served_points"][0]["point_id"] = "point-1"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope("frontier_report", body)


def test_collect_walks_the_legal_chain_and_publishes_the_advanced_state(tmp_path):
    """A submitted phase reaches 'complete' only via running -> collecting."""

    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(freeze_manifest(resolved_manifest())), encoding="utf-8"
    )
    phase = PHASE_DAG[0].phase
    state_in = tmp_path / "state.json"
    states = {
        name: {"state": "planned", "attempt": 1, "reason": ""}
        for name in initial_phase_states()
    }
    states[phase] = {"state": "submitted", "attempt": 1, "reason": ""}
    state_in.write_text(json.dumps(states), encoding="utf-8")

    receipts = tmp_path / "receipts.json"
    receipts.write_text(
        json.dumps(
            {
                "task_ids": ["task-1"],
                "receipts": [
                    {"task_id": "task-1", "action_key": "f" * 64, "receipt_sha256": _B}
                ],
            }
        ),
        encoding="utf-8",
    )
    state_out = tmp_path / "state-out.json"
    result = _run(
        "collect",
        "--manifest", str(frozen),
        "--phase", phase,
        "--state", str(state_in),
        "--receipts", str(receipts),
        "--state-out", str(state_out),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(state_out.read_text())[phase]["state"] == "complete"


def test_collect_refuses_a_phase_that_was_never_submitted(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(freeze_manifest(resolved_manifest())), encoding="utf-8"
    )
    receipts = tmp_path / "receipts.json"
    receipts.write_text(
        json.dumps(
            {
                "task_ids": ["task-1"],
                "receipts": [
                    {"task_id": "task-1", "action_key": "f" * 64, "receipt_sha256": _B}
                ],
            }
        ),
        encoding="utf-8",
    )
    result = _run(
        "collect",
        "--manifest", str(frozen),
        "--phase", PHASE_DAG[0].phase,
        "--receipts", str(receipts),
    )
    assert result.returncode == 2
    assert "not a legal transition" in result.stderr


def test_a_state_file_may_not_carry_an_unknown_phase_state(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(freeze_manifest(resolved_manifest())), encoding="utf-8"
    )
    state_in = tmp_path / "state.json"
    states = {
        name: {"state": "planned", "attempt": 1, "reason": ""}
        for name in initial_phase_states()
    }
    states[PHASE_DAG[0].phase]["state"] = "nearly_done"
    state_in.write_text(json.dumps(states), encoding="utf-8")
    result = _run(
        "status", "--manifest", str(frozen), "--state", str(state_in)
    )
    assert result.returncode == 2
    assert "unknown state" in result.stderr


@pytest.mark.parametrize("kind", sorted(ENVELOPE_VALIDATORS))
def test_an_envelope_may_not_declare_another_envelopes_schema(kind):
    body = envelope_body(kind)
    body["schema"] = "prismaquant.quality_prefill_experiment.not_this.v1"
    with pytest.raises(QualityPrefillContractError):
        seal_envelope(kind, body)
