"""The adapter's own half: what it emits, what it refuses, and what it will not do.

Nothing here touches PrismaBuild.  These are the properties that have to hold
before a request is worth submitting at all -- that one frozen plan emits one
set of bytes, that a plan missing a measured cost is refused rather than
defaulted, that a child cannot answer for a task outside its own batch, and that
a missing fleet capability is reported by name instead of routed around.

The other half -- that PrismaBuild reads what this emits, cuts it into the
children we expect, and closes the group on an exact cover -- is in
``test_quality_prefill_pb_adapter_serve_once.py``, which runs in process against
a checkout of PB #518.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prismaquant import quality_prefill_pb_adapter as adapter


EVIDENCE = "cas:sha256:" + "1e" * 32
OTHER_EVIDENCE = "cas:sha256:" + "2f" * 32
PLAN_DIGEST = "3a" * 32

#: 66 items at 8.2 s under a 45 s setup, a 0.20 setup fraction and a 300 s wall.
#: Rearranged, those limits say each batch must carry at least
#: 45 * (1 - 0.2) / 0.2 = 180 s of useful work and at most 300 - 45 = 255 s, so a
#: batch holds between 22 and 31 items and the most batches 66 items admit is
#: three of 22.  That arithmetic is PrismaBuild's, and this roster exists so the
#: expected child count is a number rather than "however many it made".
ROSTER_SIZE = 66
EXPECTED_CHILDREN = 3
ITEM_SECONDS = 8.2
SETUP_SECONDS = 45.0


def work_item(index: int, *, residency: str = "glm53.prefill") -> dict:
    """One unit/rate work item, in the shape the emitter copies into a roster."""

    return {
        "task_id": f"qp/phase-a/t{index:04d}",
        "output_id": f"qp/phase-a/out{index:04d}",
        "residency_key": residency,
        "estimated_seconds": ITEM_SECONDS,
        "estimate_evidence": EVIDENCE,
        "payload": {
            "schema": adapter.TASK_PAYLOAD_SCHEMA,
            "serving_unit": f"model.layers.{index // 8}.mlp.gate_up_proj",
            "members": [
                f"model.layers.{index // 8}.mlp.gate_proj",
                f"model.layers.{index // 8}.mlp.up_proj",
            ],
            "family": "e4m3",
            "rate": 832 + index,
            "context_id": "prefill-2048",
            "currency": "joint_aura_predicted_dloss",
        },
    }


def phase_plan(
    *,
    cwd: str = "/home/rob/prismaquant",
    interpreter: str = "/home/rob/venvs/pq-cu130/bin/python",
    # The measurement producer is another work package; a phase plan names it
    # and this adapter never imports it.  The serve-once harness substitutes a
    # producer of its own here.
    module: str = "experiments.tessera_quality_prefill_producer",
    env: dict | None = None,
    size: int = ROSTER_SIZE,
) -> dict:
    """A frozen phase plan whose cut is known in advance."""

    return {
        "schema": adapter.PHASE_PLAN_SCHEMA,
        "phase_id": "qp/phase-a",
        "experiment_plan_sha256": PLAN_DIGEST,
        "command": {"interpreter": interpreter, "module": module},
        "cwd": cwd,
        "env": dict(env or {}),
        "demand": {"cpu": 1, "mem_gb": 4},
        "gpu_memory_gb": None,
        "data_manifest": None,
        "residencies": [
            {
                "key": "glm53.prefill",
                "setup_seconds": SETUP_SECONDS,
                "setup_evidence": OTHER_EVIDENCE,
            }
        ],
        "batch_limits": {
            "max_setup_fraction": 0.20,
            "max_estimated_wall_seconds": 300.0,
        },
        "work_items": [work_item(index) for index in range(size)],
    }


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_one_frozen_plan_emits_byte_identical_requests() -> None:
    """Acceptance criterion 1.

    A resumed campaign lands on the parent key these bytes hash to, so bytes
    that vary between two runs of one plan are two parents for one phase.
    """

    plan = phase_plan()
    assert adapter.logical_request_bytes(plan) == adapter.logical_request_bytes(plan)


def test_key_order_and_repeated_emission_do_not_move_the_bytes() -> None:
    """The same statement, written differently, is the same request.

    JSON object order carries no meaning, so a plan whose sections were built in
    another order must not emit a different parent.  Emitting from a deep copy
    as well shows the emitter did not mutate the plan it was handed.
    """

    plan = phase_plan()
    shuffled = {key: copy.deepcopy(plan[key]) for key in reversed(list(plan))}
    shuffled["work_items"] = [
        {key: copy.deepcopy(item[key]) for key in reversed(list(item))}
        for item in plan["work_items"]
    ]
    first = adapter.logical_request_bytes(plan)
    assert adapter.logical_request_bytes(shuffled) == first
    assert adapter.logical_request_bytes(copy.deepcopy(plan)) == first


def test_emitting_reads_nothing_from_the_ambient_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Determinism has to survive being emitted on another box.

    Moving the interpreter, the working directory and the hostname out from
    under the emitter changes the bytes only if it read one of them; it reads
    the plan.
    """

    plan = phase_plan()
    before = adapter.logical_request_bytes(plan)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.executable", "/some/other/python")
    monkeypatch.setenv("HOSTNAME", "not-sparky")
    assert adapter.logical_request_bytes(plan) == before


def test_the_emitted_request_is_the_plan_one_for_one() -> None:
    """The roster is the phase's work items; the emitter adds no opinion."""

    plan = phase_plan()
    request = adapter.emit_logical_request(plan)
    assert request["schema"] == adapter.LOGICAL_REQUEST_SCHEMA
    assert request["common"]["argv"] == [
        plan["command"]["interpreter"], "-m", plan["command"]["module"],
        adapter.TASK_BATCH_PLACEHOLDER,
    ]
    assert request["common"]["cwd"] == plan["cwd"]
    assert [task["id"] for task in request["roster"]["tasks"]] == [
        item["task_id"] for item in plan["work_items"]
    ]
    assert request["batch_policy"]["residencies"] == plan["residencies"]
    assert (request["batch_policy"]["max_setup_fraction"]
            == plan["batch_limits"]["max_setup_fraction"])
    assert (request["batch_policy"]["max_estimated_wall_seconds"]
            == plan["batch_limits"]["max_estimated_wall_seconds"])


def test_the_batch_placeholder_is_one_whole_argument_in_the_emitted_command() -> None:
    """Part of acceptance criterion 5, on the producing side.

    PrismaBuild refuses a command that embeds the placeholder in a larger
    argument, and an emitter that produced one would be caught at submission.
    It is asserted here so a change to the command shape fails in PrismaQuant's
    own suite rather than at the fleet boundary.
    """

    argv = adapter.emit_logical_request(phase_plan())["common"]["argv"]
    assert argv.count(adapter.TASK_BATCH_PLACEHOLDER) == 1
    assert not [
        part for part in argv
        if adapter.TASK_BATCH_PLACEHOLDER in part
        and part != adapter.TASK_BATCH_PLACEHOLDER
    ]


def test_write_logical_request_publishes_exactly_the_emitted_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "request.json"
    digest = adapter.write_logical_request(phase_plan(), path)
    assert path.read_bytes() == adapter.logical_request_bytes(phase_plan())
    assert digest == adapter.document_sha256(
        adapter.emit_logical_request(phase_plan()))
    # A request file is JSON a campaign tool loads, not only bytes we hash.
    assert json.loads(path.read_text())["schema"] == adapter.LOGICAL_REQUEST_SCHEMA


# --------------------------------------------------------------------------
# The plan is refused rather than repaired
# --------------------------------------------------------------------------

def test_an_unknown_field_is_refused_rather_than_dropped() -> None:
    """A field we do not read is a field the author believes is binding."""

    plan = phase_plan()
    plan["retry_policy"] = {"attempts": 3}
    with pytest.raises(adapter.QualityPrefillAdapterError, match="extra="):
        adapter.emit_logical_request(plan)


@pytest.mark.parametrize("section, field", [
    ("batch_limits", "max_setup_fraction"),
    ("batch_limits", "max_estimated_wall_seconds"),
])
def test_a_missing_batch_limit_is_never_defaulted(section: str, field: str) -> None:
    """How much of a GPU hour may go to setup is a measurement, not a default."""

    plan = phase_plan()
    del plan[section][field]
    with pytest.raises(adapter.QualityPrefillAdapterError, match="missing="):
        adapter.emit_logical_request(plan)


def test_evidence_without_a_digest_is_not_evidence() -> None:
    """Specification §3.2: path strings are not identities."""

    plan = phase_plan()
    plan["work_items"][0]["estimate_evidence"] = "measured on sparky last week"
    with pytest.raises(adapter.QualityPrefillAdapterError,
                       match="estimate_evidence"):
        adapter.emit_logical_request(plan)


def test_a_non_finite_estimate_is_refused() -> None:
    plan = phase_plan()
    plan["work_items"][0]["estimated_seconds"] = float("inf")
    with pytest.raises(adapter.QualityPrefillAdapterError, match="finite"):
        adapter.emit_logical_request(plan)


def test_a_boolean_does_not_satisfy_an_integer_field() -> None:
    """Specification §3.1 names this one explicitly."""

    plan = phase_plan()
    plan["demand"]["cpu"] = True
    with pytest.raises(adapter.QualityPrefillAdapterError, match="demand"):
        adapter.emit_logical_request(plan)


def test_a_repeated_task_id_is_refused_before_a_parent_exists() -> None:
    """Two rows under one id cannot both be answered once."""

    plan = phase_plan()
    plan["work_items"][5]["task_id"] = plan["work_items"][4]["task_id"]
    with pytest.raises(adapter.QualityPrefillAdapterError, match="repeats"):
        adapter.emit_logical_request(plan)


def test_an_unpriced_residency_is_refused() -> None:
    plan = phase_plan()
    plan["work_items"][0]["residency_key"] = "glm53.decode"
    with pytest.raises(adapter.QualityPrefillAdapterError, match="not priced"):
        adapter.emit_logical_request(plan)


def test_an_identifier_prismabuild_would_refuse_is_refused_here() -> None:
    """The grammar is PrismaBuild's, because the id reaches a sealed action key.

    ``@`` is legal in PrismaQuant's own campaign ids and illegal in
    PrismaBuild's, so validating against the wrong one would freeze a plan the
    fleet then refuses.
    """

    plan = phase_plan()
    plan["work_items"][0]["task_id"] = "qp/phase-a/unit@832"
    with pytest.raises(adapter.QualityPrefillAdapterError, match="task_id"):
        adapter.emit_logical_request(plan)


def test_a_currency_outside_the_specifications_two_is_refused() -> None:
    """A scalar screen and a measured joint price are different claims."""

    plan = phase_plan()
    plan["work_items"][0]["payload"]["currency"] = "held_out_kl"
    with pytest.raises(adapter.QualityPrefillAdapterError, match="currency"):
        adapter.emit_logical_request(plan)


def test_a_member_listed_twice_in_one_unit_is_refused() -> None:
    """Specification §2.2: every mutable member occurs exactly once."""

    plan = phase_plan()
    payload = plan["work_items"][0]["payload"]
    payload["members"] = [payload["members"][0], payload["members"][0]]
    with pytest.raises(adapter.QualityPrefillAdapterError, match="repeats"):
        adapter.emit_logical_request(plan)


# --------------------------------------------------------------------------
# The capability gap is named, not filled
# --------------------------------------------------------------------------

def _runtime_tree(root: Path, *, decomposition: bool, entry: bool) -> Path:
    (root / "src" / "prismabuild").mkdir(parents=True)
    (root / "tools" / "fleet").mkdir(parents=True)
    if decomposition:
        (root / "src" / "prismabuild" / "decomposition.py").write_text("x = 1\n")
    body = "def submit_row(row):\n    return row\n"
    if entry:
        body += "def decompose(request, *, transport, priority):\n    return []\n"
    (root / "tools" / "fleet" / "pbcampaign.py").write_text(body)
    return root


def test_a_runtime_without_decomposition_is_reported_by_name(tmp_path: Path) -> None:
    """Acceptance criterion 6.

    The refusal has to name what is missing and what would fix it, because the
    fix is a merge and a runtime publication that no agent performs.
    """

    root = _runtime_tree(tmp_path / "old", decomposition=False, entry=False)
    support = adapter.decomposition_support(root)
    assert support["supported"] is False
    with pytest.raises(adapter.DecompositionUnavailable) as refusal:
        adapter.require_decomposition_support(root)
    message = str(refusal.value)
    assert "#517" in message and "#518" in message
    assert "runtime generation" in message and "published" in message
    assert "decomposition.py" in message


def test_a_merged_module_without_a_campaign_entry_is_still_unsupported(
    tmp_path: Path,
) -> None:
    """Half a capability is not a capability.

    A runtime carrying the module but no ``pbcampaign.decompose`` cannot publish
    a plan's children, and a probe that answered "supported" there would send a
    campaign at a fleet that drops it.
    """

    root = _runtime_tree(tmp_path / "half", decomposition=True, entry=False)
    assert adapter.decomposition_support(root)["supported"] is False
    with pytest.raises(adapter.DecompositionUnavailable, match="pbcampaign"):
        adapter.require_decomposition_support(root)


def test_a_runtime_with_both_halves_is_accepted(tmp_path: Path) -> None:
    root = _runtime_tree(tmp_path / "new", decomposition=True, entry=True)
    assert adapter.require_decomposition_support(root)["supported"] is True
    request = adapter.prepare_logical_request(phase_plan(), runtime_root=root)
    assert request == adapter.emit_logical_request(phase_plan())


def test_the_submission_boundary_refuses_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    """Acceptance criterion 6, the part that matters.

    The failure mode this guards against is not a crash -- it is a helpful
    adapter that notices the fleet cannot decompose and submits the phase as one
    long opaque action, or cuts the roster itself.  Either would make
    PrismaQuant a second scheduler, which is what PB #517 exists to remove.
    """

    root = _runtime_tree(tmp_path / "old", decomposition=False, entry=False)
    with pytest.raises(adapter.DecompositionUnavailable):
        adapter.prepare_logical_request(phase_plan(), runtime_root=root)

    source = Path(adapter.__file__).read_text(encoding="utf-8")
    # No rescue of the refusal anywhere in the module: a caught
    # DecompositionUnavailable is the shape a fallback would have.
    assert "except DecompositionUnavailable" not in source
    assert "DecompositionUnavailable" in source


def test_the_module_records_the_deployment_dependency() -> None:
    """Acceptance criterion 6's doc half.

    A reader who finds this module before finding the pull request needs to know
    from the file itself that it targets something the fleet does not run yet.
    """

    doc = adapter.__doc__ or ""
    assert "#518" in doc and "#517" in doc
    assert "not deployed to the fleet" in doc
    assert "published" in doc
    assert "in process" in doc
    assert "no fallback" in doc.lower()


def test_the_deployed_runtime_is_reported_rather_than_asserted() -> None:
    """What the fleet carries today is a measurement, so it is measured.

    This asserts the probe answers about the published runtime without asserting
    which answer it gives: the answer flips the day a generation carrying #518
    is published, and a test that pinned "unsupported" would then fail for the
    good reason.
    """

    support = adapter.decomposition_support()
    assert set(support) == {
        "runtime_root", "decomposition_module", "campaign_decompose_entry",
        "supported",
    }
    assert support["runtime_root"] == str(adapter.DEPLOYED_RUNTIME_ROOT)
    assert isinstance(support["supported"], bool)


# --------------------------------------------------------------------------
# The producer contract, on the child's side
# --------------------------------------------------------------------------

def batch_envelope(*, ordinal: int = 0, count: int = 3) -> dict:
    """A batch envelope of the shape PrismaBuild hands a child."""

    items = [work_item(index) for index in range(count)]
    return {
        "schema": adapter.BATCH_ENVELOPE_SCHEMA,
        "parent_key": "aa" * 32,
        "plan_key": "bb" * 32,
        "roster_sha256": "cc" * 32,
        "batch_policy_sha256": "dd" * 32,
        "child_ordinal": ordinal,
        "result_manifest_path": f"pb-child-{ordinal:05d}.result-manifest.json",
        "tasks": [
            {
                "id": item["task_id"],
                "payload": item["payload"],
                "residency_key": item["residency_key"],
                "estimated_seconds": item["estimated_seconds"],
                "estimate_evidence": item["estimate_evidence"],
                "output_id": item["output_id"],
            }
            for item in items
        ],
    }


def test_a_child_writes_its_manifest_where_its_action_declared_it(
    tmp_path: Path,
) -> None:
    envelope = batch_envelope()
    results = {task["id"]: {"ms": 1.5, "n": 8} for task in envelope["tasks"]}
    path = adapter.write_child_result_manifest(envelope, results, root=tmp_path)

    assert path == tmp_path / envelope["result_manifest_path"]
    manifest = adapter.validate_child_result_manifest(
        adapter.load_strict_json(path.read_bytes(), where="manifest"))
    assert manifest["schema"] == adapter.CHILD_RESULT_MANIFEST_SCHEMA
    assert manifest["parent_key"] == envelope["parent_key"]
    assert manifest["child_ordinal"] == envelope["child_ordinal"]
    assert [entry["task_id"] for entry in manifest["results"]] == [
        task["id"] for task in envelope["tasks"]
    ]
    # The output id is the roster's, never the producer's choice.
    assert [entry["output_id"] for entry in manifest["results"]] == [
        task["output_id"] for task in envelope["tasks"]
    ]
    assert manifest["results"][0]["value_sha256"] == adapter.document_sha256(
        {"ms": 1.5, "n": 8})


def test_two_runs_of_one_child_publish_identical_manifest_bytes(
    tmp_path: Path,
) -> None:
    """Nothing in a manifest may vary, or a retry disagrees with its own result."""

    envelope = batch_envelope()
    results = {task["id"]: {"ms": 1.5} for task in envelope["tasks"]}
    first = adapter.build_child_result_manifest(envelope, results)
    reordered = {key: results[key] for key in reversed(list(results))}
    assert adapter.document_bytes(
        adapter.build_child_result_manifest(envelope, reordered)
    ) == adapter.document_bytes(first)


def test_a_child_cannot_answer_for_a_task_outside_its_own_batch() -> None:
    """The producer half of acceptance criterion 4.

    PrismaBuild's exact cover catches this at the merge; catching it here means
    the child fails where it ran, naming the task, instead of a campaign ending
    with an open cover and no local explanation.
    """

    envelope = batch_envelope()
    results = {task["id"]: {"ms": 1.0} for task in envelope["tasks"]}
    results["qp/phase-a/t0099"] = {"ms": 1.0}
    with pytest.raises(adapter.QualityPrefillAdapterError,
                       match="outside its own batch"):
        adapter.build_child_result_manifest(envelope, results)


def test_a_child_that_skipped_one_of_its_tasks_refuses_to_publish() -> None:
    """A batch that passed while answering nothing is the failure to prevent."""

    envelope = batch_envelope()
    results = {task["id"]: {"ms": 1.0} for task in envelope["tasks"][:-1]}
    with pytest.raises(adapter.QualityPrefillAdapterError, match="no result for"):
        adapter.build_child_result_manifest(envelope, results)


def test_a_result_that_is_not_canonical_json_is_refused() -> None:
    envelope = batch_envelope()
    results = {task["id"]: {"ms": float("nan")} for task in envelope["tasks"]}
    with pytest.raises(adapter.QualityPrefillAdapterError, match="canonical JSON"):
        adapter.build_child_result_manifest(envelope, results)


def test_reading_an_envelope_refuses_a_duplicate_json_key(tmp_path: Path) -> None:
    """A repeated key is two statements under one name; ``json`` keeps the last."""

    path = tmp_path / "batch.json"
    path.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(adapter.QualityPrefillAdapterError, match="repeats the JSON key"):
        adapter.read_batch_envelope(path)


def test_reading_an_envelope_refuses_a_non_finite_number(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text('{"estimated_seconds": Infinity}', encoding="utf-8")
    with pytest.raises(adapter.QualityPrefillAdapterError, match="non-finite"):
        adapter.read_batch_envelope(path)


def test_an_envelope_whose_result_path_escapes_the_checkout_is_refused(
    tmp_path: Path,
) -> None:
    """Every child gets its own private checkout; a manifest never leaves it."""

    envelope = batch_envelope()
    envelope["result_manifest_path"] = "../pb-child-00000.result-manifest.json"
    with pytest.raises(adapter.QualityPrefillAdapterError,
                       match="result_manifest_path"):
        adapter.validate_batch_envelope(envelope)


def test_an_envelope_round_trips_through_the_reader(tmp_path: Path) -> None:
    envelope = batch_envelope()
    path = tmp_path / "batch.json"
    path.write_bytes(adapter.document_bytes(envelope))
    assert adapter.read_batch_envelope(path) == adapter.validate_batch_envelope(
        envelope)
