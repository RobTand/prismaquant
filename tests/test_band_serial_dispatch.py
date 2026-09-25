"""Band-serial Stage B rows (PQ #996): the derived readset and the dispatch edge.

Four bound executable records on a two-band fixture (stride 2, checkpoints
2 and 4): band 4 holds layers 3 and 2, band 2 holds layers 1 and 0. Layer 3
hands its cotangent to layer 2, and layer 1 to layer 0.

PrismaBuild has no dependency between actions, so the edge is publication
order: a producer row declares its handoff as a produced output, and its
consumer is published only after the producer executed, reported complete
and published a handoff the consumer binds. The consumer stages a readset
derived from its sealed chain manifest and that handoff, and the quantum
re-derives the same bytes before it reads.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA  # noqa: E402
from prismaquant.joint_adjoint_checkpoints import (  # noqa: E402
    exact_entry_record, write_adjoint_checkpoint)
from prismaquant.joint_quantum_handoff import (  # noqa: E402
    BAND_SERIAL_READSET_SCHEMA,
    HANDOFF_LOAD_PHASE,
    HandoffEmitter,
    QuantumHandoffRefused,
    band_serial_manifest,
    band_serial_manifest_bytes,
    handoff_read_entries,
    load_quantum_handoff,
    require_band_serial_readset,
)
from prismaquant.perturbed_x_cache import (  # noqa: E402
    write_exact_activation_cache_entry)
from test_quantum_executable_readset import (  # noqa: E402
    CALIB, N_BATCHES, N_PROBES, RENDER_PREREQ, SESSION, STRIDED,
    _bind_slice, _tiny_records,
)
from test_stageb_prepared_render_inputs import (  # noqa: E402
    _prepared_inputs, _render_files)

TIER = "prismabuild-stage:dl380g10"
ARTIFACT_MAX = 1 << 24
PREFETCH = 2


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Run outside campaign scope with no staged-tier policy (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _receipt(tmp_path, campaign, *, run_identity_extra=None):
    """``_tiny_receipt`` with boundary 0 too, so band 2 binds both layers."""
    space = tmp_path / "adjoint"
    (space / "entries").mkdir(parents=True, exist_ok=True)
    boundary_entries = {}
    for boundary in (0, 1, 2, 3):
        rows = []
        for batch in range(N_BATCHES):
            reference = write_exact_activation_cache_entry(
                space / "entries", f"boundary-{batch}-{boundary}-at-{boundary}",
                torch.zeros(2, 4),
                identity={"session": dict(SESSION),
                          "slot": f"boundary-{batch}-{boundary}",
                          "kind": "boundary",
                          "coordinates": {"batch": batch, "boundary": boundary,
                                          "probe": None}},
                max_tensor_bytes=1 << 20, max_file_bytes=1 << 20)
            rows.append(exact_entry_record(reference))
        boundary_entries[str(boundary)] = rows
    checkpoints = [write_adjoint_checkpoint(
        space, boundary=boundary,
        session={"generation": SESSION["generation"],
                 "kind": "adjoint_checkpoint",
                 "run_identity_sha256": SESSION["run_identity_sha256"]},
        cotangents={(p, b): torch.zeros(2, 4) for p in range(N_PROBES)
                    for b in range(N_BATCHES)},
        shared_adjoint={(p, b): {"scale": 1.0} for p in range(N_PROBES)
                        for b in range(N_BATCHES)},
        shared_pass={b: {"mask": [0, 1]} for b in range(N_BATCHES)})
        for boundary in STRIDED]
    return {
        "schema": jl.ADJOINT_CAPTURE_SCHEMA,
        "run_identity": {"plan_sha256": campaign["plan_sha256"],
                         "prepared_sha256": campaign["prepared_sha256"],
                         "campaign_scope": campaign["campaign_scope"],
                         **(run_identity_extra or {})},
        "stride": {"value": 2, "source": None,
                   "boundaries": sorted(STRIDED, reverse=True),
                   "max_chain_layers": 1},
        "boundary_storage": {"session": dict(SESSION),
                             "policy": {"prefetch_batches": PREFETCH},
                             "directory": str(space / "entries")},
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "status": "complete",
    }


def _campaign(tmp_path, *, run_identity_extra=None):
    """Bound executable records for layers 0-3, with prepared inputs."""
    root = str(tmp_path / "run")
    records, parent = _tiny_records(tmp_path)
    receipt = _receipt(tmp_path, records[0]["campaign"],
                       run_identity_extra=run_identity_extra)
    files = _render_files(tmp_path)
    bound = {}
    production = None
    for record in records:
        record, _digest = _bind_slice(record, receipt, tmp_path / "adjoint-slices")
        if production is None:
            # Production capture files seal the campaign digests top-level
            # too; receipt-only fields, so the slice digest is unchanged.
            production = copy.deepcopy(receipt)
            production["plan_sha256"] = record["campaign"]["plan_sha256"]
            production["prepared_sha256"] = record["campaign"]["prepared_sha256"]
        prepared = _prepared_inputs(record, files)
        manifest = jl.build_quantum_executable_manifest(
            record, production, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ), prepared_inputs=prepared)
        wire = jl.seal_manifest_bytes(manifest)
        path = Path(jl.bound_readset_directory(root)) / (
            f"{record['quantum_id']}.executable.json.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wire)
        bound[record["layer"]] = jl.bind_quantum_executable(
            record, production, parent, manifest=manifest,
            manifest_path=str(path),
            manifest_sha256=hashlib.sha256(wire).hexdigest(), output_root=root,
            strided_boundaries=STRIDED, n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ), prepared_inputs=prepared)
    return bound, production


def _slice(record):
    return json.loads(Path(record["adjoint"]["slice_path"]).read_text())


def _sealed(record):
    block = record["executable_readset"]
    return json.loads(gzip.decompress(Path(block["manifest_path"]).read_bytes()))


class _Owner:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)


def _emit(record, *, shift=0.0, kernel=None):
    """Publish ``record``'s handoff as its final pass would (no PB owner).

    ``kernel`` names the KDA capture kernel of a kernel-mode producer
    (PQ #1214); its handoff then carries the kernel's name and identity.
    """
    emitter = HandoffEmitter(
        record=record, adjoint_slice=_slice(record),
        boundary_storage={"schema": BOUNDARY_STORAGE_SCHEMA,
                          "directory": "/unused", "max_resident_bytes": 1 << 24,
                          "max_auxiliary_bytes": 1 << 24,
                          "max_artifact_bytes": ARTIFACT_MAX,
                          "prefetch_batches": PREFETCH},
        capture_batch=1)
    plane = {(p, b): torch.full((2, 4), 10.0 * p + b + shift)
             for p in range(N_PROBES) for b in range(N_BATCHES)}
    owners = [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
              for p in range(N_PROBES)]
    stamp = (None if kernel is None
             else {"name": kernel, "identity_sha256": "k" * 64})
    return emitter.emit(grad_plane=plane, cotangent_owners=owners,
                        n_probes=N_PROBES, n_batches=N_BATCHES,
                        **({} if stamp is None else {"kda_capture_kernel": stamp}))


def _complete(record, published):
    """Write the producer's results and its complete status, as it would."""
    space = Path(record["output_space"]["root"])
    space.mkdir(parents=True, exist_ok=True)
    Path(record["output_space"]["results"]).write_text(json.dumps(
        {"quantum_id": record["quantum_id"], "passed": True,
         **({"handoff": published} if published is not None else {})}))
    (space / "status.json").write_text(json.dumps(
        {"quantum_id": record["quantum_id"],
         "identity_sha256": record["identity_sha256"], "status": "complete"}))


# -- the derived readset -----------------------------------------------------

def test_the_band_serial_readset_swaps_the_chain_for_the_handoff(tmp_path):
    bound, receipt = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer, adjoint_slice=_slice(consumer))
    sealed = _sealed(consumer)
    derived = band_serial_manifest(
        sealed, handoff, _slice(consumer)["checkpoint"],
        sealed_manifest_sha256=consumer["executable_readset"]["manifest_sha256"])

    names = [phase["name"] for phase in derived["read_plan"]["phases"]]
    sealed_names = [phase["name"] for phase in sealed["read_plan"]["phases"]]
    assert sealed_names[:4] == ["head", "checkpoint-load", "chain-003-source",
                                "chain-003-bound"]
    assert names == ["head", HANDOFF_LOAD_PHASE, *sealed_names[4:]]

    def rows(manifest, name):
        phase = next(p for p in manifest["read_plan"]["phases"] if p["name"] == name)
        return [manifest["entries"][index] for index in phase["entry_indices"]]

    # handoff-load stages exactly what load_handoff_inputs reads, in order.
    assert rows(derived, HANDOFF_LOAD_PHASE) == handoff_read_entries(
        handoff, _slice(consumer)["checkpoint"])
    # Every kept phase stages the sealed bytes, unchanged.
    for name in ["head", *sealed_names[4:]]:
        assert rows(derived, name) == rows(sealed, name), name
    # No checkpoint cotangent and no chain boundary is staged any more.
    staged = {entry["path"] for entry in derived["entries"]}
    checkpoint_plane = {entry["path"] for entry in
                        _slice(consumer)["checkpoint"]["activation_entries"]}
    assert not staged & checkpoint_plane
    assert not staged & {entry["path"]
                         for entry in receipt["boundary_entries"]["3"]}
    assert staged >= {entry["path"] for entry in receipt["boundary_entries"]["2"]}
    # Prepared windows point at the same render files in the new index space.
    for sealed_window, window in zip(
            sealed["annotations"]["prepared_input"]["windows"],
            derived["annotations"]["prepared_input"]["windows"]):
        assert [derived["entries"][i] for i in window["entry_indices"]] == [
            sealed["entries"][i] for i in sealed_window["entry_indices"]]
    assert derived["annotations"]["band_serial"] == {
        "schema": BAND_SERIAL_READSET_SCHEMA,
        "sealed_manifest_sha256": consumer["executable_readset"]["manifest_sha256"],
        "handoff_sha256": handoff["handoff_sha256"],
        "producer": "layer-003",
        "generation": handoff["session"]["generation"],
        "replaced_phases": ["checkpoint-load", "chain-003-source",
                            "chain-003-bound"]}
    # Counts follow the entries; the chain annotations stay the record's.
    assert derived["annotations"]["chain_layers"] == [3]
    assert derived["entry_count"] == len(derived["entries"])
    assert derived["total_bytes"] == sum(e["bytes"] for e in derived["entries"])
    assert derived["read_plan"]["read_bytes"] == sum(
        phase["bytes"] for phase in derived["read_plan"]["phases"])
    assert len({(e["path"], e["offset"]) for e in derived["entries"]}) == len(
        derived["entries"])
    # Deterministic bytes: the dispatcher and the quantum derive the same wire.
    assert band_serial_manifest_bytes(
        consumer, handoff, _slice(consumer)["checkpoint"],
        output_root=tmp_path) == jl.seal_manifest_bytes(derived)


def test_the_band_serial_readset_passes_the_pb_phase_planner(tmp_path):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    bound, _receipt_doc = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer, adjoint_slice=_slice(consumer))
    derived = band_serial_manifest(
        _sealed(consumer), handoff, _slice(consumer)["checkpoint"],
        sealed_manifest_sha256=consumer["executable_readset"]["manifest_sha256"])
    prefix = str(tmp_path)
    for entry in derived["entries"]:
        if entry["path"].startswith("/fixture"):
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len("/fixture"):]
        elif entry["path"].startswith(prefix):
            entry["path"] = "/mnt/shared/fixture-run" + entry["path"][len(prefix):]
    ranges = tiers.manifest_phase_ranges(core.validate_data_manifest(derived))
    assert [item["name"] for item in ranges] == [
        phase["name"] for phase in derived["read_plan"]["phases"]]
    assert ranges[-1]["end_bytes"] == derived["read_plan"]["read_bytes"]


def test_the_quantum_refuses_a_staged_manifest_it_does_not_derive(tmp_path):
    bound, _receipt_doc = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    checkpoint = _slice(consumer)["checkpoint"]
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer, adjoint_slice=_slice(consumer))
    wire = band_serial_manifest_bytes(consumer, handoff, checkpoint,
                                      output_root=tmp_path)
    require_band_serial_readset(
        consumer, handoff, checkpoint, output_root=tmp_path,
        data_manifest_sha256=hashlib.sha256(wire).hexdigest())
    for digest, match in (
            (None, "needs --data-manifest-sha256"),
            (consumer["executable_readset"]["manifest_sha256"],
             "not the band-serial readset"),
            (consumer["campaign"]["read_manifest_sha256"],
             "not the band-serial readset")):
        with pytest.raises(QuantumHandoffRefused, match=match):
            require_band_serial_readset(
                consumer, handoff, checkpoint, output_root=tmp_path,
                data_manifest_sha256=digest)
    legacy = {key: value for key, value in consumer.items()
              if key != "executable_readset"}
    with pytest.raises(QuantumHandoffRefused, match="executable rows"):
        band_serial_manifest_bytes(legacy, handoff, checkpoint,
                                   output_root=tmp_path)
    # A sealed manifest that no longer hashes to the record's digest.
    path = Path(consumer["executable_readset"]["manifest_path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(QuantumHandoffRefused, match="does not hash"):
        band_serial_manifest_bytes(consumer, handoff, checkpoint,
                                   output_root=tmp_path)


def test_the_handoff_load_phase_stages_every_read_the_consumer_makes(
        tmp_path, monkeypatch):
    """Every file the consumer's head opens in handoff mode is staged in
    ``handoff-load``, in the order it is opened: a read outside the phase
    would open a pool path under a strict tier policy."""
    from prismaquant import joint_adjoint_checkpoints as checkpoints
    from prismaquant.joint_quantum_handoff import load_handoff_inputs

    bound, _receipt_doc = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    checkpoint = _slice(consumer)["checkpoint"]
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer, adjoint_slice=_slice(consumer))
    opened = []
    read_tensors = checkpoints.read_exact_entry_tensors
    read_payload = checkpoints._read_shared_state_payload

    def spy_tensors(records, **kwargs):
        records = list(records)
        opened.extend(entry["path"] for entry in records)
        return read_tensors(records, **kwargs)

    def spy_payload(path, entry):
        opened.append(str(path))
        return read_payload(path, entry)

    monkeypatch.setattr(checkpoints, "read_exact_entry_tensors", spy_tensors)
    monkeypatch.setattr(checkpoints, "_read_shared_state_payload", spy_payload)
    plane, owners, shared_pass = load_handoff_inputs(
        handoff, checkpoint, n_probes=N_PROBES, n_batches=N_BATCHES)
    assert sorted(plane) == sorted(owners) == [
        (p, b) for p in range(N_PROBES) for b in range(N_BATCHES)]
    assert sorted(shared_pass) == list(range(N_BATCHES))
    derived = json.loads(gzip.decompress(band_serial_manifest_bytes(
        consumer, handoff, checkpoint, output_root=tmp_path)))
    phase = next(p for p in derived["read_plan"]["phases"]
                 if p["name"] == HANDOFF_LOAD_PHASE)
    assert opened == [derived["entries"][index]["path"]
                      for index in phase["entry_indices"]]


def test_a_handoff_reads_the_checkpoint_plane_by_its_coordinates(tmp_path):
    """The consumer's plane check reads each checkpoint entry's sealed
    coordinates, never its name (the name is a writer convention)."""
    bound, _receipt_doc = _campaign(tmp_path)
    published = _emit(bound[3])
    consumer = bound[2]
    adjoint_slice = _slice(consumer)
    entries = adjoint_slice["checkpoint"]["activation_entries"]
    for index, entry in enumerate(entries):
        entry["name"] = f"renamed-{index}"
    load_quantum_handoff(published["path"], published["sha256"],
                         record=consumer, adjoint_slice=adjoint_slice)
    repeated = copy.deepcopy(adjoint_slice)
    repeated["checkpoint"]["activation_entries"][1]["metadata"]["identity"][
        "coordinates"] = dict(entries[0]["metadata"]["identity"]["coordinates"])
    with pytest.raises(QuantumHandoffRefused, match="repeats probe"):
        load_quantum_handoff(published["path"], published["sha256"],
                             record=consumer, adjoint_slice=repeated)
    entries[0]["metadata"]["identity"]["coordinates"] = {"probe": 0}
    with pytest.raises(QuantumHandoffRefused, match="carries no coordinates"):
        load_quantum_handoff(published["path"], published["sha256"],
                             record=consumer, adjoint_slice=adjoint_slice)


# -- the dispatch edge --------------------------------------------------------

def _pin_published_helper_root(monkeypatch):
    """The published PrismaBuild generation, named, never taken from the env.

    A producer row's handoff template is validated by PrismaBuild's own
    ``produced_output``, which the dispatcher loads through the lease
    helper root. Inside a PrismaBuild action that root is injected, so
    these tests passed there and failed in CI, where nothing injects it
    (8 failures on main 4b310ce59f6). The root is now the published
    generation itself, with the injected variable removed, and a box with
    no published runtime skips.
    """
    import prismaquant.staged_lease as staged_lease

    fleet = Path("/mnt/shared/prismabuild-fleet")
    try:
        receipt = json.loads((fleet / "repo" / "RUNTIME_VERSION.json").read_text())
        root = fleet / "runtime-generations" / str(receipt["generation"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        pytest.skip(f"no published PrismaBuild runtime to validate a handoff "
                    f"template against: {exc}")
    if not (root / "src" / "prismabuild" / "produced_output.py").is_file():
        pytest.skip(f"the published PrismaBuild generation is unreadable: {root}")
    monkeypatch.delenv(staged_lease.HELPER_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(staged_lease, "_HELPER_ROOT", str(root))


def _dispatch_layout(tmp_path, monkeypatch, *, run_identity_extra=None):
    import dispatch_joint_quanta as dispatch
    _pin_published_helper_root(monkeypatch)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    monkeypatch.setattr(dispatch, "_pbrun_seals_produced_output", lambda *a: True)
    bound, receipt = _campaign(tmp_path, run_identity_extra=run_identity_extra)
    plan = Path(bound[0]["campaign"]["plan_path"])
    plan.write_text(json.dumps({"execution": {"boundary_storage": {
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": ARTIFACT_MAX, "prefetch_batches": PREFETCH}}}))
    records = tmp_path / "records"
    records.mkdir()
    for layer, record in bound.items():
        (records / f"{record['quantum_id']}.json").write_text(json.dumps(record))
    receipt_path = tmp_path / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    out = tmp_path / "out"
    out.mkdir()
    return dispatch, bound, records, receipt_path, out


def _main(dispatch, gateway, records, receipt_path, out, *extra):
    return dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path), *extra], _gateway=gateway,
        # The fixture model has no checkpoint; the source-read coverage gate
        # has its own tests (test_readset_coverage_1095).
        _coverage=lambda rows: [])


def _inner(argv):
    tail = argv[argv.index("--") + 1:]
    return tail[tail.index("--", tail.index("--spec")) + 1:]


def _phases(argv):
    return [argv[i + 1].split("=", 1)[0] for i, word in enumerate(argv)
            if word == "--progress-phase"]


def _by_id(gateway):
    return {row["quantum_id"]: row for row in gateway.submitted}


def test_band_serial_publishes_producers_then_their_consumers(
        tmp_path, monkeypatch, capsys):
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch)
    band = ("--band-serial", "--handoff-tier", TIER)

    # Run 1: the band tops publish and declare a handoff; their successors
    # wait, because PrismaBuild would run them before the handoff exists.
    first = dispatch.FakeGateway()
    assert _main(dispatch, first, records, receipt_path, out, *band) == 0
    rows = _by_id(first)
    assert sorted(rows) == ["layer-001", "layer-003"]
    printed = capsys.readouterr().out
    assert '"band_serial_pending": "layer-002"' in printed
    assert '"band_serial_pending": "layer-000"' in printed
    for quantum_id, layer in (("layer-003", 3), ("layer-001", 1)):
        argv = rows[quantum_id]["argv"]
        inner = _inner(argv)
        assert "--emit-adjoint-handoff" in inner
        assert "--adjoint-handoff" not in inner
        template_path = Path(argv[argv.index("--produced-output-template") + 1])
        assert argv.index("--produced-output-template") < argv.index("--")
        template = json.loads(template_path.read_text())
        # The template names the producer's handoff directory under
        # --output-root, where the file itself lives (PQ #1200). The fixture
        # records name <tmp>/run; a quantum run under this root refuses them.
        from prismaquant.joint_quantum_handoff import handoff_root
        assert template["output_prefix"] == str(
            handoff_root(out / "layer-quanta" / quantum_id).resolve())
        assert template_path.parent == out / "layer-quanta" / "band-serial"
        assert template["durable_maxima"]["payload_max_bytes"] == ARTIFACT_MAX
        assert template["permitted_tiers"] == [TIER]
        # Write-only (PQ #1075): the producer never reads its handoff back,
        # so it reserves no stage window and commits each group at origin.
        assert template["write_only"] is True
        assert template["working_demands"] == {
            TIER: {"minimum_gib": 0, "window_gib": 0}}
        assert _phases(argv)[:2] == ["head", "checkpoint-load"]

    # Layer 3 executes, publishes its handoff and reports complete; layer 1
    # is still running.
    published = _emit(bound[3])
    _complete(bound[3], published)
    first.mark_terminal(rows["layer-003"]["action_key"])

    # Run 2 continues the same state file; the gateway keeps what it knows.
    second = first
    second.submitted = []
    assert _main(dispatch, second, records, receipt_path, out, *band) == 0
    rows2 = _by_id(second)
    assert sorted(rows2) == ["layer-001", "layer-002"]
    # A resubmitted producer is the same sealed action: identical argv.
    assert rows2["layer-001"]["argv"] == rows["layer-001"]["argv"]
    consumer = rows2["layer-002"]["argv"]
    inner = _inner(consumer)
    assert inner[inner.index("--adjoint-handoff") + 1] == published["path"]
    assert inner[inner.index("--adjoint-handoff-sha256") + 1] == published["sha256"]
    assert "--emit-adjoint-handoff" not in inner  # layer 1 is another band
    staged = Path(consumer[consumer.index("--data-manifest") + 1])
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    assert inner[inner.index("--data-manifest-sha256") + 1] == digest
    assert staged.parent == out / "layer-quanta" / "band-serial"
    phases = _phases(consumer)
    assert phases[:3] == ["head", HANDOFF_LOAD_PHASE, "own-002-source"]
    assert "checkpoint-load" not in phases
    assert not any(name.startswith("chain-") for name in phases)
    grace = dict(consumer[i + 1].split("=", 1) for i, word in enumerate(consumer)
                 if word == "--progress-phase")
    assert int(grace[HANDOFF_LOAD_PHASE]) == dispatch.HEAD_PROGRESS_GRACE_S
    # The quantum re-derives the staged bytes from what it is given.
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=bound[2], adjoint_slice=_slice(bound[2]))
    require_band_serial_readset(bound[2], handoff, _slice(bound[2])["checkpoint"],
                                output_root=out, data_manifest_sha256=digest)
    events = [json.loads(line) for line in
              (out / "layer-quanta" / "campaign-state.json").read_text().splitlines()]
    event = [e for e in events if e.get("quantum_id") == "layer-002"][-1]
    assert event["cotangent_source"] == {
        "mode": "handoff", "path": published["path"],
        "sha256": published["sha256"], "producer": "layer-003",
        "handoff_sha256": handoff["handoff_sha256"]}
    assert event["data_manifest_sha256"] == digest
    assert event["handoff_template"] is None

    # Run 3 without --band-serial: in-flight rows keep the mode they were
    # submitted in, so a resubmission is never a second action.
    third = second
    third.submitted = []
    assert _main(dispatch, third, records, receipt_path, out) == 0
    rows3 = _by_id(third)
    assert rows3["layer-002"]["argv"] == consumer
    assert rows3["layer-001"]["argv"] == rows["layer-001"]["argv"]
    assert "layer-000" in rows3  # never submitted, so plain chain mode
    assert "--adjoint-handoff" not in _inner(rows3["layer-000"]["argv"])


@pytest.mark.parametrize("defect", ["not-executed", "gapped", "foreign-identity",
                                    "altered-handoff"])
def test_a_consumer_waits_until_its_producer_published_a_whole_handoff(
        tmp_path, monkeypatch, capsys, defect):
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch)
    band = ("--band-serial", "--handoff-tier", TIER)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out, *band) == 0
    key = _by_id(gateway)["layer-003"]["action_key"]
    published = _emit(bound[3])
    _complete(bound[3], published)
    status = Path(bound[3]["output_space"]["root"]) / "status.json"
    if defect != "not-executed":
        gateway.mark_terminal(key)
    if defect == "gapped":
        status.write_text(json.dumps({"identity_sha256": bound[3]["identity_sha256"],
                                      "status": "gapped"}))
    elif defect == "foreign-identity":
        status.write_text(json.dumps({"identity_sha256": "f" * 64,
                                      "status": "complete"}))
    elif defect == "altered-handoff":
        path = Path(published["path"])
        path.write_bytes(path.read_bytes() + b" ")
    capsys.readouterr()
    gateway.submitted = []
    assert _main(dispatch, gateway, records, receipt_path, out, *band) == 0
    assert "layer-002" not in _by_id(gateway)
    assert '"band_serial_pending": "layer-002"' in capsys.readouterr().out


def test_a_fused_batch_one_regime_runs_band_serial(tmp_path, monkeypatch):
    """Probe fusion at a batch size of one keeps the per-sample arithmetic,
    so its rows run band-serial like the default regime's."""
    from prismaquant.joint_adjoint_slices import chain_regime_identity
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch, run_identity_extra={"chain_regime": chain_regime_identity(
            {"batch_size": 1, "probe_fusion": True})})
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--band-serial", "--handoff-tier", TIER) == 0
    rows = _by_id(gateway)
    assert sorted(rows) == ["layer-001", "layer-003"]
    assert all("--emit-adjoint-handoff" in _inner(row["argv"]) for row in rows.values())


def test_a_producer_that_published_no_handoff_leaves_its_consumer_on_the_chain(
        tmp_path, monkeypatch):
    """A producer submitted in chain mode completes without a handoff; its
    consumer then rebuilds the chain, which gives the same bytes."""
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out) == 0
    key = _by_id(gateway)["layer-003"]["action_key"]
    assert "--emit-adjoint-handoff" not in _inner(_by_id(gateway)["layer-003"]["argv"])
    _complete(bound[3], None)
    gateway.mark_terminal(key)
    gateway.submitted = []
    # layer-002 was submitted in chain mode in run 1, so it stays chain; a
    # fresh consumer of a handoff-less producer is chain as well.
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--band-serial", "--handoff-tier", TIER) == 0
    argv = _by_id(gateway)["layer-002"]["argv"]
    assert "--adjoint-handoff" not in _inner(argv)
    assert _phases(argv)[:2] == ["head", "checkpoint-load"]


def test_a_fresh_consumer_of_a_handoff_less_producer_runs_the_chain(
        tmp_path, monkeypatch):
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch)
    state = out / "layer-quanta" / "campaign-state.json"
    state.parent.mkdir(parents=True)
    # Layer 3 ran (in chain mode) before band-serial dispatch existed.
    state.write_text(json.dumps({"event": "quantum-submitted",
                                 "quantum_id": "layer-003",
                                 "identity_sha256": bound[3]["identity_sha256"],
                                 "action_key": "fake-old"}) + "\n")
    _complete(bound[3], None)
    gateway = dispatch.FakeGateway()
    gateway.mark_terminal("fake-old")
    assert _main(dispatch, gateway, records, receipt_path, out,
                 "--band-serial", "--handoff-tier", TIER) == 0
    rows = _by_id(gateway)
    assert "layer-003" not in rows
    assert "--adjoint-handoff" not in _inner(rows["layer-002"]["argv"])
    assert _phases(rows["layer-002"]["argv"])[:2] == ["head", "checkpoint-load"]


@pytest.mark.parametrize("case", ["no-tier", "legacy-rows", "batched-regime",
                                  "foreign-handoff"])
def test_band_serial_refuses_what_it_cannot_run(tmp_path, monkeypatch, capsys, case):
    from prismaquant.joint_adjoint_slices import chain_regime_identity
    extra = ({"chain_regime": chain_regime_identity(
        {"batch_size": 8, "probe_fusion": False})}
             if case == "batched-regime" else None)
    dispatch, bound, records, receipt_path, out = _dispatch_layout(
        tmp_path, monkeypatch, run_identity_extra=extra)
    flags = ["--band-serial"] + ([] if case == "no-tier" else ["--handoff-tier", TIER])
    if case == "legacy-rows":
        for path in records.glob("layer-*.json"):
            record = json.loads(path.read_text())
            del record["executable_readset"]
            path.write_text(json.dumps(record))
    gateway = dispatch.FakeGateway()
    if case == "foreign-handoff":
        assert _main(dispatch, gateway, records, receipt_path, out, *flags) == 0
        key = _by_id(gateway)["layer-003"]["action_key"]
        # Layer 1's handoff (band 2) named as layer 3's: layer 2 refuses it.
        _complete(bound[3], _emit(bound[1]))
        gateway.mark_terminal(key)
        gateway.submitted = []
    capsys.readouterr()
    assert _main(dispatch, gateway, records, receipt_path, out, *flags) == 3
    refusal = capsys.readouterr().err
    assert {"no-tier": "needs --handoff-tier",
            "legacy-rows": "executable rows only",
            "batched-regime": "batch size 8",
            "foreign-handoff": "refuses the handoff"}[case] in refusal
    assert gateway.submitted == []


@pytest.mark.parametrize("spec_kernel,producer_kernel",
                         [("kda_gram_v1", None), (None, "kda_gram_v1")],
                         ids=["kernel-spec-fallback-producer",
                              "fallback-spec-kernel-producer"])
def test_a_band_runs_in_one_kda_kernel_mode(tmp_path, monkeypatch, capsys,
                                            spec_kernel, producer_kernel):
    """A consumer binds only a handoff produced in its own launch's mode (PQ #1214).

    One dispatch wraps every row in one spec, so a band dispatched once is in
    one mode. A band resumed under a spec in the other mode is the case the
    handoff's stamp refuses, before the consumer is published.
    """
    dispatch, bound, records, receipt_path, out = _dispatch_layout(tmp_path, monkeypatch)
    flags = ("--band-serial", "--handoff-tier", TIER)
    gateway = dispatch.FakeGateway()
    assert _main(dispatch, gateway, records, receipt_path, out, *flags) == 0
    key = _by_id(gateway)["layer-003"]["action_key"]
    _complete(bound[3], _emit(bound[3], kernel=producer_kernel))
    gateway.mark_terminal(key)
    gateway.submitted = []
    spec = json.loads(Path(dispatch.SPEC_PATH).read_text())
    spec["env"] = ({} if spec_kernel is None
                   else {"PRISMAQUANT_STAGE_B_KDA_KERNEL": spec_kernel})
    Path(dispatch.SPEC_PATH).write_text(json.dumps(spec))
    capsys.readouterr()
    assert _main(dispatch, gateway, records, receipt_path, out, *flags) == 3
    refusal = capsys.readouterr().err
    assert "refuses the handoff" in refusal and "one mode" in refusal
    assert gateway.submitted == []
