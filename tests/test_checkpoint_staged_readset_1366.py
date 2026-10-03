"""CPU prerequisite checks; context doubles do not qualify staged lifetimes/GPU."""
from contextlib import contextmanager
import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import torch

import test_quantum_executable_readset as fixture
from test_quantum_executable_readset import (CALIB, N_PROBES, RENDER_PREREQ, STRIDED,
                                           _tiny_records, _tiny_receipt)
from prismaquant import joint_layer_quanta as quanta

MODE = quanta.CHECKPOINT_INCOMING_STAGED


def _setup(tmp_path, layer=3):
    records, parent = _tiny_records(tmp_path)
    record = next(row for row in records if row["layer"] == layer)
    receipt = _tiny_receipt(tmp_path, record["campaign"])
    record, _ = fixture._bind_slice(record, receipt, tmp_path / "slices")
    return {"record": record, "parent": parent, "receipt": receipt}


def _manifest(setup, selected=MODE, replay_mode="spill"):
    return quanta.build_quantum_executable_manifest(
        setup["record"], setup["receipt"], setup["parent"],
        strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=CALIB, render_prerequisite=RENDER_PREREQ,
        replay_mode=replay_mode, checkpoint_incoming_mode=selected)


def _bound(setup, manifest):
    # The fixture's campaign output root is independent of its quantum space.
    root = str(Path(setup["record"]["campaign"]["plan_path"]).parent / "run")
    return quanta.bind_quantum_executable(
        setup["record"], setup["receipt"], setup["parent"], manifest=manifest,
        manifest_path=f"{root}/layer-quanta/adjoint/bound-readsets/{setup['record']['quantum_id']}.executable.json.gz",
        manifest_sha256=hashlib.sha256(quanta.seal_manifest_bytes(manifest)).hexdigest(),
        output_root=root, strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=CALIB, render_prerequisite=RENDER_PREREQ, replay_mode="spill",
        checkpoint_incoming_mode=MODE)


def _slice(setup):
    record = setup["record"]
    return quanta.bind_adjoint_slice(
        setup["receipt"], record["layer"], plan_sha256=record["campaign"]["plan_sha256"],
        prepared_sha256=record["campaign"]["prepared_sha256"],
        scope=record["campaign"]["campaign_scope"], checkpoints=STRIDED)[0]


def _phase_rows(manifest, name):
    phase = next(p for p in manifest["read_plan"]["phases"] if p["name"] == name)
    return [manifest["entries"][i] for i in phase["entry_indices"]]


def test_original_cotangents_leave_the_head_phase(tmp_path):
    setup = _setup(tmp_path)
    manifest = _manifest(setup)
    activation = _slice(setup)["checkpoint"]["activation_entries"]
    paths = {row["path"] for row in _phase_rows(manifest, "checkpoint-load")}
    assert not paths & {row["path"] for row in activation}
    assert manifest["annotations"]["checkpoint_incoming_mode"] == MODE
    for probe in range(N_PROBES):
        staged = _phase_rows(manifest, f"spill-p{probe}")
        own = setup["receipt"]["boundary_entries"]["3"]
        assert [row["path"] for row in staged[:len(own)]] == [row["path"] for row in own]
        incoming = [row for row in activation if row["metadata"]["identity"]["coordinates"]["probe"] == probe]
        assert [row["path"] for row in staged[len(own):]] == [row["path"] for row in incoming]
    quanta.check_checkpoint_incoming_readset(_bound(setup, manifest), manifest, _slice(setup))


def test_default_entry_and_phase_contract_unchanged(tmp_path):
    setup = _setup(tmp_path)
    default = _manifest(setup, selected=None)
    selected = _manifest(setup)
    assert selected["entries"] == default["entries"]
    assert selected["total_bytes"] == default["total_bytes"]
    assert selected["read_plan"]["read_bytes"] == default["read_plan"]["read_bytes"]
    assert [p["name"] for p in selected["read_plan"]["phases"]] == [p["name"] for p in default["read_plan"]["phases"]]
    assert "checkpoint_incoming_mode" not in default["annotations"]
    activation = {r["path"] for r in _slice(setup)["checkpoint"]["activation_entries"]}
    assert activation <= {r["path"] for r in _phase_rows(default, "checkpoint-load")}


@pytest.mark.parametrize("mode", [True, 1, {}, "stream_once_research", "unknown"])
def test_invalid_seal_refuses(tmp_path, mode):
    with pytest.raises(ValueError, match="checkpoint incoming mode"):
        _manifest(_setup(tmp_path), selected=mode)


def test_chain_empty_windowed_and_copied_chain_refuse(tmp_path):
    with pytest.raises(ValueError, match="requires spill"):
        _manifest(_setup(tmp_path / "empty"), replay_mode="windowed")
    with pytest.raises(ValueError, match="referenced owner"):
        _manifest(_setup(tmp_path / "chain", layer=2))


@pytest.mark.parametrize("change", ["mode", "probe", "head", "order", "foreign_phase", "work", "digest"])
def test_rehashed_wrong_consuming_plan_refuses(tmp_path, change):
    setup = _setup(tmp_path)
    manifest = _manifest(setup)
    bound = _bound(setup, manifest)
    changed = copy.deepcopy(manifest)
    phases = {p["name"]: p for p in changed["read_plan"]["phases"]}
    if change == "mode":
        changed["annotations"].pop("checkpoint_incoming_mode")
    elif change == "probe":
        changed["annotations"]["n_probes"] += 1
    elif change == "head":
        index = phases["spill-p0"]["entry_indices"].pop()
        phases["checkpoint-load"]["entry_indices"].append(index)
    elif change == "order":
        phases["spill-p0"]["entry_indices"].reverse()
    elif change == "foreign_phase":
        phases["head"]["entry_indices"].append(phases["spill-p0"]["entry_indices"][-1])
    elif change == "work":
        changed["annotations"]["checkpoint_incoming"]["streamed_incoming"]["spill-p0"] = 0
    else:
        index = phases["spill-p0"]["entry_indices"][-1]
        changed["entries"][index]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        quanta.check_checkpoint_incoming_readset(bound, changed, _slice(setup))
    # A consistent new wire hash cannot make this producer accept altered membership.
    with pytest.raises(ValueError):
        _bound(setup, changed)


@pytest.mark.parametrize("dev_mode", ["0", "1"])
def test_foreign_incoming_slice_refuses_in_both_modes(tmp_path, monkeypatch, dev_mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    setup = _setup(tmp_path)
    manifest = _manifest(setup)
    bound = _bound(setup, manifest)
    manifest["annotations"]["slice_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint incoming readset binds a foreign slice"):
        quanta.check_checkpoint_incoming_readset(bound, manifest, _slice(setup))


def test_claim_digest_and_pb_decoder_are_used(tmp_path, monkeypatch):
    """Real bounded CAS read/SDK decoder, with only the claim identity doubled."""
    import prismabuild.client as sdk
    from prismaquant import staged_lease as lease
    setup = _setup(tmp_path)
    manifest = _manifest(setup)
    # PB checks the declared mount even when only metadata is read here.
    for row in manifest["entries"]:
        row["path"] = "/mnt/shared/checkpoint-fixture/" + row["path"].lstrip("/")
    wire = quanta.seal_manifest_bytes(manifest)
    digest = hashlib.sha256(wire).hexdigest()
    cas = tmp_path / "cas"
    blob = cas / "blobs" / digest[:2] / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(wire)
    monkeypatch.setattr(lease, "client_sdk", lambda: sdk)
    monkeypatch.setattr(lease, "resolve_sealed_readset", lambda: (str(cas), digest, len(wire)))
    assert lease.load_sealed_manifest(digest) == manifest
    with pytest.raises(lease.ReadsetUnbound, match="claim row names manifest"):
        lease.load_sealed_manifest("0" * 64)
    blob.write_bytes(wire[:-1] + bytes([wire[-1] ^ 1]))
    with pytest.raises(lease.ReadsetUnbound, match="does not hash"):
        lease.load_sealed_manifest(digest)


@pytest.mark.parametrize("case", ["inactive", "no_map", "unbound", "wrong_phase", "foreign_probe"])
def test_runtime_requires_authenticated_context(tmp_path, monkeypatch, case):
    from prismaquant import joint_cost_quantum as runtime, staged_lease, staged_tier_policy
    setup = _setup(tmp_path)
    manifest = _manifest(setup)
    bound = _bound(setup, manifest)
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP", "context-double")
    monkeypatch.setattr(staged_tier_policy, "active_policy", lambda: frozenset(("ram", "ssd")))
    monkeypatch.setattr(staged_lease, "load_sealed_manifest", lambda digest: manifest)
    if case == "inactive":
        monkeypatch.setattr(staged_tier_policy, "active_policy", lambda: None)
    elif case == "no_map":
        monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP")
    elif case == "unbound":
        def unbound(digest):
            raise staged_lease.ReadsetUnbound("missing claimed row")
        monkeypatch.setattr(staged_lease, "load_sealed_manifest", unbound)
    elif case == "wrong_phase":
        manifest["read_plan"]["phases"][1]["entry_indices"].append(manifest["read_plan"]["phases"][-1]["entry_indices"][-1])
    probes = N_PROBES + (case == "foreign_probe")
    with pytest.raises(runtime.QuantumIdentityRefused, match="checkpoint incoming"):
        runtime._authenticate_checkpoint_incoming_readset(bound, _slice(setup), {"n_probes": probes})


def test_dispatch_compute_counts_exclude_incoming_rows():
    from tools.dispatch_joint_quanta import phase_work_entries
    annotations = {"checkpoint_incoming": {"streamed_incoming": {"chain-003-bound": 6, "spill-p0": 3}}}
    assert phase_work_entries("chain-003-bound", {"entries": 9}, annotations) == 3
    assert phase_work_entries("spill-p0", {"entries": 6}, annotations) == 3
    assert phase_work_entries("checkpoint-load", {"entries": 2}, annotations) == 2
    annotations["band_serial"] = {"streamed_incoming": {"spill-p0": 4}}
    assert phase_work_entries("spill-p0", {"entries": 6}, annotations) is None


@pytest.mark.parametrize("layer,regime", [(0, None), (1, None),
    (1, {"capture_batch": 4, "accumulation": "operator_gemm", "chunk_rows": 65536})])
def test_real_cpu_operands_plane_costs_and_complete_resume(tmp_path, monkeypatch, layer, regime):
    """Real arithmetic/exact reads; authenticated-context double, no source/lease claim."""
    from prismaquant import joint_adjoint_checkpoints as checkpoint, joint_cost_quantum as runtime
    import test_stageb_one_pass_spill as spill
    monkeypatch.setattr(fixture, "_expert_fixture", spill._bf16_expert_fixture)
    if regime is not None:
        import test_joint_cost_quantum_runtime as rt
        boundary_policy = rt._boundary_policy
        monkeypatch.setattr(rt, "_boundary_policy", lambda *a, **kw: {
            **boundary_policy(*a, **kw), "prefetch_batches": 4})
    setup = fixture._acceptance_setup_expert(tmp_path, monkeypatch)
    if regime is not None:
        make_execution = setup["make_execution"]
        def execution_with_regime(*args, **kwargs):
            return {**make_execution(*args, **kwargs), "replay_regime": regime}
        setup["make_execution"] = execution_with_regime
    root = spill._spill_root(tmp_path, needs_direct_io=False)
    from prismaquant import perturbed_x_cache
    monkeypatch.setattr(perturbed_x_cache, "_direct_io_block", lambda fd: 4096)
    monkeypatch.setenv(spill.spill_mod.SPILL_ENV[0], str(root))
    monkeypatch.setenv(spill.spill_mod.SPILL_ENV[1], str(1 << 30))
    captured, slots = [], {}
    held, incoming, store = checkpoint.PlaneHostStaging.held, checkpoint.PlaneHostStaging.incoming, checkpoint.PlaneHostStaging.store
    def raw(tensor):
        return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    @contextmanager
    def held_logged(self):
        slots[id(self)] = {"incoming": [], "final": {}}
        try:
            with held(self):
                yield
                captured.append(slots[id(self)])
        finally:
            slots.pop(id(self))
    def incoming_logged(self, keys, **kwargs):
        keys = list(keys)
        value = incoming(self, keys, **kwargs)
        slots[id(self)]["incoming"].append((keys, raw(value)))
        return value
    def store_logged(self, keys, rows, gradient):
        keys = list(keys)
        store(self, keys, rows, gradient)
        slots[id(self)]["final"].update({key: raw(self.plane[key]) for key in keys})
    monkeypatch.setattr(checkpoint.PlaneHostStaging, "held", held_logged)
    monkeypatch.setattr(checkpoint.PlaneHostStaging, "incoming", incoming_logged)
    monkeypatch.setattr(checkpoint.PlaneHostStaging, "store", store_logged)
    def drive(selected=None, resume=False):
        return fixture._drive_quantum(tmp_path, monkeypatch, setup, layer=layer, resume=resume,
                                      replay_mode="spill", prepared_render_phases=True,
                                      checkpoint_incoming_mode=selected)
    _, _, baseline, _ = drive()
    baseline_plane = captured[-1]
    assert baseline_plane["incoming"] and baseline_plane["final"]
    _, _, baseline_resumed, _ = drive(resume=True)
    assert baseline_resumed["costs"] == baseline["costs"]
    baseline_resume_plane = captured[-1]
    record = setup["records"][f"layer-{layer:03d}"]
    checkpoint_dir = Path(record["output_space"]["checkpoint_dir"])
    baseline_journal = {p.name: p.read_bytes() for p in checkpoint_dir.glob("*.pkl")}
    shutil.rmtree(record["output_space"]["root"])
    manifests = []
    original_builder = fixture.build_quantum_executable_manifest
    def builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        manifests.append(result)
        return result
    monkeypatch.setattr(fixture, "build_quantum_executable_manifest", builder)
    def context_double(bound, adjoint_slice, execution):
        # This separately exercises real phase-derived binding, without pretending
        # fixture source installs or local opens are protected PB lease events.
        quanta.check_checkpoint_incoming_readset(bound, manifests[-1], adjoint_slice)
        assert execution["n_probes"] == manifests[-1]["annotations"]["n_probes"]
    monkeypatch.setattr(runtime, "_authenticate_checkpoint_incoming_readset", context_double)
    events, manifest, candidate, _ = drive(MODE)
    if record["adjoint"]["chain_layers"]:
        phase = quanta.executable_bound_phase_name(record["adjoint"]["chain_layers"][0])
        current, seen, opened = None, set(), []
        for event in events:
            if event[0] == "report":
                current = event[1]
            elif current == phase and event[0] == "boundary-path-open" and event[1] not in seen:
                seen.add(event[1])
                opened.append(event[1])
        assert opened == [row["path"] for row in _phase_rows(manifest, phase)]
    assert candidate["costs"] == baseline["costs"]
    assert captured[-1] == baseline_plane
    assert {p.name: p.read_bytes() for p in checkpoint_dir.glob("*.pkl")} == baseline_journal
    _, _, resumed, _ = drive(MODE, resume=True)
    assert resumed["costs"] == baseline["costs"]
    # With no pending prices the existing path omits spill and runs single
    # final batches even for a grouped capture seal; compare like resumes.
    assert captured[-1] == baseline_resume_plane
    shutil.rmtree(record["output_space"]["root"])
    import prismaquant.aura_cost as aura
    atomic = aura.atomic_write_bytes
    published = []
    def fail_second(path, body):
        if Path(path).suffix == ".pkl":
            if published:
                raise OSError("partial staged checkpoint journal")
            published.append(str(path))
        return atomic(path, body)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "atomic_write_bytes", fail_second)
        with pytest.raises(OSError, match="partial staged checkpoint journal"):
            drive(MODE)
    assert len(published) == 1 and Path(published[0]).is_file()
    _, _, partial, _ = drive(MODE, resume=True)
    assert partial["costs"] == baseline["costs"]
    assert captured[-1] == baseline_plane
    assert {p.name: p.read_bytes() for p in checkpoint_dir.glob("*.pkl")} == baseline_journal
    assert [p["name"] for p in manifest["read_plan"]["phases"]] == list(quanta.quantum_executable_phase_names(
        record["adjoint"]["chain_layers"], layer, n_probes=setup["n_probes"],
        replay_windows=len(record["windows"]), render_phases=True, replay_mode="spill"))


@pytest.mark.parametrize("sealed,launched", [(None, MODE), (MODE, None), (MODE, "unknown"), (MODE, "stream_once_research")])
def test_explicit_launch_cannot_override_sealed_mode(monkeypatch, sealed, launched):
    from types import SimpleNamespace
    from prismaquant import joint_cost_quantum as runtime
    record = {"executable_readset": {"checkpoint_incoming_mode": sealed}}
    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible launch reached authenticated payload/source seam")
    monkeypatch.setattr(runtime, "_authenticate_checkpoint_incoming_readset", forbidden)
    with pytest.raises(runtime.QuantumIdentityRefused, match="checkpoint incoming"):
        runtime.run_layer_quantum_core(
            SimpleNamespace(device="cpu"), None, None, {}, record=record,
            adjoint_slice={}, execution={"checkpoint_incoming_mode": launched},
            output_root="unused", resolved_windows=[], counters=None, progress=None)
