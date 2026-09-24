"""Static prepared-input route: existing prepared renders as staged inputs (PQ #917).

Pre-fix RED (PB action ``8c7c53b534d1`` / ``5d8dd89f6ca7``, 2 failed):
the executable manifest declared calibration, checkpoint, chain/own
source and boundary replay reads but no phase for the existing prepared
rendered weights the strict PWC reader must load inside each retained
window, and production dispatcher ``main`` refused every executable row
through the normal CLI. These tests assert the desired end state through
existing APIs only.

GREEN route: the builder seals an explicit prepared-input mode (one
``render-{w:02d}`` phase per retained window, before that window's
replays, bound to the sealed prepared/production-pickle digest and
roster), the binder carries the sealed membership into the bound record
block, and production ``main`` accepts exactly that complete contract
through the normal CLI + bound files + FakeGateway. Legacy
sequencing-only rows and malformed/foreign prepared contracts keep the
typed ``ExecutableBindingUnsupported`` refusal. The strict-PWC
acceptance fixture below serves two real retained windows through the
real leased stack (real PB writers, real resolver, real pinned SDK):
phase entry precedes each window's file opens, payload bytes come from
stage with ``bytes_from_pool == 0``, serving records carry pin IDs, and
pins release.
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

from prismaquant import joint_layer_quanta as jl
from test_quantum_executable_readset import (
    CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bound_inputs,
)
import test_strict_reader_tier_enforcement as strict
# The strict-PWC tests below drive ``strict._leased_fixture``, so they take
# the same state reset its own module does, before and after each test. The
# lease helper root, the test-only SDK injection and the residency resolver
# are process-global in ``staged_lease`` and ``staged_tier``, and a PB action
# also inherits ``PRISMABUILD_READER_HELPER_ROOT`` (the worker's runtime
# generation). Without the reset, ``_sdk()`` checks that generation against
# whatever ``prismabuild.reader_lease`` an earlier module in the same process
# imported (the qualification bundle's, through ``_pb_source``), and the
# window refuses ``lease-helper-divergent`` (PQ #1032, PB ``e75555fbef1c``).
# The injection these tests make was also never cleared, so it outlived them.
from test_strict_reader_tier_enforcement import _forget_state  # noqa: E402,F401 (autouse)

WINDOW_PAIRS = (
    (0, (("unit-w00-a", "FMT-A"), ("unit-w00-b", "FMT-A"))),
    (1, (("unit-w01-a", "FMT-A"), ("unit-w01-b", "FMT-A"))),
)


def _render_files(tmp_path):
    """Real tiny serialized render files with real size/digest evidence."""
    files = {}
    for window, pairs in WINDOW_PAIRS:
        for name, fmt in pairs:
            path = tmp_path / "renders" / f"{name}.{fmt}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.arange(8, dtype=torch.float32) + window, path)
            raw = path.read_bytes()
            files[(name, fmt)] = {
                "qname": name, "fmt": fmt, "path": str(path),
                "offset": 0, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    return files


def _prepared_inputs(record, files):
    """The explicit prepared-input mode argument from sealed digests."""
    return {
        "schema": jl.PREPARED_INPUT_SCHEMA,
        "production_pkl_sha256": RENDER_PREREQ["production_pkl_sha256"],
        "unit_roster_sha256": RENDER_PREREQ["unit_roster_sha256"],
        "prepared_sha256": record["campaign"]["prepared_sha256"],
        "windows": [
            {"window_index": window,
             "members": [[name, fmt] for name, fmt in pairs],
             "entries": [dict(files[pair]) for pair in pairs]}
            for window, pairs in WINDOW_PAIRS],
    }


def _rebound_record_receipt(record, receipt):
    """A production-shaped receipt binds the same record (PQ #993).

    The tiny capture receipt carries campaign digests under run_identity
    only; production capture files seal them top-level too. Those are
    receipt-only fields outside every slice, so the record already bound to
    its slice of the tiny receipt is bound to the production-shaped one:
    the slice owner gives the same digest and nothing is re-sealed.
    """
    receipt = copy.deepcopy(receipt)
    campaign = record["campaign"]
    receipt["plan_sha256"] = campaign["plan_sha256"]
    receipt["prepared_sha256"] = campaign["prepared_sha256"]
    _slice, digest = jl.bind_adjoint_slice(
        receipt, record["layer"], plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["campaign_scope"], checkpoints=STRIDED)
    assert digest == record["adjoint"]["slice_sha256"]
    return record, receipt


#: The spill root a spill-sealed fixture row's spec declares.
SPILL_ROOT = "/home/rob/pb-scratch/stage-b-spill"


def _fixture_spill_bound():
    """A small, valid sealed spill bound (one dense target, batch 1)."""
    from prismaquant.joint_replay_spill import (
        SpillTarget, seal_spill_bound, spill_geometry)
    return seal_spill_bound(
        spill_geometry({"t": SpillTarget(8, 16)}, [("t",)], pending={"t"},
                       batch_tokens=[4], n_probes=N_PROBES, element_size=2,
                       experts_per_token=None),
        block=4096, capture_batch=1, element_dtype="bfloat16")


def _spill_spec(ceiling, *, regime=None):
    """A container spec declaring the Stage B spill with a spec ceiling."""
    env = {"PRISMAQUANT_STAGE_B_SPILL_ROOT": SPILL_ROOT,
           "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES": str(ceiling)}
    if regime is not None:
        env["PRISMAQUANT_STAGE_B_REPLAY_REGIME"] = regime
    return {"container": {"image": "sha256:" + "0" * 64,
                          "mounts": [{"source": SPILL_ROOT, "target": SPILL_ROOT,
                                      "readonly": False}]},
            "env": env}


def _bind_prepared(tmp_path, root, *, replay_mode=None, spill_bound=None):
    """Real builder + real binder prepared-input sealing (no hand-seal)."""
    record, receipt, parent, kwargs = _bound_inputs(tmp_path, root=root)
    record, receipt = _rebound_record_receipt(record, receipt)
    files = _render_files(tmp_path)
    prepared = _prepared_inputs(record, files)
    manifest = jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ),
        prepared_inputs=prepared, replay_mode=replay_mode)
    wire = jl.seal_manifest_bytes(manifest)
    manifest_path = Path(kwargs["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(wire)
    bound = jl.bind_quantum_executable(
        record, receipt, parent, manifest=manifest,
        manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(wire).hexdigest(),
        output_root=root, strided_boundaries=STRIDED, n_probes=N_PROBES,
        calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ),
        prepared_inputs=prepared, replay_mode=replay_mode,
        spill_bound=spill_bound)
    return bound, receipt, parent, manifest, files, prepared


def _dispatch_prepared(tmp_path, monkeypatch, *, replay_mode=None,
                       spill_bound=None):
    """Normal-CLI dispatch layout around the binder-sealed row.

    A spill-sealed row seals ``spill_bound`` (a small valid one by default)
    under a spec that declares the spill, as the dispatcher requires.
    """
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    if replay_mode == "spill":
        if spill_bound is None:
            spill_bound = _fixture_spill_bound()
        spec.write_text(json.dumps(_spill_spec(1 << 30)))
    else:
        spec.write_text(json.dumps(
            {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    root = str(tmp_path / "run")
    bound, receipt, parent, manifest, files, prepared = _bind_prepared(
        tmp_path, root, replay_mode=replay_mode, spill_bound=spill_bound)
    records = tmp_path / "records"
    records.mkdir(parents=True)
    (records / "layer-002.json").write_text(json.dumps(bound))
    plan = Path(bound["campaign"]["plan_path"])
    if not plan.exists():
        plan.write_text("{}")  # the row reads its plan before publishing
    receipt_path = tmp_path / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    out = tmp_path / "out"
    out.mkdir()
    return dispatch, bound, receipt, manifest, records, receipt_path, out


# -- builder ---------------------------------------------------------------

def test_builder_declares_one_render_phase_per_retained_window(tmp_path):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    files = _render_files(tmp_path)
    manifest = jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ),
        prepared_inputs=_prepared_inputs(record, files))
    names = [phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert names == list(jl.quantum_executable_phase_names(
        [3], 2, n_probes=N_PROBES, replay_windows=len(record["windows"]),
        render_phases=True))
    for window, _ in WINDOW_PAIRS:
        render = f"render-{window:02d}"
        replay = f"replay-{window:02d}-p0"
        assert names.index(render) + 1 == names.index(replay), (
            f"{render} is not sealed immediately before its replay phases")
    annotation = manifest["annotations"]["prepared_input"]
    assert annotation["schema"] == jl.PREPARED_INPUT_SCHEMA
    assert annotation["prepared_sha256"] == record["campaign"][
        "prepared_sha256"]
    assert [window["window_index"] for window in annotation["windows"]] == [
        window["window_index"] for window in record["windows"]]
    for window in annotation["windows"]:
        phase = next(phase for phase in manifest["read_plan"]["phases"]
                     if phase["name"] == f"render-{window['window_index']:02d}")
        assert phase["entry_indices"] == window["entry_indices"]
        assert len(window["members"]) == len(window["entry_indices"]) == 2


def test_builder_keeps_legacy_bytes_without_prepared_inputs(tmp_path):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    manifest = jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    assert "prepared_input" not in manifest["annotations"]
    assert [phase["name"] for phase in manifest["read_plan"]["phases"]] == (
        list(jl.quantum_executable_phase_names(
            [3], 2, n_probes=N_PROBES,
            replay_windows=len(record["windows"]))))
    assert manifest == kwargs["manifest"]


def test_builder_refuses_foreign_prepared_roster(tmp_path):
    record, receipt, parent, _kwargs = _bound_inputs(tmp_path)
    files = _render_files(tmp_path)
    prepared = _prepared_inputs(record, files)
    prepared["unit_roster_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="foreign production pickle|foreign"):
        jl.build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ),
            prepared_inputs=prepared)


def test_builder_refuses_uncovered_window(tmp_path):
    record, receipt, parent, _kwargs = _bound_inputs(tmp_path)
    files = _render_files(tmp_path)
    prepared = _prepared_inputs(record, files)
    prepared["windows"] = prepared["windows"][:1]
    with pytest.raises(ValueError, match="not the sealed retained windows"):
        jl.build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ),
            prepared_inputs=prepared)


def test_binder_carries_prepared_membership_into_record(tmp_path):
    root = str(tmp_path / "run")
    bound, _receipt, _parent, manifest, _files, _prepared = _bind_prepared(
        tmp_path, root)
    block = bound["executable_readset"]
    announced = manifest["annotations"]["prepared_input"]
    carried = block["prepared_input"]
    for key in ("schema", "production_pkl_sha256", "unit_roster_sha256",
                "prepared_sha256"):
        assert carried[key] == announced[key]
    assert len(carried["windows"]) == len(announced["windows"])
    for carried_window, announced_window in zip(
            carried["windows"], announced["windows"]):
        for key in ("window_index", "members", "entry_indices"):
            assert carried_window[key] == announced_window[key]
        assert carried_window["entries"] == [
            {key: manifest["entries"][index][key]
             for key in ("path", "offset", "bytes", "sha256")}
            for index in announced_window["entry_indices"]]
        assert all(item["offset"] == 0 for item in
                   carried_window["entries"])
    assert block["phases"] == [phase["name"]
                               for phase in manifest["read_plan"]["phases"]]


def test_prepared_membership_against_resolved_windows():
    resolved = [
        {"window_index": 0, "names": ["unit-w00-a", "unit-w00-b"]},
        {"window_index": 1, "names": ["unit-w01-a", "unit-w01-b"]},
    ]
    sealed = [
        {"window_index": 0,
         "members": [["unit-w00-a", "FMT-A"], ["unit-w00-b", "FMT-A"]]},
        {"window_index": 1,
         "members": [["unit-w01-a", "FMT-A"], ["unit-w01-b", "FMT-A"]]},
    ]
    jl.check_prepared_windows_against_resolved(
        sealed, resolved, quantum_id="layer-002")
    foreign = copy.deepcopy(sealed)
    foreign[1]["members"] = [["unit-w01-a", "FMT-A"],
                             ["unit-evil", "FMT-A"]]
    with pytest.raises(ValueError, match="disagree with the sealed budget"):
        jl.check_prepared_windows_against_resolved(
            foreign, resolved, quantum_id="layer-002")
    with pytest.raises(ValueError, match="admits"):
        jl.check_prepared_windows_against_resolved(
            sealed[:1], resolved, quantum_id="layer-002")


# -- production dispatch ----------------------------------------------------

def test_dispatch_main_accepts_complete_prepared_input_contract(
        tmp_path, monkeypatch, capsys):
    (dispatch, bound, _receipt, manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway,
        # No checkpoint behind the fixture; coverage is tested on its own
        # (test_readset_coverage_1095).
        _coverage=lambda rows: [])
    refused = capsys.readouterr().err
    assert code == 0, (
        f"production main refuses the complete prepared-input contract "
        f"(exit {code}: {refused.strip()}); every executable row still "
        "refuses before submission")
    assert len(gateway.submitted) == 1
    argv = gateway.submitted[0]["argv"]
    wire = Path(bound["executable_readset"]["manifest_path"]).read_bytes()
    digest = hashlib.sha256(wire).hexdigest()
    assert digest in " ".join(argv)
    manifest_at = argv[argv.index("--data-manifest") + 1]
    assert hashlib.sha256(Path(manifest_at).read_bytes()).hexdigest() == (
        digest)
    declared = [argv[i + 1].split("=", 1)[0]
                for i, word in enumerate(argv)
                if word == "--progress-phase"]
    assert "render-00" in declared and "render-01" in declared
    assert declared.index("render-00") + 1 == declared.index(
        "replay-00-p0")


def test_dispatch_main_still_refuses_legacy_sequencing_row(
        tmp_path, monkeypatch, capsys):
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    record, receipt, parent, kwargs = _bound_inputs(
        tmp_path, root=str(tmp_path / "run"))
    record, receipt = _rebound_record_receipt(record, receipt)
    manifest = jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    wire = jl.seal_manifest_bytes(manifest)
    kwargs = dict(kwargs, manifest=manifest,
                      manifest_sha256=hashlib.sha256(
                          wire).hexdigest())
    assert "prepared_input" not in manifest["annotations"]
    manifest_path = Path(kwargs["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(jl.seal_manifest_bytes(manifest))
    bound = jl.bind_quantum_executable(
        record, receipt, parent, **kwargs)
    records = tmp_path / "records"
    records.mkdir(parents=True)
    (records / "layer-002.json").write_text(json.dumps(bound))
    receipt_path = tmp_path / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    out = tmp_path / "out"
    out.mkdir()
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway)
    refused = capsys.readouterr().err
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "no accepted PB produced-output binding" in refused
    assert gateway.submitted == []


def test_dispatch_refuses_foreign_prepared_roster(
        tmp_path, monkeypatch, capsys):
    (dispatch, bound, _receipt, manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch)
    tampered = copy.deepcopy(manifest)
    tampered["annotations"]["prepared_input"][
        "production_pkl_sha256"] = "0" * 64
    wire = jl.seal_manifest_bytes(tampered)
    manifest_path = Path(bound["executable_readset"]["manifest_path"])
    manifest_path.write_bytes(wire)
    bound["executable_readset"]["manifest_sha256"] = hashlib.sha256(
        wire).hexdigest()
    (records / "layer-002.json").write_text(json.dumps(bound))
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway)
    refused = capsys.readouterr().err
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "foreign" in refused
    assert gateway.submitted == []


def test_dispatch_refuses_mover_reference_shaped_contract(
        tmp_path, monkeypatch, capsys):
    """A valid-looking produced mover reference is not a read capability."""
    (dispatch, bound, _receipt, manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch)
    impostor = copy.deepcopy(manifest)
    impostor["annotations"]["prepared_input"] = {
        "schema": "prismabuild.produced_output_batch.v1",
        "material": "f" * 64,
        "windows": impostor["annotations"]["prepared_input"]["windows"],
    }
    wire = jl.seal_manifest_bytes(impostor)
    manifest_path = Path(bound["executable_readset"]["manifest_path"])
    manifest_path.write_bytes(wire)
    bound["executable_readset"]["manifest_sha256"] = hashlib.sha256(
        wire).hexdigest()
    bound["executable_readset"]["prepared_input"] = copy.deepcopy(
        impostor["annotations"]["prepared_input"])
    (records / "layer-002.json").write_text(json.dumps(bound))
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway)
    refused = capsys.readouterr().err
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "sequencing-only" in refused
    assert gateway.submitted == []


def test_prepared_manifest_passes_pb_phase_planner(tmp_path):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    record, receipt, parent, _kwargs = _bound_inputs(tmp_path)
    files = _render_files(tmp_path)
    manifest = jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ),
        prepared_inputs=_prepared_inputs(record, files))
    prefix = str(tmp_path)
    for entry in manifest["entries"]:
        if entry["path"].startswith("/fixture"):
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len(
                "/fixture"):]
        elif entry["path"].startswith(prefix):
            entry["path"] = "/mnt/shared/fixture-renders" + entry[
                "path"][len(prefix):]
    normalized = core.validate_data_manifest(manifest)
    ranges = tiers.manifest_phase_ranges(normalized)
    assert [item["name"] for item in ranges] == [
        phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert "render-00" in [item["name"] for item in ranges]
    assert ranges[-1]["end_bytes"] == manifest["read_plan"]["read_bytes"]


# -- strict PWC acceptance (real leased stack, two retained windows) --------

def _leased_renders(tmp_path, monkeypatch, files):
    """Stage the declared render files through the REAL PB writers.

    Returns (cache, keys_by_window, resolver, consumer): the PWC serves
    the DECLARED pool paths while the strict resolver pins the staged
    copies -- no pool/HDD byte may serve.
    """
    from prismaquant.production_weight_cache import ProductionWeightCache
    staged = {}
    for pair, spec in files.items():
        declared = Path(spec["path"])
        target = strict._stage_root(tmp_path) / declared.name
        target.write_bytes(declared.read_bytes())
        staged[pair] = (declared, target, None)
    rows = {name: (declared, target, None)
            for name, (declared, target, _ram) in staged.items()}
    resolver, consumer, _mover = strict._leased_fixture(
        tmp_path, monkeypatch, rows)
    weights = {(pair[0], pair[1]): files[pair]["path"] for pair in files}
    cache = ProductionWeightCache(weights=weights, levers={})
    cache.enable_lru(1 << 20)
    digests = {pair: files[pair]["sha256"] for pair in files}
    cache.require_file_load_sha256(
        digests,
        max_file_bytes=max(spec["bytes"] for spec in files.values()))
    return cache, resolver, consumer


def test_strict_pwc_serves_two_windows_with_phase_order(
        tmp_path, monkeypatch):
    files = _render_files(tmp_path)
    cache, resolver, consumer = _leased_renders(
        tmp_path, monkeypatch, files)
    order: list[str] = []
    current = {"phase": "none"}
    real_prefetch = cache.prefetch

    def recording_prefetch(keys=None, max_workers=1):
        for key in (keys or ()):
            order.append(f"open:{current['phase']}:{key[0]}")
        return real_prefetch(keys, max_workers=max_workers)

    cache.prefetch = recording_prefetch

    def enter_phase(name):
        current["phase"] = name
        order.append(f"phase:{name}")

    for window, pairs in WINDOW_PAIRS:
        keys = [(name, fmt) for name, fmt in pairs]
        name = f"render-{window:02d}"
        # Readiness at the window boundary with the existing API: check
        # every selected file without deserializing, before the phase is
        # entered and the loading pool reads.
        cache.plan_retained_window(
            keys, max_resident_bytes=1 << 20, max_workers=1,
            max_load_buffer_bytes=1 << 20)
        # This isolated PWC fixture announces its phase directly. The
        # generator/dispatcher and actual production readiness callback
        # are exercised by test_stageb_prepared_inputs_bridge.py.
        enter_phase(name)
        with cache.retained_window(
                keys, max_resident_bytes=1 << 20, max_workers=1,
                max_load_buffer_bytes=1 << 20) as _receipt:
            for key in keys:
                tensor = cache.get(*key)
                assert torch.equal(
                    tensor,
                    torch.arange(8, dtype=torch.float32) + window)
                receipt = cache.file_load_receipt(key, tensor)
                assert receipt["serving_tier"] == "stage"
    assert order.index("phase:render-00") < order.index(
        "open:render-00:unit-w00-a")
    assert order.index("phase:render-01") < order.index(
        "open:render-01:unit-w01-a")
    report = resolver.report()
    assert report["bytes_from_pool"] == 0
    assert report["bytes_from_stage"] > 0
    assert report["serving_tiers"] and all(
        tier["pin_id"] for tier in report["serving_tiers"])
    assert strict._pins_live(tmp_path, consumer) == []


def test_strict_pwc_corrupt_staged_payload_fails_clear(
        tmp_path, monkeypatch):
    files = _render_files(tmp_path)
    cache, resolver, _consumer = _leased_renders(
        tmp_path, monkeypatch, files)
    pair = ("unit-w00-a", "FMT-A")
    staged = strict._stage_root(tmp_path) / Path(files[pair]["path"]).name
    raw = bytearray(staged.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    staged.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        with cache.retained_window(
                [pair], max_resident_bytes=1 << 20, max_workers=1,
                max_load_buffer_bytes=1 << 20):
            cache.prefetch([pair], max_workers=1)
    assert resolver.report()["bytes_from_pool"] == 0


def test_strict_pwc_undeclared_render_refuses(tmp_path, monkeypatch):
    files = _render_files(tmp_path)
    from prismaquant.production_weight_cache import ProductionWeightCache
    ghost = tmp_path / "renders" / "ghost.FMT-A.pt"
    torch.save(torch.arange(8, dtype=torch.float32), ghost)
    weights = {(pair[0], pair[1]): files[pair]["path"] for pair in files}
    weights[("ghost", "FMT-A")] = str(ghost)
    cache = ProductionWeightCache(weights=weights, levers={})
    cache.enable_lru(1 << 20)
    cache.require_file_load_sha256(
        {pair: files[pair]["sha256"] for pair in files} | {
            ("ghost", "FMT-A"): hashlib.sha256(
                ghost.read_bytes()).hexdigest()},
        max_file_bytes=max(
            [spec["bytes"] for spec in files.values()]
            + [ghost.stat().st_size]))
    # The ghost file is declared to PWC but staged NOWHERE: the strict
    # resolver must refuse before any pool byte serves.
    rows = {}
    for pair, spec in files.items():
        declared = Path(spec["path"])
        target = strict._stage_root(tmp_path) / declared.name
        target.write_bytes(declared.read_bytes())
        rows[pair] = (declared, target, None)
    resolver, _consumer, _mover = strict._leased_fixture(
        tmp_path, monkeypatch, rows)
    with pytest.raises(Exception):
        cache.prefetch([("ghost", "FMT-A")], max_workers=1)
    assert resolver.report()["bytes_from_pool"] == 0


def test_progress_reports_render_phases_without_pricing_units():
    from prismaquant.joint_cost_quantum import ChunkFrontier, QuantumProgress
    frontier = ChunkFrontier(
        chunks=[{"name": "layer-002-c000", "start_bytes": 0,
                 "end_bytes": 200}],
        windows=[{"render_file_upper_bound_bytes": 200}])
    progress = QuantumProgress(frontier=frontier, base_units=0,
                               log=lambda message: None)
    progress._phases = ["head", "checkpoint-load",
                        "chain-003-source", "chain-003-bound",
                        "own-002-source",
                        "render-00", "replay-00-p0", "replay-00-p1",
                        "render-01", "replay-01-p0", "replay-01-p1"]
    progress.enter_head(3)
    units = progress.units()
    progress.enter_read_phase("render-00")
    assert progress._phase == "render-00"
    assert progress.units() == units
    progress.enter_read_phase("replay-00-p0")
    assert progress._phase == "replay-00-p0"
    assert progress.units() == units
