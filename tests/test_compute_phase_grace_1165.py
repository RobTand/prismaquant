"""A Stage B compute phase's grace covers its pass (PQ #1165).

``chain-NNN-bound`` rolls a chain layer's backward, ``spill-pP`` captures one
probe into the spill, a spill row's ``render-NN`` (NN >= 1) replays its window
from the spill, and a windowed row's ``replay-NN-pP`` replays one probe. Each
pass writes only disposable scratch (the cotangent scratch, the ``O_TMPFILE``
spill), which a resume recomputes, so no unit is durable and none commits
inside the pass. PrismaBuild's no-progress clock therefore runs for the whole
pass, and the phase's grace must be the pass's whole time budget: the read
term plus the pass's compute, measured per unit on the same regime.

Before this, every such phase took the legacy chunk lane's flat 900 s
(``CHUNK_PROGRESS_GRACE_S``), and R13's layer-043 row was ended as
``no_progress`` 967 s into a chain-044-bound roll of 2048 rows (PB
93247fc291c0).

The dispatcher tests drive ``dispatch_joint_quanta.main`` on a binder-sealed
row; the runtime test drives the real quantum and counts the units the
dispatcher prices.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_stageb_prepared_render_inputs import _dispatch_prepared  # noqa: E402

#: Spelled out, not imported: the red half must fail on the defect's own
#: assertion on a dispatcher that has no compute grace yet.
COMPUTE_STAMP = "prismaquant.compute_phase_grace.v1"
LOAD_STAMP = "prismaquant.load_phase_grace.v1"
FIXTURE_KEY = "f" * 64


def _ceiling(kind, unit_s, *, equal=None, at_most=None, tag="gb10"):
    scope = {"consumer_tag": tag, "equal": dict(equal or {}),
             "basis": "fixture: every row in scope takes no longer per unit"}
    if at_most is not None:
        scope["at_most"] = dict(at_most)
    return {"schema": "prismaquant.compute_unit_ceiling.v1", "kind": kind,
            "unit_s": unit_s, "method": "fixture", "scope": scope,
            "sources": [{"action_key": FIXTURE_KEY}]}


#: Seconds per unit far above the flat 900 s, so a grace that ignores the
#: pass cannot pass by accident.
FIXTURE_CEILINGS = (_ceiling("chain-roll", 400.0),
                    _ceiling("spill-capture", 500.0))


def _dispatch(tmp_path, monkeypatch, capsys, *, replay_mode, ceilings,
              extra=()):
    (dispatch, _bound, _receipt, manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch, replay_mode=replay_mode)
    monkeypatch.setattr(dispatch, "COMPUTE_UNIT_CEILINGS", tuple(ceilings),
                        raising=False)
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path), *extra], _gateway=gateway,
        # No checkpoint behind the fixture; coverage has its own tests.
        _coverage=lambda rows: [])
    assert code == 0, capsys.readouterr().err
    (row,) = gateway.submitted
    return dispatch, manifest, row["argv"]


def _graces(argv):
    return {name: int(value) for name, value in (
        argv[i + 1].split("=", 1) for i, word in enumerate(argv)
        if word == "--progress-phase")}


def _stamps(argv):
    flag = "--progress-grace-derivation"
    return json.loads(argv[argv.index(flag) + 1]) if flag in argv else []


def _plan(manifest):
    return {phase["name"]: phase for phase in manifest["read_plan"]["phases"]}


def _read_term(dispatch, phase_bytes):
    """W + ceil(bytes / floor): the fixture dispatch has one row, so the
    built-in one-reader floor applies."""
    from prismaquant.residency_shard_reader import staged_range_wait_from_env

    env = json.loads(Path(dispatch.SPEC_PATH).read_text()).get("env") or {}
    floor = dispatch.LOAD_PHASE_FLOOR_ONE_READER["floor_bytes_per_s"]
    return math.ceil(staged_range_wait_from_env(env)) + -(-phase_bytes // floor)


# -- red: the defect, at the dispatcher ------------------------------------


def test_chain_roll_and_spill_capture_graces_cover_their_pass(
        tmp_path, monkeypatch, capsys):
    """Every chain-*-bound and spill-p* grace is at least its read term plus
    its units at the measured seconds per unit."""
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS)
    graces, plan = _graces(argv), _plan(manifest)
    n_probes = manifest["annotations"]["n_probes"]
    unit_s = {doc["kind"]: doc["unit_s"] for doc in FIXTURE_CEILINGS}
    passes = {}
    for name, phase in plan.items():
        entries = len(phase["entry_indices"])
        if name.startswith("chain-") and name.endswith("-bound"):
            passes[name] = ("chain-roll", n_probes * entries)
        elif name.startswith("spill-p"):
            # The fixture launches the default regime: capture batch 1.
            passes[name] = ("spill-capture", entries)
    assert {kind for kind, _units in passes.values()} == {
        "chain-roll", "spill-capture"}, sorted(plan)
    for name, (kind, units) in sorted(passes.items()):
        need = (_read_term(dispatch, plan[name]["bytes"])
                + math.ceil(units * unit_s[kind]))
        assert graces[name] >= need, (
            f"{name}: a {graces[name]} s grace cannot cover its pass, which "
            f"commits no unit: read {_read_term(dispatch, plan[name]['bytes'])} s "
            f"+ {units} {kind} units x {unit_s[kind]} s = {need} s")
    stamps = {stamp.get("phase"): stamp for stamp in _stamps(argv)}
    for name in passes:
        assert stamps.get(name, {}).get("schema") == COMPUTE_STAMP, (
            f"{name} carries no compute grace stamp")


def test_spill_window_replays_carry_their_pass(tmp_path, monkeypatch, capsys):
    """A spill row replays windows 1 and later under their render phase: the
    replay is unmeasured, so the grace adds the blanket and says so."""
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS)
    graces, plan = _graces(argv), _plan(manifest)
    replays = [name for name in plan if name.startswith("render-")
               and int(name.split("-")[1]) >= 1]
    assert replays, sorted(plan)
    blanket = dispatch.HEAD_PROGRESS_GRACE_S
    for name in replays:
        need = _read_term(dispatch, plan[name]["bytes"]) + blanket
        assert graces[name] >= need, (
            f"{name}: a {graces[name]} s grace cannot cover the window's "
            f"spill replay (read + the {blanket} s blanket = {need} s)")
    stamps = {stamp.get("phase"): stamp for stamp in _stamps(argv)}
    last = list(plan)[-1]
    for name in replays:
        term, *rest = stamps[name]["compute"]
        assert (term["kind"], term["mode"]) == ("spill-replay", "blanket")
        assert "replay_wall_s" in term["reason"]
        # The last declared phase also runs the quantum's tail (PQ #1190).
        assert [item["kind"] for item in rest] == (
            ["tail"] if name == last else [])
    # render-00 only reads: it keeps the chunk grace and no stamp.
    assert graces["render-00"] == dispatch.CHUNK_PROGRESS_GRACE_S
    assert "render-00" not in stamps


def test_windowed_replays_carry_their_pass(tmp_path, monkeypatch, capsys):
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode=None,
        ceilings=FIXTURE_CEILINGS)
    graces, plan = _graces(argv), _plan(manifest)
    replays = [name for name in plan if name.startswith("replay-")]
    assert replays, sorted(plan)
    for name in replays:
        need = (_read_term(dispatch, plan[name]["bytes"])
                + dispatch.HEAD_PROGRESS_GRACE_S)
        assert graces[name] >= need, (
            f"{name}: a {graces[name]} s grace cannot cover a replay "
            f"backward over {len(plan[name]['entry_indices'])} stored batches")
    for name in plan:
        if name.startswith("render-"):
            assert graces[name] == dispatch.CHUNK_PROGRESS_GRACE_S, name


def test_a_row_outside_the_measured_scope_takes_the_blanket_and_says_why(
        tmp_path, monkeypatch, capsys):
    ceilings = (_ceiling("chain-roll", 400.0, equal={"n_probes": 99}),
                _ceiling("spill-capture", 500.0, tag="some-other-class"))
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill", ceilings=ceilings)
    graces, plan = _graces(argv), _plan(manifest)
    blanket = dispatch.HEAD_PROGRESS_GRACE_S
    bound = next(name for name in plan if name.endswith("-bound"))
    assert graces[bound] == _read_term(dispatch, plan[bound]["bytes"]) + blanket, (
        f"{bound}: outside every measurement, the roll takes the {blanket} s "
        "blanket on top of its read term")
    stamps = {stamp.get("phase"): stamp for stamp in _stamps(argv)}
    (term,) = stamps[bound]["compute"]
    assert term["mode"] == "blanket" and term["grace_s"] == blanket
    n_probes = manifest["annotations"]["n_probes"]
    assert f"n_probes is {n_probes}, measured at 99" in term["reason"]
    capture = stamps["spill-p0"]["compute"][0]
    assert capture["kind"] == "spill-capture" and capture["mode"] == "blanket"
    assert "'some-other-class'" in capture["reason"]
    assert stamps[bound]["mode"] == "blanket"


# -- the derivation, pinned ------------------------------------------------


def test_no_compute_phase_carries_the_load_bound_alone(
        tmp_path, monkeypatch, capsys):
    import dispatch_joint_quanta as dispatch

    _dispatch_module, _manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS)
    stamps = _stamps(argv)
    loads = [stamp["phase"] for stamp in stamps
             if stamp.get("schema") == LOAD_STAMP]
    assert loads == ["checkpoint-load"]
    computes = [stamp for stamp in stamps if stamp.get("schema") == COMPUTE_STAMP]
    graces = _graces(argv)
    for stamp in computes:
        assert stamp["compute"], stamp["phase"]
        assert stamp["grace_s"] == graces[stamp["phase"]] == (
            stamp["read"]["grace_s"]
            + sum(term["grace_s"] for term in stamp["compute"]))
    (basis,) = [stamp for stamp in stamps if stamp.get("schema")
                == dispatch.COMPUTE_GRACE_BASIS_SCHEMA]
    assert basis["bound"] == dispatch.COMPUTE_PHASE_BOUND
    assert sorted(basis["ceilings"]) == ["chain-roll", "spill-capture"]
    assert basis["floor"]["floor_bytes_per_s"] == (
        dispatch.LOAD_PHASE_FLOOR_ONE_READER["floor_bytes_per_s"])


def test_a_compute_ceiling_file_replaces_the_built_in_of_its_kind(
        tmp_path, monkeypatch, capsys):
    document = _ceiling("chain-roll", 7.0)
    path = tmp_path / "chain-roll-ceiling.json"
    path.write_text(json.dumps(document))
    dispatch, manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill",
        ceilings=FIXTURE_CEILINGS, extra=("--compute-ceiling", str(path)))
    stamps = _stamps(argv)
    bound = next(stamp for stamp in stamps
                 if stamp.get("phase", "").endswith("-bound"))
    (term,) = bound["compute"]
    assert (term["mode"], term["unit_s"]) == ("derived", 7.0)
    basis = stamps[-1]
    assert basis["ceilings"]["chain-roll"]["document"]["path"] == str(path)
    assert basis["ceilings"]["spill-capture"]["unit_s"] == 500.0


@pytest.mark.parametrize("edit,why", [
    (lambda d: d.update(kind="render"), "kind must be one of"),
    (lambda d: d.update(unit_s=0), "unit_s must be a positive"),
    (lambda d: d.update(unit_s=True), "unit_s must be a positive"),
    (lambda d: d["scope"].pop("basis"), "scope.basis"),
    (lambda d: d["scope"].update(at_most={"x": "1"}), "at_most values"),
    (lambda d: d.update(sources=[{"action_key": "short"}]), "64-hex"),
])
def test_a_malformed_ceiling_refuses(edit, why):
    import dispatch_joint_quanta as dispatch

    document = _ceiling("chain-roll", 1.0)
    edit(document)
    with pytest.raises(dispatch.DispatchRefused, match=why):
        dispatch.compute_unit_ceiling(document)


def test_phase_work_follows_the_runtime_order():
    import dispatch_joint_quanta as dispatch

    work = dispatch.compute_phase_work
    common = {"entries": 512, "n_probes": 4, "capture_batch": 4}
    (roll,) = work("chain-044-bound", replay_mode="spill", **common)
    assert (roll["kind"], roll["units"]) == ("chain-roll", 2048)
    capture, first_replay = work("spill-p2", replay_mode="spill", **common)
    assert (capture["kind"], capture["units"]) == ("spill-capture", 128)
    assert (first_replay["kind"], first_replay["units"]) == ("spill-replay", 1)
    (replay,) = work("render-03", replay_mode="spill", **common)
    assert (replay["kind"], replay["units"]) == ("spill-replay", 4)
    (windowed,) = work("replay-03-p1", replay_mode="windowed", **common)
    assert (windowed["kind"], windowed["units"]) == ("windowed-replay", 512)
    for name, mode in (("render-00", "spill"), ("render-03", "windowed"),
                       ("head", "spill"), ("checkpoint-load", "spill"),
                       ("chain-044-source", "spill"), ("own-044-source", "spill"),
                       ("spill-p0", "windowed"), ("replay-00-p0", "spill")):
        assert work(name, replay_mode=mode, **common) is None, (name, mode)
    # Uncounted units stay uncounted; the term then takes the blanket.
    (roll,) = work("chain-044-bound", replay_mode="spill", entries=None,
                   n_probes=4, capture_batch=4)
    assert roll["units"] is None


# The R13 row shape, from the sealed layer-044 and layer-043 records
# (meta-045-fae344d): W = 300 s, one link reader.
V7_CONTEXT = {
    "consumer_tags": ["gb10"], "n_probes": 4, "entry_bytes": 16779369,
    "replay_regime.capture_batch": 4, "replay_regime.accumulation": "operator_gemm",
    "replay_regime.chunk_rows": 65536, "spill.capture_batch": 4,
    "spill.element_dtype": "bfloat16", "spill.block": 4096,
    "spill.geometry.batch_bytes": 629145600,
    "spill.geometry.batch_x_bytes": 327155712, "spill.geometry.element_size": 2,
    "spill.geometry.experts_per_token": 8, "spill.geometry.largest_tensor_bytes": 16777216,
    "spill.geometry.max_batch_tokens": 2048, "spill.geometry.n_probes": 4,
    "chain_regime.batch_size": 4, "chain_regime.probe_fusion": True,
}


def _r13_grace(name, *, phase_bytes, entries, context=V7_CONTEXT):
    import dispatch_joint_quanta as dispatch

    work = dispatch.compute_phase_work(
        name, replay_mode="spill", entries=entries, n_probes=4, capture_batch=4)
    return dispatch.compute_phase_grace(
        name, work=work, phase_bytes=phase_bytes, staged_wait_s=300.0,
        link=dispatch.link_readers(rows=1), context=context,
        ceilings=dispatch.compute_ceilings())


def test_the_r13_row_shape_derives_from_the_built_in_measurements():
    bound = _r13_grace("chain-044-bound", phase_bytes=8_591_036_928, entries=512)
    assert (bound["mode"], bound["grace_s"]) == ("derived", 300 + 137 + 1686)
    spill = _r13_grace("spill-p0", phase_bytes=8_591_036_928, entries=512)
    assert [term["grace_s"] for term in spill["compute"]] == [4393, 1800]
    assert (spill["mode"], spill["grace_s"]) == ("blanket", 300 + 137 + 4393 + 1800)
    render = _r13_grace("render-01", phase_bytes=5_940_033_993, entries=354)
    assert (render["mode"], render["grace_s"]) == ("blanket", 300 + 95 + 1800)
    # Layer 43 spills less per batch than the measured layer 44: in scope.
    smaller = {**V7_CONTEXT, "spill.geometry.batch_bytes": 545259520,
               "spill.geometry.batch_x_bytes": 243269632}
    spill = _r13_grace("spill-p0", phase_bytes=8_591_036_928, entries=512,
                       context=smaller)
    assert spill["compute"][0]["mode"] == "derived"
    # More per batch, another regime or another device class: out of scope.
    for edit in ({"spill.geometry.batch_bytes": 629145601},
                 {"replay_regime.chunk_rows": 32768},
                 {"consumer_tags": ["x86"]}):
        spill = _r13_grace("spill-p0", phase_bytes=8_591_036_928, entries=512,
                           context={**V7_CONTEXT, **edit})
        assert spill["compute"][0]["mode"] == "blanket", edit
    for edit in ({"chain_regime.probe_fusion": False}, {"entry_bytes": 1}):
        bound = _r13_grace("chain-044-bound", phase_bytes=8_591_036_928,
                           entries=512, context={**V7_CONTEXT, **edit})
        assert bound["compute"][0]["mode"] == "blanket", edit


# -- the runtime: what the dispatcher prices is what the quantum runs -------


def test_the_quantum_commits_nothing_inside_a_pass_and_runs_the_priced_units(
        tmp_path, monkeypatch):
    """The real quantum, spill mode, one chain layer: no report lands between
    the first and last unit of a chain roll or a spill capture, and the units
    it runs under each phase are the units the dispatcher prices for it."""
    import dispatch_joint_quanta as dispatch
    import test_quantum_executable_readset as phases
    import test_stageb_one_pass_spill as one_pass
    import prismaquant.joint_adjoint_checkpoints as checkpoints
    import prismaquant.joint_replay_spill as spill_mod

    monkeypatch.setattr(phases, "_expert_fixture", one_pass._bf16_expert_fixture)
    for name in spill_mod.SPILL_ENV:
        monkeypatch.delenv(name, raising=False)
    setup = phases._acceptance_setup_expert(tmp_path, monkeypatch)
    record0 = setup["records"]["layer-000"]
    assert list(record0["adjoint"]["chain_layers"]) == [1]
    spill_root = one_pass._spill_root(tmp_path)
    monkeypatch.setenv(spill_mod.SPILL_ENV[0], str(spill_root))
    monkeypatch.setenv(spill_mod.SPILL_ENV[1], str(1 << 30))

    import prismaquant.joint_cost_quantum as qc

    log = []

    def current():
        return next((event[1] for event in reversed(log)
                     if event[0] == "report"), None)

    original_commit = qc.QuantumProgress.commit

    def logging_commit(self):
        advanced = original_commit(self)
        if advanced:
            log.append(("report", self._phase, self.units()))
        return advanced

    monkeypatch.setattr(qc.QuantumProgress, "commit", logging_commit)
    original_roll = checkpoints.render_free_layer_roll

    def counting_roll(*args, roll, **kwargs):
        def counted(tensor, batch, probe):
            log.append(("unit", current(), "chain-roll"))
            return roll(tensor, batch, probe)
        return original_roll(*args, roll=counted, **kwargs)

    monkeypatch.setattr(checkpoints, "render_free_layer_roll", counting_roll)
    original_end = spill_mod._SpillObserver.end_batch

    def counting_end(self):
        log.append(("unit", current(), "spill-capture"))
        return original_end(self)

    monkeypatch.setattr(spill_mod._SpillObserver, "end_batch", counting_end)
    _events, manifest, _payload, _resolved = phases._drive_quantum(
        tmp_path, monkeypatch, setup, layer=0, resume=False,
        replay_mode="spill")
    assert not os.listdir(spill_root)
    shutil.rmtree(spill_root, ignore_errors=True)

    plan = _plan(manifest)
    n_probes = manifest["annotations"]["n_probes"]
    ran = {}
    for event in log:
        if event[0] == "unit":
            ran.setdefault((event[1], event[2]), []).append(event)
    priced = {}
    for name, phase in plan.items():
        for term in dispatch.compute_phase_work(
                name, replay_mode="spill",
                entries=len(phase["entry_indices"]), n_probes=n_probes,
                capture_batch=1) or ():
            if term["kind"] in ("chain-roll", "spill-capture"):
                priced[(name, term["kind"])] = term["units"]
    assert {kind for _name, kind in priced} == {"chain-roll", "spill-capture"}
    assert {key: len(units) for key, units in ran.items()} == priced
    # Nothing commits between a pass's first and last unit.
    for key in priced:
        indices = [i for i, event in enumerate(log)
                   if event[0] == "unit" and (event[1], event[2]) == key]
        inside = [log[i] for i in range(indices[0], indices[-1])
                  if log[i][0] == "report"]
        assert inside == [], (key, inside)
