"""The joint run says what it is doing, on a clock, in names the window matches.

Action ``ad8803aa`` (2026-09-18) printed one line between 16:24:17 and
18:41:42 while 23 040 durable entry files landed, and committed nothing at all
for the 4 h 14 min head walk before that.  PrismaBuild's residency window
advances on the consumer's accepted progress, so it advanced on nothing: 19 of
46 phases staged, 1 egressed, and a 744 GB stage reached 0 B available
(RobTand/prismabuild#632).

What these tests hold is the pair of properties that failure needed, not that
the writer called the helper:

* a unit moves only when a file is durable, and the count never regresses;
* the phase NAME the run commits under is one the sealed read plan carries --
  ``residency_plan.remaining`` reads any other name as "this consumer has
  passed nothing", which is a window that never advances and never releases.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch
import tessera_campaign_container as container

from prismaquant import joint_run_progress as jrp

REPO = Path(__file__).resolve().parents[1]


class Clock:
    """A clock the test moves, so the cadence is tested and not the wall."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Recorder:
    def __init__(self):
        self.lines = []
        self.records = []

    def log(self, message):
        self.lines.append(message)

    def commit(self, phase, units):
        self.records.append((phase, units))
        return True


def run_phases(layers):
    """The phase table a joint ``run`` read set seals, in read order."""
    return ("head", *(jrp.layer_phase_name(index) for index in range(layers)))


def reporter(layers=4, partitions=8, **kwargs):
    recorder = kwargs.pop("recorder", None) or Recorder()
    clock = kwargs.pop("clock", None) or Clock()
    return recorder, clock, jrp.JointRunProgress(
        layers=layers, partitions=partitions, log=recorder.log,
        clock=clock, commit=recorder.commit, interval_s=60.0,
        phases=kwargs.pop("phases", run_phases(layers)),
        resolver=kwargs.pop("resolver", None), **kwargs)


# -- the clock -------------------------------------------------------------

def test_the_line_is_emitted_on_the_clock_and_not_per_entry():
    recorder, clock, progress = reporter()
    for partition in range(8):
        progress.entry(layer=0, partition=partition)
    assert recorder.lines == []
    clock.advance(61)
    progress.entry(layer=0, partition=8)
    assert len(recorder.lines) == 1


def test_the_cadence_does_not_change_with_the_model_size():
    small = reporter(layers=4, partitions=8)
    large = reporter(layers=92, partitions=512)
    for recorder, clock, progress in (small, large):
        for index in range(200):
            progress.entry(layer=0, partition=index)
        assert recorder.lines == [], "a per-entry line would scale with the model"
        clock.advance(61)
        progress.entry(layer=0, partition=200)
        assert len(recorder.lines) == 1


def test_an_unreadable_interval_is_refused_rather_than_replaced(monkeypatch):
    monkeypatch.setenv(jrp.INTERVAL_ENV, "not-a-number")
    with pytest.raises(ValueError):
        jrp.interval_seconds()
    monkeypatch.setenv(jrp.INTERVAL_ENV, "0")
    with pytest.raises(ValueError):
        jrp.interval_seconds()
    monkeypatch.delenv(jrp.INTERVAL_ENV)
    assert jrp.interval_seconds() == jrp.DEFAULT_INTERVAL_S


# -- what the line says ----------------------------------------------------

class FakeResolver:
    def __init__(self):
        self.counters = {"hits": 0, "misses": 0, "range_hits": 0,
                         "range_misses": 0, "fallback_count": 0}

    def report(self):
        return dict(self.counters)


def test_the_line_carries_the_counts_a_reader_needs():
    resolver = FakeResolver()
    recorder, clock, progress = reporter(layers=4, partitions=8, resolver=resolver)
    for partition in range(5):
        progress.entry(layer=2, partition=partition)
    resolver.counters["range_hits"] = 7
    resolver.counters["misses"] = 2
    clock.advance(61)
    progress.flush(force=True)
    line = recorder.lines[-1]
    assert "layers 3/4" in line
    assert "partitions 5/8" in line
    assert "entries 5" in line
    assert "/s)" in line
    assert "range_hits+7" in line and "misses+2" in line
    assert "fallback_count+0" in line


def test_the_residency_counters_are_the_delta_since_the_last_line():
    resolver = FakeResolver()
    recorder, clock, progress = reporter(resolver=resolver)
    resolver.counters["range_hits"] = 10
    clock.advance(61)
    progress.flush(force=True)
    assert "range_hits+10" in recorder.lines[-1]
    resolver.counters["range_hits"] = 13
    clock.advance(61)
    progress.flush(force=True)
    assert "range_hits+3" in recorder.lines[-1]


def test_an_unset_residency_map_says_so_rather_than_printing_zeros(monkeypatch):
    from prismaquant import residency_map
    monkeypatch.setattr(residency_map, "residency_resolver", lambda: None)
    recorder, clock, progress = reporter(resolver=None)
    clock.advance(61)
    progress.flush(force=True)
    assert "residency unset" in recorder.lines[-1]


# -- the unit --------------------------------------------------------------

def test_a_unit_is_cumulative_across_phases_and_never_regresses():
    recorder, clock, progress = reporter()
    progress.base_units = 512           # what the head walk already committed
    seen = []
    for layer in range(4):
        for partition in range(8):
            progress.entry(layer=layer, partition=partition)
            seen.append(progress.units)
    assert seen == sorted(seen)
    assert seen[0] == 513 and seen[-1] == 512 + 32
    # The reverse pass journals priced units beside the cotangent entries.
    progress.priced_units(9)
    assert progress.units == 512 + 32 + 9
    progress.priced_units(4)            # a stale read may not take it back
    assert progress.units == 512 + 32 + 9


def test_nothing_is_committed_before_the_file_is_durable():
    recorder, clock, progress = reporter()
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == [], (
        "a run that has published no entry has committed no unit, and a phase "
        "named on entering a loop is the counter PB #480 refuses to buy time on")
    progress.entry(layer=0, partition=0)
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == [("layer-0", 1)]


def test_a_replayed_count_is_not_sent_to_the_worker():
    recorder, clock, progress = reporter()
    progress.entry(layer=0, partition=0)
    clock.advance(61)
    progress.flush(force=True)
    sent = len(recorder.records)
    clock.advance(61)
    progress.flush(force=True)
    assert len(recorder.records) == sent
    assert progress.suppressed >= 1
    assert len(recorder.lines) > sent, "the line still says the run is alive"


# -- the phase -------------------------------------------------------------

def test_every_committed_phase_is_one_the_read_plan_carries():
    phases = run_phases(4)
    recorder, clock, progress = reporter(layers=4, phases=phases)
    for layer in range(4):
        for partition in range(8):
            progress.entry(layer=layer, partition=partition)
        clock.advance(61)
        progress.flush(force=True)
    assert recorder.records, "a run that commits nothing is the defect under test"
    assert all(phase in phases for phase, _ in recorder.records)


def test_the_phase_index_never_moves_backwards_on_the_reverse_pass():
    phases = run_phases(4)
    recorder, clock, progress = reporter(layers=4, phases=phases)
    for layer in range(4):
        progress.entry(layer=layer, partition=0)
    assert progress.phase == "layer-3"
    # The reverse pass walks the layers down and writes cotangents as it goes.
    for layer in reversed(range(4)):
        progress.entry(layer=layer, partition=0, kind="cotangent")
        progress.enter(jrp.layer_phase_name(layer))
    assert progress.phase == "layer-3"
    indices = [phases.index(phase) for phase, _ in recorder.records]
    assert indices == sorted(indices)


def test_an_undeclared_phase_commits_nothing_and_says_why():
    recorder, clock, progress = reporter(layers=4, phases=("head", "layer-0"))
    progress.enter("cost_capture_002")
    assert progress.phase is None
    assert any("not one this action declared" in line for line in recorder.lines)
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == []


def test_a_launch_with_no_declared_phases_commits_nothing(monkeypatch):
    monkeypatch.delenv(jrp.PHASES_ENV, raising=False)
    assert jrp.declared_phases() is None
    recorder, clock, progress = reporter(phases=())
    progress.entry(layer=0, partition=0)
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records == []
    assert recorder.lines, "the log line is not conditional on the fleet"


def test_the_declared_list_is_read_as_data_from_the_environment(monkeypatch):
    monkeypatch.setenv(jrp.PHASES_ENV, json.dumps(["head", "layer-0"]))
    assert jrp.declared_phases() == ("head", "layer-0")
    for bad in ("", "[]", "{}", "not json", '["", "layer-0"]', '[1]'):
        monkeypatch.setenv(jrp.PHASES_ENV, bad)
        assert jrp.declared_phases() is None


def test_no_channel_is_byte_identical_to_this_module_being_absent(monkeypatch, tmp_path):
    from prismaquant import prismabuild_progress
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_PATH", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", raising=False)
    monkeypatch.delenv(jrp.PHASES_ENV, raising=False)
    assert prismabuild_progress.report("head", 1) is False
    clock = Clock()
    lines = []
    progress = jrp.JointRunProgress(layers=2, partitions=2, log=lines.append,
                                    clock=clock, interval_s=60.0)
    progress.entry(layer=0, partition=0)
    clock.advance(61)
    assert progress.flush(force=True) is True
    assert progress.commits == 0
    assert not list(tmp_path.iterdir())


# -- one definition of the name ---------------------------------------------

def test_the_manifest_and_the_reporter_share_one_layer_phase_name():
    source = (REPO / "experiments" / "glm_data_manifests.py").read_text()
    assert "_run_progress_module.layer_phase_name(layer)" in source
    assert 'f"layer-{layer}"' not in source, (
        "a second spelling of the phase name is how a consumer commits a name "
        "the residency plan cannot match")
    assert jrp.layer_phase_name(7) == "layer-7"


def test_the_joint_pass_builder_names_its_layers_through_that_function():
    """The phase table the window walks, checked where the run is submitted."""
    import ast

    path = REPO / "experiments" / "glm_data_manifests.py"
    source = path.read_text()
    tree = ast.parse(source)
    builder = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "build_joint_pass_manifest")
    body = ast.get_source_segment(source, builder)
    assert "_run_progress_module.layer_phase_name(layer)" in body
    assert 'f"layer-' not in body and "'layer-" not in body, (
        "the builder spells a layer phase name itself; the reporter would then "
        "commit under the other spelling and the window would match neither")


def test_a_joint_run_submission_declares_the_manifest_s_own_phase_names():
    manifest = {"annotations": {"phases": [
        {"name": name} for name in run_phases(3)]}}
    assert dispatch._declared_phase_names(manifest) == run_phases(3)


def test_an_oversized_read_plan_is_refused_rather_than_truncated():
    manifest = {"annotations": {"phases": [
        {"name": f"layer-{index}"} for index in range(2049)]}}
    with pytest.raises(RuntimeError, match="2048"):
        dispatch._declared_phase_names(manifest)


def test_the_run_command_declares_phases_at_all():
    source = (REPO / "tools" / "dispatch_tessera_campaign.py").read_text()
    assert 'command == "run":' in source and "_declared_phase_names(manifest)" in source


# -- the container boundary --------------------------------------------------

def test_the_container_carries_the_declared_phase_list_inside():
    spec = {"container": {"mounts": [
        {"source": "/mnt/shared", "target": "/mnt/shared"}]}}
    environ = {
        container.PATH_ENV: "/mnt/shared/pb-queue/claimed/k.progress",
        container.TOKEN_ENV: "tok",
        container.PHASES_ENV: json.dumps(["head", "layer-0"]),
    }
    assert container.progress_environment(spec, environ) == {
        container.PATH_ENV: "/mnt/shared/pb-queue/claimed/k.progress",
        container.TOKEN_ENV: "tok",
        container.PHASES_ENV: json.dumps(["head", "layer-0"]),
    }


def test_a_container_launch_without_a_channel_forwards_nothing():
    spec = {"container": {"mounts": [{"source": "/mnt/shared", "target": "/mnt/shared"}]}}
    assert container.progress_environment(spec, {}) == {}


# -- the durable boundary ----------------------------------------------------

def test_every_published_entry_file_moves_the_count(tmp_path):
    """The hook is at publication, which is where a unit becomes durable."""
    torch = pytest.importorskip("torch")
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 1})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    recorder, clock, progress = reporter(layers=3, partitions=2)
    storage.watch_progress(progress)
    for boundary in range(3):
        for batch in range(2):
            storage.write(torch.zeros(4), batch_index=batch, boundary_index=boundary)
    assert storage.telemetry["written_entries"] == 6
    assert progress.units == 6
    assert progress.phase == "layer-2"
    clock.advance(61)
    progress.flush(force=True)
    assert recorder.records[-1] == ("layer-2", 6)


def test_a_storage_with_no_reporter_behaves_exactly_as_before(tmp_path):
    torch = pytest.importorskip("torch")
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 1})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    storage.write(torch.zeros(4), batch_index=0, boundary_index=0)
    assert storage.telemetry["written_entries"] == 1
