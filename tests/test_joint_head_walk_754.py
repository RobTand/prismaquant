"""The joint head walk banks its progress and fans out over PB's placement (#754).

Every action restart re-paid the whole roster walk: the head resolved 36,423
units, committed each one's count to PrismaBuild, and banked nothing, so the
withdrawn campaign night paid the walk three times. Two properties close that,
and neither may bend the other:

* **Resumable** -- each verified unit is journalled under the campaign's own
  checkpoint machinery (``prepare_journal``/``write_unit``), keyed to the
  inputs it verified, and a resume re-verifies the banked prefix against the
  bytes it was banked from before trusting a single row of it. Unverifiable
  state is discarded and re-walked, never trusted.
* **Parallel** -- per-unit walks overlap on the CPUs PrismaBuild assigned the
  container, while commitment stays in the walk's one deterministic order, so
  the journal's banked units are always a roster prefix and the durable
  sequence means what it meant serially.

The fixture is the campaign intake fixture of ``test_tessera_joint_aura``:
two units, one measured rung each, both renders already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import unit_path
from tests.test_tessera_joint_aura import fixture


def _reports(monkeypatch, bridge):
    seen = []
    monkeypatch.setattr(bridge, "_pb_commit",
                        lambda units, phase, unit=None: seen.append((units, phase, unit)))
    return seen


def _fresh_state(data):
    """The equivalence target: everything downstream consumes from here."""
    return (data.cells, data.formats_by_qname, data.progress_committed,
            data.synthesized_now, dict(data.payload))


# -- resumability -----------------------------------------------------------


def test_a_walk_interrupted_midway_resumes_and_matches_a_fresh_walk(tmp_path, monkeypatch):
    """The defect: an interrupted walk banked nothing and restarted from zero."""
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    journal = tmp_path / "head-walk"
    seen = _reports(monkeypatch, bridge)
    fresh = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                              head_checkpoint=tmp_path / "fresh-walk")
    seen.clear()

    # An interruption on the second unit: the first is committed, and the
    # crash-window flush banks exactly the completed prefix.
    real_load = bridge._load_unit
    def break_on_second(path, *, qname, **kwargs):
        if qname == names[1]:
            raise RuntimeError("interrupted mid-walk")
        return real_load(path, qname=qname, **kwargs)
    monkeypatch.setattr(bridge, "_load_unit", break_on_second)
    with pytest.raises(RuntimeError, match="interrupted mid-walk"):
        bridge.load_measured_anchor_input(config, verify_payloads=False,
                                          head_checkpoint=journal)
    assert unit_path(journal, names[0]).is_file(), "the committed prefix is banked"
    assert not unit_path(journal, names[1]).exists()

    monkeypatch.setattr(bridge, "_load_unit", real_load)
    resumed = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                head_checkpoint=journal, head_resume=True)
    assert _fresh_state(resumed) == _fresh_state(fresh)
    assert resumed.head_walk_resumed_units == 1
    # The resumed walk re-verifies and re-reports the banked unit, in roster
    # order, never a count ahead of what it has re-checked.
    assert [count for count, _, _ in seen] == [1, 2]
    assert [unit for _, _, unit in seen] == sorted(names)


def test_a_corrupt_checkpoint_is_discarded_fail_closed_and_rewalked(tmp_path):
    """A checkpoint that fails its own verification is never trusted."""
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    journal = tmp_path / "head-walk"
    bridge.load_measured_anchor_input(config, verify_payloads=False, head_checkpoint=journal)
    unit_path(journal, names[1]).write_bytes(b"corrupt envelope")

    data = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                             head_checkpoint=journal, head_resume=True)
    assert data.head_walk_resumed_units == 0, "nothing unverifiable is reused"
    assert (tmp_path / "head-walk.stale").is_dir(), "the corrupt journal is kept aside"
    fresh = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                              head_checkpoint=tmp_path / "again")
    assert (data.cells, data.formats_by_qname, data.progress_committed) == \
           (fresh.cells, fresh.formats_by_qname, fresh.progress_committed)


def test_a_banked_unit_whose_inputs_drifted_is_rewalked_not_trusted(tmp_path):
    """The cursor re-verifies each banked unit against the bytes it names."""
    from prismaquant import tessera_joint_aura as bridge
    from prismaquant.production_weight_cache import _cache_weight_filename

    config, names, fmt, _payload, _states = fixture(tmp_path)
    journal = tmp_path / "head-walk"
    bridge.load_measured_anchor_input(config, verify_payloads=False, head_checkpoint=journal)
    render = tmp_path / "campaign/rows/row-0000/cache" / _cache_weight_filename(names[1], fmt)
    render.write_bytes(render.read_bytes() + b"drifted")

    data = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                             head_checkpoint=journal, head_resume=True)
    assert data.head_walk_resumed_units == 1, "the undrifted prefix re-verifies"
    fresh = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                              head_checkpoint=tmp_path / "again")
    assert (data.cells, data.formats_by_qname) == (fresh.cells, fresh.formats_by_qname)


def test_a_journal_keyed_to_another_plan_is_ignored_not_reused(tmp_path):
    """A mismatched key is a fresh walk, not a refusal and not a reuse."""
    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    journal = tmp_path / "head-walk"
    bridge.load_measured_anchor_input(config, verify_payloads=False, head_checkpoint=journal)
    manifest = json.loads((journal / "manifest.json").read_text())
    manifest["identity"]["census"] = "0" * 64
    (journal / "manifest.json").write_text(json.dumps(manifest))

    data = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                             head_checkpoint=journal, head_resume=True)
    assert data.head_walk_resumed_units == 0
    assert (tmp_path / "head-walk.stale").is_dir()


def test_a_journal_without_resume_is_refused_like_every_other_checkpoint(tmp_path):
    """The head journal is a checkpoint, not a cache: ``--resume`` opens it."""
    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    journal = tmp_path / "head-walk"
    bridge.load_measured_anchor_input(config, verify_payloads=False, head_checkpoint=journal)
    with pytest.raises(RuntimeError, match="pass --resume"):
        bridge.load_measured_anchor_input(config, verify_payloads=False,
                                          head_checkpoint=journal)


def test_a_scoped_read_cannot_bank_a_head_journal(tmp_path):
    """A partial roster is not the campaign's input and must not be journalled."""
    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    with pytest.raises(ValueError, match="scoped read"):
        bridge.load_measured_anchor_input(config, verify_payloads=False, unit_scope=(0, 2),
                                          head_checkpoint=tmp_path / "head-walk")


# -- parallelism ------------------------------------------------------------


def test_the_parallel_walk_commits_and_banks_exactly_what_the_serial_walk_does(
        tmp_path, monkeypatch):
    """However the walks finish, the durable sequence is the roster's order."""
    import os
    import time
    from prismaquant import tessera_joint_aura as bridge

    # The worker ceiling is the PB-assigned affinity by contract, so the test
    # states the assignment it fans out under rather than borrowing whatever
    # box the shard landed on.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    config, names, _fmt, _payload, _states = fixture(tmp_path)
    seen = _reports(monkeypatch, bridge)
    # The FIRST roster unit finishes last; the driver must still commit it
    # first, or the banked prefix and the reported counts would reorder.
    real_load = bridge._load_unit
    def slow_first(path, *, qname, **kwargs):
        if qname == names[0]:
            time.sleep(0.05)
        return real_load(path, qname=qname, **kwargs)
    monkeypatch.setattr(bridge, "_load_unit", slow_first)
    parallel = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                 head_checkpoint=tmp_path / "parallel-walk",
                                                 head_walk_workers=4)
    monkeypatch.setattr(bridge, "_load_unit", real_load)
    serial = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                               head_checkpoint=tmp_path / "serial-walk",
                                               head_walk_workers=1)
    assert _fresh_state(parallel) == _fresh_state(serial)
    assert [count for count, _, _ in seen] == [1, 2, 1, 2]
    assert [unit for _, _, unit in seen] == sorted(names) * 2


def test_worker_count_is_bounded_by_the_pb_assigned_affinity(monkeypatch):
    """PB owns placement; the walk never guesses cores it was not assigned."""
    import os
    from prismaquant import tessera_joint_aura as bridge

    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    assert bridge._head_walk_worker_count() == 2
    assert bridge._head_walk_worker_count(
        environ={bridge.HEAD_WALK_WORKERS_ENV: "1"}) == 1, "the A/B knob forces the serial path"
    assert bridge._head_walk_worker_count(1) == 1
    with pytest.raises(ValueError, match="affinity"):
        bridge._head_walk_worker_count(environ={bridge.HEAD_WALK_WORKERS_ENV: "3"})
    with pytest.raises(ValueError, match="affinity"):
        bridge._head_walk_worker_count(8)
    with pytest.raises(ValueError, match="worker count"):
        bridge._head_walk_worker_count(environ={bridge.HEAD_WALK_WORKERS_ENV: "x"})
    with pytest.raises(ValueError, match="worker count"):
        bridge._head_walk_worker_count(environ={bridge.HEAD_WALK_WORKERS_ENV: "0"})
    # A whole-box reservation does not mint a thread per unit.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)))
    assert bridge._head_walk_worker_count() == 16


def test_the_serial_driver_commits_each_unit_before_walking_the_next():
    from prismaquant.tessera_joint_aura import _drive_ordered_walk

    events = []
    _drive_ordered_walk(["a", "b"], lambda name: events.append(("walk", name)) or {},
                        lambda name, result: events.append(("commit", name)), workers=1)
    assert events == [("walk", "a"), ("commit", "a"), ("walk", "b"), ("commit", "b")]


def test_the_parallel_driver_commits_in_roster_order_however_walks_finish():
    import time
    from prismaquant.tessera_joint_aura import _drive_ordered_walk

    finishes = []
    def walk(name):
        time.sleep({"a": 0.05}.get(name, 0.0))
        finishes.append(name)
        return {}
    committed = []
    _drive_ordered_walk(["a", "b", "c"], walk,
                        lambda name, result: committed.append(name), workers=3)
    assert committed == ["a", "b", "c"]
    assert finishes and finishes[-1] == "a", "the roster's first unit finished last"


def test_a_failing_unit_stops_the_walk_with_its_prefix_committed():
    from prismaquant.tessera_joint_aura import _drive_ordered_walk

    def walk(name):
        if name == "b":
            raise ValueError("boom")
        return {}
    committed = []
    with pytest.raises(ValueError, match="boom"):
        _drive_ordered_walk(["a", "b", "c"], walk,
                            lambda name, result: committed.append(name), workers=2)
    assert committed == ["a"], "the bankable state is always a roster prefix"


def test_a_walk_without_a_checkpoint_reports_and_behaves_as_before(tmp_path, monkeypatch):
    """The journal is opt-in: no checkpoint, the same walk, the same reports."""
    import os
    from prismaquant import tessera_joint_aura as bridge
    from prismaquant.joint_prewarm_phases import HEAD_PHASE

    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)))
    config, names, _fmt, _payload, _states = fixture(tmp_path)
    seen = _reports(monkeypatch, bridge)
    data = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                             progress_phase=HEAD_PHASE,
                                             head_walk_workers=2)
    assert [count for count, _, _ in seen] == [1, 2]
    assert [unit for _, _, unit in seen] == sorted(names)
    assert data.progress_committed == 2
    assert data.head_walk_resumed_units == 0
    assert not (tmp_path / "head-walk").exists()
