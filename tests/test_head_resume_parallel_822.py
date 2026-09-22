"""Resume authenticates independent files concurrently, committing only a prefix."""
import os
from pathlib import Path
import threading

from prismaquant import cost_stage_checkpoint as checkpoint
from prismaquant import tessera_joint_aura as bridge
from tests.test_tessera_joint_aura import fixture


def _overlap(reader):
    lock = threading.Lock()
    both = threading.Event()
    state = {'active': 0, 'peak': 0}

    def read(*args, **kwargs):
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            if state['active'] == 2:
                both.set()
        try:
            # Model independent blocked filesystem reads. No duration is a
            # pass criterion; simultaneous live authentication is the claim.
            both.wait(0.5)
            return reader(*args, **kwargs)
        finally:
            with lock:
                state['active'] -= 1
    return read, state


def _case(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0, 1})
    config, names, _, _, _ = fixture(tmp_path)
    journal = tmp_path / 'head-walk'
    baseline = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                head_checkpoint=journal,
                                                progress_phase=None)
    return config, names, journal, baseline


def test_resume_envelopes_overlap_without_changing_verified_rows(tmp_path, monkeypatch):
    config, names, journal, baseline = _case(tmp_path, monkeypatch)
    read, concurrency = _overlap(checkpoint._load_unit)
    monkeypatch.setattr(checkpoint, '_load_unit', read)
    resumed = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                head_checkpoint=journal, head_resume=True,
                                                head_walk_workers=2, progress_phase=None)
    assert resumed.cells == baseline.cells
    assert resumed.head_walk_resumed_units == len(names)
    assert concurrency['peak'] == 2


def test_resume_fences_overlap_without_skipping_source_hashes(tmp_path, monkeypatch):
    config, names, journal, baseline = _case(tmp_path, monkeypatch)
    parts = tmp_path / 'merged/cost.anchors.json.parts'
    paths = {checkpoint.unit_path(parts, name) for name in names}
    original = bridge._sha
    read, concurrency = _overlap(original)
    hashed = []

    def guarded(path):
        if Path(path) in paths:
            hashed.append(Path(path))
            return read(path)
        return original(path)

    monkeypatch.setattr(bridge, '_sha', guarded)
    resumed = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                head_checkpoint=journal, head_resume=True,
                                                head_walk_workers=2, progress_phase=None)
    assert resumed.cells == baseline.cells
    assert resumed.head_walk_resumed_units == len(names)
    assert sorted(hashed) == sorted(paths)
    assert concurrency['peak'] == 2
