"""A durable reverified prefix needs one cumulative publication, not N writes."""
import json
import os

from prismaquant import tessera_joint_aura as bridge
from prismaquant.cost_stage_checkpoint import unit_path
from prismaquant.production_weight_cache import _cache_weight_filename
from tests.test_tessera_joint_aura import fixture


def _case(tmp_path, monkeypatch):
    config, names, fmt, _, _ = fixture(tmp_path)
    journal = tmp_path / 'head-walk'
    bridge.load_measured_anchor_input(config, verify_payloads=False,
                                     head_checkpoint=journal, progress_phase=None)
    progress = tmp_path / 'progress.json'
    monkeypatch.setenv('PRISMABUILD_ACTION_PROGRESS_PATH', str(progress))
    monkeypatch.setenv('PRISMABUILD_ACTION_PROGRESS_TOKEN', 'test-token')
    monkeypatch.delenv('PRISMAQUANT_DEV_PROGRESS_STAMP', raising=False)
    writes = []
    replace = os.replace

    def record_replace(source, target):
        replace(source, target)
        if str(target) == str(progress):
            writes.append(json.loads(progress.read_text()))

    monkeypatch.setattr(os, 'replace', record_replace)
    return config, sorted(names), fmt, journal, writes


def test_full_resume_publishes_one_actual_cumulative_record(tmp_path, monkeypatch):
    config, names, _, journal, writes = _case(tmp_path, monkeypatch)
    resumed = bridge.load_measured_anchor_input(
        config, verify_payloads=False, head_checkpoint=journal, head_resume=True)
    assert resumed.head_walk_resumed_units == len(names)
    assert resumed.progress_committed == len(names)
    assert [(r['units_completed'], r['unit']) for r in writes] == [(len(names), names[-1])]


def test_drifted_suffix_is_not_reported_as_replayed(tmp_path, monkeypatch):
    config, names, fmt, journal, writes = _case(tmp_path, monkeypatch)
    render = tmp_path / 'campaign/rows/row-0000/cache' / _cache_weight_filename(names[1], fmt)
    render.write_bytes(render.read_bytes() + b'drift')
    real_load = bridge._load_unit

    def refuse_suffix(path, *, qname, **kwargs):
        if qname == names[1]:
            # Prefix publication happens after its fences pass, before any
            # unbanked suffix can claim a cumulative unit it has not finished.
            assert [r['units_completed'] for r in writes] == [1]
            raise ValueError('unverified suffix')
        return real_load(path, qname=qname, **kwargs)

    monkeypatch.setattr(bridge, '_load_unit', refuse_suffix)
    import pytest
    with pytest.raises(ValueError, match='unverified suffix'):
        bridge.load_measured_anchor_input(config, verify_payloads=False,
                                         head_checkpoint=journal, head_resume=True)
    assert [r['unit'] for r in writes] == [names[0]]


def test_corrupt_first_envelope_reports_no_replayed_prefix(tmp_path, monkeypatch):
    config, names, _, journal, writes = _case(tmp_path, monkeypatch)
    unit_path(journal, names[0]).write_bytes(b'corrupt')
    real_load = bridge._load_unit

    def fresh_only(path, *, qname, **kwargs):
        if qname == names[0]:
            assert writes == [], 'corrupt bank must not report replayed units'
        return real_load(path, qname=qname, **kwargs)

    monkeypatch.setattr(bridge, '_load_unit', fresh_only)
    resumed = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                               head_checkpoint=journal, head_resume=True,
                                               head_walk_workers=1)
    assert resumed.head_walk_resumed_units == 0
    assert [r['units_completed'] for r in writes] == [1, 2]


def _replay_drive_writes(monkeypatch, writes):
    """How many commits landed while the replay drive itself was running."""
    seen = []
    drive = bridge._drive_ordered_walk

    def observed(roster, walk, commit, **kwargs):
        try:
            return drive(roster, walk, commit, **kwargs)
        finally:
            seen.append(len(writes))

    monkeypatch.setattr(bridge, '_drive_ordered_walk', observed)
    return seen


def test_replay_longer_than_the_cadence_commits_during_the_drive(tmp_path, monkeypatch):
    # #1518: a --head-resume replay that outlives its cadence reports its
    # cumulative verified prefix while the drive runs, not only at its end.
    config, names, _, journal, writes = _case(tmp_path, monkeypatch)
    allowance_s = 0.2
    cadence_s = allowance_s / bridge.PROGRESS_CADENCE_SAFETY_FACTOR
    import time as _time
    real_sha = bridge._sha

    def slow_sha(path):
        # Every banked unit re-hashes its journal shard once; making that
        # slower than the cadence makes the replay outlive every window.
        _time.sleep(cadence_s * 1.5)
        return real_sha(path)

    monkeypatch.setattr(bridge, '_sha', slow_sha)
    during = _replay_drive_writes(monkeypatch, writes)
    resumed = bridge.load_measured_anchor_input(
        config, verify_payloads=False, head_checkpoint=journal, head_resume=True,
        head_walk_workers=1, progress_allowance_s=allowance_s)
    assert resumed.head_walk_resumed_units == len(names)
    units = [r['units_completed'] for r in writes]
    assert during[0] >= 2, f'the replay drive committed {during[0]} times while it ran'
    assert units == sorted(set(units)), 'cumulative counts never repeat or regress'
    assert units[0] == 1 and units[-1] == len(names)
    assert {r['phase'] for r in writes} == {'synthesize'}
    assert resumed.progress_committed == len(names)


def test_replay_cadence_is_the_allowance_over_the_stated_factor():
    cadence = bridge._ProgressCadence('synthesize', 600)
    assert bridge.PROGRESS_CADENCE_SAFETY_FACTOR == 4
    assert cadence.cadence_s == 600 / bridge.PROGRESS_CADENCE_SAFETY_FACTOR == 150.0
    assert not bridge._ProgressCadence('synthesize', None).active
    assert not bridge._ProgressCadence(None, 600).active
    import pytest
    for bad in (0, -1, float('inf'), float('nan'), '600', True):
        with pytest.raises(ValueError, match='progress_allowance_s'):
            bridge._ProgressCadence('synthesize', bad)


def test_replay_inside_one_window_writes_its_first_and_final_count_only(tmp_path, monkeypatch):
    # O(1) writes per window (#822's substance): a fast replay under a long
    # allowance commits the first verified unit and the final count, nothing per unit.
    config, names, _, journal, writes = _case(tmp_path, monkeypatch)
    resumed = bridge.load_measured_anchor_input(
        config, verify_payloads=False, head_checkpoint=journal, head_resume=True,
        head_walk_workers=1, progress_allowance_s=600)
    assert resumed.head_walk_resumed_units == len(names)
    assert [(r['units_completed'], r['unit']) for r in writes] == [(1, names[0]), (len(names), names[-1])]


def test_overlay_fence_continues_the_count_on_the_cadence(tmp_path, monkeypatch):
    # #1519: the candidate overlay's fence runs after the replay's last
    # commit. Its admitted cells continue the cumulative count on the same
    # cadence, and progress_committed carries the final count onward.
    from prismaquant import joint_catalog_extension as jce
    config, names, _, journal, writes = _case(tmp_path, monkeypatch)
    allowance_s = 0.2
    cadence_s = allowance_s / bridge.PROGRESS_CADENCE_SAFETY_FACTOR
    overlay_cells = 3
    calls = []

    def slow_overlay(data, bound, **kwargs):
        import time as _time
        calls.append(kwargs)
        progress = kwargs.get('progress')
        for index in range(overlay_cells):
            _time.sleep(cadence_s * 1.5)
            data.cells[names[0], f'OVERLAY_{index}'] = {'overlay': index}
            if progress is not None:
                progress(index + 1, f'{names[0]}@OVERLAY_{index}')
        return data

    monkeypatch.setattr(jce, 'attach_candidate_overlay', slow_overlay)
    config = dict(config, candidate_overlay={'path': 'fake', 'sha256': '0' * 64})
    before = len(writes)
    resumed = bridge.load_measured_anchor_input(
        config, verify_payloads=False, head_checkpoint=journal, head_resume=True,
        head_walk_workers=1, progress_allowance_s=allowance_s)
    assert calls and calls[0]['hash_workers'] == 1
    overlay = [r['units_completed'] for r in writes[before:] if r['units_completed'] > len(names)]
    assert len(overlay) >= 2, f'the overlay fence committed {len(overlay)} times'
    assert overlay[-1] == len(names) + overlay_cells
    units = [r['units_completed'] for r in writes]
    assert units == sorted(set(units))
    assert resumed.progress_committed == len(names) + overlay_cells
