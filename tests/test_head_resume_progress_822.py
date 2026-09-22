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
