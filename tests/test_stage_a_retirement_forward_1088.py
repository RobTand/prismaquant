"""Actual scientific forward records retire only exact selected owned files."""
import hashlib
import json
from pathlib import Path
import pytest
from prismaquant import stage_a_chain_resume as resume_mod, stage_a_retirement as retire
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.stage_a_produced_output import read_produced_selection
from test_stage_a_retirement_1073 import scientific_case, snapshot, _offline_tier_policy


def forward_case(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    state = json.loads(chain_state_path(case.space).read_text())
    owned = retire._owned_forward_paths(state, space=case.space, session=case.pub.session)
    verified = read_produced_selection(case.selected)
    groups = [group for group in verified.groups if
        all(row['path'] in owned for row in group['entries'])]
    assert groups and len(groups[0]['entries']) == 2
    paths = {row['path'] for group in groups for row in group['entries']}
    assert paths <= set(case.plan.entry_paths)
    return case, groups, paths


def rewrite_forward_records(case, mutate):
    path = chain_state_path(case.space)
    state = json.loads(path.read_text())
    mutate(state['boundary_entries'])
    path.write_text(json.dumps(resume_mod._seal(
        {k: v for k, v in state.items() if k != 'chain_state_sha256'})))


def test_unbound_forward_origin_batch_is_reclaimed_exactly(tmp_path, monkeypatch):
    case, groups, paths = forward_case(tmp_path, monkeypatch)
    before = snapshot(case.root)
    report = retire.apply_retirement(case.plan)
    assert set(group['batch_id'] for group in groups) <= set(report['batches'])
    assert not any(Path(path).exists() for path in paths)
    assert report['entry_bytes_removed'] == sum(len(before[p]) for p in case.plan.entry_paths)
    assert all(Path(p).read_bytes() == raw for p, raw in before.items() if p not in case.plan.entry_paths)


@pytest.mark.parametrize('name', ['seed.json', 'forward-capsule.json'])
def test_forward_path_binding_refuses_before_any_unlink(tmp_path, monkeypatch, name):
    case, _, paths = forward_case(tmp_path, monkeypatch)
    binding = case.bindings / name
    binding.write_text(json.dumps({'forward_inputs': [{'path': sorted(paths)[0]}]}))
    before = snapshot(case.root)
    with pytest.raises(retire.RetirementRefused, match='live binding') as caught:
        retire.plan_retirement(case.root, **case.arguments)
    assert name in str(caught.value) and snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_foreign_forward_records_do_not_admit_unlink(tmp_path, monkeypatch):
    case, groups, paths = forward_case(tmp_path, monkeypatch)
    def foreign(rows):
        for entries in rows.values():
            for entry in entries:
                identity = entry['metadata']['identity']
                identity['session'] = {**identity['session'], 'generation': 'foreign-generation'}
    rewrite_forward_records(case, foreign)
    plan = retire.plan_retirement(case.root, **case.arguments)
    assert not paths.intersection(plan.entry_paths)
    assert {group['batch_id'] for group in groups} <= set(plan.untouched)
    before = {p: Path(p).read_bytes() for p in paths}
    retire.apply_retirement(plan)
    assert {p: Path(p).read_bytes() for p in paths} == before


def test_preexisting_retirement_record_never_adds_forward_targets(tmp_path, monkeypatch):
    case, groups, paths = forward_case(tmp_path, monkeypatch)
    # A valid state may omit forward references. Seal its smaller target set.
    state_path = chain_state_path(case.space)
    state = json.loads(state_path.read_text())
    with monkeypatch.context() as patch:
        patch.setattr(retire, '_owned_forward_paths', lambda *a, **k: set())
        plan = retire.plan_retirement(case.root, **case.arguments)
        retire.apply_retirement(plan)
    sealed = retire.retirement_record_path(case.space).read_bytes()
    assert not paths.intersection(plan.entry_paths)
    resumed = retire.plan_retirement(case.root, **case.arguments)
    assert resumed.record == json.loads(sealed)
    assert retire.apply_retirement(resumed)['entries_removed'] == 0
    assert all(Path(p).is_file() for p in paths)
    assert retire.retirement_record_path(case.space).read_bytes() == sealed
    assert json.loads(state_path.read_text()) == state


@pytest.mark.parametrize('field', ['escaping-path', 'wrong-kind', 'missing-session'])
def test_malformed_owned_forward_refuses_before_unlink(tmp_path, monkeypatch, field):
    case, _, _ = forward_case(tmp_path, monkeypatch)
    def malformed(rows):
        entry = next(iter(rows.values()))[0]
        if field == 'escaping-path':
            entry['path'] = str(tmp_path / 'unrelated.pt')
        elif field == 'wrong-kind':
            entry['metadata']['identity']['kind'] = 'cotangent'
        else:
            entry['metadata']['identity']['session'] = None
    rewrite_forward_records(case, malformed)
    before = snapshot(case.root)
    with pytest.raises(retire.RetirementRefused, match='forward entry'):
        retire.plan_retirement(case.root, **case.arguments)
    assert snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_changed_forward_origin_refuses_before_unlink(tmp_path, monkeypatch):
    case, _, paths = forward_case(tmp_path, monkeypatch)
    path = Path(sorted(paths)[0]); path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b'changed')
    before = snapshot(case.root)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        retire.plan_retirement(case.root, **case.arguments)
    assert snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_mixed_forward_and_unsealed_tail_batch_refuses(tmp_path, monkeypatch):
    case, groups, _ = forward_case(tmp_path, monkeypatch)
    removed = groups[0]['entries'][0]['path']
    def remove_one(rows):
        for boundary, entries in rows.items():
            rows[boundary] = [entry for entry in entries if entry['path'] != removed]
    rewrite_forward_records(case, remove_one)
    before = snapshot(case.root)
    with pytest.raises(retire.RetirementRefused, match='mixes eligible and unrelated'):
        retire.plan_retirement(case.root, **case.arguments)
    assert snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_interrupted_forward_unlink_reuses_sealed_targets(tmp_path, monkeypatch):
    case, _, paths = forward_case(tmp_path, monkeypatch)
    original = retire.unlink_owned
    removed = []
    def interrupt(path, *args, **kwargs):
        if path in paths and removed:
            raise OSError('controlled forward retirement interruption')
        result = original(path, *args, **kwargs)
        if path in paths:
            removed.append(path)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(retire, 'unlink_owned', interrupt)
        with pytest.raises(OSError, match='controlled forward'):
            retire.apply_retirement(case.plan)
    sealed = retire.retirement_record_path(case.space).read_bytes()
    assert len(removed) == 1 and not Path(removed[0]).exists()
    plan = retire.plan_retirement(case.root, **case.arguments)
    assert plan.record == json.loads(sealed)
    retire.apply_retirement(plan)
    assert not any(Path(p).exists() for p in case.plan.entry_paths)
    assert retire.retirement_record_path(case.space).read_bytes() == sealed
