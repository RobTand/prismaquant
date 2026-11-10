"""Selected native checkpoint retirement; historical filename retained for traceability."""
import json
from pathlib import Path
import pytest
from prismaquant import stage_a_retirement as retire
from prismaquant.stage_a_produced_output import BoundaryProducedBindingError
from test_stage_a_retirement_1073 import scientific_case, snapshot, _offline_tier_policy


def test_retirement_reclaims_owned_pins_and_committed_forward_batches(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    before = snapshot(case.root)
    pinned, _ = retire._checkpoint_planes(case.space, case.plan.record['session'])
    forward = retire._owned_forward_paths(json.loads(
        (case.space / 'chain-state.json').read_text()), space=case.space,
        session=case.plan.record['session'])
    targets = set(case.plan.entry_paths)
    assert set(pinned) <= targets and targets.intersection(forward)
    report = retire.apply_retirement(case.plan)
    assert report['entries_removed'] == len(targets)
    assert report['entry_bytes_removed'] == sum(len(before[path]) for path in targets)
    assert not any(Path(path).exists() for path in targets)
    assert all(Path(path).read_bytes() == raw for path, raw in before.items() if path not in targets)


def test_a_changed_pinned_file_refuses(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    pinned, _ = retire._checkpoint_planes(case.space, case.plan.record['session'])
    path = Path(pinned[0]); raw = path.read_bytes()
    # Keep the former inode live, so replacement cannot rely on inode reuse.
    displaced = path.with_suffix('.retained-original')
    path.rename(displaced); path.write_bytes(raw)
    before = snapshot(case.root)
    with pytest.raises(ValueError, match='native runtime file changed'):
        retire.plan_retirement(case.root, **case.arguments)
    assert snapshot(case.root) == before
    assert displaced.read_bytes() == raw and path.read_bytes() == raw
    assert not retire.retirement_record_path(case.space).exists()


def test_a_rerun_refuses_when_an_immutable_batch_record_is_gone(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    retire.apply_retirement(case.plan)
    sealed = retire.retirement_record_path(case.space).read_bytes()
    binding = next(iter(case.pub._groups.values()))
    path = Path(binding['path']); original = path.read_bytes(); path.unlink()
    before = snapshot(case.root)
    with pytest.raises((OSError, RuntimeError)):
        retire.plan_retirement(case.root, **case.arguments)
    assert snapshot(case.root) == before
    assert original and not path.exists()
    assert retire.retirement_record_path(case.space).read_bytes() == sealed


def test_new_binding_after_plan_refuses_apply_before_any_unlink(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    consumer = case.bindings / 'new-consumer.json'
    consumer.write_text(json.dumps({'checkpoint': case.plan.entry_paths[0]}))
    before = snapshot(case.root)
    with pytest.raises(retire.RetirementRefused, match='live binding'):
        retire.apply_retirement(case.plan)
    assert snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_replaced_target_directory_refuses_without_foreign_mutation(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    target = Path(case.plan.entry_paths[0])
    directory = target.parent
    original = directory.with_name(directory.name + '-retained')
    directory.rename(original)
    foreign = tmp_path / 'foreign-fixture'; foreign.mkdir()
    (foreign / target.name).write_bytes(b'independently known foreign target')
    (foreign / 'control.bin').write_bytes(b'retained control')
    directory.symlink_to(foreign, target_is_directory=True)
    before = snapshot(foreign)
    owned_before = snapshot(original)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        retire.apply_retirement(case.plan)
    assert snapshot(foreign) == before and snapshot(original) == owned_before
    assert directory.is_symlink()
    assert not retire.retirement_record_path(case.space).exists()
