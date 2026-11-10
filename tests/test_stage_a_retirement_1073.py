"""Retire real scientific checkpoint files through selected application batches.

Terminal execution/containment is an explicit offline seam here. The admitted
native controller proves those boundaries separately. Checkpoint serialization,
scientific capture, application publication, binding scan and exact unlink are real.
"""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from prismaquant import stage_a_chain_resume as resume_mod, stage_a_retirement as retire
from prismaquant.joint_adjoint_band import BandRefused, build_band_receipt
from prismaquant.joint_adjoint_checkpoints import adjoint_space, checkpoint_directory, write_adjoint_receipt
from prismaquant.stage_a_chain_resume import ChainResumeRefused, chain_state_path
from prismaquant.stage_a_produced_output import read_produced_selection, publish_document
from produced_native_fixtures import publication, selected_publications
from test_stage_a_chain_resume import DIGESTS, TWO, _at, _interrupted, _run, _offline_tier_policy


def scientific_case(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    pub = publication(tmp_path / 'launch', output_prefix=root, group_size=2, tensor_bytes=256,
        payload_max=1 << 28, spool_bytes=1 << 25)
    register = selected_publications(monkeypatch)
    selected = []
    def finalize(*args, **kwargs):
        selection, result = register(pub)
        selected.append(selection)
        return result
    monkeypatch.setattr(pub, 'finalize', finalize)
    space = adjoint_space(root)
    receipt = _run(root, monkeypatch, produced_output=pub)
    write_adjoint_receipt(space, receipt)
    assert len(selected) == 1
    verified = read_produced_selection(selected[0])
    assert verified.groups and verified.index['session'] == receipt['boundary_storage']['session']
    checkpoint_paths, _ = retire._checkpoint_planes(space, verified.index['session'])
    assert checkpoint_paths and all(Path(p).is_file() for p in checkpoint_paths)
    succ_root = tmp_path / 'successor'
    _interrupted(succ_root, monkeypatch, implementation=TWO, generation=1002, interrupt=_at(3, 1, 2))
    successor = chain_state_path(adjoint_space(succ_root))
    bindings = tmp_path / 'bindings'; bindings.mkdir()
    monkeypatch.setattr(resume_mod, 'require_producer_contained', lambda producer: None)
    arguments = {'successor_path': successor,
        'successor_sha256': hashlib.sha256(successor.read_bytes()).hexdigest(),
        'binding_roots': [bindings],
        'producer_selections': [{'selection': selected[0], 'checkpoint_exit': None}]}
    plan = retire.plan_retirement(root, **arguments)
    assert set(checkpoint_paths) <= set(plan.entry_paths)
    return SimpleNamespace(root=root, space=space, pub=pub, selected=selected[0],
        arguments=arguments, bindings=bindings, plan=plan)


def snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_a_superseded_scientific_run_retires_exact_selected_files_and_refuses_reuse(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    before = snapshot(case.root)
    target = set(case.plan.entry_paths)
    assert target and target < set(before)
    record = dict(case.plan.record)
    report = retire.apply_retirement(case.plan)
    assert report['entries_removed'] == len(target)
    assert not any(Path(path).exists() for path in target)
    assert {path: raw for path, raw in snapshot(case.root).items()
        if path in before} == {path: raw for path, raw in before.items() if path not in target}
    sealed = retire.retirement_record_path(case.space).read_bytes()
    assert json.loads(sealed) == record
    assert retire.apply_retirement(retire.plan_retirement(case.root, **case.arguments))['entries_removed'] == 0
    assert retire.retirement_record_path(case.space).read_bytes() == sealed
    with pytest.raises(ChainResumeRefused, match='retires this Stage A run'):
        resume_mod.plan_chain_resume(case.space, json.loads(chain_state_path(case.space).read_text()),
            recomputed={}, running_implementation_sha256='1' * 64)
    with pytest.raises(BandRefused, match='retires this Stage A run'):
        build_band_receipt(output_root=case.root, boundary=2, stride_value=2, bind_identity={}, **DIGESTS)


@pytest.mark.parametrize('kind', ['seed-spec', 'band-by-digest', 'stage-b-slice-gz'])
def test_a_live_binding_refuses_before_any_unlink(tmp_path, monkeypatch, kind):
    case = scientific_case(tmp_path, monkeypatch)
    if kind == 'seed-spec':
        data = {'checkpoint': {'path': str(case.space / 'checkpoints')}}
    elif kind == 'band-by-digest':
        data = {'checkpoint': next(iter(case.plan.record['digests']))}
    else:
        data = {'entries': [case.plan.entry_paths[0]]}
    path = case.bindings / ('consumer.json.gz' if kind.endswith('gz') else 'consumer.json')
    raw = json.dumps(data).encode()
    path.write_bytes(gzip.compress(raw) if kind.endswith('gz') else raw)
    before = snapshot(case.root)
    with pytest.raises(retire.RetirementRefused, match='live binding') as caught:
        retire.plan_retirement(case.root, **case.arguments)
    assert path.name in str(caught.value) and snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


@pytest.mark.parametrize('mode', ['same-identity', 'inside-run', 'live-producer', 'zero-roots', 'missing-selection', 'wrong-producer'])
def test_every_successor_containment_and_selection_join_precedes_unlink(tmp_path, monkeypatch, mode):
    case = scientific_case(tmp_path, monkeypatch)
    args = dict(case.arguments)
    if mode in ('same-identity', 'inside-run'):
        original = chain_state_path(case.space)
        path = original if mode == 'inside-run' else tmp_path / 'same-chain.json'
        if path != original:
            path.write_bytes(original.read_bytes())
        args.update(successor_path=path, successor_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    elif mode == 'live-producer':
        def live(producer):
            raise ChainResumeRefused('the owner is still running')
        monkeypatch.setattr(resume_mod, 'require_producer_contained', live)
    elif mode == 'zero-roots':
        args['binding_roots'] = []
    elif mode == 'missing-selection':
        args['producer_selections'] = []
    else:
        original = retire._producers
        monkeypatch.setattr(retire, '_producers', lambda *a: [{**original(*a)[0], 'job_id': '124'}])
    before = snapshot(case.root)
    with pytest.raises((retire.RetirementRefused, ChainResumeRefused)):
        retire.plan_retirement(case.root, **args)
    assert snapshot(case.root) == before
    assert not retire.retirement_record_path(case.space).exists()


def test_interrupted_retirement_resumes_only_its_immutable_targets(tmp_path, monkeypatch):
    case = scientific_case(tmp_path, monkeypatch)
    original = retire.unlink_owned
    removed = []
    def interrupt(path, *args, **kwargs):
        if removed:
            raise OSError('controlled interruption after first unlink')
        result = original(path, *args, **kwargs)
        removed.append(path)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(retire, 'unlink_owned', interrupt)
        with pytest.raises(OSError, match='controlled interruption'):
            retire.apply_retirement(case.plan)
    sealed = retire.retirement_record_path(case.space).read_bytes()
    assert len(removed) == 1 and not Path(removed[0]).exists()
    resumed = retire.plan_retirement(case.root, **case.arguments)
    assert resumed.record == case.plan.record
    report = retire.apply_retirement(resumed)
    assert report['entries_removed'] == len(case.plan.entry_paths) - 1
    assert not any(Path(path).exists() for path in case.plan.entry_paths)
    assert retire.retirement_record_path(case.space).read_bytes() == sealed


def test_a_seed_refuses_an_already_retiring_checkpoint(tmp_path, monkeypatch):
    from prismaquant.stage_a_chain_seed import ChainSeedRefused, _sealed_checkpoint
    case = scientific_case(tmp_path, monkeypatch)
    manifest = next((case.space / 'checkpoints').glob('*/checkpoint.json'))
    binding = {'path': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    _sealed_checkpoint(binding, 'checkpoint')
    retire.retirement_record_path(case.space).write_text('{}')
    with pytest.raises(ChainSeedRefused, match='retires this Stage A run'):
        _sealed_checkpoint(binding, 'checkpoint')
