"""Real metadata/session owners only; no original GPU or source admission claim."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_cost_stage_a as stage_a, source_generation as sg
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.stage_a_selected_row_diagnostic import (
    SESSION_PREPARATION_OWNER,
    load_original_diagnostic_context,
    load_original_diagnostic_issued_context,
    prepare_original_diagnostic_session,
)
from test_capture_original_material import _bound
from test_original_source_admission import authority_case, original_runner, original_model, material, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


def _context(case):
    session_input = case['plan']['original_session_preparation']
    return load_original_diagnostic_issued_context(session_input,
        base_plan_input=case['plan']['original_source']['base_plan'], authority=case['authority'])


def test_real_issued_context_preserves_pending_namespace_without_admission(authority_case):
    case = authority_case
    context = _context(case)
    session = context['session_preparation']['session']
    directory = case['entries_path'].parent
    before = {path: path.read_bytes() for path in directory.rglob('*.json')}
    assert session == case['authority']['session']
    assert session['run_identity_sha256'] == canonical_json_sha256(
        context['session_identity'], where='exact boundary source')
    assert context['execution']['n_probes'] == 1
    assert context['base_plan']['selected_row_diagnostic']['global_token_count'] == 262144
    assert context['session_preparation']['source_computation'] is False
    assert context['session_preparation']['cuda_computation'] is False
    assert context['session_preparation']['source_admitted'] is False
    assert json.loads((directory / 'generation.json').read_bytes())['status'] == 'running'
    assert json.loads((directory / 'owners' / (SESSION_PREPARATION_OWNER + '.json')).read_bytes())['status'] == 'complete'
    assert not any(case['entries_path'].iterdir())
    assert _context(case)['session_identity'] == context['session_identity']
    assert before == {path: path.read_bytes() for path in directory.rglob('*.json')}


def test_existing_owner_rebinds_same_session_and_never_mints_another(authority_case, monkeypatch):
    context = _context(authority_case)
    identity, session = context['session_identity'], context['session_preparation']['session']
    def forbidden_bind(*args, **kwargs):
        pytest.fail('runtime minted another artifact generation')
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'bind', forbidden_bind)
    with StreamedBoundaryArtifacts(context['execution']['boundary_storage']) as owner:
        owner.rebind(session, identity=identity, n_probes=1,
                     owner_label='original-diagnostic-cpu-session-control')
        assert owner.session == session
        assert owner.directory.name == session['generation']
        assert not owner._references


@pytest.mark.parametrize('damage', ['generation', 'policy', 'owner', 'computed', 'entry'])
def test_pending_generation_or_metadata_owner_tampering_is_not_a_capture(authority_case, damage):
    case = authority_case
    directory = case['entries_path'].parent
    if damage == 'entry':
        (case['entries_path'] / 'foreign.pt').write_bytes(b'not an emitted source entry')
    else:
        path = directory / ('generation.json' if damage in ('generation', 'policy') else
                            'owners/' + SESSION_PREPARATION_OWNER + '.json')
        document = json.loads(path.read_bytes())
        if damage == 'generation':
            document['status'] = 'complete'
        elif damage == 'policy':
            document['policy']['max_artifact_bytes'] += 1
        elif damage == 'owner':
            document['owner']['label'] = 'foreign'
        else:
            document['owner']['source_computation'] = True
        path.write_text(json.dumps(document))
    with pytest.raises((RuntimeError, ValueError)):
        _context(case)


@pytest.mark.parametrize('damage', ['prepared', 'recursive-base', 'N', 'execution', 'directory', 'artifact-cap'])
def test_wrong_control_context_fails_before_any_new_namespace(authority_case, damage):
    case = authority_case
    context = _context(case)
    base, prepared, execution = copy.deepcopy(case['base']), copy.deepcopy(case['prepared']), copy.deepcopy(case['execution'])
    base['output_root'] = str(case['tmp'] / 'uncreated-control-refusal')
    execution['boundary_storage']['directory'] = str(Path(base['output_root']) / 'layer-quanta' / 'adjoint' / 'exact-boundaries')
    if damage == 'prepared':
        prepared['schema'] = 'prismaquant.joint_prepared.v3'
    elif damage == 'recursive-base':
        base['authority'] = case['authority_input']
    elif damage == 'N':
        base['selected_row_diagnostic']['global_token_count'] = 512
    elif damage == 'execution':
        execution['n_probes'] = 4
    elif damage == 'directory':
        execution['boundary_storage']['directory'] += '-foreign'
    else:
        execution['boundary_storage']['max_artifact_bytes'] = context['resources']['artifact_bytes'] + 1
    base['prepared'] = _bound(case['tmp'] / 'wrong-prepared.json', prepared)
    base['execution'] = _bound(case['tmp'] / 'wrong-execution.json', execution)
    base_input = _bound(case['tmp'] / 'wrong-base.json', base)
    with pytest.raises((RuntimeError, ValueError)):
        load_original_diagnostic_context(base_input, context['static_authority_input'])
    assert not Path(base['output_root']).exists()


def test_metadata_issuer_does_not_reuse_an_existing_session_root(authority_case):
    case = authority_case
    context = _context(case)
    before = {path: path.read_bytes() for path in case['entries_path'].parent.rglob('*.json')}
    with pytest.raises(ValueError, match='session root already exists'):
        prepare_original_diagnostic_session(context['base_plan_input'], context['static_authority_input'])
    assert before == {path: path.read_bytes() for path in case['entries_path'].parent.rglob('*.json')}


def test_missing_original_cuda_admission_precedes_every_legacy_pricing_join(authority_case, monkeypatch):
    case = authority_case
    from prismaquant import cost_streaming, model_profiles, tessera_joint_aura
    before = case['owner'].receipt()
    def forbidden(*args, **kwargs):
        pytest.fail('runtime entered source/profile/pricing/model/output before actual original device refusal')
    monkeypatch.setattr(case['owner'], '_open_source_state', forbidden)
    monkeypatch.setattr(model_profiles, 'detect_profile', forbidden)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', forbidden)
    monkeypatch.setattr(tessera_joint_aura, '_preflight_run_prepared', forbidden)
    monkeypatch.setattr(tessera_joint_aura, 'seed_source_identity_cache', forbidden)
    with pytest.raises(RuntimeError, match='GPU loads/transfers are not qualified'):
        stage_a.run_original_diagnostic_capture(case['authority_input'], case['plan_input'],
            case['plan']['original_session_preparation'], source_authentication=case['owner'])
    assert case['owner'].receipt() == before
    assert not any(case['entries_path'].iterdir())


def test_plain_execution_grammar_preserves_full_draw_and_refuses_legacy_or_recursive_fields(authority_case):
    case = authority_case
    assert sg.normalize_original_diagnostic_execution(case['execution']) == case['execution']
    for key, value in [('source_derivative', {}), ('canonical_capture', {}), ('source_identity_cache', {}),
                       ('authority', case['authority_input']), ('session', case['session'])]:
        document = dict(case['execution'], **{key: value})
        with pytest.raises(RuntimeError):
            sg.normalize_original_diagnostic_execution(document)


def test_typed_context_cannot_enter_ordinary_core_without_selected_diagnostic(tmp_path, monkeypatch):
    from test_stage_a_chain_resume import _dense_runner
    from test_layer_major_boundary_capture import draw
    runner = _dense_runner()
    root = tmp_path / 'uncreated-core'
    with pytest.raises(stage_a.AdjointIdentityRefused, match='only a fresh selected-row diagnostic'):
        stage_a.run_adjoint_capture_core(runner, draw(), execution={}, output_root=root, stride=1,
            source_model_identity={}, unit_roster_sha256='1' * 64, plan_sha256='2' * 64,
            prepared_sha256='3' * 64, read_manifest_sha256='4' * 64, implementation_sha256='5' * 64,
            original_diagnostic_context={})
    assert not root.exists()


def test_context_bound_bytes_digest_is_real_not_an_authority_restamp(authority_case):
    context = _context(authority_case)
    reference = context['base_plan']['prepared']
    assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256']
    assert context['session_identity']['prepared_sha256'] == reference['sha256']
    assert context['session_identity']['prepared_sha256'] != context['base_plan']['static_authority_sha256']
    assert 'authority' not in context['session_identity']
    assert context['session_identity']['read_manifest_sha256'] == authority_case['authority']['readset']['sha256']
    assert context['session_identity']['read_manifest_sha256'] != authority_case['final_input']['sha256']
