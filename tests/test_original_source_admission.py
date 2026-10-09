"""Real CPU owner/control joins, never fabricated original CUDA/root admission."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import source_generation as sg, tessera_calibration_cache as cc
from prismaquant.calibration_data import load_calibration_input
from prismaquant.cost_streaming import build_streamed_model_identity
from prismaquant.digests import canonical_json_sha256
from prismaquant.joint_aura import source_execution_identity
from prismaquant.staged_lease import LeaseRefused, resolve_context
from prismaquant.joint_adjoint_checkpoints import adjoint_space, boundary_entry_directory
from prismaquant.stage_a_selected_row_diagnostic import prepare_original_diagnostic_session
from test_capture_original_material import _bound, material, _forget_state  # noqa: F401
from test_original_source_identity import original_runner  # noqa: F401
from test_original_streaming_bootstrap_2010 import original_model  # noqa: F401
from test_strict_reader_tier_enforcement import _publish_readset_on_the_claim

pytestmark = pytest.mark.own_process

# CEO D32: the suite default is certified mode; a test names the mode it pins.
CERTIFIED, DEV = '0', '1'
MODES = pytest.mark.parametrize('mode', [CERTIFIED, DEV], ids=['certified', 'dev'])


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def authority_case(original_runner, original_model, tmp_path):
    state, material = original_runner, original_model
    owner, runner = state['owner'], state['runner']
    # Native source fixture remains the existing real Llama material/decoder;
    # the full token plane is independently saved and read by the real loader.
    ids = torch.arange(512 * 512, dtype=torch.int64).reshape(512, 512).remainder(32)
    text = b'real deterministic CPU contract corpus\n'
    provenance = dict(fit_ids_sha256=_digest(ids.to(torch.int32).numpy().tobytes()),
        fit_tokens=ids.numel(), fit_tokens_min=13, model=str(owner.root), nsamples=512,
        seed=0, seqlen=512, source='CPU original-material contract draw',
        split_role='calibration', text_sha256=_digest(text))
    calibration_path = tmp_path / 'full-calibration.safetensors'
    save_file({'calibration_ids': ids}, str(calibration_path),
              metadata={'calibration_provenance': json.dumps(provenance)})
    calibration_input = dict(path=str(calibration_path), sha256=_digest(calibration_path.read_bytes()))
    observed_ids, calibration = load_calibration_input(calibration_path,
        expected_sha256=calibration_input['sha256'], n_samples=512, seqlen=512)
    assert torch.equal(observed_ids, ids)
    sdk, claim = resolve_context()
    queue_root = Path(claim['queue_root'])
    claim_path = queue_root / 'claimed' / (claim['action_key'] + '.json')
    claim_row = json.loads(claim_path.read_bytes())
    claim_row['resources'] = {'cpu': 1, 'mem_gb': 1}
    claim_path.write_text(json.dumps(claim_row))
    prefetch = dict(max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1,
        cache_headroom_gb=1, prefetch_min_available_gb=1, require_prefetched_residency=True)
    resources = dict(schema='prismaquant.original_source_resources.v1', cpu_bytes=1024**3,
        material_bytes=material['options']['max_material_bytes'], source_cache_bytes=1024**2,
        source_prefetch=prefetch, copy_bytes=1024**2, gpu_bytes=0, native_bytes=0,
        serialization_bytes=16 * 1024**2, artifact_bytes=16 * 1024**2,
        deadline_seconds=60, stall_seconds=10, host_floor_bytes=1, margin_bytes=0,
        claim_demand=claim_row['resources'])
    resources_input = _bound(tmp_path / 'resources.json', resources)
    runtime = sg.original_source_runtime(runner, owner)
    source = build_streamed_model_identity(runner, str(owner.root))
    authority = dict(schema=sg.ORIGINAL_AUTHORITY_SCHEMA, scope=sg.ORIGINAL_AUTHORITY_SCOPE,
        publisher=dict(id=material['options']['publisher_id'], revision=material['options']['publisher_revision'],
                       input=material['options']['publisher_input']),
        producer=_bound(tmp_path / 'producer.json', material['producer']),
        source_paths=_bound(tmp_path / 'paths.json', material['paths']),
        readset=material['options']['readset_input'], runtime=_bound(tmp_path / 'runtime.json', runtime),
        qualification=None, root_admission=None, calibration=calibration,
        source_model_identity=source, source_execution=source_execution_identity(runner.model),
        session=None, resources=resources_input)
    static = sg.original_authority_static_sha256(authority)
    output_root = tmp_path / 'diagnostic-metadata-root'
    static_input = _bound(tmp_path / 'static-authority.json',
        {key: authority[key] for key in sg.ORIGINAL_STATIC_AUTHORITY_KEYS})
    policy = dict(schema='prismaquant.aura.boundary_storage.v2', capture_order='layer_major',
        directory=str(boundary_entry_directory(adjoint_space(output_root))), max_resident_bytes=1024**2,
        max_auxiliary_bytes=1024**2, max_artifact_bytes=16 * 1024**2, prefetch_batches=1)
    execution = dict(schema='prismaquant.original_diagnostic_execution.v1', n_calib_samples=512,
        calib_seqlen=512, n_probes=1, seed_base=7000, probe_microbatch=1, token_scope='all',
        temperature=1.0, boundary_storage=policy, stride=1, chain_batch_size=1, chain_probe_fusion=False)
    execution_input = _bound(tmp_path / 'execution.json', execution)
    prepared = dict(schema='prismaquant.original_diagnostic_preparation.v1',
        scope='render_free_original_source_context', static_authority_sha256=static,
        implementation_sha256=runtime['prismaquant_source_sha256'], source_model_identity=source,
        source_execution=authority['source_execution'], calibration=calibration, resources=resources_input,
        head_source={'tensors': sorted(name for name in material['producer']['tensors']
                                      if not name.startswith('model.layers.'))})
    prepared_input = _bound(tmp_path / 'prepared.json', prepared)
    diagnostic = dict(schema='prismaquant.stage_a.selected_row_diagnostic.v1',
        calibration_shape=[512, 512], calibration_dtype='torch.int64',
        calibration_tensor_sha256=calibration['calibration_sha256'], selected_global_row=0,
        probe_seed=7000, global_token_count=262144, vocab_size=154880, through=6)
    base = dict(schema='prismaquant.original_diagnostic_base_plan.v1', model=str(owner.root),
        output_root=str(output_root), static_authority_sha256=static,
        prepared=prepared_input, read_manifest=authority['readset'], execution=execution_input,
        calibration_input=calibration_input, selected_row_diagnostic=diagnostic)
    base_input = _bound(tmp_path / 'base-plan.json', base)
    issued = prepare_original_diagnostic_session(base_input, static_input)
    authority['session'] = dict(issued['receipt']['session'])
    session_input = {key: issued[key] for key in ('path', 'sha256')}
    authority_input = _bound(tmp_path / 'authority.json', authority)
    final = copy.deepcopy(material['readset'])
    final['entries'].append(dict(path=authority_input['path'], offset=0,
        bytes=Path(authority_input['path']).stat().st_size, sha256=authority_input['sha256']))
    final['entry_count'] = len(final['entries'])
    final['total_bytes'] = sum(row['bytes'] for row in final['entries'])
    final_input = _bound(tmp_path / 'final-readset.json', final)
    cas_root = tmp_path / 'native-cas'
    blob = cas_root / 'blobs' / final_input['sha256'][:2] / final_input['sha256']
    blob.parent.mkdir(parents=True)
    blob.write_bytes(Path(final_input['path']).read_bytes())
    _publish_readset_on_the_claim(queue_root, claim['action_key'], cas_root,
                                 final_input['sha256'], blob.stat().st_size)
    plan = dict(model=str(owner.root), calibration_input=calibration_input, source_prefetch=prefetch,
        original_session_preparation=session_input,
        original_source=dict(authority=authority_input, base_plan=base_input, prepared=prepared_input,
                             read_manifest=final_input, execution=execution_input))
    plan_input = _bound(tmp_path / 'final-plan.json', plan)
    packet = sg.observe_original_source_execution(authority_input, plan_input, resource_check=owner.resource_check)
    yield dict(owner=owner, material=material, authority=authority, authority_input=authority_input,
               plan=plan, plan_input=plan_input, packet=packet,
               session=authority['session'], entries_path=Path(policy['directory']) / authority['session']['generation'] / 'entries',
               base=base, prepared=prepared, execution=execution, claim_path=claim_path,
               final=final, final_input=final_input, tmp=tmp_path)


@pytest.mark.parametrize("owner", ["static", "preparation", "session"])
def test_original_owners_keep_the_512_by_512_draw(authority_case, owner):
    case = authority_case
    preparation = copy.deepcopy(case["prepared"])
    calibration = preparation["calibration"]
    # Same flat token payload/digest and same token count, different shape.
    calibration["shape"] = [128, 2048]
    calibration["provenance"]["nsamples"] = 128
    calibration["provenance"]["seqlen"] = 2048
    with pytest.raises(RuntimeError, match="full calibration shape"):
        if owner == "static":
            authority = {key: copy.deepcopy(case["authority"][key])
                         for key in sg.ORIGINAL_STATIC_AUTHORITY_KEYS}
            authority["calibration"] = calibration
            sg.normalize_original_source_static_authority(authority)
        elif owner == "preparation":
            sg.normalize_original_diagnostic_preparation(preparation)
        else:
            sg.original_diagnostic_session_identity(
                base_plan=case["base"], base_plan_sha256="a" * 64, prepared=preparation,
                execution_sha256=case["base"]["execution"]["sha256"])


def test_original_draw_requires_its_minimum_fit_tokens(authority_case):
    preparation = copy.deepcopy(authority_case["prepared"])
    del preparation["calibration"]["provenance"]["fit_tokens_min"]
    with pytest.raises(RuntimeError, match="provenance fields"):
        sg.normalize_original_diagnostic_preparation(preparation)


@pytest.mark.parametrize("dev_mode", ["1", "0"])
def test_session_identity_directly_compares_shape_after_preparation_validation(
        authority_case, monkeypatch, dev_mode):
    case = authority_case
    preparation = copy.deepcopy(case["prepared"])
    # Isolate the session join: upstream preparation validation has its own
    # original-shape tests above. Keep the actual flat tensor digest unchanged.
    preparation["calibration"]["shape"] = [128, 2048]
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    monkeypatch.setattr(sg, "normalize_original_diagnostic_preparation", sg._snapshot)
    with pytest.raises(RuntimeError, match="prepared/base full calibration shape"):
        sg.original_diagnostic_session_identity(
            base_plan=case["base"], base_plan_sha256="a" * 64, prepared=preparation,
            execution_sha256=case["base"]["execution"]["sha256"])


def test_original_json_snapshot_keeps_direct_order_unicode_and_owned_values():
    value = {10: ['café', '\ud800', -0.0], 9: (True, None, 1.0)}
    expected = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    snapshot = sg._snapshot(value)
    assert snapshot == expected
    assert list(snapshot) == ['9', '10']
    assert type(snapshot['9']) is list
    snapshot['10'].append('independent mutation')
    assert value[10] == ['café', '\ud800', -0.0]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'),
                                  {'mixed': 1, 2: 'key'}, {'not_json': object()}])
def test_original_json_snapshot_keeps_stdlib_strict_error(value):
    with pytest.raises((TypeError, ValueError)) as previous:
        json.dumps(value, sort_keys=True, allow_nan=False)
    with pytest.raises(type(previous.value)) as current:
        sg._snapshot(value)
    assert str(current.value) == str(previous.value)
    assert current.value.__cause__ is None


def test_original_identity_comparison_reuses_stage_owner_without_changing_errors():
    from prismaquant.stage_inputs import same

    assert sg._same.func is same
    assert sg._same.keywords == {'contract': sg._contract}
    assert same({'key': 1}, {'key': 1}, 'source') is None
    assert sg._same({'key': 1}, {'key': 1}, 'source') is None
    with pytest.raises(ValueError) as default:
        same('a', 'b', 'source')
    with pytest.raises(RuntimeError) as original:
        sg._same('a', 'b', 'source')
    assert str(default.value) == 'source: identity mismatch'
    assert str(original.value) == 'original generation: source: identity mismatch'
    assert default.value.__cause__ is None and original.value.__cause__ is None


def test_original_recorded_comparison_reuses_the_stage_seal_with_the_original_vocabulary(
        monkeypatch, capsys):
    """A recorded-versus-selected join is the stage owner's D32 seal.

    Certified mode refuses with this module's exception and message. Dev mode
    prints one ``[DEV-MODE]`` line and continues.
    """
    from prismaquant.stage_inputs import recorded_same

    assert sg._recorded_same.func is recorded_same
    assert sg._recorded_same.keywords['contract'] is sg._contract
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', CERTIFIED)
    assert sg._recorded_same({'key': 1}, {'key': 1}, 'source') is None
    with pytest.raises(ValueError) as default:
        recorded_same('a', 'b', 'source')
    with pytest.raises(RuntimeError) as original:
        sg._recorded_same('a', 'b', 'source')
    assert str(default.value) == 'source: identity mismatch'
    assert str(original.value) == 'original generation: source: identity mismatch'
    assert default.value.__cause__ is None and original.value.__cause__ is None
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', DEV)
    capsys.readouterr()
    assert recorded_same('a', 'b', 'source') is None
    assert sg._recorded_same('a', 'b', 'source') is None
    out = capsys.readouterr().out.splitlines()
    assert out == [
        '[DEV-MODE] seal source differs (Stage A/B recorded versus running provenance): '
        'expected b, actual a; sealing is off (PQ #1147), continuing with the stored data',
        '[DEV-MODE] seal source differs (original source qualification): '
        'expected b, actual a; sealing is off (PQ #1147), continuing with the stored data']


def test_original_execution_retains_nullable_mapped_selectors_and_mapping_envelope():
    value = MappingProxyType({'schema': 'prismaquant.joint_aura.source_execution.v1',
        'modules': MappingProxyType({'u': {'attention': None, 'experts': {'é': '\ud800', 'a': None}}})})
    assert sg._source_execution(value) is value


@pytest.mark.parametrize(('modules', 'message'), [
    ([], 'original execution modules must be an object'),
    ({}, 'original execution has no resolved selectors'),
    ({2: {'attention': 'eager'}}, 'invalid original execution selector'),
    ({'u': {}}, 'invalid original execution selector'),
    ({'u': {'unknown': 'eager'}}, 'invalid original execution selector'),
    ({'u': {'attention': True}}, 'original execution u.attention needs resolved selector values'),
    ({'u': {'attention': float('nan')}}, 'original execution u.attention needs resolved selector values'),
    ({'u': {'attention': {'x': 1}}}, 'original execution u.attention needs resolved selector values'),
    ({'u': {'attention': {1: 'eager'}}}, 'original execution u.attention needs resolved selector values'),
    ({'u': {'attention': {'x': ''}}}, 'original execution u.attention needs resolved selector values'),
    ({'u': {'experts': []}}, 'original execution u.experts needs resolved selector values'),
    ({'u': {'experts': ''}}, 'original execution u.experts needs resolved selector values'),
    ({'u': {'experts': {}}}, 'original execution u.experts needs resolved selector values'),
])
def test_original_execution_retains_exact_selector_refusals(modules, message):
    value = {'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': modules}
    with pytest.raises(RuntimeError) as caught:
        sg._source_execution(value)
    assert str(caught.value) == 'original generation: ' + message
    assert caught.value.__cause__ is None


def test_original_execution_retains_exact_string_types():
    class Selector(str):
        pass

    for modules, message in [
        ({Selector('u'): {'attention': 'eager'}}, 'invalid original execution selector'),
        ({'u': {'attention': Selector('eager')}}, 'original execution u.attention needs resolved selector values'),
        ({'u': {'attention': {Selector('a'): 'eager'}}}, 'original execution u.attention needs resolved selector values'),
    ]:
        value = {'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': modules}
        with pytest.raises(RuntimeError) as caught:
            sg._source_execution(value)
        assert str(caught.value) == 'original generation: ' + message


def test_source_execution_identity_keeps_direct_ascii_snapshot_and_selector_ownership():
    selectors = {10: 'é', 9: ['\ud800', None, -0.0]}
    model = torch.nn.Module()
    model.config = SimpleNamespace(_attn_implementation=selectors, _experts_implementation='grouped_mm')
    identity = source_execution_identity(model)
    expected = json.loads(json.dumps(selectors, sort_keys=True, allow_nan=False))
    assert identity == {'schema': 'prismaquant.joint_aura.source_execution.v1',
                        'modules': {'': {'attention': expected, 'experts': 'grouped_mm'}}}
    assert list(identity['modules']['']['attention']) == ['9', '10']
    selectors[9].append('caller mutation')
    assert identity['modules']['']['attention'] == expected


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'), object(),
                                  {'mixed': 1, 2: 'key'}])
def test_source_execution_identity_keeps_direct_json_refusal(value):
    model = torch.nn.Module()
    model.config = SimpleNamespace(_attn_implementation=value)
    with pytest.raises((TypeError, ValueError)) as previous:
        json.dumps(value, sort_keys=True, allow_nan=False)
    with pytest.raises(type(previous.value)) as current:
        source_execution_identity(model)
    assert str(current.value) == str(previous.value)
    assert current.value.__cause__ is None


def _forbid_source_work(case, monkeypatch):
    owner = case['owner']
    before = owner.receipt()
    def forbidden(*args, **kwargs):
        pytest.fail('authority intake opened source material/profile/output')
    monkeypatch.setattr(owner, '_open_source_state', forbidden)
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, 'detect_profile', forbidden)
    return before


@MODES
@pytest.mark.parametrize('axis', ['schema', 'scope', 'publisher', 'producer', 'source_paths',
                                'readset', 'calibration', 'source_model_identity', 'source_execution'])
def test_selected_reader_cannot_substitute_another_target_authority(
        authority_case, monkeypatch, axis, mode):
    """Real CPU control inputs; no selected result or qualified source is fabricated.

    The target, readset, calibration and model joins say which data the proof
    is about. They are correctness joins, not seals, and refuse in both modes.
    """
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    reader_authority = copy.deepcopy(case['authority'])
    if axis in ('schema', 'scope'):
        reader_authority[axis] += '.other'
    elif axis == 'publisher':
        reader_authority[axis]['revision'] = '0' * 40
    elif axis in ('producer', 'source_paths', 'readset'):
        reader_authority[axis]['sha256'] = '0' * 64
    elif axis == 'calibration':
        reader_authority[axis]['artifact_sha256'] = '0' * 64
    elif axis == 'source_model_identity':
        reader_authority[axis]['content_sha256'] = '0' * 64
    else:
        reader_authority[axis]['modules']['']['attention'] = 'another_dispatch'
    with pytest.raises(RuntimeError, match=f'qualified reader target {axis}'):
        sg._require_original_reader_target(reader_authority, case['authority'])
    assert case['owner'].receipt() == before


def test_reader_target_agreement_is_not_producer_runtime_resource_or_session_equality(
        authority_case, monkeypatch):
    """Target agreement only, never a positive public reader qualification."""
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    reader_authority = copy.deepcopy(case['authority'])
    reader_authority['runtime'] = _bound(case['tmp'] / 'different-reader-runtime.json',
                                       copy.deepcopy(case['packet']['runtime']))
    reader_authority['resources'] = _bound(case['tmp'] / 'different-reader-resources.json',
                                         copy.deepcopy(case['packet']['resources']))
    reader_authority['session'] = dict(case['session'], generation='different_reader_session')
    assert sg._require_original_reader_target(reader_authority, case['authority']) is None
    assert case['owner'].receipt() == before


@pytest.mark.parametrize('missing', [None, {}])
def test_reader_observations_without_sdk_selected_native_context_refuse(authority_case, missing):
    """Absence is a refusal; an SDK4 result is not mock-upgraded to SDK5."""
    result = {} if missing is None else {'producer_context': None}
    with pytest.raises(RuntimeError, match='requires selected native producer context'):
        sg._require_original_reader_producer(authority_case['owner'].receipt(),
            authority_case['authority'], result, authority_case['authority']['runtime'])


@pytest.mark.parametrize('axis', ['prismaquant_source_sha256', 'tessera_source_sha256',
    'container_content_sha256', 'modeling_source', 'model_class', 'profile', 'config',
    'versions', 'arithmetic', 'material_pipeline', 'prismabuild'])
def test_producer_runtime_requires_its_independently_bound_complete_expectation(authority_case, axis):
    """Actual CPU runtime metadata, not a mock selected producer/source qualification."""
    observed = authority_case['packet']['runtime']
    expected = copy.deepcopy(observed)
    if axis in ('prismaquant_source_sha256', 'tessera_source_sha256', 'container_content_sha256'):
        expected[axis] = '0' * 64
    elif axis in ('model_class', 'profile'):
        expected[axis] += '.other'
    elif axis == 'modeling_source':
        expected[axis]['sha256'] = '0' * 64
    elif axis == 'config':
        expected[axis]['independent_other_config'] = True
    elif axis == 'versions':
        expected[axis]['python'] = '0.0.0'
    elif axis == 'arithmetic':
        expected[axis]['allow_tf32'] = not expected[axis]['allow_tf32']
    elif axis == 'material_pipeline':
        expected[axis]['target_dtype'] = 'another_loader_dtype'
    else:
        expected[axis]['runtime_generation'] += '.other'
    binding = _bound(authority_case['tmp'] / f'independent-runtime-{axis}.json', expected)
    with pytest.raises(RuntimeError, match='observed original source runtime'):
        sg._validate_original_source_runtime(observed, binding,
            sdk_version=observed['prismabuild']['sdk_version'])


def test_current_consumer_and_independent_producer_sdk_policies_stay_exact(authority_case):
    observed = authority_case['packet']['runtime']
    binding = _bound(authority_case['tmp'] / 'independent-producer-runtime.json', copy.deepcopy(observed))
    version = observed['prismabuild']['sdk_version']
    assert sg._validate_original_source_runtime(observed, binding, sdk_version=version) == observed
    with pytest.raises(RuntimeError, match='original runtime SDK version'):
        sg._validate_original_source_runtime(observed, binding, sdk_version=version + 1)
    changed = copy.deepcopy(observed)
    changed['prismabuild']['sdk_version'] += 1
    with pytest.raises(RuntimeError, match='original runtime SDK version'):
        sg.validate_original_source_runtime(changed, changed)


def _selected_reader_producer_fixtures(case):
    """Real receipt claim rows beside the one identity a SDK5 result carries.

    The claim rows are the fixture owner's actual launch-env delivery
    observations; the context dict assembles only the identity fields the
    consumer join reads, from those same real rows and the observed real
    resources. It is not an SDK result, no queue is read, and nothing here
    qualifies a producer or a source.
    """
    receipt = case['owner'].receipt()
    claim = receipt['deliveries'][0]['native_delivery']['claim']
    _, runtime = sg._control(case['authority']['runtime'], 'fixture runtime')
    identity = {key: claim[key] for key in ('queue_root', 'action_key', 'nonce', 'scope_id',
                                            'worker', 'host', 'incarnation', 'helper_root')}
    _, resources = sg._control(case['authority']['resources'], 'fixture resources')
    context = dict(identity,
        schema='prismabuild.native_producer_context.v1', published_unix=1791000000.0,
        attempt=1, generation='fixture-generation', receipt_sha256='a' * 64,
        runtime_sha256='b' * 64, resources=resources['claim_demand'],
        resources_semantics='selected-claim-sealed-demand',
        attempt_source='selected-immutable-attempt')
    result = {'action_key': identity['action_key'], 'published_unix': context['published_unix'],
              'attempt': context['attempt'], 'generation': context['generation'],
              'host': identity['host'], 'worker_id': identity['worker'],
              'producer_context': context,
              'receipt': {'receipt_sha256': context['receipt_sha256'],
                          'producer': {'runtime': {'runtime_sha256': context['runtime_sha256']}}}}
    return receipt, context, result


def _seal(monkeypatch, capsys, mode, join, refusal):
    """Run one D32 seal under ``mode``; return the ``[DEV-MODE]`` output it printed.

    Certified mode refuses with ``refusal`` and prints no stamp. Dev mode
    completes the join and stamps the mismatch instead (CEO D32).
    """
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    capsys.readouterr()
    if mode == CERTIFIED:
        with pytest.raises(RuntimeError, match=refusal):
            join()
        assert '[DEV-MODE]' not in capsys.readouterr().out
        return ''
    assert join() is None
    return capsys.readouterr().out


@MODES
@pytest.mark.parametrize('axis', ['queue_root', 'action_key', 'nonce', 'scope_id',
                                  'worker', 'host', 'helper_root'])
def test_reader_delivery_cannot_substitute_another_selected_producer(
        authority_case, monkeypatch, capsys, axis, mode):
    """Every delivery joins the actual selected producer, axis by axis.

    Each mutation also satisfies the context/result mirrors that precede the
    delivery join, so the join that fires is the delivery identity join
    itself and not an earlier policy check. It compares the reader's record
    of its producer run with the SDK's selection, a D32 seal: certified mode
    refuses, dev mode stamps the first differing field and continues.
    """
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    receipt, context, result = _selected_reader_producer_fixtures(case)
    authority = case['authority']
    if axis == 'helper_root':
        _, runtime = sg._control(authority['runtime'], 'fixture runtime')
        runtime = copy.deepcopy(runtime)
        context[axis] = '/foreign-selected-helper'
        runtime['prismabuild']['helper_root'] = context['helper_root']
        runtime['prismabuild']['runtime_generation'] = Path(context['helper_root']).name
        authority = dict(authority, runtime=_bound(
            case['tmp'] / 'foreign-helper-runtime.json', runtime))
    elif axis == 'queue_root':
        context[axis] = '/foreign-queue'
    elif axis == 'action_key':
        context[axis] = 'f' * 64
        result['action_key'] = context['action_key']
    elif axis in ('nonce', 'scope_id'):
        context[axis] = 'f' * 32
    elif axis == 'worker':
        context['worker'] = context['incarnation'] = 'another-worker'
        result['worker_id'] = 'another-worker'
    else:
        context['host'] = 'another-host'
        result['host'] = 'another-host'
    # The worker axis moves two fields; the first one by name is incarnation.
    field = 'incarnation' if axis == 'worker' else axis
    out = _seal(monkeypatch, capsys, mode,
                lambda: sg._require_original_reader_producer(receipt, authority, result,
                                                             authority['runtime']),
                'actual reader delivery selected producer')
    if mode == DEV:
        assert ('[DEV-MODE] seal actual reader delivery selected producer differs at '
                f'{field} (original source qualification): expected {context[axis]}, ') in out
    assert case['owner'].receipt() == before


@MODES
@pytest.mark.parametrize('axis', ['helper_root', 'helper_generation', 'reservation',
                                  'launch_provenance'])
def test_reader_runtime_reservation_and_launch_are_seals_of_the_selected_producer(
        authority_case, monkeypatch, capsys, axis, mode):
    """Each is the reader's record of its producer run against the SDK's selection.

    Certified mode refuses by name. Dev mode stamps the one differing value
    and the join continues with the stored record.
    """
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    receipt, context, result = _selected_reader_producer_fixtures(case)
    authority = case['authority']
    if axis in ('helper_root', 'helper_generation'):
        _, runtime = sg._control(authority['runtime'], 'fixture runtime')
        runtime = copy.deepcopy(runtime)
        generation = Path(context['helper_root']).name
        if axis == 'helper_root':
            label = 'selected reader actual helper root'
            recorded, selected = f'/foreign-recorded-helper/{generation}', context['helper_root']
            runtime['prismabuild']['helper_root'] = recorded
        else:
            label = 'selected reader actual helper generation'
            recorded, selected = 'foreign-recorded-generation', generation
            runtime['prismabuild']['runtime_generation'] = recorded
        stamp = f'differs (original source qualification): expected {selected}, actual {recorded};'
        authority = dict(authority, runtime=_bound(case['tmp'] / f'{axis}-runtime.json', runtime))
    elif axis == 'reservation':
        label = 'selected reader actual producer reservation'
        context['resources'] = {**context['resources'], 'mem_gb': 99}
        stamp = 'differs at mem_gb (original source qualification): expected 99, actual 1;'
    else:
        label = 'actual reader delivery launch provenance'
        receipt['deliveries'][0]['native_delivery']['claim']['attempt_source'] = (
            'selected-immutable-attempt')
        stamp = ('differs (original source qualification): expected launch-env, '
                 'actual selected-immutable-attempt;')
    out = _seal(monkeypatch, capsys, mode,
                lambda: sg._require_original_reader_producer(receipt, authority, result,
                                                             authority['runtime']),
                label)
    if mode == DEV:
        assert f'[DEV-MODE] seal {label} {stamp}' in out
    assert case['owner'].receipt() == before


@MODES
@pytest.mark.parametrize('damage', ['provenance', 'semantics', 'worker', 'incarnation',
                                    'receipt', 'runtime'])
def test_reader_context_policy_fields_cannot_be_relabelled(
        authority_case, monkeypatch, damage, mode):
    """The SDK answer's own mirrors are correctness joins: both modes refuse."""
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    receipt, context, result = _selected_reader_producer_fixtures(case)
    if damage == 'provenance':
        context['attempt_source'] = 'launch-env'
        expected = 'selected reader context provenance'
    elif damage == 'semantics':
        context['resources_semantics'] = 'ledger-allowance'
        expected = 'selected reader reservation semantics'
    elif damage == 'worker':
        context['worker'] = 'another-worker'
        expected = 'selected reader full worker identity'
    elif damage == 'incarnation':
        context['incarnation'] = 'another-incarnation'
        expected = 'selected reader full incarnation'
    elif damage == 'receipt':
        context['receipt_sha256'] = 'c' * 64
        expected = 'selected reader execution receipt'
    else:
        context['runtime_sha256'] = 'd' * 64
        expected = 'selected reader attested runtime'
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    with pytest.raises(RuntimeError, match=expected):
        sg._require_original_reader_producer(receipt, case['authority'], result,
                                             case['authority']['runtime'])
    assert case['owner'].receipt() == before


@MODES
def test_reader_runtime_config_is_the_source_config_in_both_modes(
        authority_case, monkeypatch, mode):
    """The model configuration the proof ran is a target axis, never a stamp."""
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    receipt, context, result = _selected_reader_producer_fixtures(case)
    _, runtime = sg._control(case['authority']['runtime'], 'fixture runtime')
    runtime = copy.deepcopy(runtime)
    runtime['config']['independent_other_config'] = True
    authority = dict(case['authority'], runtime=_bound(case['tmp'] / 'other-config-runtime.json', runtime))
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    with pytest.raises(RuntimeError, match='selected reader runtime source config'):
        sg._require_original_reader_producer(receipt, authority, result, authority['runtime'])
    assert case['owner'].receipt() == before


def _sealed_helper_generation(root):
    """A real sealed generation, ``src/prismabuild`` beside other helper files."""
    package = root / 'src' / 'prismabuild'
    package.mkdir(parents=True)
    (package / 'client.py').write_text('SDK_VERSION = 5\n')
    (root / 'worker.py').write_text('print("worker")\n')
    return root


def _reader_rows_over(case, root):
    """Claim rows and a runtime record that name ``root`` and the real digests of its tree."""
    from prismaquant.production_weight_cache import _production_cache_source_sha256

    receipt, context, result = _selected_reader_producer_fixtures(case)
    context['helper_root'] = str(root)
    for row in receipt['deliveries']:
        row['native_delivery']['claim']['helper_root'] = str(root)
    _, runtime = sg._control(case['authority']['runtime'], 'fixture runtime')
    runtime = copy.deepcopy(runtime)
    runtime['prismabuild'].update(
        helper_root=str(root), runtime_generation=root.name,
        source_tree={'package_sha256': _production_cache_source_sha256(root / 'src' / 'prismabuild'),
                     'helper_tree_sha256': _production_cache_source_sha256(root)})
    authority = dict(case['authority'],
                     runtime=_bound(case['tmp'] / 'sealed-generation-runtime.json', runtime))
    return receipt, authority, result


def test_reader_join_accepts_the_sealed_helper_generation_it_recorded(authority_case, monkeypatch):
    """Certified mode hashes the tree on disk and finds the digests the reader recorded."""
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    root = _sealed_helper_generation(case['tmp'] / 'runtime-generations' / 'fixture-generation-0')
    receipt, authority, result = _reader_rows_over(case, root)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', CERTIFIED)
    assert sg._require_original_reader_producer(receipt, authority, result,
                                                authority['runtime']) is None
    assert case['owner'].receipt() == before


@MODES
@pytest.mark.parametrize('moved', ['package', 'helper'])
def test_reader_helper_tree_is_hashed_from_disk_only_in_certified_mode(
        authority_case, monkeypatch, capsys, moved, mode):
    """A byte that moved after the reader recorded its tree.

    A byte in the package moves both recorded digests. A byte elsewhere in the
    generation moves only the complete-tree digest. Certified mode re-hashes
    the generation and refuses either. Dev mode computes no digest over
    existing data: it stamps ``not computed`` beside the recorded tree and the
    helper tree is never read.
    """
    from prismaquant import production_weight_cache

    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    root = _sealed_helper_generation(case['tmp'] / 'runtime-generations' / 'fixture-generation-0')
    receipt, authority, result = _reader_rows_over(case, root)
    moved_file = root / 'src' / 'prismabuild' / 'client.py' if moved == 'package' else root / 'worker.py'
    moved_file.write_text(moved_file.read_text() + '# moved\n')

    def hashed(*args, **kwargs):
        pytest.fail('dev mode hashed the helper tree to seal a run')
    if mode == DEV:
        monkeypatch.setattr(production_weight_cache, '_production_cache_source_sha256', hashed)
    out = _seal(monkeypatch, capsys, mode,
                lambda: sg._require_original_reader_producer(receipt, authority, result,
                                                             authority['runtime']),
                'selected reader actual complete helper tree')
    if mode == DEV:
        assert out.count('[DEV-MODE]') == 1
        assert ('[DEV-MODE] seal selected reader actual complete helper tree not computed '
                '(original source qualification): recorded ') in out
    assert case['owner'].receipt() == before


@MODES
def test_reader_delivery_join_holds_until_the_unshared_helper_tree(
        authority_case, monkeypatch, capsys, mode):
    """The real claim rows satisfy every producer join this consumer owns.

    The unmutated join runs to its final check. Certified mode stops there: the
    fixture's launch helper root is the fleet generation while the SDK is the
    test-injected install, so the two never share one root and the complete-tree
    digest cannot agree. That refusal is the honest current boundary; a sealed
    SDK5 generation (PB #1485 and the separately authorized helper selection) is
    the positive prerequisite. Dev mode completes with that one tree stamp.
    """
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    receipt, context, result = _selected_reader_producer_fixtures(case)
    # The launch helper root in a pool action is the serving interpreter
    # environment, not a sealed src/ generation, so the source-digest owner
    # refuses to read it; on a sealed generation the digest join itself
    # refuses instead. Either named refusal is the honest boundary.
    out = _seal(monkeypatch, capsys, mode,
                lambda: sg._require_original_reader_producer(receipt, case['authority'], result,
                                                             case['authority']['runtime']),
                'selected reader actual complete helper tree|cannot read package root')
    if mode == DEV:
        assert out.count('[DEV-MODE]') == 1
        assert 'seal selected reader actual complete helper tree not computed' in out
    assert case['owner'].receipt() == before


@MODES
@pytest.mark.parametrize('damage', [None, 'foreign-node', 'restamped-snapshot',
                                    'null-compatibility', 'target-source', 'target-runtime'])
def test_reader_source_snapshot_cannot_bridge_family_acceptance(
        authority_case, monkeypatch, capsys, damage, mode):
    """The reader's executed snapshot binds through the same family owner.

    The restamp join compares the row's recorded source with the family's
    accepted old source, a D32 seal: dev mode stamps it. The family, node,
    compatibility and target joins are correctness joins and refuse in both modes.
    """
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    runtime = case['packet']['runtime']
    snapshot_parent = 'b' * 40
    request = {'params': {'checkout_snapshot': {
        'schema': 'prismaquant.prismabuild.pbrun_checkout_snapshot.v2',
        'commit': 'c' * 40, 'parent': snapshot_parent, 'refs': {},
        'subdirectory': '.', 'input': {'id': 'fixture-snapshot'}}}}
    family = {'schema': 'prismaquant.original_source_unchanged_family.v1',
              'old_source': snapshot_parent, 'new_source': 'd' * 40,
              'compatibility': _bound(case['tmp'] / 'reader-compat.json', {'accepted': True}),
              'controls': ['tests/original-source-admission-reader'],
              'target_prismaquant_source_sha256': runtime['prismaquant_source_sha256'],
              'target_runtime_sha256': canonical_json_sha256(runtime, where='fixture target runtime')}
    accepted = {'tests/original-source-admission-reader': family}
    reader = {'node_id': 'tests/original-source-admission-reader',
              'source_snapshot': snapshot_parent, 'compatibility': family['compatibility']}
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    capsys.readouterr()
    assert sg._require_original_qualified_source(reader, request, accepted, runtime) is None
    assert '[DEV-MODE]' not in capsys.readouterr().out
    if damage is None:
        assert case['owner'].receipt() == before
        return
    if damage == 'foreign-node':
        reader['node_id'] = 'tests/another-reader'
        expected = 'qualified member lacks independently selected source-family acceptance'
    elif damage == 'restamped-snapshot':
        # The executed request and the reader row both claim the foreign
        # snapshot; only the family's accepted old_source stays real, so the
        # join that fires is the family restamp join.
        reader['source_snapshot'] = 'e' * 40
        request['params']['checkout_snapshot']['parent'] = 'e' * 40
        expected = 'original executed member source is not restamped'
    elif damage == 'null-compatibility':
        reader['compatibility'] = None
        expected = 'every qualified member requires'
    elif damage == 'target-source':
        accepted[reader['node_id']] = dict(family, target_prismaquant_source_sha256='0' * 64)
        expected = 'qualified member actual target source implementation'
    else:
        accepted[reader['node_id']] = dict(family, target_runtime_sha256='0' * 64)
        expected = 'qualified member actual target runtime'
    join = lambda: sg._require_original_qualified_source(reader, request, accepted, runtime)
    if damage == 'restamped-snapshot':
        out = _seal(monkeypatch, capsys, mode, join, expected)
        if mode == DEV:
            assert ('[DEV-MODE] seal original executed member source is not restamped differs '
                    f'(original source qualification): expected {snapshot_parent}, '
                    f'actual {"e" * 40};') in out
    else:
        with pytest.raises(RuntimeError, match=expected):
            join()
    assert case['owner'].receipt() == before


def test_owned_control_join_is_nonactivating_and_independently_frozen(authority_case, monkeypatch):
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    joined = sg._normalize_original_source_authority(case['owner'], case['authority_input'],
                                                    case['plan_input'], case['packet'])
    assert joined['source_model_identity']['checkpoint_weight_map'] == case['material']['producer']['tensors']
    assert joined['session'] == case['session']
    assert joined['calibration']['provenance'] == case['authority']['calibration']['provenance']
    joined['publisher']['id'] = 'external/mutation'
    assert case['owner']._original['publisher_id'] == case['authority']['publisher']['id']
    for owner in (None, case['owner']):
        with pytest.raises(RuntimeError, match='full64'):
            cc.require_original_source_authority(owner, case['authority_input'], case['plan_input'], case['packet'])
    assert case['owner'].receipt() == before
    assert not any(case['entries_path'].iterdir())
    with pytest.raises(RuntimeError, match='GPU loads/transfers are not qualified'):
        case['owner'].require_material_device('cuda:0')
    with pytest.raises(RuntimeError, match='qualified immutable source'):
        cc.require_automatic_capture_source_recording()


@pytest.mark.parametrize('damage', ['missing', 'changed', 'demand-only'])
def test_native_claim_resources_cannot_be_missing_rebound_or_replaced_by_demand(authority_case, monkeypatch, damage):
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    row = json.loads(case['claim_path'].read_bytes())
    if damage == 'changed':
        row['resources'] = {**row['resources'], 'mem_gb': 2}
    else:
        resources = row.pop('resources')
        if damage == 'demand-only':
            row['demand'] = resources
    case['claim_path'].write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='native reservation/resource demand'):
        sg._normalize_original_source_authority(case['owner'], case['authority_input'],
                                               case['plan_input'], case['packet'])
    assert case['owner'].receipt() == before
    assert not any(case['entries_path'].iterdir())


@pytest.mark.parametrize('damage', ['runtime', 'session', 'session-identity', 'plan', 'claim', 'owner-resource'])
def test_wrong_bound_owner_runtime_session_plan_or_claim_cannot_reach_source(authority_case, monkeypatch, damage):
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    packet = dict(case['packet'])
    if damage == 'runtime':
        packet['runtime'] = copy.deepcopy(packet['runtime'])
        packet['runtime']['versions']['torch'] += '-other-build'
    elif damage == 'session':
        packet['session'] = dict(packet['session'], generation='f' * 32)
    elif damage == 'session-identity':
        packet['session_identity'] = dict(packet['session_identity'], read_manifest_sha256='f' * 64)
    elif damage == 'plan':
        packet['plan_input'] = dict(packet['plan_input'], sha256='f' * 64)
    elif damage == 'claim':
        row = json.loads(case['claim_path'].read_bytes())
        row['resource_scope']['nonce'] = 'superseded-real-claim'
        case['claim_path'].write_text(json.dumps(row))
    else:
        # A foreign callback cannot adopt the real constructor's ownership.
        packet['resource_check'] = case['material']['checks'].append
    refusal = LeaseRefused if damage == 'claim' else (RuntimeError, ValueError)
    with pytest.raises(refusal):
        cc.require_original_source_authority(case['owner'], case['authority_input'], case['plan_input'], packet)
    assert case['owner'].receipt() == before
    assert not any(case['entries_path'].iterdir())


def test_duplicate_control_and_changed_independent_digest_fail_before_material(authority_case, monkeypatch):
    case = authority_case
    before = _forbid_source_work(case, monkeypatch)
    raw = Path(case['authority_input']['path']).read_bytes()
    duplicated = b'{"schema":"duplicate",' + raw[1:]
    path = case['tmp'] / 'duplicate-authority.json'
    path.write_bytes(duplicated)
    reference = dict(path=str(path), sha256=_digest(duplicated))
    packet = dict(case['packet'], authority_input=reference)
    with pytest.raises(RuntimeError, match='duplicate key schema'):
        cc.require_original_source_authority(None, reference, case['plan_input'], packet)
    reference = dict(case['authority_input'], sha256='0' * 64)
    packet = dict(case['packet'], authority_input=reference)
    with pytest.raises((RuntimeError, ValueError), match='owned bytes'):
        cc.require_original_source_authority(None, reference, case['plan_input'], packet)
    assert case['owner'].receipt() == before


def test_actual_native_receipt_preserves_reacquisitions_and_rejects_wrong_fd_identity(authority_case):
    case, owner = authority_case, authority_case['owner']
    path = owner.root / 'two.safetensors'
    for _ in range(2):
        with owner.material_window([path]):
            with owner.safe_open(safe_open, path, framework='pt') as reader:
                tensor = reader.get_tensor(reader.keys()[0])
                assert tensor.device.type == 'cpu'
            del tensor
        gc.collect()
    receipt = owner.receipt()
    rows = [row for row in receipt['deliveries'] if row['name'] == path.name]
    assert len(rows) >= 2
    assert len({row['delivery_index'] for row in rows}) == len(rows)
    assert all(row['payload_reads'] > 0 and not row['held'] for row in rows[-2:])
    assert not receipt['copy_completions'] and not receipt['pending_copy_completions']
    checked = cc.validate_original_source_material_receipt(receipt, case['authority'])
    checked['deliveries'][0]['native_delivery']['entry']['file_id']['ino'] += 1
    assert cc.validate_original_source_material_receipt(owner.receipt(), case['authority']) == receipt
    with pytest.raises(RuntimeError, match='portable native descriptor identity'):
        cc.validate_original_source_material_receipt(checked, case['authority'])
    missing = copy.deepcopy(receipt)
    missing['deliveries'] = [row for row in missing['deliveries'] if row['name'] != 'config.json']
    with pytest.raises(RuntimeError):
        cc.validate_original_source_material_receipt(missing, case['authority'])
    assert owner.receipt() == receipt


def test_real_partial_qualification_metadata_is_not_renamed_full64(authority_case):
    case = authority_case
    partial = dict(schema='prismaquant.original_source_qualification.v1', reader=None,
                   cuda=[], unchanged_family=[])
    authority = dict(case['authority'], qualification=_bound(case['tmp'] / 'partial.json', partial))
    # A rejection control only: never manufacture passed executions, events,
    # positive source qualifications or an independently approved root record.
    with pytest.raises(RuntimeError, match='partial full64'):
        sg._require_original_source_proofs(authority, case['owner'].resource_check)


def test_actual_cas_publication_cannot_adopt_an_independently_rebound_artifact(
        tmp_path, material, authority_case):
    """Real sealed action/CAS receipt bytes, not a stubbed result-reader echo."""
    from test_strict_reader_tier_enforcement import _pb

    import subprocess

    sdk, _pool, _map = _pb()
    from prismabuild import core, movement_actions

    checkout = tmp_path / 'artifact-checkout'
    checkout.mkdir()
    (checkout / 'publication.py').write_text('# CPU parser artifact fixture, not a CUDA execution\n')
    node = 'tests/private_cpu_artifact_join'
    receipt_binding = _bound(tmp_path / 'reader-receipt.json', {'observed_cpu_bytes': 17})
    authority_binding = _bound(tmp_path / 'reader-authority.json', {'independent_cpu_source': 'first'})
    artifacts = {role: dict(binding, bytes=Path(binding['path']).stat().st_size)
                 for role, binding in (('receipt', receipt_binding), ('authority', authority_binding))}
    publication = dict(schema='prismaquant.original_source_artifact_publication.v1',
                       node_id=node, artifacts=artifacts)
    payload = sg.ORIGINAL_ARTIFACT_PUBLICATION_PREFIX + json.dumps(
        publication, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    command = ['/bin/echo', node]
    def git(*args):
        return subprocess.run(['git', '-C', str(checkout), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    # Same real Git/bundle/ingestion recipe as the selected-result owner
    # fixture in test_pilot_verified_result_1293, not a partial descriptor.
    git('init', '-q', '-b', 'master')
    git('add', 'publication.py')
    git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-q', '-m', 'foreign baseline source')
    parent = git('rev-parse', 'HEAD')
    (checkout / 'publication.py').write_text('# actual foreign CPU snapshot source\n')
    git('add', 'publication.py')
    git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-q', '-m', 'foreign selected source')
    commit = git('rev-parse', 'HEAD')
    bundle = tmp_path / 'actual-foreign-source.bundle'
    git('bundle', 'create', str(bundle), '--all')
    cas = core.PrismaBuildCAS(tmp_path / 'artifact-cas')
    descriptor, _ = cas.ingest_input(bundle, input_id='pbrun.checkout-snapshot')
    action = core.seal_action({
        'schema': core.ACTION_SCHEMA_V2,
        'task': {'definition_id': 'tests/original-artifact-binding', 'definition_version': 'v1',
            'task_class': 'generation', 'determinism': 'deterministic', 'artifact_family': 'generic',
            'artifact_kind': 'generic', 'argv': movement_actions.standard_capture_argv(
                command, 'result.txt', path_prefix='/opt/pb-tools'),
            'working_directory': '.', 'result_path': 'result.txt'},
        'inputs': [descriptor], 'code_closure': core.build_code_closure(checkout, ['publication.py']),
        'params': {'command': command, 'cwd': '.', 'checkout_snapshot': {
            'schema': 'prismaquant.prismabuild.pbrun_checkout_snapshot.v2',
            'commit': commit, 'parent': parent, 'refs': {}, 'subdirectory': '.', 'input': descriptor}},
        'environment': {'variables': {'PATH': '/opt/pb-tools:/usr/bin:/bin'}, 'toolchain': {}},
        'execution_scope': {'portability': 'portable', 'platform_key': None, 'host_class': None},
    })
    cas.publish_action_request(action)
    output = tmp_path / 'actual-captured-payload.txt'
    output.write_bytes(payload)
    attestation = core.preflight_action(action, cas_root=cas.root, checkout_root=checkout)
    receipt, _ = cas.publish_result(action, output, attestation=attestation, return_execution_receipt=True)
    assert sdk.cas_receipt_self_check(receipt) is None
    owned = cas.read_declared_blob(dict(id='actual-result', **receipt['result']),
                                   max_bytes=len(payload), where='actual CPU result fixture')
    assert owned == payload and _digest(owned) == receipt['result']['sha256']
    published = sg._original_artifact_publication(owned, node_id=node, roles={'receipt', 'authority'})
    assert sg._published_original_artifact(published, 'receipt', receipt_binding,
        resource_check=material['options']['resource_check'], max_bytes=1024) == {'observed_cpu_bytes': 17}
    # Same pathname, new valid independent SHA: the selected CAS receipt
    # remains about the original observed artifact, not the replacement.
    changed = _bound(Path(receipt_binding['path']), {'observed_cpu_bytes': 17000})
    with pytest.raises(RuntimeError, match='selected published receipt binding'):
        sg._published_original_artifact(published, 'receipt', changed,
            resource_check=material['options']['resource_check'], max_bytes=1024)
    with pytest.raises(RuntimeError, match='selected node'):
        sg._original_artifact_publication(owned, node_id='another-reader', roles={'receipt', 'authority'})
    with pytest.raises(RuntimeError, match='old receipts remain unqualified'):
        sg._original_artifact_publication(b'old pytest summary without sidecar digests\n',
                                         node_id=node, roles={'receipt', 'authority'})
    # The same real sealed request cannot bypass actual target-source
    # transfer proof by declaring the foreign baseline and null compatibility.
    # No accepted-family or positive CUDA record is fabricated here.
    member = dict(node_id=node, source_snapshot=action['params']['checkout_snapshot']['parent'],
                  compatibility=None)
    with pytest.raises(RuntimeError, match='every qualified member requires'):
        sg._require_original_qualified_source(member, action, {}, authority_case['packet']['runtime'])
