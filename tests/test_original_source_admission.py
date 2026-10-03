"""Real CPU owner/control joins, never fabricated original CUDA/root admission."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import source_generation as sg, tessera_calibration_cache as cc
from prismaquant.calibration_data import load_calibration_input
from prismaquant.cost_streaming import build_streamed_model_identity
from prismaquant.joint_aura import source_execution_identity
from prismaquant.staged_lease import resolve_context
from prismaquant.joint_adjoint_checkpoints import adjoint_space, boundary_entry_directory
from prismaquant.stage_a_selected_row_diagnostic import prepare_original_diagnostic_session
from test_capture_original_material import _bound, material, _forget_state  # noqa: F401
from test_original_source_identity import original_runner  # noqa: F401
from test_original_streaming_bootstrap_2010 import original_model  # noqa: F401
from test_strict_reader_tier_enforcement import _publish_readset_on_the_claim

pytestmark = pytest.mark.own_process


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
    claim_row['demand'] = {'cpu': 1, 'mem_gb': 1}
    claim_path.write_text(json.dumps(claim_row))
    prefetch = dict(max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1,
        cache_headroom_gb=1, prefetch_min_available_gb=1, require_prefetched_residency=True)
    resources = dict(schema='prismaquant.original_source_resources.v1', cpu_bytes=1024**3,
        material_bytes=material['options']['max_material_bytes'], source_cache_bytes=1024**2,
        source_prefetch=prefetch, copy_bytes=1024**2, gpu_bytes=0, native_bytes=0,
        serialization_bytes=16 * 1024**2, artifact_bytes=16 * 1024**2,
        deadline_seconds=60, stall_seconds=10, host_floor_bytes=1, margin_bytes=0,
        claim_demand=claim_row['demand'])
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


def _forbid_source_work(case, monkeypatch):
    owner = case['owner']
    before = owner.receipt()
    def forbidden(*args, **kwargs):
        pytest.fail('authority intake opened source material/profile/output')
    monkeypatch.setattr(owner, '_open_source_state', forbidden)
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, 'detect_profile', forbidden)
    return before


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
    with pytest.raises((RuntimeError, ValueError)):
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
    with pytest.raises(RuntimeError, match='bootstrap auxiliary'):
        cc.validate_original_source_material_receipt(missing, case['authority'])


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
    action = core.seal_action({
        'schema': core.ACTION_SCHEMA_V2,
        'task': {'definition_id': 'tests/original-artifact-binding', 'definition_version': 'v1',
            'task_class': 'generation', 'determinism': 'deterministic', 'artifact_family': 'generic',
            'artifact_kind': 'generic', 'argv': movement_actions.standard_capture_argv(
                command, 'result.txt', path_prefix='/opt/pb-tools'),
            'working_directory': '.', 'result_path': 'result.txt'},
        'inputs': [], 'code_closure': core.build_code_closure(checkout, ['publication.py']),
        'params': {'command': command, 'checkout_snapshot': {
            'parent': '4d0a88e0057c7f01ff170fffb4744d0c5b5d29fe'}},
        'environment': {'variables': {'PATH': '/opt/pb-tools:/usr/bin:/bin'}, 'toolchain': {}},
        'execution_scope': {'portability': 'portable', 'platform_key': None, 'host_class': None},
    })
    cas = core.PrismaBuildCAS(tmp_path / 'artifact-cas')
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
    with pytest.raises(RuntimeError, match='every CUDA member requires'):
        sg._require_original_qualified_source(member, action, {}, authority_case['packet']['runtime'])
