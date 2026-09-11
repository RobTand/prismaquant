"""Selected capture reuse authenticates consumed source objects, once per row."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def selected_source_fixture(monkeypatch, tmp_path, *, priced=False):
    """One selected-source campaign on CPU: a real shard reader, a real capture.

    Returns the campaign module, the full selected argv (streaming, units,
    census, capture and its hash), and the observation lists the
    authentication test reads.  ``priced`` gives the unit a one-rung menu so
    the run encodes and journals one anchor instead of stopping at the
    empty-menu refusal.
    """
    import prismaquant
    from safetensors.torch import save_file
    from prismaquant import autoscale, cost_streaming, layer_streaming
    from prismaquant import tessera_calibration_cache as cc
    from test_tessera_campaign_resume import _main_fixture, UNIT

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    campaign, _, argv, model, inputs = _main_fixture(monkeypatch, tmp_path, priced=priced)
    model.config = SimpleNamespace(_attn_implementation='eager')
    model.lm_head = torch.nn.Linear(256, 32, bias=False, dtype=torch.bfloat16)
    selected = model.model.layers[0].proj.weight.detach().clone()
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    tensors = {
        'head.safetensors': {'lm_head.weight': model.lm_head.weight.detach().clone()},
        'selected.safetensors': {UNIT+'.weight': selected},
        'unused.safetensors': {'unused.weight': torch.zeros(1024, 1024, dtype=torch.bfloat16)},
    }
    weight_map = {}
    for name, values in tensors.items():
        save_file(values, str(source/name))
        weight_map.update(dict.fromkeys(values, name))
    (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    version = importlib.metadata.version('transformers')
    contract = dict(schema='prismaquant.streaming_initialization.v1',
        scope='streamed_text_source_forward', status='completed',
        transformers_version=version, model_class='SyntheticSource',
        dtype='torch.bfloat16', layers_prefix='model.layers.', num_layers=1,
        persistent_tensors=2, derived_buffers=0,
        state_sha256='a'*64, source_map_sha256='b'*64)
    calibration = campaign.th.calibration_identity(inputs['text'], inputs['tokens'], fit_tokens=4,
        source='wikitext-2-raw-v1/train', split_role='calibration', model=str(source),
        seed=0, nsamples=32, seqlen=512, fit_tokens_min=4)
    census = campaign.calibration_census({UNIT:4}, {UNIT:3.},
        args=SimpleNamespace(model=str(source), nsamples=32, seqlen=512, seed=0, layer_stride=1),
        groups={'u:'+UNIT:[UNIT]}, dense_targets=[UNIT], expert_targets=[],
        shapes={UNIT:[32,256]}, identity=calibration, model_load_contract=contract,
        attention_implementation='eager', capture_runtime=dict(torch=torch.__version__,
            cuda=torch.version.cuda, transformers=version))
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    canonical = cc.capture_identity(census_path, calibration=calibration, max_act_rows=4,
        model_load_contract=contract, attention_implementation='eager')
    complete = cc.publish_capture(tmp_path/'capture', census_path=census_path,
        identity=canonical, acts={UNIT:inputs['rows']}, hessians={UNIT:inputs['hessian']},
        counts=census['counts'], maxima=census['max_abs'])
    manifest_before = Path(complete['path']).read_bytes()
    copied, hashed, identities = [], [], []
    original_hash, original_identity = cc.sha256, cc.capture_identity

    def counting_hash(path, **kwargs):
        if Path(path).suffix == '.safetensors':
            hashed.append((Path(path).name, Path(path).stat().st_size))
        return original_hash(path, **kwargs)

    def record_identity(*args, **kwargs):
        result = original_identity(*args, **kwargs)
        identities.append(result)
        return result

    def build(*_args, **kwargs):
        # Keep the public campaign orchestration, and the actual shard reader.
        # Only model construction is the existing small orchestration fixture.
        authentication = kwargs.get('source_authentication')
        owned = {} if authentication is None else {'source_authentication': authentication}
        shards, keys = layer_streaming._build_weight_map(str(source), **owned)
        layer_streaming._materialize(model, ['lm_head.'], shards, keys,
            torch.device('cpu'), torch.bfloat16, **owned)

        def snapshot(names, **_kwargs):
            assert names == [UNIT]
            values = layer_streaming._read_layer_to_device('model.layers.0.', shards, keys,
                torch.bfloat16, torch.device('cpu'), **owned)
            value = values[UNIT+'.weight'].clone()
            copied.append(value)
            return {UNIT:value}, dict(schema='prismaquant.selected_source_weights.v1',
                source_forward_count=0)

        return SimpleNamespace(model=model, snapshot_selected_weights=snapshot, shutdown=lambda: None)

    monkeypatch.setattr(cc, 'sha256', counting_hash)
    monkeypatch.setattr(cc, 'capture_identity', record_identity)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', build)
    monkeypatch.setattr(autoscale, 'selected_anchor_resources', lambda *a, **k: dict(
        memory_bytes=1024**3, selected_source_weight_bytes=selected.numel()*2,
        # The campaign builds the encoder memo with the capacity the plan
        # charged for, so a stub plan publishes one too.
        encoder_memo_capacity=1,
        phases={'resident_anchors': {'factorization_scratch_bytes': 4*256**2*4}}))
    monkeypatch.setattr(campaign, '_collect_activations', lambda *a, **k: pytest.fail('repeated forward'))
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(schema=campaign.UNITS_SCHEMA,
        groups=[dict(key='u:'+UNIT, members=[UNIT])])) )
    argv[argv.index('--model')+1] = str(source)
    argv[argv.index('--hessian')+1] = 'require'
    selected_argv = [*argv, '--attention-implementation', 'eager', '--streaming',
        '--streaming-cache-headroom-gb', '0', '--units', str(selection),
        '--calibration-census', str(census_path), '--calibration-cache', complete['path'],
        '--calibration-cache-sha256', complete['sha256'], '--nsamples', '32',
        '--seqlen', '512', '--layer-stride', '1', '--max-act-rows', '4']
    state = dict(copied=copied, hashed=hashed, identities=identities, selected=selected,
                 canonical=canonical, complete=complete, manifest_before=manifest_before,
                 source=source)
    return campaign, selected_argv, state


def test_selected_campaign_hashes_only_consumed_shards_and_preserves_identity(
    monkeypatch, tmp_path,
):
    """The public selected path must not hash an untouched payload shard."""
    campaign, selected_argv, state = selected_source_fixture(monkeypatch, tmp_path)
    copied, hashed, identities = state['copied'], state['hashed'], state['identities']
    selected, canonical, complete = state['selected'], state['canonical'], state['complete']
    manifest_before, source = state['manifest_before'], state['source']
    assert campaign.main([*selected_argv, '--campaign-identity-bytes', '1048576']) == campaign.EXIT_EMPTY_MENU
    assert copied and torch.equal(copied[0], selected)
    assert identities and all(value == canonical for value in identities)
    assert hashlib.sha256(Path(complete['path']).read_bytes()).hexdigest() == complete['sha256']
    assert Path(complete['path']).read_bytes() == manifest_before
    expected = sorted((name, (source/name).stat().st_size)
        for name in ('head.safetensors', 'selected.safetensors'))
    print({'hashed_shards': hashed, 'hashed_payload_bytes': sum(size for _,size in hashed),
           'required_payload_bytes': sum(size for _,size in expected)})
    assert sorted(hashed) == expected


@pytest.fixture
def complete_source(tmp_path):
    from safetensors.torch import save_file
    from prismaquant import tessera_calibration_cache as cc
    root = tmp_path/'model'
    root.mkdir()
    (root/'config.json').write_text('{}')
    (root/'chat_template.jinja').write_text('immutable auxiliary')
    tensors = {'head.safetensors': {'lm_head.weight': torch.ones(2, 2)},
        'selected.safetensors': {'model.layers.0.a.weight': torch.arange(4.).reshape(2, 2),
                                 'model.layers.0.b.weight': torch.ones(2, 2)*3},
        'unused.safetensors': {'model.layers.1.a.weight': torch.ones(2, 2)*9}}
    mapping = {}
    for name, values in tensors.items():
        save_file(values, str(root/name))
        mapping.update(dict.fromkeys(values, name))
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    version = importlib.metadata.version('transformers')
    contract = dict(schema='prismaquant.streaming_initialization.v1',
        scope='streamed_text_source_forward', status='completed', transformers_version=version,
        model_class='SyntheticSource', dtype='torch.bfloat16', layers_prefix='model.layers.',
        num_layers=2, persistent_tensors=4, derived_buffers=0,
        state_sha256='a'*64, source_map_sha256='b'*64)
    source = dict(files={name: cc.sha256(root/name) for name in tensors}, tensors=mapping,
        config_sha256=cc.sha256(root/'config.json'),
        auxiliary_sha256={'chat_template.jinja': cc.sha256(root/'chat_template.jinja')})
    census = dict(model=str(root), model_load_contract=contract,
        capture_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=version),
        attention_implementation='eager', nsamples=2, seqlen=2, seed=0, layer_stride=1,
        counts={'a':2}, max_abs={'a':3.}, unit_shapes={'a':[2,2]},
        anchor_groups={'u:a':['a']}, expert_projection={'producer': {'source':source}})
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    canonical = cc.capture_identity(census_path, calibration={'fit_ids_sha256':'draw'},
        max_act_rows=2, model_load_contract=contract, attention_implementation='eager')
    capture = cc.publish_capture(tmp_path/'capture', census_path=census_path, identity=canonical,
        acts={'a':torch.ones(2,2)}, hessians={'a':torch.eye(2)},
        counts=census['counts'], maxima=census['max_abs'])
    kwargs = dict(census_path=census_path, capture_path=capture['path'],
        expected_sha256=capture['sha256'], model=str(root), max_act_rows=2,
        attention_implementation='eager', calibration_parameters=dict(nsamples=2,seqlen=2,seed=0))
    return SimpleNamespace(root=root, census=census, canonical=canonical, kwargs=kwargs,
                           tensors=tensors, capture=capture)


def _mutate_preserving_mtime(path):
    import os
    before = path.stat()
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        old = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([old[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.mark.parametrize('name', ['config.json', 'model.safetensors.index.json', 'chat_template.jinja'])
def test_metadata_tampering_refused_before_source_construction(complete_source, name):
    from prismaquant import tessera_calibration_cache as cc
    _mutate_preserving_mtime(complete_source.root/name)
    with pytest.raises(RuntimeError, match='content differs'):
        cc.authenticate_selected_capture_source(**complete_source.kwargs)


@pytest.mark.parametrize('change', ['missing_hash', 'wrong_hash', 'partial', 'missing_entry',
                                    'identity_units', 'census_draw', 'census_roster'])
def test_complete_capture_and_census_binding_fail_closed(complete_source, change):
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    kwargs = dict(f.kwargs)
    if change == 'missing_hash':
        kwargs['expected_sha256'] = None
    elif change == 'wrong_hash':
        kwargs['expected_sha256'] = '0'*64
    elif change.startswith('census_'):
        census = dict(f.census)
        if change == 'census_draw':
            census['seed'] = 1
        else:
            census['unit_shapes'] = {'unknown':[2,2]}
        kwargs['census_path'].write_text(json.dumps(census))
    else:
        path = Path(kwargs['capture_path'])
        manifest = json.loads(path.read_text())
        if change == 'partial':
            manifest['status'] = 'capturing'
        elif change == 'missing_entry':
            manifest['entries'].clear()
        else:
            manifest['identity']['units']['a'] = [4,2]
        path.write_text(json.dumps(manifest))
        kwargs['expected_sha256'] = cc.sha256(path)
    with pytest.raises(RuntimeError):
        cc.authenticate_selected_capture_source(**kwargs)


def test_new_source_roster_entry_is_not_accepted(complete_source):
    from prismaquant import tessera_calibration_cache as cc
    (complete_source.root/'new.json').write_text('{}')
    with pytest.raises(RuntimeError, match='roster changed'):
        cc.authenticate_selected_capture_source(**complete_source.kwargs)


def test_unconsumed_payload_is_not_hashed_but_later_consumption_is_refused(complete_source, monkeypatch):
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    _mutate_preserving_mtime(f.root/'unused.safetensors')
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        hashed = []
        original = cc.sha256
        def count(path, **kwargs):
            hashed.append(Path(path).name)
            assert kwargs['file_descriptor'] >= 0
            return original(path, **kwargs)
        monkeypatch.setattr(cc, 'sha256', count)
        with owner.safe_open(safe_open, f.root/'unused.safetensors', framework='pt') as handle:
            assert handle.get_slice('model.layers.1.a.weight').get_shape() == [2,2]
        assert not hashed
        with owner.safe_open(safe_open, f.root/'selected.safetensors', framework='pt') as handle:
            assert torch.equal(handle.get_tensor('model.layers.0.a.weight'), f.tensors['selected.safetensors']['model.layers.0.a.weight'])
        with pytest.raises(RuntimeError, match='content differs'):
            with owner.safe_open(safe_open, f.root/'unused.safetensors', framework='pt') as handle:
                handle.get_slice('model.layers.1.a.weight')[:]
        assert hashed == ['selected.safetensors', 'unused.safetensors']
        assert owner.receipt()['payload_bytes_hashed'] == (f.root/'selected.safetensors').stat().st_size


def test_threaded_shared_shard_hashes_once_and_preserves_read_bytes(complete_source, monkeypatch):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        original, hashed = cc.sha256, []
        def count(path, **kwargs):
            fd = kwargs['file_descriptor']
            assert os.fstat(fd).st_ino == Path(path).stat().st_ino
            hashed.append((Path(path).name, fd))
            return original(path, **kwargs)
        monkeypatch.setattr(cc, 'sha256', count)
        barrier = threading.Barrier(8)
        def read(_index):
            with owner.safe_open(safe_open, f.root/'selected.safetensors', framework='pt') as handle:
                barrier.wait(timeout=10)
                a = handle.get_slice('model.layers.0.a.weight')[:].clone()
                b = handle.get_tensor('model.layers.0.b.weight').clone()
                return a, b
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(read, range(8)))
        for a, b in values:
            assert torch.equal(a, f.tensors['selected.safetensors']['model.layers.0.a.weight'])
            assert torch.equal(b, f.tensors['selected.safetensors']['model.layers.0.b.weight'])
        assert len(hashed) == 1
        row = next(row for row in owner.receipt()['verified_files'] if row['name'].endswith('.safetensors'))
        assert row['payload_reads'] == 16
        fd = hashed[0][1]
    with pytest.raises(OSError):
        os.fstat(fd)


def test_owned_descriptor_refuses_path_replacement_before_factory_returns(complete_source):
    import os
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    owner = cc.authenticate_selected_capture_source(**f.kwargs)
    path = f.root/'selected.safetensors'
    held, released = [], []
    class Tracked:
        def __init__(self, context):
            self.context = context
        def __exit__(self, *args):
            released.append(True)
            return self.context.__exit__(*args)
    def replace(alias, **kwargs):
        held.append(int(alias.rsplit('/', 1)[1]))
        assert Path(alias).stat().st_ino == path.stat().st_ino
        replacement = f.root/'replacement.tmp'
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        return Tracked(safe_open(alias, **kwargs))
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.safe_open(replace, path, framework='pt')
    assert released == [True]
    assert owner._readers == 0
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.close()
    with pytest.raises(OSError):
        os.fstat(held[0])


def test_mutation_after_hash_is_rejected_and_descriptors_close(complete_source, monkeypatch):
    import os
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    owner = cc.authenticate_selected_capture_source(**f.kwargs)
    original, descriptors = cc.sha256, []
    def mutate(path, **kwargs):
        descriptors.append(kwargs['file_descriptor'])
        result = original(path, **kwargs)
        _mutate_preserving_mtime(Path(path))
        return result
    monkeypatch.setattr(cc, 'sha256', mutate)
    with pytest.raises(RuntimeError, match='changed during consumption'):
        with owner.safe_open(safe_open, f.root/'selected.safetensors', framework='pt') as handle:
            handle.get_tensor('model.layers.0.a.weight')
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.close()
    assert descriptors
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_read_lease_prevents_close_and_slice_escape(complete_source):
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    owner = cc.authenticate_selected_capture_source(**f.kwargs)
    with owner.safe_open(safe_open, f.root/'selected.safetensors', framework='pt') as handle:
        view = handle.get_slice('model.layers.0.a.weight')
        with pytest.raises(RuntimeError, match='active readers'):
            owner.close()
        assert torch.equal(view[:], f.tensors['selected.safetensors']['model.layers.0.a.weight'])
    with pytest.raises(RuntimeError, match='outside its read lease'):
        view[:]
    owner.close()
    owner.close()
    with pytest.raises(RuntimeError, match='closed'):
        owner.read_json(f.root/'config.json')


def test_unknown_source_path_refused_before_open(complete_source):
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        with pytest.raises(RuntimeError, match='absent from the sealed roster'):
            owner.safe_open(safe_open, f.root/'../other.safetensors', framework='pt')


def test_layer_reader_drains_other_owned_chunks_before_error(complete_source, monkeypatch):
    import threading
    import time
    from safetensors import safe_open
    from prismaquant import layer_streaming as ls, tessera_calibration_cache as cc
    f = complete_source
    started, finished = threading.Event(), threading.Event()
    class Reader:
        def __init__(self, alias, **kwargs):
            self.context = safe_open(alias, **kwargs)
        def __enter__(self):
            self.handle = self.context.__enter__()
            return self
        def __exit__(self, *args):
            return self.context.__exit__(*args)
        def get_tensor(self, name):
            if name.endswith('a.weight'):
                assert started.wait(timeout=5)
                raise RuntimeError('intentional first chunk failure')
            started.set()
            time.sleep(0.05)
            result = self.handle.get_tensor(name)
            finished.set()
            return result
    monkeypatch.setattr(ls, 'safe_open', Reader)
    monkeypatch.setattr(ls, 'layer_read_threads', lambda: 2)
    monkeypatch.setattr(ls, '_LAYER_READ_MIN_TENSORS', 1)
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        shards, keys = ls._build_weight_map(str(f.root), source_authentication=owner)
        with pytest.raises(RuntimeError, match='intentional first chunk failure'):
            ls._read_layer_to_device('model.layers.0.', shards, keys, torch.float32,
                torch.device('cpu'), source_authentication=owner)
        assert finished.is_set()
        assert owner._readers == 0


def test_projected_tensor_and_scale_readers_share_source_authentication(complete_source, monkeypatch):
    from prismaquant import layer_streaming as ls, tessera_calibration_cache as cc
    from prismaquant.tessera_expert_projection import source_unit_weight
    f = complete_source
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        original, hashed = cc.sha256, []
        def count(path, **kwargs):
            hashed.append(Path(path).name)
            return original(path, **kwargs)
        monkeypatch.setattr(cc, 'sha256', count)
        key = 'model.layers.0.a.weight'
        source = f.census['expert_projection']['producer']['source']
        unit = dict(source_tensor=key, rows=2, cols=2)
        for _ in range(2):
            weight = source_unit_weight(f.root, source, unit, source_authentication=owner)
            assert torch.equal(weight, f.tensors['selected.safetensors'][key])
        out = {'quantized': torch.ones(4,4)}
        scales = ls.Fp8ScaleInvMap({'quantized':(str(f.root/'unused.safetensors'),
            'model.layers.1.a.weight')}, block=(2,2))
        assert ls._apply_fp8_dequant_inplace(out, scales, torch.device('cpu'),
            source_authentication=owner) == 1
        assert torch.equal(out['quantized'], torch.ones(4,4)*9)
        assert hashed == ['selected.safetensors', 'unused.safetensors']


def test_metadata_read_lease_prevents_concurrent_close(complete_source, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        entered, resume = threading.Event(), threading.Event()
        original = cc.json.load
        def blocked(handle):
            entered.set()
            assert resume.wait(timeout=5)
            return original(handle)
        monkeypatch.setattr(cc.json, 'load', blocked)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(owner.read_json, f.root/'config.json')
            assert entered.wait(timeout=5)
            try:
                with pytest.raises(RuntimeError, match='active readers'):
                    owner.close()
            finally:
                resume.set()
            assert future.result(timeout=5) == {}


def test_manifest_hash_and_parse_use_the_same_bytes(complete_source, monkeypatch):
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    path = Path(f.capture['path'])
    original = Path.read_bytes
    def swap_after_read(value):
        raw = original(value)
        if value == path:
            value.write_text('{"status":"untrusted replacement"}')
        return raw
    monkeypatch.setattr(Path, 'read_bytes', swap_after_read)
    manifest = cc.require_capture_contract(path, expected_sha256=f.capture['sha256'])
    assert manifest['identity'] == f.canonical
    assert manifest['status'] == 'complete'


def test_fifo_source_replacement_refuses_without_blocking(complete_source):
    import multiprocessing
    import os
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    path = f.root/'selected.safetensors'
    path.unlink()
    os.mkfifo(path)
    # A forked, owned probe makes a blocking-open regression bounded too.
    context = multiprocessing.get_context('fork')
    outcomes = context.Queue()
    def open_source():
        owner = cc.CaptureSourceAuthentication(f.root, f.canonical,
            f.census['expert_projection']['producer']['source'], manifest_sha256=f.capture['sha256'])
        try:
            owner.file_stat(path)
        except RuntimeError as exc:
            outcomes.put(str(exc))
        finally:
            owner.close()
    process = context.Process(target=open_source)
    process.start()
    process.join(timeout=2)
    hung = process.is_alive()
    if hung:
        process.terminate()
        process.join(timeout=5)
    try:
        assert not hung, 'source open blocked on a FIFO before regular-file validation'
        assert process.exitcode == 0
        assert 'regular file' in outcomes.get(timeout=2)
    finally:
        outcomes.close()
        outcomes.join_thread()


@pytest.mark.parametrize('failure', ['enter', 'exit'])
def test_underlying_context_failure_releases_read_lease(complete_source, failure):
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    calls = []
    class Failing:
        def __init__(self, *_args, **_kwargs):
            pass
        def __enter__(self):
            calls.append('enter')
            if failure == 'enter':
                raise RuntimeError('context enter failure')
            return self
        def __exit__(self, *_args):
            calls.append('exit')
            if failure == 'exit':
                raise RuntimeError('context exit failure')
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        with pytest.raises(RuntimeError, match=f'context {failure} failure'):
            with owner.safe_open(Failing, f.root/'selected.safetensors', framework='pt'):
                pass
        assert calls == ['enter', 'exit']
        assert owner._readers == 0


def test_legitimate_hf_style_source_symlink_preserves_authentication(complete_source, tmp_path):
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    f = complete_source
    path = f.root/'selected.safetensors'
    blob = tmp_path/'hub-blob'
    path.rename(blob)
    path.symlink_to(blob)
    with cc.authenticate_selected_capture_source(**f.kwargs) as owner:
        with owner.safe_open(safe_open, path, framework='pt') as handle:
            value = handle.get_tensor('model.layers.0.a.weight')
            assert torch.equal(value, f.tensors['selected.safetensors']['model.layers.0.a.weight'])
        assert owner.receipt()['payload_bytes_hashed'] == blob.stat().st_size
