"""A streamed capture reads its source once and its own output never (PQ #1896).

The streamed capture hashed every source file before its forward, then read
the same bytes again for the tensors; a capture chain's prep did the same.
The writer then re-read every entry it had just written to hash it, and the
seal read them a third time. These tests hold the explicit legacy recording/write-digest mechanism.
The end-to-end fixture bypasses automatic admission only to exercise that
preserved mechanism; it is not positive immutable-provider qualification.
Automatic campaigns are separately refused by
``test_automatic_capture_source_qualification.py``. The mechanism checks:

* the source is hashed by the read that consumes it, through the descriptor
  the tensors are read through, before the first tensor reaches the capture;
* metadata-observable changes between the hash and a later read refuse;
  same-signature mutation detection remains an unmet requirement (#2010);
* a census producer digest is compared at that first use and refuses before
  any tensor is read;
* a chain prep hashes nothing, and its join refuses quanta that recorded
  different digests for one file;
* each capture entry is hashed while it is written and never read back, and
  the seal holds it to the stat fingerprint taken then.
"""
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import capture  # noqa: F401  (fixture)


def _mutate_preserving_mtime(path):
    """Flip one payload byte and restore mtime; ctime cannot be restored."""
    before = path.stat()
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        old = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([old[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _source(tmp_path, *, producer=None):
    """Two real safetensors shards, a config, and a census naming them."""
    from safetensors.torch import save_file
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    save_file({'w': torch.arange(64, dtype=torch.float32).reshape(8, 8)},
              str(source / 'model-00001.safetensors'))
    save_file({'v': torch.ones(4, 4)}, str(source / 'model-00002.safetensors'))
    census = dict(model=str(source))
    if producer is not None:
        census['expert_projection'] = {'producer': {'source': producer}}
    path = tmp_path / 'census.json'
    path.write_text(json.dumps(census))
    return source, path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- the source side ------------------------------------------------------------

def test_the_read_that_consumes_a_file_hashes_it_through_its_own_descriptor(tmp_path, monkeypatch):
    from safetensors import safe_open
    source, census = _source(tmp_path)
    shard = source / 'model-00001.safetensors'
    hashes = []
    original = cc.sha256

    def spy(path, **kwargs):
        hashes.append((Path(path).name, kwargs.get('file_descriptor')))
        return original(path, **kwargs)
    monkeypatch.setattr(cc, 'sha256', spy)
    with cc.record_capture_source(census, model=source) as owner:
        with owner.safe_open(safe_open, shard, framework='pt') as reader:
            assert hashes == []  # opening reads the header only
            assert torch.equal(reader.get_tensor('w'),
                               torch.arange(64, dtype=torch.float32).reshape(8, 8))
            reader.get_tensor('w')
        held = owner._files[shard.name]['fd']
        assert hashes == [(shard.name, held)]  # once, through the held descriptor
        receipt = owner.receipt()
    assert receipt['schema'] == cc.RECORDING_RECEIPT_SCHEMA
    assert [(row['name'], row['sha256'], row['payload_reads'])
            for row in receipt['verified_files']] == [(shard.name, _digest(shard), 2)]
    assert receipt['metadata_only_shards'] == []


def test_bytes_that_change_between_the_hash_and_the_read_refuse(tmp_path):
    from safetensors import safe_open
    source, census = _source(tmp_path)
    shard = source / 'model-00001.safetensors'
    owner = cc.record_capture_source(census, model=source)
    reader = owner.safe_open(safe_open, shard, framework='pt')
    with pytest.raises(RuntimeError, match='changed during consumption'):
        with reader as handle:
            handle.get_tensor('w')  # the hash, then the read
            _mutate_preserving_mtime(shard)
            handle.get_tensor('w')
    # Nothing the owner recorded can be sealed: the object it hashed is gone.
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.recorded_source_files()
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.close()


def test_a_change_after_the_last_read_refuses_at_the_read_lease_exit(tmp_path):
    """Exercise the exit fence with deterministic observable drift, even on ZFS.

    Immediate writes can preserve ctime within a filesystem timestamp tick
    (#2010); the separate test above retains the mtime-restoration control.
    """
    from safetensors import safe_open
    from prismaquant.file_identity import file_stat_signature
    source, census = _source(tmp_path)
    shard = source / 'model-00001.safetensors'
    owner = cc.record_capture_source(census, model=source)
    with pytest.raises(RuntimeError, match='changed during consumption'):
        with owner.safe_open(safe_open, shard, framework='pt') as handle:
            handle.get_tensor('w')
            before = shard.stat()
            _mutate_preserving_mtime(shard)
            os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            assert file_stat_signature(shard.stat()) != file_stat_signature(before)
    with pytest.raises(RuntimeError, match='changed during consumption'):
        owner.close()


def test_a_census_producer_digest_refuses_before_the_first_tensor(tmp_path):
    source, census = _source(tmp_path, producer={'files': {'model-00001.safetensors': '0' * 64}})
    shard = source / 'model-00001.safetensors'
    reads = []

    class _Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_tensor(self, name):
            reads.append(name)
            return torch.zeros(1)

    with cc.record_capture_source(census, model=source) as owner:
        with pytest.raises(RuntimeError, match='differs from census producer'):
            with owner.safe_open(lambda path, **_kw: _Handle(), shard) as handle:
                handle.get_tensor('w')
    assert reads == []  # refused before the read, not after it


def test_a_matching_census_producer_digest_is_recorded_as_verified(tmp_path):
    from safetensors import safe_open
    source, census = _source(tmp_path)
    digest = _digest(source / 'model-00001.safetensors')
    census.write_text(json.dumps(dict(model=str(source), expert_projection={'producer': {
        'source': {'files': {'model-00001.safetensors': digest}}}})))
    with cc.record_capture_source(census, model=source) as owner:
        with owner.safe_open(safe_open, source / 'model-00001.safetensors', framework='pt') as handle:
            handle.get_tensor('w')
        assert owner.receipt()['producer_verified'] == ['model-00001.safetensors']


def test_the_sealed_identity_is_the_full_capture_identity(capture):  # noqa: F811
    """Traversal identity plus the recorded roster is what full hashing seals."""
    _root, path, census, identity, *_rest = capture
    with cc.record_capture_source(path, model=census['model']) as owner:
        traversal = cc.capture_identity(path, calibration={'fit_ids_sha256': 'draw'},
            max_act_rows=2, model_load_contract=identity['model_load_contract'],
            attention_implementation='eager', source_authentication=owner)
        assert 'source_files' not in traversal
        assert owner.receipt()['verified_files'] == []  # the identity read nothing
        owner.authenticate_complete_source()
        assert cc.bind_capture_source(traversal, owner.recorded_source_files()) == identity
    with pytest.raises(RuntimeError, match='already binds'):
        cc.bind_capture_source(identity, identity['source_files'])


def test_a_source_file_added_during_the_capture_refuses_its_seal(tmp_path):
    source, census = _source(tmp_path)
    with cc.record_capture_source(census, model=source) as owner:
        owner.authenticate_complete_source()
        (source / 'tokenizer.json').write_text('{}')
        with pytest.raises(RuntimeError, match='roster changed during the capture'):
            owner.recorded_source_files()


def test_a_capture_reads_the_source_its_census_names(tmp_path):
    source, census = _source(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(RuntimeError, match='the source its census names'):
        cc.record_capture_source(census, model=other)


def test_retained_pages_go_after_the_last_consumer(tmp_path, monkeypatch):
    """The payload hash keeps a file's pages; the traversal releases them later."""
    from safetensors import safe_open
    source, census = _source(tmp_path)
    advised = []
    original = os.posix_fadvise

    def spy(fd, offset, length, advice):
        advised.append((fd, offset, length, advice))
        return original(fd, offset, length, advice)
    monkeypatch.setattr(os, 'posix_fadvise', spy)
    first, second = source / 'model-00001.safetensors', source / 'model-00002.safetensors'
    with cc.record_capture_source(census, model=source, release_read_pages=True) as owner:
        for shard, key in ((first, 'w'), (second, 'v')):
            with owner.safe_open(safe_open, shard, framework='pt') as handle:
                handle.get_tensor(key)
        assert advised == []  # the hash dropped nothing the reads needed
        assert owner.release_retained_pages(keep=[second]) == (first.name,)
        fd = owner._files[first.name]['fd']
        assert advised == [(fd, 0, 0, os.POSIX_FADV_DONTNEED)]
        assert owner.release_retained_pages(keep=[second]) == ()
        assert owner.release_retained_pages() == (second.name,)
        owner.authenticate_complete_source()  # config.json: nobody reads it
        config_fd = owner._files['config.json']['fd']
        assert owner.receipt()['released_files'] == [first.name, second.name]
        advised.clear()
    # Nothing the owner retained outlives it.
    assert advised == [(config_fd, 0, 0, os.POSIX_FADV_DONTNEED)]
    # A capture without the bounded page policy leaves page cache alone.
    advised.clear()
    with cc.record_capture_source(census, model=source) as owner:
        with owner.safe_open(safe_open, first, framework='pt') as handle:
            handle.get_tensor('w')
        assert owner.release_retained_pages() == ()
    assert advised == []


# -- the capture chain ------------------------------------------------------------

@pytest.mark.parametrize('alias, definition', [
    ('fragment_path', 'capture_fragment_path'),
    ('generation_directory', 'capture_generation_directory'),
    ('owner_status_path', 'capture_owner_status_path'),
    ('_seal', '_seal_capture_document'),
    ('_read_sealed', '_read_capture_document'),
    ('prepare', 'prepare_capture_chain'),
])
def test_capture_chain_preserves_domain_qualified_definitions_and_aliases(alias, definition):
    from prismaquant import capture_layer_chain as chain
    owner = getattr(chain, definition)
    assert getattr(chain, alias) is owner
    assert owner.__name__ == definition


def test_a_capture_chain_prep_reads_no_source_payload(tmp_path, monkeypatch):
    from test_capture_layer_chain import _Chain
    hashed = []
    original = cc.sha256

    def spy(path, **kwargs):
        hashed.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(cc, 'sha256', spy)
    fixture = _Chain(tmp_path)
    assert hashed == []
    assert 'source_files' not in fixture.prep['identity']


def test_a_v1_capture_chain_prep_is_refused(tmp_path):
    from prismaquant import capture_layer_chain as chain
    from test_capture_layer_chain import _Chain
    fixture = _Chain(tmp_path)
    fixture.reseal(chain.prep_path(fixture.root), 'prep_sha256',
                   lambda document: document.update(schema='prismaquant.capture_layer_chain.prep.v1'))
    with pytest.raises(chain.CaptureChainRefused, match='not a prismaquant.capture_layer_chain.prep.v2'):
        chain.read_prep(fixture.root)


def test_the_join_refuses_quanta_that_read_different_source_bytes(tmp_path):
    from prismaquant import capture_layer_chain as chain
    from test_capture_layer_chain import _Chain
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))

    def recorded(digest):
        def change(document):
            document['source_authentication']['verified_files'] = [
                {'name': 'model.safetensors', 'sha256': digest, 'payload_reads': 1}]
        return change
    fixture.reseal(chain.fragment_path(fixture.root, 0, 1), 'fragment_sha256', recorded('a' * 64))
    fixture.reseal(chain.fragment_path(fixture.root, 1, 2), 'fragment_sha256', recorded('b' * 64))
    with pytest.raises(chain.CaptureChainRefused, match='read different source bytes'):
        chain.join(fixture.root, census_path=fixture.census)
    assert not (fixture.root / 'capture_manifest.json').exists()


def test_the_join_binds_what_the_quanta_recorded_without_reading_it(tmp_path, monkeypatch):
    from prismaquant import capture_layer_chain as chain
    from test_capture_layer_chain import _Chain
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    shard = fixture.source / 'model.safetensors'

    def recorded(document):
        document['source_authentication']['verified_files'] = [
            {'name': shard.name, 'sha256': _digest(shard), 'payload_reads': 1}]
    fixture.reseal(chain.fragment_path(fixture.root, 0, 1), 'fragment_sha256', recorded)
    hashed = []
    original = cc.sha256

    def spy(path, **kwargs):
        hashed.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(cc, 'sha256', spy)
    record = chain.join(fixture.root, census_path=fixture.census)
    assert shard.name not in hashed and 'config.json' in hashed
    assert record['source_authentication']['schema'] == cc.RECORDING_RECEIPT_SCHEMA
    rows = {row['name']: row for row in record['source_authentication']['verified_files']}
    assert rows[shard.name]['sha256_source'] == 'recorded_by_capture_reader'
    published = json.loads(Path(record['manifest']['path']).read_text())
    assert published['identity']['source_files'] == {
        name: _digest(fixture.source / name) for name in ('config.json', shard.name)}


# -- the output side ------------------------------------------------------------

def _entry_reads(monkeypatch, root):
    """Every whole-file hash or ``torch.load`` of a file under ``root``."""
    reads = []
    original_hash, original_load = cc.sha256, torch.load
    root = Path(root).resolve()

    def hash_spy(path, **kwargs):
        if root in Path(path).resolve().parents:
            reads.append(('sha256', Path(path).name))
        return original_hash(path, **kwargs)

    def load_spy(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and root in Path(path).resolve().parents:
            reads.append(('load', Path(path).name))
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(cc, 'sha256', hash_spy)
    monkeypatch.setattr(torch, 'load', load_spy)
    return reads


def test_the_capture_writer_hashes_each_entry_as_it_writes_it(capture, tmp_path, monkeypatch):  # noqa: F811
    _root, path, census, identity, acts, hessians, monolith = capture
    root = tmp_path / 'written'
    reads = _entry_reads(monkeypatch, root)
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    record = writer.finish(model_load_contract=identity['model_load_contract'])
    assert reads == []
    published = json.loads(Path(record['path']).read_text())
    assert published == json.loads(Path(monolith['path']).read_text())
    for name, entry in published['entries'].items():
        assert entry['sha256'] == _digest(root / entry['path'])
    assert record['sha256'] == _digest(record['path'])


def test_publish_capture_never_reads_back_an_entry_it_wrote(capture, tmp_path, monkeypatch):  # noqa: F811
    _root, path, census, identity, acts, hessians, monolith = capture
    root = tmp_path / 'published'
    reads = _entry_reads(monkeypatch, root)
    record = cc.publish_capture(root, census_path=path, identity=identity, acts=acts,
                                hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    assert reads == []
    assert Path(record['path']).read_bytes() == Path(monolith['path']).read_bytes()
    assert record['sha256'] == _digest(record['path'])


def test_the_seal_refuses_an_entry_changed_after_its_writer_hashed_it(capture, tmp_path):  # noqa: F811
    _root, path, census, identity, acts, hessians, _monolith = capture
    root = tmp_path / 'written'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    entry = root / 'inputs' / 'a.pt'
    observed = entry.stat()
    os.utime(entry, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match='changed since its writer'):
        writer.finish(model_load_contract=identity['model_load_contract'])
    assert not (root / 'capture_manifest.json').exists()


def test_a_traversal_identity_seals_only_with_its_recorded_source(capture, tmp_path):  # noqa: F811
    _root, path, census, identity, acts, hessians, monolith = capture
    traversal = {key: value for key, value in identity.items() if key != 'source_files'}
    root = tmp_path / 'traversal'
    writer = cc.CaptureWriter(root, census_path=path, identity=traversal)
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    with pytest.raises(RuntimeError, match='binds no source files'):
        writer.finish(model_load_contract=identity['model_load_contract'])
    record = writer.finish(model_load_contract=identity['model_load_contract'],
                           source_files=identity['source_files'])
    assert Path(record['path']).read_bytes() == Path(monolith['path']).read_bytes()
    # Downstream readers verify against the sealed manifest, as before.
    cc.require_capture_contract(record['path'], record['sha256'])
    values, _ = cc.prefetch_capture(record['path'], expected_identity=identity, census=census,
                                    names=['a'], device='cpu', expected_sha256=record['sha256'])
    assert torch.equal(values[0]['a'], acts['a'])


# -- admission ------------------------------------------------------------------

def test_the_retained_window_follows_shards_that_interleave_layers():
    """A file read by layers 0 and 3 is held from layer 0 until layer 3 is read.

    GLM-5.3-Flash's shards interleave this way (one is read by layers 4 and
    40), so a window of consecutive layers under-counts what the owner holds.
    """
    from prismaquant.autoscale import retained_source_page_bytes
    layer_files = {0: {'x'}, 1: {'b'}, 2: {'c'}, 3: {'x'}}
    size = {'x': 1000, 'b': 1, 'c': 10}
    # One slot: layer 2 runs holding c and x, which layer 3 still reads. The
    # largest single layer is only x (1000).
    assert retained_source_page_bytes(layer_files, size, 1) == 1010
    # Two slots: layer 1 runs with layer 2 read ahead, holding b, c and x. Two
    # consecutive layers' files are at most x + c (1010).
    assert retained_source_page_bytes(layer_files, size, 2) == 1011
    # Without the bounded policy nothing goes before the owner closes.
    assert retained_source_page_bytes(layer_files, size, 1, released=False) == 1011


def test_admission_reports_the_retained_window_outside_the_plan(tmp_path):
    """Retained pages are clean page cache: reported, never charged (PQ #1896).

    The guard's committed reading omits them and the kernel reclaims them
    before it refuses an allocation, so they are not in ``memory_bytes`` or
    any phase; what they decide is whether the source is read once.
    """
    from prismaquant.autoscale import (retained_source_page_bytes,
                                       streamed_calibration_resources)
    from test_capture_layer_chain_glm import _three_layer_config, _write_sharded_checkpoint
    from test_glm5_next_streamed_forward_parity import _build_model
    source = tmp_path / 'source'
    _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    common = dict(unit_shapes={}, counts={}, nsamples=2, seqlen=257, max_act_rows=7,
                  prefetch_workers=1, headroom_gb=0)
    for policy, released in (('legacy', False), ('shared-inputs-bounded-v1', True)):
        for slots in (2, 3):
            plain = streamed_calibration_resources(source, cache_slots=slots,
                                                   capture_policy=policy, **common)
            plan = streamed_calibration_resources(source, cache_slots=slots,
                capture_policy=policy, source_recording=True, **common)
            assert 'source_retained_page_bytes' not in plain
            layer_files = {int(k): set(v) for k, v in plan['body_source_shards'].items()}
            sizes = {name: (source / name).stat().st_size
                     for files in layer_files.values() for name in files}
            assert plan['source_retained_page_bytes'] == retained_source_page_bytes(
                layer_files, sizes, slots, released=released) > 0
            assert plan['memory_bytes'] == plain['memory_bytes']
            for phase in (plan.get('phases') or {}).values():
                assert 'source_retained_page_bytes' not in phase
            assert 'source_retained_page_bytes' not in (plan.get('terms') or {})


# -- the streamed capture, end to end -------------------------------------------

def _glm_conv_kernel_is_cuda_only():
    """Whether transformers bound GLM's ``causal_conv1d_fn`` to the CUDA-only package.

    ``use_kernel_func_from_hub_with_fallback`` chooses once, when
    ``modeling_glm5_next`` is imported: the ``causal_conv1d`` package's
    ``causal_conv1d_fn`` when the package imports and resolves it, otherwise
    the torch path. The package rejects CPU tensors, whatever
    ``torch.cuda.is_available`` says later (PQ #1939). This mirrors that
    choice; a spec lookup would not, because a package whose extension fails
    to import still has a spec, and transformers falls back to torch for it.
    """
    try:
        getattr(importlib.import_module('causal_conv1d'), 'causal_conv1d_fn')
    except Exception:
        return False
    return True


def _glm_source(tmp_path, monkeypatch, *, device):
    from test_capture_layer_chain_glm import _three_layer_config, _write_sharded_checkpoint
    from test_glm5_next_streamed_forward_parity import _build_model
    from projection_producer_fixture import require_projection_producer
    require_projection_producer(monkeypatch)
    monkeypatch.setenv('PRISMAQUANT_TMPDIR', str(tmp_path / 'staging'))
    if device == 'cpu':
        # The campaign places its model on CUDA whenever CUDA is available.
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(1896)
    source = tmp_path / 'source'
    shards = _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    return source, shards


@pytest.mark.parametrize('policy', ['legacy', 'shared-inputs-bounded-v1'])
def test_a_streamed_capture_reads_each_source_file_once(tmp_path, monkeypatch, policy):
    """The capture with the model on CPU.

    It runs where GLM's convolution takes CPU tensors: a worker whose
    transformers bound the torch path. A GB10 worker binds the CUDA-only
    package, so there this case skips and the CUDA case below runs instead.
    """
    if _glm_conv_kernel_is_cuda_only():
        pytest.skip('transformers bound GLM causal_conv1d_fn to the CUDA-only causal_conv1d '
                    'package at import; this CPU case runs on CPU-only workers (PQ #1939)')
    _reads_each_source_file_once(tmp_path, monkeypatch, policy, device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA; certifies nothing when skipped')
@pytest.mark.parametrize('policy', ['legacy', 'shared-inputs-bounded-v1'])
def test_a_streamed_capture_reads_each_source_file_once_on_cuda(tmp_path, monkeypatch, request, policy):
    """The capture with the model on CUDA, the device a GB10 capture uses.

    A bounded CUDA capture needs its release policy before the process starts,
    so that case runs in a child pytest (#1096); the spies below run in it.
    """
    from test_glm_campaign_streaming import bounded_capture_child_needed, run_in_bounded_capture_child
    if bounded_capture_child_needed(policy, cuda=True, environ=os.environ):
        run_in_bounded_capture_child(request, tmp_path)
        return
    _reads_each_source_file_once(tmp_path, monkeypatch, policy, device='cuda')


def _reads_each_source_file_once(tmp_path, monkeypatch, policy, *, device):
    """The monolith hashes every file exactly once, each through the descriptor it reads.

    The census pass runs first, unobserved. During the capture every whole-file
    hash of the source is recorded with the descriptor it read through, and
    every tensor payload read through the owner with its file. Each roster file
    is hashed once; the files the forward reads are hashed by that read, and the
    rest (the vision tower, the tokenizer assets) once at the seal.
    """
    from prismaquant import tessera_campaign as campaign
    # Controlled legacy-mechanism fixture, not a qualified immutable provider.
    # The independent automatic-admission matrix exercises the real refusal.
    monkeypatch.setattr(cc, 'require_automatic_capture_source_recording', lambda: None)
    source, shards = _glm_source(tmp_path, monkeypatch, device=device)
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1),
              torch.arange(257).flip(0).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens', lambda *_: (tokens, 'tiny GLM frozen draw'))
    census = tmp_path / 'census.json'
    common = ['--model', str(source), '--out', str(tmp_path / 'unused.pkl'),
              '--menu-mode', 'research', '--nsamples', '2', '--seqlen', '257',
              '--max-act-rows', '7', '--attention-implementation', 'eager', '--streaming',
              '--streaming-cache-headroom-gb', '0']
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'census-cache'),
                          '--census-out', str(census)]) == 0
    hashed, payload = [], []
    original_hash, original_payload = cc.sha256, cc._CaptureSourceSafeOpen._payload

    def hash_spy(path, **kwargs):
        if Path(path).resolve().parent == source.resolve():
            hashed.append((Path(path).name, kwargs.get('file_descriptor') is not None))
        return original_hash(path, **kwargs)

    def payload_spy(self):
        payload.append(self.name)
        return original_payload(self)
    monkeypatch.setattr(cc, 'sha256', hash_spy)
    monkeypatch.setattr(cc._CaptureSourceSafeOpen, '_payload', payload_spy)
    out, cache_dir = tmp_path / 'capture', tmp_path / 'capture-cache'
    argv = [*common, '--calibration-census', str(census), '--cache-dir', str(cache_dir),
            '--capture-calibration-out', str(out)]
    if policy != 'legacy':
        argv += ['--streaming-capture-policy', policy]
    assert campaign.main(argv) == 0
    roster = {path.name for path in cc.capture_source_files(source)}
    assert Counter(name for name, _fd in hashed) == Counter(roster)  # once each
    assert all(through_descriptor for _name, through_descriptor in hashed)
    read = set(payload)
    assert {'model-head.safetensors', *(f'model-layer-{i:03d}.safetensors' for i in range(3))} <= read
    assert 'model-visual.safetensors' not in read  # sealed by hash, never read
    manifest = json.loads((out / 'capture_manifest.json').read_text())
    assert manifest['identity']['source_files'] == {name: _digest(source / name) for name in roster}
    receipt = json.loads((cache_dir / 'capture-source-authentication.json').read_text())
    assert receipt['schema'] == cc.RECORDING_RECEIPT_SCHEMA
    assert {row['name'] for row in receipt['verified_files']} >= roster
    if policy == 'shared-inputs-bounded-v1':
        # Each file the forward read was released after its last layer.
        assert {name for name in read if name.endswith('.safetensors')} <= set(receipt['released_files'])
