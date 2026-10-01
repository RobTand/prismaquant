"""A streamed capture reads its source once and its own output never (PQ #1896).

The streamed capture hashed every source file before its forward, then read
the same bytes again for the tensors; a capture chain's prep did the same.
The writer then re-read every entry it had just written to hash it, and the
seal read them a third time. These tests hold the single-pass contract:

* the source is hashed by the read that consumes it, through the descriptor
  the tensors are read through, before the first tensor reaches the capture;
* bytes that change between the hash and a later read refuse;
* a census producer digest is compared at that first use and refuses before
  any tensor is read;
* a chain prep hashes nothing, and its join refuses quanta that recorded
  different digests for one file;
* each capture entry is hashed while it is written and never read back, and
  the seal holds it to the stat fingerprint taken then.
"""
from collections import Counter
import hashlib
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
    from safetensors import safe_open
    source, census = _source(tmp_path)
    shard = source / 'model-00001.safetensors'
    owner = cc.record_capture_source(census, model=source)
    with pytest.raises(RuntimeError, match='changed during consumption'):
        with owner.safe_open(safe_open, shard, framework='pt') as handle:
            handle.get_tensor('w')
            _mutate_preserving_mtime(shard)
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

def test_admission_charges_the_retained_source_window(tmp_path):
    from prismaquant.autoscale import streamed_calibration_resources
    from test_capture_layer_chain_glm import _three_layer_config, _write_sharded_checkpoint
    from test_glm5_next_streamed_forward_parity import _build_model
    source = tmp_path / 'source'
    _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    sizes = [(source / f'model-layer-{layer:03d}.safetensors').stat().st_size for layer in range(3)]
    common = dict(unit_shapes={}, counts={}, nsamples=2, seqlen=257, max_act_rows=7,
                  prefetch_workers=1, headroom_gb=0)
    plain = streamed_calibration_resources(source, cache_slots=2, **common)
    assert 'source_retained_page_bytes' not in plain['terms']
    for slots in (2, 3):
        plan = streamed_calibration_resources(source, cache_slots=slots, source_recording=True,
                                              **common)
        window = max(sum(sizes[first:first + slots]) for first in range(3))
        assert plan['terms']['source_retained_page_bytes'] == window
        assert plan['memory_bytes'] == sum(plan['terms'].values())
    bounded = streamed_calibration_resources(source, cache_slots=2, source_recording=True,
        capture_policy='shared-inputs-bounded-v1', **common)
    window = max(sum(sizes[first:first + 2]) for first in range(3))
    for name in ('source_validation', 'forward', 'materialization'):
        assert bounded['phases'][name]['source_retained_page_bytes'] == window


# -- the streamed capture, end to end -------------------------------------------

def _glm_source(tmp_path, monkeypatch):
    from test_capture_layer_chain_glm import _three_layer_config, _write_sharded_checkpoint
    from test_glm5_next_streamed_forward_parity import _build_model
    pinned = '/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97'
    producer = Path(os.environ.get('TESSERA_REPO') or pinned)
    if not producer.is_dir():
        pytest.skip('TESSERA_REPO must name the pinned producer checkout '
                    f'(unset, and {pinned} is absent)')
    monkeypatch.setenv('TESSERA_REPO', str(producer))
    monkeypatch.setenv('PRISMAQUANT_TMPDIR', str(tmp_path / 'staging'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(1896)
    source = tmp_path / 'source'
    shards = _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    return source, shards


@pytest.mark.parametrize('policy', ['legacy', 'shared-inputs-bounded-v1'])
def test_a_streamed_capture_reads_each_source_file_once(tmp_path, monkeypatch, policy):
    """The monolith hashes every file exactly once, each through the descriptor it reads.

    The census pass runs first, unobserved. During the capture every whole-file
    hash of the source is recorded with the descriptor it read through, and
    every tensor payload read through the owner with its file. Each roster file
    is hashed once; the files the forward reads are hashed by that read, and the
    rest (the vision tower, the tokenizer assets) once at the seal.
    """
    from prismaquant import tessera_campaign as campaign
    source, shards = _glm_source(tmp_path, monkeypatch)
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
