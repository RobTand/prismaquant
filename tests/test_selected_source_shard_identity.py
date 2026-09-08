"""Selected-source rows byte-verify only the shards they read (#388).

`capture_identity` hashes every source file it globs and returns the digests
inside the identity, so a selected row that never installs most layers still
re-hashed the whole checkpoint. With `verify_shards` set, the row hashes every
small file and only the named shards; every other on-disk shard inherits its
digest from the census-sealed producer roster, and the identity must still equal
the canonical capture's. Whatever cannot be inherited or checked refuses.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest

from prismaquant import tessera_calibration_cache as cc

SHARDS = ('model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
          'model-00003-of-00003.safetensors')
SMALL = ('chat_template.jinja', 'config.json', 'notes.txt', 'tokenizer.json',
         'tokenizer.model')
GLOBBED_SMALL = ('config.json', 'notes.txt', 'tokenizer.json', 'tokenizer.model')
ROSTER_ORIGIN = 'expert_projection.producer.source'


def runtime():
    import torch
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                transformers=importlib.metadata.version('transformers'))


def streaming_contract(*, layers_prefix='model.layers.', num_layers=2,
                       model_class='SyntheticSource'):
    return dict(schema='prismaquant.streaming_initialization.v1',
        scope='streamed_text_source_forward', status='completed',
        transformers_version=importlib.metadata.version('transformers'),
        model_class=model_class, dtype='torch.bfloat16', layers_prefix=layers_prefix,
        num_layers=num_layers, persistent_tensors=1, derived_buffers=0,
        state_sha256='a'*64, source_map_sha256='b'*64)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def producer_roster(source, tensors):
    """The census-sealed `expert_projection.producer.source` for a fixture root."""
    shards = sorted(p.name for p in source.glob('*.safetensors'))
    auxiliary = sorted(p.name for p in source.iterdir() if p.is_file() and
                       not p.name.endswith(('.safetensors', '.bin')) and p.name != 'config.json')
    return dict(files={name: file_sha256(source/name) for name in shards},
                auxiliary_sha256={name: file_sha256(source/name) for name in auxiliary},
                config_sha256=file_sha256(source/'config.json'), tensors=dict(tensors))


class HashLog:
    """Record every `tessera_calibration_cache.sha256` call: (name, bytes)."""

    def __init__(self, monkeypatch):
        self.events = []
        real = cc.sha256
        def wrapped(path, **kwargs):
            digest = real(path, **kwargs)
            self.events.append((Path(path).name, Path(path).stat().st_size))
            return digest
        monkeypatch.setattr(cc, 'sha256', wrapped)

    def names(self):
        return sorted({name for name, _ in self.events})

    def shard_bytes(self):
        return sum(size for name, size in self.events if name.endswith('.safetensors'))


@pytest.fixture
def sharded_source(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{"architectures": ["SyntheticSource"]}')
    (source/'tokenizer.json').write_text('{}')
    (source/'tokenizer.model').write_bytes(b'spm')
    (source/'notes.txt').write_text('notes')
    (source/'chat_template.jinja').write_text('{{ messages }}')  # sealed, outside the glob
    (source/'weights.bin').write_bytes(b'outside the glob and the roster')
    for index, name in enumerate(SHARDS):
        (source/name).write_bytes(bytes([index+1])*(64*(index+1)))
    tensors = {'model.layers.0.proj.weight': SHARDS[0], 'model.layers.1.proj.weight': SHARDS[1],
               'model.embed_tokens.weight': SHARDS[2], 'lm_head.weight': SHARDS[2]}
    census = dict(model=str(source), model_load_contract=streaming_contract(),
        capture_runtime=runtime(), attention_implementation='eager',
        unit_shapes={'model.layers.0.proj': [3, 2], 'model.layers.1.proj': [3, 2]},
        counts={'model.layers.0.proj': 5, 'model.layers.1.proj': 5},
        max_abs={'model.layers.0.proj': 1.0, 'model.layers.1.proj': 1.0},
        expert_projection=dict(producer=dict(source=producer_roster(source, tensors))))
    path = tmp_path/'census.json'
    path.write_text(json.dumps(census))
    return source, path, census


def identity(path, census, **extra):
    return cc.capture_identity(path, calibration={'fit_ids_sha256': 'draw'}, max_act_rows=2,
        model_load_contract=census['model_load_contract'], attention_implementation='eager',
        **extra)


def identity_with_verification(path, census, **extra):
    return cc.capture_identity_with_verification(path, calibration={'fit_ids_sha256': 'draw'},
        max_act_rows=2, model_load_contract=census['model_load_contract'],
        attention_implementation='eager', **extra)


def rewrite_census(path, census, mutate):
    census = json.loads(json.dumps(census))
    mutate(census)
    path.write_text(json.dumps(census))
    return census


def test_unset_verify_shards_hashes_exactly_the_globbed_roster(sharded_source, monkeypatch):
    """Design A: the default path hashes today's file set, no more and no less."""
    source, path, census = sharded_source
    log = HashLog(monkeypatch)
    result = identity(path, census)
    assert log.names() == sorted([*SHARDS, *SMALL, 'census.json'])
    assert set(result['source_files']) == {*SHARDS, *GLOBBED_SMALL}
    assert log.shard_bytes() == sum((source/name).stat().st_size for name in SHARDS)


def test_verify_shards_reads_only_named_shards_and_reproduces_the_canonical_identity(
        sharded_source, monkeypatch):
    source, path, census = sharded_source
    canonical = identity(path, census)
    log = HashLog(monkeypatch)
    result, verification = identity_with_verification(path, census, verify_shards={SHARDS[1]})
    assert result == canonical
    assert log.names() == sorted([SHARDS[1], *SMALL, 'census.json'])
    assert log.shard_bytes() == (source/SHARDS[1]).stat().st_size
    assert verification == dict(byte_verified=[SHARDS[1]],
        inherited_from_census_roster=[SHARDS[0], SHARDS[2]],
        byte_verified_auxiliary=sorted(SMALL), roster_origin=ROSTER_ORIGIN)
    assert identity(path, census, verify_shards=frozenset({SHARDS[1]})) == canonical


def test_empty_verify_shards_hashes_only_small_files_without_refusing(sharded_source, monkeypatch):
    source, path, census = sharded_source
    canonical = identity(path, census)
    log = HashLog(monkeypatch)
    result, verification = identity_with_verification(path, census, verify_shards=frozenset())
    assert result == canonical
    assert log.shard_bytes() == 0
    assert verification['byte_verified'] == []
    assert verification['inherited_from_census_roster'] == list(SHARDS)


def test_verify_shards_refuses_an_unsealed_unread_shard_on_disk(sharded_source):
    source, path, census = sharded_source
    (source/'model-00004-of-00003.safetensors').write_bytes(b'unsealed')
    with pytest.raises(RuntimeError, match='neither read nor sealed.*model-00004'):
        identity_with_verification(path, census, verify_shards={SHARDS[1]})


def test_verify_shards_refuses_a_tampered_read_shard(sharded_source):
    source, path, census = sharded_source
    (source/SHARDS[1]).write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match=f'differs from census producer: {SHARDS[1]}'):
        identity_with_verification(path, census, verify_shards={SHARDS[1]})


def test_verify_shards_refuses_a_read_shard_the_roster_never_sealed(sharded_source):
    source, path, census = sharded_source
    census = rewrite_census(path, census, lambda c: c['expert_projection']['producer']
                            ['source']['files'].pop(SHARDS[1]))
    with pytest.raises(RuntimeError, match=f'absent from the census producer roster.*{SHARDS[1]}'):
        identity_with_verification(path, census, verify_shards={SHARDS[1]})


def test_verify_shards_refuses_a_read_shard_missing_on_disk(sharded_source):
    source, path, census = sharded_source
    with pytest.raises(RuntimeError, match='not on disk.*model-00009'):
        identity_with_verification(path, census,
                                   verify_shards={'model-00009-of-00003.safetensors'})


def test_verify_shards_refuses_without_a_sealed_roster(sharded_source):
    source, path, census = sharded_source
    census = rewrite_census(path, census, lambda c: c.pop('expert_projection'))
    with pytest.raises(RuntimeError, match='producer shard roster'):
        identity_with_verification(path, census, verify_shards={SHARDS[1]})
    # The default path is unchanged by the missing roster: it still hashes everything.
    assert set(identity(path, census)['source_files']) == {*SHARDS, *GLOBBED_SMALL}
