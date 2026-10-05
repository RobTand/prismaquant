"""A retained full hash proof transfers with all original mutation fences."""
import functools,hashlib,json
from pathlib import Path
import pytest
from prismaquant import cost_streaming as cs
from prismaquant.tessera_source_digest_adoption import adopt_source_digests, main
from test_source_identity_validate_derivation import checkpoint,_build_cache,_llama_config_dict


@pytest.fixture
def authority(checkpoint,monkeypatch):
    root,shards=checkpoint;config=_llama_config_dict();config['_name_or_path']=str(root)
    cache,identity=_build_cache(root,shards,config)
    monkeypatch.setattr(cs,'live_streaming_runner_config',lambda _:config)
    return root,shards,cache,identity,{'path':str(cache),'sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}


def test_adopts_original_hashes_without_payload_reads(authority,tmp_path,monkeypatch):
    from tessera.source_digest_cache import SourceDigestCache
    from tessera import serving_parts
    root,shards,path,identity,binding=authority
    def forbidden(*a,**k):raise AssertionError('adoption or warm use rehashed weight bytes')
    monkeypatch.setattr(cs,'_file_sha256',forbidden);monkeypatch.setattr(serving_parts,'sha256_file',forbidden)
    out=tmp_path/'digests';out.mkdir()
    with pytest.raises(AssertionError,match='rehashed'):
        SourceDigestCache(out,source=root).sha256(next(iter(shards.values())))
    result=adopt_source_digests(root,binding,out,expected_content_sha256=identity['content_sha256'],
                                quiescent_seconds=0)
    assert result['shards']==2 and result['fresh_source_payload_reads']==0
    cache=SourceDigestCache(out,source=root)
    for row in identity['shards']:assert cache.sha256(Path(row['path']))==row['sha256']
    receipt=cache.receipt();assert receipt['cached_shards']==2
    assert all(row['writer']['authority']==binding and row['writer']['fresh_payload_read'] is False for row in receipt['shards'])
    # Written through Tessera's SourceDigestCache.adopt, which stamps its own
    # adoption record (host, pid, device, quiescence) beside this owner's writer.
    assert all(row['writer']['adopted']['quiescent_seconds']==0 and row['writer']['adopted']['pid']
               for row in receipt['shards'])


def test_a_shard_changed_inside_the_quiescence_window_is_not_adopted(authority,tmp_path):
    root,shards,path,identity,binding=authority
    with pytest.raises(ValueError,match='not quiescent'):
        adopt_source_digests(root,binding,tmp_path/'fresh',expected_content_sha256=identity['content_sha256'],
                             quiescent_seconds=3600)
    assert not any((tmp_path/'fresh').glob('*.json'))


@pytest.mark.parametrize('change',['proof','source','expected'])
def test_changed_authority_or_source_cannot_publish_a_cache(authority,tmp_path,change):
    root,shards,path,identity,binding=authority;expected=identity['content_sha256']
    if change=='proof':path.write_bytes(path.read_bytes()+b' ')
    elif change=='source':next(iter(shards.values())).write_bytes(b'x'*65536)
    else:expected='0'*64
    out=tmp_path/'refused'
    with pytest.raises((ValueError,RuntimeError)):
        adopt_source_digests(root,binding,out,expected_content_sha256=expected,quiescent_seconds=0)
    assert not out.exists()


def test_device_only_portability_is_explicit_and_preserved(authority,tmp_path,monkeypatch):
    from tessera.source_digest_cache import SourceDigestCache
    root,shards,path,identity,binding=authority
    document=json.loads(path.read_text())
    for row in document['fingerprints']:row['device']+=123
    path.write_text(json.dumps(document));binding['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='stat drifted'):
        adopt_source_digests(root,binding,tmp_path/'strict',expected_content_sha256=identity['content_sha256'],
                             quiescent_seconds=0)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE','1')
    out=tmp_path/'portable';result=adopt_source_digests(root,binding,out,expected_content_sha256=identity['content_sha256'],
                                                        quiescent_seconds=0)
    assert result['device_portable_shards']==2
    cache=SourceDigestCache(out,source=root)
    for shard in shards.values():cache.sha256(shard)
    assert all(row['writer']['upstream_dev_portable_device'] is True for row in cache.receipt()['shards'])


def test_main_prints_the_published_adoption_receipt(authority,tmp_path,capsys,monkeypatch):
    """The CLI prints exactly the receipt it published, from the real path.

    Runs tool ``main`` end to end over the existing authority fixture: the
    real adoption, the real validator and Tessera's real byte writes. The
    only substitution is timing (functools.partial of the real
    SourceDigestCache with quiescent_seconds=0, plus its original fingerprint
    staticmethod, in tessera.source_digest_cache), matching the fixture's
    explicit zero-quiescence setting -- no fake cache, no mocked
    adopt_source_digests, no mocked receipt.
    """
    from tessera import source_digest_cache as tsdc
    real=tsdc.SourceDigestCache
    zero_quiescence=functools.partial(real,quiescent_seconds=0)
    zero_quiescence.fingerprint=real.fingerprint
    monkeypatch.setattr(tsdc,'SourceDigestCache',zero_quiescence)
    root,shards,path,identity,binding=authority
    out=tmp_path/'cli'
    argv=['--model',str(root),'--source-cache',str(path),'--source-cache-sha256',binding['sha256'],
          '--expected-content-sha256',identity['content_sha256'],'--out',str(out)]
    assert main(argv)==0
    stdout=json.loads(capsys.readouterr().out)
    receipt=json.loads((out/'adoption-receipt.json').read_text())
    assert stdout==receipt
    assert receipt['shards']==len(identity['shards'])
    # The lazy cache's receipt() reports the keys this instance served, not
    # every disk entry (same contract the first test in this file follows):
    # warm each identity shard through cache.sha256 before reading it.
    cache=real(out,source=root)
    for row in identity['shards']:assert cache.sha256(Path(row['path']))==row['sha256']
    assert cache.receipt()['cached_shards']==len(identity['shards'])
