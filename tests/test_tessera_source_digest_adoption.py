"""A retained full hash proof transfers with all original mutation fences."""
import hashlib,json
from pathlib import Path
import pytest
from prismaquant import cost_streaming as cs
from prismaquant.tessera_source_digest_adoption import adopt_source_digests
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
    result=adopt_source_digests(root,binding,out,expected_content_sha256=identity['content_sha256'])
    assert result['shards']==2 and result['fresh_source_payload_reads']==0
    cache=SourceDigestCache(out,source=root)
    for row in identity['shards']:assert cache.sha256(Path(row['path']))==row['sha256']
    receipt=cache.receipt();assert receipt['cached_shards']==2
    assert all(row['writer']['authority']==binding and row['writer']['fresh_payload_read'] is False for row in receipt['shards'])


@pytest.mark.parametrize('change',['proof','source','expected'])
def test_changed_authority_or_source_cannot_publish_a_cache(authority,tmp_path,change):
    root,shards,path,identity,binding=authority;expected=identity['content_sha256']
    if change=='proof':path.write_bytes(path.read_bytes()+b' ')
    elif change=='source':next(iter(shards.values())).write_bytes(b'x'*65536)
    else:expected='0'*64
    out=tmp_path/'refused'
    with pytest.raises((ValueError,RuntimeError)):
        adopt_source_digests(root,binding,out,expected_content_sha256=expected)
    assert not out.exists()


def test_device_only_portability_is_explicit_and_preserved(authority,tmp_path,monkeypatch):
    from tessera.source_digest_cache import SourceDigestCache
    root,shards,path,identity,binding=authority
    document=json.loads(path.read_text())
    for row in document['fingerprints']:row['device']+=123
    path.write_text(json.dumps(document));binding['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='stat drifted'):
        adopt_source_digests(root,binding,tmp_path/'strict',expected_content_sha256=identity['content_sha256'])
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE','1')
    out=tmp_path/'portable';result=adopt_source_digests(root,binding,out,expected_content_sha256=identity['content_sha256'])
    assert result['device_portable_shards']==2
    cache=SourceDigestCache(out,source=root)
    for shard in shards.values():cache.sha256(shard)
    assert all(row['writer']['upstream_dev_portable_device'] is True for row in cache.receipt()['shards'])
