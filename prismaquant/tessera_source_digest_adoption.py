"""Carry a validated original source hash proof into Tessera's existing cache.

No weight hash is recomputed or relabelled as fresh. The owning PrismaQuant
validator checks the complete source/config/fences first; Tessera's own
``SourceDigestCache.adopt`` then re-takes each shard's fingerprint, applies its
quiescence rule and publishes the same digest under its mutation-sensitive key,
recording this owner as the writer.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def adopt_source_digests(source, binding, output, *, expected_content_sha256,
                         quiescent_seconds=None):
    """Seed Tessera's source digest cache from a validated retained hash proof.

    ``quiescent_seconds`` is passed to ``SourceDigestCache``; ``None`` keeps
    Tessera's default. A shard changed more recently than that is refused by
    ``adopt`` exactly as a fresh read would not be recorded.
    """
    from . import cost_streaming as owner
    from .tessera_joint_allocation import _read_bound, _bound_stat_fence
    from .cost_stage_checkpoint import publish_new_bytes
    from tessera.source_digest_cache import SourceDigestCache
    source=Path(source).resolve();path=Path(binding['path'])
    before=_bound_stat_fence(path);raw=_read_bound(binding,'retained source hash proof');document=json.loads(raw)
    identity=owner.validate_cached_streamed_model_identity(source,path,require_complete_checkpoint=True)
    if (document['identity']!=identity or _bound_stat_fence(path)!=before
            or identity['content_sha256']!=expected_content_sha256):
        raise ValueError('retained source proof or expected source identity differs')
    old={str(Path(row['path']).resolve()):row for row in document['fingerprints']}
    observed=[]
    # Reopen/fstat through the cache's owning contract. Validate the full set
    # before publishing any imported metadata; no partial roster is authority.
    for row in identity['shards']:
        shard=Path(row['path']);fp=SourceDigestCache.fingerprint(shard)
        live={'path':str(shard.resolve()),'device':fp['_dev'],'inode':fp['ino'],
              'size':fp['size'],'mtime_ns':fp['mtime_ns'],'ctime_ns':fp['ctime_ns']}
        prior=old[str(shard.resolve())]
        reuse=owner.stat_fingerprint_reuse(live,prior)
        if reuse is None:
            raise ValueError('retained source fence changed before adoption: '+str(shard))
        observed.append((shard,row['sha256'],fp,prior,reuse!='exact'))
    out=Path(output);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    cache=(SourceDigestCache(out,source=source) if quiescent_seconds is None
           else SourceDigestCache(out,source=source,quiescent_seconds=quiescent_seconds))
    for shard,digest,fp,prior,portable in observed:
        # adopt re-takes the fingerprint and refuses one that differs from the
        # fence taken above, so a shard changed since then publishes nothing.
        cache.adopt(shard,digest,fingerprint=fp,writer={
            'kind':'adopted_verified_prismaquant_streamed_identity',
            'authority':dict(binding),'source_content_sha256':identity['content_sha256'],
            'upstream_fingerprint':prior,'upstream_dev_portable_device':portable,
            'fresh_payload_read':False})
    for shard,_,fp,_,_ in observed:
        if SourceDigestCache.fingerprint(shard)!=fp:
            raise ValueError('source fence changed before adoption completed: '+str(shard))
    if _bound_stat_fence(path)!=before:raise ValueError('retained proof changed during adoption')
    fd=os.open(out,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    result={'schema':'prismaquant.tessera_source_digest_adoption.v1','status':'adopted_existing_hashes',
        'authority':dict(binding),'source':str(source),'source_content_sha256':identity['content_sha256'],
        'cache_directory':str(out.resolve()),'shards':len(observed),'source_bytes':sum(fp['size'] for _,_,fp,_,_ in observed),
        'device_portable_shards':sum(portable for *_,portable in observed),
        'fresh_source_payload_reads':0,'writer_kind':'adopted_verified_prismaquant_streamed_identity'}
    receipt=out/'adoption-receipt.json';encoded=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    if not publish_new_bytes(receipt,encoded) and receipt.read_bytes()!=encoded:
        raise ValueError('existing adoption receipt differs')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--source-cache',required=True)
    parser.add_argument('--source-cache-sha256',required=True);parser.add_argument('--expected-content-sha256',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    result=adopt_source_digests(args.model,{'path':args.source_cache,'sha256':args.source_cache_sha256},args.out,
                              expected_content_sha256=args.expected_content_sha256)
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
