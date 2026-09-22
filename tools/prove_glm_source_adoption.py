"""Real metadata-only adoption/readback, with any weight hash trapped."""
import json
from pathlib import Path
from prismaquant import cost_streaming
from prismaquant.tessera_source_digest_adoption import adopt_source_digests
from tessera import serving_parts
from tessera.source_digest_cache import SourceDigestCache
source=Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
binding={'path':'/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-joint-01/prepare/source-identity.json','sha256':'543afb5077e531adef1c72d6eeaf0b0d66dbd3d4f828a1b9bea0a0657d6a1f08'}
root=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/cpu-export-codex-20260922');out=root/'source-digests-adopted'
original=serving_parts.sha256_file
checks=[]
def no_body(path):
 if str(path).endswith('.safetensors'):raise AssertionError('unexpected source payload hash')
 checks.append(str(path));return original(path)
def no_pq_hash(*a,**kw):raise AssertionError('unexpected PQ source hash')
cost_streaming._file_sha256=no_pq_hash;serving_parts.sha256_file=no_body
result=adopt_source_digests(source,binding,out,expected_content_sha256='3a4ff1472f4a3fd08f55da96add839d25dbbefef0797c576707aca985e48828c')
cache=SourceDigestCache(out,source=source);whole=serving_parts.source_identity(source,digest_cache=cache)
original_identity=json.loads(Path(binding['path']).read_text())['identity']
assert whole['files']=={Path(row['path']).name:row['sha256'] for row in original_identity['shards']}
assert whole['tensors']==original_identity['checkpoint_weight_map']
assert cache.receipt()['cached_shards']==120 and cache.receipt()['hashed_shards']==0
result.update(whole_source_readback_matches=True,source_digest_receipt=cache.receipt(),fresh_nonshard_hashes=checks,source_header_validation='all120headers',full_model_exported=False)
(root/'source-adoption-readback.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_digest_receipt','fresh_nonshard_hashes')},sort_keys=True))
