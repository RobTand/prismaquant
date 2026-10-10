"""Prepare a metadata-only L05 readset proposal; never read original weights.

Publisher index and LFS declarations, plus existing retained capture metadata,
are enough to identify whole files and exact capture coordinates. This does
not validate original delivery or admit a GPU consumer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EVIDENCE = Path('/mnt/shared/astra-pq-2010-evidence-20261002/bf16-publisher')
REVISION = 'a6c167b62691b2bac901344b65cb651a70f53e43'
MODEL = Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
CAMPAIGN = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913')


def load(path, identities):
    raw = path.read_bytes()
    identities.append({'path': str(path), 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identities = []
    publisher, _ = load(EVIDENCE / 'revision.json', identities)
    assert publisher['sha'] == REVISION and publisher['id'] == 'zai-org/GLM-5.3-Flash-BF16'
    proofs, _ = load(EVIDENCE / 'publisher-auxiliary-proofs.json', identities)
    index, index_raw = load(EVIDENCE / 'publisher-auxiliary/model.safetensors.index.json', identities)
    proof = next(row for row in proofs if row['name'] == 'model.safetensors.index.json')
    assert proof['native_git_object_verified'] and proof['sha256'] == hashlib.sha256(index_raw).hexdigest()
    assert hashlib.sha1(b'blob ' + str(len(index_raw)).encode() + b'\0' + index_raw).hexdigest() == proof['publisher_git_blob_sha1']
    config, _ = load(EVIDENCE / 'publisher-auxiliary/config.json', identities)
    assert config['text_config']['layer_types'][5] == 'linear_attention'
    names = {name: file for name, file in index['weight_map'].items()
             if name.startswith('model.language_model.layers.5.')}
    files = sorted(set(names.values()))
    siblings = {row['rfilename']: row for row in publisher['siblings']}
    shards = []
    for name in files:
        row = siblings[name]
        assert row['size'] == row['lfs']['size']
        shards.append({'name': name, 'path': str(MODEL / name),
                       'bytes': row['size'], 'sha256': row['lfs']['sha256']})
    slice_path = CAMPAIGN / 'r13-stageb-20260923/meta-045-fae344d/adjoint-slices/layer-005.json'
    handoff_path = CAMPAIGN / ('r13-stageb-20260923/a4/overlay/layer-quanta/layer-006/'
                              'handoff/31f5259905fe4c35b91e678809f967f2/handoff.json')
    adjoint, _ = load(slice_path, identities)
    handoff, _ = load(handoff_path, identities)
    boundaries = [row for row in adjoint['boundary_entries']['5']
                  if row['metadata']['identity']['coordinates']['batch'] in range(4)]
    cotangents = [row for row in handoff['activation_entries']
                 if row['metadata']['identity']['coordinates']['batch'] in range(4)]
    assert len(boundaries) == 4 and len(cotangents) == 16
    assert {(row['metadata']['identity']['coordinates']['batch'],
             row['metadata']['identity']['coordinates']['probe']) for row in cotangents} == {
                 (batch, probe) for batch in range(4) for probe in range(4)}
    for row in boundaries + cotangents:
        assert row['shape'] == [1, 512, 4, 4096] and row['dtype'] == 'torch.bfloat16'
    result = {'schema': 'prismaquant.research.l05_protocol_proposal.v1',
              'status': 'metadata-only; original-weight reads await Astra preflight and actual GPU lifetime gate',
              'inputs_read': identities, 'publisher_revision': REVISION,
              'low_level_source': '27b7cc60641fb901d0482de47ee7b0e02c94495b',
              'reviewed_delivery_source': '6385aeb162d33841f8b8ad4a31c9c6fdc787fd4a',
              'layer': 5, 'source_layer_tensor_count': len(names),
              'source_layer_weight_map': names, 'source_shards': shards,
              'all_held_shard_bytes': sum(row['bytes'] for row in shards),
              'largest_shard_bytes': max(row['bytes'] for row in shards),
              'boundary_entries': boundaries, 'cotangent_entries': cotangents,
              'serialized_capture_file_bytes': sum(row['file_bytes'] for row in boundaries + cotangents),
              'decoded_capture_tensor_bytes': sum(row['tensor_bytes'] for row in boundaries + cotangents),
              'calibration_rows': [0, 1, 2, 3], 'probe_count': 4, 'global_token_count': 262144,
              'scope': 'local actual-width row/QDQ/contraction discriminator, not draw totals or full-model cotangent validation',
              'lifetime_gate': 'Reuse #1934 actual asynchronous copy/native-alias completion control; never the CPU-only original-material owner',
              'missing_before_execution': ['#1934 accepted GPU lifetime receipt',
                                           'exact layer consumer allocation/cleanup protocol and aggregate peak reservation',
                                           'full CPU entrypoint qualification using synthetic inputs',
                                           'Astra final preflight', 'sealed PB action and staged leases']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: exact publisher-index/LFS and stored first-group coordinates; no original weights read')
    print(f'Whole staged source shard bytes: {result["all_held_shard_bytes"]}; capture files: {result["serialized_capture_file_bytes"]}')


if __name__ == '__main__':
    main()
