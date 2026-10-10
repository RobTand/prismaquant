"""Prepare full-draw replay slices for GLM layers 5 and 9 (prismaquant#2572).

Metadata only: publisher index and LFS declarations, the qualified
activation contract, plus existing retained Stage B capture metadata,
identify whole files and exact capture coordinates. No weight shard or
retained tensor payload is opened. One PB CPU run emits every slice
binding, staged-readset manifest and prospective resource ledger.

Each slice replays a contiguous 16-sequence range through the staged
``l05_source_replay`` entry point, whose validator admits the emitted
scope. Slice ranges tile batches 0-511 without overlap, so the gate
analysis can sum slice totals against the recorded Stage B unit totals.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers.models.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration

from experiments.alloc_lead_asd.l05_source_replay import (
    AUXILIARY_DIGESTS,
    PUBLISHER_REVISION,
    qualify_historical_activation_policy,
    record,
    replay_slice_scope,
    write_json,
)
from experiments.alloc_lead_asd.qualify_tiny_glm import DERIVATIVE
from prismaquant.glm_source_derivative import bind_source_derivative
from prismaquant.layer_streaming import _model_tensor_dtypes
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

EVIDENCE = Path('/mnt/shared/astra-pq-2010-evidence-20261002/bf16-publisher')
REVISION = PUBLISHER_REVISION
MODEL = Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
CAMPAIGN = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913')
CONTRACT = Path('/mnt/shared/tessera-measurements/pq1962-sol-20261002/historical-l05-backend-contract.json')
CONTRACT_SHA256 = '9c2f7167c65efb26c4b213369cc6e8616cee974891b09d3229f18e9818717d81'
LAYERS = (5, 9)
FACT_EXPERTS = {5: (240,), 9: (131,)}
FACT_ROLES = ('gate_proj', 'up_proj', 'down_proj')
SEQUENCE_COUNT = 512
PROBES = (0, 1, 2, 3)
CAPTURE_SHAPE = [1, 512, 4, 4096]
CAPTURE_DTYPE = 'torch.bfloat16'
LIMITS = dict(cpu_cgroup_bytes=52 << 30, device_envelope_bytes=44 << 30,
              torch_allocator_bytes=40 << 30, native_device_allowance_bytes=4 << 30,
              aggregate_bytes=96 << 30, host_runtime_metadata_bytes=4 << 30,
              host_trace_and_output_bytes=2 << 30)


def load(path, identities):
    raw = path.read_bytes()
    identities.append({'path': str(path), 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw), raw


def publisher_layer(EVIDENCE_root, layer, identities):
    publisher, _ = load(EVIDENCE_root / 'revision.json', identities)
    assert publisher['sha'] == REVISION and publisher['id'] == 'zai-org/GLM-5.3-Flash-BF16'
    proofs, _ = load(EVIDENCE_root / 'publisher-auxiliary-proofs.json', identities)
    index, index_raw = load(EVIDENCE_root / 'publisher-auxiliary/model.safetensors.index.json', identities)
    proof = next(row for row in proofs if row['name'] == 'model.safetensors.index.json')
    assert proof['native_git_object_verified'] and proof['sha256'] == hashlib.sha256(index_raw).hexdigest()
    assert hashlib.sha1(b'blob ' + str(len(index_raw)).encode() + b'\0' + index_raw).hexdigest() == proof['publisher_git_blob_sha1']
    config, _ = load(EVIDENCE_root / 'publisher-auxiliary/config.json', identities)
    assert config['text_config']['layer_types'][layer] == 'linear_attention'
    names = {name: file for name, file in index['weight_map'].items()
             if name.startswith(f'model.language_model.layers.{layer}.')}
    files = sorted(set(names.values()))
    siblings = {row['rfilename']: row for row in publisher['siblings']}
    shards = []
    for name in files:
        row = siblings[name]
        assert row['size'] == row['lfs']['size']
        shards.append({'name': name, 'path': str(MODEL / name),
                       'bytes': row['size'], 'sha256': row['lfs']['sha256']})
    return config, names, shards


def layer_facts(layer, identities):
    slice_path = CAMPAIGN / f'r13-stageb-20260923/meta-045-fae344d/adjoint-slices/layer-{layer:03d}.json'
    handoffs = glob.glob(str(CAMPAIGN / f'r13-stageb-20260923/a4/overlay/layer-quanta/layer-{layer + 1:03d}/handoff/*/handoff.json'))
    assert len(handoffs) == 1
    adjoint, _ = load(slice_path, identities)
    handoff, _ = load(Path(handoffs[0]), identities)
    boundaries = adjoint['boundary_entries'][str(layer)]
    assert len(boundaries) == SEQUENCE_COUNT
    assert sorted(row['metadata']['identity']['coordinates']['batch'] for row in boundaries) == list(range(SEQUENCE_COUNT))
    cotangents = handoff['activation_entries']
    assert len(cotangents) == SEQUENCE_COUNT * len(PROBES)
    assert {(row['metadata']['identity']['coordinates']['batch'],
             row['metadata']['identity']['coordinates']['probe']) for row in cotangents} == {
                 (batch, probe) for batch in range(SEQUENCE_COUNT) for probe in PROBES}
    for row in boundaries + cotangents:
        assert row['shape'] == CAPTURE_SHAPE and row['dtype'] == CAPTURE_DTYPE
    for row in boundaries:
        assert row['metadata']['identity']['coordinates']['probe'] is None
        assert row['metadata']['identity']['coordinates']['boundary'] == layer
    return boundaries, cotangents


def qualify_layer(config, layer):
    profile = Glm5NextProfile()
    config = Glm5NextConfig.from_dict(dict(config))
    config._attn_implementation = config.text_config._attn_implementation = 'eager'
    config._experts_implementation = config.text_config._experts_implementation = 'grouped_mm'
    with torch.device('meta'):
        model = Glm5NextForConditionalGeneration(config)
    derivative = bind_source_derivative(model, profile, DERIVATIVE)
    dtypes = _model_tensor_dtypes(model, torch.bfloat16)
    prefix = f'model.language_model.layers.{layer}.'
    rows = []
    for kind, tensors in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in tensors:
            if name.startswith(prefix):
                assert value.is_meta
                dtype = dtypes.get(name, torch.bfloat16 if value.is_floating_point() else value.dtype)
                rows.append({'kind': kind, 'name': name, 'shape': list(value.shape),
                             'dtype': str(dtype), 'bytes': value.numel() * torch.empty((), dtype=dtype).element_size()})
    assert rows
    assert not torch.cuda.is_initialized()
    resident = sum(row['bytes'] for row in rows)
    packed = sum(row['bytes'] for row in rows if '.mlp.experts.' in row['name'])
    largest = max(row['bytes'] // (row['shape'][0] * (2 if row['name'].endswith('gate_up_proj') else 1))
                  if '.mlp.experts.' in row['name'] else row['bytes'] for row in rows)
    keys = sorted({row['name'] for row in rows})
    return rows, keys, resident, packed, largest, derivative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', default='5,9')
    parser.add_argument('--slice-seqs', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    layers = tuple(sorted({int(part) for part in args.layers.split(',')}))
    assert set(layers) <= set(LAYERS) and layers
    assert SEQUENCE_COUNT % args.slice_seqs == 0
    torch.set_num_threads(1)
    contract_raw = CONTRACT.read_bytes()
    assert hashlib.sha256(contract_raw).hexdigest() == CONTRACT_SHA256
    contract = json.loads(contract_raw)
    activation_policy = qualify_historical_activation_policy(contract)
    binary = contract['projection_config']['binary']
    output = args.output_dir
    if output.exists():
        raise RuntimeError('output directory must be unused')
    output.mkdir(parents=True)
    index = []
    for layer in layers:
        identities: list = []
        config, names, shards = publisher_layer(EVIDENCE, layer, identities)
        boundaries, cotangents = layer_facts(layer, identities)
        rows, keys, resident, packed, largest, derivative = qualify_layer(config, layer)
        profile = Glm5NextProfile()
        mapped = sorted({profile.checkpoint_to_live_name(name, multimodal=True) for name in names})
        assert len(mapped) == len(names) and mapped
        assert all(name.startswith(f'model.language_model.layers.{layer}.') for name in mapped)
        model = Path(shards[0]['path']).parent
        metadata = []
        for name in ('config.json', 'model.safetensors.index.json'):
            row = next(row for row in identities if Path(row['path']).name == name)
            assert row['sha256'] == AUXILIARY_DIGESTS[name]
            metadata.append(dict(path=row['path'], name=name, bytes=row['bytes'], sha256=row['sha256'],
                                 logical_path=str(model / name)))
        fact_units = [f'model.language_model.layers.{layer}.mlp.experts.{expert}.{role}'
                      for expert in FACT_EXPERTS[layer] for role in FACT_ROLES]
        units = {name.removesuffix('.weight') for name in mapped if name.endswith('.weight')}
        assert all(unit in units for unit in fact_units)
        by_batch = {}
        for row in boundaries:
            by_batch.setdefault(row['metadata']['identity']['coordinates']['batch'], row)
        cot_grid = {}
        for row in cotangents:
            coords = row['metadata']['identity']['coordinates']
            cot_grid.setdefault((coords['probe'], coords['batch']), row)
        for start in range(0, SEQUENCE_COUNT, args.slice_seqs):
            stop = start + args.slice_seqs
            sequences = list(range(start, stop))
            group = output / f'l{layer:02d}-s{start:03d}-{stop - 1:03d}'
            group.mkdir()
            slice_boundaries = [by_batch[batch] for batch in sequences]
            slice_cotangents = [cot_grid[probe, batch] for probe in PROBES for batch in sequences]
            binding = dict(scope=replay_slice_scope(layer, sequences), model=str(model), layer=layer,
                publisher_revision=REVISION, sequences=sequences, sequence_length=512,
                n_probes=len(PROBES), global_token_count=262144, probe_seed_base=7000, temperature=1.0,
                loss_positions='all', cotangents_are_banked_fixed_inputs=True,
                activation_format='TESSERA_E4M3_K1_R1024', dtype='torch.bfloat16', layer_keys=mapped,
                fact_units=fact_units, activation_policy=activation_policy,
                historical_probe_identity=contract['probe_identity'],
                layer_cache_bytes=resident, metadata=metadata, shards=shards,
                boundaries=slice_boundaries, cotangents=slice_cotangents, allocator_limits=LIMITS,
                trace_host_reserve_bytes=LIMITS['host_trace_and_output_bytes'],
                comparison_staging_bound_bytes=2 * largest,
                projection_contract=dict(path=str(CONTRACT), sha256=CONTRACT_SHA256),
                derivative=derivative,
                limitations=['full-draw replay slice: local fixed-cotangent mechanism only',
                             'slice sums gate against the recorded Stage B unit totals in analysis',
                             'does not qualify legacy full-model cotangents'])
            write_json(group / 'binding.json', binding)
            entries = [dict(path=str(group / 'binding.json'), offset=0,
                            **{key: value for key, value in record(group / 'binding.json').items()
                               if key in ('bytes', 'sha256')})]
            for row in metadata + shards:
                entries.append(dict(path=row['path'], offset=0, bytes=row['bytes'], sha256=row['sha256']))
            for row in slice_boundaries + slice_cotangents:
                entries.append(dict(path=row['path'], offset=0, bytes=row['file_bytes'], sha256=row['sha256']))
            entries.append(dict(path=str(CONTRACT), offset=0, bytes=len(contract_raw), sha256=CONTRACT_SHA256))
            entries.append(dict(path=binary['path'], offset=0, bytes=1523600, sha256=binary['sha256']))
            if len({row['path'] for row in entries}) != len(entries):
                raise RuntimeError('duplicate slice readset entries')
            total = sum(row['bytes'] for row in entries)
            manifest = dict(schema='prismaquant.prismabuild.data_manifest.v1', mount_prefix='/mnt/shared',
                produced_by=dict(tool='pq2572-layer-slice-builder', source_commit=os.environ['PRISMAQUANT_IDENTITY_GIT_COMMIT'],
                                 size_source='independently_bound_publisher_and_retained_entry_metadata'),
                entry_count=len(entries), total_bytes=total,
                annotations=dict(scope='slice readset: exact source/capture/code bytes before replay',
                                 allowed_tiers='ram,ssd',
                                 phases=[dict(name='startup', bytes=total, cumulative_bytes=total,
                                              note='owned exact slice source/capture/code readset before local control')]),
                entries=entries)
            write_json(group / 'data-manifest.json', manifest)
            host = dict(source_client_file_cache=sum(row['bytes'] for row in shards),
                        held_sealed_source=sum(row['bytes'] for row in shards),
                        capture_client_file_cache=sum(row['file_bytes'] for row in slice_boundaries + slice_cotangents),
                        runtime_and_metadata=LIMITS['host_runtime_metadata_bytes'],
                        trace_and_outputs=LIMITS['host_trace_and_output_bytes'])
            load_phase = dict(pin_and_conversion=2 * largest)
            compare = dict(post_fence_source_conversion_and_d2h=2 * largest)
            graph = dict(decoded_host_captures=sum(row['tensor_bytes'] for row in slice_boundaries + slice_cotangents),
                         one_host_concat_or_sealed_decode_bound=max(2, len(sequences)) *
                            max(row['tensor_bytes'] for row in slice_boundaries + slice_cotangents))
            host_peak = sum(host.values()) + max(sum(load_phase.values()), sum(compare.values()), sum(graph.values()))
            write_json(group / 'resource-ledger.json', dict(
                scope='prospective enforced phase envelopes; no actual-width peak measured',
                common_host=host, host_phases=dict(load=load_phase, source_comparison=compare, graph=graph),
                host_component_peak_bytes=host_peak, host_cgroup_bytes=LIMITS['cpu_cgroup_bytes'],
                full_resident_cuda_bytes=resident, extra_packed_source_retention_cuda_bytes=packed,
                load_cuda_component_bound=resident + packed,
                torch_allocator_hard_cap_bytes=LIMITS['torch_allocator_bytes'],
                prospective_native_device_allowance_bytes=LIMITS['native_device_allowance_bytes'],
                device_shared_subset_bytes=LIMITS['device_envelope_bytes'], aggregate_bytes=LIMITS['aggregate_bytes'],
                comparison_staging_bytes=2 * largest,
                unknowns=['actual-width graph/QDQ/native/profile peaks; reject at enforced caps and preserve partial evidence'],
                binding=record(group / 'binding.json'), manifest=record(group / 'data-manifest.json')))
            index.append(dict(layer=layer, sequences=sequences, directory=str(group),
                              binding=record(group / 'binding.json'), manifest=record(group / 'data-manifest.json')))
    write_json(output / 'slices-index.json', dict(
        scope='pq2572 full-draw replay slices for layers 5 and 9',
        publisher_revision=REVISION, contract=dict(path=str(CONTRACT), sha256=CONTRACT_SHA256),
        slice_seqs=args.slice_seqs, slices=index))
    print(json.dumps(dict(slices=len(index), layers=layers,
                          index=record(output / 'slices-index.json')), indent=2))


if __name__ == '__main__':
    main()
