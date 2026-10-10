"""Prepare the bounded original L05 protocol from qualified metadata only.

Never opens a weight shard or a retained tensor. The produced full readset is
a proposal for Astra review, not authorization to stage or execute it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.alloc_lead_asd.l05_source_replay import qualify_historical_activation_policy, record, write_json
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.residency_map import bind_residency_manifest
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_whole_file


def load(path, digest):
    return json.loads(read_staged_whole_file(path, digest, label='original proposal metadata'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposal', 'allocation', 'proofs', 'projection-contract'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--data-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    proposal = load(args.proposal, args.proposal_sha256)
    allocation = load(args.allocation, args.allocation_sha256)
    proofs = load(args.proofs, args.proofs_sha256)
    contract_raw = read_staged_whole_file(args.projection_contract, args.projection_contract_sha256,
                                         label='historical projection metadata')
    contract = json.loads(contract_raw)
    activation_policy = qualify_historical_activation_policy(contract)
    if (proposal['publisher_revision'] != 'a6c167b62691b2bac901344b65cb651a70f53e43'
            or proposal['layer'] != 5 or proposal['source_layer_tensor_count'] != 892
            or proposal['probe_count'] != 4 or proposal['calibration_rows'] != [0, 1, 2, 3]
            or proposal['global_token_count'] != 262144
            or allocation['protocol_sha256'] != args.proposal_sha256):
        raise RuntimeError('proposal or qualified allocation differs from the bounded historical contract')
    model = Path(proposal['source_shards'][0]['path']).parent
    metadata = []
    for name in ('config.json', 'model.safetensors.index.json'):
        row = next(row for row in proposal['inputs_read'] if Path(row['path']).name == name)
        proof = next(row for row in proofs if row['name'] == name)
        raw = read_staged_whole_file(Path(row['path']), row['sha256'], label='publisher auxiliary ' + name)
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if (not proof['native_git_object_verified'] or proof['sha256'] != row['sha256']
                or proof['publisher_git_blob_sha1'] != git_blob):
            raise RuntimeError('independently bound publisher auxiliary proof differs')
        metadata.append(dict(path=row['path'], name=name, bytes=row['bytes'], sha256=row['sha256'],
                             logical_path=str(model / name), publisher_git_blob=git_blob))
    profile = Glm5NextProfile()
    keys = [profile.checkpoint_to_live_name(name, multimodal=True)
            for name in proposal['source_layer_weight_map']]
    if len(set(keys)) != 892 or any(not name or not name.startswith('model.language_model.layers.5.') for name in keys):
        raise RuntimeError('publisher keys do not map to the exact isolated body selection')
    boundaries = sorted(proposal['boundary_entries'], key=lambda row: row['metadata']['identity']['coordinates']['batch'])
    cotangents = sorted(proposal['cotangent_entries'], key=lambda row: (
        row['metadata']['identity']['coordinates']['probe'], row['metadata']['identity']['coordinates']['batch']))
    limits = dict(cpu_cgroup_bytes=52 << 30, device_envelope_bytes=44 << 30,
                  torch_allocator_bytes=40 << 30, native_device_allowance_bytes=4 << 30,
                  aggregate_bytes=96 << 30, host_runtime_metadata_bytes=4 << 30,
                  host_trace_and_output_bytes=2 << 30)
    largest = max(row['bytes'] // (row['shape'][0] * (2 if row['name'].endswith('gate_up_proj') else 1))
                  if '.mlp.experts.' in row['name'] else row['bytes'] for row in allocation['live_tensors'])
    binding = dict(scope='original L05 local mechanism control', model=str(model), layer=5,
        publisher_revision=proposal['publisher_revision'], sequences=[0, 1, 2, 3], sequence_length=512,
        n_probes=4, global_token_count=262144, probe_seed_base=7000, temperature=1.0,
        loss_positions='all', cotangents_are_banked_fixed_inputs=True,
        activation_format='TESSERA_E4M3_K1_R1024', dtype='torch.bfloat16', layer_keys=sorted(keys),
        activation_policy=activation_policy, historical_probe_identity=contract['probe_identity'],
        layer_cache_bytes=allocation['resident_layer_bytes'], metadata=metadata, shards=proposal['source_shards'],
        boundaries=boundaries, cotangents=cotangents, allocator_limits=limits,
        trace_host_reserve_bytes=limits['host_trace_and_output_bytes'],
        comparison_staging_bound_bytes=2 * largest,
        projection_contract=dict(path=str(args.projection_contract), sha256=args.projection_contract_sha256),
        limitations=['local fixed-cotangent mechanism only', 'not matched full calibration pricing',
                     'does not qualify legacy full-model cotangents', 'explicit historical DEV projection fallback'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'binding.json', binding)
    entries = [dict(path=str(args.output / 'binding.json'), offset=0, **{key: value for key, value in
               record(args.output / 'binding.json').items() if key in ('bytes', 'sha256')})]
    for row in metadata + proposal['source_shards']:
        entries.append(dict(path=row['path'], offset=0, bytes=row['bytes'], sha256=row['sha256']))
    for row in boundaries + cotangents:
        entries.append(dict(path=row['path'], offset=0, bytes=row['file_bytes'], sha256=row['sha256']))
    entries.append(dict(path=str(args.projection_contract), offset=0,
                        bytes=len(contract_raw), sha256=args.projection_contract_sha256))
    binary = contract['projection_config']['binary']
    entries.append(dict(path=binary['path'], offset=0, bytes=1523600, sha256=binary['sha256']))
    if len({row['path'] for row in entries}) != len(entries):
        raise RuntimeError('duplicate full original readset entries')
    total = sum(row['bytes'] for row in entries)
    manifest = dict(schema='prismaquant.prismabuild.data_manifest.v1', mount_prefix='/mnt/shared',
        produced_by=dict(tool='pq1962-original-protocol-metadata-only',
                         source_commit=os.environ['PRISMAQUANT_IDENTITY_GIT_COMMIT'],
                         size_source='independently_bound_publisher_and_retained_entry_metadata'),
        entry_count=len(entries), total_bytes=total,
        annotations=dict(scope='PROPOSAL ONLY; no original payload was read or staged', allowed_tiers='ram,ssd',
                         phases=[dict(name='startup', bytes=total, cumulative_bytes=total,
                                      note='owned exact original source/capture/code readset before local control')]), entries=entries)
    write_json(args.output / 'data-manifest.json', manifest)
    # The prospective hard envelope replaces the unproven8 GiB graph guess.
    host = dict(source_client_file_cache=proposal['all_held_shard_bytes'],
                held_sealed_source=proposal['all_held_shard_bytes'],
                capture_client_file_cache=proposal['serialized_capture_file_bytes'],
                runtime_and_metadata=limits['host_runtime_metadata_bytes'],
                trace_and_outputs=limits['host_trace_and_output_bytes'])
    load_phase = dict(pin_and_conversion=2 * largest)
    compare = dict(post_fence_source_conversion_and_d2h=2 * largest)
    graph = dict(decoded_host_captures=proposal['decoded_capture_tensor_bytes'],
                 one_host_concat_or_sealed_decode_bound=max(2, len(binding['sequences'])) *
                    max(row['tensor_bytes'] for row in boundaries + cotangents))
    host_peak = sum(host.values()) + max(sum(load_phase.values()), sum(compare.values()), sum(graph.values()))
    write_json(args.output / 'resource-ledger.json', dict(scope='prospective enforced phase envelopes; no actual-width peak measured',
        common_host=host, host_phases=dict(load=load_phase, source_comparison=compare, graph=graph),
        host_component_peak_bytes=host_peak, host_cgroup_bytes=limits['cpu_cgroup_bytes'],
        full_resident_cuda_bytes=allocation['resident_layer_bytes'],
        extra_packed_source_retention_cuda_bytes=allocation['packed_expert_bytes'],
        load_cuda_component_bound=allocation['resident_layer_bytes'] + allocation['packed_expert_bytes'],
        torch_allocator_hard_cap_bytes=limits['torch_allocator_bytes'],
        prospective_native_device_allowance_bytes=limits['native_device_allowance_bytes'],
        device_shared_subset_bytes=limits['device_envelope_bytes'], aggregate_bytes=limits['aggregate_bytes'],
        comparison_staging_bytes=2 * largest,
        unknowns=['actual-width graph/QDQ/native/profile peaks; reject at enforced caps and preserve partial evidence'],
        binding=record(args.output / 'binding.json'), manifest=record(args.output / 'data-manifest.json')))
    print('PASS: prepared bounded original binding/readset from metadata only; no original source or retained payload read')
    print(json.dumps(dict(binding=record(args.output / 'binding.json'), manifest=record(args.output / 'data-manifest.json'),
                         entries=len(entries), bytes=total, aggregate_gib=96, cpu_cgroup_gib=52, gpu_subset_gib=44)))


if __name__ == '__main__':
    main()
