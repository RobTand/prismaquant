"""Metadata-only admission arithmetic for the selected-source authentication A/B.

Run through PrismaBuild. This reads source headers, never original payloads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
SCHEMA = 'prismaquant.selected_source_authentication_ab.resources.v1'


def sealed_json(binding):
    raw = Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('sealed JSON input changed')
    return json.loads(raw)


def resource_plan(base, shapes, *, full_source_bytes, source_file_count):
    if base.get('schema') != 'prismaquant.selected_anchor_resources.v2':
        raise ValueError('requires the existing selected source resource owner')
    prep = dict(base['phases']['source_preparation'])
    if prep['declared_headroom_bytes'] != 24*GIB:
        raise ValueError('the source comparison preserves 24 GiB loader headroom')
    weight = base['selected_source_weight_bytes']
    widest = max(math.prod(shape)*4 for shape in shapes.values())
    residue, profile, hash_buffer, margin = 4*GIB, 2*GIB, 16*MIB, 2*GIB
    common = dict(reference_gpu_bytes=weight, profiler_memory_bytes=profile,
                  legacy_source_page_residue_bytes=residue, hash_buffer_bytes=hash_buffer)
    physical = dict(source_preparation=dict(prep, **common),
        legacy_full_authentication=dict(
            nonbody_source_bytes=prep['nonbody_source_bytes'],
            declared_headroom_bytes=prep['declared_headroom_bytes'],
            reference_gpu_bytes=weight, profiler_memory_bytes=profile,
            # Hash may transiently retain more pages than it may carry into
            # the next phase. Both bounds are checked against memory.stat.
            absolute_file_charge_bytes=8*GIB, hash_buffer_bytes=hash_buffer),
        projection_and_parity=dict(
            selected_source_weight_bytes=weight, reference_gpu_bytes=weight,
            declared_headroom_bytes=prep['declared_headroom_bytes'],
            legacy_source_page_residue_bytes=residue, profiler_memory_bytes=profile,
            source_and_cpu_comparison_bytes=2*widest, hash_buffer_bytes=hash_buffer))
    gpu = dict(source_preparation={key:prep[key] for key in (
        'selected_source_weight_bytes','nonbody_source_bytes','source_window_bytes',
        'loader_transient_bytes')})
    gpu['source_preparation'].update(reference_gpu_bytes=weight, device_headroom_bytes=8*GIB)
    gpu['legacy_full_authentication']=dict(nonbody_source_bytes=prep['nonbody_source_bytes'],
        reference_gpu_bytes=weight, device_headroom_bytes=8*GIB)
    gpu['projection_and_parity']=dict(selected_source_weight_bytes=weight,
        reference_gpu_bytes=weight, device_headroom_bytes=8*GIB)
    envelopes={phase:sum(terms.values())+sum(gpu[phase].values())+margin
               for phase,terms in physical.items()}
    maximum=max(envelopes.values())
    unconditional=dict(physical['legacy_full_authentication'])
    unconditional['absolute_file_charge_bytes']=full_source_bytes
    unconditional_guard=sum(unconditional.values())+sum(gpu['legacy_full_authentication'].values())+margin
    return dict(schema=SCHEMA, status='DERIVED_UNMEASURED_CONDITIONAL',
        physical_phases=physical, gpu_phases=gpu, guard_phase_envelopes=envelopes,
        guard_envelope_bytes=maximum, physical_bytes=max(map(lambda p:sum(p.values()),physical.values())),
        gpu_bytes=max(map(lambda p:sum(p.values()),gpu.values())),
        requested_mem_gib=math.ceil(maximum/GIB), requested_gpu_mem_gib=math.ceil(
            max(map(lambda p:sum(p.values()),gpu.values()))/GIB),
        maximum_physical_admission_bytes=104*GIB,
        fits_104gib_conditional_envelope=maximum<=104*GIB,
        guard_margin_bytes=margin, reference_bytes=weight,
        max_selected_bytes=weight, max_parity_cpu_bytes=2*widest,
        legacy_hash_absolute_file_cap_bytes=8*GIB,
        legacy_post_hash_absolute_file_cap_bytes=residue,
        legacy_unconditional_file_residue_bytes=full_source_bytes,
        legacy_unconditional_guard_bytes=unconditional_guard,
        legacy_unconditional_fits_104gib=unconditional_guard<=104*GIB,
        full_source_payload_bytes=full_source_bytes, full_source_shard_count=source_file_count,
        profiler_memory_reserve_bytes=profile,
        in_process_instruments=['cProfile main phases and hash worker threads', '/proc/self/io',
            'observations at existing source hash and safetensors reader seams'],
        gpu_kernel_attribution='UNMEASURED; no retained Torch profiler event state',
        max_source_event_bytes=64*MIB, hash_read_buffer_bytes=hash_buffer,
        cpu_reservation=7, cpu_roles=dict(main=1, source_read_workers=4,prefetch=1,telemetry=1),
        native_threads_per_worker=1, cache_slots=2, prefetch_workers=1,
        cache_headroom_gib=24, device_headroom_bytes=8*GIB,
        arm_deadlines_seconds=dict(full=1800, selected=900), total_deadline_seconds=3600,
        order=['full','selected'], automatic_followup=False,
        source_forward_count=0, anchor_render_count=0,
        interpretation='One ordered pilot pair. No order-independent speed or full anchor throughput claim.',
        residue_policy='Existing page advice only; absolute cgroup file charge must fit the legacy hash cap '
            'and the stricter post-hash/between-arm cap. Refuse if reclamation does not meet the bounds; '
            'never drop global caches, retry, or replace the full baseline with a subset hash.',
        limitations=['The unconditional full-page-residue envelope does not qualify a 104 GiB completion.',
            'The conditional envelope requires observed cgroup and CUDA guard inequalities throughout.',
            'Physical and CUDA charges may overlap; the guard deliberately adds both.',
            'Deadlines are hard refusal limits, not completion predictions.',
            'Source file cache warmth, order, and other-host activity remain recorded confounders.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inputs=json.loads(args.inputs.read_text())
    census=sealed_json(inputs['census'])
    names=inputs['selected_units']
    if len(names)!=4 or len(set(names))!=4 or any(name not in census['unit_shapes'] for name in names):
        raise ValueError('requires the four reviewed source units')
    if any(census.get(k)!=v for k,v in dict(nsamples=512,seqlen=512,seed=0,layer_stride=1).items()):
        raise ValueError('requires the unchanged full GLM census draw')
    shapes={name:census['unit_shapes'][name] for name in sorted(names)}
    from prismaquant.autoscale import selected_anchor_resources
    base=selected_anchor_resources(census['model'],unit_shapes=shapes,counts=census['counts'],
        max_act_rows=512,cache_slots=2,prefetch_workers=1,headroom_gb=24,anchor_batch_size=1)
    root=Path(census['model'])
    files=[]
    expected=census['expert_projection']['producer']['source']['files']
    for path in sorted(root.glob('*.safetensors')):
        if path.name not in expected:
            raise ValueError('source shard is absent from the frozen producer roster')
        state=path.stat()
        files.append(dict(name=path.name,bytes=state.st_size,sha256=expected[path.name],
                          provenance='SHA copied from frozen census; only file length inspected here'))
    if set(expected)!=set(row['name'] for row in files):
        raise ValueError('source payload roster differs from the full census')
    result=resource_plan(base,shapes,full_source_bytes=sum(row['bytes'] for row in files),
                         source_file_count=len(files))
    result.update(inputs=inputs,selected_shapes=shapes,selected_anchor_resources=base,
                  source_payload_roster=files)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    args.output.write_bytes(raw)
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
        status=result['status'],physical_bytes=result['physical_bytes'],gpu_bytes=result['gpu_bytes'],
        guard_bytes=result['guard_envelope_bytes'],mem_gib=result['requested_mem_gib'],
        full_payload_bytes=result['full_source_payload_bytes'],
        unconditional_guard_bytes=result['legacy_unconditional_guard_bytes'])))

if __name__=='__main__':
    main()
