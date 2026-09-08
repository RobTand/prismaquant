"""CPU-only preparation of the bounded GLM v3 original-wire screen.

This does not submit work, load model payloads, or authorize native execution.
Run through PrismaBuild because resource derivation is a qualification probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

GIB = 1024**3
MIB = 1024**2


def sealed_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"sealed input changed: {path}")
    return json.loads(raw)


def validate_cells(proposal, census, groups):
    if proposal.get('schema') != 'prismaquant.glm_native_candidate_screen.proposal.v3':
        raise ValueError('requires the v3 native proposal')
    names = proposal['selected_logical_units']
    if len(names) != 4 or len(set(names)) != 4 or len(proposal['cells']) != 28:
        raise ValueError('screen must contain four units and 28 distinct cells')
    seen = set()
    for cell in proposal['cells']:
        name, fmt = cell['qname'], cell['format']
        if name not in names or (name, fmt) in seen:
            raise ValueError('unknown unit or duplicate native cell')
        seen.add((name, fmt))
        if cell['shape'] != census['unit_shapes'][name]:
            raise ValueError('native shape differs from the original census')
        group = groups['groups'][cell['source_group']]
        if name not in group['members']:
            raise ValueError('native cell is outside its complete source group')
        allowed = {f"{family['family']}_R{rate}"
                   for family in group['families']
                   for rate in family['round_one_rungs_q256']}
        if fmt not in allowed:
            raise ValueError('native cell is outside the complete-group initial grid')
        if type(cell['memory_bytes']) is not int or cell['memory_bytes'] <= 0:
            raise ValueError('native cell requires positive payload bytes')
    if any(sum(name == pair[0] for pair in seen) != 7 for name in names):
        raise ValueError('each native unit requires seven cells')
    return {name: census['unit_shapes'][name] for name in sorted(names)}


def resource_plan(base, shapes, counts, cells):
    """Conservative phase arithmetic, not a measured native fit certificate.

    Preserve the existing source-window and loader-transient accounting.
    Capture loading reserves the verified loader's two serialized-file owners
    plus bounded scratch, decoded CPU storage and a GPU transfer destination.
    CPU source/page owners are excluded from the explicit GPU subset.
    """
    if base.get('schema') != 'prismaquant.selected_anchor_resources.v2':
        raise ValueError('requires selected_anchor_resources v2')
    phases = {name: dict(terms) for name, terms in base['phases'].items()}
    weight = base['selected_source_weight_bytes']
    storages = {name: 4*(shape[1]**2 + min(counts[name], 512)*shape[1])
                for name, shape in shapes.items()}
    total, largest = sum(storages.values()), max(storages.values())
    largest_render = max(math.prod(shape)*2 for shape in shapes.values())
    identity_staging = 2*max(largest_render,
        max(shape[1]**2*4 for shape in shapes.values()),
        max(min(counts[name], 512)*shape[1]*4 for name, shape in shapes.items()))
    file_cap, scratch_cap = GIB, 64*MIB
    # This is an enforced file cap, not an assertion about future file sizes.
    if largest > file_cap:
        raise ValueError('selected storage cannot fit the serialized file cap')
    headroom = phases['source_preparation']['declared_headroom_bytes']
    if headroom < 24*GIB:
        raise ValueError('screen preparation requires the declared 24 GiB headroom')
    phases['capture_prefetch'] = dict(
        selected_source_weight_bytes=weight, declared_headroom_bytes=headroom,
        selected_device_capture_bytes=total,
        decoded_cpu_entry_bytes=largest, transfer_destination_bytes=largest,
        serialized_private_buffer_bytes=file_cap,
        # Advice is best effort. Charge every selected file page even if no
        # completed file's pages are reclaimed before the last transfer.
        source_page_cache_bytes=len(shapes)*file_cap,
        validation_scratch_bytes=scratch_cap)
    wire_cap = max(cell['memory_bytes'] for cell in cells) + 64*MIB
    phases['resident_anchors'].update(
        # Covers PWC read, serialized render, independent wire decode and hash
        # staging. Finished rungs remain disk-backed in the existing PWC.
        render_qualification_bytes=4*largest_render,
        original_wire_buffer_and_pages_bytes=3*wire_cap,
        capture_serialized_buffer_bytes=file_cap,
        capture_source_page_cache_bytes=len(shapes)*file_cap,
        capture_load_scratch_bytes=scratch_cap,
        completed_artifact_page_cache_bytes=sum(
            math.prod(cell['shape'])*2 + cell['memory_bytes'] + 64*MIB for cell in cells),
        final_input_identity_staging_bytes=identity_staging)
    # One serialized Netdata reader; only bounded timing/freshness summaries
    # remain in Python. Charge the entire permitted JSONL page owner as well.
    for terms in phases.values():
        terms.update(telemetry_state_and_response_bytes=32*MIB,
                     telemetry_output_page_cache_bytes=256*MIB)
    gpu_headroom = 8*GIB
    prep = phases['source_preparation']
    # Loader transients are conservatively charged wholly to the GPU subset;
    # no claim is made that they are actually all CUDA allocations.
    gpu = dict(source_preparation=dict(
        selected_source_weight_bytes=weight,
        nonbody_source_bytes=prep['nonbody_source_bytes'],
        source_window_bytes=prep['source_window_bytes'],
        loader_transient_bytes=prep['loader_transient_bytes'],
        device_headroom_bytes=gpu_headroom))
    enc = phases['resident_anchors']
    gpu['resident_anchors'] = {key: enc[key] for key in (
        'selected_source_weight_bytes', 'selected_hessian_bytes', 'selected_prefix_bytes',
        'encoder_memo_bytes', 'factorization_scratch_bytes', 'compatible_batch_weight_bytes')}
    gpu['resident_anchors'].update(device_headroom_bytes=gpu_headroom,
        render_qualification_bytes=4*largest_render,
        original_wire_decode_bytes=wire_cap)
    gpu['capture_prefetch'] = dict(selected_source_weight_bytes=weight,
        selected_device_capture_bytes=total, transfer_destination_bytes=largest,
        device_headroom_bytes=gpu_headroom)
    physical_peak = max(map(lambda terms: sum(terms.values()), phases.values()))
    gpu_peak = max(map(lambda terms: sum(terms.values()), gpu.values()))
    if gpu_peak > physical_peak:
        raise ValueError('GPU subset exceeds physical bound')
    # CaptureMemoryGuard intentionally adds the entire CUDA reservation even
    # on hosts that also charge it in the cgroup. Keep this admission bound
    # separate from physical ownership; do not advertise it as measured DRAM.
    guard_phases = {name: sum(phases[name].values()) + sum(terms.values()) + 2*GIB
                    for name, terms in gpu.items()}
    guard_peak = max(guard_phases.values())
    return dict(schema='prismaquant.glm_native_screen_resource_plan.v1',
        status='DERIVED_UNMEASURED', physical_phases=phases, gpu_phases=gpu,
        physical_bytes=physical_peak, gpu_bytes=gpu_peak,
        physical_gib=math.ceil(physical_peak/GIB), gpu_gib=math.ceil(gpu_peak/GIB),
        guard_envelope_bytes=guard_peak, guard_phase_envelopes=guard_phases,
        requested_mem_gib=math.ceil(max(physical_peak, guard_peak)/GIB),
        cpu_reservation=6, cpu_roles=dict(main=1, source_read_workers=4, prefetch=1),
        native_threads_per_worker=1, cache_slots=2, prefetch_workers=1,
        encoder_memo_capacity=1, selected_source_weight_bytes=weight,
        selected_capture_bytes=total, max_capture_storage_bytes=largest,
        max_capture_file_bytes=file_cap, max_validation_scratch_bytes=scratch_cap,
        max_original_wire_file_bytes=wire_cap,
        max_resident_render_bytes=largest_render,
        final_input_identity_staging_bytes=identity_staging,
        profiler_policy=dict(in_process='per-phase cProfile aggregate statistics',
            torch_event_retention_bytes=0, gpu_trace='disabled',
            encode_gpu_kernel_attribution='unavailable',
            scope='wire correctness; no kernel-performance or saturation qualification'),
        execution_deadline_seconds=3600,
        deadline_semantics='PB hard termination cap; completion time is unmeasured',
        max_attempts=1, source_forward_count=0,
        limitations=['No measured native peak or completion prediction.',
            'GPU source transients conservatively retain the whole existing loader bound.',
            'Complete-capture source authentication and frozen review remain prerequisites.',
            '28 cells do not qualify the 864-member expert group or serving.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposal', 'census', 'group-contract', 'union-proposal'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proposal = sealed_json(args.proposal, args.proposal_sha256)
    census = sealed_json(args.census, args.census_sha256)
    groups = sealed_json(args.group_contract, args.group_contract_sha256)
    union = sealed_json(args.union_proposal, args.union_proposal_sha256)
    if (union['fixed']['census_sha256'] != args.census_sha256 or
            union['fixed']['model'] != census['model'] or
            union['native_acceptance_before_full_campaign']['bounded_candidate_cells'] != proposal['cells']):
        raise ValueError('native screen differs from the v3 union proposal')
    if proposal['group_contract_sha256'] != args.group_contract_sha256:
        raise ValueError('native proposal and complete-group contract disagree')
    shapes = validate_cells(proposal, census, groups)
    from prismaquant.autoscale import selected_anchor_resources
    base = selected_anchor_resources(census['model'], unit_shapes=shapes,
        counts=census['counts'], max_act_rows=512, cache_slots=2,
        prefetch_workers=1, headroom_gb=24, anchor_batch_size=1)
    result = resource_plan(base, shapes, census['counts'], proposal['cells'])
    result['inputs'] = {name: dict(path=str(getattr(args, name)),
        sha256=getattr(args, name+'_sha256')) for name in ('proposal', 'census', 'group_contract', 'union_proposal')}
    result['selected_anchor_resources'] = base
    result['cells'] = proposal['cells']
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict({key: result[key] for key in ('status', 'physical_gib', 'gpu_gib',
        'requested_mem_gib', 'physical_bytes', 'gpu_bytes', 'selected_capture_bytes')},
        resource_plan_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
