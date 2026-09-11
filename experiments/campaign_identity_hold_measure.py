"""Measure the campaign identity hold against its planned bound on CPU.

A CPU action: build real ``_BoundCheckpointUnitIdentity`` holders through the
campaign's own roster and planner over the GLM routed-expert closed roster
(every ``TESSERA_E4M3_K1`` rung the family admits), at the row's unit shapes,
and report ``observed_metadata_bytes`` per unit against the planned bound,
the construction transient against the planned scratch, and the extrapolated
864-unit reservation.  Synthetic weights and Hessians at the real shapes;
the hash values are irrelevant to the object sizes and are not reported as
identities of anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--census', required=True, type=Path)
    parser.add_argument('--units', nargs='+', default=[
        'model.language_model.layers.28.mlp.experts.0.gate_proj',
        'model.language_model.layers.28.mlp.experts.0.up_proj',
        'model.language_model.layers.28.mlp.experts.0.down_proj'])
    parser.add_argument('--row-units', type=int, default=864)
    parser.add_argument('--reservation-bytes', type=int, default=256*1024**2)
    args = parser.parse_args(argv)

    import torch
    from prismaquant import tessera_campaign as tc, tessera_hessian as th
    from prismaquant.tessera_formats import get_tessera_family
    from prismaquant.tessera_menu import expand_tessera_menu, MENU_READABLE

    shapes = json.loads(args.census.read_text())['unit_shapes']
    family = get_tessera_family('TESSERA_E4M3_K1')
    weights, hessians, menus, projections = {}, {}, {}, {}
    for name in args.units:
        out_features, in_features = (int(v) for v in shapes[name])
        generator = torch.Generator().manual_seed(len(name))
        weights[name] = torch.randn(out_features, in_features, generator=generator).to(torch.bfloat16)
        hessians[name] = torch.eye(in_features)
        menus[name] = expand_tessera_menu((out_features, in_features), mode=MENU_READABLE,
                                          families=(family,))
        # A packed routed expert's projection record, as the campaign's
        # projected_units carry it; the sizes are what matter here.
        expert = int(name.split('.experts.')[1].split('.')[0])
        projection = name.rsplit('.', 1)[-1]
        projections[name] = dict(tensor=name + '.weight', source_tensor=name.rsplit('.experts.', 1)[0]
            + '.experts.' + ('w13' if projection != 'down_proj' else 'w2') + '.weight',
            source_layout='packed', source_slice={'expert': expert}, expert=expert,
            projection=projection, group='w13' if projection != 'down_proj' else 'w2',
            rows=out_features, cols=in_features)
    source = th.activation_source(hessians, th.calibration_identity(
        'identity-hold-measurement', [torch.arange(16).reshape(1, 16)], fit_tokens=16))
    static_scales = {}
    started = time.perf_counter()
    bounds, scratch = tc._campaign_identity_metadata_plan(
        weights=weights, menus=menus, calibration_source=source,
        projected_units=projections, static_scales=static_scales)
    plan_seconds = time.perf_counter() - started
    report = dict(schema='prismaquant.campaign_identity_hold_measurement.v1',
        python=sys.version, torch=torch.__version__, units={}, plan_seconds=plan_seconds,
        planned_scratch_bytes=scratch, roster_widths={n: len(m) for n, m in menus.items()})
    # Where the bind time goes: the per-format roster helpers, the producer's
    # tensor hash (whose ``tobytes`` copy is also the construction peak), and
    # the rest.  Wrapped on the modules the campaign resolves them from.
    from tessera import cached_unit
    from prismaquant import tessera_formats, tessera_render
    timings = {}

    def timed(owner, attr, label):
        original = getattr(owner, attr)
        def wrapped(*a, **k):
            t = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                entry = timings.setdefault(label, [0, 0.0])
                entry[0] += 1; entry[1] += time.perf_counter() - t
        setattr(owner, attr, wrapped)
        return owner, attr, original

    restore = [timed(tessera_formats, 'tessera_wire_recipe', 'tessera_wire_recipe'),
               timed(tessera_render, 'rung_accepts_hessian', 'rung_accepts_hessian'),
               timed(cached_unit, 'tensor_identity', 'tensor_identity'),
               timed(tc, '_checkpoint_anchor_identity', '_checkpoint_anchor_identity')]
    tracemalloc.start()
    held = {}
    try:
        for name in args.units:
            timings.clear()
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            t0 = time.perf_counter()
            roster = tc._campaign_identity_anchor_roster(name, menus[name],
                calibration_source=source, static_scales=static_scales)
            roster_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            held[name] = tc.bind_checkpoint_unit_identity(roster, source_weight=weights[name],
                calibration_source=source, projected_unit=projections[name],
                static_scales=static_scales, retain_source_receipt=False)
            bind_seconds = time.perf_counter() - t0
            split = {label: dict(calls=calls, seconds=seconds) for label, (calls, seconds) in timings.items()}
            current, peak = tracemalloc.get_traced_memory()
            del roster
            observed = held[name].observed_metadata_bytes()
            t1 = time.perf_counter()
            derived = held[name].derive(source_weight=weights[name], qname=name,
                format_name=menus[name][0].format_name, grid=family.payload_grid(),
                rung=int(menus[name][0].body_rate_q256), activation=source,
                projected_unit=projections[name])
            derive_seconds = time.perf_counter() - t1
            # The producer's tensor_identity makes one bytes() copy of the
            # tensor it hashes; the Hessian's is the largest live transient
            # of construction and exists on the historical per-anchor path
            # too.  The roster's own transient is what is left after it.
            hash_copy = max(weights[name].numel() * weights[name].element_size(),
                            hessians[name].numel() * hessians[name].element_size())
            transient = (peak - before) - (current - before)
            report['units'][name] = dict(shape=shapes[name], formats=len(menus[name]),
                planned_bound_bytes=bounds[name], observed_metadata_bytes=observed,
                within_bound=observed <= bounds[name],
                traced_retained_bytes=current - before, traced_construction_peak_bytes=peak - before,
                construction_transient_bytes=transient,
                producer_hash_copy_bytes=hash_copy,
                roster_transient_bytes=max(0, transient - hash_copy),
                roster_seconds=roster_seconds, bind_seconds=bind_seconds, bind_split=split,
                derive_seconds=derive_seconds,
                template_json_bytes=len(json.dumps(derived, sort_keys=True).encode()))
    finally:
        tracemalloc.stop()
        for owner, attr, original in restore:
            setattr(owner, attr, original)
        for unit in held.values():
            unit.close()
    widest = max(report['units'].values(), key=lambda u: u['planned_bound_bytes'])
    transient = max(u['roster_transient_bytes'] for u in report['units'].values())
    report['row'] = dict(units=args.row_units, reservation_bytes=args.reservation_bytes,
        planned_total_bytes=args.row_units * widest['planned_bound_bytes'] + scratch,
        observed_extrapolated_bytes=args.row_units * max(
            u['observed_metadata_bytes'] for u in report['units'].values()) + transient,
        planned_fits_reservation=(args.row_units * widest['planned_bound_bytes'] + scratch
                                  <= args.reservation_bytes),
        scratch_covers_roster_transient=scratch >= transient,
        producer_hash_copy_peak_bytes=max(u['producer_hash_copy_bytes'] for u in report['units'].values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['row']), flush=True)
    for name, unit in report['units'].items():
        print(json.dumps(dict(unit=name, **{k: unit[k] for k in (
            'formats', 'planned_bound_bytes', 'observed_metadata_bytes', 'within_bound',
            'construction_transient_bytes', 'roster_transient_bytes', 'roster_seconds',
            'bind_seconds', 'bind_split', 'derive_seconds')})), flush=True)
    if not all(u['within_bound'] for u in report['units'].values()):
        raise SystemExit('observed identity metadata exceeded the planned bound')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
