"""Measure capture sealing on one real resident row, then qualify its wires.

The ordinary campaign owns intake, residency, encoding and publication. This
observer compares fresh plain and reference-backed sources only after the
campaign has published commitments over its resident H. It bounds the existing
first-round batch list, without changing the selected row or its calibration.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def set_option(argv, key, value):
    if key in argv:
        argv[argv.index(key)+1] = str(value)
    else:
        argv.extend([key, str(value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-spec')
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-result', required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--units', type=int, default=16)
    parser.add_argument('campaign_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.campaign_args
    if command[:1] == ['--']:
        command = command[1:]
    if args.container_spec:
        from tools.tessera_campaign_container import main as launch
        spec = json.loads(args.container_spec)
        profile = Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        spec['container']['mounts'].append(dict(source=str(profile.parent), target='/pb-profile'))
        spec['env']['PRISMABUILD_PROFILE_TORCH_OUT'] = '/pb-profile/'+profile.name
        return launch(['--spec', json.dumps(spec), '--', 'python3', '-u', '-m',
            'experiments.glm_hessian_seal_profile', '--out', args.out,
            '--reference-result', args.reference_result, '--reference-sha256', args.reference_sha256,
            '--producer-sha256', args.producer_sha256, '--units', str(args.units), '--', *command])

    import torch
    from tessera import cached_unit
    from tessera.manifest import ScalePlaneKind
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_hessian as th
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    from experiments.glm_full_capture_profile import CaptureObserver

    assert digest(args.reference_result) == args.reference_sha256
    assert cached_unit.encoder_source_sha256() == args.producer_sha256
    reference = json.loads(Path(args.reference_result).read_text())
    reference = {s['qname']: s for s in reference['arms'][0]['signatures']}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='prismaquant.glm_hessian_seal_profile.v1', status='running',
        started_unix=time.time(), producer_source_sha256=args.producer_sha256,
        torch=torch.__version__, cuda=torch.version.cuda, command=command,
        reference=dict(path=args.reference_result, sha256=args.reference_sha256),
        scope='resident row capture sealing plus bounded native wire qualification',
        full_campaign_complete=False, arms=[], prepare_calls=[], scheduled=[])

    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    original_source = th.activation_source
    original_identity = cached_unit.tensor_identity
    original_batches = campaign._anchor_batches
    original_prepare = campaign._prepare_anchor
    intercepted = [False]

    def observed_source(hessians, identity, **kwargs):
        if kwargs.get('reference_path') is None:
            return original_source(hessians, identity, **kwargs)
        assert not intercepted[0]
        intercepted[0] = True
        descriptor = json.loads(Path(kwargs['reference_path']).read_text())
        expected_seal = descriptor['capture_sha256']
        result['resident_units'] = len(hessians)
        result['resident_hessian_bytes'] = sum(h.numel()*h.element_size() for h in hessians.values())
        result['reference_capture'] = dict(path=str(kwargs['reference_path']),
            sha256=digest(kwargs['reference_path']), capture_sha256=expected_seal)
        # The same actual unit supplies the before/after factorisation witness.
        name = sorted(hessians)[0]
        prepared = None
        for index, enabled in enumerate((False, True, False, True, True, False)):
            profiled = index < 2
            label = ('profile' if profiled else 'measured')+f'-{index}-'+('resident' if enabled else 'plain')
            count = dict(calls=0, bytes=0)

            def counted(tensor):
                count['calls'] += 1
                count['bytes'] += tensor.numel()*tensor.element_size()
                return original_identity(tensor)

            cached_unit.tensor_identity = counted
            profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) if profiled else nullcontext())
            record = dict(label=label, resident=enabled, profiled=profiled)
            try:
                with ExitStack() as scope, profiler as prof:
                    torch.cuda.synchronize()
                    record['started_unix'] = time.time()
                    started = time.perf_counter()
                    with torch.profiler.record_function('construct_and_seal_resident_capture'):
                        source = original_source(hessians, identity, **(
                            dict(reference_path=kwargs['reference_path'], source_scope=scope) if enabled else {}))
                        seal = source.capture_sha256()
                    torch.cuda.synchronize()
                    record.update(seconds=time.perf_counter()-started, finished_unix=time.time(),
                        tensor_identity=dict(count), capture_sha256=seal)
                    assert seal == expected_seal
                    assert count['calls'] == (0 if enabled else len(hessians))
                    if profiled:
                        before = count['calls']
                        with torch.profiler.record_function('consume_one_actual_hessian'):
                            values = source.for_unit(name+'.weight', hessians[name].shape[0], 'cuda',
                                                     scale_plane=ScalePlaneKind.CHANNEL)
                        torch.cuda.synchronize()
                        assert count['calls'] == before+1
                        if prepared is None:
                            prepared = values
                        else:
                            assert values.keys() == prepared.keys()
                            for key, value in values.items():
                                assert torch.equal(value, prepared[key]) if isinstance(value, torch.Tensor) else value == prepared[key]
                            prepared = values = None
                            record['factor_and_metric_exact_parity'] = True
                    if enabled:
                        record['reader'] = source.hessians.receipt()
                        assert record['reader']['loaded_entries'] == 0
                if profiled:
                    path = out/(label+'.trace.json.gz')
                    prof.export_chrome_trace(str(path))
                    record['profile'] = dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)
                    shutil.copyfile(path, os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
            finally:
                cached_unit.tensor_identity = original_identity
            result['arms'].append(record)
            save()
        return original_source(hessians, identity, **kwargs)

    def batches(*pos, **kw):
        selected = []
        units = 0
        for batch in original_batches(*pos, **kw):
            if units+len(batch) > args.units:
                break
            selected.append(batch)
            units += len(batch)
        assert units == args.units
        result['scheduled'] = [list(item) for batch in selected for item in batch]
        return selected

    def prepare(*pos, **kw):
        started = time.perf_counter()
        try:
            return original_prepare(*pos, **kw)
        finally:
            result['prepare_calls'].append(dict(qname=kw.get('qname'), seconds=time.perf_counter()-started))

    th.activation_source = observed_source
    campaign._anchor_batches = batches
    campaign._prepare_anchor = prepare
    run = list(command)
    for key, value in (('--out',out/'cost.pkl'), ('--cache-dir',out/'cache'),
            ('--checkpoint',out/'cost.anchors.json'), ('--max-rounds',1)):
        set_option(run, key, value)
    save()
    try:
        with CaptureObserver(out/'observer', profile_layers=()):
            rc = campaign.main(run)
            assert rc == 0 and intercepted[0]
            manifest = json.loads((out/'cost.anchors.json').read_text())
            signatures = []
            for name, _, _ in result['scheduled']:
                state = _load_unit(unit_path(out/'cost.anchors.json.parts',name),
                    stage=manifest['stage'], qname=name, identity_sha256=manifest['identity_sha256'])
                assert len(state['anchors']) == len(state['wire_records']) == 1
                anchor = state['anchors'][0]
                wire = state['wire_records'][anchor['format_name']]
                path = out/'cache/wire'/wire['file']
                assert digest(path) == wire['blob_sha256'] and path.stat().st_size == wire['blob_bytes']
                signature = dict(qname=name, format=anchor['format_name'], dloss=anchor['dloss'], wire_sha256=wire['blob_sha256'])
                assert all(signature[key] == reference[name][key] for key in ('qname','dloss','wire_sha256'))
                signatures.append(signature)
            with (out/'cost.pkl').open('rb') as handle:
                payload = pickle.load(handle)
            result.update(returncode=rc, signatures=signatures, exact_wire_and_score_parity=True,
                completed_guard=payload['provenance']['selected_source_preparation']['memory_guard'])
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        th.activation_source = original_source
        campaign._anchor_batches = original_batches
        campaign._prepare_anchor = original_prepare
        cached_unit.tensor_identity = original_identity
        result['finished_unix'] = time.time()
        save()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
