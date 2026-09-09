"""Measure a fixed anchor prefix after the campaign's complete resident prefetch.

This is a PB benchmark, not a completed pricing action. It uses the existing
campaign, observer and checkpoint writer and stops before the next encode.
"""
from __future__ import annotations

import argparse
import time

import torch

from experiments.glm_full_capture_profile import AnchorObserver, selected_anchor_command


class PrefixComplete(BaseException):
    """Bypass the campaign's ordinary per-anchor encoder-error recovery."""


class PrefixEncodingFailed(BaseException):
    """A benchmark cannot skip a failed encode and compare different work."""


def _resident_bytes(tensors):
    return sum(t.numel()*t.element_size() for t in tensors.values())


def run_prefix(campaign, command, observer, *, limit, expected_source_units):
    if type(limit) is not int or not 0 < limit <= 1024:
        raise ValueError('prefix limit must be in [1, 1024]')
    if type(expected_source_units) is not int or expected_source_units < 1:
        raise ValueError('prefix needs a positive complete source-unit count')
    methods = ('_measure_anchor', '_measure_anchor_batch', '_prefetch_selected_capture')
    originals = {name: getattr(campaign, name) for name in methods}
    completed = 0
    prefetch_count = 0
    calls = []
    stopped = False
    observer.result.update(work_scope='fixed anchor prefix after complete selected resident prefetch',
        campaign_completed=False, requested_anchor_units=limit, expected_source_units=expected_source_units,
        prefix_calls=calls)

    def prefetch(*args, **kwargs):
        nonlocal prefetch_count
        values, capture, receipt = originals['_prefetch_selected_capture'](*args, **kwargs)
        acts, hessians, counts, maxima = values
        if (set(acts) != set(hessians) or set(acts) != set(kwargs['names']) or
                len(acts) != expected_source_units):
            raise RuntimeError('prefix did not prefetch its complete selected source scope')
        prefetch_count += 1
        observer.result['resident_prefetch'] = dict(units=len(acts),
            hessian_bytes=_resident_bytes(hessians), activation_bytes=_resident_bytes(acts),
            devices=sorted({str(t.device) for t in [*acts.values(), *hessians.values()]}),
            cuda_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
            cuda_reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
            capture=capture, receipt=receipt, finished_unix=time.time())
        return values, capture, receipt

    def wrap(original):
        observed = observer.wrap_anchor(original)
        def measure(*args, **kwargs):
            nonlocal completed
            names = list(kwargs['qnames']) if 'qnames' in kwargs else [kwargs['qname']]
            if completed == limit:
                raise PrefixComplete()
            if completed+len(names) > limit:
                raise PrefixEncodingFailed('prefix limit cuts across a producer batch')
            if prefetch_count != 1:
                raise PrefixEncodingFailed('encode did not follow exactly one complete prefetch')
            record = dict(qnames=names, format_name=kwargs['format_name'], started_unix=time.time())
            try:
                result = observed(*args, **kwargs)
            except Exception as error:
                raise PrefixEncodingFailed(f'prefix encoder failed: {error!r}') from error
            record['finished_unix'] = time.time()
            calls.append(record)
            completed += len(names)
            return result
        return measure

    try:
        campaign._prefetch_selected_capture = prefetch
        for method in methods[:2]:
            setattr(campaign, method, wrap(originals[method]))
        try:
            campaign.main(command)
        except PrefixComplete:
            stopped = True
    finally:
        for method, original in originals.items():
            setattr(campaign, method, original)
        observer.result.update(completed_anchor_units=completed, prefix_boundary_reached=stopped)
    if not stopped or completed != limit or prefetch_count != 1:
        raise RuntimeError('campaign did not reach the requested measured-prefix boundary')
    identities = [(name, call['format_name']) for call in calls for name in call['qnames']]
    if len(set(identities)) != limit:
        raise RuntimeError('prefix measured a repeated unit/rung')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-out', required=True)
    parser.add_argument('--limit-anchors', type=int, required=True)
    parser.add_argument('--expected-source-units', type=int, required=True)
    parser.add_argument('--profile-calls', required=True)
    parser.add_argument('--trace-max-bytes', type=int, required=True)
    parser.add_argument('--profile-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    selected_anchor_command(command)
    if not torch.cuda.is_available():
        raise RuntimeError('native campaign prefix profiling requires CUDA')
    from prismaquant import tessera_campaign as campaign
    observer = AnchorObserver(args.evidence_out,
        profile_calls=[int(x) for x in args.profile_calls.split(',')],
        trace_max_bytes=args.trace_max_bytes, command=command, cuda_only=True,
        window_seconds=args.profile_seconds)
    with observer:
        run_prefix(campaign, command, observer, limit=args.limit_anchors,
                   expected_source_units=args.expected_source_units)
        if observer.result['resident_prefetch']['devices'] != ['cuda:0']:
            raise RuntimeError('native prefix inputs were not all resident on the selected GPU')


if __name__ == '__main__':
    main()
