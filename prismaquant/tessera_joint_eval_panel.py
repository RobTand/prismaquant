"""Bound diagnostic evaluation rows inside an unchanged encoding calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from .calibration_data import load_calibration_input
from .cluster_campaign import _atomic_write_new_bytes

SCHEMA = 'prismaquant.tessera_joint_eval_panel.v1'
STATUS = 'diagnostic_pilot'
ALGORITHM = 'python_random_permutation_prefix_v1'


def observation_status(count):
    """An invocation with zero token rows does not observe that expert."""
    if (not isinstance(count, dict) or not {'tokens', 'calls'} <= set(count)
            or any(type(count[key]) is not int or count[key] < 0 for key in ('tokens', 'calls'))
            or (count['tokens'] > 0 and count['calls'] == 0)):
        raise ValueError('joint evaluation observation counts are invalid')
    return 'observed' if count['tokens'] > 0 else 'unknown_unobserved'


def _sha_ids(ids):
    return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()


def _indices(total, seed, size):
    if (type(total) is not int or type(seed) is not int or type(size) is not int
            or not 1 <= size < total):
        raise ValueError('joint evaluation needs a strict positive subset and integer seed')
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return order[:size]


def make_panel(ids, *, artifact_sha256, seed, size):
    """Freeze a nested permutation prefix; order is part of the token hash."""
    indices = _indices(len(ids), seed, size)
    selected = ids[indices].contiguous()
    return {'schema': SCHEMA, 'status': STATUS,
            'calibration_input_sha256': artifact_sha256,
            'selection': {'algorithm': ALGORITHM, 'seed': seed, 'size': size,
                          'indices': indices},
            'shape': list(selected.shape), 'eval_ids_sha256': _sha_ids(selected)}


def validate_panel_descriptor(panel, *, n_samples, seqlen, artifact_sha256):
    """Refuse malformed selections at plan intake, before the expensive run."""
    if not isinstance(panel, dict) or set(panel) != {
            'schema', 'status', 'calibration_input_sha256', 'selection', 'shape',
            'eval_ids_sha256'} or panel['schema'] != SCHEMA or panel['status'] != STATUS:
        raise ValueError('joint evaluation requires a complete diagnostic v1 panel')
    selection = panel['selection']
    if (not isinstance(selection, dict) or set(selection) !=
            {'algorithm', 'seed', 'size', 'indices'} or selection['algorithm'] != ALGORITHM):
        raise ValueError('joint evaluation requires an exact permutation-prefix selection')
    indices = _indices(n_samples, selection['seed'], selection['size'])
    if (selection['indices'] != indices or panel['shape'] != [selection['size'], seqlen]
            or panel['calibration_input_sha256'] != artifact_sha256
            or not isinstance(panel['eval_ids_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', panel['eval_ids_sha256']) is None):
        raise ValueError('joint evaluation indices, artifact, shape or hash descriptor differ')
    return panel


def select_panel(ids, calibration, panel):
    """Replay every selection coordinate and hash against the original IDs."""
    if panel is None:
        return ids, None
    validate_panel_descriptor(panel, n_samples=len(ids), seqlen=ids.shape[1],
                              artifact_sha256=calibration['artifact_sha256'])
    selection = panel['selection']
    expected = make_panel(ids, artifact_sha256=calibration['artifact_sha256'],
                          seed=selection['seed'], size=selection['size'])
    if panel != expected:
        raise ValueError('joint evaluation indices, artifact, shape or token hash differ')
    return ids[selection['indices']].contiguous(), expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args(argv)
    with open(args.base_plan) as source:
        plan = json.load(source)
    if plan.get('schema') != 'prismaquant.tessera_joint_aura.plan.v1' or 'joint_eval' in plan:
        raise ValueError('joint evaluation plan must extend an original v1 plan once')
    execution = plan['execution']
    ids, calibration = load_calibration_input(plan['calibration_input']['path'],
        expected_sha256=plan['calibration_input']['sha256'],
        n_samples=execution['n_calib_samples'], seqlen=execution['calib_seqlen'])
    plan['joint_eval'] = make_panel(ids, artifact_sha256=calibration['artifact_sha256'],
                                    seed=args.seed, size=args.size)
    # The old output root is sealed to its 512-window plan: a caller must
    # provide a new destination before publishing a pilot plan.
    if (not args.output_root or Path(args.output_root).resolve() ==
            Path(plan['output_root']).resolve()):
        raise ValueError('diagnostic joint evaluation requires a distinct output root')
    plan['output_root'] = args.output_root
    storage = execution.get('boundary_storage')
    if not isinstance(storage, dict) or 'directory' not in storage:
        raise ValueError('diagnostic joint evaluation requires exact boundary storage')
    storage['directory'] = str(Path(args.output_root) / 'exact-boundaries')
    _atomic_write_new_bytes(Path(args.output), (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
