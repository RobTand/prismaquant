"""Bound diagnostic evaluation rows inside an unchanged encoding calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from .calibration_data import load_calibration_input
from .cluster_campaign import _atomic_write_new_bytes
from .dev_mode import seal_check

from .joint_eval_observation import STATUS, observation_status  # noqa: F401  (re-exported)
from .digests import DIRECT_ASCII_INDENT2_LAX, bytes_sha256hex

SCHEMA = 'prismaquant.tessera_joint_eval_panel.v1'
DRAW_SCHEMA = 'prismaquant.tessera_joint_eval_draw.v1'
ALGORITHM = 'python_random_permutation_prefix_v1'


def _sha_ids(ids):
    return bytes_sha256hex(ids.contiguous().numpy().tobytes())


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
            or not isinstance(panel['eval_ids_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', panel['eval_ids_sha256']) is None):
        raise ValueError("joint evaluation indices, shape or hash descriptor differ")
    seal_check("panel encoding calibration binding", artifact_sha256,
               panel["calibration_input_sha256"], where="joint evaluation panel")
    return panel


def select_panel(ids, calibration, panel):
    """Replay every selection coordinate and hash against the original IDs."""
    if panel is None:
        return ids, None
    validate_panel_descriptor(panel, n_samples=len(ids), seqlen=ids.shape[1],
                              artifact_sha256=calibration['artifact_sha256'])
    selection = panel['selection']
    expected = make_panel(ids, artifact_sha256=panel["calibration_input_sha256"],
                          seed=selection["seed"], size=selection["size"])
    if panel != expected:
        raise ValueError("joint evaluation indices, shape or token hash differ")
    return ids[selection["indices"]].contiguous(), panel


def _draw_sha256(value, where):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'joint evaluation draw {where} requires a SHA256 digest')
    return value


def _draw_path(value):
    """Structural path-string checks only; the byte reader owns file reality."""
    if not isinstance(value, str):
        raise ValueError('joint evaluation draw calibration_input requires a pinned path string')
    if not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('joint evaluation draw calibration_input requires an absolute '
                         'path without parent traversal')
    return value


def validate_eval_draw_descriptor(draw, *, encoding_artifact_sha256):
    """Refuse malformed independent draws at plan intake, before any load.

    Closed grammar, digest formats, positive dimensions and a structural
    absolute path raise in both modes. The old-encoding binding is a seal
    (D32): ``seal_check`` stamps and continues in dev mode, refusing only a
    certified run. There is no distinctness or file-existence gate here;
    the byte reader refuses actual contents that miss their pin.
    """
    if not isinstance(draw, dict) or set(draw) != {
            'schema', 'status', 'encoding_calibration_input_sha256',
            'calibration_input', 'shape', 'calibration_sha256'} \
            or draw['schema'] != DRAW_SCHEMA or draw['status'] != STATUS:
        raise ValueError('joint evaluation draw requires a complete independent v1 descriptor')
    _draw_sha256(draw['encoding_calibration_input_sha256'], 'old encoding binding')
    seal_check('old encoding calibration binding', encoding_artifact_sha256,
               draw['encoding_calibration_input_sha256'],
               where='joint evaluation draw descriptor')
    _draw_sha256(draw['calibration_sha256'], 'token identity')
    calibration_input = draw['calibration_input']
    if (not isinstance(calibration_input, dict)
            or set(calibration_input) != {'path', 'sha256'}):
        raise ValueError('joint evaluation draw requires a closed calibration_input reference')
    _draw_sha256(calibration_input['sha256'], 'artifact pin')
    shape = draw['shape']
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(dim) is not int or dim < 1 for dim in shape)):
        raise ValueError('joint evaluation draw requires positive integer rows and seqlen')
    _draw_path(calibration_input['path'])
    return draw


def load_eval_draw(draw, *, encoding_calibration):
    """Load exactly the independent draw its descriptor pins, from its file.

    Returns ``(ids, calibration, draw)``: the fresh tokens exactly as the
    artifact stores them (no reshape, no recapture, no subset), the loaded
    calibration record with all of its provenance, and the validated
    descriptor. Byte and shape integrity stay hard in both modes: the
    artifact's own digest pin, the loaded int64 ``calibration_sha256`` and
    the exact declared shape refuse actual contents that differ. Provenance
    source/model/text_sha256 is a seal (D32): ``seal_check`` stamps and
    continues in dev mode, refusing only a certified run. The split is a
    pricing-use constraint: held-out validation, test and final-benchmark
    tokens are not training calibration and refuse in both modes.
    """
    validate_eval_draw_descriptor(draw,
        encoding_artifact_sha256=encoding_calibration['artifact_sha256'])
    ids, calibration = load_calibration_input(draw['calibration_input']['path'],
        expected_sha256=draw['calibration_input']['sha256'],
        n_samples=draw['shape'][0], seqlen=draw['shape'][1])
    if calibration['calibration_sha256'] != draw['calibration_sha256']:
        raise ValueError('joint evaluation draw token identity differs from the '
                         'loaded int64 draw')
    provenance, encoding = calibration['provenance'], encoding_calibration['provenance']
    for field in ('source', 'model', 'text_sha256'):
        seal_check(f'draw provenance {field}', encoding.get(field),
                   provenance.get(field), where='joint evaluation draw')
    if provenance.get('split_role') != 'calibration':
        raise ValueError('joint evaluation draw requires the calibration split, '
                         'not a sealed final panel')
    return ids, calibration, draw


def evaluation_execution(config):
    """The execution a runtime plans with, tightened to its bound evaluation.

    Default no-eval plans return their execution untouched. A bound
    diagnostic panel or independent draw overrides the declared rows and
    context with its own exact shape after its descriptor validates against
    the encoding calibration the plan pins, so planners declare the tokens
    the evaluation actually reads instead of the encoding draw's window.
    """
    execution = config['execution']
    if 'joint_eval_draw' in config:
        if 'joint_eval' in config:
            raise ValueError('joint evaluation draw and diagnostic panel are '
                             'mutually exclusive plans')
        rows, seqlen = validate_eval_draw_descriptor(
            config['joint_eval_draw'],
            encoding_artifact_sha256=config['calibration_input']['sha256'])['shape']
    elif 'joint_eval' in config:
        rows, seqlen = validate_panel_descriptor(
            config['joint_eval'], n_samples=execution['n_calib_samples'],
            seqlen=execution['calib_seqlen'],
            artifact_sha256=config['calibration_input']['sha256'])['shape']
    else:
        return execution
    bounded = dict(execution)
    bounded['n_calib_samples'] = rows
    bounded['calib_seqlen'] = seqlen
    return bounded


def select_evaluation(ids, encoding_calibration, config):
    """Return ``(ids, calibration, descriptor)`` for the plan's draw owner.

    A bound independent draw loads its own artifact and returns the fresh
    calibration record; the diagnostic panel subsets the encoding draw and
    returns the encoding calibration unchanged; neither returns the full
    encoding draw with a ``None`` descriptor, exactly as ``select_panel``
    always did.
    """
    if 'joint_eval_draw' in config:
        if 'joint_eval' in config:
            raise ValueError('joint evaluation draw and diagnostic panel are '
                             'mutually exclusive plans')
        return load_eval_draw(config['joint_eval_draw'],
                              encoding_calibration=encoding_calibration)
    selected, panel = select_panel(ids, encoding_calibration, config.get('joint_eval'))
    return selected, encoding_calibration, panel


def _selected_formats(formats, wanted):
    """The available entry's own order, narrowed to the targeted names."""
    if isinstance(formats, dict):
        return {fmt: formats[fmt] for fmt in formats if fmt in wanted}
    return [fmt for fmt in formats if fmt in wanted]


def evaluation_formats(config, available_formats):
    """The formats an evaluation actually prices, when the plan narrows them.

    A plan may bind ``joint_eval_targets`` -- exact qname to a nonempty,
    duplicate-free list of format strings -- only alongside a diagnostic
    evaluation (a bound draw or the existing panel): this is measurement
    scope, deciding that named units are priced in named formats instead of
    repricing every available format. Names are exact: an unknown qname or
    format refuses rather than guessing a mapping, and no prefix or tail
    matching is performed. The selected mapping keeps the available
    mapping's own deterministic order; without targets the available
    mapping is returned unchanged.
    """
    if 'joint_eval_targets' not in config:
        return available_formats
    if 'joint_eval_draw' not in config and 'joint_eval' not in config:
        raise ValueError('joint evaluation targets require a diagnostic draw '
                         'or panel plan')
    targets = config['joint_eval_targets']
    if not isinstance(targets, dict) or not targets:
        raise ValueError('joint evaluation targets require a nonempty qname mapping')
    for qname, wanted in targets.items():
        if not isinstance(qname, str) or not qname:
            raise ValueError('joint evaluation targets require exact nonempty qnames')
        if qname not in available_formats:
            raise ValueError(f'joint evaluation targets name unavailable '
                             f'qname {qname!r}')
        if (not isinstance(wanted, list) or not wanted
                or any(not isinstance(fmt, str) or not fmt for fmt in wanted)
                or len(set(wanted)) != len(wanted)):
            raise ValueError(f'joint evaluation targets require a nonempty '
                             f'unique format list for {qname!r}')
        unavailable = [fmt for fmt in wanted if fmt not in available_formats[qname]]
        if unavailable:
            raise ValueError(f'joint evaluation targets name unavailable '
                             f'formats {unavailable} for {qname!r}')
    return {qname: _selected_formats(formats, targets[qname])
            for qname, formats in available_formats.items() if qname in targets}


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
    _atomic_write_new_bytes(Path(args.output), DIRECT_ASCII_INDENT2_LAX.encoded(plan) + b'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
