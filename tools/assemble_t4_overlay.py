"""Prepare immutable candidate metadata after every new render is qualified.

Does not issue any Stage A capture reuse authority or activate Stage B.

The catalog is named by path and SHA-256 (``--catalog``). A result file binds
the digest of the cell it qualified. With ``--rebinding``
(``rebind_t4_qualified_results.py``), a result that qualified the previous
catalog's cell is admitted for the new cell when the two differ only in
``anchor``. The rebinding proves this, and it is checked again here per cell.

A catalog may add several formats (PQ #1432). Each unit's added formats go,
sorted, before its terminal BF16 (``joint_catalog_extension.extended_roster``),
the order the loader and the pair check use. ``--qualified-dir`` is the default
result directory and ``--format-qualified-dir FORMAT=DIR`` overrides it for one
format. Intake accepts pair-keyed names or legacy qname-keyed names when just
one exists; both present is ambiguous and refuses unless ``--qualified-key``
explicitly selects ``pair`` or ``qname``. No result is renamed or rewritten.
A rebinding may bind this catalog, or a catalog this one declares in
``carried_from``; its rows admit the cells they name, and every other cell
needs a result that binds it directly. The published PWC holds the original
PWC's cells plus every catalog cell, and that count is derived, not declared.

The original plan and prepared are named by ``--original-plan`` and
``--original-prepared``, each with its SHA-256; a path without its digest, or
a digest without its path, refuses. With neither, the assembler binds
``OLDPLAN`` and ``OLDPREP``, the pair every overlay before R13 extended
(RobTand/prismaquant#1117). A digest mismatch refuses before anything is
published.
"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.digests import bytes_sha256hex
from prismaquant.joint_catalog_extension import _artifact_fence, extended_roster
from prismaquant.tessera_joint_aura import render_origin_census
from rebind_t4_qualified_results import SCHEMA as REBINDING_SCHEMA
from rebind_t4_qualified_results import cell_sha256, require_rebound, result_path

PANEL = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel')
OLDPLAN = PANEL / 'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02.plan.json'
OLDPREP = PANEL / 'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json'


sha = bytes_sha256hex


def doc(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def publish(path, raw):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    return {'path': str(path), 'sha256': sha(raw)}


def add_overlay_format(formats_by_qname, qname, formats):
    """Insert ``formats`` (one or several) before ``qname``'s terminal BF16, as the loader reads it.

    Several added formats go in sorted, once per unit.
    ``attach_candidate_overlay`` and ``verify_catalog_pair`` use the same
    order, and Stage B compares the prepared roster with the loaded one in
    order (RobTand/prismaquant#990).
    """
    formats_by_qname[qname] = list(extended_roster(formats_by_qname[qname], formats))


def add_overlay_formats(formats_by_qname, cells):
    """Insert every catalog cell's format into its unit's roster, one insertion per unit."""
    added = {}
    for cell in cells:
        added.setdefault(cell['qname'], []).append(cell['format'])
    for qname, formats in added.items():
        add_overlay_format(formats_by_qname, qname, formats)


def qualified_dirs(parser, default, overrides):
    """``format -> directory`` from repeated ``FORMAT=DIR`` flags; other formats use ``default``."""
    dirs = {}
    for value in overrides or ():
        fmt, sep, directory = value.partition('=')
        if not (sep and fmt and directory) or fmt in dirs:
            parser.error(f'--format-qualified-dir {value!r}: expected one FORMAT=DIR per format')
        dirs[fmt] = Path(directory)
    return lambda fmt: dirs.get(fmt, Path(default))


def rebinding_rows_for(rebinding, *, catalog_binding, catalog, qualified_dir):
    """The rebinding's rows by ``(qname, format)``, checked against this catalog.

    The rebinding binds this catalog, or one this catalog declares it carried
    cells from (``carried_from``). Each row names a cell of this catalog, and
    the result directory it was proven in is the one this run reads that
    cell's format from.
    """
    assert rebinding['schema'] == REBINDING_SCHEMA
    carried = catalog.get('carried_from', [])
    assert rebinding['catalog'] == catalog_binding or rebinding['catalog'] in carried, \
        'rebinding binds neither this catalog nor one it declares carried'
    rows = {(row['qname'], row['format']): row for row in rebinding['rows']}
    assert len(rows) == len(rebinding['rows'])
    pairs = {(cell['qname'], cell['format']) for cell in catalog['cells']}
    assert set(rows) <= pairs, 'rebinding names a cell this catalog does not carry'
    if rebinding['catalog'] == catalog_binding:
        assert len(rows) == len(catalog['cells']), 'a rebinding of this catalog covers every cell'
    for fmt in {fmt for _qname, fmt in rows}:
        assert Path(rebinding['qualified_dir']) == qualified_dir(fmt), (fmt, 'rebinding qualified_dir')
    return rows


def original_input(parser, path, digest, default, flag):
    """The bytes and binding of one original input, by flag or by default.

    The bytes are read once, so the binding describes what was assembled.
    """
    if (path is None) != (digest is None):
        parser.error(f'{flag} and {flag}-sha256 go together')
    path = default if path is None else Path(path)
    raw = path.read_bytes()
    binding = {'path': str(path), 'sha256': sha(raw)}
    if digest is not None and binding['sha256'] != digest:
        parser.error(f'{flag} {path}: SHA-256 is {binding["sha256"]}, not {digest}')
    return raw, binding


def bind_stage_b_resources(plan, prepared, policy_binding, resources):
    """Carry a Stage B resource policy onto the extended plan and prepared file.

    The policy may replace only the retained-window budget and the GPU byte
    ceiling (``joint_catalog_extension.CANDIDATE_PLAN_FIELDS``);
    ``verify_catalog_pair`` re-derives it and ``prepare_extended_joint_quanta``
    requires it.
    """
    plan['stage_b_resource_policy'] = policy_binding
    plan['execution']['retained_operator_windows']['budget'] = resources['budget']
    plan['max_gpu_bytes'] = resources['limits']['gpu_bytes']
    prepared['stage_b_resource_policy'] = policy_binding


def fence_cell_artifacts(cell):
    """Hold a catalog cell's wire and render to the stat recorded at catalog build.

    The catalog's own fence: a wire whose stat drifted at the same size (a later
    hardlink moves ctime) is re-hashed against its recorded blob, not refused.
    """
    for key in ('wire', 'render'):
        path = Path(cell[key])
        _artifact_fence(path, path.stat(), cell[key + '_stat'],
                        cell['record']['blob_sha256'] if key == 'wire' else None,
                        'assemble catalog ' + key + ' fence')


def qualified_result(cell, raw, rebinding_rows):
    """The result that qualified ``cell``, directly or through a rebinding row."""
    if rebinding_rows is None or (cell['qname'], cell['format']) not in rebinding_rows:
        value = json.loads(raw)
        assert value['cell_sha256'] == cell_sha256(cell), cell['qname']
        return value
    return require_rebound(cell, raw, rebinding_rows[cell['qname'], cell['format']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--served-activation-policy', required=True)
    parser.add_argument('--served-activation-policy-sha256', required=True)
    parser.add_argument('--stage-b-resource-policy', required=True)
    parser.add_argument('--stage-b-resource-policy-sha256', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--qualified-dir', required=True,
                        help='result directory for every format without its own')
    parser.add_argument('--qualified-key', choices=('auto', 'pair', 'qname'), default='auto',
                        help='input filenames: unique pair/qname match, or an explicit layout')
    parser.add_argument('--format-qualified-dir', action='append', metavar='FORMAT=DIR',
                        help='result directory for one added format (repeatable)')
    parser.add_argument('--rebinding')
    parser.add_argument('--rebinding-sha256')
    parser.add_argument('--original-plan', help='the plan the overlay extends (default: OLDPLAN)')
    parser.add_argument('--original-plan-sha256')
    parser.add_argument('--original-prepared', help='the prepared the overlay extends (default: OLDPREP)')
    parser.add_argument('--original-prepared-sha256')
    parser.add_argument('--out', required=True, help='overlay root; must not exist')
    args = parser.parse_args()
    planraw, planorigin = original_input(parser, args.original_plan, args.original_plan_sha256,
                                         OLDPLAN, '--original-plan')
    prepraw, preporigin = original_input(parser, args.original_prepared, args.original_prepared_sha256,
                                         OLDPREP, '--original-prepared')
    policy = {'path': args.served_activation_policy, 'sha256': args.served_activation_policy_sha256}
    assert sha(Path(policy['path']).read_bytes()) == policy['sha256']
    resource_policy = {'path': args.stage_b_resource_policy, 'sha256': args.stage_b_resource_policy_sha256}
    raw = Path(resource_policy['path']).read_bytes()
    assert sha(raw) == resource_policy['sha256']
    resources = json.loads(raw)
    catalogpath = Path(args.catalog)
    raw = catalogpath.read_bytes()
    assert sha(raw) == args.catalog_sha256
    catalog = json.loads(raw)
    qualified_dir = qualified_dirs(parser, args.qualified_dir, args.format_qualified_dir)
    rebinding_rows = None
    if args.rebinding is not None:
        assert args.rebinding_sha256, '--rebinding needs --rebinding-sha256'
        raw = Path(args.rebinding).read_bytes()
        assert sha(raw) == args.rebinding_sha256
        rebinding_rows = rebinding_rows_for(
            json.loads(raw), catalog_binding={'path': str(catalogpath), 'sha256': args.catalog_sha256},
            catalog=catalog, qualified_dir=qualified_dir)
    oldprep = json.loads(prepraw)
    plan = json.loads(planraw)
    del planraw, prepraw
    raw = Path(oldprep['production_cache']['path']).read_bytes()
    assert sha(raw) == oldprep['production_cache']['sha256']
    cache = pickle.loads(raw)
    del raw
    extended_cells = len(cache.weights) + len(catalog['cells'])
    overlay = Path(args.out)
    assert not overlay.exists(), 'immutable proposed overlay already exists'
    plan['inputs']['candidate_overlay'] = {'path': str(catalogpath), 'sha256': args.catalog_sha256}
    plan['output_root'] = str(overlay)
    plan['served_activation_policy'] = policy
    prepared = copy.deepcopy(oldprep)
    prepared['served_activation_policy'] = policy
    bind_stage_b_resources(plan, prepared, resource_policy, resources)
    for cell in catalog['cells']:
        q, fmt = cell['qname'], cell['format']
        pair = (q, fmt)
        assert pair not in cache.weights
        value = qualified_result(cell, result_path(qualified_dir(fmt), q, fmt,
                                                   key=args.qualified_key).read_bytes(), rebinding_rows)
        receipt = value['verified_cell']
        if 'verified_cell_sha256' in value:
            assert value['verified_cell_sha256'] == cell_sha256(receipt)
        for key in ('source_weight', 'activation', 'encoding_identity_sha256', 'render_origin',
                    'render_comparison', 'catalog_source_adoption'):
            assert receipt[key] == cell[key], (pair, key)
        fence_cell_artifacts(cell)
        assert receipt['render_file_sha256'] and receipt['rendered_weight']['content_sha256']
        cache.weights[pair] = cell['render']
        cache._lru_paths[pair] = cell['render']
        cache.metadata['verified_cells'][pair] = receipt
    add_overlay_formats(prepared['formats_by_qname'], catalog['cells'])
    assert len(cache.weights) == extended_cells, 'original PWC cells plus every catalog cell'
    planbinding = publish(overlay / 'plan.json', doc(plan))
    cache.metadata['inputs'] = plan['inputs']
    cache.metadata['plan_sha256'] = planbinding['sha256']
    census = render_origin_census(r['render_origin'] for r in cache.metadata['verified_cells'].values())
    cache.metadata.update(census)
    prepared.update(census)
    prepared['measured_cells'] = len(cache.weights)
    prepared['plan_sha256'] = planbinding['sha256']
    prepared['production_cache'] = publish(overlay / 'prepare/production.pkl',
                                           pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    prepbinding = publish(overlay / 'prepare/prepared.json', doc(prepared))
    inputs = {'original_plan': planorigin, 'original_prepared': preporigin,
              'extended_plan': planbinding, 'extended_prepared': prepbinding}
    publish(overlay / 'catalog-pair-inputs.json', doc(inputs))
    print(json.dumps({'status': 'proposed_candidate_metadata_only', 'inputs': inputs,
                      'cells': len(cache.weights)}), flush=True)


if __name__ == '__main__':
    main()
