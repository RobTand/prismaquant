"""Describe immutable logical cells; PrismaBuild alone partitions them.

Prior qualifications are explicit, digest-bound inputs. No pilot, parent, model
or format is inferred, and only the new logical request is written.
"""
import argparse
import json
from pathlib import Path

from prismaquant.digests import DIRECT_ASCII_LAX, bytes_sha256hex, is_sha256hex
from prismaquant.joint_catalog_extension import catalog_view
from tools.rebind_t4_qualified_results import cell_task_id


def verified_json(path, expected):
    raw = Path(path).read_bytes()
    if not is_sha256hex(expected) or bytes_sha256hex(raw) != expected:
        raise ValueError(f'{path}: SHA256 mismatch')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='campaign root for new qualified/<pair-id>.json outputs')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--python', required=True, help='interpreter on the worker, pinned to the current Tessera')
    parser.add_argument('--qualifier-checkout', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--prior-result', nargs=2, action='append', default=[], metavar=('PATH', 'SHA256'),
                        help='verified per-cell qualification JSON, reused in place (repeatable)')
    parser.add_argument('--prior-child', nargs=2, action='append', default=[], metavar=('SHA256', 'PARENT_KEY'),
                        help='CAS child result manifest with pair task/output IDs (repeatable)')
    parser.add_argument('--cas-root', default='/mnt/shared/prismabuild-fleet/cas',
                        help='CAS containing prior child manifests in blobs/<prefix>/<digest>')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    catalog = verified_json(args.catalog, args.catalog_sha256)
    formats = catalog_view(catalog)['formats']
    cells = {}
    for cell in catalog['cells']:
        if not isinstance(cell.get('qname'), str) or not cell['qname']:
            raise ValueError('catalog cell has no qname')
        task_id = cell_task_id(cell)
        if task_id in cells:
            raise ValueError(f'duplicate catalog pair: {cell["qname"]}, {cell["format"]}')
        cells[task_id] = cell
    existing = {}
    prior_paths = set()

    def adopt_result(path, expected, *, task_id=None):
        path = Path(path).resolve()
        prior_paths.add(path)
        result = verified_json(path, expected)
        result_id = cell_task_id(result)
        if result_id not in cells or (task_id is not None and task_id != result_id):
            raise ValueError(f'{path}: result pair is not the declared catalog task')
        if result.get('cell_sha256') != DIRECT_ASCII_LAX.sha256(cells[result_id]):
            raise ValueError(f'{path}: result cell SHA256 differs from catalog')
        if (not isinstance(result.get('verified_cell'), dict)
                or result.get('verified_cell_sha256') != DIRECT_ASCII_LAX.sha256(result['verified_cell'])):
            raise ValueError(f'{path}: verified cell SHA256 mismatch')
        if result_id in existing and existing[result_id][1] != expected:
            raise ValueError(f'{path}: conflicting prior result for {result_id}')
        existing.setdefault(result_id, (str(path), expected))

    for path, expected in args.prior_result:
        adopt_result(path, expected)
    for digest, parent_key in args.prior_child:
        if not is_sha256hex(digest) or not is_sha256hex(parent_key):
            raise ValueError('prior child requires SHA256 and full parent key')
        path = (Path(args.cas_root) / 'blobs' / digest[:2] / digest).resolve()
        prior_paths.add(path)
        child = verified_json(path, digest)
        if child.get('schema') != 'prismabuild.child_result_manifest.v1' or child.get('parent_key') != parent_key:
            raise ValueError(f'{path}: child schema or parent key mismatch')
        for row in child['results']:
            task_id = row['task_id']
            if task_id not in cells or row.get('output_id') != task_id:
                raise ValueError(f'{path}: child requires rostered pair task/output IDs')
            adopt_result(root / 'qualified' / (task_id + '.json'), row['value_sha256'], task_id=task_id)

    tasks = []
    output_owners = {}
    for task_id, cell in cells.items():
        payload = {'cell': cell, 'output': str(root / 'qualified' / (task_id + '.json')), 'reads': []}
        if task_id in existing:
            payload['output'], payload['existing_result_sha256'] = existing[task_id]
        else:
            for kind in ('wire', 'render'):
                payload['reads'].append({
                    'path': str(Path(cell[kind]).relative_to('/mnt/shared')), 'offset': 0,
                    'bytes': cell[kind + '_stat']['bytes'],
                    'sha256': cell['record']['blob_sha256'] if kind == 'wire' else None})
        output_path = Path(payload['output']).resolve()
        if output_path in output_owners:
            raise ValueError(f'output path {output_path} aliases distinct catalog pairs: '
                             f'{output_owners[output_path]}, {task_id}')
        output_owners[output_path] = task_id
        payload['output'] = str(output_path)
        tasks.append({
            'id': task_id, 'payload': payload, 'residency_key': cell['format'],
            'estimated_seconds': 0.001 if task_id in existing else 0.6,
            'estimate_evidence': 'Planning estimate only: historical PB strict pilot parent74d1ca9cdf1c, '
                                 'warm per-cell 0.51-0.84s with two overlapped readers; '
                                 'verified prior results metadata-only; not a measurement of this catalog',
            'output_id': task_id})
    fresh_temporaries = set()
    for task in tasks:
        payload = task['payload']
        if 'existing_result_sha256' in payload:
            continue
        # Mirror qualify_t4_overlay.commit_task: the qualification output's suffix is
        # replaced with .tmp, not appended, before its wb write and rename.
        temporary = Path(payload['output']).with_suffix('.tmp')
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f'temporary output path {temporary} is occupied; refusing overwrite')
        temporary = temporary.resolve()
        if temporary in prior_paths or temporary in output_owners or temporary in fresh_temporaries:
            raise ValueError(f'temporary output path {temporary} aliases a prior input or task destination')
        fresh_temporaries.add(temporary)
    request = {
        'schema': 'prismabuild.logical_request.v1',
        'common': {
            'argv': [args.python, 'tools/qualify_t4_overlay.py', '--pb-task-batch', '{pb.task_batch}',
                     '--allowed-tiers', 'ram,ssd'],
            'cwd': args.qualifier_checkout, 'demand': {'cpu': 3, 'mem_gb': 4, 'gpu': 1},
            'gpu_memory_gb': 2, 'data_manifest': None,
            'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'PYTHONPATH': '.', 'T4_QUALIFY_READ_WORKERS': '2'},
            'tags': ['gb10'], 'timeout_s': 600},
        'roster': {'schema': 'prismabuild.logical_task_roster.v1', 'tasks': tasks},
        'batch_policy': {
            'schema': 'prismabuild.roster_batch_policy.v1',
            'residencies': [
                {'key': fmt, 'setup_seconds': 9,
                 'setup_evidence': 'Historical PB pilot24 wall15s minus ~10.5s percell work plus cold '
                                   'firstcell4.4s; conservative startup allowance, not throughput '
                                   'measurement or a measurement of this catalog'} for fmt in formats],
            'max_setup_fraction': 0.1, 'max_estimated_wall_seconds': 120},
        'task_data_manifest': {
            'schema': 'prismabuild.task_data_manifest.v1', 'payload_field': 'reads',
            'mount_prefix': '/mnt/shared', 'residency_tier': None, 'residency_ram': 'off',
            'mover_readers': 2, 'mover_mem_gb': 1}}
    out = Path(args.out)
    raw = DIRECT_ASCII_LAX.encoded(request) + b'\n'
    with out.open('xb') as handle:
        handle.write(raw)
    print(json.dumps({'path': str(out), 'sha256': bytes_sha256hex(raw), 'tasks': len(tasks),
                      'reused': len(existing), 'status': 'ready_for_verified_PB_client62047'}))


if __name__ == '__main__':
    main()
