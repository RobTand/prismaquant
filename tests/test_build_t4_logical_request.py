"""Model-independent qualification requests (PQ #1503), using synthetic files only."""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from prismaquant.digests import DIRECT_ASCII_LAX, bytes_sha256hex

SPEC = importlib.util.spec_from_file_location(
    "build_t4_logical_request", Path(__file__).parents[1] / "tools/build_t4_logical_request.py")
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def identity(cell):
    return DIRECT_ASCII_LAX.sha256([cell['qname'], cell['format']])


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = DIRECT_ASCII_LAX.encoded(value) + b'\n'
    path.write_bytes(raw)
    return bytes_sha256hex(raw)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    cells: list[dict[str, Any]] = [{'qname': 'model.layer.unit', 'format': fmt,
                  'wire': f'/mnt/shared/synthetic/{fmt}/wire',
                  'render': f'/mnt/shared/synthetic/{fmt}/render',
                  'wire_stat': {'bytes': 12}, 'render_stat': {'bytes': 24},
                  'record': {'blob_sha256': 'a' * 64}}
             for fmt in ('TESSERA_E2M1_K2_R896', 'TESSERA_E4M3_K1_R1024')]
    catalog = {'schema': 'prismaquant.t4_adopted_catalog.v2', 'cells': cells,
               'formats': sorted(c['format'] for c in cells),
               'sources': [{'cost': {'path': 'synthetic', 'sha256': 'b' * 64}}],
               'cell_sources': [0, 0]}
    root = tmp_path / 'new-model'
    root.mkdir()
    catalog_path = tmp_path / 'catalog.json'
    digest = put(catalog_path, catalog)
    out = tmp_path / 'request.json'

    def run(*extra):
        monkeypatch.setattr(sys, 'argv', ['builder', '--root', str(root),
                            '--catalog', str(catalog_path), '--catalog-sha256', digest,
                            '--python', '/pinned/python', '--qualifier-checkout', '/qualifier',
                            '--out', str(out), *map(str, extra)])
        m.main()
        return json.loads(out.read_bytes())

    return cells, catalog, root, catalog_path, out, run


def qualified(path, cell, **updates):
    verified = {'qualification_seconds': 0.1}
    return put(path, {'qname': cell['qname'], 'format': cell['format'],
                      'cell_sha256': DIRECT_ASCII_LAX.sha256(cell),
                      'verified_cell': verified,
                      'verified_cell_sha256': DIRECT_ASCII_LAX.sha256(verified), **updates})


def child(cas, cells, digests, **updates):
    value = {'schema': 'prismabuild.child_result_manifest.v1', 'parent_key': 'c' * 64,
             'plan_key': 'd' * 64, 'child_ordinal': 0,
             'results': [{'task_id': identity(c), 'output_id': identity(c),
                          'value_sha256': d} for c, d in zip(cells, digests, strict=True)], **updates}
    digest = bytes_sha256hex(DIRECT_ASCII_LAX.encoded(value) + b'\n')
    put(cas / 'blobs' / digest[:2] / digest, value)
    return digest


def test_fresh_multiformat_request_has_pair_ids_outputs_and_format_residencies(campaign):
    cells, _, _, _, _, run = campaign
    request = run()
    tasks = request['roster']['tasks']
    assert [t['id'] for t in tasks] == [identity(c) for c in cells]
    assert [t['output_id'] for t in tasks] == [t['id'] for t in tasks]
    assert len({t['payload']['output'] for t in tasks}) == 2
    assert {t['residency_key'] for t in tasks} == {c['format'] for c in cells}
    assert {r['key'] for r in request['batch_policy']['residencies']} == {c['format'] for c in cells}
    assert all(len(t['payload']['reads']) == 2 for t in tasks)
    assert tasks[0]['payload']['reads'][0] == {
        'path': cells[0]['wire'].removeprefix('/mnt/shared/'), 'offset': 0,
        'bytes': 12, 'sha256': 'a' * 64}
    assert request['schema'] == 'prismabuild.logical_request.v1'
    assert request['roster']['schema'] == 'prismabuild.logical_task_roster.v1'
    assert request['common'] == {
        'argv': ['/pinned/python', 'tools/qualify_t4_overlay.py', '--pb-task-batch',
                 '{pb.task_batch}', '--allowed-tiers', 'ram,ssd'],
        'cwd': '/qualifier', 'demand': {'cpu': 3, 'mem_gb': 4, 'gpu': 1},
        'gpu_memory_gb': 2, 'tags': ['gb10'], 'timeout_s': 600, 'data_manifest': None,
        'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'PYTHONPATH': '.', 'T4_QUALIFY_READ_WORKERS': '2'}}
    assert request['batch_policy']['schema'] == 'prismabuild.roster_batch_policy.v1'
    assert request['batch_policy']['max_setup_fraction'] == 0.1
    assert request['batch_policy']['max_estimated_wall_seconds'] == 120
    assert all(r['setup_seconds'] == 9 for r in request['batch_policy']['residencies'])
    assert request['task_data_manifest'] == {
        'schema': 'prismabuild.task_data_manifest.v1', 'payload_field': 'reads',
        'mount_prefix': '/mnt/shared', 'residency_tier': None, 'residency_ram': 'off',
        'mover_readers': 2, 'mover_mem_gb': 1}


def test_verified_explicit_legacy_filename_reuses_only_its_format_and_never_writes(campaign):
    cells, _, root, _, _, run = campaign
    path = root / 'old-qualified.json'
    digest = qualified(path, cells[0])
    before = path.read_bytes()
    request = run('--prior-result', path, digest)
    reused, fresh = request['roster']['tasks']
    assert reused['payload']['output'] == str(path)
    assert reused['payload']['existing_result_sha256'] == digest
    assert reused['payload']['reads'] == []
    assert len(fresh['payload']['reads']) == 2
    assert path.read_bytes() == before
    assert not (root / 'qualified').exists()


def test_verified_cas_child_with_arbitrary_parent_and_count_reuses_pair(campaign, tmp_path):
    cells, _, root, _, _, run = campaign
    path = root / 'qualified' / (identity(cells[0]) + '.json')
    digest = qualified(path, cells[0])
    cas = tmp_path / 'cas'
    key = child(cas, cells[:1], [digest])
    request = run('--cas-root', cas, '--prior-child', key, 'c' * 64)
    assert request['roster']['tasks'][0]['payload']['existing_result_sha256'] == digest
    assert len(request['roster']['tasks'][1]['payload']['reads']) == 2


@pytest.mark.parametrize('defect', ['manifest_digest', 'parent', 'task_id', 'output_id',
                                  'result_digest', 'result_cell', 'missing_result', 'schema'])
def test_bad_cas_child_refuses_before_request_publication(campaign, tmp_path, defect):
    cells, _, root, _, out, run = campaign
    path = root / 'qualified' / (identity(cells[0]) + '.json')
    digest = qualified(path, cells[0])
    cas = tmp_path / 'cas'
    updates = {}
    if defect in ('task_id', 'output_id'):
        row = {'task_id': identity(cells[0]), 'output_id': identity(cells[0]),
               'value_sha256': digest}
        row[defect] = cells[0]['qname']  # qname-only legacy manifests are ambiguous.
        updates['results'] = [row]
    if defect == 'schema':
        updates['schema'] = 'unexpected'
    key = child(cas, cells[:1], [digest], **updates)
    if defect == 'manifest_digest':
        (cas / 'blobs' / key[:2] / key).write_bytes(b'changed')
    if defect == 'result_digest':
        path.write_bytes(b'changed')
    if defect == 'result_cell':
        digest = qualified(path, cells[0], cell_sha256='0' * 64)
        key = child(cas, cells[:1], [digest])
    if defect == 'missing_result':
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        run('--cas-root', cas, '--prior-child', key, 'e' * 64 if defect == 'parent' else 'c' * 64)
    assert not out.exists()


@pytest.mark.parametrize('defect', ['digest', 'cell', 'unknown_pair', 'verified_cell'])
def test_bad_explicit_result_refuses(campaign, defect):
    cells, _, root, _, out, run = campaign
    path = root / 'result.json'
    updates = {'cell_sha256': '0' * 64} if defect == 'cell' else {}
    if defect == 'unknown_pair':
        updates['format'] = 'UNROSTERED'
    if defect == 'verified_cell':
        updates['verified_cell_sha256'] = '0' * 64
    digest = qualified(path, cells[0], **updates)
    with pytest.raises(ValueError):
        run('--prior-result', path, '0' * 64 if defect == 'digest' else digest)
    assert not out.exists()


def test_conflicting_prior_records_refuse(campaign):
    cells, _, root, _, out, run = campaign
    one, two = root / 'one.json', root / 'two.json'
    first = qualified(one, cells[0], qualification_seconds=1)
    second = qualified(two, cells[0], qualification_seconds=2)
    with pytest.raises(ValueError, match='conflict'):
        run('--prior-result', one, first, '--prior-result', two, second)
    assert not out.exists()


def test_catalog_digest_refuses_before_publication(campaign):
    _, _, _, path, out, run = campaign
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        run()
    assert not out.exists()


def test_duplicate_catalog_pair_refuses(campaign):
    cells, catalog, _, path, out, run = campaign
    catalog['cells'] = [cells[0], cells[0]]
    catalog['formats'] = [cells[0]['format']]
    digest = put(path, catalog)
    # Override the fixture's original digest on argv (last argparse value wins).
    with pytest.raises(ValueError, match='duplicate'):
        run('--catalog-sha256', digest)
    assert not out.exists()


def test_v1_catalog_derives_residency_from_its_format(campaign):
    cells, catalog, _, path, _, run = campaign
    catalog.update(schema='prismaquant.t4_adopted_catalog.v1', cells=cells[:1],
                   format=cells[0]['format'], cost={'path': 'synthetic'})
    digest = put(path, catalog)
    request = run('--catalog-sha256', digest)
    assert request['roster']['tasks'][0]['residency_key'] == cells[0]['format']
    assert [r['key'] for r in request['batch_policy']['residencies']] == [cells[0]['format']]


def test_explicit_files_and_multiple_cas_children_combine_without_mutations(campaign, tmp_path):
    cells, _, root, _, _, run = campaign
    cas = tmp_path / 'cas'
    paths = [root / 'qualified' / (identity(c) + '.json') for c in cells]
    digests = [qualified(p, c) for p, c in zip(paths, cells, strict=True)]
    keys = [child(cas, [c], [d]) for c, d in zip(cells, digests, strict=True)]
    before = {p: p.read_bytes() for p in paths}
    request = run('--prior-result', paths[0], digests[0], '--cas-root', cas,
                  '--prior-child', keys[0], 'c' * 64, '--prior-child', keys[1], 'c' * 64)
    assert all(t['payload']['reads'] == [] for t in request['roster']['tasks'])
    assert {p: p.read_bytes() for p in paths} == before


def test_unrequested_legacy_pilot_is_not_read(campaign):
    _, _, root, _, _, run = campaign
    path = root / 'pilot24-results.json'
    path.write_bytes(b'invalid JSON, not a declared input')
    request = run()
    assert all(len(t['payload']['reads']) == 2 for t in request['roster']['tasks'])
    assert path.read_bytes() == b'invalid JSON, not a declared input'


@pytest.mark.parametrize('alias', ['direct', 'dot_segments', 'symlink'])
def test_prior_result_cannot_alias_another_pairs_fresh_output(campaign, alias):
    cells, _, root, _, out, run = campaign
    fresh_output = root / 'qualified' / (identity(cells[1]) + '.json')
    digest = qualified(fresh_output, cells[0])
    path = fresh_output
    if alias == 'dot_segments':
        path = root / 'qualified' / '..' / 'qualified' / fresh_output.name
    elif alias == 'symlink':
        path = root / 'alias.json'
        path.symlink_to(fresh_output)
    before = fresh_output.read_bytes()
    with pytest.raises(ValueError, match='output path'):
        run('--prior-result', path, digest)
    assert not out.exists()
    assert fresh_output.read_bytes() == before


def test_relative_prior_result_is_bound_to_builder_cwd_not_worker_cwd(campaign, monkeypatch):
    cells, _, root, _, _, run = campaign
    path = root / 'old-qualified.json'
    digest = qualified(path, cells[0])
    monkeypatch.chdir(root)
    request = run('--prior-result', path.name, digest)
    payload = request['roster']['tasks'][0]['payload']
    assert payload['output'] == str(path.resolve())
    worker = root / 'independent-worker-cwd'
    worker.mkdir()
    monkeypatch.chdir(worker)
    assert bytes_sha256hex(Path(payload['output']).read_bytes()) == digest


@pytest.mark.parametrize('missing', ['verified_cell', 'verified_cell_sha256', 'cell_sha256'])
def test_pilot_era_unbound_qualification_content_is_not_admitted(campaign, missing):
    cells, _, root, _, out, run = campaign
    path = root / 'pilot-era.json'
    qualified(path, cells[0])
    value = json.loads(path.read_bytes())
    del value[missing]
    digest = put(path, value)
    with pytest.raises(ValueError):
        run('--prior-result', path, digest)
    assert not out.exists()


@pytest.mark.parametrize('alias', ['direct', 'dot_segments', 'symlink'])
@pytest.mark.parametrize('duplicate_unused', [False, True])
def test_fresh_temporary_cannot_alias_any_declared_prior_result(campaign, alias, duplicate_unused):
    cells, _, root, _, out, run = campaign
    temporary = (root / 'qualified' / (identity(cells[1]) + '.json')).with_suffix('.tmp')
    digest = qualified(temporary, cells[0])
    path = temporary
    if alias == 'dot_segments':
        path = root / 'qualified' / '..' / 'qualified' / temporary.name
    elif alias == 'symlink':
        path = root / 'prior-alias.json'
        path.symlink_to(temporary)
    extra = []
    originals = {temporary: temporary.read_bytes()}
    if duplicate_unused:
        selected = root / 'selected-prior.json'
        assert qualified(selected, cells[0]) == digest
        originals[selected] = selected.read_bytes()
        # This second, equal-digest input is ignored by setdefault for task A,
        # but its file must still be protected from fresh task B's temporary.
        extra = ['--prior-result', selected, digest]
    with pytest.raises(ValueError, match='temporary output path'):
        run(*extra, '--prior-result', path, digest)
    assert not out.exists()
    assert all(p.exists() and p.read_bytes() == raw for p, raw in originals.items())


@pytest.mark.parametrize('target_kind', ['task_output', 'fresh_temporary'])
def test_fresh_temporary_cannot_alias_other_destinations(campaign, target_kind):
    cells, _, root, _, out, run = campaign
    outputs = [root / 'qualified' / (identity(c) + '.json') for c in cells]
    outputs[0].parent.mkdir()
    target = outputs[0] if target_kind == 'task_output' else outputs[0].with_suffix('.tmp')
    target.write_bytes(b'synthetic destination; must not be touched')
    originals = target.read_bytes()
    outputs[1].with_suffix('.tmp').symlink_to(target)
    with pytest.raises(ValueError, match='temporary output path'):
        run()
    assert not out.exists()
    assert target.read_bytes() == originals


def test_fresh_temporary_cannot_alias_declared_cas_child_manifest(campaign, tmp_path):
    cells, _, root, _, out, run = campaign
    prior = root / 'qualified' / (identity(cells[0]) + '.json')
    result_digest = qualified(prior, cells[0])
    cas = tmp_path / 'synthetic-cas'
    key = child(cas, cells[:1], [result_digest])
    manifest = cas / 'blobs' / key[:2] / key
    temporary = (root / 'qualified' / (identity(cells[1]) + '.json')).with_suffix('.tmp')
    temporary.symlink_to(manifest)
    originals = {prior: prior.read_bytes(), manifest: manifest.read_bytes()}
    with pytest.raises(ValueError, match='temporary output path'):
        run('--cas-root', cas, '--prior-child', key, 'c' * 64)
    assert not out.exists()
    assert all(p.exists() and p.read_bytes() == raw for p, raw in originals.items())


def test_fresh_temporary_hardlink_to_prior_is_refused_without_mutation(campaign):
    cells, _, root, _, out, run = campaign
    prior = root / 'retained-prior.json'
    digest = qualified(prior, cells[0])
    temporary = (root / 'qualified' / (identity(cells[1]) + '.json')).with_suffix('.tmp')
    temporary.parent.mkdir()
    temporary.hardlink_to(prior)
    originals = prior.read_bytes()
    assert temporary.resolve() != prior.resolve()
    assert temporary.stat().st_ino == prior.stat().st_ino
    with pytest.raises(ValueError, match='temporary output path'):
        run('--prior-result', prior, digest)
    assert not out.exists()
    assert prior.read_bytes() == temporary.read_bytes() == originals


@pytest.mark.parametrize('occupied', ['file', 'directory', 'broken_symlink'])
def test_occupied_fresh_temporary_is_refused(campaign, occupied):
    cells, _, root, _, out, run = campaign
    temporary = (root / 'qualified' / (identity(cells[0]) + '.json')).with_suffix('.tmp')
    temporary.parent.mkdir()
    if occupied == 'file':
        temporary.write_bytes(b'synthetic stale temporary')
    elif occupied == 'directory':
        temporary.mkdir()
    else:
        temporary.symlink_to(root / 'missing-target.json')
    with pytest.raises(ValueError, match='temporary output path'):
        run()
    assert not out.exists()
    if occupied == 'file':
        assert temporary.read_bytes() == b'synthetic stale temporary'
    elif occupied == 'directory':
        assert temporary.is_dir() and list(temporary.iterdir()) == []
    else:
        assert temporary.is_symlink() and temporary.readlink() == root / 'missing-target.json'
        assert not (root / 'missing-target.json').exists()


def test_unoccupied_fresh_temporaries_cannot_share_normalized_destination(campaign):
    cells, _, root, _, out, run = campaign
    outputs = [root / 'qualified' / (identity(c) + '.json') for c in cells]
    outputs[0].parent.mkdir()
    targets = [root / 'same-stem.json', root / 'same-stem.result']
    for output, target in zip(outputs, targets, strict=True):
        output.symlink_to(target)
    assert not (root / 'same-stem.tmp').exists()
    with pytest.raises(ValueError, match='temporary output path'):
        run()
    assert not out.exists()
    assert not (root / 'same-stem.tmp').exists()
    assert all(p.is_symlink() and p.readlink() == t for p, t in zip(outputs, targets, strict=True))


def test_unoccupied_fresh_temporary_cannot_alias_normalized_task_output(campaign):
    cells, _, root, _, out, run = campaign
    outputs = [root / 'qualified' / (identity(c) + '.json') for c in cells]
    outputs[0].parent.mkdir()
    other_temporary = outputs[1].with_suffix('.tmp')
    outputs[0].symlink_to(other_temporary)
    assert not other_temporary.exists()
    with pytest.raises(ValueError, match='temporary output path'):
        run()
    assert not out.exists()
    assert outputs[0].is_symlink() and outputs[0].readlink() == other_temporary
    assert not other_temporary.exists()


def test_reused_tasks_do_not_reserve_unused_temporary_destinations(campaign):
    cells, _, root, _, _, run = campaign
    own_temporary_name = (root / 'qualified' / (identity(cells[0]) + '.json')).with_suffix('.tmp')
    digest = qualified(own_temporary_name, cells[0])
    originals = own_temporary_name.read_bytes()
    request = run('--prior-result', own_temporary_name, digest)
    assert request['roster']['tasks'][0]['payload']['existing_result_sha256'] == digest
    assert len(request['roster']['tasks'][1]['payload']['reads']) == 2
    assert own_temporary_name.read_bytes() == originals


def test_request_output_is_exclusive(campaign):
    _, _, _, _, out, run = campaign
    out.write_bytes(b'previous request')
    with pytest.raises(FileExistsError):
        run()
    assert out.read_bytes() == b'previous request'
