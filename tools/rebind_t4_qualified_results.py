"""Rebind the qualified T4 results to a catalog that changed only its anchors.

Each qualification result binds the digest of the catalog cell it qualified
(``cell_sha256``). Rebuilding the catalog with journal anchors changed every
cell's ``anchor`` and nothing else, and ``qualify_t4_overlay.py`` never reads
the anchor, so the verdicts still hold. This tool proves that per cell and
writes one rebinding document. It does not modify any result file.

For each ``(qname, format)``, the new cell must equal the previous cell in every
field except ``anchor``. The result file must bind the previous cell's digest
and its own ``verified_cell`` digest. Each rebinding row records the result
file's digest, both cell digests and the previous anchor. With those,
``assemble_t4_overlay.py`` rebuilds the previous cell from the new one and
checks the result's digest without reading the previous catalog.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = 'prismaquant.t4_qualified_rebinding.v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cell_sha256(cell):
    return sha(json.dumps(cell, sort_keys=True, separators=(',', ':')).encode())


def result_path(qualified_dir, qname):
    return Path(qualified_dir) / (sha(qname.encode()) + '.json')


def bound_json(path, expected):
    raw = Path(path).read_bytes()
    assert sha(raw) == expected, (path, 'SHA256 mismatch')
    return json.loads(raw)


def rebind_cell(new, previous, raw_result):
    """One rebinding row, or an AssertionError naming what does not hold."""
    pair = (new['qname'], new['format'])
    assert (previous['qname'], previous['format']) == pair, pair
    assert set(new) == set(previous), (pair, 'cell fields differ')
    changed = sorted(key for key in new if new[key] != previous[key])
    assert changed in ([], ['anchor']), (pair, 'fields other than the anchor changed', changed)
    value = json.loads(raw_result)
    assert (value['qname'], value['format']) == pair, pair
    previous_sha = cell_sha256(previous)
    assert value['cell_sha256'] == previous_sha, (pair, 'result does not bind the previous cell')
    if 'verified_cell_sha256' in value:  # absent on the 37 pilot-era results
        assert value['verified_cell_sha256'] == cell_sha256(value['verified_cell']), (pair, 'verified_cell digest')
    return {'qname': pair[0], 'format': pair[1], 'result_sha256': sha(raw_result),
            'previous_cell_sha256': previous_sha, 'cell_sha256': cell_sha256(new),
            'previous_anchor': previous['anchor']}


def require_rebound(cell, raw_result, row):
    """The assembler's check: ``raw_result`` qualified ``cell`` up to its anchor."""
    assert (row['qname'], row['format']) == (cell['qname'], cell['format']), row['qname']
    assert sha(raw_result) == row['result_sha256'], (row['qname'], 'result file changed')
    assert cell_sha256(cell) == row['cell_sha256'], (row['qname'], 'cell changed')
    previous = {**cell, 'anchor': row['previous_anchor']}
    assert cell_sha256(previous) == row['previous_cell_sha256'], (row['qname'], 'previous cell')
    value = json.loads(raw_result)
    assert value['cell_sha256'] == row['previous_cell_sha256'], (row['qname'], 'result binding')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--previous-catalog', required=True)
    parser.add_argument('--previous-catalog-sha256', required=True)
    parser.add_argument('--qualified-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists(), out
    catalog = bound_json(args.catalog, args.catalog_sha256)
    previous = bound_json(args.previous_catalog, args.previous_catalog_sha256)
    previous_cells = {(cell['qname'], cell['format']): cell for cell in previous['cells']}
    assert len(previous_cells) == len(previous['cells']) == len(catalog['cells'])
    for key in set(catalog) | set(previous):
        if key not in ('cells', 'anchor_journal'):
            assert catalog.get(key) == previous.get(key), ('catalog field changed', key)
    rows = []
    for cell in catalog['cells']:
        raw = result_path(args.qualified_dir, cell['qname']).read_bytes()
        rows.append(rebind_cell(cell, previous_cells[cell['qname'], cell['format']], raw))
    document = {'schema': SCHEMA,
                'catalog': {'path': args.catalog, 'sha256': args.catalog_sha256},
                'previous_catalog': {'path': args.previous_catalog, 'sha256': args.previous_catalog_sha256},
                'qualified_dir': str(Path(args.qualified_dir)),
                'scope': 'each new cell equals its previous cell except anchor; the qualifier never reads the anchor',
                'rows': rows}
    raw = (json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(raw)
    print(json.dumps({'path': str(out), 'sha256': sha(raw), 'rows': len(rows),
                      'anchor_changed': sum(r['cell_sha256'] != r['previous_cell_sha256'] for r in rows)}),
          flush=True)


if __name__ == '__main__':
    main()
