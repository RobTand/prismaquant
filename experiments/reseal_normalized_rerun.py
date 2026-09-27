"""Offline re-compare of G2 identity-proof arms with the PQ #1520 settings set aside.

The v2 prefix arms matched every cell but failed the checkpoint identity on
``settings.source_identity_cache``: the pq-g2e tree binds #1520's two
``--source-identity-cache`` flags into every identity it writes, and the
stored rows predate them.  This re-runs ``reseal_identity_proof.py compare``
on each finished arm's own produced journal with ``--produced-only-setting``
for exactly those two names, in a fresh process per arm.  It reads the pins,
the stored row and the produced run from the arm's ``result.json``; it never
re-encodes anything.

``controls`` runs the negative controls on one arm, in process, against the
same ``compare_rows``:

* baseline: the unmodified normalized comparison (must pass);
* flip: one other produced settings key changed in the loaded identity (must
  refuse, naming that key);
* stored_binds: the stored identity carrying ``source_identity_cache`` (must
  refuse before comparing);
* undeclared: an extra name outside ``PRODUCED_ONLY_SETTINGS`` (must refuse).

Run it with this checkout as the working directory; its ``prismaquant/`` is
the tree the comparison imports.  Every output is created, never replaced.

usage:
  reseal_normalized_rerun.py arms OUTDIR ARM...
  reseal_normalized_rerun.py controls OUTDIR ARM
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROOF = Path('/mnt/shared/tessera-measurements/claude-ldlq-perf-20260927/g2-identity-proof')
PQ_1520 = ('source_identity_cache', 'source_identity_cache_sha256')
EXPECTED_CELLS = 128
HARNESS = Path(__file__).resolve().parent/'reseal_identity_proof.py'


def _arm(name):
    result = json.loads((PROOF/'arms-v2'/name/'result.json').read_text())
    if result.get('status') != 'finished':
        raise ValueError(f'{name}: arm status is {result.get("status")!r}, not finished')
    comparison = result['comparison']
    return result, comparison['produced'], result['stored_row'], comparison['old_pins'], comparison['new_pins']


def _create(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True, default=str) + '\n')


def run_arms(outdir, names):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        _result, produced, stored, old, new = _arm(name)
        out = outdir/f'{name}.compare.json'
        if out.exists():
            raise FileExistsError(out)
        argv = [sys.executable, str(HARNESS), 'compare', '--out', str(out),
                '--prismaquant-root', str(Path.cwd()), '--produced', produced, '--stored-row', stored,
                '--expected-cells', str(EXPECTED_CELLS), '--prefix',
                '--old-prismaquant-pin', old['prismaquant_source_sha256'],
                '--old-encoder-pin', old['encoder_source_sha256'],
                '--new-prismaquant-pin', new['prismaquant_source_sha256'],
                '--new-encoder-pin', new['encoder_source_sha256']]
        for setting in PQ_1520:
            argv += ['--produced-only-setting', setting]
        started = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True)
        item = dict(argv=argv, returncode=proc.returncode, seconds=round(time.time() - started, 1),
                    stdout_tail=proc.stdout[-2000:], stderr_tail=proc.stderr[-4000:])
        if out.exists():
            got = json.loads(out.read_text())
            item.update(ok=got['ok'], cells=len(got['cells']),
                        cells_ok=sum(1 for cell in got['cells'] if cell['ok']),
                        byte_identical=sum(1 for cell in got['cells'] if cell.get('byte_identical')),
                        strata=got.get('strata'), failures=got['failures'][:6],
                        expected_identity_sha256=got.get('expected_identity_sha256'),
                        normalized_produced_identity_sha256=got.get('normalized_produced_identity_sha256'),
                        identity_matches_after_normalization=got.get('identity_matches_after_normalization'),
                        produced_only_settings=got.get('produced_only_settings'))
        summary[name] = item
        print(json.dumps({name: {k: item.get(k) for k in ('returncode', 'ok', 'cells', 'cells_ok',
                                                            'identity_matches_after_normalization')}}), flush=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    _create(outdir/f'summary.{stamp}.json', summary)
    return 0 if all(item.get('ok') is True and item.get('cells_ok') == EXPECTED_CELLS
                    and item.get('identity_matches_after_normalization') is True
                    for item in summary.values()) else 1


def _flipped(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + 'x'
    return 1 if value is None else [value]


def run_controls(outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir/f'controls.{name}.json'
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(Path.cwd()))
    import experiments.reseal_identity_proof as proof
    proof.bind_prismaquant(Path.cwd())
    _result, produced, stored, old, new = _arm(name)
    original_load = proof.load_manifest
    record = dict(arm=name, produced=produced, stored=stored, controls={})

    def compare(names=PQ_1520):
        return proof.compare_rows(produced, stored, old=old, new=new, expected_cells=EXPECTED_CELLS,
                                  prefix=True, produced_only_settings=names)

    def summarize(result):
        return dict(ok=result['ok'], cells=len(result['cells']),
                    cells_ok=sum(1 for cell in result['cells'] if cell['ok']),
                    failures=result['failures'][:6],
                    identity_matches_after_normalization=result.get('identity_matches_after_normalization'),
                    normalized_produced_identity_sha256=result.get('normalized_produced_identity_sha256'),
                    expected_identity_sha256=result.get('expected_identity_sha256'))

    def refusal(call):
        try:
            result = call()
        except ValueError as error:
            return dict(refused=True, error=str(error))
        return dict(refused=False, **summarize(result))

    baseline = compare()
    record['controls']['baseline'] = dict(summarize(baseline), passed=baseline['ok'] is True)

    produced_settings = original_load(produced, 'stream')['identity']['settings']
    flip_key = 'nsamples' if 'nsamples' in produced_settings else sorted(
        k for k in produced_settings if k not in PQ_1520)[0]
    flip_to = _flipped(produced_settings[flip_key])

    def flipped_load(root, journal='parts'):
        manifest = original_load(root, journal)
        if Path(root) == Path(produced):
            manifest['identity']['settings'][flip_key] = flip_to
        return manifest
    proof.load_manifest = flipped_load
    try:
        flipped = compare()
    finally:
        proof.load_manifest = original_load
    fields = [f.get('field') for f in flipped['failures'] if f['what'] == 'identity']
    record['controls']['flip'] = dict(summarize(flipped), key=flip_key,
                                      value=produced_settings[flip_key], flipped_to=flip_to,
                                      passed=flipped['ok'] is False and fields == [f'settings.{flip_key}'])

    def stored_binds_load(root, journal='parts'):
        manifest = original_load(root, journal)
        if Path(root) == Path(stored):
            manifest['identity']['settings']['source_identity_cache'] = None
        return manifest
    proof.load_manifest = stored_binds_load
    try:
        outcome = refusal(compare)
    finally:
        proof.load_manifest = original_load
    record['controls']['stored_binds'] = dict(outcome, passed=outcome['refused'] is True
                                              and 'stored identity binds setting(s)' in outcome['error'])

    outcome = refusal(lambda: compare(PQ_1520 + (flip_key,)))
    record['controls']['undeclared'] = dict(outcome, key=flip_key, passed=outcome['refused'] is True
                                            and 'are not declared produced-only settings' in outcome['error'])

    record['passed'] = all(item['passed'] for item in record['controls'].values())
    _create(out, record)
    print(json.dumps({key: item['passed'] for key, item in record['controls'].items()}), flush=True)
    return 0 if record['passed'] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[0] not in ('arms', 'controls'):
        raise SystemExit(__doc__)
    outdir = Path(argv[1])
    if argv[0] == 'arms':
        return run_arms(outdir, argv[2:])
    if len(argv) != 3:
        raise SystemExit('controls takes exactly one arm')
    return run_controls(outdir, argv[2])


if __name__ == '__main__':
    sys.exit(main())
