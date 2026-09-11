"""Proof generator for the campaign identity re-seal (tools/reseal_campaign_identity.py).

The two source seals bound into every campaign checkpoint identity
(``prismaquant_source_sha256``, ``encoder_source_sha256``) are pins over
source trees, not measured values.  Amending them in already-written rows is
only honest when the candidate sources reproduce the stored bytes and scores.
This module produces that evidence, one arm per PB action:

``fixture-id``  (CPU, x86)  ``encoder_fixture_id()`` + per-fixture digests +
                the source seal, computed by each producer tree in its own
                process, so old and new can be compared in one record.
``prefix``      (GPU)  re-encode a fixed anchor prefix of a completed routed
                expert row under the candidate PQ + producer, into a fresh
                directory, then compare every produced cell against the
                stored row: wire bytes, receipt, anchor score, and the full
                checkpoint identity with only the two pins substituted.
``dense``       (GPU)  the same for a complete dense row, including the
                numeric content of ``cost.pkl``.
``compare``     (CPU)  the comparator alone, for re-verification on the host.

Every arm writes ``result.json`` under its ``--out``; the migration tool
assembles those into the proof bundle it requires before rewriting a row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 'prismaquant.reseal_identity_proof.v1'
PIN_KEYS = ('prismaquant_source_sha256', 'encoder_source_sha256')
ANCHOR_VOLATILE = ('seconds', 'encoding_batch_size')
STRIPPED_FLAGS = ('--seed-checkpoint', '--seed-wire-dir')


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def bind_prismaquant(root):
    """Put the candidate PQ tree first and drop the cwd entry ``-m`` adds.

    The checkout PB snapshots is the driver's home, not the code under test:
    ``python -m`` puts the cwd at ``sys.path[0]`` ahead of PYTHONPATH, so
    without this the campaign would import from the snapshot and the arm
    would silently measure the wrong tree.  The result records what was
    actually imported, and the comparator checks the identity's pin against
    the requested new pin, so a wrong binding cannot pass.
    """
    root = Path(root).resolve()
    if 'prismaquant' in sys.modules:
        raise RuntimeError('prismaquant was imported before the proof bound its tree')
    cwd = Path.cwd().resolve()
    sys.path[:] = [p for p in sys.path if p not in ('', '.') and Path(p or '.').resolve() != cwd]
    sys.path.insert(0, str(root))
    import prismaquant
    actual = Path(prismaquant.__file__).resolve().parent
    if actual != root/'prismaquant':
        raise RuntimeError(f'prismaquant bound to {actual}, wanted {root/"prismaquant"}')
    return prismaquant


def environment_record(*, with_pins):
    record = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  hostname=platform.node(), cwd=os.getcwd(),
                  env={k: os.environ.get(k) for k in ('PYTHONPATH', 'TESSERA_REPO', 'TESSERA_WINDOW_BEST_FORM',
                       'TESSERA_WINDOW_BEST_TILE', 'PRISMAQUANT_DETERMINISTIC', 'PRISMAQUANT_CONTAINER_CONTENT_SHA256',
                       'PRISMABUILD_ACTION_KEY', 'CUDA_VISIBLE_DEVICES')})
    if with_pins:
        import prismaquant
        from prismaquant.production_weight_cache import _production_cache_source_sha256
        import tessera
        from tessera import cached_unit
        from tessera.encoder_identity import encoder_fixture_id
        record.update(prismaquant_file=prismaquant.__file__, tessera_file=tessera.__file__,
                      prismaquant_source_sha256=_production_cache_source_sha256(),
                      encoder_source_sha256=cached_unit.encoder_source_sha256(),
                      encoder_fixture_id=encoder_fixture_id().hex())
        try:
            import torch
            record.update(torch=torch.__version__, cuda=torch.version.cuda,
                          device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        except Exception as error:  # pragma: no cover - torch absent on a CPU host
            record['torch'] = repr(error)
    return record


def sha256_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str)+'\n')
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# campaign command surgery
# ---------------------------------------------------------------------------

def redirect_outputs(command, out):
    """Point ``--out/--cache-dir/--checkpoint`` at a fresh directory; drop seeding."""
    out = Path(out)
    command = list(command)
    for flag in STRIPPED_FLAGS:
        while flag in command:
            index = command.index(flag)
            del command[index:index+2]
    for flag, value in {'--out': out/'cost.pkl', '--cache-dir': out/'cache',
                        '--checkpoint': out/'cost.anchors.json'}.items():
        if flag not in command:
            raise ValueError(f'campaign command lacks {flag}')
        command[command.index(flag)+1] = str(value)
    return command


def batch_class(batch):
    classes = {item[0].rsplit('.', 1)[-1] for item in batch}
    if classes <= {'down_proj'}:
        return 'down_proj'
    if classes <= {'gate_proj', 'up_proj'}:
        return 'gate_up'
    raise ValueError(f'batch crosses shape classes: {sorted(classes)}')


def shape_interleaved(batches, *, classes, per_class):
    """The first ``per_class`` batches of each shape class, then the rest.

    Membership of every batch is preserved; only whole batches move.  With
    the unit-major order of ``_anchor_batches`` the first three batches of a
    class are its first chunk at each of the three rung rates, so a prefix of
    ``per_class * batch_size`` anchors per class covers every rate.
    """
    chosen, rest, taken = [], [], {name: 0 for name in classes}
    for batch in batches:
        kind = batch_class(batch)
        if kind in taken and taken[kind] < per_class:
            taken[kind] += 1
            chosen.append(batch)
        else:
            rest.append(batch)
    short = [name for name, count in taken.items() if count < per_class]
    if short:
        raise ValueError(f'shape classes without enough batches: {short}')
    return chosen + rest


class _SilentObserver:
    """``run_prefix`` needs a result dict and a pass-through anchor wrapper."""

    def __init__(self):
        self.result = {}

    @staticmethod
    def wrap_anchor(function):
        return function


# ---------------------------------------------------------------------------
# comparator
# ---------------------------------------------------------------------------

def substitute_pins(identity, *, old, new):
    identity = json.loads(json.dumps(identity))
    for key in PIN_KEYS:
        if identity.get(key) != old[key]:
            raise ValueError(f'stored identity {key}={identity.get(key)} is not the declared old pin {old[key]}')
        identity[key] = new[key]
    return identity


def deep_equal(a, b, where='', diffs=None):
    """Exact structural equality; NaN equals NaN; arrays compared by bytes."""
    diffs = [] if diffs is None else diffs
    if type(a) is not type(b):
        try:
            import numpy as np
            if isinstance(a, np.generic) or isinstance(b, np.generic):
                return deep_equal(_plain(a), _plain(b), where, diffs)
        except ImportError:
            pass
        diffs.append((where, f'type {type(a).__name__} vs {type(b).__name__}'))
        return diffs
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((where, f'keys {sorted(set(a) ^ set(b))}'))
            return diffs
        for key in sorted(a, key=str):
            deep_equal(a[key], b[key], f'{where}.{key}', diffs)
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append((where, f'length {len(a)} vs {len(b)}'))
            return diffs
        for index, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, f'{where}[{index}]', diffs)
        return diffs
    if isinstance(a, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            diffs.append((where, f'{a!r} vs {b!r}'))
        return diffs
    if hasattr(a, 'tobytes') and hasattr(a, 'shape'):
        if tuple(a.shape) != tuple(b.shape) or str(a.dtype) != str(b.dtype) or a.tobytes() != b.tobytes():
            diffs.append((where, 'array content'))
        return diffs
    if a != b:
        diffs.append((where, f'{a!r} vs {b!r}'))
    return diffs


def _plain(value):
    return value.item() if hasattr(value, 'item') else value


def load_manifest(root):
    manifest = json.loads((Path(root)/'cost.anchors.json').read_text())
    for key in ('identity', 'identity_sha256', 'schema', 'stage', 'units'):
        if key not in manifest:
            raise ValueError(f'{root}: manifest lacks {key}')
    return manifest


def load_units(root, manifest, qnames):
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    parts = Path(root)/'cost.anchors.json.parts'
    return {name: _load_unit(unit_path(parts, name), stage=manifest['stage'], qname=name,
                             identity_sha256=manifest['identity_sha256']) for name in qnames}


def compare_rows(produced, stored, *, old, new, expected_cells=None, require_cost=False):
    """Compare a produced run against the stored row it re-encodes.

    Returns a result dict with ``ok`` and the cell table.  Nothing here is
    tolerant: a produced cell must match the stored one in every wire byte,
    every receipt field but the producer seal, and every anchor field but
    the two timing/batching fields the campaign itself does not compare.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256, canonical_json
    from prismaquant.production_weight_cache import first_identity_difference
    produced, stored = Path(produced), Path(stored)
    result = dict(schema=SCHEMA, kind='comparison', produced=str(produced), stored=str(stored),
                  old_pins=dict(old), new_pins=dict(new), failures=[], cells=[])
    fail = result['failures'].append
    pm, sm = load_manifest(produced), load_manifest(stored)
    expected_identity = substitute_pins(sm['identity'], old=old, new=new)
    difference = first_identity_difference(pm['identity'], canonical_json(expected_identity, where='expected identity'))
    if difference is not None:
        fail(dict(what='identity', field=difference[0], produced=str(difference[1])[:300], expected=str(difference[2])[:300]))
    for key in PIN_KEYS:
        if pm['identity'].get(key) != new[key]:
            fail(dict(what='identity_pin', field=key, produced=pm['identity'].get(key), expected=new[key]))
    expected_sha = canonical_json_sha256(canonical_json(expected_identity, where='expected identity'), where='expected identity')
    result.update(stored_identity_sha256=sm['identity_sha256'], produced_identity_sha256=pm['identity_sha256'],
                  expected_identity_sha256=expected_sha, identity_matches_with_pins_substituted=(pm['identity_sha256'] == expected_sha))
    if pm['identity_sha256'] != expected_sha:
        fail(dict(what='identity_sha256', produced=pm['identity_sha256'], expected=expected_sha))
    if pm['stage'] != sm['stage'] or pm['schema'] != sm['schema']:
        fail(dict(what='manifest', produced=[pm['schema'], pm['stage']], stored=[sm['schema'], sm['stage']]))

    parts = produced/'cost.anchors.json.parts'/'units'
    produced_names = [u['qname'] for u in pm['units'] if (parts/Path(u['file']).name).is_file()]
    if not produced_names:
        fail(dict(what='units', detail='produced run journaled no unit'))
    p_units = load_units(produced, pm, produced_names)
    s_units = load_units(stored, sm, produced_names)
    seen = set()
    for name in produced_names:
        p_state, s_state = p_units[name], s_units[name]
        if bool(p_state.get('unservable')) or bool(s_state.get('unservable')):
            fail(dict(what='unservable', unit=name, produced=p_state.get('unservable'), stored=s_state.get('unservable')))
            continue
        stored_anchors = {a['format_name']: a for a in s_state['anchors']}
        for anchor in p_state['anchors']:
            fmt = anchor['format_name']
            cell = dict(qname=name, format_name=fmt, family=anchor.get('family'), body_rate_q256=anchor.get('body_rate_q256'),
                        dloss=anchor.get('dloss'), ok=True, problems=[])
            problem = cell['problems'].append
            if (name, fmt) in seen:
                problem('duplicate cell')
            seen.add((name, fmt))
            s_anchor = stored_anchors.get(fmt)
            if s_anchor is None:
                problem('stored row has no anchor at this format')
            else:
                a = {k: v for k, v in anchor.items() if k not in ANCHOR_VOLATILE}
                b = {k: v for k, v in s_anchor.items() if k not in ANCHOR_VOLATILE}
                for where, detail in deep_equal(a, b, 'anchor'):
                    problem(f'{where}: {detail}')
                cell['stored_dloss'] = s_anchor.get('dloss')
            p_rec, s_rec = p_state['wire_records'].get(fmt), s_state['wire_records'].get(fmt)
            if p_rec is None or s_rec is None:
                problem('wire record missing on ' + ('produced' if p_rec is None else 'stored') + ' side')
            else:
                p_file, s_file = produced/'cache'/'wire'/p_rec['file'], stored/'cache'/'wire'/s_rec['file']
                p_bytes = p_file.read_bytes() if p_file.is_file() else None
                s_bytes = s_file.read_bytes() if s_file.is_file() else None
                cell.update(blob_bytes=p_rec['blob_bytes'], blob_sha256=p_rec['blob_sha256'],
                            stored_blob_sha256=s_rec['blob_sha256'], produced_file=str(p_file), stored_file=str(s_file))
                if p_bytes is None or s_bytes is None:
                    problem('wire file missing on ' + ('produced' if p_bytes is None else 'stored') + ' side')
                else:
                    cell['byte_identical'] = p_bytes == s_bytes
                    if not cell['byte_identical']:
                        problem('wire bytes differ')
                    for side, blob, rec in (('produced', p_bytes, p_rec), ('stored', s_bytes, s_rec)):
                        if hashlib.sha256(blob).hexdigest() != rec['blob_sha256'] or len(blob) != rec['blob_bytes']:
                            problem(f'{side} wire record does not describe its file')
                p_id, s_id = dict(p_rec['identity']), dict(s_rec['identity'])
                if s_id.get('encoder_source_sha256') != old['encoder_source_sha256']:
                    problem('stored receipt seal is not the declared old producer pin')
                if p_id.get('encoder_source_sha256') != new['encoder_source_sha256']:
                    problem('produced receipt seal is not the declared new producer pin')
                p_id.pop('encoder_source_sha256', None)
                s_id.pop('encoder_source_sha256', None)
                for where, detail in deep_equal(p_id, s_id, 'receipt'):
                    problem(f'{where}: {detail}')
                cell['encoder_fixture_id'] = p_rec['identity'].get('encoder_fixture_id')
                cell['stored_encoder_fixture_id'] = s_rec['identity'].get('encoder_fixture_id')
                if cell['encoder_fixture_id'] != cell['stored_encoder_fixture_id']:
                    problem('encoder_fixture_id moved')
            cell['ok'] = not cell['problems']
            result['cells'].append(cell)
    if expected_cells is not None and len(result['cells']) != expected_cells:
        fail(dict(what='cell_count', produced=len(result['cells']), expected=expected_cells))
    bad = [c for c in result['cells'] if not c['ok']]
    if bad:
        fail(dict(what='cells', failing=len(bad), sample=[dict(qname=c['qname'], format_name=c['format_name'], problems=c['problems'][:5]) for c in bad[:8]]))

    strata = {}
    for cell in result['cells']:
        key = f"{cell['family']}@R{cell['body_rate_q256']}:{'routed' if '.experts.' in cell['qname'] else 'dense'}"
        strata[key] = strata.get(key, 0) + 1
    result['strata'] = strata

    cost_path = produced/'cost.pkl'
    if cost_path.is_file():
        with cost_path.open('rb') as stream:
            p_cost = pickle.load(stream)
        with (stored/'cost.pkl').open('rb') as stream:
            s_cost = pickle.load(stream)
        report = dict(keys_compared=[], provenance_excluded=True)
        if set(p_cost) != set(s_cost):
            fail(dict(what='cost_keys', produced=sorted(p_cost), stored=sorted(s_cost)))
        for key in sorted(set(p_cost) & set(s_cost)):
            if key == 'provenance':
                continue
            a, b = p_cost[key], s_cost[key]
            if key == 'tessera_expert_wires':
                a, b = _strip_wire_seals(a), _strip_wire_seals(b)
            diffs = deep_equal(a, b, key)
            report['keys_compared'].append(key)
            if diffs:
                fail(dict(what='cost_content', key=key, sample=[f'{w}: {d}' for w, d in diffs[:8]]))
        result['cost_pkl'] = report
    elif require_cost:
        fail(dict(what='cost_pkl', detail='produced run wrote no cost.pkl'))
    result['ok'] = not result['failures']
    return result


def _strip_wire_seals(records):
    records = json.loads(json.dumps(records, default=str))
    for unit in records.values():
        for record in unit.values():
            if isinstance(record, dict) and isinstance(record.get('identity'), dict):
                record['identity'].pop('encoder_source_sha256', None)
    return records


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def _pins(args):
    return (dict(prismaquant_source_sha256=args.old_prismaquant_pin, encoder_source_sha256=args.old_encoder_pin),
            dict(prismaquant_source_sha256=args.new_prismaquant_pin, encoder_source_sha256=args.new_encoder_pin))


def _campaign_argv(args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('campaign argv required after --')
    if command[:1] == ['python3'] or command[:1] == ['python']:
        # roster argv carries the interpreter and -u -m prismaquant.tessera_campaign
        module = command.index('prismaquant.tessera_campaign')
        command = command[module+1:]
    return command


def run_gpu_arm(args, *, prefix):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = out/'run'
    if run.exists():
        raise RuntimeError(f'{run} exists; a proof arm never resumes into an existing run')
    bind_prismaquant(args.prismaquant_root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('the re-encode proof runs on the campaign GPU platform')
    old, new = _pins(args)
    command = redirect_outputs(_campaign_argv(args), run)
    started = time.time()
    record = dict(schema=SCHEMA, kind='prefix' if prefix else 'dense', out=str(out), run=str(run),
                  stored_row=str(args.stored_row), command=command, started_unix=started,
                  environment=environment_record(with_pins=True))
    for key, expected in (('prismaquant_source_sha256', new['prismaquant_source_sha256']),
                          ('encoder_source_sha256', new['encoder_source_sha256'])):
        if record['environment'][key] != expected:
            raise RuntimeError(f'running {key}={record["environment"][key]} is not the declared new pin {expected}')
    write_json(out/'result.json', dict(record, status='running'))
    from prismaquant import tessera_campaign as campaign
    if prefix:
        from experiments.campaign_prefix_profile import run_prefix
        classes = args.shape_classes.split(',')
        original = campaign._anchor_batches
        observer = _SilentObserver()
        campaign._anchor_batches = lambda *a, **kw: shape_interleaved(original(*a, **kw), classes=classes, per_class=args.batches_per_class)
        try:
            run_prefix(campaign, command, observer, limit=args.limit_anchors, expected_source_units=args.expected_source_units)
        finally:
            campaign._anchor_batches = original
        record['prefix'] = {k: v for k, v in observer.result.items() if k != 'resident_prefetch'}
        record['resident_prefetch'] = {k: v for k, v in observer.result.get('resident_prefetch', {}).items()
                                       if k in ('units', 'hessian_bytes', 'activation_bytes', 'devices', 'finished_unix')}
        expected_cells = args.limit_anchors
    else:
        campaign.main(command)
        expected_cells = args.expected_cells
    record['campaign_finished_unix'] = time.time()
    comparison = compare_rows(run, args.stored_row, old=old, new=new, expected_cells=expected_cells, require_cost=not prefix)
    record.update(comparison=comparison, ok=comparison['ok'], finished_unix=time.time(), status='finished')
    write_json(out/'result.json', record)
    print(json.dumps(dict(ok=record['ok'], cells=len(comparison['cells']), strata=comparison['strata'],
                          failures=comparison['failures'][:4], seconds=record['finished_unix']-started)), flush=True)
    return 0 if record['ok'] else 1


_FIXTURE_SNIPPET = r'''
import json, sys
import tessera
from tessera import cached_unit
from tessera.encoder_identity import encoder_fixture_id, fixture_digests
import torch
print(json.dumps(dict(tessera_file=tessera.__file__, torch=torch.__version__,
    encoder_source_sha256=cached_unit.encoder_source_sha256(),
    encoder_fixture_id=encoder_fixture_id().hex(), fixture_digests=fixture_digests())))
'''


def run_fixture_id(args):
    out = Path(args.out)
    record = dict(schema=SCHEMA, kind='fixture_id', started_unix=time.time(), producers=[],
                  environment=environment_record(with_pins=False),
                  window_env=dict(TESSERA_WINDOW_BEST_FORM=args.window_best_form, TESSERA_WINDOW_BEST_TILE=args.window_best_tile))
    for entry in args.producer:
        label, src = entry.split('=', 1)
        src = Path(src).resolve()
        if not (src/'tessera').is_dir():
            raise ValueError(f'{src} has no tessera package')
        env = dict(os.environ, PYTHONPATH=str(src), TESSERA_WINDOW_BEST_FORM=args.window_best_form,
                   TESSERA_WINDOW_BEST_TILE=args.window_best_tile, PYTHONDONTWRITEBYTECODE='1')
        env.pop('TESSERA_REPO', None)
        started = time.time()
        proc = subprocess.run([sys.executable, '-P', '-c', _FIXTURE_SNIPPET], env=env, capture_output=True, text=True, cwd=str(out))
        item = dict(label=label, src=str(src), returncode=proc.returncode, seconds=time.time()-started,
                    stderr_tail=proc.stderr[-4000:])
        if proc.returncode == 0:
            item.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            if not Path(item['tessera_file']).resolve().is_relative_to(src):
                raise RuntimeError(f'{label}: tessera imported from {item["tessera_file"]}, not {src}')
        record['producers'].append(item)
    ids = {p['label']: p.get('encoder_fixture_id') for p in record['producers']}
    seals = {p['label']: p.get('encoder_source_sha256') for p in record['producers']}
    record.update(encoder_fixture_ids=ids, encoder_source_sha256=seals,
                  fixture_id_equal=len(set(ids.values())) == 1 and None not in ids.values(),
                  ok=all(p['returncode'] == 0 for p in record['producers']) and len(record['producers']) >= 2,
                  finished_unix=time.time())
    if record['ok'] and not record['fixture_id_equal']:
        digests = [p['fixture_digests'] for p in record['producers']]
        record['moved_fixtures'] = sorted(k for k in digests[0] if any(d.get(k) != digests[0][k] for d in digests[1:]))
    write_json(out/'result.json', record)
    print(json.dumps(dict(ok=record['ok'], fixture_id_equal=record['fixture_id_equal'], ids=ids, seals=seals)), flush=True)
    return 0 if record['ok'] else 1


def run_compare(args):
    bind_prismaquant(args.prismaquant_root)
    old, new = _pins(args)
    result = compare_rows(args.produced, args.stored_row, old=old, new=new, expected_cells=args.expected_cells,
                          require_cost=args.require_cost)
    write_json(args.out, result)
    print(json.dumps(dict(ok=result['ok'], cells=len(result['cells']), strata=result['strata'], failures=result['failures'][:4])), flush=True)
    return 0 if result['ok'] else 1


def _add_pins(parser):
    for name in ('old-prismaquant-pin', 'old-encoder-pin', 'new-prismaquant-pin', 'new-encoder-pin'):
        parser.add_argument('--'+name, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='arm', required=True)
    for arm in ('prefix', 'dense'):
        p = sub.add_parser(arm)
        p.add_argument('--out', required=True)
        p.add_argument('--prismaquant-root', required=True, help='candidate PQ checkout (its prismaquant/ is hashed as the new pin)')
        p.add_argument('--stored-row', required=True)
        _add_pins(p)
        if arm == 'prefix':
            p.add_argument('--limit-anchors', type=int, required=True)
            p.add_argument('--expected-source-units', type=int, required=True)
            p.add_argument('--shape-classes', default='down_proj,gate_up')
            p.add_argument('--batches-per-class', type=int, default=3)
        else:
            p.add_argument('--expected-cells', type=int, required=True)
        p.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('fixture-id')
    p.add_argument('--out', required=True)
    p.add_argument('--producer', action='append', required=True, help='LABEL=/path/to/src (containing tessera/)')
    p.add_argument('--window-best-form', default='1')
    p.add_argument('--window-best-tile', default='64,4,2')
    p = sub.add_parser('compare')
    p.add_argument('--out', required=True)
    p.add_argument('--prismaquant-root', required=True)
    p.add_argument('--produced', required=True)
    p.add_argument('--stored-row', required=True)
    p.add_argument('--expected-cells', type=int)
    p.add_argument('--require-cost', action='store_true')
    _add_pins(p)
    args = parser.parse_args(argv)
    if args.arm == 'prefix':
        return run_gpu_arm(args, prefix=True)
    if args.arm == 'dense':
        return run_gpu_arm(args, prefix=False)
    if args.arm == 'fixture-id':
        return run_fixture_id(args)
    return run_compare(args)


if __name__ == '__main__':
    sys.exit(main())
