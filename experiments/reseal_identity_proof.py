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
import functools
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = 'prismaquant.reseal_identity_proof.v1'
PIN_KEYS = ('prismaquant_source_sha256', 'encoder_source_sha256')
ANCHOR_VOLATILE = ('seconds', 'encoding_batch_size')
# Wall-clock and batching fields the campaign stores beside each priced rung;
# not scores, and never equal across two runs of the same encode.
COST_VOLATILE = ('encode_seconds', 'encode_seconds_accounting', 'encoding_batch_size')
STRIPPED_FLAGS = ('--seed-checkpoint', '--seed-wire-dir')
# The driver checkout: the tree this file was loaded from, which PB snapshots
# and the container mounts at the working directory. Its ``experiments``
# helpers drive the arm; the PrismaQuant tree under test is --prismaquant-root.
DRIVER_EXPERIMENTS = Path(__file__).resolve().parent
# The tessera#486 fused encoder paths, and what each call says about them:
# a refusal function returns None when the fused path admits the call, a fused
# function returns None when its tripwire hands the call back to the reference.
FUSED_PROBES = (
    ('tessera.lut_fused', 'lut_swap_refusal', 'refusal'),
    ('tessera.lut_fused', 'swap_passes_fused', 'fused'),
    ('tessera.encode', '_lut_swap_passes_reference', 'reference'),
    ('tessera.tcq_fused', 'tcq_fused_refusal', 'refusal'),
    ('tessera.tcq_fused', 'viterbi_columns_fused', 'fused'),
)
FUSED_ENV = ('TESSERA_LUT_FUSED', 'TESSERA_TCQ_FUSED', 'TESSERA_TCQ_GRAPH')
# The stage-2 counters Tessera keeps itself: fused fits, tripwire fallbacks, nonfinite fallbacks.
FUSED_STATS = ('tessera.lut_fused', 'STATS')
# The encode entry points PrismaQuant calls on ``tessera.export`` (tessera_render
# looks each up on the module at call time); the engagement summary is scoped to
# the window from the first call's entry to the last call's exit.
FUSED_ENCODE = ('tessera.export', ('encode_linear', 'encode_linears', 'encode_linear_planes', 'encode_linears_planes'))


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


def pin_driver_experiments():
    """Make ``experiments.*`` resolve from the driver checkout, whatever ``sys.path`` holds.

    ``experiments`` has no ``__init__.py``, so it is a namespace package whose
    ``__path__`` is recomputed from ``sys.path`` on each import.
    ``bind_prismaquant`` drops the driver checkout (the cwd) from ``sys.path``,
    after which ``experiments.campaign_prefix_profile`` would be looked up in
    the bound PrismaQuant root: the code under test, which need not carry the
    helper (PQ 9753a5b7c5 does not, and both prefix arms on it failed with
    ModuleNotFoundError). Pinning ``__path__`` moves only ``experiments``;
    ``prismaquant`` still resolves from the bound root alone.
    """
    package = sys.modules.get('experiments')
    if package is None:
        spec = importlib.machinery.ModuleSpec('experiments', None, is_package=True)
        spec.submodule_search_locations = [str(DRIVER_EXPERIMENTS)]
        package = importlib.util.module_from_spec(spec)
        sys.modules['experiments'] = package
    package.__path__ = [str(DRIVER_EXPERIMENTS)]
    return package


def driver_modules():
    """Every loaded ``experiments`` module with a file; refuse one outside the driver checkout."""
    loaded = {name: Path(module.__file__).resolve() for name, module in list(sys.modules.items())
              if name.startswith('experiments.') and getattr(module, '__file__', None)}
    outside = {name: str(path) for name, path in loaded.items() if path.parent != DRIVER_EXPERIMENTS}
    if outside:
        raise RuntimeError(f'experiments modules loaded from outside the driver checkout {DRIVER_EXPERIMENTS}: {outside}')
    return {name: dict(file=str(path), sha256=sha256_file(path)) for name, path in sorted(loaded.items())}


def import_prefix_helper():
    """``experiments.campaign_prefix_profile`` from the driver checkout, with the modules it loaded."""
    pin_driver_experiments()
    helper = importlib.import_module('experiments.campaign_prefix_profile')
    return helper, driver_modules()


def _count_difference(after, before):
    """Per-probe outcome counts ``after - before``, keeping only outcomes that moved."""
    moved = {}
    for key, labels in after.items():
        diff = {label: n - before.get(key, {}).get(label, 0) for label, n in labels.items()}
        diff = {label: n for label, n in diff.items() if n}
        if diff:
            moved[key] = diff
    return moved


class FusedEngagement:
    """Count what the tessera#486 fused encoder stages did inside this arm's encodes.

    Stage 1 is the fused TCQ trellis, stage 2 the fused LUT swap passes. An arm
    whose cells never took a fused stage proves nothing about the fused kernels.
    Stage 1 has no Tessera counter, so each probe wraps one module attribute
    and counts its outcomes: a refusal gate by reason (``admitted`` when it
    returned None, integers in a reason masked so one reason is one class), a
    fused function by ``ran`` or ``fell_back``, and the reference swap passes by
    calls. The encoder imports the gates and fused functions from their modules
    at call time, so the wrappers see every call; they return what they wrap
    and read nothing from the tensors. Stage 2 also keeps
    ``tessera.lut_fused.STATS`` (``fused``, ``tripped``, ``nonfinite``).

    Fits that never reach a gate run the reference silently and nothing counts
    them: a CPU matrix (``encoder_fixture_id``), ``swaps == 0``, a replaced
    ``_lut_cost``, ``TESSERA_LUT_FUSED=0``. So the summary is scoped to the
    encode window: the probe counts and STATS are read on entry to the first
    call of a ``tessera.export`` encode entry point and on exit from the last
    one, and the summary is their difference. A call nested in another encode
    call does not open a window of its own. A module the producer does not
    have is recorded as absent.
    """

    MAX_REASONS = 32
    MAX_VERBATIM = 8

    def __init__(self, probes=FUSED_PROBES, stats=FUSED_STATS, encode=FUSED_ENCODE):
        self.probes = tuple(probes)
        self.stats_source = stats
        self.encode_source = encode
        self.installed = {}
        self.counts = {}
        self.verbatim = {}
        self.windows = []
        self.first = None
        self.last = None
        self._depth = 0
        self._originals = []
        self._lock = threading.RLock()

    def _stats(self):
        if self.stats_source is None:
            return None
        module_name, attr = self.stats_source
        try:
            stats = getattr(importlib.import_module(module_name), attr, None)
        except ImportError:
            return None
        return None if not isinstance(stats, dict) else {k: int(v) for k, v in stats.items()}

    def _snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self.counts)), self._stats()

    def _patch(self, module_name, attr, role, wrap):
        key = f'{module_name}.{attr}'
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            self.installed[key] = f'absent: {error}'
            return
        original = getattr(module, attr, None)
        if original is None:
            self.installed[key] = 'absent: no such attribute'
            return
        setattr(module, attr, wrap(key, original))
        self._originals.append((module, attr, original))
        self.installed[key] = role

    def install(self):
        for module_name, attr, role in self.probes:
            self._patch(module_name, attr, role, functools.partial(self._wrap, role=role))
        if self.encode_source is not None:
            module_name, entries = self.encode_source
            for attr in entries:
                self._patch(module_name, attr, 'encode', self._window)
        return self

    def _window(self, key, original):
        def encode(*args, **kwargs):
            with self._lock:
                self._depth += 1
                outermost = self._depth == 1
                if outermost and self.first is None:
                    self.first = self._snapshot()
            started = time.time()
            try:
                return original(*args, **kwargs)
            finally:
                with self._lock:
                    self._depth -= 1
                    if outermost:
                        batch = args[0] if args else None
                        units = len(batch) if isinstance(batch, (list, tuple)) else 1
                        self.windows.append(dict(entry=key, start_unix=started, end_unix=time.time(), units=units))
                        self.last = self._snapshot()
        return functools.wraps(original)(encode)

    def uninstall(self):
        for module, attr, original in reversed(self._originals):
            setattr(module, attr, original)
        self._originals.clear()

    def _count(self, key, label):
        with self._lock:
            entry = self.counts.setdefault(key, {})
            if label not in entry and len(entry) >= self.MAX_REASONS:
                label = 'other'
            entry[label] = entry.get(label, 0) + 1

    def _wrap(self, key, original, *, role):
        def probe(*args, **kwargs):
            value = original(*args, **kwargs)
            if role == 'refusal':
                if value is None:
                    label = 'admitted'
                else:
                    reason = str(value)[:400]
                    label = re.sub(r'\d+', '<n>', reason)[:200]
                    with self._lock:
                        seen = self.verbatim.setdefault(key, {})
                        if reason in seen or len(seen) < self.MAX_VERBATIM:
                            seen[reason] = seen.get(reason, 0) + 1
            elif role == 'fused':
                label = 'fell_back' if value is None else 'ran'
            else:
                label = 'calls'
            self._count(key, label)
            return value
        return functools.wraps(original)(probe)

    def record(self):
        with self._lock:
            counts = json.loads(json.dumps(self.counts))
            verbatim = json.loads(json.dumps(self.verbatim))
            windows = [dict(window) for window in self.windows]
            first, last = self.first, self.last
        scoped = first is not None and last is not None
        in_window = _count_difference(last[0], first[0]) if scoped else {}
        before, after = (first[1], last[1]) if scoped else (None, None)
        delta = None if before is None or after is None else {
            k: after.get(k, 0) - before.get(k, 0) for k in sorted(set(before) | set(after))}

        def outcome(name, label):
            return in_window.get(name, {}).get(label, 0)

        def refusals(name):
            return {k: v for k, v in in_window.get(name, {}).items() if k != 'admitted'}
        tcq_gate, lut_gate = 'tessera.tcq_fused.tcq_fused_refusal', 'tessera.lut_fused.lut_swap_refusal'
        lut_fused = 'tessera.lut_fused.swap_passes_fused'
        summary = dict(
            encode_windows=len(windows), encode_units=sum(window['units'] for window in windows),
            stage1_admitted=outcome(tcq_gate, 'admitted'), stage1_refusals=refusals(tcq_gate),
            stage1_fused=outcome('tessera.tcq_fused.viterbi_columns_fused', 'ran'),
            stage2_admitted=outcome(lut_gate, 'admitted'), stage2_refusals=refusals(lut_gate),
            stage2_fused=None if delta is None else delta.get('fused', 0),
            stage2_tripped=None if delta is None else delta.get('tripped', 0),
            stage2_nonfinite=None if delta is None else delta.get('nonfinite', 0),
            lut_reference_calls=outcome('tessera.encode._lut_swap_passes_reference', 'calls'))
        summary['stage1_refused'] = sum(summary['stage1_refusals'].values())
        summary['stage2_refused'] = sum(summary['stage2_refusals'].values())
        # An admitted trellis call runs the fused trellis; the probe on
        # swap_passes_fused and STATS count the same fits: one ``fused`` per
        # run, one ``tripped`` or ``nonfinite`` per fallback.
        summary['stage1_counts_agree'] = summary['stage1_admitted'] == summary['stage1_fused']
        summary['stage2_counts_agree'] = None if delta is None else (
            outcome(lut_fused, 'ran') == delta.get('fused', 0)
            and outcome(lut_fused, 'fell_back') == delta.get('tripped', 0) + delta.get('nonfinite', 0))
        summary['both_stages'] = bool(summary['stage1_fused'] and summary['stage2_fused'])
        summary['engaged'] = bool(summary['stage1_fused'] or summary['stage2_fused'])
        return dict(probes=dict(self.installed), counts=counts, counts_in_encode_window=in_window,
                    refusals_verbatim=verbatim, encode_windows=windows,
                    stats=dict(source='.'.join(self.stats_source) if self.stats_source else None,
                               before_first_encode=before, after_last_encode=after, delta=delta),
                    summary=summary, env={name: os.environ.get(name) for name in FUSED_ENV})


def environment_record(*, with_pins):
    record = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  hostname=platform.node(), cwd=os.getcwd(),
                  env={k: os.environ.get(k) for k in ('PYTHONPATH', 'TESSERA_REPO', 'TESSERA_WINDOW_BEST_FORM',
                       'TESSERA_WINDOW_BEST_TILE', 'TESSERA_SEAL_PREFETCH', 'PRISMAQUANT_DETERMINISTIC',
                       'PRISMAQUANT_CONTAINER_CONTENT_SHA256', *FUSED_ENV,
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

    Membership of every batch is preserved; only whole batches move.  The
    rates those batches carry follow the row's round-one rates, in the
    unit-major order of ``_anchor_batches``: on a banded row the first batches
    of a class move through the band ends, and on a single-rate row
    (``--rate-band 896,896``) they are successive member chunks at that one
    rate.  Whether a prefix reaches every class depends on the limit as well;
    ``require_prefix_classes`` checks that before anything is encoded.
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


def require_prefix_classes(batches, *, classes, limit):
    """Refuse an order whose first ``limit`` anchors encode no batch of a requested class.

    ``run_prefix`` stops after ``limit`` anchors.  At three batches per class,
    a 16-anchor prefix of a single-rate row is two down_proj batches and never
    reaches gate/up, so the arm would pass while proving one of the two shape
    classes it was asked for.
    """
    seen, done = set(), 0
    for batch in batches:
        if done + len(batch) > limit:
            break
        seen.add(batch_class(batch))
        done += len(batch)
    missing = [name for name in classes if name not in seen]
    if missing:
        raise ValueError(f'a {limit}-anchor prefix encodes no {missing} batch; '
                         'raise --limit-anchors or lower --batches-per-class')
    return batches


def restrict_groups(groups, *, classes, per_class):
    """Keep the first ``per_class`` members of each shape class in every group.

    The campaign then prices only those members: round one places the band
    ends and round two the interior rate (the bisection reaches R960 only
    after every member of the group has both ends), so a restricted group
    reaches the third rate in 3 x members anchors instead of 3 x 864.  The
    group's rate grid is the intersection over its members, and the stored
    rows measured every member at the same three rates, so the subset's
    grid is the same one.  The comparator, not this permutation, judges
    whether the rates and scores agree with the stored row.  The arm stops at
    the first leave-one-out gate after the prefix, which a ``--max-rounds 1``
    single-rate row never reaches; use ``--batches-per-class`` there.
    """
    kept = {}
    for key, members in groups.items():
        taken = {name: 0 for name in classes}
        chosen = []
        for member in members:
            kind = batch_class([(member,)])
            if kind in taken and taken[kind] < per_class:
                taken[kind] += 1
                chosen.append(member)
        if chosen:
            kept[key] = chosen
    if not kept:
        raise ValueError('no anchor group has members in the requested shape classes')
    return kept


def _prefix_done(observer, limit):
    calls = observer.result.get('prefix_calls') or []
    return sum(len(call['qnames']) for call in calls) >= limit


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

def substitute_pins(identity, *, old, new, drop=()):
    """The identity the migration will write: new pins, dropped settings gone.

    ``drop`` is the pins file's ``drop_settings``: scheduling knobs the new
    source no longer binds. ``reseal_campaign_identity.migrate`` pops exactly
    these from ``identity['settings']``, so the proof has to pop them too --
    otherwise it compares the produced row against an identity the migration
    is not going to produce, and a correct run fails on a field neither side
    disputes. A name that is not bound is refused rather than ignored, so a
    typo in the pins file cannot quietly weaken the comparison.
    """
    identity = json.loads(json.dumps(identity))
    for key in PIN_KEYS:
        if identity.get(key) != old[key]:
            raise ValueError(f'stored identity {key}={identity.get(key)} is not the declared old pin {old[key]}')
        identity[key] = new[key]
    settings = identity.get('settings') or {}
    missing = [key for key in drop if key not in settings]
    if missing:
        raise ValueError(f'stored identity binds no setting(s) {missing}, so they cannot be dropped')
    for key in drop:
        settings.pop(key)
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


def compare_rows(produced, stored, *, old, new, expected_cells=None, require_cost=False,
                 drop_settings=()):
    """Compare a produced run against the stored row it re-encodes.

    Returns a result dict with ``ok`` and the cell table.  Nothing here is
    tolerant: a produced cell must match the stored one in every wire byte,
    every receipt field but the producer seal, and every anchor field but
    the two timing/batching fields the campaign itself does not compare.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256, canonical_json
    from prismaquant.production_weight_cache import first_identity_difference
    produced, stored = Path(produced), Path(stored)
    drop_settings = tuple(drop_settings)
    result = dict(schema=SCHEMA, kind='comparison', produced=str(produced), stored=str(stored),
                  old_pins=dict(old), new_pins=dict(new), dropped_settings=list(drop_settings),
                  failures=[], cells=[])
    fail = result['failures'].append
    pm, sm = load_manifest(produced), load_manifest(stored)
    expected_identity = substitute_pins(sm['identity'], old=old, new=new, drop=drop_settings)
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
                # The recipe the encoder recorded for this cell says which fused
                # stages its encode could reach (window/TCQ body, CHANNEL/LUT16 plane).
                recipe = p_rec['identity'].get('recipe') or {}
                cell.update(recipe_body=recipe.get('body'), recipe_plane=recipe.get('plane'))
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
        report = dict(keys_compared=[], provenance_excluded=True, cost_fields_masked=list(COST_VOLATILE))
        if set(p_cost) != set(s_cost):
            fail(dict(what='cost_keys', produced=sorted(p_cost), stored=sorted(s_cost)))
        for key in sorted(set(p_cost) & set(s_cost)):
            if key == 'provenance':
                continue
            a, b = p_cost[key], s_cost[key]
            if key == 'tessera_expert_wires':
                a, b = _strip_wire_seals(a), _strip_wire_seals(b)
            elif key == 'costs':
                a, b = _mask_cost_timing(a), _mask_cost_timing(b)
            diffs = deep_equal(a, b, key)
            report['keys_compared'].append(key)
            if diffs:
                fail(dict(what='cost_content', key=key, sample=[f'{w}: {d}' for w, d in diffs[:8]]))
        result['cost_pkl'] = report
    elif require_cost:
        fail(dict(what='cost_pkl', detail='produced run wrote no cost.pkl'))
    result['ok'] = not result['failures']
    return result


def _mask_cost_timing(costs):
    return {unit: {fmt: {k: v for k, v in entry.items() if k not in COST_VOLATILE}
                   for fmt, entry in rungs.items()} for unit, rungs in costs.items()}


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
    # The prefix helper is driver code: import it now, from the driver checkout,
    # so a tree that cannot drive the arm fails before anything is measured.
    helper, driver = import_prefix_helper() if prefix else (None, driver_modules())
    old, new = _pins(args)
    command = redirect_outputs(_campaign_argv(args), run)
    started = time.time()
    record = dict(schema=SCHEMA, kind='prefix' if prefix else 'dense', out=str(out), run=str(run),
                  stored_row=str(args.stored_row), command=command, started_unix=started,
                  environment=environment_record(with_pins=True),
                  driver=dict(experiments=str(DRIVER_EXPERIMENTS), modules=driver))
    for key, expected in (('prismaquant_source_sha256', new['prismaquant_source_sha256']),
                          ('encoder_source_sha256', new['encoder_source_sha256'])):
        if record['environment'][key] != expected:
            raise RuntimeError(f'running {key}={record["environment"][key]} is not the declared new pin {expected}')
    write_json(out/'result.json', dict(record, status='running'))
    from prismaquant import tessera_campaign as campaign
    # Counted from here: the campaign's own encodes, not the pins check above.
    engagement = FusedEngagement().install()
    try:
        if prefix:
            run_prefix, PrefixComplete = helper.run_prefix, helper.PrefixComplete
            classes = args.shape_classes.split(',')
            original = campaign._anchor_batches
            original_groups = campaign.resolve_anchor_groups
            original_loo = campaign._loo_for
            observer = _SilentObserver()
            if args.members_per_class:
                # Restricted groups; the campaign's own batching order applies.
                # The first leave-one-out evaluation after the requested prefix
                # is the third round's gate on the restricted members; stop
                # there, before the campaign finalizes a table for units it
                # never measured.
                # ``select_anchor_groups`` resolves the same groups to check the
                # --units selection covers every member; only the pricing loop's
                # resolution (tessera_campaign._main, after selection) is
                # restricted, so the selection gate still sees the whole stack.
                # Both resolutions happen in tessera_campaign._main: the scope
                # groups first (which the --units selection is checked against,
                # member for member) and the pricing groups second (which the
                # round loop pends anchors from).  Only the second is restricted.
                calls = []

                def restricted_groups(*a, **kw):
                    groups = original_groups(*a, **kw)
                    calls.append(len(groups))
                    if len(calls) == 1:
                        return groups
                    return restrict_groups(groups, classes=classes, per_class=args.members_per_class)
                campaign.resolve_anchor_groups = restricted_groups

                def stop_after_prefix(*a, **kw):
                    if observer.result.get('completed_anchor_units', 0) >= args.limit_anchors or _prefix_done(observer, args.limit_anchors):
                        raise PrefixComplete()
                    return original_loo(*a, **kw)
                campaign._loo_for = stop_after_prefix
                record['group_restriction'] = dict(classes=classes, members_per_class=args.members_per_class, resolutions=calls)
            else:
                checked = []

                def interleaved(*a, **kw):
                    order = shape_interleaved(original(*a, **kw), classes=classes, per_class=args.batches_per_class)
                    if not checked:
                        # The campaign calls this once per round with every pending
                        # anchor; the first call is round one, before any encode.
                        require_prefix_classes(order, classes=classes, limit=args.limit_anchors)
                        checked.append(True)
                    return order
                campaign._anchor_batches = interleaved
            try:
                run_prefix(campaign, command, observer, limit=args.limit_anchors, expected_source_units=args.expected_source_units)
            finally:
                campaign._anchor_batches = original
                campaign.resolve_anchor_groups = original_groups
                campaign._loo_for = original_loo
            record['prefix'] = {k: v for k, v in observer.result.items() if k != 'resident_prefetch'}
            record['resident_prefetch'] = {k: v for k, v in observer.result.get('resident_prefetch', {}).items()
                                           if k in ('units', 'hessian_bytes', 'activation_bytes', 'devices', 'finished_unix')}
            expected_cells = args.limit_anchors
        else:
            campaign.main(command)
            expected_cells = args.expected_cells
    except BaseException as error:
        # A campaign that raises leaves a record saying so, with whatever the
        # fused paths did before it stopped, instead of a result left 'running'.
        record.update(status='failed', error=repr(error)[:4000], failed_unix=time.time(),
                      fused_engagement=engagement.record())
        write_json(out/'result.json', record)
        raise
    finally:
        engagement.uninstall()
    record['fused_engagement'] = engagement.record()
    record['campaign_finished_unix'] = time.time()
    comparison = compare_rows(run, args.stored_row, old=old, new=new, expected_cells=expected_cells,
                              require_cost=not prefix, drop_settings=args.drop_setting)
    record.update(comparison=comparison, ok=comparison['ok'], finished_unix=time.time(), status='finished')
    write_json(out/'result.json', record)
    print(json.dumps(dict(ok=record['ok'], cells=len(comparison['cells']), strata=comparison['strata'],
                          failures=comparison['failures'][:4], seconds=record['finished_unix']-started,
                          fused=record['fused_engagement']['summary'])), flush=True)
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
    out.mkdir(parents=True, exist_ok=True)
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
                          require_cost=args.require_cost, drop_settings=args.drop_setting)
    write_json(args.out, result)
    print(json.dumps(dict(ok=result['ok'], cells=len(result['cells']), strata=result['strata'], failures=result['failures'][:4])), flush=True)
    return 0 if result['ok'] else 1


def _add_pins(parser):
    for name in ('old-prismaquant-pin', 'old-encoder-pin', 'new-prismaquant-pin', 'new-encoder-pin'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--drop-setting', action='append', default=[], metavar='NAME',
                        help='a settings key the migration drops (the pins file\'s '
                             'drop_settings); repeatable. The expected identity has it '
                             'removed, so the arm compares against what migrate writes')


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
            p.add_argument('--members-per-class', type=int, default=0,
                           help='restrict every anchor group to its first N members per shape class, so the bisection rate is reached')
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
