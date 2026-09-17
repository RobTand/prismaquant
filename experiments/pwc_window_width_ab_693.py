#!/usr/bin/env python3
"""Interleaved A/B for the PWC resident-window change of #693.

Both arms run in ONE process against ONE file set, so the only thing that
differs between them is the code under test:

  legacy  -- the window rule this change replaces: the caller's byte budget is
             ``min(max_render_bytes, max_load_buffer_bytes)`` and a quantum is
             additionally capped at ``max_workers`` keys, and ``_window_file``
             re-reads every archive directory on every reach.
  current -- the quantum is as wide as the two admitted byte budgets allow, and
             each file's archive storage is priced once per window lifetime.

It replays ``prepare_cache``'s consumption shape exactly: plan the unit's
renders, then, for each planned quantum, open a ``resident_window`` with
``release_file_pages=True`` and borrow every key once. What it does NOT do is
the GPU verify, so the wall time here is the load leg alone.

Reported per arm: wall seconds, quanta opened, archive-directory reads, and
``open()`` calls on the render files. The two counts are deterministic; the
wall time is not, which is why the arms interleave and why both are printed.
"""
from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path


def build_unit_files(root: Path, *, units: int, renders: int, bytes_each: int):
    """Write ``units`` x ``renders`` Torch archives of the production size."""
    import torch

    root.mkdir(parents=True, exist_ok=True)
    elements = bytes_each // 2
    side = int(elements ** 0.5)
    layout = []
    for unit in range(units):
        keys = []
        for render in range(renders):
            path = root / f"unit{unit:03d}.render{render}.pt"
            if not path.is_file() or path.stat().st_size < bytes_each:
                torch.save(torch.full((side, side), float(unit + render),
                                      dtype=torch.bfloat16), path)
            keys.append((f"unit{unit:03d}", f"FMT{render}", path))
        layout.append(keys)
    return layout


class Counters:
    def __init__(self):
        self.archive_reads = 0
        self.file_opens = 0
        self.quanta = 0


def instrument(counters, render_paths):
    """Count archive-directory reads and render-file opens, without changing them."""
    from prismaquant import perturbed_x_cache as pxc

    original_archive = pxc.torch_archive_storage_bytes

    def counted_archive(source, **kwargs):
        if isinstance(source, (str, Path)):
            counters.archive_reads += 1
        return original_archive(source, **kwargs)

    pxc.torch_archive_storage_bytes = counted_archive

    original_open = builtins.open
    names = {str(path) for path in render_paths}

    def counted_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and str(file) in names:
            counters.file_opens += 1
        return original_open(file, *args, **kwargs)

    builtins.open = counted_open
    original_path_open = Path.open

    def counted_path_open(self, *args, **kwargs):
        if str(self) in names:
            counters.file_opens += 1
        return original_path_open(self, *args, **kwargs)

    Path.open = counted_path_open
    return lambda: (setattr(pxc, 'torch_archive_storage_bytes', original_archive),
                    setattr(builtins, 'open', original_open),
                    setattr(Path, 'open', original_path_open))


def run_arm(layout, *, legacy: bool, max_render_bytes: int,
            max_load_buffer_bytes: int, workers: int):
    """One pass over every unit, in ``prepare_cache``'s consumption shape."""
    from prismaquant.production_weight_cache import ProductionWeightCache

    weights = {(name, fmt): str(path) for keys in layout for name, fmt, path in keys}
    paths = [path for keys in layout for _, _, path in keys]
    cache = ProductionWeightCache(weights=dict(weights), levers={})
    cache.enable_lru(max_render_bytes)
    counters = Counters()
    restore = instrument(counters, paths)
    if legacy:
        # The rule this change replaces, re-imposed on the current planner so
        # both arms share one code path for everything else.
        original_memo = cache._window_archive_memo
        cache._window_archive_memo = lambda: {}
    try:
        started = time.perf_counter()
        for keys in layout:
            unit_keys = tuple((name, fmt) for name, fmt, _ in keys)
            if legacy:
                planned = cache.plan_resident_windows(
                    unit_keys, max_resident_bytes=min(max_render_bytes,
                                                      max_load_buffer_bytes),
                    max_workers=workers)
                windows = tuple(window[index:index + workers]
                                for window in planned
                                for index in range(0, len(window), workers))
            else:
                windows = cache.plan_resident_windows(
                    unit_keys, max_resident_bytes=max_render_bytes,
                    max_load_buffer_bytes=max_load_buffer_bytes,
                    max_workers=workers)
            for window in windows:
                counters.quanta += 1
                with cache.resident_window(
                        window, max_resident_bytes=max_render_bytes,
                        max_workers=workers,
                        max_load_buffer_bytes=max_load_buffer_bytes,
                        release_file_pages=True):
                    for key in window:
                        borrowed = cache.get_resident(*key)
                        assert borrowed is not None
                        borrowed = None
        elapsed = time.perf_counter() - started
    finally:
        restore()
        if legacy:
            cache._window_archive_memo = original_memo
    return {'seconds': elapsed, 'quanta': counters.quanta,
            'archive_reads': counters.archive_reads,
            'render_file_opens': counters.file_opens}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='directory for the render files (never /tmp)')
    parser.add_argument('--units', type=int, default=8)
    parser.add_argument('--renders', type=int, default=5)
    parser.add_argument('--render-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--max-render-bytes', type=int, default=536870912)
    parser.add_argument('--max-load-buffer-bytes', type=int, default=536870912)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--json-out')
    parser.add_argument('--only', choices=('legacy', 'current'),
                        help='run one arm only, for a per-arm profile')
    args = parser.parse_args()
    if str(args.root).startswith('/tmp'):
        raise SystemExit('refusing to write render files under /tmp')

    root = Path(args.root)
    layout = build_unit_files(root, units=args.units, renders=args.renders,
                              bytes_each=args.render_bytes)
    shared = dict(max_render_bytes=args.max_render_bytes,
                  max_load_buffer_bytes=args.max_load_buffer_bytes,
                  workers=args.workers)
    # One untimed pass so neither arm pays first-touch costs the other avoids.
    if args.only in (None, 'legacy'):
        run_arm(layout, legacy=True, **shared)
    if args.only in (None, 'current'):
        run_arm(layout, legacy=False, **shared)

    rows = []
    for repeat in range(args.repeats):
        order = ['legacy', 'current'] if repeat % 2 == 0 else ['current', 'legacy']
        if args.only:
            order = [args.only]
        for arm in order:
            result = run_arm(layout, legacy=(arm == 'legacy'), **shared)
            rows.append({'repeat': repeat, 'arm': arm, **result})
            print(json.dumps(rows[-1]), flush=True)

    summary = {}
    for arm in (('legacy', 'current') if not args.only else (args.only,)):
        seconds = [row['seconds'] for row in rows if row['arm'] == arm]
        sample = next(row for row in rows if row['arm'] == arm)
        summary[arm] = {
            'median_seconds': statistics.median(seconds),
            'min_seconds': min(seconds),
            'seconds': seconds,
            'quanta': sample['quanta'],
            'archive_reads': sample['archive_reads'],
            'render_file_opens': sample['render_file_opens'],
        }
    if not args.only:
        summary['ratio_median_seconds'] = (summary['legacy']['median_seconds']
                                           / summary['current']['median_seconds'])
    summary['inputs'] = {'units': args.units, 'renders': args.renders,
                         'render_bytes': args.render_bytes, 'root': str(root),
                         'repeats': args.repeats, **shared}
    print(json.dumps({'summary': summary}, indent=1), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
