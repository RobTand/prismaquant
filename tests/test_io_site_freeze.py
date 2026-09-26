"""Freeze the repo's thread and executor construction sites (PQ #1294, P1).

The repo grew one hand-rolled loader per stage: 32 thread or executor sites
in ``prismaquant/`` and 18 in ``tools/`` on 2026-09-25. None is a textual
copy of another, so clone-finding audits never flagged them, but most do the
same job: move bytes from a tier to a consumer, verify them and decode them.
Each carries its own worker count and depth constant, and most run load and
compute in turn (#1116, #1128, #725, #997, #1253, #1291).

Byte movement belongs to one engine, ``prismaquant/io_engine.py`` (#1294).
This test is the mechanical half of that rule. It scans every module with
``ast`` and finds each call to ``ThreadPoolExecutor``,
``ProcessPoolExecutor``, ``Thread``, ``multiprocessing.Pool`` or
``multiprocessing.Process``, plus every ``Thread`` subclass. It then
requires the result to equal ``ALLOWED`` exactly:

- a new site, or one more site under an existing key, fails, so new code
  goes through the engine;
- a key that has lost sites fails until ``ALLOWED`` is edited to match, so
  each migration records the shrink in the same change.

The allowlist only shrinks. The end state is an empty ``ALLOWED``. Keys are
``path::qualname`` and never line numbers, which drift. The role tag on each
entry is provisional, read from the function's name. It orders the
migration and is not checked.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("prismaquant", "tools")
ENGINE = "prismaquant/io_engine.py"
_CALLEES = {"ThreadPoolExecutor", "ProcessPoolExecutor", "Thread"}
_MP_CALLEES = {"Pool", "Process"}
_MP_BASES = {"multiprocessing", "mp", "ctx"}

# key -> (site count, provisional role). Shrink only.
ALLOWED: dict[str, tuple[int, str]] = {
    "prismaquant/cost_stage_checkpoint.py::_drive_ordered_units": (1, "reader"),
    "prismaquant/cost_streaming.py::_hash_source_shards": (1, "hasher"),
    "prismaquant/incremental_probe.py::_run_body_streaming_shard": (1, "writer"),
    # The one sampler thread every periodic sampler is built on (PQ #1299).
    "prismaquant/io_spans.py::PeriodicSampler.__init__": (1, "sampler"),
    "prismaquant/joint_adjoint_checkpoints.py::stream_exact_entry_tensors": (1, "reader"),
    "prismaquant/joint_quantum_handoff.py::HandoffStream.__enter__": (1, "writer"),
    "prismaquant/joint_replay_spill.py::StageBReplaySpill._chunks": (1, "reader"),
    "prismaquant/joint_replay_spill.py::StageBReplaySpill._start_arenas": (1, "writer"),
    "prismaquant/joint_replay_spill.py::StageBReplaySpill._start_read_buffers": (1, "reader"),
    "prismaquant/layer_streaming.py::_layer_read_pool": (1, "reader"),
    "prismaquant/perturbed_x_cache.py::_exact_read_pool": (1, "reader"),
    "prismaquant/produced_stager.py::ProducedStager.__init__": (1, "writer"),
    "prismaquant/production_weight_cache.py::ProductionWeightCache.prefetch": (1, "reader"),
    "prismaquant/production_weight_cache.py::ProductionWeightCache.retained_window": (1, "reader"),
    "prismaquant/residency_shard_reader.py::_chunk_pool": (1, "reader"),
    "prismaquant/source_prefetch.py::prefetch_files_to_page_cache": (1, "reader"),
    "prismaquant/streaming_model.py::_build_streaming_context": (1, "reader"),
    "prismaquant/tessera_calibration_cache.py::_parallel_prefetch_capture": (1, "reader"),
    "prismaquant/tessera_campaign.py::_SealAhead.__init__": (1, "writer"),
    "prismaquant/tessera_campaign.py::_campaign_bound_identities": (1, "hasher"),
    "prismaquant/tessera_campaign.py::_verify_wire_records_on_threads": (1, "hasher"),
    "prismaquant/tessera_census_cache.py::census_selected_cached_units_manifest": (1, "hasher"),
    "prismaquant/tessera_census_cache.py::load_selected_wire_records": (1, "reader"),
    "prismaquant/tessera_joint_aura.py::load_measured_anchor_input": (1, "hasher"),
    "prismaquant/tessera_joint_aura.py::prepare_cache": (1, "reader"),
    "prismaquant/tessera_publication.py::BoundedPublisher.__init__": (1, "writer"),
    "prismaquant/tessera_row_stream.py::RowStream.__init__": (1, "reader"),
    "tools/audit_t4_render_paths.py::main": (1, "tool"),
    "tools/benchmarks/joint_validation_pair.py::main": (1, "tool"),
    "tools/build_t4_overlay_catalog.py::main": (1, "tool"),
    "tools/measure_wire_rehash_readers.py::main": (1, "tool"),
    "tools/staged_exact_read_bench.py::child_ceiling": (1, "tool"),
    "tools/staged_exact_read_bench.py::child_sink_ceiling": (1, "tool"),
    "tools/tp_decode_feasibility.py::_spawn_loopback": (1, "tool"),
    "tools/unit_journal_bench.py::cmd_child": (1, "tool"),
}


def _callee(node: ast.expr) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return node.id, None
    if isinstance(node, ast.Attribute):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        return node.attr, base
    return None, None


def _sites(source: str) -> list[str]:
    found: list[str] = []

    def walk(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(child, ast.ClassDef) and any(
                        _callee(base)[0] == "Thread" for base in child.bases):
                    found.append(".".join(scope + [child.name]) + " (subclass)")
                walk(child, scope + [child.name])
                continue
            if isinstance(child, ast.Call):
                name, base = _callee(child.func)
                if name in _CALLEES or (name in _MP_CALLEES and base in _MP_BASES):
                    found.append(".".join(scope) or "<module>")
            walk(child, scope)

    walk(ast.parse(source), [])
    return found


def scan(root: Path = ROOT) -> Counter:
    counts: Counter = Counter()
    for top in SCANNED:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel == ENGINE or "/archive/" in f"/{rel}":
                continue
            for site in _sites(path.read_text(encoding="utf-8")):
                counts[f"{rel}::{site}"] += 1
    return counts


def test_scanner_sees_every_construction_form():
    source = (
        "import threading, multiprocessing as mp\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "import concurrent.futures as cf\n"
        "def a():\n    ThreadPoolExecutor(2)\n"
        "def b():\n    cf.ProcessPoolExecutor()\n"
        "class C:\n    def d(self):\n        threading.Thread(target=print)\n"
        "def e():\n    mp.Pool(2)\n"
        "class T(threading.Thread):\n    pass\n"
        "def f():\n    subprocess_pool = dict(Pool=1)\n"
    )
    assert sorted(_sites(source)) == ["C.d", "T (subclass)", "a", "b", "e"]


def test_thread_and_executor_sites_equal_the_frozen_allowlist():
    live = scan()
    allowed = {key: count for key, (count, _) in ALLOWED.items()}
    new = {k: v for k, v in live.items() if k not in allowed}
    grown = {k: (allowed[k], v) for k, v in live.items() if k in allowed and v > allowed[k]}
    shrunk = {k: (n, live.get(k, 0)) for k, n in allowed.items() if live.get(k, 0) < n}
    assert not new, (
        f"new thread/executor sites {sorted(new)}: move bytes through {ENGINE} "
        "(PQ #1294) instead of a new pool")
    assert not grown, f"allowlisted keys gained sites (allowed, live): {grown}"
    assert not shrunk, (
        f"sites migrated or removed (allowed, live): {shrunk}; shrink ALLOWED "
        "in the same change")
