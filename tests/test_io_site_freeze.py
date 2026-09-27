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
``multiprocessing.Process``, plus their subclasses. Explicit import and
simple assignment aliases, including multiprocessing contexts, retain their
constructor identity. Bindings stay lexical; possible branch/rebinding targets
are conservative unions. This is a static guard, not execution of arbitrary
factory functions or reflection. It requires ``ALLOWED`` to match exactly:

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
    "prismaquant/joint_replay_spill.py::StageBReplaySpill._start_arenas": (1, "writer"),
    "prismaquant/layer_streaming.py::_layer_read_pool": (1, "reader"),
    "prismaquant/perturbed_x_cache.py::_exact_read_pool": (1, "reader"),
    "prismaquant/produced_stager.py::ProducedStager.__init__": (1, "writer"),
    "prismaquant/production_weight_cache.py::ProductionWeightCache.prefetch": (1, "reader"),
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


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _qualified(node: ast.expr, aliases: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, {node.id})
    if isinstance(node, ast.Attribute):
        return {f"{base}.{node.attr}" for base in _qualified(node.value, aliases)}
    if isinstance(node, ast.Call) and "multiprocessing.get_context" in _qualified(
            node.func, aliases):
        return {"multiprocessing.context"}
    return set()


def _scope_nodes(node: ast.AST):
    """Inspect a lexical scope, not the bodies of its nested definitions."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _SCOPES):
            yield from _scope_nodes(child)


def _scope_aliases(node: ast.AST, inherited: dict[str, set[str]]) -> dict[str, set[str]]:
    nodes = list(_scope_nodes(node))
    imports, assignments, local = [], [], set()
    for child in nodes:
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imports.append(child)
            local.update(a.asname or a.name.split(".")[0] for a in child.names)
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            local.update(names)
            if child.value is not None:
                assignments.append((names, child.value))
        elif isinstance(child, ast.arg):
            local.add(child.arg)
    aliases = {k: set(v) for k, v in inherited.items() if k not in local}
    # Collect imports before uses: a function may use a module import written
    # below its definition. Keep all possibilities on rebinding/branches, so
    # a later non-pool import cannot erase an earlier constructor call.
    for child in imports:
        for alias in child.names:
            if isinstance(child, ast.Import):
                name = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else name
            else:
                name = alias.asname or alias.name
                target = f"{child.module}.{alias.name}"
                if alias.name == "*" and child.module == "multiprocessing":
                    for member in _MP_CALLEES:
                        aliases.setdefault(member, set()).add(f"multiprocessing.{member}")
            aliases.setdefault(name, set()).add(target)
    # Fixed-point resolution follows simple constructor/context aliases without
    # executing source. Lexical locals do not leak into sibling functions.
    for _ in range(len(assignments)):
        for names, value in assignments:
            for name in names:
                aliases.setdefault(name, set()).update(_qualified(value, aliases))
    return aliases


def _construction(node: ast.expr, aliases: dict[str, set[str]]) -> bool:
    for qualified in _qualified(node, aliases):
        base, _, name = qualified.rpartition(".")
        if name in _CALLEES or (name in _MP_CALLEES and (
                base in _MP_BASES or base.startswith("multiprocessing."))):
            return True
    return False


def _sites(source: str) -> list[str]:
    found: list[str] = []

    def walk(node: ast.AST, scope: list[str], aliases: dict[str, set[str]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                if isinstance(child, ast.ClassDef) and any(
                        _construction(base, aliases) for base in child.bases):
                    found.append(".".join(scope + [child.name]) + " (subclass)")
                walk(child, scope + [child.name], _scope_aliases(child, aliases))
                continue
            if isinstance(child, ast.Call) and _construction(child.func, aliases):
                found.append(".".join(scope) or "<module>")
            walk(child, scope, aliases)

    tree = ast.parse(source)
    walk(tree, [], _scope_aliases(tree, {}))
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
