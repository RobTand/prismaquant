"""The reviewed DIRECT_ASCII_SPACED_LAX tools bundle routes through the owner (issue #1301).

Scope: the 47 sites selected in the reviewed residual map's
``DIRECT_ASCII_SPACED_LAX-tools-47-sites`` bundle. 58 calls in 45 scopes now
call ``prismaquant.digests.DIRECT_ASCII_SPACED_LAX`` (text/encoded/sha256 as
each caller already needed); ``tools/tessera_fleet/model_worker.py`` is the
one selected file left unchanged, because ``tools.tessera_fleet.dispatch_model``
copies it alone into sealed workspaces and pinned producer images where it
runs with the standard library alone and cannot import the owner (recorded in
the PR body).

RED: before the routing edit every pin named ``*_routes_*`` fails -- the
scope still hand-rolls ``json.dumps(..., sort_keys=True)`` and contains no
profile call. The byte fixtures pass on both sides: they prove the profile
the site now routes to reproduces the recipe's exact bytes, so identity bytes
never move. Mixed-recipe neighbors (9 other-exact + 4 new-recipe calls in the
10 mixed scopes) keep their own spellings, and their pins fail if a neighbor
is collapsed onto the profile.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from prismaquant.digests import DIRECT_ASCII_SPACED_LAX  # noqa: E402

#: Every routed scope of the reviewed bundle, with the profile method each
#: caller already needed. ``<module>`` scopes are module-level scripts.
ROUTED_SCOPES = {
    "tools/audit_t4_render_paths.py::main": "text",
    "tools/band_serial_handoff_live_pair.py::main": "text",
    "tools/bench_fence_rehash.py::_finish": "text",
    "tools/build_glm_derivative_image.py::main": "encoded",
    "tools/build_stagea_forward_recovery_package.py::main": "text",
    "tools/build_stagea_seed_package.py::main": "text",
    "tools/build_stagea_split_package.py::main": "text",
    "tools/build_tessera_census_cache.py::main": "text",
    "tools/build_tessera_hessian_collection.py::main": "text",
    "tools/build_tessera_mtp_cached_units.py::main": "text",
    "tools/capture_glm_routing_replay.py::main": "text",
    "tools/chain_roll_bench.py::cmd_child": "sha256",
    "tools/chain_roll_bench.py::cmd_spec": "text",
    "tools/check_campaign_activation_identity.py::<module>": "text",
    "tools/check_stageb_a4_group_quantizer.py::<module>": "text",
    "tools/check_stageb_a4_quantizer.py::<module>": "text",
    "tools/checkpoint_parse_probe.py::_report": "text",
    "tools/checkpoint_parse_probe.py::main": "text",
    "tools/compare_stage_a_checkpoints.py::main": "text",
    "tools/compose_tessera_cached_units.py::main": "text",
    "tools/dispatch_joint_quanta.py::_append_state": "text",
    "tools/dispatch_joint_quanta.py::_container_wrap": "text",
    "tools/dispatch_joint_quanta.py::_dispatch": "text",
    "tools/dispatch_tessera_campaign.py::_pbrun_argv": "text",
    "tools/dispatch_tessera_campaign.py::_row": "text",
    "tools/expand_probe_onto_tessera_census.py::main": "text",
    "tools/measure_wire_rehash_readers.py::main": "text",
    "tools/nvfp4_served_qdq_bench.py::main": "text",
    "tools/pq282_equality/recheck_refusal_order.py::<module>": "text",
    "tools/profile_stage_b_head.py::main": "text",
    "tools/prove_glm_source_adoption.py::<module>": "text",
    "tools/pwc_window_load_bench.py::_quantum_identity": "text",
    "tools/pwc_window_load_bench.py::run_window": "encoded",
    "tools/qualify_t4_overlay.py::main": "text",
    "tools/regenerate_joint_quanta.py::_check_authorized_diff": "text",
    "tools/regenerate_joint_quanta.py::_check_authorized_metadata_diff": "text",
    "tools/regenerate_joint_quanta.py::_compare_existing_generation": "text",
    "tools/regenerate_joint_quanta.py::main": "text",
    "tools/render_window_bench.py::cmd_analyze": "text",
    "tools/reselect_mtp_fixed.py::main": "text",
    "tools/retire_stage_a_run.py::main": "text",
    "tools/seal_tessera_census_identity.py::main": "text",
    "tools/stage_b_preparation_submission.py::main": "text",
    "tools/stagea_forward_recovery.py::main": "text",
    "tools/staged_exact_read_bench.py::child_main": "text",
}

#: The reviewed bundle's skipped file, and why: dispatch_model copies this
#: file alone into the sealed workspace (``worker.py``) and runs it with
#: ``/usr/bin/python3`` under the standard library alone.
SKIPPED_WORKER = "tools/tessera_fleet/model_worker.py"

#: The 10 mixed scopes keep one non-profile JSON recipe each; these are the
#: kwargs that distinguish the neighbor calls the bundle must not move.
MIXED_NEIGHBOR_KWARGS = {
    "tools/build_glm_derivative_image.py::main": {"separators"},
    "tools/build_stagea_forward_recovery_package.py::main": {"indent"},
    "tools/build_tessera_census_cache.py::main": {"indent"},
    "tools/build_tessera_mtp_cached_units.py::main": {"separators", "allow_nan"},
    "tools/compare_stage_a_checkpoints.py::main": {"indent"},
    "tools/compose_tessera_cached_units.py::main": {"separators", "allow_nan"},
    "tools/dispatch_joint_quanta.py::_dispatch": {"indent"},
    "tools/measure_wire_rehash_readers.py::main": {"indent"},
    "tools/render_window_bench.py::cmd_analyze": {"indent"},
    "tools/seal_tessera_census_identity.py::main": {"indent", "allow_nan"},
}


def _parse(rel: str) -> ast.Module:
    return ast.parse((REPOSITORY / rel.split("::")[0]).read_text(encoding="utf-8"))


def _scope_node(tree: ast.Module, name: str) -> ast.AST:
    if name == "<module>":
        return tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _scope_statements(node: ast.AST) -> list[ast.stmt]:
    """The scope's own statements; nested defs own their calls separately."""
    if isinstance(node, ast.Module):
        return [s for s in node.body
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert isinstance(node, ast.FunctionDef)
    return [s for s in node.body
            if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _dumps_calls(scope_statements: list[ast.stmt]) -> list[ast.Call]:
    found = []
    for stmt in scope_statements:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dumps"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "json"):
                found.append(node)
    return found


def _recipe_kwarg_names(call: ast.Call) -> set[str]:
    return {k.arg for k in call.keywords}


def _imports_profile(rel: str) -> bool:
    for node in ast.walk(_parse(rel)):
        if (isinstance(node, ast.ImportFrom) and node.module == "prismaquant.digests"
                and any(alias.name == "DIRECT_ASCII_SPACED_LAX" for alias in node.names)):
            return True
    return False


# -- the old-vs-new byte equivalence every routed site relies on -------------

#: Representative values for the inherited recipe: empty containers, nested
#: containers, non-ASCII and escapes, the NaN/Infinity spellings the lax
#: profile accepts, and the key-order/coercion corners of the stdlib encoder.
REPRESENTATIVE_VALUES = [
    {},
    [],
    {"a": 1},
    {"b": 1, "a": 2, "c": [3, {"z": 4, "a": 5}]},
    {"nested": {"deep": [{"x": {"y": None}}]}, "empty_list": [], "empty_obj": {}},
    {"nonascii": "éωλ—" },
    {"κ": 1, "a": "κ"},
    {"control": "é\x00\r\n", "quotes": '"\\'},
    {"nan": float("nan"), "inf": float("inf"), "neginf": float("-inf")},
    {"negzero": -0.0, "tiny": 0.1, "big": 10 ** 20},
    {"flags": [True, False, None]},
    {"coerced": {1: "a", True: "b"}},
    "top-level string",
    17,
]


@pytest.mark.parametrize("value", REPRESENTATIVE_VALUES)
def test_profile_reproduces_the_inherited_recipe_bytes(value):
    """The profile the sites route to emits the recipe's exact bytes."""
    old_text = json.dumps(value, sort_keys=True)
    assert DIRECT_ASCII_SPACED_LAX.text(value) == old_text
    assert DIRECT_ASCII_SPACED_LAX.encoded(value) == old_text.encode("utf-8")
    assert DIRECT_ASCII_SPACED_LAX.sha256(value) == hashlib.sha256(
        old_text.encode("utf-8")).hexdigest()


def test_profile_keeps_the_recipe_native_refusal():
    """A non-serializable object fails exactly as json.dumps refused it."""
    with pytest.raises(TypeError) as refused:
        DIRECT_ASCII_SPACED_LAX.text(object())
    assert str(refused.value) == "Object of type object is not JSON serializable"
    assert refused.value.__cause__ is None


# -- per-site routing pins (RED before the edit: the raw recipe remains) -----


@pytest.mark.parametrize(("rel", "method"), sorted(ROUTED_SCOPES.items()))
def test_selected_scope_routes_through_the_existing_profile(rel, method):
    tree = _parse(rel)
    file_name, scope_name = rel.split("::")
    node = _scope_node(tree, scope_name)
    calls = _dumps_calls(_scope_statements(node))
    raw_spaced = [c for c in calls if _recipe_kwarg_names(c) == {"sort_keys"}
                  and c.keywords[0].value.value is True]
    assert not raw_spaced, (
        f"{rel} still hand-rolls the spaced recipe at lines "
        f"{[c.lineno for c in raw_spaced]}: route it through "
        "prismaquant.digests.DIRECT_ASCII_SPACED_LAX")
    profile_calls = [n for stmt in _scope_statements(node) for n in ast.walk(stmt)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name)
                     and n.func.value.id == "DIRECT_ASCII_SPACED_LAX"]
    assert profile_calls, f"{rel}::{scope_name} has no profile call"
    assert all(n.func.attr == method for n in profile_calls), (
        f"{rel}::{scope_name} must use DIRECT_ASCII_SPACED_LAX.{method}, "
        f"found {[n.func.attr for n in profile_calls]}")
    assert _imports_profile(file_name), f"{rel} does not import the profile"


# -- per-site byte identity for the callable seams ---------------------------


def test_append_state_line_bytes_are_unchanged(tmp_path, monkeypatch):
    from tools.dispatch_joint_quanta import _append_state
    import time as time_module
    monkeypatch.setattr(time_module, "time", lambda: 1234.5)
    path = tmp_path / "state" / "campaign-state.jsonl"
    _append_state(path, {"quantum_id": "q-1"})
    assert path.read_bytes() == b'{"quantum_id": "q-1", "unix": 1234.5}\n'


def test_append_state_keeps_the_appended_event_shape(tmp_path, monkeypatch):
    from tools.dispatch_joint_quanta import _append_state
    import time as time_module
    monkeypatch.setattr(time_module, "time", lambda: 0.0)
    path = tmp_path / "events.jsonl"
    _append_state(path, {"z": 1, "a": [2, {"é": 3}]})
    assert path.read_bytes() == (
        json.dumps({"z": 1, "a": [2, {"é": 3}], "unix": 0.0}, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_probe_report_line_bytes_are_unchanged(capsys, monkeypatch):
    from tools import checkpoint_parse_probe as probe
    monkeypatch.setattr(probe, "_gib", lambda field: 0.5)
    probe._report("parse", elapsed_s=1.5)
    assert capsys.readouterr().out == (
        '{"elapsed_s": 1.5, "peak_rss_gib": 0.5, "rss_gib": 0.5, "step": "parse"}\n')


def test_fence_finish_report_bytes_are_unchanged(tmp_path, capsys):
    from tools.bench_fence_rehash import _finish
    root = tmp_path / "scratch"
    root.mkdir()
    report = {"runs": [{"wall_s": 3.0}, {"wall_s": 1.0}, {"wall_s": 2.0}]}
    assert _finish(report, str(root), 100.0) == 0
    assert not root.exists()
    assert capsys.readouterr().out == (
        "BENCH "
        + json.dumps({"runs": [{"wall_s": 3.0}, {"wall_s": 1.0}, {"wall_s": 2.0}],
                      "median_wall_s": 2.0, "median_bytes_per_s": 50.0},
                     sort_keys=True) + "\n")


# -- the skipped worker stays exactly as it is -------------------------------


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skipped_worker_stays_stdlib_only_and_unrouted():
    """The sealed worker file keeps its recipe: the owner cannot import there."""
    tree = _parse(SKIPPED_WORKER)
    assert not _imports_profile(SKIPPED_WORKER), (
        "tools/tessera_fleet/model_worker.py must not import prismaquant: "
        "dispatch_model stages it alone and runs it stdlib-only")
    atomic = _scope_node(tree, "atomic_json")
    main = _scope_node(tree, "main")
    assert [c for c in _dumps_calls(_scope_statements(atomic))
            if _recipe_kwarg_names(c) == {"sort_keys"}]
    assert [c for c in _dumps_calls(_scope_statements(main))
            if _recipe_kwarg_names(c) == {"sort_keys"}]


def test_skipped_worker_atomic_json_bytes_are_unchanged(tmp_path):
    worker = _load_by_path("worker_1301", SKIPPED_WORKER)
    path = tmp_path / "prepared" / "identity.json"
    worker.atomic_json(path, {"b": 1, "a": "é"})
    assert path.read_bytes() == b'{"a": "\\u00e9", "b": 1}\n'
    again = tmp_path / "again.json"
    worker.atomic_json(again, {})
    assert again.read_bytes() == b'{}\n'


# -- mixed-recipe neighbors keep their own spellings --------------------------


@pytest.mark.parametrize(("rel", "kwargs"), sorted(MIXED_NEIGHBOR_KWARGS.items()))
def test_mixed_recipe_neighbors_keep_their_own_recipe(rel, kwargs):
    tree = _parse(rel)
    node = _scope_node(tree, rel.split("::")[1])
    calls = _dumps_calls(_scope_statements(node))
    keepers = [c for c in calls if kwargs & _recipe_kwarg_names(c)]
    assert keepers, (
        f"{rel} lost its non-profile JSON recipe (kwargs {sorted(kwargs)}): "
        "the reviewed bundle must not move mixed-recipe neighbors")


def test_no_routed_scope_grew_a_new_raw_spaced_site():
    """The whole bundle is either routed (pins above) or the recorded skip."""
    routed_files = {rel.split("::")[0] for rel in ROUTED_SCOPES}
    for rel in sorted(routed_files):
        tree = _parse(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            raw = [c for c in _dumps_calls(_scope_statements(node))
                   if _recipe_kwarg_names(c) == {"sort_keys"}
                   and c.keywords[0].value.value is True]
            assert not raw, (
                f"{rel}::{node.name} still hand-rolls the spaced recipe at "
                f"{[c.lineno for c in raw]}")
