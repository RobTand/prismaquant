"""PQ #1301 closure: current call coverage, justified retention and fixture coverage.

The #2540 refresh is historical evidence and #2637 pins part of it verbatim.
Keep it; place the reconciled snapshot in ``current_1301`` using the existing
refresh row format. These checks do not execute code embedded in the census,
modify the scanner's population, or require historical Git objects.

``digest_parent_completion_1301.json`` records the research base's call
population, NOT expected hashes computed by a replacement owner. For every
removed/replaced call, ``completed_calls`` in current_1301 must link independent
old-consumer fixtures and executable compatibility tests. Review must still
verify fixture independence, consumer semantics, retention eligibility and the
supplemental audit's limits; metadata alone cannot establish those facts.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools import duplication_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "docs/audits/digest_site_census_pq1301_2026-10-04.json"
ORIGINAL = ROOT / "tests/fixtures/digest_parent_completion_1301.json"
RECIPE_FIELDS = {
    "algorithm", "encoding_options", "input_order", "framing", "digest_bits",
    "result_type", "result_width", "read_boundaries", "refusal_behavior",
}


def _family(call, callee):
    if callee and callee.startswith("hashlib."):
        return "hashlib"
    if callee in {"json.dumps", "json.JSONEncoder", "json.dump"}:
        if callee != "json.dump" and any(
            k.arg == "sort_keys" and isinstance(k.value, ast.Constant)
            and k.value.value is True for k in call.keywords
        ):
            return "sorted-json"
        return "supplemental-json"
    return None


def _live_calls():
    """Use the unchanged scanner traversal, retaining individual calls too."""
    rows = []
    for path in inventory._files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        for call, scope, aliases in inventory._iter_scoped_calls(ast.parse(source)):
            callee = inventory._resolved_callee(call.func, aliases)
            family = _family(call, callee)
            if family is None:
                continue
            site = f"{family}:{rel}::{scope}"
            rows.append({
                "id": f"{site}@{call.lineno}:{call.col_offset}",
                "site": site, "family": family, "file": rel, "scope": scope,
                "line": call.lineno, "column": call.col_offset,
                "source_blob": blob, "callee": callee,
                "expression": ast.get_source_segment(source, call),
                "syntax": ast.dump(call, include_attributes=False),
            })
    return sorted(rows, key=lambda row: row["id"])


def _current():
    document = json.loads(CENSUS.read_text(encoding="utf-8"))
    # Fall back only to produce useful old-code RED showing the stale census,
    # rather than an import, missing-key or test-discovery error.
    return document.get("current_1301", document["refresh_2540"])


def _rows(snapshot):
    return snapshot["calls"] + snapshot["supplemental_calls"]


def _repo_path(name):
    assert isinstance(name, str) and name, "missing repository evidence path"
    path = ROOT / name
    assert not Path(name).is_absolute() and path.resolve().is_relative_to(ROOT), name
    assert path.is_file(), f"evidence file does not exist: {name}"
    return path


def _test_reference(reference):
    parts = reference.split("::")
    assert len(parts) >= 2, f"expected an executable test node: {reference}"
    path = _repo_path(parts[0])
    assert parts[0].startswith("tests/test_") and path.suffix == ".py", reference
    assert path != Path(__file__), "completion checks are not consumer compatibility fixtures"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    for name in parts[1:]:
        matches = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and n.name == name]
        assert len(matches) == 1, f"missing compatibility test node: {reference}"
        nodes = matches[0].body
    assert parts[-1].startswith("test_"), reference


def test_current_census_matches_every_live_call_and_recipe():
    current = _current()
    live = _live_calls()
    recorded = _rows(current)
    assert len({row["id"] for row in recorded}) == len(recorded), "duplicate census call"
    actual = {row["id"]: row for row in live}
    indexed = {row["id"]: row for row in recorded}
    assert indexed.keys() == actual.keys(), (
        "census is not current (historical refresh_2540 must remain separate)",
        {"missing": sorted(actual.keys() - indexed.keys())[:12],
         "obsolete": sorted(indexed.keys() - actual.keys())[:12]},
    )
    primary = {row["site"] for row in live if row["family"] != "supplemental-json"}
    assert set(current["scan_sites"]) == primary == set(inventory.digest_sites(ROOT))
    gated = {site for site in primary
             if site.split(":", 1)[1].split("::", 1)[0] not in inventory.DIGEST_OWNERS}
    baseline = json.loads(inventory.BASELINE.read_text(encoding="utf-8"))
    assert gated == set(current["ratchet_sites"]) == set(baseline["primitive_digest_sites"])
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    assert gated <= set(original["ratchet_sites"]), "closure must not grow the digest baseline"
    for call_id, row in indexed.items():
        observed = actual[call_id]
        for field in ("file", "scope", "callee", "source_blob", "line", "column"):
            assert row[field] == observed[field], (call_id, field)
        assert ast.dump(ast.parse(row["expression"], mode="eval").body, include_attributes=False) == observed["syntax"]
        assert row["classification"] in {"load-bearing", "ephemeral"}, call_id
        assert row["classification_reason"] and row["consumer_evidence"], call_id
        assert RECIPE_FIELDS <= row["recipe"].keys(), call_id
        assert current["source_blobs"][row["file"]]["git_blob"] == row["source_blob"], call_id
    # This deliberately does NOT broaden the primitive ratchet. The reviewer
    # also checks the separately documented validators/aliases/generated/shell
    # population and exclusions; an unusual recipe is not retention evidence.
    assert current["exclusions"], "record supplemental audit boundaries"


def test_no_unresolved_eligible_primitive_or_json_calls_at_parent_closure():
    rows = _rows(_current())
    unresolved = [row["id"] for row in rows
                  if row["file"] not in inventory.DIGEST_OWNERS
                  and (row["eligible"] or row["disposition"] != "retain")]
    assert not unresolved, (
        "parent closure still has eligible/unresolved calls; migrate exactly or prove retention",
        unresolved[:20],
    )
    for row in rows:
        if row["file"] in inventory.DIGEST_OWNERS:
            assert row["disposition"] == "owner-implementation", row["id"]
            continue
        evidence = row["retention_evidence"]
        assert evidence and evidence["reference"] and evidence["kind"], row["id"]
        assert row["disposition_reason"], row["id"]
        # Every retention names an existing contract witness, not a generic
        # assertion that a recipe differs. Semantic validity is independently
        # reviewed, including staged layouts and reference independence.
        _repo_path(evidence["reference"].split("::", 1)[0])
        assert evidence["source_blob"] == row["source_blob"], row["id"]


def _identity(row):
    # Lines may move and imports may be rearranged without being migrations.
    syntax_hash = row.get("syntax_sha256")
    if syntax_hash is None:
        syntax = row.get("syntax") or ast.dump(
            ast.parse(row["expression"], mode="eval").body, include_attributes=False)
        syntax_hash = hashlib.sha256(syntax.encode("utf-8")).hexdigest()
    return row["file"], row["scope"], row["callee"], syntax_hash


def test_replaced_calls_have_independent_consumer_fixture_references():
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    remaining = Counter(_identity(row) for row in _live_calls())
    replaced = []
    for row in original["calls"]:
        if row["file"] in inventory.DIGEST_OWNERS:
            continue
        key = _identity(row)
        if remaining[key]:
            remaining[key] -= 1
        else:
            replaced.append(row)
    current = _current()
    completed = current.get("completed_calls", [])
    assert len({row["original_id"] for row in completed}) == len(completed), "duplicate completion evidence"
    indexed = {row["original_id"]: row for row in completed}
    missing = [row["id"] for row in replaced if row["id"] not in indexed]
    assert not missing, ("replaced calls need per-site old-consumer fixtures", missing[:20])
    # Until the first replacement, old code must not pass the completion check
    # merely because an empty ledger is vacuously complete.
    assert completed, "no completed per-call migrations recorded for the remaining parent work"
    for row in replaced:
        evidence = indexed[row["id"]]
        assert evidence["source_commit"] == original["source_commit"], row["id"]
        assert evidence["source_blob"] == row["source_blob"], row["id"]
        assert evidence["fixtures"] and evidence["tests"], row["id"]
        for name in evidence["fixtures"]:
            path = _repo_path(name)
            assert name.startswith("tests/") and path != ORIGINAL, name
        for reference in evidence["tests"]:
            _test_reference(reference)
        assert evidence["owner_route"], row["id"]


def test_allocator_pareto_owner_route_preserves_original_publication(tmp_path):
    """Exercise the real CPU publication block, not a replacement-owner oracle.

    This isolates publication from the CLI's model/probe setup. The original
    source fixture and revised block receive identical records. In addition to
    complete files/stdout, compare all hash input bytes (streamed or one-shot).
    """
    import contextlib
    import copy
    import io
    from types import SimpleNamespace
    from unittest.mock import patch

    from prismaquant import allocator

    source = (ROOT / "prismaquant/allocator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    blocks = [n for n in main.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == "args.pareto_output_dir"
              and any(isinstance(child, ast.Name) and child.id == "seen_payloads"
                      for child in ast.walk(n))]
    assert len(blocks) == 1, "locate the actual Pareto publication block"
    revised = ast.Module(body=[blocks[0]], type_ignores=[])
    original_source = (ROOT / "tests/fixtures/digest_allocator_pareto_1301.py").read_text(encoding="utf-8")
    original = ast.parse(original_source)
    base = {
        "target_bits": 4.25, "achieved_bits": 4.0, "bits_total": 40.0,
        "bits_total_with_aux": 41.0, "predicted_dloss": 0.125,
        "total_predicted_dloss_with_aux": 0.25, "variable_predicted_dloss": 0.1,
        "aux_fixed_predicted_dloss": 0.025, "fixed_predicted_dloss": 0.125,
        "format_counts": {"NVFP4": 1},
    }
    records = [
        {**base, "assignment": {"z": "FP8", "é": "NVFP4"}},
        {**base, "target_bits": 4.5, "assignment": {"é": "NVFP4", "z": "FP8"}},
        {**base, "assignment": {"z": "NVFP4", "é": "FP8"}},
    ]
    real_sha256 = hashlib.sha256

    def exercise(block, directory):
        feeds = []

        def recording_sha256(data=b"", **kwargs):
            feed = bytearray(data)
            feeds.append(feed)
            delegate = real_sha256(data, **kwargs)

            class Digest:
                def update(self, raw):
                    feed.extend(raw)
                    delegate.update(raw)

                def hexdigest(self):
                    return delegate.hexdigest()

                def digest(self):
                    return delegate.digest()

            return Digest()

        args = SimpleNamespace(
            pareto_output_dir=str(directory), target_disk_gb=None,
            artifact_overhead_reserve_bytes=None, probe="probe-input",
            costs="cost-input",
        )
        namespace = dict(vars(allocator))
        namespace.update(
            args=args, pareto_seed_records=copy.deepcopy(records), target_profile="fixture",
            measured_runtime_table=None, specs_sorted=[SimpleNamespace(name="NVFP4")],
            targets=[4.25, 4.5], knee_summary={"enabled": False},
        )
        stdout = io.StringIO()
        with patch.object(hashlib, "sha256", recording_sha256), contextlib.redirect_stdout(stdout):
            exec(compile(block, "<allocator-pareto-consumer>", "exec"), namespace)
        files = {p.name: p.read_bytes().replace(str(directory).encode(), b"<OUTPUT>")
                 for p in sorted(directory.iterdir())}
        return files, stdout.getvalue().replace(str(directory), "<OUTPUT>"), list(map(bytes, feeds))

    expected = exercise(original, tmp_path / "original")
    actual = exercise(revised, tmp_path / "revised")
    assert actual == expected, "publication bytes, duplicate suppression, labels or hash feed changed"
    manifest = json.loads(actual[0]["manifest.json"])
    assert len(manifest["candidates"]) == 2
    for candidate in manifest["candidates"]:
        assert len(candidate["label"].rsplit("_", 1)[1]) == 12
    # Native mixed-key refusal must occur at the original point with no output.
    # Successful publication above also checks Unicode and insertion-order ties.
    def refused(block, directory):
        bad = {**base, "assignment": {1: "NVFP4", "z": "FP8"}}
        namespace = dict(vars(allocator))
        namespace.update(args=SimpleNamespace(pareto_output_dir=str(directory)),
                         pareto_seed_records=[bad])
        try:
            exec(compile(block, "<allocator-pareto-consumer>", "exec"), namespace)
        except Exception as error:
            return type(error), str(error), list(directory.iterdir())
        raise AssertionError("mixed keys unexpectedly accepted")

    assert refused(revised, tmp_path / "bad-new") == refused(original, tmp_path / "bad-old")

    # This fails on old code AFTER the independent consumer comparisons pass.
    # Routing, not a change in the stored twelve-character identity, is missing.
    raw = [call for call, scope, aliases in inventory._iter_scoped_calls(tree)
           if scope == "main" and (inventory._resolved_callee(call.func, aliases) or "").startswith("hashlib.")]
    assert not raw, "allocator.main's remaining Pareto identity must use its exact digest owner"
