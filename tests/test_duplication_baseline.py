"""Duplicated functionality may shrink and never grow (PQ #1295).

``tools/duplication_inventory.py`` finds cross-file structural
near-duplicates and module-level helpers that share a name across modules,
which are the duplication textual audits miss.
``tests/fixtures/duplication_baseline.json`` records what existed on 2026-09-25.
A new near-duplicate pair, a new same-name group, or a module joining an
existing group fails. So does a baseline entry that no longer exists: each
consolidation shrinks the baseline in the same change, by running
``python tools/duplication_inventory.py --write-baseline``.

The baseline also ratchets primitive digest sites (PQ #1508): a new raw
``hashlib`` constructor or literal ``sort_keys=True`` JSON encoding outside
``prismaquant/digests.py`` and ``prismaquant/tensor_digests.py`` fails, and
consolidating a site onto an owner lowers the baseline the same way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import duplication_inventory as inventory  # noqa: E402


def _baseline():
    return json.loads(inventory.BASELINE.read_text(encoding="utf-8"))


def test_scanner_finds_a_renamed_copy_and_ignores_same_file_pairs(tmp_path):
    body = (
        "    total = 0\n"
        "    for item in items:\n"
        "        if item.ready:\n"
        "            total += item.size\n"
        "        else:\n"
        "            total -= helper(item)\n"
        "    return {'total': total, 'count': len(items)}\n"
    )
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "prismaquant" / "a.py").write_text(
        "def _sha(items):\n" + body + "\ndef other(things):\n" + body.replace("items", "things"))
    (tmp_path / "tools" / "b.py").write_text(
        "def _sha(rows):\n" + body.replace("items", "rows").replace("total", "acc"))
    live = inventory.scan(tmp_path)
    assert ["prismaquant/a.py::_sha", "tools/b.py::_sha"] in live["near_duplicates"]
    assert ["prismaquant/a.py::other", "tools/b.py::_sha"] in live["near_duplicates"]
    assert all(a.split("::")[0] != b.split("::")[0] for a, b in live["near_duplicates"])
    assert live["same_name_helpers"] == {"_sha": ["prismaquant/a.py", "tools/b.py"]}


def test_near_duplicate_pairs_only_shrink():
    live = {tuple(p) for p in inventory.scan()["near_duplicates"]}
    base = {tuple(p) for p in _baseline()["near_duplicates"]}
    assert not live - base, (
        f"new near-duplicate functions {sorted(live - base)}: reuse the existing "
        "implementation instead of a copy (PQ #1295)")
    assert not base - live, (
        f"baseline pairs gone {sorted(base - live)}: shrink the baseline with "
        "tools/duplication_inventory.py --write-baseline")


def test_same_name_helpers_only_shrink():
    live = inventory.scan()["same_name_helpers"]
    base = _baseline()["same_name_helpers"]
    grown = {n: sorted(set(m) - set(base.get(n, ()))) for n, m in live.items()
             if set(m) - set(base.get(n, ()))}
    assert not grown, (
        f"helpers defined again in another module {grown}: import the existing "
        "one (PQ #1295)")
    shrunk = {n: sorted(set(m) - set(live.get(n, ()))) for n, m in base.items()
              if set(m) - set(live.get(n, ()))}
    assert not shrunk, (
        f"baseline entries gone {shrunk}: shrink the baseline with "
        "tools/duplication_inventory.py --write-baseline")


def test_must_differ_pairs_are_live_and_give_a_reason():
    """A pair kept as two implementations says why, and still exists (#1302)."""
    base = _baseline()
    pairs = {tuple(p) for p in base["near_duplicates"]}
    seen = set()
    for row in base.get("must_differ", []):
        assert set(row) == {"pair", "reason"}, row
        pair = tuple(row["pair"])
        assert pair in pairs, f"must_differ names a pair the baseline lacks: {pair}"
        assert pair not in seen, f"must_differ repeats {pair}"
        seen.add(pair)
        assert len(row["reason"].split()) >= 8, f"{pair} needs a real reason"


def test_primitive_digest_sites_only_shrink():
    """Raw digest sites outside the owners may only disappear (#1508)."""
    live = set(inventory.scan()["primitive_digest_sites"])
    base = set(_baseline()["primitive_digest_sites"])
    assert not live - base, (
        f"new primitive digest sites outside {sorted(inventory.DIGEST_OWNERS)}: "
        f"{sorted(live - base)} -- use the digest owners' named profiles instead "
        "of a new raw hashlib or sorted-JSON call (PQ #1508)")
    assert not base - live, (
        f"baseline digest sites gone {sorted(base - live)}: shrink the baseline "
        "with tools/duplication_inventory.py --write-baseline")


def test_the_digest_site_ratchet_fails_on_a_new_raw_site_and_passes_the_owner(tmp_path):
    """RED/GREEN for the gate itself: one raw site outside the owners fails.

    A fixture module with a fresh ``hashlib.sha256(json.dumps(...,
    sort_keys=True))`` site is an offender against an empty baseline (the RED
    side: this is exactly the new-code pattern the ratchet exists to stop),
    while the same raw recipe inside ``prismaquant/digests.py`` is not
    ratcheted at all because that is where consolidation moves sites to.
    """
    raw = (
        "import hashlib\nimport json\n"
        "def _digest(value):\n"
        "    return hashlib.sha256(\n"
        "        json.dumps(value, sort_keys=True).encode()).hexdigest()\n"
    )
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "prismaquant" / "digests.py").write_text(raw)
    (tmp_path / "tools" / "fresh_site.py").write_text(raw)
    live = set(inventory.scan(tmp_path)["primitive_digest_sites"])
    # The owner's own raw site is not ratcheted.
    assert not any("prismaquant/digests.py" in site for site in live)
    # Against an empty baseline the fresh raw site is a violation (RED).
    assert sorted(live - set()) == [
        "hashlib:tools/fresh_site.py::_digest",
        "sorted-json:tools/fresh_site.py::_digest",
    ]
    # With the site recorded, the same set arithmetic the shrink test uses
    # reports no violation (GREEN).
    baseline = {"hashlib:tools/fresh_site.py::_digest",
                "sorted-json:tools/fresh_site.py::_digest"}
    assert not live - baseline and not baseline - live


def test_image_side_patch_scripts_are_outside_the_scan_and_their_loader_is_not(tmp_path):
    """A serving-runtime patch script runs inside the pinned image, where the
    digest owners cannot be imported, and its bytes are bound to the image
    digest; the exclusion covers that directory and nothing beside it."""
    raw = ("import hashlib\nfrom pathlib import Path\n"
           "DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n")
    patch_dir = tmp_path / "prismaquant" / "serving_runtime_patches" / "some_set"
    patch_dir.mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (patch_dir / "patch_some_set.py").write_text(raw)
    (tmp_path / "prismaquant" / "serving_runtime_patch_set.py").write_text(raw)
    live = set(inventory.scan(tmp_path)["primitive_digest_sites"])
    assert live == {"hashlib:prismaquant/serving_runtime_patch_set.py::<module>"}
