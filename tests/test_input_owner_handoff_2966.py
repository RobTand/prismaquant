"""Portable input deliveries for #2966, not a synthetic coverage proof.

The existing input owners supply docs/measurements/pq2966-input-handoff/
{a8s,balanced,qwen}.json and their referenced files. Each reference is
{path: repository-relative path, sha256: hex digest, bytes: positive integer}.
No fleet reads, historical git objects, models, or downloads are used here.

Population handoff envelopes contain assignment_sha256, source_binding,
metadata and candidates. Metadata is the original metadata object (or its
lossless JSON extraction), retaining population, target_profile and
tessera_serving_scope.
Qwen envelopes contain source_binding and references named measurement_scope,
consumer_expectations, refusals and wire_evidence. The wire_evidence JSON
contains a nonempty wires list of references to the actual serialized bytes.
source_binding records origin, source_sha256, extraction and owner_evidence.

These checks establish readable, content-bound delivery and basic population
integrity only. They cannot authenticate an owner assertion or prove an
extraction lossless without the original inputs; the reviewer must verify
that separately. Passing is NOT image-X coverage or Qwen corpus admission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs" / "measurements" / "pq2966-input-handoff"
ASSIGNMENTS = {
    "a8s": "153a4d53f2672098242a1e6b2d5e6acfb2fee2ca17496b66f97e2cbf395adc9d",
    "balanced": "b36e6634dae0cacd0ce6469eebe921e67bc10b4d5b49329fcb14e759e4b3ab8f",
}


def _delivery(name):
    path = HANDOFF / f"{name}.json"
    assert path.is_file(), (
        f"#2966 authentic {name} input handoff is missing: {path.relative_to(ROOT)}; "
        "supply it through the existing owner, not from export counts or fabricated memberships"
    )
    payload = json.loads(path.read_text())
    source = payload["source_binding"]
    for key in ("origin", "extraction", "owner_evidence"):
        assert isinstance(source[key], str) and source[key].strip(), key
    assert re.fullmatch(r"[0-9a-f]{64}", source["source_sha256"])
    return payload


def _bytes(ref):
    relative = Path(ref["path"])
    assert not relative.is_absolute() and ".." not in relative.parts, ref
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), ref
    assert type(ref["bytes"]) is int and ref["bytes"] > 0, ref
    assert path.stat().st_size == ref["bytes"], ref
    assert re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), ref
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == ref["sha256"], ref
    return path


def _json(ref):
    value = json.loads(_bytes(ref).read_text())
    assert value, "an empty extraction does not supply the missing input"
    return value


@pytest.mark.parametrize("allocation", ["a8s", "balanced"])
def test_retained_pre_route_population_is_delivered(allocation):
    handoff = _delivery(allocation)
    assert handoff["assignment_sha256"] == ASSIGNMENTS[allocation]
    metadata = _json(handoff["metadata"])
    for key in ("population", "target_profile", "tessera_serving_scope"):
        assert metadata.get(key), f"{allocation}: missing original {key}"
    population = metadata["population"]
    all_units = set()
    for structure in ("dense", "routed_experts"):
        enumerated = population["enumerated"][structure]
        priced = population["priced"][structure]
        unpriced = population["unpriced"][structure]
        assert isinstance(enumerated, list) and isinstance(priced, list)
        assert isinstance(unpriced, dict)
        assert all(isinstance(unit, str) and unit for unit in enumerated)
        assert len(set(enumerated)) == len(enumerated), "duplicate population membership"
        assert len(set(priced)) == len(priced), "duplicate priced membership"
        assert not set(priced) & set(unpriced), "priced/unpriced overlap"
        assert set(enumerated) == set(priced) | set(unpriced), "lossy population partition"
        assert all(isinstance(reason, str) and reason for reason in unpriced.values())
        assert not all_units & set(enumerated), "conflicting structures"
        all_units.update(enumerated)
    assert all_units, "aggregate counts or export-only rosters are not selectable populations"
    # Preserve the existing owner's candidate representation rather than invent
    # a second scientific/menu schema in this delivery test.
    _json(handoff["candidates"])


def test_qwen_scientific_inputs_and_actual_wire_bytes_are_delivered():
    handoff = _delivery("qwen")
    for role in ("measurement_scope", "consumer_expectations", "refusals"):
        _json(handoff[role])
    evidence = _json(handoff["wire_evidence"])
    assert isinstance(evidence["wires"], list) and evidence["wires"], (
        "wire formulas, aggregate byte totals and profile layouts are not actual wire evidence"
    )
    for wire in evidence["wires"]:
        _bytes(wire)
