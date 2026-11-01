"""Real research packer and schedule-summary boundaries; no native reader claim."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.test_pq_block_helpers import QNAME, make_synthetic_parents, wire

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-block-parents")
    parents, entries = make_synthetic_parents(directory)
    tags = (np.arange(32).reshape(2, 16) % 2).astype(np.int64)
    blob, breakdown = wire.pack_projection(parents, tags, 64, 2)
    padded_blob, padded = wire.pack_projection(parents, tags, 64, 2, target_bytes=len(blob) + 32)
    assert breakdown["pad_bytes"] == 0 and padded["pad_bytes"] == 32
    return parents, entries, tags, (blob, breakdown), (padded_blob, padded)


@pytest.fixture
def schedule(monkeypatch):
    # The public script resolves its sibling modules from tools/. Reproduce
    # that import context while calling its actual main in this PB process.
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return importlib.import_module("pq_block_schedule_cost")


def _invoke(schedule, encoded, document, blob, entrypoint, tmp_path, monkeypatch):
    parents, entries, tags, _zero, _padded = encoded
    if entrypoint == "library":
        return schedule.summarize_schedule(parents, tags, 64, 2, document)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"schema": schedule.PARENT_BANK_SCHEMA,
                                "qname": QNAME, "parents": entries}))
    selection = tmp_path / "selection.npy"
    np.save(selection, tags)
    breakdown = tmp_path / "breakdown.json"
    breakdown.write_text(json.dumps(document))
    packed = tmp_path / "packed.bin"
    packed.write_bytes(blob)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        str(ROOT / "tools/pq_block_schedule_cost.py"),
        "--parent-bank", str(bank), "--selection", str(selection),
        "--breakdown", str(breakdown), "--packed-blob", str(packed), "--output", str(output)])
    schedule.main()
    return json.loads(output.read_bytes())


@pytest.mark.parametrize("entrypoint", ["library", "cli"])
def test_negative_padding_cannot_publish_a_successful_stored_byte_count(
    schedule, encoded, tmp_path, monkeypatch, entrypoint,
):
    blob, original = encoded[3]
    bad = copy.deepcopy(original)
    bad["pad_bytes"] = -1
    bad["total_bytes"] = len(blob) - 1
    refused = None
    summary = None
    try:
        summary = _invoke(schedule, encoded, bad, blob, entrypoint, tmp_path, monkeypatch)
    except (ValueError, RuntimeError, SystemExit) as exc:
        refused = str(exc)
    assert refused is not None, f"accepted impossible padding: {summary['stored_bytes']}"
    assert "pad" in refused.lower(), refused
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("entrypoint", ["library", "cli"])
@pytest.mark.parametrize("padding", [0, 32])
def test_actual_zero_and_legitimate_padded_wires_keep_exact_byte_totals(
    schedule, encoded, tmp_path, monkeypatch, entrypoint, padding,
):
    blob, document = encoded[3 if padding == 0 else 4]
    summary = _invoke(schedule, encoded, document, blob, entrypoint, tmp_path, monkeypatch)
    stored = summary["stored_bytes"]
    assert stored["planes_bytes"]["pad_bytes"] == padding
    assert stored["total_bytes"] == len(blob)
    assert stored["total_bits_per_stored_weight"] == len(blob) * 8 / (128 * 32)
    assert sum(stored["planes_bytes"].values()) == len(blob)
    assert summary["claims"]["native_reader_qualified"] is False


def test_actual_packer_already_refuses_a_budget_smaller_than_its_content(encoded):
    parents, _entries, tags, (blob, _breakdown), _padded = encoded
    with pytest.raises(wire.BlockWireFormatError, match="overflow"):
        wire.pack_projection(parents, tags, 64, 2, target_bytes=len(blob) - 1)


@pytest.mark.parametrize("entrypoint", ["library", "cli"])
@pytest.mark.parametrize("padding, coerced_total_extra", [
    (True, 1), (False, 0), (1.5, 1), (1.0, 1), (0.0, 0),
    ("0", 0), ("1", 1), (float("nan"), 0),
    (float("inf"), 0), (float("-inf"), 0), (None, 0),
], ids=["bool-true", "bool-false", "fractional", "float-integer", "float-zero",
        "string-zero", "string-one", "nan", "inf", "negative-inf", "null"])
def test_noninteger_padding_is_refused_at_breakdown_intake(
    schedule, encoded, tmp_path, monkeypatch, entrypoint, padding, coerced_total_extra,
):
    blob, original = encoded[3]
    bad = copy.deepcopy(original)
    bad["pad_bytes"] = padding
    # Keep the total that the old coercion could accept for finite values.
    # The public API must refuse the raw domain before publishing any report.
    bad["total_bytes"] = len(blob) + coerced_total_extra
    with pytest.raises(ValueError, match="pad_bytes.*nonnegative integer"):
        _invoke(schedule, encoded, bad, blob, entrypoint, tmp_path, monkeypatch)
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("entrypoint", ["library", "cli"])
def test_required_padding_count_cannot_be_omitted(
    schedule, encoded, tmp_path, monkeypatch, entrypoint,
):
    blob, original = encoded[3]
    bad = copy.deepcopy(original)
    del bad["pad_bytes"]
    with pytest.raises(ValueError, match="pad_bytes.*nonnegative integer"):
        _invoke(schedule, encoded, bad, blob, entrypoint, tmp_path, monkeypatch)
    assert not (tmp_path / "summary.json").exists()
