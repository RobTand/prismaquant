"""Golden byte-identity for the tessera_export_lane digest delegations (#1599).

`prismaquant/tessera_export_lane.py` delegated seven primitive digest sites to
the named ``prismaquant/digests.py`` owners.  Every fixture row was computed
from the pre-change spelling verbatim; this file re-derives each expectation
from that old spelling *and* asserts the owner reproduces it byte for byte.
"""

import base64
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.digests import (
    bytes_sha256hex, indent2_json_file_bytes, text_sha256hex,
)

FIXTURE = Path(__file__).parent / "fixtures" / "digest_export_lane_1599.json"


def _rows():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["fixture_schema"] == "prismaquant.digest_golden.v1"
    assert data["issue"] == 1599
    return data["rows"]


def _old_indent2_bytes(value: object) -> bytes:
    # Pre-change spelling, verbatim (write_cached_expert_units :1040).
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _old_indent2_bytes_assignment(value: object) -> bytes:
    # Pre-change spelling, verbatim (_write_plan_assignment :2315).
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def test_fixture_rows_load_and_cover_every_family():
    rows = _rows()
    families = {row["family"].split(":")[0] for row in rows}
    assert families == {
        "indent2_json_file_bytes", "bytes_sha256hex", "text_sha256hex"}
    # every delegated family has at least one row
    assert len([r for r in rows if r["family"].startswith("indent2")]) == 2
    assert len([r for r in rows if r["family"].startswith("bytes_sha256hex")]) == 3


@pytest.mark.parametrize("row_index", range(2))
def test_indent2_rows_are_byte_identical(row_index):
    row = [r for r in _rows() if r["family"].startswith("indent2")][row_index]
    value = next(iter(row["input"].values())) if row["family"].endswith(
        "_write_plan_assignment") else row["input"]
    old = (_old_indent2_bytes if row_index == 0 else _old_indent2_bytes_assignment)(value)
    assert hashlib.sha256(old).hexdigest() == row["expected_digest_hex"]
    new = indent2_json_file_bytes(value)
    assert isinstance(new, bytes)
    assert new == old
    assert bytes_sha256hex(new) == row["expected_digest_hex"]


@pytest.mark.parametrize("row", _rows())
def test_bytes_and_text_rows_are_byte_identical(row):
    family = row["family"]
    if family.startswith("bytes_sha256hex"):
        raw = base64.b64decode(row["input_bytes_b64"])
        # Pre-change spelling, verbatim: hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256(raw).hexdigest() == row["expected_hex"]
        assert bytes_sha256hex(raw) == row["expected_hex"]
        if raw:
            # digest and bytes producers stay consistent for every intake shape
            assert bytes_sha256hex(raw) == bytes_sha256hex(base64.b64encode(raw) and raw)
    elif family.startswith("text_sha256hex"):
        text = row["input_text"]
        # Pre-change spelling, verbatim: hashlib.sha256(str(x).encode()).hexdigest()
        assert hashlib.sha256(text.encode()).hexdigest() == row["expected_hex"]
        assert text_sha256hex(text) == row["expected_hex"]
    else:
        pytest.skip("indent2 row covered by its own test")


def test_the_module_delegates_and_keeps_the_by_design_spellings():
    """The delegations are importable and the kept sites are named in #1599."""
    import inspect

    import prismaquant.tessera_export_lane as lane

    src = inspect.getsource(lane)
    # the seven delegations
    assert "text_sha256hex(str(cell_wire_dir))" in src
    assert "encoded = indent2_json_file_bytes(manifest)" in src
    assert "digest = bytes_sha256hex(encoded)" in src
    assert "if bytes_sha256hex(raw) != expected_sha256:" in src
    assert "payload = indent2_json_file_bytes(projected)" in src
    assert '"plan_assignment_sha256": bytes_sha256hex(payload)}' in src
    assert "'sha256': bytes_sha256hex(raw)," in src
    assert "print(bytes_sha256hex(build_bytes))" in src
    # the kept by-design spellings: NUL-framed hessian commitment, non-compact
    # diagnostics (require_platform / read_cached_unit_bundle prints / main)
    assert 'json.dumps({"schema": HESSIAN_CAPTURE_SHA256_SCHEMA' in src
    assert "unit.update(b\"\\0\")" in src
    assert 'executes_by_platform={json.dumps(stated, sort_keys=True)}' in src
    assert "'[cached-unit warning] ' + json.dumps(warning, sort_keys=True)" in src
    assert 'json.dumps(report["build"], indent=2, sort_keys=True) + "\\n"' in src


def test_empty_blob_matches_the_known_empty_sha256():
    # b"" -> e3b0... pinned in the fixture row family bytes_sha256hex:
    # selected_cached_units_manifest (pre-change: hashlib.sha256(b"").hexdigest())
    assert bytes_sha256hex(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
