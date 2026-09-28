"""Golden byte tests for the tessera_calibration_cache digest consolidation (PQ #1512).

The fixture records the OLD outputs of every consolidated site, captured on the
pre-change tree (ratchet head bddf455bdba) by a PrismaBuild run of the golden
record script.  Each test recomputes the same row through the NEW code path and
asserts equality, so receipts, seals and identity chains recorded before the
change still verify.
"""

import json
from pathlib import Path

import pytest

from prismaquant.digests import (
    DIRECT_ASCII_LAX,
    DIRECT_ASCII_STRICT,
    bytes_sha256hex,
    hex_chain_sha256hex,
    indent2_json_file_bytes,
    text_sha256hex,
)
from prismaquant.tessera_calibration_cache import (
    _json,
    _load_execution,
    fold_load_receipt,
    merge_load_execution,
)

FIXTURE = Path(__file__).parent / "fixtures" / "digest_calibration_cache_1512.json"


def _table():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document["inputs"], document["old_rows"]


def test_the_pretty_file_profile_reproduces_the_old_capture_form():
    inputs, rows = _table()
    for index, value in enumerate(inputs["json_file_values"]):
        assert indent2_json_file_bytes(value).hex() == rows[f"A{index}"]["hex"]


def test_the_changed_writer_writes_those_bytes(tmp_path):
    inputs, rows = _table()
    for index, value in enumerate(inputs["json_file_values"]):
        target = tmp_path / f"capture-{index}.json"
        _json(target, value)
        assert target.read_bytes().hex() == rows[f"A{index}"]["hex"]


def test_strict_compact_text_reproduces_the_old_identity_and_policy_rows():
    inputs, rows = _table()
    for index, value in enumerate(inputs["identities"]):
        assert DIRECT_ASCII_STRICT.text(value) == rows[f"B{index}"]
    for index, value in enumerate(inputs["policies"]):
        assert DIRECT_ASCII_STRICT.text(value) == rows[f"C{index}"]


def test_the_streamed_descriptor_hash_reproduces_the_old_lax_encoder():
    inputs, rows = _table()
    descriptors = [
        dict(schema="prismaquant.capture_load_execution.v1", policy=inputs["policies"][0],
             capture_identity=inputs["identities"][0]),
        dict(schema="prismaquant.capture_load_execution.v1", policy=inputs["policies"][1],
             capture_identity=inputs["identities"][1]),
        dict(schema="prismaquant.capture_load_execution.v1",
             policy=dict(inputs["policies"][0], relaxed=float("nan")), capture_identity={}),
    ]
    for index, descriptor in enumerate(descriptors):
        assert DIRECT_ASCII_LAX.sha256_streamed(descriptor) == rows[f"D{index}"]
        recorded = rows[f"D{index}_as_text"]
        if isinstance(recorded, dict) and "error" in recorded:
            with pytest.raises(ValueError):
                DIRECT_ASCII_STRICT.text(descriptor)
        else:
            assert DIRECT_ASCII_STRICT.text(descriptor) == recorded


def test_streamed_and_materialized_profile_hashes_agree_for_accepted_values():
    value = {"b": 1, "a": [2, 3.5, None, True], "s": "naïve ✓"}
    assert DIRECT_ASCII_LAX.sha256_streamed(value) == DIRECT_ASCII_LAX.sha256(value)
    assert DIRECT_ASCII_STRICT.sha256_streamed(value) == DIRECT_ASCII_STRICT.sha256(value)


def test_raw_byte_digest_sites_reproduce_the_old_hex():
    inputs, rows = _table()
    for index, blob_hex in enumerate(inputs["blobs_hex"]):
        assert bytes_sha256hex(bytes.fromhex(blob_hex)) == rows[f"E{index}"]


def test_receipt_chains_recorded_before_the_change_still_verify():
    inputs, rows = _table()
    policy = inputs["policies"][0]
    execution = _load_execution(policy, None, identity_sha256=None)
    assert execution["identity_sha256"] == rows["F_identity"]
    assert execution["ordered_load_identities_sha256"] == rows["F_seed"]
    for index, identity in enumerate(inputs["chain_hex"][1:]):
        receipt = {"identity_sha256": identity, "source_read_bytes": 1024 * (index + 1),
                   "file_bytes": 4096 * (index + 1), "archive_storage_bytes": 512 * (index + 1)}
        fold_load_receipt(execution, receipt)
        assert execution["ordered_load_identities_sha256"] == rows[f"F_fold{index}"]
    total = _load_execution(policy, None, identity_sha256=None)
    partial = _load_execution(policy, None, identity_sha256=None)
    fold_load_receipt(partial, {"identity_sha256": inputs["chain_hex"][2], "source_read_bytes": 7,
                                "file_bytes": 9, "archive_storage_bytes": 1})
    merge_load_execution(total, partial)
    assert total["ordered_load_identities_sha256"] == rows["F_merge"]


def test_the_chain_owner_is_the_old_recipe():
    left = "ab" * 32
    right = "cd" * 32
    assert hex_chain_sha256hex(left, right) == text_sha256hex(left + right)
    assert hex_chain_sha256hex("", "") == bytes_sha256hex(b"")
