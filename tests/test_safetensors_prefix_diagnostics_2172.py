"""Exact content/metadata prefix refusals; no serving qualification."""
import hashlib
import os

import pytest
import torch
from safetensors.torch import save

from prismaquant import export_structure, shipcard
from prismaquant.source_read_plan import safetensors_prefix_length


@pytest.mark.parametrize("raw", [b"", b"\x00", b"\xff" * 7,
    bytes(8), (1).to_bytes(8, "little"), (2**64 - 1).to_bytes(8, "little")])
def test_real_file_prefix_refusals_retain_exact_distinct_text(tmp_path, raw):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(raw)
    short = len(raw) < 8
    length = int.from_bytes(raw, "little")
    with pytest.raises(ValueError) as exc:
        export_structure._read_metadata(path, shard=True)
    expected = (f"{path}: truncated header length" if short else
                f"{path}: invalid or truncated metadata length {length}")
    assert str(exc.value) == expected
    with path.open("rb") as handle:
        with pytest.raises(ValueError) as exc:
            shipcard._verify_open_safetensors_fd(
                handle.fileno(), name=path.name, initial_stat=os.fstat(handle.fileno()))
    expected = ("truncated safetensors content during verification" if short else
                f"{path.name}: invalid safetensors header length {length}")
    assert str(exc.value) == expected


def test_over_cap_sparse_file_refuses_before_header_body(tmp_path):
    path = tmp_path / "over-cap.safetensors"
    length = shipcard._MAX_SAFETENSORS_HEADER_BYTES + 1
    with path.open("wb") as handle:
        handle.write(length.to_bytes(8, "little"))
        handle.truncate(length + 8)
    with pytest.raises(ValueError) as exc:
        export_structure._read_metadata(path, shard=True)
    assert str(exc.value) == f"{path}: invalid or truncated metadata length {length}"
    with path.open("rb") as handle:
        with pytest.raises(ValueError) as exc:
            shipcard._verify_open_safetensors_fd(
                handle.fileno(), name=path.name, initial_stat=os.fstat(handle.fileno()))
        assert handle.tell() == 8
    assert str(exc.value) == f"{path.name}: invalid safetensors header length {length}"


def test_content_hash_and_metadata_bytes_keep_their_independent_read_contracts(tmp_path):
    tensor = torch.tensor([0, 255, 17, 0, 128], dtype=torch.uint8)
    raw = save({"weight": tensor})
    path = tmp_path / "weights.safetensors"
    path.write_bytes(raw)
    header_length = int.from_bytes(raw[:8], "little")
    metadata, consumed, _ = export_structure._read_metadata(path, shard=True)
    assert metadata == raw[8:8 + header_length]
    assert consumed == 8 + header_length
    with path.open("rb") as handle:
        record, consumed, calls = shipcard._verify_open_safetensors_fd(
            handle.fileno(), name=path.name, initial_stat=os.fstat(handle.fileno()))
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["tensor_sha256"] == {"weight": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    assert consumed == len(raw)
    assert calls == 3


def test_json_index_size_is_not_interpreted_as_a_prefixed_shard(tmp_path):
    path = tmp_path / "index.json"
    path.write_bytes(b"{}")
    assert export_structure._read_metadata(path, shard=False)[:2] == (b"{}", 2)
    path.write_bytes(b"")
    with pytest.raises(ValueError) as exc:
        export_structure._read_metadata(path, shard=False)
    assert str(exc.value) == f"{path}: invalid or truncated metadata length 0"


def test_owner_keeps_literal_string_subclasses_literal():
    class Message(str):
        def __call__(self, length):
            pytest.fail("a literal error string is not a diagnostic factory")
    message = Message("literal range message")
    with pytest.raises(ValueError) as exc:
        safetensors_prefix_length(bytes(8), 8, max_bytes=2,
                                  short_error="short", range_error=message)
    assert exc.value.args == (message,)
