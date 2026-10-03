"""Real-byte prefix compatibility; no GPU or serving qualification."""
import hashlib
import json
import os

import pytest
import torch
from safetensors.torch import save

from prismaquant.calibration_data import _decode_calibration_buffer, load_calibration_input
from prismaquant.layer_streaming import _advise_consumed_safetensors_pages
from prismaquant.source_read_plan import safetensors_prefix_length


@pytest.mark.parametrize("raw", [b"", b"\x00", b"\xff" * 7, bytes(8),
                                 (1).to_bytes(8, "little"),
                                 (2**64 - 1).to_bytes(8, "little")])
def test_malformed_prefix_keeps_each_consumer_refusal(raw, tmp_path):
    error = ("exact calibration input has no safetensors header length"
             if len(raw) < 8 else "exact calibration input header length is out of range")
    with pytest.raises(ValueError) as exc:
        _decode_calibration_buffer(raw)
    assert str(exc.value) == error
    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as exc:
        _advise_consumed_safetensors_pages(str(path), [])
    assert str(exc.value) == "invalid safetensors header for consumed-page release"


@pytest.mark.parametrize("length,size,limit,valid", [
    (1, 9, 1, True), (2, 9, 2, False), (2, 10, 1, False),
    (100_000_000, 100_000_008, 100_000_000, True),
    (100_000_001, 100_000_009, 100_000_000, False),
    (2**64 - 1, 100_000_008, 100_000_000, False),
])
def test_owner_preserves_inherited_unsigned_bounds(length, size, limit, valid):
    raw = length.to_bytes(8, "little")
    # Independent inherited predicate; no files/payload allocation at the cap.
    assert (0 < int.from_bytes(raw, "little") <= min(limit, size - 8)) is valid
    if valid:
        assert safetensors_prefix_length(raw, size, max_bytes=limit,
                                        short_error="short", range_error="range") == length
    else:
        with pytest.raises(ValueError, match="^range$"):
            safetensors_prefix_length(raw, size, max_bytes=limit,
                                      short_error="short", range_error="range")


def test_calibration_public_api_and_buffer_preserve_exact_draw(tmp_path):
    ids = torch.arange(32, dtype=torch.int64).reshape(4, 8)
    provenance = {"source": "fixture", "fit_tokens": 32, "nsamples": 4, "seqlen": 8,
                  "fit_ids_sha256": hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    raw = save({"calibration_ids": ids}, metadata={"calibration_provenance": json.dumps(provenance)})
    decoded, metadata = _decode_calibration_buffer(raw)
    assert decoded.numpy().tobytes() == ids.numpy().tobytes()
    assert metadata == provenance
    path = tmp_path / "tokens.safetensors"
    path.write_bytes(raw)
    result = load_calibration_input(path, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                    n_samples=4, seqlen=8)
    assert result[0].numpy().tobytes() == ids.numpy().tobytes()


def test_exact_end_header_still_uses_each_local_parser(tmp_path):
    # Prefix is valid at EOF; calibration refuses missing metadata, while an
    # empty page-release roster is valid and emits no payload advice.
    raw = (2).to_bytes(8, "little") + b"{}"
    with pytest.raises(ValueError, match="^exact calibration input requires calibration_provenance JSON$"):
        _decode_calibration_buffer(raw)
    path = tmp_path / "empty.safetensors"
    path.write_bytes(raw)
    _advise_consumed_safetensors_pages(str(path), [])
    assert path.read_bytes() == raw
