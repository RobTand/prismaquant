"""A warm plan's length is a claim about a file, so it stops at the file.

``Campaign.weight_extents`` rounds each tensor's byte range out to 1 MiB
record boundaries because ZFS reads a whole record either way.  The last
tensor in a shard almost never ends on such a boundary, and rounding its end
out unconditionally declared bytes that do not exist: every one of the 16
GLM resume rows over-declared about 2.5 MB, so every warm through the
PrismaBuild prewarm loop recorded ``status: partial`` even after reading
every byte there was (rows 0085 and 0086, 2026-09-11).

The fixture is a real safetensors shard whose data ends mid-record at the end
of the file, which is the shape that produced the defect; the test asserts
that rounding would have run past the file before asserting that it does not.
"""

import json
import struct

from experiments.glm_arc_prewarm import RECORD_SIZE, Campaign

#: Ends 512 KiB into a record, so rounding its end out leaves the file.
TENSOR_BYTES = (3 * RECORD_SIZE) + (RECORD_SIZE // 2)
TENSOR = "model.layers.0.mlp.experts.0.down_proj"


def _campaign(tmp_path):
    """A one-row campaign over a one-tensor shard, on disk, nothing mocked."""

    model = tmp_path / "model"
    model.mkdir()
    header = json.dumps({
        TENSOR + ".weight": {"dtype": "F8_E4M3", "shape": [TENSOR_BYTES],
                             "data_offsets": [0, TENSOR_BYTES]},
    }).encode()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * TENSOR_BYTES)
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {TENSOR + ".weight": shard.name}}))

    units = tmp_path / "row-0000.json"
    units.write_text(json.dumps({"groups": [{"members": [TENSOR]}]}))
    captures = tmp_path / "captures"
    captures.mkdir()
    manifest = captures / "manifest.json"
    manifest.write_text(json.dumps({"entries": {}}))
    (tmp_path / "plan.json").write_text(json.dumps({
        "model": str(model),
        "calibration_cache": {"path": str(manifest)},
        # ``groups`` is the campaign's own label for the row and is what
        # ``row_plan`` reads; the real plan.json carries it on every row.
        "rows": [{"row_id": "row-0000", "units": str(units),
                  "groups": ["expert-group-0"]}],
    }))
    return Campaign(str(tmp_path)), shard


def test_a_weight_extent_never_declares_a_byte_past_the_end_of_its_shard(
    tmp_path,
):
    campaign, shard = _campaign(tmp_path)
    size = shard.stat().st_size

    extents = campaign.weight_extents("row-0000")

    assert len(extents) == 1, extents
    path, offset, length = extents[0]
    assert path == str(shard)
    # The fixture really is the case that produced the defect: the record the
    # tensor ends inside reaches past the end of the file.
    assert size % RECORD_SIZE != 0
    assert size + (-size % RECORD_SIZE) > size
    assert offset + length <= size, "the manifest declares bytes that do not exist"
    # Nothing is given up either: the warm still covers the whole tensor.
    assert offset == 0 and offset + length == size
