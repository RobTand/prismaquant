"""Shared synthetic inputs for footprint tests."""
from __future__ import annotations

import json
import struct

from prismaquant import footprint as fp


def _write_safetensors(path, tensors):
    """Write a minimal valid .safetensors file. tensors: {name: (dtype, shape)}.

    Data is zero-filled; only the header (dtype/shape/data_offsets) matters for
    the byte accounting, which reads spans, not values.
    """
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)
