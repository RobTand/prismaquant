"""Prove the device envelope against the REAL cuda allocator, on a real device.

Run inside the admitted campaign container on a GPU box:

    python3 -m tools.device_envelope_probe --envelope-gib 1

This is the check the mocked unit test cannot make for the caller: the real
``torch.cuda.set_per_process_memory_fraction`` refuses the UNSPECIFIED device
form (``Expected a torch.device with a specified index or an integer, but got:
cuda``) while ``get_device_properties`` accepts it, so an envelope set with the
plan's ``cuda`` never reached the allocator at all. The probe sets the envelope
the way the pass does, reads the fraction back through torch's own getter, and
then shows the bound BINDS: a small allocation succeeds and one larger than the
envelope is refused inside this row. It allocates nothing large, touches no
model, and leaves the process.

Exit status is the verdict; the JSON record on stdout is the receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

GiB = 1024 ** 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope-gib", type=int, default=1)
    parser.add_argument("--small-mib", type=int, default=64)
    parser.add_argument("--overshoot-gib", type=int, default=2)
    args = parser.parse_args(argv)

    import torch

    from prismaquant import memory_management as mm

    envelope_bytes = args.envelope_gib * GiB
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    record = {
        "schema": "prismaquant.device_envelope_probe.v1",
        "host": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_index": index,
        "device_name": props.name,
        "device_total_bytes": int(props.total_memory),
        "envelope_bytes": envelope_bytes,
        "resident_before_bytes": int(torch.cuda.memory_reserved(index)),
    }
    try:
        envelope = mm.enforce_device_envelope(
            "cuda", envelope_bytes, where="device envelope probe")
    except Exception as error:                      # noqa: BLE001 - the verdict
        record["enforced"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(record, indent=1, sort_keys=True), flush=True)
        return 1
    record["envelope_record"] = envelope
    assert envelope["enforced"] is True, envelope
    assert envelope["allocator_device_index"] == index, (envelope, index)
    # Read it back through torch's own api, on the index the envelope named.
    reader = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if reader is not None:
        observed = reader(index)
        record["fraction_readback"] = observed
        record["fraction_expected"] = envelope_bytes / int(props.total_memory)
        record["fraction_roundtrip_matches"] = bool(
            abs(observed - record["fraction_expected"]) < 1e-9)
        assert record["fraction_roundtrip_matches"], record
    else:
        record["fraction_readback"] = None
        record["fraction_roundtrip_matches"] = "unsupported by this torch"
    # A small allocation must be handed out ...
    small = torch.empty(args.small_mib * 1024 ** 2 // 4, dtype=torch.float32,
                        device=f"cuda:{index}")
    assert small.numel() > 0
    del small
    # ... and one larger than the envelope must be refused HERE, not on the box.
    try:
        torch.empty(args.overshoot_gib * GiB // 4, dtype=torch.float32,
                    device=f"cuda:{index}")
    except RuntimeError as error:
        record["overshoot_refused"] = "out of memory" in str(error).lower()
        record["overshoot_error"] = str(error).splitlines()[0][:200]
    else:
        record["overshoot_refused"] = False
    record["resident_after_bytes"] = int(torch.cuda.memory_reserved(index))
    record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
    record["passed"] = bool(record["overshoot_refused"])
    print(json.dumps(record, indent=1, sort_keys=True), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
