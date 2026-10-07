"""Compare the port with accepted arithmetic on real CPU tensors and arrays."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from prismaquant.digests import file_sha256hex
from prismaquant.quality_stage import artifact, write_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--retained-result", type=Path, required=True)
    parser.add_argument("--retained-array", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.reference_root.resolve(strict=True)))
    old_exl = importlib.import_module("accepted.exl3_torch")
    old_lib = importlib.import_module("accepted.g3_lib")
    old_kl = importlib.import_module("accepted.full_vocab")
    old_mixed = importlib.import_module("mixed_injection")
    from prismaquant import g3_exl3, g3_numerics, g3_activation
    x = torch.linspace(-449, 449, 7*64, dtype=torch.float32).reshape(7, 64).to(torch.bfloat16)
    x[0, 0], x[0, 1] = 0.0, -0.0
    old_codes, old_scales = old_lib.fp8_per_token_dynamic(x)
    codes, scales = g3_numerics.fp8_per_token_dynamic(x)
    checks = {"fp8_codes": torch.equal(old_codes.view(torch.uint8), codes.view(torch.uint8)),
              "fp8_scales": torch.equal(old_scales.view(torch.uint8), scales.view(torch.uint8))}
    for contract, value in ((old_mixed.T8, None), (old_mixed.T4, 1.0), (old_mixed.BF16, None)):
        for tp in (1, 2):
            left = old_mixed.quantized_rows_fp32(x, contract, input_global_scale=value, tp_splits=tp)
            right = g3_activation.quantized_rows_fp32(x, contract, input_global_scale=value, tp_splits=tp)
            checks[f"activation/{contract}/tp{tp}"] = torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    trellis = torch.arange(8*8*64, dtype=torch.int32).mul(37).to(torch.int16).reshape(8, 8, 64)
    suh = torch.linspace(0.25, 0.75, 128).half()
    svh = torch.linspace(0.125, 0.5, 128).half()
    left = old_exl.effective_weight(trellis, suh, svh)
    right = g3_exl3.effective_weight(trellis, suh, svh)
    checks["exl3_fp32"] = torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    checks["exl3_bf16"] = torch.equal(left.bfloat16().view(torch.uint8), right.bfloat16().view(torch.uint8))
    raw = old_exl.pack_wire(suh.numpy().tobytes(), svh.numpy().tobytes(), trellis.numpy().tobytes(),
                           np.array([old_exl.MCG_KERNEL], dtype=np.uint32).tobytes())
    checks["exl3_wire"] = torch.equal(old_exl.decode_wire(raw, 128, 128).view(torch.uint8),
                                       g3_exl3.decode_wire(raw, 128, 128).view(torch.uint8))
    t = torch.arange(17*33, dtype=torch.float32).reshape(17, 33)/31
    c = t+torch.linspace(-0.1, 0.1, 33)
    checks["full_vocab_fp64_kl"] = torch.equal(old_kl.token_kl(t, c, tile_rows=5, require_cuda=False),
                                               g3_numerics.token_kl(t, c, tile_rows=5, require_cuda=False))
    retained = np.load(args.retained_array, allow_pickle=False)
    result = json.loads(args.retained_result.read_bytes())
    if retained.shape != (25, 2047) or not np.isfinite(retained).all():
        raise ValueError("retained accepted G3 population differs")
    mean = float(np.concatenate(list(retained)).astype(np.float64).mean())
    expected = result["teacher2"]["mean_kl"]
    per_window = [float(row.mean()) for row in retained]
    unit_roundoff = np.finfo(np.float64).eps / 2
    terms = retained.size
    reduction_bound = (terms*unit_roundoff)/(1-terms*unit_roundoff)*float(np.abs(retained).mean())
    checks["retained_mean_within_fp64_reduction_bound"] = abs(mean-expected) <= reduction_bound
    checks["retained_window_reduction"] = per_window == result["teacher2"]["per_window_mean"]
    if not all(checks.values()):
        raise ValueError("CPU numerical parity failed: "+repr(checks))
    reference_files = [artifact(args.reference_root/path) for path in
        ("accepted/exl3_torch.py", "accepted/g3_lib.py", "accepted/full_vocab.py", "mixed_injection.py")]
    report = {"schema": "prismaquant.quality_cpu_parity/1", "checks": checks, "device": "cpu", "skips": [],
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "accepted_source_head": "658d6b56af22970d862dab167668176e4329c1d3", "reference_files": reference_files,
        "retained_result": artifact(args.retained_result), "retained_array": artifact(args.retained_array),
        "retained_mean": mean, "retained_mean_difference": mean-expected,
        "retained_mean_absolute_difference": abs(mean-expected),
        "retained_mean_relative_difference": abs(mean-expected)/abs(expected) if expected else None,
        "retained_mean_bitwise": mean == expected, "fp64_reduction_bound": reduction_bound,
        "reduction_dtype": "float64", "reduction_order": "concatenate ordered window rows, cast float64, NumPy mean",
        "reduction_bound_derivation": "gamma_N times mean(abs(values)); unit roundoff is float64 epsilon divided by two",
        "limitations": ["CPU parity does not establish native GPU equality.",
                        "Retained arrays prove reduction parity, not a new full-model measurement."]}
    write_result(args.output, report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
