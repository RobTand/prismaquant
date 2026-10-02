"""Authenticated CPU fixture operands from the completed FP32 screen.

Only the crosscheck observations are reused. Tests do not run that model or
reinterpret these operands as a new scientific measurement.
"""
import hashlib
import io
import json
from pathlib import Path

import torch


ROOT = Path('/mnt/shared/tessera-measurements/pq1962-sol-20261002/pair2')


def observations():
    raw_pricing = (ROOT / 'float32.pricing.pt').read_bytes()
    raw_summary = (ROOT / 'float32.json').read_bytes()
    assert hashlib.sha256(raw_pricing).hexdigest() == '311be7d24dfaae6764dd1da10645a8ec49454aae53cdfe03346da5fcde66faeb'
    assert hashlib.sha256(raw_summary).hexdigest() == 'b5bfe18476ceedc24486506795289dadb783d5e2309dc189a572623a4ffb2a87'
    pricing = torch.load(io.BytesIO(raw_pricing), map_location='cpu', weights_only=True)
    crosscheck = json.loads(raw_summary)['profile']['arm_crosscheck']
    return pricing['comps'][:, :1, :2, :1].double().clone(), crosscheck
