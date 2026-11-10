"""Golden replay of an actual retained receipt, without the retired SDK."""
import copy
import json
from pathlib import Path

from prismaquant.recorded_cas_receipt import cas_receipt_self_check


def test_real_recorded_receipt_and_a_one_bit_digest_change():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "docs/measurements/artifacts/pq183-lfm-bound-2026-09-05/run09-cas-receipt.json").read_bytes())
    assert cas_receipt_self_check(receipt) is None
    damaged = copy.deepcopy(receipt)
    digest = damaged["receipt_sha256"]
    damaged["receipt_sha256"] = f"{int(digest[0], 16) ^ 1:x}" + digest[1:]
    assert cas_receipt_self_check(damaged) == "cas-receipt-digest"
