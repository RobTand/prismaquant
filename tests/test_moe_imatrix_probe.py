from __future__ import annotations

import torch

from prismaquant.moe_imatrix import canonical_imatrix_sha256


def test_imatrix_value_hash_is_qname_sorted_and_value_sensitive():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert canonical_imatrix_sha256(first) == canonical_imatrix_sha256(second)
    second["b"] = torch.tensor([3.0])
    assert canonical_imatrix_sha256(first) != canonical_imatrix_sha256(second)
