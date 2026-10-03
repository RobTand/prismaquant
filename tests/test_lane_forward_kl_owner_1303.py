"""Lane replay keeps exact teacher pairing and per-row KL reductions (#1303)."""
from __future__ import annotations

import pytest
import torch

from prismaquant import kl_measurement as km
from test_forward_kl_owner_1303 import _raw, _record_owner


@pytest.mark.parametrize("full_sequence", [False, True])
@pytest.mark.parametrize("microbatch", [1, 2, 3])
def test_lane_consumer_routes_each_teacher_group_and_keeps_row_normalization(monkeypatch, full_sequence, microbatch):
    teacher_logits = (torch.arange(3 * 4 * 7).reshape(3, 4, 7) % 17 - 8).float() / 4
    teacher = torch.log_softmax(teacher_logits, -1)
    stacked = torch.stack([teacher_logits.flip(-1), teacher_logits * 2])
    refs = [teacher[i:i + microbatch] for i in range(0, 3, microbatch)]
    if not full_sequence:
        stacked = stacked[:, :, -1:, :]
    expected = torch.zeros(2)
    offset = 0
    for ref in refs:
        if not full_sequence:
            ref = ref[:, -1:, :]
        lp = torch.log_softmax(stacked[:, offset:offset + len(ref)].float(), -1)
        old = (ref.exp().unsqueeze(0) * (ref.unsqueeze(0) - lp)).sum(-1)
        expected += old.mean(dim=2).sum(dim=1)
        offset += len(ref)
    calls = _record_owner(monkeypatch, km)
    actual = km._replay_lane_kl_totals(stacked, refs, full_sequence_kl=full_sequence)
    assert _raw(actual) == _raw(expected)
    assert len(calls) == len(refs)
