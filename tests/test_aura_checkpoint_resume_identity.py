from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import prismaquant.aura_cost as aura


class _TinyLM(nn.Module):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.embed = nn.Embedding(23, 16)
        self.l1 = nn.Linear(16, 16, bias=False)
        self.l2 = nn.Linear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 23, bias=False)
        self.forward_calls = 0
        if state is not None:
            self.load_state_dict(state)

    def forward(self, input_ids):
        self.forward_calls += 1
        x = self.embed(input_ids)
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return SimpleNamespace(logits=self.lm_head(x))


def test_resident_aura_refuses_durable_checkpoints(tmp_path):
    # The resident path checkpointed only against the retired codebook lane's
    # cache-pair identity (archived 2026-09-25, #1304). With no value-bearing
    # render identity left to bind, it refuses before any forward instead of
    # resuming on model and cache names; durable AURA checkpoints are
    # streamed-only (tests/test_streamed_cost_checkpoints.py).
    model = _TinyLM()
    with pytest.raises(RuntimeError, match="no value-bearing render identity"):
        aura.compute_aura_cost(
            model,
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            ["NVFP4"],
            n_probes=2,
            n_linear_chunks=1,
            min_free_gib=0.0,
            checkpoint_dir=tmp_path / "checkpoints",
            resume=False,
        )
    assert model.forward_calls == 0
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("length", [39, 41, 63, 65])
def test_aura_git_override_rejects_non_object_id_lengths(monkeypatch, length):
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * length)
    with pytest.raises(RuntimeError, match="40- or 64-character"):
        aura._git_commit()
