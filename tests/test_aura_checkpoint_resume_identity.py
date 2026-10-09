from __future__ import annotations

import json
import subprocess
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


@pytest.fixture
def producer_git(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    source = root / "prismaquant" / "aura_cost.py"
    source.parent.mkdir(parents=True)
    source.write_text("# recorded producer\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "add", "prismaquant/aura_cost.py"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=AURA test", "-c",
                    "user.email=aura@example.invalid", "commit", "--quiet",
                    "-m", "Record the producer"], cwd=root, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(aura, "__file__", str(source))
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", raising=False)
    return root, source, commit


def test_dirty_producer_dev_mode_publishes_and_reuses_checkpoint(
    tmp_path, monkeypatch, capsys, producer_git
):
    _root, source, commit = producer_git
    source.write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    assert aura._checkpoint_git_commit() == commit
    stamp = capsys.readouterr().out
    assert "[DEV-MODE]" in stamp
    assert commit in stamp
    assert "prismaquant/aura_cost.py" in stamp
    checkpoint = tmp_path / "checkpoint"
    identity = {"git_commit": commit, "label": "dirty producer"}
    root, digest, completed = aura._prepare_aura_checkpoints(
        checkpoint, resume=False, identity=identity, names=["unit-雪"])
    assert completed == {}
    aura._write_aura_unit_checkpoint(root, qname="unit-雪", identity_sha256=digest,
                                    state={"value": [1.25]})
    retained = {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}
    assert json.loads((root / "manifest.json").read_bytes())["identity"] == identity
    assert aura._prepare_aura_checkpoints(
        root, resume=True, identity=identity, names=["unit-雪"])[1:] == (
            digest, {"unit-雪": {"value": [1.25]}})
    assert {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()} == retained


def test_dirty_producer_certified_mode_keeps_refusal(monkeypatch, capsys, producer_git):
    _root, source, commit = producer_git
    source.write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match="aura_cost.py differs"):
        aura._checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_git_diff_execution_error_refuses_in_both_modes(
    monkeypatch, capsys, producer_git, mode
):
    root, _source, commit = producer_git
    # HEAD still resolves, but Git cannot read its commit object.
    (root / ".git" / "objects" / commit[:2] / commit[2:]).unlink()
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    assert aura._git_commit() == commit
    with pytest.raises(RuntimeError, match="AURA checkpoint git identity is not exact"):
        aura._checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_git_diff_timeout_refuses_in_both_modes(monkeypatch, capsys, producer_git, mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    real_run = aura.subprocess.run

    def run(command, **kwargs):
        if command[:2] == ["git", "diff"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(aura.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        aura._checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_missing_git_refuses_in_both_modes(monkeypatch, capsys, producer_git, mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    monkeypatch.setenv("PATH", "/no-aura-git-executable")
    with pytest.raises(RuntimeError, match="cannot resolve git commit"):
        aura._checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_dirty_producer_keeps_explicit_git_override(
    monkeypatch, capsys, producer_git, mode
):
    _root, source, _commit = producer_git
    source.write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * 40)
    assert aura._checkpoint_git_commit() == "a" * 40
    assert "[DEV-MODE]" not in capsys.readouterr().out
