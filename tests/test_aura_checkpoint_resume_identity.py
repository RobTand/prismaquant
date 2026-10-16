from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import aura_checkpoint_journal, aura_provenance, joint_price_step
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
        aura_provenance.producer_git_commit()


#: Every module whose code builds, encodes, restores or stamps an AURA
#: checkpoint: the cost driver, the journal, the price step and the provenance
#: recipe itself.
CHECKPOINT_OWNERS = ("aura_cost.py", "aura_checkpoint_journal.py",
                     "joint_price_step.py", "aura_provenance.py")


@pytest.fixture
def producer_git(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    package = root / "prismaquant"
    package.mkdir(parents=True)
    for name in CHECKPOINT_OWNERS + ("unrelated.py",):
        (package / name).write_text(f"# recorded {name}\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "add", "prismaquant"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=AURA test", "-c",
                    "user.email=aura@example.invalid", "commit", "--quiet",
                    "-m", "Record the producer"], cwd=root, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(aura_provenance, "__file__", str(package / "aura_provenance.py"))
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", raising=False)
    return root, package / "aura_cost.py", commit


def test_dirty_producer_dev_mode_publishes_and_reuses_checkpoint(
    tmp_path, monkeypatch, capsys, producer_git
):
    _root, source, commit = producer_git
    source.write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    assert aura_provenance.checkpoint_git_commit() == commit
    stamp = capsys.readouterr().out
    assert "[DEV-MODE]" in stamp
    assert commit in stamp
    assert "prismaquant/aura_cost.py" in stamp
    checkpoint = tmp_path / "checkpoint"
    identity = {"git_commit": commit, "label": "dirty producer"}
    journal = aura_checkpoint_journal.AuraCheckpointJournal.open(
        checkpoint, resume=False, identity=identity, names=["unit-雪"])
    root, digest = journal.root, journal.identity_sha256
    assert journal.completed == set()
    journal.writer.publish(qname="unit-雪", state={"value": [1.25]})
    retained = {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}
    assert json.loads((root / "manifest.json").read_bytes())["identity"] == identity
    resumed = aura_checkpoint_journal.AuraCheckpointJournal.open(
        root, resume=True, identity=identity, names=["unit-雪"])
    restored = {}
    class Prices:
        def restore_unit(self, name, state):
            restored[name] = state
    assert resumed.identity_sha256 == digest
    assert resumed.resume_units(Prices()) == {"unit-雪"}
    assert restored == {"unit-雪": {"value": [1.25]}}
    assert {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()} == retained


def test_dirty_producer_certified_mode_keeps_refusal(monkeypatch, capsys, producer_git):
    _root, source, commit = producer_git
    source.write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match="aura_cost.py differs"):
        aura_provenance.checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_git_diff_execution_error_refuses_in_both_modes(
    monkeypatch, capsys, producer_git, mode
):
    root, _source, commit = producer_git
    # HEAD still resolves, but Git cannot read its commit object.
    (root / ".git" / "objects" / commit[:2] / commit[2:]).unlink()
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    assert aura_provenance.producer_git_commit() == commit
    with pytest.raises(RuntimeError, match="AURA checkpoint git identity is not exact"):
        aura_provenance.checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_git_diff_timeout_refuses_in_both_modes(monkeypatch, capsys, producer_git, mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    real_run = aura_provenance.subprocess.run

    def run(command, **kwargs):
        if command[:2] == ["git", "diff"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real_run(command, **kwargs)

    monkeypatch.setattr(aura_provenance.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        aura_provenance.checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["1", "0"])
def test_missing_git_refuses_in_both_modes(monkeypatch, capsys, producer_git, mode):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    monkeypatch.setenv("PATH", "/no-aura-git-executable")
    with pytest.raises(RuntimeError, match="cannot resolve git commit"):
        aura_provenance.checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("owner", CHECKPOINT_OWNERS)
@pytest.mark.parametrize("mode", ["1", "0"])
def test_dirty_producer_keeps_explicit_git_override(
    monkeypatch, capsys, producer_git, mode, owner
):
    root, _source, _commit = producer_git
    (root / "prismaquant" / owner).write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * 40)
    assert aura_provenance.checkpoint_git_commit() == "a" * 40
    assert "[DEV-MODE]" not in capsys.readouterr().out


def test_checkpoint_owner_set_is_the_live_producer_modules():
    # A pathspec naming a file that no longer exists matches nothing, so a
    # renamed owner would silently drop out of the exactness check.
    root = Path(aura_provenance.__file__).resolve().parents[1]
    live = {Path(module.__file__).resolve().relative_to(root).as_posix()
            for module in (aura, aura_checkpoint_journal, joint_price_step,
                           aura_provenance)}
    assert set(aura_provenance.CHECKPOINT_PRODUCER_SOURCES) == live
    assert live == {f"prismaquant/{name}" for name in CHECKPOINT_OWNERS}


@pytest.mark.parametrize("owner", CHECKPOINT_OWNERS)
def test_certified_mode_refuses_drift_in_every_checkpoint_owner(
    monkeypatch, capsys, producer_git, owner
):
    root, _source, commit = producer_git
    (root / "prismaquant" / owner).write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match=(
            rf"AURA checkpoint git identity is not exact: {re.escape(owner)} "
            rf"differs from commit {commit}; commit it before checkpoint/resume")):
        aura_provenance.checkpoint_git_commit()
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("owner", CHECKPOINT_OWNERS)
def test_dev_mode_stamps_drift_in_every_checkpoint_owner(
    monkeypatch, capsys, producer_git, owner
):
    root, _source, commit = producer_git
    (root / "prismaquant" / owner).write_text("# changed producer\n", encoding="utf-8")
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    assert aura_provenance.checkpoint_git_commit() == commit
    stamp = capsys.readouterr().out
    assert "[DEV-MODE]" in stamp
    assert commit in stamp
    assert f"(prismaquant/{owner})" in stamp


def test_owner_refusal_names_each_drifted_owner_and_staged_edits(
    monkeypatch, producer_git
):
    root, _source, commit = producer_git
    package = root / "prismaquant"
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    # The whole-package digest binds every other module; this commit label
    # certifies the checkpoint owners.
    (package / "unrelated.py").write_text("# changed\n", encoding="utf-8")
    assert aura_provenance.checkpoint_git_commit() == commit
    for name in ("joint_price_step.py", "aura_checkpoint_journal.py"):
        (package / name).write_text("# changed producer\n", encoding="utf-8")
    # A staged edit is still not the recorded commit.
    subprocess.run(["git", "add", "prismaquant/joint_price_step.py"], cwd=root, check=True)
    with pytest.raises(RuntimeError, match=(
            "not exact: aura_checkpoint_journal.py, joint_price_step.py differ "
            f"from commit {commit};")):
        aura_provenance.checkpoint_git_commit()
