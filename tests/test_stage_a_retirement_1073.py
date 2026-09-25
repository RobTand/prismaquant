"""Retiring a superseded Stage A run's checkpoints and pinned entries (PQ #1073).

Real fixture Stage A runs (``test_stage_a_chain_resume._run``) seal referenced
checkpoints, whose cotangent rows are the owner's own entries. These tests hold
the retirement to its proof and to what it removes; the PrismaBuild accounting
half is ``test_stage_a_retirement_pb_1073``:

* a superseded, completed run loses its checkpoints and exactly the entries
  they reference; its other entries stay, and a resume, a seed and a band of
  it refuse afterwards;
* a seed spec, a band or Stage B metadata under a declared binding root that
  names the run refuses the retirement, names the document, and removes
  nothing;
* a run that could still resume refuses: a successor with the run's own
  identity is a resume, and an owner not contained may still write;
* zero binding roots refuse; a retirement that stops partway finishes on a
  rerun from its sealed record.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from prismaquant import stage_a_chain_resume as resume_mod
from prismaquant import stage_a_retirement as retire
from prismaquant.joint_adjoint_band import BandRefused, build_band_receipt
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_space, checkpoint_directory, write_adjoint_receipt)
from prismaquant.stage_a_chain_resume import ChainResumeRefused, chain_state_path

from fleet_sdk import require_lease_helper
from test_stage_a_chain_resume import (  # noqa: F401  (autouse fixture)
    DIGESTS, TWO, _at, _interrupted, _offline_tier_policy, _resume, _run)


def completed(root, monkeypatch, **kw):
    write_adjoint_receipt(adjoint_space(root), _run(root, monkeypatch, **kw))
    return adjoint_space(root)


def successor(tmp_path, monkeypatch):
    """A fresh run under another implementation: another run identity."""
    root = tmp_path / "successor"
    _interrupted(root, monkeypatch, implementation=TWO, generation=1002,
                 interrupt=_at(3, 1, 2))
    path = chain_state_path(adjoint_space(root))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def plane_paths(space):
    session = json.loads(chain_state_path(space).read_text())["boundary_storage"]["session"]
    paths, digests = retire._checkpoint_planes(space, {
        "generation": str(session["generation"]),
        "run_identity_sha256": session["run_identity_sha256"]})
    return paths, digests


def argv(root, succ, roots):
    path, sha = succ
    out = ["--output-root", str(root), "--superseded-by", str(path),
           "--superseded-by-sha256", sha]
    for binding in roots:
        out += ["--binding-root", str(binding)]
    return out


def tree(root):
    return {str(p.relative_to(root)): p.stat().st_size
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def last_json(capsys):
    """The tool's JSON line: the last line of stdout (Stage A logs to it too)."""
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    return json.loads(lines[-1])


def retire_main(args):
    import tools.retire_stage_a_run as tool
    return tool.main(args)


def test_a_superseded_run_loses_its_checkpoints_and_the_entries_they_pin(
        tmp_path, monkeypatch, capsys):
    # The retirement runs through to PrismaBuild's lease release (PQ #886);
    # require_prismabuild_sdk only proves a distribution installed, which an
    # editable dev checkout with no fleet-injected helper root also passes
    # (PQ #1097) -- probe the helper the lane actually calls.
    require_lease_helper()
    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    pinned, digests = plane_paths(space)
    assert pinned and digests
    entries = space / "exact-boundaries"
    others = sorted(str(p) for p in entries.rglob("*.pt") if str(p) not in set(pinned))
    assert others, "the fixture run left no unpinned entry to keep"
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    (bindings / "unrelated.json").write_text(json.dumps({"path": "/elsewhere/x"}))

    assert retire_main(argv(root, succ, [bindings]) + ["--dry-run"]) == 0
    planned = last_json(capsys)
    assert planned["status"] == "planned" and planned["bindings_read"] == 1
    assert (space / "checkpoints").is_dir()
    assert not retire.retirement_record_path(space).exists()

    assert retire_main(argv(root, succ, [bindings])) == 0
    report = last_json(capsys)
    assert report["entries_removed"] == len(pinned)
    assert not (space / "checkpoints").exists()
    assert not any(os.path.exists(path) for path in pinned)
    assert all(os.path.exists(path) for path in others)
    record = json.loads(retire.retirement_record_path(space).read_text())
    assert record["entry_paths"] == pinned
    assert record["superseded_by"]["sha256"] == succ[1]

    # A retired run feeds nothing: no resume, no seed, no band.
    with pytest.raises(ChainResumeRefused, match="retires this Stage A run"):
        resume_mod.plan_chain_resume(space, json.loads(chain_state_path(space).read_text()),
                                     recomputed={}, running_implementation_sha256="1" * 64)
    with pytest.raises(BandRefused, match="retires this Stage A run"):
        build_band_receipt(output_root=root, boundary=2, stride_value=2,
                           bind_identity={}, **DIGESTS)

    # A rerun over a finished retirement removes nothing more and succeeds.
    assert retire_main(argv(root, succ, [bindings])) == 0
    assert last_json(capsys)["entries_removed"] == 0


@pytest.mark.parametrize("kind", ["seed-spec", "band-by-digest", "stage-b-slice-gz"])
def test_a_live_binding_refuses_and_names_the_document(tmp_path, monkeypatch, capsys, kind):
    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    _pinned, digests = plane_paths(space)
    boundary = max(digests)
    manifest = checkpoint_directory(space, boundary) / "checkpoint.json"
    bindings = tmp_path / "campaign" / "live"
    bindings.mkdir(parents=True)
    if kind == "seed-spec":
        name = "seed.json"
        (bindings / name).write_text(json.dumps({
            "schema": "prismaquant.stage_a.chain_seed.v1",
            "checkpoint": {"path": str(manifest), "sha256": digests[boundary]}}))
    elif kind == "band-by-digest":
        name = "band.json"
        (bindings / name).write_text(json.dumps({"checkpoints": [
            {"boundary": boundary, "sha256": digests[boundary]}]}))
    else:
        name = "layer-003.json.gz"
        (bindings / name).write_bytes(gzip.compress(json.dumps({"checkpoint": {
            "entries": [str(space / "exact-boundaries" / "x.pt")]}}).encode()))
    before = tree(tmp_path / "run")

    assert retire_main(argv(root, succ, [tmp_path / "campaign"])) == 3
    err = capsys.readouterr().err
    assert name in err and "a live binding holds the run" in err
    assert tree(tmp_path / "run") == before
    assert not retire.retirement_record_path(space).exists()


def test_a_run_that_could_still_resume_refuses(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    space = adjoint_space(root)
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    before = tree(root)

    # A "successor" carrying the run's own identity is the run resumed.
    same = tmp_path / "copy" / "chain-state.json"
    same.parent.mkdir()
    same.write_bytes(chain_state_path(space).read_bytes())
    same_sha = hashlib.sha256(same.read_bytes()).hexdigest()
    assert retire_main(argv(root, (same, same_sha), [bindings])) == 3
    assert "a resume of the run" in capsys.readouterr().err
    # The run's own chain state is inside the run.
    own = chain_state_path(space)
    assert retire_main(argv(root, (own, same_sha), [bindings])) == 3
    assert "inside the run" in capsys.readouterr().err
    assert tree(root) == before

    # An owner that is not contained may still write.
    producer = {"owner_action_key": "e" * 64, "owner_attempt": {"nonce": "n", "scope_id": "s"},
                "queue_root": str(tmp_path / "queue")}
    state = json.loads(own.read_text())
    state["producer"] = producer
    own.write_text(json.dumps(resume_mod._seal(
        {k: v for k, v in state.items() if k != "chain_state_sha256"})))

    def live(owner):
        raise ChainResumeRefused("the owner is still running")

    monkeypatch.setattr(resume_mod, "require_producer_contained", live)
    succ = successor(tmp_path, monkeypatch)
    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "the owner is still running" in capsys.readouterr().err
    assert (space / "checkpoints").is_dir()


def test_zero_binding_roots_prove_nothing(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    assert retire_main(argv(root, succ, [])) == 3
    assert "zero roots prove nothing" in capsys.readouterr().err
    assert (space / "checkpoints").is_dir()


def test_a_retirement_that_stops_partway_finishes_on_a_rerun(tmp_path, monkeypatch, capsys):
    # The retirement runs through to PrismaBuild's lease release (PQ #886);
    # require_prismabuild_sdk only proves a distribution installed, which an
    # editable dev checkout with no fleet-injected helper root also passes
    # (PQ #1097) -- probe the helper the lane actually calls.
    require_lease_helper()
    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    pinned, _digests = plane_paths(space)
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    real_unlink = os.unlink
    calls = []

    def dies(path, *args, **kwargs):
        if str(path) in pinned:
            calls.append(path)
            if len(calls) == 2:
                raise OSError("killed mid-retirement")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(retire.os, "unlink", dies)
        assert retire_main(argv(root, succ, [bindings])) == 3
    capsys.readouterr()
    assert retire.retirement_record_path(space).exists()
    assert not (space / "checkpoints").exists()
    assert sum(os.path.exists(path) for path in pinned) == len(pinned) - 1

    assert retire_main(argv(root, succ, [bindings])) == 0
    assert last_json(capsys)["entries_removed"] == len(pinned) - 1
    assert not any(os.path.exists(path) for path in pinned)


def test_a_seed_refuses_a_checkpoint_of_a_space_being_retired(tmp_path, monkeypatch):
    from prismaquant.stage_a_chain_seed import ChainSeedRefused, _sealed_checkpoint

    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    _pinned, digests = plane_paths(space)
    boundary = max(digests)
    binding = {"path": str(checkpoint_directory(space, boundary) / "checkpoint.json"),
               "sha256": digests[boundary]}
    record, _source = _sealed_checkpoint(binding, "checkpoint")
    assert record["boundary"] == boundary
    # The record is sealed before the first unlink, so a space mid-retirement,
    # its checkpoints still present, already refuses.
    retire.retirement_record_path(space).write_text("{}")
    with pytest.raises(ChainSeedRefused, match="retires this Stage A run"):
        _sealed_checkpoint(binding, "checkpoint")
