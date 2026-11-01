"""Actual freeze CLI, real valid plans and a coordinated filesystem winner."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from experiments import tessera_quality_prefill as cli
from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.quality_prefill_contract import parse_manifest
from tests.test_quality_prefill_contract import resolved_manifest


def _freeze(tmp_path, name, manifest):
    draft = tmp_path / f"{name}-draft.json"
    output = tmp_path / f"{name}-frozen.json"
    draft.write_text(json.dumps(manifest))
    assert cli.main(["freeze", "--manifest", str(draft), "--out", str(output)]) == 0
    raw = output.read_bytes()
    assert parse_manifest(raw.decode(), mode="frozen")["identity_sha256"]
    return raw


@pytest.fixture
def plans(tmp_path):
    requested = resolved_manifest()
    other = copy.deepcopy(requested)
    other["experiment_id"] = "quality-prefill-concurrent-winner"
    expected = _freeze(tmp_path, "expected", requested)
    distinct = _freeze(tmp_path, "distinct", other)
    assert distinct != expected
    draft = tmp_path / "requested.json"
    draft.write_text(json.dumps(requested))
    output = tmp_path / "contended-frozen.json"
    argv = ["freeze", "--manifest", str(draft), "--out", str(output)]
    return argv, output, expected, distinct


def _publish_at_real_publication(monkeypatch, output, winner):
    replace, link = os.replace, os.link
    observed = []

    def rendezvous(real, source, target, *args, **kwargs):
        target_path = Path(target)
        if kwargs.get("dst_dir_fd") is not None:
            target_path = Path(os.readlink(f"/proc/self/fd/{kwargs['dst_dir_fd']}")) / target_path
        if not observed and target_path == output:
            observed.append(True)
            assert publish_new_bytes(output, winner, nofollow=True)
            assert output.read_bytes() == winner
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", lambda *a, **kw: rendezvous(replace, *a, **kw))
    monkeypatch.setattr(os, "link", lambda *a, **kw: rendezvous(link, *a, **kw))
    return observed


@pytest.mark.parametrize("matching", [False, True])
def test_concurrent_frozen_plan_is_preserved_and_loser_refuses(plans, monkeypatch, capsys, matching):
    argv, output, expected, distinct = plans
    winner = expected if matching else distinct
    observed = _publish_at_real_publication(monkeypatch, output, winner)
    capsys.readouterr()
    code = cli.main(argv)
    captured = capsys.readouterr()
    assert observed, "fixture did not reach actual filesystem publication"
    assert output.read_bytes() == winner, "an acknowledged valid frozen plan was replaced"
    assert code == 2, "published-once freeze accepted an already published winner"
    assert "published once" in captured.err


def test_ordinary_freeze_publishes_the_actual_valid_plan(plans, capsys):
    argv, output, expected, _distinct = plans
    capsys.readouterr()
    assert cli.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert output.read_bytes() == expected
    frozen = parse_manifest(expected.decode(), mode="frozen")
    assert summary["identity_sha256"] == frozen["identity_sha256"]


@pytest.mark.parametrize("matching", [False, True])
def test_existing_frozen_plan_is_refused_unchanged(plans, capsys, matching):
    argv, output, expected, distinct = plans
    winner = expected if matching else distinct
    assert publish_new_bytes(output, winner, nofollow=True)
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert "published once" in capsys.readouterr().err
    assert output.read_bytes() == winner


@pytest.mark.parametrize("parent", ["relative", "alias"])
def test_freeze_output_parent_preserves_its_valid_plan(plans, tmp_path, monkeypatch, parent):
    argv, _output, expected, _distinct = plans
    if parent == "relative":
        monkeypatch.chdir(tmp_path)
        requested = Path("new-parent/frozen.json")
        actual = tmp_path / requested
    else:
        actual_parent = tmp_path / "actual-parent"
        actual_parent.mkdir()
        alias = tmp_path / "alias-parent"
        alias.symlink_to(actual_parent, target_is_directory=True)
        requested = alias / "frozen.json"
        actual = actual_parent / "frozen.json"
    argv[argv.index("--out") + 1] = str(requested)
    assert cli.main(argv) == 0
    assert actual.read_bytes() == expected
    assert parse_manifest(expected.decode(), mode="frozen")["identity_sha256"]


def test_dangling_output_leaf_is_occupied_and_refused_unchanged(plans, tmp_path, capsys):
    argv, output, _expected, _distinct = plans
    target = tmp_path / "absent-owned-target.json"
    output.symlink_to(target)
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert "published once" in capsys.readouterr().err
    assert output.is_symlink()
    assert not target.exists()
