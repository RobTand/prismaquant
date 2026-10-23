"""Actual research replay CLI publication with a real create-only winner."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.tessera_anchored_surface import replay_campaign
from tests.test_tessera_anchored_surface import make_campaign
from tools import tessera_surface_replay as cli


def _encoded(report):
    return (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


@pytest.fixture
def replay(tmp_path):
    costs, checkpoint, plan, _values = make_campaign(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "report.json"
    expected = _encoded(replay_campaign(costs, checkpoint, plan))
    other = copy.deepcopy(plan)
    other["segments"][0]["max_absolute_log10_error"] = 0.02
    distinct = _encoded(replay_campaign(costs, checkpoint, other))
    assert distinct != expected
    argv = ["--costs", str(costs), "--checkpoint", str(checkpoint),
            "--plan", str(plan_path), "--out", str(output)]
    return argv, output, expected, distinct


def _publish_at_real_publication(monkeypatch, output, winner):
    """Pause before the real replace/link; publish a valid winner, then resume."""
    replace, link = os.replace, os.link
    observed = []

    def rendezvous(real, source, target, *args, **kwargs):
        target_path = Path(target)
        if kwargs.get("dst_dir_fd") is not None:
            target_path = Path(os.readlink(f"/proc/self/fd/{kwargs['dst_dir_fd']}")) / target_path
        if not observed and target_path == output:
            observed.append(True)
            # The existing no-follow owner stages a distinct UUID inode even
            # in this PID, so it cannot collide with the paused writer's stage.
            assert publish_new_bytes(output, winner, nofollow=True)
            assert output.read_bytes() == winner
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", lambda *a, **kw: rendezvous(replace, *a, **kw))
    monkeypatch.setattr(os, "link", lambda *a, **kw: rendezvous(link, *a, **kw))
    return observed


def test_distinct_concurrent_report_winner_is_refused_and_preserved(replay, monkeypatch):
    argv, output, _expected, distinct = replay
    observed = _publish_at_real_publication(monkeypatch, output, distinct)
    refusal = None
    try:
        cli.main(argv)
    except SystemExit as exc:
        refusal = exc
    assert observed, "fixture did not reach a real publication primitive"
    assert output.read_bytes() == distinct, "acknowledged distinct research report was replaced"
    assert refusal is not None and refusal.code == 2


def test_matching_concurrent_report_winner_is_idempotent(replay, monkeypatch):
    argv, output, expected, _distinct = replay
    observed = _publish_at_real_publication(monkeypatch, output, expected)
    assert cli.main(argv) == 0
    assert observed
    assert output.read_bytes() == expected


def test_ordinary_replay_and_identical_reuse_keep_actual_report_bytes(replay):
    argv, output, expected, _distinct = replay
    assert cli.main(argv) == cli.main(argv) == 0
    assert output.read_bytes() == expected


def test_existing_distinct_report_is_refused_without_replacement(replay):
    argv, output, _expected, distinct = replay
    assert publish_new_bytes(output, distinct, nofollow=True)
    with pytest.raises(SystemExit) as refusal:
        cli.main(argv)
    assert refusal.value.code == 2
    assert output.read_bytes() == distinct


@pytest.mark.parametrize("parent", ["relative", "alias"])
def test_output_parent_spellings_preserve_the_actual_report(replay, tmp_path, monkeypatch, parent):
    argv, _output, expected, _distinct = replay
    if parent == "relative":
        monkeypatch.chdir(tmp_path)
        requested = Path("new-parent/report.json")
        actual = tmp_path / requested
    else:
        actual_parent = tmp_path / "actual-parent"
        actual_parent.mkdir()
        alias = tmp_path / "alias-parent"
        alias.symlink_to(actual_parent, target_is_directory=True)
        requested = alias / "report.json"
        actual = actual_parent / "report.json"
    argv[argv.index("--out") + 1] = str(requested)
    assert cli.main(argv) == cli.main(argv) == 0
    assert actual.read_bytes() == expected


@pytest.mark.parametrize("matching", [False, True])
def test_existing_leaf_alias_keeps_its_report_and_reuse_policy(replay, tmp_path, matching):
    argv, output, expected, distinct = replay
    winner = expected if matching else distinct
    backing = tmp_path / "owned-report.json"
    assert publish_new_bytes(backing, winner, nofollow=True)
    output.symlink_to(backing)
    if matching:
        assert cli.main(argv) == 0
    else:
        with pytest.raises(SystemExit) as refusal:
            cli.main(argv)
        assert refusal.value.code == 2
    assert output.is_symlink()
    assert backing.read_bytes() == winner
