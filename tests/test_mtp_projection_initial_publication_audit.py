"""Exercise the real tiny-checkpoint projection and its final publication race."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_glm_mtp_capture import mtp_source  # noqa: F401


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projection_args(env, name, *, q256=256):
    census = copy.deepcopy(env.census)
    for request in census["expert_projection"]["request"].values():
        request["q256"] = q256
    census_path = env.root / f"{name}-census.json"
    census_path.write_text(json.dumps(census))
    plan_path = env.root / f"{name}-plan.json"
    plan_path.write_text(json.dumps({
        "model": str(env.source),
        "inputs": {"census": {"path": str(census_path), "sha256": _sha(census_path)}},
    }))
    out = env.root / name
    return ["--phase", "projection", "--plan", str(plan_path),
            "--plan-sha256", _sha(plan_path), "--out", str(out)], out


def _produce(env, name, *, q256=256):
    from tools import glm_mtp_capture as cli

    argv, out = _projection_args(env, name, q256=q256)
    assert cli.main(argv) == 0
    path = out / "mtp-projection.json"
    report = json.loads((out / "projection-run.json").read_bytes())
    assert report["projection"]["sha256"] == _sha(path)
    return path.read_bytes()


def _interleave_winner(monkeypatch, target, winner):
    from prismaquant.cost_stage_checkpoint import publish_new_bytes

    real_exists = Path.exists
    observed = {"absent_checks": 0, "published": False}

    def exists(path, *args, **kwargs):
        absent = not real_exists(path, *args, **kwargs)
        if path == target and absent and not observed["published"]:
            observed["absent_checks"] += 1
            if observed["absent_checks"] == 2:
                observed["published"] = True
                assert publish_new_bytes(target, winner)
        # This is the result of the real check before the coordinated writer.
        return not absent

    monkeypatch.setattr(Path, "exists", exists)
    return observed


def test_distinct_projection_winner_is_preserved_and_loser_refuses(mtp_source, monkeypatch):
    from tools import glm_mtp_capture as cli

    env = mtp_source
    winner = _produce(env, "winner", q256=512)
    expected_current = _produce(env, "current-control", q256=256)
    assert json.loads(winner)["request"] != json.loads(expected_current)["request"]
    argv, out = _projection_args(env, "contended", q256=256)
    target = out / "mtp-projection.json"
    observed = _interleave_winner(monkeypatch, target, winner)
    refusal = None
    try:
        cli.main(argv)
    except RuntimeError as exc:
        refusal = str(exc)
    assert observed["published"], "the real final absence check was not reached"
    assert target.read_bytes() == winner, "an acknowledged distinct projection was replaced"
    assert refusal is not None and ("exist" in refusal or "publish" in refusal)


def test_unchanged_projection_publishes_its_actual_digest(mtp_source):
    raw = _produce(mtp_source, "unchanged")
    projection = json.loads(raw)
    assert projection["stacks"]
    assert projection["producer"]["source"] == mtp_source.census["expert_projection"]["producer"]["source"]


def test_same_byte_existing_projection_refuses_unchanged(mtp_source):
    from tools import glm_mtp_capture as cli

    env = mtp_source
    winner = _produce(env, "same-byte")
    argv, out = _projection_args(env, "same-byte")
    target = out / "mtp-projection.json"
    with pytest.raises(RuntimeError, match="publishes once"):
        cli.main(argv)
    assert target.read_bytes() == winner


@pytest.mark.parametrize("parent", ["relative", "alias"])
def test_projection_output_parent_preserves_its_real_receipt(mtp_source, monkeypatch, parent):
    from tools import glm_mtp_capture as cli

    env = mtp_source
    argv, _out = _projection_args(env, "path-control")
    if parent == "relative":
        monkeypatch.chdir(env.root)
        requested = Path("relative-projection")
        actual = env.root / requested
    else:
        actual_parent = env.root / "actual-parent"
        actual_parent.mkdir()
        alias = env.root / "alias-parent"
        alias.symlink_to(actual_parent, target_is_directory=True)
        requested = alias / "projection"
        actual = actual_parent / "projection"
    argv[argv.index("--out") + 1] = str(requested)
    assert cli.main(argv) == 0
    projection = actual / "mtp-projection.json"
    report = json.loads((actual / "projection-run.json").read_bytes())
    assert report["projection"]["sha256"] == _sha(projection)
    assert report["projection"]["path"] == str(projection.resolve())
