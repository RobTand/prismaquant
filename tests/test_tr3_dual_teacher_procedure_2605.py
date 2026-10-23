"""Doc lint for the dual-teacher masked-tile TR3 procedure (PQ #2605).

The procedure doc carries PB-submittable serve and score commands for the
masked-tile candidate. This lint parses every scorer command in the doc
against the live scorer argparse, so a renamed flag or path breaks here
on CPU instead of on a GPU reservation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/measurements/tr3-dual-teacher-masked-tile-20261010.md"

TEACHER1_SHA = "1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae"
TEACHER2_SHA = "da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b"
PANEL_SHA = "35f0c5c973be614f29db757e9bd4bce407ea218b974a8407ec7e64c571aad72b"
SERVE_IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
               "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")


def _read_doc():
    assert DOC.is_file(), f"procedure doc is missing: {DOC}"
    return DOC.read_text(encoding="utf-8")


def _scorer_commands(text):
    """(raw line, argv) for each scorer invocation in fenced bash blocks."""
    found = []
    for block in re.findall(r"```bash(.*?)```", text, re.S):
        logical = re.sub(r"\\\n", " ", block)
        for line in logical.splitlines():
            if "measure_glm_tr3_vllm.py" not in line:
                continue
            tokens = shlex.split(line, posix=True)
            cut = next(i for i, token in enumerate(tokens)
                       if token.endswith("measure_glm_tr3_vllm.py"))
            raw = tokens[cut + 1:]
            if ">" in raw:
                raw = raw[:raw.index(">")]
            argv = ["measure_glm_tr3_vllm.py"]
            substitutions = {"$MASTER_PORT": "29500"}
            argv += [substitutions.get(token, "DUMMY") if token.startswith("$")
                     else token for token in raw]
            found.append((line.strip(), raw, argv))
    return found


def _parse(argv):
    from experiments import measure_glm_tr3_vllm as served
    captured = {}
    with mock.patch.object(served, "measure", lambda args: captured.update(vars(args))), \
        mock.patch.object(sys, "argv", argv):
        served.main()
    return captured


def test_doc_names_both_teachers_and_the_matched_input_set():
    text = _read_doc()
    assert TEACHER1_SHA in text
    assert TEACHER2_SHA in text
    assert PANEL_SHA in text
    assert "tr3-teacher-04" in text
    assert "tr3-teacher-exl3ref-01" in text
    assert "25 windows" in text
    assert "2048" in text


def test_doc_gives_parseable_dual_teacher_serve_and_score_commands():
    text = _read_doc()
    commands = _scorer_commands(text)
    assert len(commands) >= 2, "doc must carry serve (qualify) and score commands"
    flags = {raw[i] for _, raw, _ in commands for i in range(len(raw))}
    assert "--qualify-hook" in flags, "one command must serve the one-window qualification"
    assert "--qualification" in flags, "the score command must replay the qualification"
    for line, raw, argv in commands:
        parsed = _parse(argv)
        assert parsed["teacher_sha256"] == TEACHER1_SHA, line
        assert parsed["teacher2_sha256"] == TEACHER2_SHA, line
        assert parsed["teacher"].endswith("tr3-teacher-04/artifact/teacher.json"), line
        assert parsed["teacher2"].endswith("tr3-teacher-exl3ref-01/artifact/teacher.json"), line
        assert parsed["serve_image"] == SERVE_IMAGE, line
        assert parsed["tensor_parallel_size"] == 2, line
        assert parsed["panel"].endswith("final_panel_handoff.json"), line
    model_values = [raw[raw.index("--model") + 1] for _, raw, _ in commands]
    assert all(value.startswith("$CANDIDATE") for value in model_values), \
        "candidate path is a named parameter, not an invented path"


def test_doc_defines_the_receipt_path_and_format():
    text = _read_doc()
    assert "docs/measurements" in text
    assert "docs/measurements/tr3-dual-teacher-masked-tile-20261010/" in text
    assert "hook-qualification.json" in text
    assert "full-vocabulary-kl.json" in text
    assert "second_teacher_full_vocabulary_kl" in text
    assert "teacher2_sha256" in text
    assert "runtime_binding" in text
    assert "candidate_identity" in text
    assert "prismaquant.glm_tr3_full_vocabulary_kl/1" in text


def test_doc_exports_tr3_arm_and_runs_from_tessera_worktree():
    text = _read_doc()
    assert re.search(r"^export TR3_ARM=\S+", text, re.M), \
        "TR3_ARM must be exported before RECEIPT_ROOT uses it"
    assert "TESSERA_CHECKOUT" in text
    assert 'cd "$TESSERA_CHECKOUT"' in text
    assert '--cwd "$PRISMAQUANT_CHECKOUT"' in text


# Shell-boundary test doubles record the submission and worker/container argv
# without invoking PB, Docker, an image, or measure(). This covers the prefix
# the argparse lint omits. No submission entry point is executed, even a fake
# script named pbrun.py (the admitted test environment rightly prohibits that).
_RECORD_LAUNCH = r'''
import hashlib, json, os, pathlib, subprocess, sys
tool, *args = sys.argv[1:]

def record(kind, **fields):
    with open(os.environ["TR3_TEST_LOG"], "a") as stream:
        stream.write(json.dumps(dict(kind=kind, argv=args, cwd=os.getcwd(), **fields)) + "\n")

if tool == "pbrun.py":
    if "--withdraw" in args:
        record("withdraw")
    else:
        boundary = args.index("--")
        env = os.environ.copy()
        for i, flag in enumerate(args[:boundary]):
            if flag == "--env":
                key, value = args[i + 1].split("=", 1)
                env[key] = value
        record("pb")
        worker = args[boundary + 1:]
        assert worker[:2] == ["bash", "-c"]
        assert worker[2].lstrip().startswith("exec docker run ")
        # Mock the exec boundary: never invoke a real Docker entry point.
        worker[2] = worker[2].replace("exec docker ",
            'exec python3 "$PB_FLEET_TOOLS/record_launch.py" docker ', 1)
        key = hashlib.sha256(json.dumps(args).encode()).hexdigest()
        # Model PB's different closure-stamped source identity for every action.
        snapshot = pathlib.Path(env["TR3_TEST_SNAPSHOT"]) / key
        snapshot.mkdir()
        subprocess.run(worker, cwd=snapshot, env=env, check=True)
        print(json.dumps({"action_key": key}))
elif tool == "pbwait.py":
    record("wait")
    records = [json.loads(line) for line in pathlib.Path(os.environ["TR3_TEST_LOG"]).read_text().splitlines()]
    withdrawn = any(r["kind"] == "withdraw" and r["argv"][1] == args[-1] for r in records)
    complete = not (withdrawn and os.environ.get("TR3_TEST_CLEANUP_INCOMPLETE"))
    print(json.dumps({"schema": "prismabuild.endings.v1", "complete": complete, "endings": [
        {"status": "withdrawn" if withdrawn else "executed", "succeeded": not withdrawn}]}))
    sys.exit(1 if withdrawn else 0)
elif tool == "docker":
    env = {}
    for i, flag in enumerate(args):
        if flag == "--env":
            key, sep, value = args[i + 1].partition("=")
            env[key] = value if sep else os.environ[key]
    image = args.index(os.environ["SERVE_IMAGE"])
    # bash -c SCRIPT tr3-container COMMAND...
    command = args[image + 4:]
    record("container", environment=env, command=command)
    # A placeholder receipt only lets the documented hash/wait sequence run.
    # No receipt content here is measurement evidence.
    if "--output" in command:
        pathlib.Path(command[command.index("--output") + 1]).write_text("{}")
else:
    raise AssertionError(tool)
'''


def _run_documented_shell(tmp_path, **overrides):
    tools = tmp_path / "client tools"
    tools.mkdir()
    (tools / "record_launch.py").write_text(_RECORD_LAUNCH)
    env = dict(os.environ)
    env.update({
        "PATH": str(tools) + os.pathsep + env["PATH"],
        "PB_FLEET_TOOLS": str(tools),
        "COORD_HOST_TAG": "test-coordinator", "PEER_HOST_TAG": "test-peer",
        "COORD_IP": "192.0.2.1", "PEER_IP": "192.0.2.2",
        "COORD_IFACE": "test-net-0", "PEER_IFACE": "test-net-1",
        "MASTER_PORT": "29500", "TR3_RUN_ID": "test-run", "TR3_ARM": "test-arm",
        "TESSERA_WHEEL_SHA256": "a" * 64,
        "PRISMAQUANT_SERVE_COMMIT": "b" * 40,
        "TR3_TEST_LOG": str(tmp_path / "launches.jsonl"),
    })
    for key in ("TESSERA_CHECKOUT", "PRISMAQUANT_CHECKOUT", "TR3_TEST_SNAPSHOT",
                "PRISMAQUANT_SERVE_SOURCE", "CANDIDATE_DIR", "RECEIPT_ROOT"):
        path = tmp_path / (key.lower() + " with spaces")
        path.mkdir()
        env[key] = str(path)
    for key in ("CANDIDATE_DIGEST_CACHE", "TESSERA_SERVE_WHEEL",
                "COORD_NAMESPACE", "PEER_NAMESPACE"):
        path = tmp_path / (key.lower() + " with spaces")
        path.touch()
        env[key] = str(path)
    for i, key in enumerate(("QUAL_COORD_D38", "QUAL_PEER_D38", "SCORE_COORD_D38", "SCORE_PEER_D38")):
        env[key] = str(i + 1) * 64
    env.update(overrides)
    mock_shell = r'''
python3() {
  case "$1" in
    "$PB_FLEET_TOOLS/pbrun.py"|"$PB_FLEET_TOOLS/pbwait.py")
      local tool=${1##*/}
      shift
      command python3 "$PB_FLEET_TOOLS/record_launch.py" "$tool" "$@" ;;
    *) command python3 "$@" ;;
  esac
}
'''
    shell = mock_shell + "\n".join(re.findall(r"```bash(.*?)```", _read_doc(), re.S))
    result = subprocess.run(["bash", "-c", shell], env=env, text=True, capture_output=True)
    log = Path(env["TR3_TEST_LOG"])
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, env, records


def _options(argv):
    """Compare stock peer flags irrespective of order, retaining bool flags."""
    result = {}
    while argv:
        flag, *argv = argv
        assert flag.startswith("--"), flag
        value = True
        if argv and not argv[0].startswith("--"):
            value, *argv = argv
        assert flag not in result, flag
        result[flag] = value
    return result


def test_documented_jobs_enter_image_and_match_two_host_engine_profile(tmp_path, monkeypatch):
    from experiments import measure_glm_tr3_vllm as served
    from tools.gold_engine_options import gold_engine_kwargs, headless_peer_argv

    result, env, records = _run_documented_shell(tmp_path)
    assert result.returncode == 0, result.stderr
    submissions = [r for r in records if r["kind"] == "pb"]
    containers = [r for r in records if r["kind"] == "container"]
    assert len(submissions) == len(containers) == 4
    assert len({r["cwd"] for r in containers}) == 4, "distinct PB snapshot identities"
    for record in records:
        if record["kind"] == "wait":
            assert record["argv"][:3] == ["--endings", "--until", "cleaned"]
    assert [r["kind"] for r in records] == [
        "pb", "container", "pb", "container", "wait", "withdraw", "wait",
        "pb", "container", "pb", "container", "wait", "withdraw", "wait",
    ]
    assert len({tuple(r["argv"]) for r in submissions}) == 4, "never reuse a cached peer"
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    for i, (submission, container) in enumerate(zip(submissions, containers)):
        peer = i % 2 == 0
        host = "PEER" if peer else "COORD"
        assert submission["cwd"] == env["TESSERA_CHECKOUT"]
        argv = submission["argv"]
        assert argv[argv.index("--cwd") + 1] == env["PRISMAQUANT_CHECKOUT"]
        assert argv[argv.index("--tag") + 1] == env[host + "_HOST_TAG"]
        assert argv[argv.index("--host-class") + 1] == "gb10"
        assert "--gpu" in argv and "--exclusive" in argv
        assert argv[argv.index("--container-image") + 1] == SERVE_IMAGE
        stage = "QUAL" if i < 2 else "SCORE"
        assert argv[argv.index("--d38-receipt") + 1] == env[stage + "_" + host + "_D38"]
        assert argv[argv.index("--d38-namespace") + 1] == env[host + "_NAMESPACE"]
        dargv = container["argv"]
        assert Path(container["cwd"]).parent == Path(env["TR3_TEST_SNAPSHOT"])
        assert container["cwd"] != submission["cwd"], "mount worker snapshot, not client cwd"
        assert dargv[:2] == ["run", "--rm"]
        assert dargv[dargv.index("--entrypoint") + 1] == "/bin/bash"
        assert dargv[dargv.index("--network") + 1] == "host"
        assert dargv[dargv.index("--ipc") + 1] == "host"
        assert dargv[dargv.index("--gpus") + 1] == "all"
        assert dargv[dargv.index("--workdir") + 1] == "/workspace/prismaquant"
        mounts = [dargv[j + 1] for j, flag in enumerate(dargv) if flag == "--mount"]
        assert mounts == [
            f'type=bind,src={container["cwd"]},dst=/workspace/submission,readonly',
            f'type=bind,src={env["PRISMAQUANT_SERVE_SOURCE"]},dst=/workspace/prismaquant,readonly',
            *[f'type=bind,src={env[key]},dst={env[key]},readonly' for key in
              ("CANDIDATE_DIR", "CANDIDATE_DIGEST_CACHE")],
            "type=bind,src=/mnt/shared/tessera-measurements/glm-canonical-census-20260908,"
            "dst=/mnt/shared/tessera-measurements/glm-canonical-census-20260908,readonly",
            f'type=bind,src={env["TESSERA_SERVE_WHEEL"]},dst={env["TESSERA_SERVE_WHEEL"]},readonly',
            f'type=bind,src={env["RECEIPT_ROOT"]},dst={env["RECEIPT_ROOT"]}',
        ]
        inside = container["environment"]
        assert inside["PYTHONPATH"] == "/workspace/prismaquant"
        assert inside["PQ_GOLD_PEER_DRIVER"] == "experiments.measure_glm_tr3_vllm"
        assert inside["TESSERA_SERVE_MODE"] == "resident"
        assert inside["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
        assert inside["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert inside["NCCL_IB_DISABLE"] == "1"
        assert inside["VLLM_HOST_IP"] == env[host + "_IP"]
        assert inside["NCCL_SOCKET_IFNAME"] == inside["GLOO_SOCKET_IFNAME"] == env[host + "_IFACE"]
        assert inside["TR3_RUN_ID"] == env["TR3_RUN_ID"]
        assert inside["TR3_STAGE"] == ("qualify" if i < 2 else "score")
        script = dargv[dargv.index(SERVE_IMAGE) + 2]
        assert 'sha256sum -c -' in script and '--no-index --no-deps --force-reinstall' in script
        assert 'exec "$@"' in script
        assert inside["PRISMAQUANT_SERVE_COMMIT"] == env["PRISMAQUANT_SERVE_COMMIT"]
        assert 'test "$(git rev-parse HEAD)" = "$PRISMAQUANT_SERVE_COMMIT"' in script
        assert 'test -z "$(git status --porcelain)"' in script
        assert container["command"][0] == "python3", "use image interpreter"

    for peer, coordinator in ((containers[0], containers[1]), (containers[2], containers[3])):
        parsed = _parse(coordinator["command"][1:])
        assert parsed["nnodes"] == 2 and parsed["node_rank"] == 0
        assert parsed["master_addr"] == env["COORD_IP"]
        assert parsed["master_port"] == int(env["MASTER_PORT"])
        args = argparse.Namespace(**parsed)
        kwargs = served.scorer_engine_kwargs(
            args, model=args.model, topology=gold_engine_kwargs(args))
        expected = headless_peer_argv(kwargs, node_rank=1)
        actual = peer["command"][2:]
        assert actual[:2] == expected[:2] == ["serve", env["CANDIDATE_DIR"]]
        assert _options(actual[2:]) == _options(expected[2:])
        assert (ROOT / peer["command"][1]).is_file()
        assert (ROOT / coordinator["command"][1]).is_file()


def test_documented_reservation_rejects_same_host(tmp_path):
    result, _, records = _run_documented_shell(tmp_path, PEER_HOST_TAG="test-coordinator")
    assert result.returncode != 0
    assert not records, "refuse same-host TP2 before submission"


def test_documented_jobs_stop_before_score_if_peer_cleanup_is_incomplete(tmp_path):
    result, _, records = _run_documented_shell(tmp_path, TR3_TEST_CLEANUP_INCOMPLETE="1")
    assert result.returncode != 0
    assert "cleanup incomplete" in result.stderr
    assert len([r for r in records if r["kind"] == "pb"]) == 2
    assert len([r for r in records if r["kind"] == "withdraw"]) == 1
