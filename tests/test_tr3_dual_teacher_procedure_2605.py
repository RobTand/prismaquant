"""Doc lint for the dual-teacher masked-tile TR3 procedure (PQ #2605).

The procedure doc carries PB-submittable serve and score commands for the
masked-tile candidate. This lint parses every scorer command in the doc
against the live scorer argparse, so a renamed flag or path breaks here
on CPU instead of on a GPU reservation.
"""
from __future__ import annotations

import re
import shlex
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
            argv = ["measure_glm_tr3_vllm.py"]
            argv += ["DUMMY" if token.startswith("$") else token for token in raw]
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
    assert "second_teacher_full_vocabulary_kl" in text
    assert "teacher2_sha256" in text
    assert "runtime_binding" in text
    assert "candidate_identity" in text
    assert "prismaquant.glm_tr3_full_vocabulary_kl/1" in text
