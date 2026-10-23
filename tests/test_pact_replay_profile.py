"""Bounded profiling retains one-item evidence across replay supervision."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"


@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_profile_entry_retains_first_item_metadata(tmp_path, supervised, failure):
    entry = tmp_path / "science.py"
    entry.write_text(
        "from bounded_profile import capture_item\n"
        "from replay_progress import ReplayProgress, supervise\n"
        "from pathlib import Path\n"
        "import sys\n"
        "assert 'torch' not in sys.modules\n"
        "def work(notify):\n"
        "    progress = ReplayProgress(Path(sys.argv[1]), 30, notify=notify)\n"
        "    for sample in (384, 385):\n"
        "        with capture_item():\n"
        "            import torch\n"
        "            with torch.profiler.record_function('science-item-' + str(sample)):\n"
        "                value = torch.arange(16) + sample\n"
        "        progress('layer-42-stream-null', 'complete band sequence ' + str(sample))\n"
        + ("    raise ValueError('science refusal')\n" if failure else "")
        + ("import multi_stream_replay as replay\n"
           "from types import SimpleNamespace\n"
           "from unittest.mock import patch\n"
           "args = SimpleNamespace(stall_seconds=30, dry_run_cpu=False, prepare_readset=False)\n"
           "with patch.object(replay, 'cli_parser') as parser, "
           "patch.object(replay, 'run_replay', lambda args, notify: work(notify)):\n"
           "    parser.return_value.parse_args.return_value = args\n"
           "    replay.main()\n"
           "assert 'torch' not in sys.modules\n"
           if supervised else "work(lambda record: None)\n")
    )
    output = tmp_path / "profile"
    history = tmp_path / "history.jsonl"
    result = subprocess.run(
        [sys.executable, str(REPLAY_ROOT / "profile_entry.py"), str(output), str(entry),
         json.dumps([str(history)]), "cpu"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            (str(REPLAY_ROOT.parents[1]), os.environ.get("PYTHONPATH", "")))},
    )
    if failure:
        assert result.returncode != 0
        assert "ValueError: science refusal" in result.stderr
        assert (output / "trace.json").is_file()
        assert not (output / "profile-summary.json").exists()
        assert "profiled_entry_completed" not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "profile-summary.json").read_text())
    assert summary["schema"] == "pact.bounded_rate_profile.v1"
    assert summary["capture_item_cap"] == summary["capture_items"] == 1
    assert (summary["started_unix"] <= summary["capture_started_unix"]
            <= summary["capture_ended_unix"] <= summary["ended_unix"])
    assert summary["entry_wall_seconds"] == summary["ended_unix"] - summary["started_unix"]
    assert summary["trace"] == str(output / "trace.json")
    events = json.loads(Path(summary["trace"]).read_text())["traceEvents"]
    names = {event.get("name") for event in events}
    assert "science-item-384" in names
    assert "science-item-385" not in names
    assert summary["operator_tables"] is summary["full_run_operator_totals"] is False
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["sequence"] for row in records[1:]] == [384, 385]
    assert json.loads(result.stdout.splitlines()[-1])["profiled_entry_completed"] is True
