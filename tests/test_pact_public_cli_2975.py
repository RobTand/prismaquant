"""Focused W3 public-CLI admission gates, not full D38 equivalence evidence.

W1 (#2973) owns the authoritative closure/portable scientific inputs and W2
(#2974) owns the repository harness. Neither is present at the research base.
Locate the real multi_stream_replay.py under the repository's application or
experiment tree; do not prescribe W2's package location. This search locates
ONE entry, and explicitly does not establish the declared caller closure.
These tests fail with an explicit missing-public-entry assertion in the test
body, not an import/discovery error, until the prerequisite is integrated.

The small receipt/roster below exercise real argparse binding and the real
pre-source-read admission path through main(). They are NOT teacher arrays,
checkpoint fixtures, resident-source data, or a substitute for W1's inputs.
Only the dependency byte reader is local, and setup is a tripwire: no source
read, numerical consumer, GPU execution or scientific publication is attempted.
No planner, roster validator, deadline validator or lease guard is replaced.

Expected GLM planning values are literal excerpts of the public contract:
45 decoder layers, fixed prefixed64, independent explicit stream roster, and
contiguous committed windows. The approved 3500-second bound comes from #2975;
the inspected older canonical snapshot's 1700-second ceiling is NOT its oracle.
Full canonical/migrated consumer equivalence and x86 D38 remain separate gates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
class _ReachedSourceSetup(Exception):
    """Stop after public argument/plan admission and before science I/O."""


@pytest.fixture
def public_entry(monkeypatch, bound_inputs):
    # Only metadata reads are admitted in this narrow test. Missing resident
    # data is represented by staged_range returning None, not by replacing the
    # real strict_input_leases owner or allowing unguarded scientific reads.
    _, receipt_path, roster_path = bound_inputs
    observed = {"reads": []}
    loaded_names = set()
    saved = {}

    def load():
        candidates = sorted(path for root in (ROOT / "prismaquant", ROOT / "experiments")
                            for path in root.rglob("multi_stream_replay.py"))
        assert len(candidates) == 1, (
            "#2975 needs one real repository-owned public multi_stream_replay "
            "entry; consume #2973's closure and #2974's harness. Adapter-only "
            f"tests are not caller migration evidence. Found: {candidates}; "
            f"CPU architecture={platform.machine()} (admission check, NOT D38 evidence)"
        )
        entry_path = candidates[0]
        entry_dir = entry_path.parent
        monkeypatch.syspath_prepend(str(entry_dir))
        # Support W2's selected package OR the canonical standalone layout,
        # without catching an import failure and silently trying another owner.
        parents = entry_path.relative_to(ROOT).parts[:-1]
        packaged = all((ROOT.joinpath(*parents[:i]) / "__init__.py").is_file()
                       for i in range(1, len(parents) + 1))
        prefix = ".".join(parents) + "." if packaged else ""
        names = {path.stem for path in entry_dir.glob("*.py")}
        names |= {prefix + name for name in tuple(names)}
        for name in names:
            saved[name] = sys.modules.get(name)
            sys.modules.pop(name, None)
        loaded_names.update(names)
        residency = types.ModuleType("g3_residency")
        residency.staged_range = lambda path, offset, size: None

        def read_file(path):
            path = Path(path)
            assert path in (receipt_path, roster_path), (
                "This pre-source admission test unexpectedly reached science I/O: " + str(path)
            )
            observed["reads"].append(path)
            size = path.stat().st_size
            staged = residency.staged_range(path, 0, size)
            return path.read_bytes() if staged is None else bytes(staged)

        residency.read_file = read_file
        monkeypatch.setitem(sys.modules, "g3_residency", residency)
        monkeypatch.delenv("PRISMABUILD_RESIDENCY_MAP", raising=False)
        entry = importlib.import_module(prefix + "multi_stream_replay")
        stage = importlib.import_module(prefix + "stage1a_energy")
        original_resolve = entry.resolve_cli_run

        def resolve(*args, **kwargs):
            result = original_resolve(*args, **kwargs)
            observed["run"] = result
            return result

        def stop_at_source(args):
            observed["args"] = args
            raise _ReachedSourceSetup

        monkeypatch.setattr(entry, "resolve_cli_run", resolve)
        monkeypatch.setattr(stage, "setup", stop_at_source)
        return entry, observed

    try:
        yield load
    finally:
        for name in loaded_names:
            previous = saved[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def bound_inputs(tmp_path):
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 45, "num_nextn_predict_layers": 1,
            "hidden_size": 4096, "vocab_size": 154880,
        },
    }))
    receipt = {
        "schema": "pact.prefixed_clean_frontier.v1",
        "layer_start": 0, "layer_stop": 45,
        "session": "clean-stop", "start_reference_owner": {"session": "clean-start"},
    }
    roster = {
        "schema": "pact.stream_roster.v1", "band": "3:9",
        "streams": [{
            "stream_id": "clean-null", "class": "routed", "kind": "W_T8",
            "weight_source": "fixture-T8", "amplitude": 1, "null_replay": True,
        }, {
            "stream_id": "routed-one", "class": "routed", "kind": "W_T8",
            "weight_source": "fixture-T8", "amplitude": 1, "null_replay": False,
        }],
    }
    receipt_path = tmp_path / "boundaries.json"
    roster_path = tmp_path / "roster.json"
    receipt_path.write_text(json.dumps(receipt))
    roster_path.write_text(json.dumps(roster))
    # Non-admission inputs are intentionally only bound paths, never opened.
    argv = []
    roots = {
        "pq-root": ROOT, "tessera-src": tmp_path / "tessera",
        "g3-source": tmp_path / "g3", "source-root": config_dir,
        "a8-root": tmp_path / "a8", "t8r-root": tmp_path / "t8r",
        "a4-root": tmp_path / "a4", "exl3-root": tmp_path / "exl3",
        "manifest": tmp_path / "manifest.json",
        "reference-manifest": tmp_path / "reference.json",
        "capture-root": tmp_path / "captures", "draw": tmp_path / "draw.safetensors",
        "output": tmp_path / "out" / "energy.jsonl",
        "run-manifest-out": tmp_path / "out" / "run.json",
        "teacher-frontier": tmp_path / "teacher.json",
        "teacher-content": tmp_path / "teacher-content.json",
        "energy-rows": tmp_path / "energy.jsonl", "boundaries": receipt_path,
        "stream-manifest": roster_path, "boundary-dir": tmp_path / "working-boundaries",
        "output-dir": tmp_path / "streams", "checkpoint-dir": tmp_path / "checkpoints",
    }
    for name, value in roots.items():
        argv.extend(["--" + name, str(value)])
    argv.extend([
        "--draw-sha256", "0" * 64, "--teacher-content-sha256", "0" * 64,
        "--input-contract", "prefixed_514", "--sample-range", "384:448",
        "--device", "cpu", "--dry-run-cpu", "--band", "3:9",
        "--layer-range", "3:9", "--weight-source", "fixture-T8",
        "--window-layers", "9", "--window-start", "0", "--window-stop", "9",
    ])
    return argv, receipt_path, roster_path


def _set_arg(argv, option, value):
    result = list(argv)
    result[result.index(option) + 1] = str(value)
    return result


def _main(monkeypatch, entry, argv):
    monkeypatch.setattr(sys, "argv", [entry.__file__, *argv])
    entry.main()


@pytest.mark.parametrize("deadline", [None, 1700, 3500], ids=["default", "legacy-1700", "approved-3500"])
def test_normal_cli_admits_bound_glm_plan(public_entry, bound_inputs, monkeypatch, deadline):
    entry, observed = public_entry()
    argv, receipt_path, roster_path = bound_inputs
    if deadline is not None:
        argv += ["--deadline-seconds", str(deadline)]
    with pytest.raises(_ReachedSourceSetup):
        _main(monkeypatch, entry, argv)
    args, run = observed["args"], observed["run"]
    assert args.device == "cpu" and args.dry_run_cpu
    if deadline is not None:
        assert args.deadline_seconds == deadline
    else:
        assert 0 < args.deadline_seconds <= 3500
    assert observed["reads"] == [receipt_path, roster_path]
    assert args.source_root == receipt_path.parent / "model"
    assert args.input_contract == "prefixed_514"
    assert args.sample_range == "384:448"
    assert args.prefix_ids == [154822, 154824]
    assert {key: run[key] for key in (
        "band_start", "band_stop", "replay_start", "replay_stop",
        "window_start", "window_stop", "window_layers",
    )} == {
        "band_start": 3, "band_stop": 9, "replay_start": 0, "replay_stop": 45,
        "window_start": 0, "window_stop": 9, "window_layers": 9,
    }
    assert run["plan"] == [
        {"window_start": 0, "window_stop": 9},
        {"window_start": 9, "window_stop": 18},
        {"window_start": 18, "window_stop": 27},
        {"window_start": 27, "window_stop": 36},
        {"window_start": 36, "window_stop": 45},
    ]
    assert run["manifest_streams"] == json.loads(roster_path.read_text())["streams"]
    assert run["generations"] == {
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "source_window": [0, 45], "start_generation": "clean-start",
        "stop_generation": "clean-stop",
    }


@pytest.mark.parametrize("deadline", [0, -1, 3501])
def test_normal_cli_refuses_deadline_bounds_before_source_reads(public_entry, bound_inputs, monkeypatch, deadline):
    entry, observed = public_entry()
    argv, _, _ = bound_inputs
    with pytest.raises(ValueError, match="(?i)deadline"):
        _main(monkeypatch, entry, [*argv, "--deadline-seconds", str(deadline)])
    assert "args" not in observed, "An invalid deadline reached source setup"


@pytest.mark.parametrize(("option", "value", "reason"), [
    ("--band", "9:3", "band|cohort|range"),
    ("--layer-range", "0:9", "band|cohort|range"),
    ("--window-start", "1", "window|grid"),
])
def test_normal_cli_refuses_geometry_before_source_setup(public_entry, bound_inputs, monkeypatch, option, value, reason):
    entry, observed = public_entry()
    argv, _, _ = bound_inputs
    with pytest.raises(ValueError, match=reason):
        _main(monkeypatch, entry, _set_arg(argv, option, value))
    assert "args" not in observed, "Invalid coordinates reached source setup"


@pytest.mark.parametrize("mutation", ["other-band", "duplicate-stream"])
def test_normal_cli_refuses_roster_mismatch_before_source_reads(public_entry, bound_inputs, monkeypatch, mutation):
    entry, observed = public_entry()
    argv, _, roster_path = bound_inputs
    roster = json.loads(roster_path.read_text())
    if mutation == "other-band":
        roster["band"] = "9:15"
    else:
        roster["streams"].append(dict(roster["streams"][0]))
    roster_path.write_text(json.dumps(roster))
    with pytest.raises(ValueError, match="roster|stream|band"):
        _main(monkeypatch, entry, argv)
    assert "args" not in observed, "A conflicting roster reached source setup"
