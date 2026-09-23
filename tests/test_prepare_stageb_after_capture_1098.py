"""The Stage B preparation's pbrun call declares its read phases as progress phases (PQ #1098).

``tools/prepare_stageb_after_capture.sh`` submits the preparation with a v2
data manifest, and pbrun refuses a v2 read plan unless every read phase is
also a progress phase, in read-plan order
(``pbrun.require_linear_read_plan_progress``). Without ``--progress-phase``
the submission is refused before it is sealed.

The script runs for real here. Two stubs stand in for what it must not touch:
``PQ_PYTHON`` (the preparation step, whose outputs the test writes first) and
a ``python3`` that records the pbrun argv instead of submitting, while every
other ``python3`` call runs the real interpreter.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant import stage_b_prep_io as io  # noqa: E402

SCRIPT = ROOT / "tools" / "prepare_stageb_after_capture.sh"
PBRUN = "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"


def _entries(count):
    return [{"path": f"/mnt/shared/campaign/input-{index}.json", "offset": 0,
             "bytes": 10 + index, "sha256": f"{index:064x}"} for index in range(count)]


def _builder_manifest():
    return io.preparation_read_manifest(
        _entries(2), produced_by={"tool": "test"}, annotations={})


def _two_phase_manifest():
    manifest = _builder_manifest()
    manifest["read_plan"] = {"phases": [
        {"name": "control", "entry_indices": [0], "bytes": 10, "cumulative_bytes": 10},
        {"name": "shards", "entry_indices": [1], "bytes": 11, "cumulative_bytes": 21}],
        "read_bytes": 21}
    return manifest


def _run(tmp_path, manifest):
    checkout = tmp_path / "checkout"
    (checkout / "tools").mkdir(parents=True)
    (checkout / "tools" / "resolve_tessera_dev_pin.py").write_text(
        "print('0123456789abcdef0123456789abcdef01234567')\n")
    metadata = tmp_path / "metadata"
    submission = tmp_path / "metadata.submission"
    submission.mkdir()
    (submission / "read-manifest.json.gz").write_bytes(
        gzip.compress(json.dumps(manifest).encode()))
    (submission / "submission.json").write_text(json.dumps({
        "strict_read_flags": ["--data-manifest-sha256", "a" * 64,
                              "--allowed-tiers", "ram,ssd"]}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "pbrun-argv.json"
    stub = bin_dir / "python3"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "${{1:-}}" == {PBRUN} ]]; then\n'
        f'  exec {sys.executable} -c \'import json,sys; json.dump(sys.argv[2:], open(sys.argv[1], "w"))\' '
        f'{record} "${{@:2}}"\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n')
    stub.chmod(0o755)
    prepare = bin_dir / "pq-python"
    prepare.write_text("#!/usr/bin/env bash\nexit 0\n")
    prepare.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               PQ_CHECKOUT=str(checkout), PQ_PYTHON=str(prepare))
    result = subprocess.run(
        ["bash", str(SCRIPT), "pair.json", "b" * 64, "capture.json", "c" * 64,
         str(metadata)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    argv = json.loads(record.read_text())
    return argv[:argv.index("--")]


@pytest.mark.parametrize("build", [_builder_manifest, _two_phase_manifest],
                         ids=["preparation-manifest", "two-read-phases"])
def test_every_read_phase_is_a_progress_phase_in_order(tmp_path, build):
    manifest = build()
    options = _run(tmp_path, manifest)
    declared = [options[index + 1] for index, option in enumerate(options)
                if option == "--progress-phase"]
    timeout = options[options.index("--timeout-s") + 1]
    names = [phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert declared == [f"{name}={timeout}" for name in names]
    assert "--data-manifest" in options and "--residency" in options

