"""Execute the real Tessera driver arm without encoding or serving work."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from prismaquant import pipeline


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "a" * 64

#: The producer checkout's packaged contract, as the arm reads it for the
#: ``--producer-authority`` decision. v40 publishes the option; v39 (the live
#: campaign's ``a3e83875``) does not.
V40_CONTRACT = {"contract_version": 40, "producer_interface": {
    "schema": "tessera.producer-interface.v1",
    "reuse_authority": {"option": "--producer-authority",
                        "attribute": "PRODUCER_AUTHORITY",
                        "protocol": "tessera.cached_unit.ReuseAuthority",
                        "canonical_capture_attribute": "canonical_hessian_capture",
                        "drivers": ["src/tessera/export_serving.py"]}}}
V39_CONTRACT = {"contract_version": 39}


def _execute_plan_driver(script, *, env):
    """Keep a finite deadline without mistaking cold startup for a refusal.

    The admitted batch exceeded 30 seconds before its Python helpers finished.
    Runtime and binding assertions below remain authoritative; only the test
    harness gets a larger, still bounded cold-start allowance.
    """
    return subprocess.run(["bash", "-c", script], env=env,
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


def _run(tmp_path, *, changed=False, manifest="bound", mode="compiled",
         derived=False, translate=False, corrupt_derived=False, extra_env=None,
         contract=V40_CONTRACT):
    producer = tmp_path / "producer tree" / "src" / "tessera" / "serving"
    producer.mkdir(parents=True)
    (producer / "runtime_contract.json").write_text(json.dumps(contract))
    work = tmp_path / "work with spaces"
    for sub in ("artifacts", "logs", "exported"):
        (work / sub).mkdir(parents=True)
    assignment = work / "artifacts/layer_config.json"
    assignment.write_text('{"layer": "BF16"}')
    plan = work / "artifacts/tessera_plan.json"
    settings = {
        "MODEL_PATH": str(tmp_path / "model"), "TESSERA_PLAN_COVER": "as-allocated",
        "TESSERA_PLATFORM": "sm_121", "TESSERA_RUNTIME_IMAGE": IMAGE,
        "TESSERA_EXECUTION_MODE": mode, "TESSERA_RESIDENCY": "resident",
    }
    document = pipeline.stage_settings_document(settings)
    stage_path = work / "artifacts/stage_settings.json"
    stage_path.write_text(json.dumps(document))
    if manifest == "bound":
        code, messages = pipeline.check_stage_settings(
            plan, "tessera-plan", document,
            overrides={"ASSIGNMENT_DIGEST": hashlib.sha256(assignment.read_bytes()).hexdigest()},
        )
        assert code == 0, messages
    elif manifest == "other-stage":
        Path(str(plan) + ".settings.json").write_text(json.dumps({
            "schema": pipeline.STAGE_MANIFEST_SCHEMA, "stages": {"other": {}},
        }))
    if not translate:
        plan.write_text("old translated plan")
    derived_path = work / "artifacts/derived layer.json"
    derived_digest = ""
    if derived:
        derived_path.write_bytes(assignment.read_bytes())
        derived_digest = hashlib.sha256(derived_path.read_bytes()).hexdigest()
        if corrupt_derived:
            derived_path.write_text("changed after preflight")
    (work / "exported/shipcard.json").write_text("{}")
    if changed:
        assignment.write_text('{"layer": "TESSERA_E4M3_K1_R1024"}')
    script = (ROOT / "prismaquant/run-pipeline.sh").read_text()
    helper = script[script.index("require_stage_settings() {"):].split("\n}\n", 1)[0] + "\n}\n"
    start = script.index('  # Tessera lane: one blob per vLLM module')
    start = script.rfind('if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then', 0, start)
    end = script.index('\nif [[ "$EXPORT_CONTAINER" == "gguf" ]]; then', start)
    block = script[start:end]
    preamble = r'''
set -euo pipefail
TESSERA_SCOPE_ARGS=()
python3() {
  if [[ "$1" == "-m" && "$2" == "prismaquant.pipeline" ]]; then
    "$PYTEST_PYTHON" "$@"
  elif [[ "$1" == "-c" ]]; then
    "$PYTEST_PYTHON" "$@"
  elif [[ "$1" == "-m" && "$2" == "prismaquant.tessera_export_lane" ]]; then
    # The real preflight WRITES the build anchor the arm hands it, and the arm
    # now reads the priced expert wires' manifest path back out of it
    # (PrismaQuant #183). A stub that returns 0 without producing the anchor
    # models a preflight that did not do its job.
    echo "TEST_PREFLIGHT_ARGS:$*" >&2
    local previous="" build="" cached=""
    for argument in "$@"; do
      if [[ "$previous" == "--write-build-json" ]]; then
        build="$argument"
      elif [[ "$previous" == "--cached-units" ]]; then
        cached="$argument"
      fi
      previous="$argument"
    done
    # A preflight handed a composed bundle names it back in the anchor.
    "$PYTEST_PYTHON" -c 'import json, os, sys; from prismaquant.dev_mode import dev_mode_enabled; d = {"cached_encoder_source_proof_mode": "permissive" if dev_mode_enabled() else "strict"}; p = os.environ.get("TEST_PLAN_ASSIGNMENT"); d.update(plan_assignment=p, plan_assignment_sha256=os.environ["TEST_PLAN_DIGEST"]) if p else None; d.update(cached_units=sys.argv[2]) if sys.argv[2] else None; open(sys.argv[1], "w").write(json.dumps(d))' "$build" "$cached"
    return 0
  elif [[ "$1" == "-m" && "$2" == "prismaquant.tessera_plan_writer" ]]; then
    echo "TEST_PLAN_WRITER_REACHED:$3" >&2
    return 81
  elif [[ "$1" == "-m" && "$2" == "tessera.export_serving" ]]; then
    echo "TEST_EXPORT_REACHED"
    echo "TEST_EXPORT_ARGS:$*" >&2
    return 0
  else
    echo "unexpected worker: $*" >&2
    return 99
  fi
}
'''
    env = dict(os.environ, WORK_DIR=str(work), EXPORT_CONTAINER="tessera",
               TESSERA_PLAN_COVER="as-allocated", MODEL_PATH=settings["MODEL_PATH"],
               TESSERA_RUNTIME_IMAGE=IMAGE, TESSERA_EXECUTION_MODE=mode,
               TESSERA_PLATFORM="sm_121", TESSERA_RESIDENCY="resident",
               TESSERA_SERVE_MODE="resident", TESSERA_REPO=str(tmp_path / "producer tree"),
               TARGET_PROFILE_RESOLVED="vllm_tessera", EXPORT_DEVICE="cuda",
               PIPELINE_SCRIPT_DIR=str(ROOT / "prismaquant"),
               STAGE_SETTINGS_PATH=str(stage_path), PYTEST_PYTHON=sys.executable,
               TEST_PLAN_ASSIGNMENT=str(derived_path) if derived else "",
               TEST_PLAN_DIGEST=derived_digest)
    env.update(extra_env or {})
    return _execute_plan_driver(preamble + helper + block, env=env)


#: The admitted batch's cold start exceeded the older 30-second harness
#: deadline before its Python helpers finished (#1963), which is why the
#: deadline was raised to a larger, still finite allowance.
OBSERVED_COLD_START_S = 30


def test_driver_harness_deadline_absorbs_the_observed_cold_start(monkeypatch, tmp_path):
    """The plan-driver deadline must stay finite and above the cold start.

    A cold start past the older 30-second deadline false-failed the admitted
    batch, so the harness must allow more without becoming unbounded. Four
    delayed-startup cases reproduced that start with a real ``sleep 31`` each
    -- 31 wall-seconds apiece, 124 s of this file's own wall under serial
    execution -- and only re-asserted binding outcomes the real driver cases
    below already assert for themselves. Reading the deadline the driver
    actually applies keeps the same guarantee: a ``None`` or infinite deadline
    is refused at the harness boundary before the subprocess starts, and the
    old 30-second budget is caught by the remaining-budget assertion below
    after the (fast, real) driver run, without spending the wall time.
    """
    requested = {}
    real_run = subprocess.run

    def capture_deadline(argv, **kwargs):
        deadline = kwargs.get("timeout")
        requested["timeout"] = deadline
        # Check at the boundary, before delegating: an unbounded or infinite
        # deadline must be refused by this assertion, not by the OS later.
        assert deadline is not None, "the plan-driver harness must keep a finite deadline"
        assert math.isfinite(deadline), (
            "the plan-driver harness must keep a finite deadline, "
            f"got {deadline!r}")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_deadline)
    result = _run(tmp_path, changed=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ASSIGNMENT_DIGEST" in result.stdout + result.stderr
    assert requested["timeout"] > OBSERVED_COLD_START_S, (
        "the older 30-second deadline false-failed the admitted batch's cold "
        f"start; the harness must exceed {OBSERVED_COLD_START_S} seconds, "
        f"got {requested['timeout']!r}")


def test_actual_driver_refuses_old_plan_after_allocation_bytes_change(tmp_path):
    result = _run(tmp_path, changed=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ASSIGNMENT_DIGEST" in result.stdout + result.stderr
    assert "TEST_EXPORT_REACHED" not in result.stdout


@pytest.mark.parametrize("manifest", ["missing", "other-stage"])
def test_actual_driver_refuses_plan_without_independent_allocation_binding(tmp_path, manifest):
    result = _run(tmp_path, manifest=manifest)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "allocation" in (result.stdout + result.stderr).lower()
    assert "TEST_EXPORT_REACHED" not in result.stdout


def test_actual_driver_reuses_plan_for_identical_allocation(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEST_EXPORT_REACHED" in result.stdout
    assert "TEST_PLAN_WRITER_REACHED" not in result.stderr
    export = next(line for line in result.stdout.splitlines()
                  if line.startswith("TEST_EXPORT_ARGS:"))
    assert export.startswith("TEST_EXPORT_ARGS:-m tessera.export_serving"), export


@pytest.mark.parametrize("mode,eager", [("eager", "1"), ("compiled", "0")])
def test_printed_serve_recipe_retains_exact_runtime_scope(tmp_path, mode, eager):
    result = _run(tmp_path, mode=mode)
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(line.split("Serve:", 1)[1].strip() for line in result.stdout.splitlines()
                   if "Serve:" in line)
    tokens = shlex.split(command)
    assert f"IMAGE={IMAGE}" in tokens
    assert f"TESSERA_LANE_EAGER={eager}" in tokens
    assert f"TS={tmp_path / 'producer tree'}" in tokens
    assert str(tmp_path / "work with spaces/exported") in tokens


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_printed_census_recipe_keeps_raw_scope_and_bound_allocation(tmp_path, mode):
    result = _run(tmp_path, mode=mode)
    assert result.returncode == 0, result.stdout + result.stderr
    census = next(line.split("Route census:", 1)[1].strip()
                  for line in result.stdout.splitlines() if "Route census:" in line)
    tokens = shlex.split(census)
    assert "--log" not in tokens
    assert tokens[tokens.index("--runtime-image") + 1] == IMAGE
    assert ("--compiled" in tokens) == (mode == "compiled")
    close = next(line.split("Close census:", 1)[1].strip()
                 for line in result.stdout.splitlines() if "Close census:" in line)
    tokens = shlex.split(close)
    assert tokens[tokens.index("--layer-config") + 1] == str(tmp_path / "work with spaces/artifacts/layer_config.json")
    assert tokens[tokens.index("--model-dir") + 1] == str(tmp_path / "work with spaces/exported")
    assert "--priced-route" not in tokens


def test_actual_driver_translates_the_preflight_source_assignment(tmp_path):
    result = _run(tmp_path, derived=True, translate=True, manifest="missing")
    assert result.returncode == 81, result.stdout + result.stderr
    assert "TEST_PLAN_WRITER_REACHED:" + str(tmp_path / "work with spaces/artifacts/derived layer.json") in result.stdout
    assert "TEST_EXPORT_REACHED" not in result.stdout


def test_actual_driver_refuses_a_changed_derived_plan_assignment(tmp_path):
    result = _run(tmp_path, derived=True, translate=True, manifest="missing", corrupt_derived=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "derived Tessera plan assignment changed" in result.stderr
    assert "TEST_PLAN_WRITER_REACHED" not in result.stderr
    assert "TEST_EXPORT_REACHED" not in result.stdout


def test_actual_driver_refuses_a_cached_plan_unbound_to_derived_assignment(tmp_path):
    result = _run(tmp_path, derived=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PLAN_ASSIGNMENT_DIGEST" in result.stdout + result.stderr
    assert "TEST_EXPORT_REACHED" not in result.stdout


def test_a_composed_cached_bundle_reaches_preflight_and_export(tmp_path):
    """A ``TESSERA_CACHED_UNITS`` bundle replaces the expert-units write (#1413).

    The arm runs under ``set -u`` without the script's top-of-file defaults,
    so an unset optional input must read as empty, not abort the driver.
    """
    bundle = str(tmp_path / "cached units.v3.json")
    result = _run(tmp_path, extra_env={"TESSERA_CACHED_UNITS": bundle,
                                       "TESSERA_SOURCE_DIGEST_CACHE": ""})
    assert result.returncode == 0, result.stdout + result.stderr
    preflight = next(line for line in result.stderr.splitlines()
                     if line.startswith("TEST_PREFLIGHT_ARGS:"))
    assert f"--cached-units {bundle}" in preflight
    assert "--write-cached-expert-units" not in preflight
    # The export's stderr is folded into its tee'd log, so read stdout.
    export = next(line for line in result.stdout.splitlines()
                  if line.startswith("TEST_EXPORT_ARGS:"))
    assert f"--cached-units {bundle}" in export
    assert "--cached-expert-units" not in export
    assert "--source-digest-cache" not in export


@pytest.mark.parametrize("value,expected", [("0", "strict"), ("1", "permissive")])
def test_cached_driver_forwards_the_producers_explicit_mode(tmp_path, value, expected):
    result = _run(tmp_path, extra_env={"PRISMAQUANT_DEV_MODE": value,
        "TESSERA_CACHED_UNITS": str(tmp_path / "cached.json")})
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("TEST_EXPORT_ARGS:"))
    assert f"--cached-encoder-source-proof-mode {expected}" in line


def test_without_a_bundle_the_preflight_writes_expert_units(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    preflight = next(line for line in result.stderr.splitlines()
                     if line.startswith("TEST_PREFLIGHT_ARGS:"))
    assert "--write-cached-expert-units" in preflight
    assert "--cached-units" not in preflight


def _export_line(result):
    # The stub echoes ``$*``, so a path with spaces and the stub preflight's
    # empty build digest are not recoverable as tokens; compare the line.
    return next(line for line in result.stdout.splitlines()
                if line.startswith("TEST_EXPORT_ARGS:"))


def test_an_old_pin_export_gets_the_argv_it_got_before_the_authority(tmp_path):
    """A v39 checkout's exporter would refuse the option as unknown."""
    old = _run(tmp_path / "old", contract=V39_CONTRACT)
    new = _run(tmp_path / "new")
    assert old.returncode == 0 == new.returncode, old.stdout + old.stderr
    old_line, new_line = _export_line(old), _export_line(new)
    authority = str(ROOT / "prismaquant" / "tessera_reuse_authority.py")
    inserted = f" --producer-authority {authority}"
    assert "--producer-authority" not in old_line
    assert new_line.count(inserted) == 1
    assert f"{inserted} --device cuda" in new_line
    assert (new_line.replace(inserted, "", 1).replace(str(tmp_path / "new"), "<run>").encode()
            == old_line.replace(str(tmp_path / "old"), "<run>").encode())
