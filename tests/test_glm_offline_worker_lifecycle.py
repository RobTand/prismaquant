"""The paired worker is stopped on every path that ends the offline launcher.

A native happy-path launch cannot show any of this.  It exercises exactly one
ending -- head exits zero, worker stops -- and the endings that leaked the box
are the other ones: an interrupted wait, a stop that failed, a launch that died
between starting the worker and starting the head, a worker box that could not
be reached at all.  So the shell the patcher emits is run directly here with
the worker box stubbed, and each ending is asked what it did.

The text under test is the patcher's own ``WORKER_ID_PATCH`` and
``WORKER_ID_TAIL_PATCH``, spliced together the way the patcher splices them
into start.sh.  Paraphrasing them here would test the paraphrase.

Two distinctions carry most of these cases.  A docker or ssh failure is not
absence: only docker saying the container does not exist proves the worker is
gone, and everything else means this run may still own a running worker.  And
a cleanup failure has to reach the exit status, because a launcher that
reports success while a worker holds the box has told the caller the opposite
of the truth.
"""

import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "tools" / "glm_offline" / "patch_start_for_offline.py"
RECIPE = Path("/home/rob/tmp/exl3-inventory/miaai-recipe/start.sh")

sys.path.insert(0, str(PATCHER.parent))
import patch_start_for_offline as patcher  # noqa: E402

CID = "a" * 64

# The recipe's own docker run is a line continuation; these stand in for the
# flags between the patched first line and the patched last one.
_MIDDLE = (
    "        --gpus all --network host --ipc host \\\n"
    "        -v '$MODEL_DIR:$MODEL_DIR:ro' \\\n"
)

HARNESS_HEAD = r"""
set -uo pipefail
LOGDIR="$TMP/logs"
WORKER_SSH=stub-worker
CONTAINER_WORKER=glm53-exl3-worker
MODEL_DIR=/model
IMAGE=stub-image
log() { echo "[log] $*"; }
die() { echo "[die] $*" >&2; exit 1; }

# The worker box.  SCENARIO decides what it reports.  Every command the cleanup
# can issue is answered, so an unhandled one surfaces as a failure rather than
# as a silent success.
worker_ssh() {
  local cmd="$*"
  case "$cmd" in
    *"docker run -d"*)
        case "$SCENARIO" in
          nocid)    return 0 ;;
          badcid)   echo "docker: invalid reference format."; return 0 ;;
          shortcid) echo "abc123"; return 0 ;;
          upcaseid) echo "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    return 0 ;;
          injectid) echo "\$(touch $TMP/PWNED)"; return 0 ;;
          banner)   echo "Warning: Permanently added host" ;;
          trailer)  echo "$LIVE_CID"; echo "Connection to host closed."; return 0 ;;
        esac
        echo "$LIVE_CID"
        return 0 ;;
    *"inspect -f"*)
        case "$SCENARIO" in
          gone)        echo "Error: No such object: $LIVE_CID" >&2; return 1 ;;
          unreachable) echo "ssh: connect to host stub-worker port 22: No route to host" >&2
                       return 255 ;;
        esac
        echo "$LIVE_CID"; return 0 ;;
    *"--format 'status"*) echo "status=running exit=0"; return 0 ;;
    *"docker logs"*)      echo "worker log line"; return 0 ;;
    *"docker stop"*)
        echo "attempt" >> "$TMP/stop-attempts"
        if [ "$SCENARIO" = "stopfail" ]; then return 1; fi
        if [ "$SCENARIO" = "stopfail_then_ok" ] \
           && [ "$(wc -l < "$TMP/stop-attempts")" = "1" ]; then return 1; fi
        echo "$LIVE_CID" >> "$TMP/stopped"; return 0 ;;
  esac
  echo "UNHANDLED: $cmd" >&2
  return 1
}
"""

HARNESS_TAIL = r"""
# What the launcher does once the worker exists is the scenario's business.
case "${ENDING:-none}" in
  interrupted)
      log "waiting for the head"
      kill -INT $$
      log "TRAP DID NOT EXIT"; exit 9 ;;
  terminated)
      log "waiting for the head"
      kill -TERM $$
      log "TRAP DID NOT EXIT"; exit 9 ;;
  partial)  die "the head container failed to start" ;;
  headfail) log "the head exited nonzero"; exit 7 ;;
  complete)
      offline_stop_paired_worker || true
      if [ "${offline_cleanup_rc:-0}" != "0" ]; then
          die "head exit 0 and the artifact is present, but the paired worker was not stopped; it still holds ${WORKER_SSH}"
      fi
      log "offline run complete"
      exit 0 ;;
esac
"""


def _harness_text():
    return (HARNESS_HEAD
            + patcher.WORKER_ID_PATCH
            + _MIDDLE
            + patcher.WORKER_ID_TAIL_PATCH
            + HARNESS_TAIL)


def _run(tmp_path, scenario, ending, offline=True):
    script = tmp_path / "harness.sh"
    script.write_text(_harness_text())
    env = {
        "PATH": "/usr/bin:/bin",
        "TMP": str(tmp_path),
        "SCENARIO": scenario,
        "ENDING": ending,
        "LIVE_CID": CID,
        "OFFLINE_ENTRY": "/opt/glm53/offline_head.py" if offline else "",
    }
    return subprocess.run(["bash", str(script)], capture_output=True,
                          text=True, env=env, timeout=60)


def _lines(tmp_path, name):
    path = tmp_path / name
    return path.read_text().split() if path.exists() else []


def _stops(tmp_path):
    return _lines(tmp_path, "stopped")


def _attempts(tmp_path):
    return _lines(tmp_path, "stop-attempts")


def test_the_spliced_harness_is_valid_shell(tmp_path):
    """If this fails every other test below is testing a syntax error."""
    script = tmp_path / "harness.sh"
    script.write_text(_harness_text())
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


# --------------------------------------------------------------------------
# The endings.
# --------------------------------------------------------------------------

def test_an_ordinary_ending_stops_the_worker_and_keeps_its_evidence(tmp_path):
    result = _run(tmp_path, "normal", "complete")
    assert result.returncode == 0, result.stderr
    assert _stops(tmp_path) == [CID]
    assert "offline run complete" in result.stdout
    # The reason to want a worker's state is usually the reason it had to be
    # stopped, so state and log are taken before the stop, not after it.
    assert (tmp_path / "logs" / "worker-final-state.txt").read_text().strip()
    assert (tmp_path / "logs" / "worker-final.log").read_text().strip()


def test_an_interrupt_stops_the_worker_and_exits_as_the_signal(tmp_path):
    """A handler that only cleans up turns Ctrl-C into a zero exit, and the
    caller can no longer tell the run was interrupted."""
    result = _run(tmp_path, "normal", "interrupted")
    assert _stops(tmp_path) == [CID], "an interrupt left the worker running"
    assert result.returncode == 128 + signal.SIGINT
    assert "TRAP DID NOT EXIT" not in result.stdout


def test_a_termination_stops_the_worker_and_exits_as_the_signal(tmp_path):
    result = _run(tmp_path, "normal", "terminated")
    assert _stops(tmp_path) == [CID]
    assert result.returncode == 128 + signal.SIGTERM
    assert "TRAP DID NOT EXIT" not in result.stdout


def test_a_launch_that_dies_before_the_head_still_stops_the_worker(tmp_path):
    """Registering cleanup after launch_cluster returns covered only a launch
    that had already fully succeeded, which was never the failing case."""
    result = _run(tmp_path, "normal", "partial")
    assert _stops(tmp_path) == [CID], "a partial launch left the worker running"
    assert result.returncode == 1


# --------------------------------------------------------------------------
# A cleanup failure reaches the exit status, and does not overwrite a more
# specific one.
# --------------------------------------------------------------------------

def test_a_stop_that_fails_cannot_end_in_run_complete(tmp_path):
    result = _run(tmp_path, "stopfail", "complete")
    assert result.returncode != 0, "a worker still holding the box exited zero"
    assert "offline run complete" not in result.stdout
    assert "did NOT stop" in result.stdout
    assert _stops(tmp_path) == []


def test_a_stop_that_fails_after_a_clean_head_exit_still_fails_the_run(tmp_path):
    """The EXIT path, not the explicit call: nothing else would catch this."""
    result = _run(tmp_path, "stopfail", "none")
    assert result.returncode == 75
    assert "exiting 75" in result.stdout


def test_a_failed_stop_does_not_overwrite_the_heads_own_failure(tmp_path):
    """A nonzero head status is the more specific failure and survives."""
    result = _run(tmp_path, "stopfail", "headfail")
    assert result.returncode == 7
    assert "did NOT stop" in result.stdout


def test_a_failed_stop_keeps_the_id_and_says_the_run_still_owns_it(tmp_path):
    result = _run(tmp_path, "stopfail", "none")
    assert "still owns" in result.stdout
    assert CID in result.stdout


def test_a_failed_stop_is_retried_at_exit(tmp_path):
    """Keeping the id is what makes a retry possible; clearing it before the
    stop made a transient failure permanent."""
    result = _run(tmp_path, "stopfail_then_ok", "complete")
    assert len(_attempts(tmp_path)) == 2
    assert _stops(tmp_path) == [CID]
    assert result.returncode == 1, "the run still reports that it had to retry"


def test_a_failed_stop_keeps_the_worker_evidence_it_managed_to_take(tmp_path):
    """State and log are captured before the stop is attempted, so the run that
    most needs them is the one that still gets them."""
    _run(tmp_path, "stopfail", "complete")
    assert (tmp_path / "logs" / "worker-final-state.txt").read_text().strip()


# --------------------------------------------------------------------------
# Absence is proven, never inferred from a failure to ask.
# --------------------------------------------------------------------------

def test_a_worker_docker_says_does_not_exist_is_treated_as_gone(tmp_path):
    result = _run(tmp_path, "gone", "complete")
    assert result.returncode == 0, result.stderr
    assert "no such container" in result.stdout
    assert _attempts(tmp_path) == []


def test_an_unreachable_worker_box_is_not_treated_as_a_stopped_worker(tmp_path):
    """A dropped link, a down daemon or a refused login says nothing about
    whether the worker is running.  Reading it as absence is how a leaked
    worker gets reported as a clean run."""
    result = _run(tmp_path, "unreachable", "complete")
    assert result.returncode != 0
    assert "NOT evidence the worker stopped" in result.stdout
    assert "could not reach stub-worker" in result.stdout
    assert _attempts(tmp_path) == [], "it tried to stop a worker it could not see"


def test_an_unreachable_box_at_exit_fails_a_run_that_otherwise_succeeded(tmp_path):
    result = _run(tmp_path, "unreachable", "none")
    assert result.returncode == 75


# --------------------------------------------------------------------------
# The id.
# --------------------------------------------------------------------------

def test_a_worker_started_without_a_usable_id_fails_the_run_closed(tmp_path):
    """Warning and continuing would leave a container running that this run
    cannot name, which is exactly the state the whole change exists to
    prevent."""
    result = _run(tmp_path, "nocid", "complete")
    assert result.returncode != 0
    assert "printed no 64-hex container id" in result.stderr
    assert "glm53-exl3-worker" in result.stderr, "it must name what to remove by hand"
    assert "offline run complete" not in result.stdout


def test_the_id_survives_ssh_noise_before_it(tmp_path):
    """worker_ssh is ssh, and ssh writes banners to stdout."""
    result = _run(tmp_path, "banner", "complete")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _stops(tmp_path) == [CID]


def test_the_id_survives_ssh_noise_after_it(tmp_path):
    """Taking the last LINE would take the trailer.  The id is the last line
    that is an id."""
    result = _run(tmp_path, "trailer", "complete")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _stops(tmp_path) == [CID]


@pytest.mark.parametrize("scenario", ["badcid", "shortcid", "upcaseid"])
def test_a_malformed_id_fails_the_run_closed(tmp_path, scenario):
    """A container id is exactly 64 lowercase hex.  Anything else names the
    wrong thing, or nothing, and must not reach a docker stop."""
    result = _run(tmp_path, scenario, "complete")
    assert result.returncode != 0
    assert "printed no 64-hex container id" in result.stderr
    assert _attempts(tmp_path) == []


def test_a_malformed_id_is_never_interpolated_into_a_remote_command(tmp_path):
    """The id is pasted into a command that runs on another box.  A string
    that is not an id must be rejected before it gets there."""
    result = _run(tmp_path, "injectid", "complete")
    assert result.returncode != 0
    assert not (tmp_path / "PWNED").exists()
    assert _attempts(tmp_path) == []


def test_the_die_message_shows_what_docker_actually_printed(tmp_path):
    """A run that fails closed here is a run someone has to clean up by hand,
    and the first thing they need is what came back."""
    result = _run(tmp_path, "badcid", "complete")
    assert "invalid reference format" in result.stderr


def test_the_worker_is_not_stopped_twice(tmp_path):
    """The ending calls cleanup and then the EXIT trap fires.  A successful
    stop clears the id, so the second call has nothing to stop; a second stop
    would target an id this run no longer owns."""
    result = _run(tmp_path, "normal", "complete")
    assert result.returncode == 0
    assert len(_attempts(tmp_path)) == 1


def test_a_serving_launch_registers_no_trap(tmp_path):
    """Only the offline branch has a head that exits by design.  A served
    cluster's worker is supposed to outlive launch_cluster."""
    result = _run(tmp_path, "normal", "headfail", offline=False)
    assert result.returncode == 7
    assert _attempts(tmp_path) == []


def test_the_id_comes_from_docker_run_not_from_a_later_name_lookup():
    """A name resolved later can name a LATER run's worker that reused it, and
    stopping that is worse than leaking this one."""
    assert patcher.WORKER_ID_PATCH.startswith('    offline_worker_cid="$(')
    assert "docker run -d" in patcher.WORKER_ID_PATCH
    body = patcher.WORKER_ID_TAIL_PATCH.split("offline_stop_paired_worker()")[1]
    body = body.split("offline_on_signal()")[0]
    assert "$CONTAINER_WORKER" not in body
    assert "docker inspect -f '{{.Id}}' '$_cid'" in body


# --------------------------------------------------------------------------
# Patch application.
#
# The recipe is a third-party file that lives outside this repo and is not
# vendored, so most of these run against a stand-in built from the patcher's
# own anchors, in the order the real file has them (worker docker run at
# start.sh:1303-1335, head docker run at :1338).  That keeps the ordering and
# refusal behaviour covered on any runner instead of skipping wherever the
# recipe is absent.  The two tests that need the real file say so and skip.
# --------------------------------------------------------------------------

def _stand_in_recipe():
    """A minimal start.sh carrying each anchor exactly once, in file order."""
    a = dict((label, anchor) for label, anchor, _ in patcher.PATCHES)
    return (
        "#!/usr/bin/env bash\n"
        + a['KV_CACHE_DTYPE forced empty']
        + '\nexec vllm serve "${MODEL_DIR}" "${ARGS[@]}"\n'
        + '\nexec vllm serve "${MODEL_DIR}" "${ARGS[@]}"\n'
        + a['head exec swap']
        + a['offline mounts and MODEL_DIR']
        + a['paired worker id at creation']
        + "        --gpus all \\\n"
        + a['worker /model bind']
        + a['paired worker cleanup registration']
        + a['head docker run mounts']
        + "        --entrypoint bash \"$IMAGE\" /start.sh >/dev/null\n"
        + "}\n"
        + a['offline completion branch']
        + "        :\n    fi\n"
    )


def test_the_stand_in_carries_every_anchor_exactly_once():
    """If it drifts from the anchors it is no longer a stand-in for anything."""
    text = _stand_in_recipe()
    for label, anchor, _ in patcher.PATCHES:
        assert text.count(anchor) == 1, f"{label}: stand-in has drifted"


def test_cleanup_is_registered_before_the_head_is_started():
    """Ordering is the whole coverage argument, and it is an ordering between
    two separate patches, so it is asserted on the emitted text."""
    generated = patcher.apply_patches(_stand_in_recipe())
    register = generated.index("trap offline_on_exit EXIT")
    head_run = generated.index('docker run -d --name "$CONTAINER_HEAD"')
    assert register < head_run


def test_the_id_is_captured_before_cleanup_is_registered():
    generated = patcher.apply_patches(_stand_in_recipe())
    capture = generated.index('offline_worker_cid="$(worker_ssh')
    register = generated.index("trap offline_on_exit EXIT")
    assert capture < register


def test_a_moved_anchor_stops_the_launch_rather_than_patching_partially():
    """A start.sh that took four of eight substitutions would launch the served
    configuration under an offline name."""
    text = _stand_in_recipe().replace(patcher.WORKER_ID_ANCHOR, "# moved\n")
    with pytest.raises(patcher.PatchError) as exc:
        patcher.apply_patches(text)
    assert "do not launch" in str(exc.value)
    assert "paired worker id at creation" in str(exc.value)


def test_a_duplicated_anchor_is_refused_too():
    """Two matches is as wrong as none: replace() would take both."""
    text = _stand_in_recipe() + patcher.LAUNCH_ANCHOR
    with pytest.raises(patcher.PatchError, match="matched 2 times"):
        patcher.apply_patches(text)


def test_a_lost_worker_exec_is_refused_after_the_patches_apply():
    """The worker must still join as a worker.  Every anchor can match and this
    can still be wrong, so it is checked separately."""
    text = _stand_in_recipe().replace(
        'exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"', 'exec something_else\n', 1)
    with pytest.raises(patcher.PatchError, match="inner-script execs"):
        patcher.apply_patches(text)


def _recipe_text():
    if not RECIPE.is_file():
        pytest.skip(f"the external GLM recipe is not on this host: {RECIPE}")
    return RECIPE.read_text()


def test_every_anchor_matches_the_real_recipe_exactly_once():
    """The anchors are literal text from a file this repo does not own, so a
    recipe update is a silent break unless something checks the real file."""
    text = _recipe_text()
    for label, anchor, _ in patcher.PATCHES:
        assert text.count(anchor) == 1, f"{label}: anchor no longer matches once"


def test_the_generated_launcher_is_valid_shell(tmp_path):
    _recipe_text()
    generated = tmp_path / "generated.sh"
    run = subprocess.run(
        [sys.executable, str(PATCHER), str(RECIPE), str(generated)],
        capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert f"({len(patcher.PATCHES)} patches applied)" in run.stdout
    check = subprocess.run(["bash", "-n", str(generated)],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
