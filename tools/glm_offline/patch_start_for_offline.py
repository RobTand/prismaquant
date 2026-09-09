#!/usr/bin/env python3
r"""Generate an offline-scoring variant of the MiaAI recipe's start.sh.

The original is never edited.  Every substitution below must match exactly
once or this exits non-zero, so a recipe update that moves one of these
anchors stops the launch instead of silently launching the served
configuration under an offline name.

What the five patches do, and why each one is needed rather than an export:

  KV       `KV_CACHE_DTYPE` is not in start.sh's caller-wins list
           (start.sh:60-82), so `.env`'s `fp8` would win over any export.  We
           force the empty string, which is what asks the engine for `auto`.
           That is a request, not an observation: the runtime's
           mla_attention.py can still select fp8_ds_mla for the sparse
           backend, so the reading is recorded as requested-auto with the
           actual dtype unmeasured.

  EXEC     the head's inner script is regenerated on every run
           (start.sh:983), so a frozen copy of it goes stale.  We let the
           recipe generate it in full -- every overlay patch, every preamble
           line -- and swap only its terminal `exec vllm serve` for our
           entrypoint.  The worker's identical exec at start.sh:1185 is
           untouched: it must still join as a worker.

  LAUNCH   the mounts and `MODEL_DIR` are settled at the top of
           `launch_cluster`, BEFORE serve_env is assembled (start.sh:1282) and
           before the worker's docker run (start.sh:1303).  serve_env carries
           MODEL_DIR to the worker, so a later override would reach the head
           only and the two ranks would name different models.

  WORKER   the worker needs the same /model bind at the same path.  Both ranks
           load the artifact themselves; only the head runs the entrypoint.

  RUN      the head docker run has no bind for our entrypoint, the panel, the
           token files or an output directory, and its `-v` list is literal.
           `offline_mounts` carries `-e` entries too, so one insertion point
           serves both; the array is expanded where docker takes either.
"""
from __future__ import annotations

import sys
from pathlib import Path

KV_ANCHOR = 'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"\n'
KV_PATCH = KV_ANCHOR + (
    '# offline scoring: forced AFTER the line above, not before it.  That line\n'
    "# uses ${VAR:-default}, which substitutes on empty as well as on unset, so\n"
    '# an empty value settled any earlier is quietly restored to fp8.  .env pins\n'
    '# fp8 and KV_CACHE_DTYPE is not in the caller-wins block (start.sh:60-82),\n'
    '# so nothing an operator exports reaches here either.\n'
    '#\n'
    '# Empty omits the flag (start.sh:1010,1114), which ASKS the engine for\n'
    '# `auto`.  A request, not an observation: mla_attention.py can still select\n'
    '# fp8_ds_mla for the sparse backend.\n'
    '#\n'
    '# OFFLINE_KV_CACHE_DTYPE is the deliberate opt-out, for asking for a dtype\n'
    '# on purpose rather than inheriting one by accident.\n'
    'KV_CACHE_DTYPE="${OFFLINE_KV_CACHE_DTYPE-}"\n'
)

EXEC_ANCHOR = '    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"\n'
EXEC_PATCH = EXEC_ANCHOR + (
    "    # offline scoring: keep every generated preamble line, swap only the\n"
    "    # head's terminal exec.  The worker's identical exec is not touched.\n"
    '    if [ -n "${OFFLINE_ENTRY:-}" ]; then\n'
    '        sed -i "s|^exec vllm serve .*|exec python3 ${OFFLINE_ENTRY}|" '
    '"$HEAD_SCRIPT"\n'
    '        grep -qx "exec python3 ${OFFLINE_ENTRY}" "$HEAD_SCRIPT" \\\n'
    '            || die "offline: the head exec swap did not take"\n'
    '    fi\n'
)

LAUNCH_ANCHOR = 'launch_cluster() {\n'
LAUNCH_PATCH = LAUNCH_ANCHOR + (
    '    # offline scoring: settled here, before serve_env is assembled\n'
    '    # (start.sh:1282) and before the worker launches (start.sh:1303).\n'
    '    # serve_env carries MODEL_DIR to the worker, so a later override\n'
    '    # would reach the head alone and the ranks would disagree.\n'
    '    offline_mounts=()\n'
    '    offline_worker_mounts=""\n'
    '    if [ -n "${OFFLINE_MODEL_HOST:-}" ]; then\n'
    '        [ -d "${OFFLINE_MODEL_HOST}" ] \\\n'
    '            || die "OFFLINE_MODEL_HOST is not a directory: ${OFFLINE_MODEL_HOST}"\n'
    '        offline_mounts+=(-v "${OFFLINE_MODEL_HOST}:/model:ro")\n'
    '        offline_worker_mounts="-v ${OFFLINE_MODEL_HOST}:/model:ro"\n'
    '        MODEL_DIR=/model\n'
    '    fi\n'
    '    if [ -n "${OFFLINE_ENTRY_HOST:-}" ]; then\n'
    '        offline_mounts+=(-v "${OFFLINE_ENTRY_HOST}:${OFFLINE_ENTRY}:ro")\n'
    '    fi\n'
    '    if [ -n "${OFFLINE_PANEL_HOST:-}" ]; then\n'
    '        offline_mounts+=(-v "${OFFLINE_PANEL_HOST}:/opt/glm53/panel.json:ro")\n'
    '    fi\n'
    '    if [ -n "${OFFLINE_INVENTORY_HOST:-}" ]; then\n'
    '        # Bound at its OWN absolute path.  The panel handoff carries\n'
    '        # absolute host tokens_path values with a sha256 beside each one;\n'
    '        # binding the tree elsewhere would need those rewritten, and a\n'
    '        # rewritten path is not the object the digest names.\n'
    '        offline_mounts+=(-v "${OFFLINE_INVENTORY_HOST}:${OFFLINE_INVENTORY_HOST}:ro")\n'
    '    fi\n'
    '    if [ -n "${OFFLINE_OUT_HOST:-}" ]; then\n'
    '        offline_mounts+=(-v "${OFFLINE_OUT_HOST}:/out")\n'
    '    fi\n'
    '    # Placeholder-entry inputs only.  The sealed adapter reads\n'
    '    # /out/scorer-args.json and takes the rest from argv, so these are\n'
    '    # passed only when a caller sets them.\n'
    '    [ -n "${PANEL_JSON:-}" ] && offline_mounts+=(-e "PANEL_JSON=${PANEL_JSON}")\n'
    '    [ -n "${OUT_JSON:-}" ] && offline_mounts+=(-e "OUT_JSON=${OUT_JSON}")\n'
    '    # Noether needs scorer, source, teacher and candidate-cache mounts on\n'
    '    # top of these.  They go in through here, one docker -v spec per line,\n'
    '    # so adding one is a variable rather than another patch anchor:\n'
    '    #   OFFLINE_EXTRA_MOUNTS=$\'/a/b:/a/b:ro\\n/c:/d:ro\'\n'
    '    # Each spec is checked for a readable host side first, because a bad\n'
    '    # bind that docker accepts becomes an empty directory inside, and an\n'
    '    # empty directory is the failure that looks like a missing file much\n'
    '    # later and somewhere else.\n'
    '    while IFS= read -r _spec; do\n'
    '        [ -n "$_spec" ] || continue\n'
    '        [ -e "${_spec%%:*}" ] \\\n'
    '            || die "OFFLINE_EXTRA_MOUNTS: no such host path: ${_spec%%:*}"\n'
    '        offline_mounts+=(-v "$_spec")\n'
    '    done <<< "${OFFLINE_EXTRA_MOUNTS:-}"\n'
    '    while IFS= read -r _spec; do\n'
    '        [ -n "$_spec" ] || continue\n'
    '        [ -e "${_spec%%:*}" ] \\\n'
    '            || die "OFFLINE_EXTRA_WORKER_MOUNTS: no such host path: ${_spec%%:*}"\n'
    '        offline_worker_mounts="${offline_worker_mounts} -v $_spec"\n'
    '    done <<< "${OFFLINE_EXTRA_WORKER_MOUNTS:-}"\n'
    '    # serve_env (start.sh:1282) is a fixed name list.  Anything outside it\n'
    '    # cannot reach the worker that way, and the adapter needs six names on\n'
    '    # BOTH ranks (PYTHONPATH, PYTHONDONTWRITEBYTECODE, PRISMAQUANT_TMPDIR\n'
    '    # and the three thread caps): the RPC imports and the BLAS calls happen\n'
    "    # on the workers too, not only on the head.  One NAME=VALUE per line.\n"
    '    while IFS= read -r _env; do\n'
    '        [ -n "$_env" ] || continue\n'
    '        case "$_env" in\n'
    '            *=*) ;;\n'
    '            *) die "OFFLINE_EXTRA_ENV needs NAME=VALUE, got: $_env" ;;\n'
    '        esac\n'
    '        offline_mounts+=(-e "$_env")\n'
    '        offline_worker_mounts="${offline_worker_mounts} -e $_env"\n'
    '    done <<< "${OFFLINE_EXTRA_ENV:-}"\n'
)

WORKER_ANCHOR = '        ${worker_preload} \\\n'
WORKER_PATCH = '        ${offline_worker_mounts} \\\n' + WORKER_ANCHOR

RUN_ANCHOR = '    docker run -d --name "$CONTAINER_HEAD" \\\n'
RUN_PATCH = RUN_ANCHOR + '        "${offline_mounts[@]}" \\\n'

COMPLETE_ANCHOR = """    launch_cluster
    if wait_for_health; then
"""
# The offline head is SUPPOSED to exit: it scores, writes its JSON and stops,
# and never opens $PORT.  wait_for_health (start.sh:1435-1437) treats any head
# exit as a failure, exit 0 included, so without this branch a perfect run is
# reported as "server did not become healthy" and the launcher's exit status
# says nothing about the result.  Wait for the exit instead and judge it on the
# two things the run actually produces: the head's exit code and the artifact.
# post_ready_warmup and on_ready are skipped entirely -- both speak HTTP to a
# server that by design never listens.
# ---------------------------------------------------------------------------
# Paired-worker lifecycle
# ---------------------------------------------------------------------------
# The offline head exits by design, so `wait_for_health` never runs and nothing
# downstream ever stopped the worker.  It held the worker box at 115 GB of
# 121 GB after two runs in one morning, both times until someone stopped it by
# hand.  A launcher that leaves half its cluster running has not finished.
#
# Four things this has to get right, and a first attempt got none of the last
# three:
#
#   identity   The id comes from `docker run -d` itself, which prints the id of
#              the container it created.  Resolving `$CONTAINER_WORKER` by name
#              later can name a LATER run's worker that reused it, and stopping
#              that is worse than leaking this one.  It is validated as exactly
#              64 lowercase hex before anything is done with it: ssh writes
#              banners to stdout, and a non-id string must never be
#              interpolated into a remote `docker stop`.
#   coverage   Registered at creation, before the head is even started, so a
#              failure between the two -- the head's `docker run`, a preflight
#              inside it, anything -- still stops the worker.  Registering
#              after `launch_cluster` returns covers only a launch that fully
#              succeeded, which is the case that was already fine.
#   interrupt  EXIT/INT/TERM, because the common way to end a long wait is
#              Ctrl-C, and that path took no cleanup at all.
#   honesty    A stop that fails must not be able to end in "run complete".
#              The head's own status stays separate and is still reported; the
#              launcher's exit status becomes nonzero, because a worker still
#              holding the box is not a finished run.  A docker or ssh failure
#              is not absence, a missing id fails the run closed rather than
#              warning, and a failed stop keeps the id so ownership can be
#              reported and the stop retried at exit.
WORKER_ID_ANCHOR = "    worker_ssh \"docker run -d --name '$CONTAINER_WORKER' \\\n"
WORKER_ID_PATCH = "    offline_worker_cid=\"$(worker_ssh \"docker run -d --name '$CONTAINER_WORKER' \\\n"

WORKER_ID_TAIL_ANCHOR = "        --entrypoint bash '$IMAGE' /start.sh\" >/dev/null\n"
WORKER_ID_TAIL_PATCH = (
    "        --entrypoint bash '$IMAGE' /start.sh\")\"\n"
    "    offline_worker_run_out=\"$(printf '%s' \"$offline_worker_cid\" | tr -d '\\r')\"\n"
    "    # A container id is exactly 64 lowercase hex characters.  Take the last\n"
    "    # line that IS one, rather than the last line: ssh writes banners and\n"
    "    # warnings to stdout, before and after, and anything that is not an id\n"
    "    # must never reach a docker stop -- both because it would name the wrong\n"
    "    # thing and because it would be interpolated into a remote command.\n"
    "    offline_worker_cid=\"$(printf '%s\\n' \"$offline_worker_run_out\" \\\n"
    "        | grep -E '^[0-9a-f]{64}$' | tail -n 1 || true)\"\n"
    "    offline_cleanup_rc=0\n"
    "    # Registered here, one statement after the worker exists and before the\n"
    "    # head is started, so every later failure path is covered.\n"
    "    #\n"
    "    # A docker or ssh failure is NOT absence.  Only docker saying the\n"
    "    # container does not exist proves the worker is gone; an unreachable\n"
    "    # daemon, a dropped link or a refused login means the worker may well\n"
    "    # still be running and this run still owns it.  The id is therefore\n"
    "    # cleared only on a proven stop or a proven absence, and kept otherwise,\n"
    "    # so ownership can be reported and the stop retried at exit.\n"
    "    offline_stop_paired_worker() {\n"
    "        [ -n \"${offline_worker_cid:-}\" ] || return 0\n"
    "        _cid=\"$offline_worker_cid\"\n"
    "        _out=\"$(worker_ssh \"docker inspect -f '{{.Id}}' '$_cid'\" 2>&1)\" && _rc=0 || _rc=$?\n"
    "        if [ \"$_rc\" != 0 ]; then\n"
    "            case \"$_out\" in\n"
    "                *'No such object'*|*'No such container'*)\n"
    "                    log \"paired worker ${_cid:0:12} is gone; docker reports no such container\"\n"
    "                    offline_worker_cid=\"\"\n"
    "                    return 0 ;;\n"
    "                *)\n"
    "                    offline_cleanup_rc=1\n"
    "                    log \"could not reach ${WORKER_SSH} to check paired worker ${_cid:0:12}: $_out\"\n"
    "                    log \"this is NOT evidence the worker stopped; the id is retained and this run still owns $_cid\"\n"
    "                    return 1 ;;\n"
    "            esac\n"
    "        fi\n"
    "        mkdir -p \"$LOGDIR\" 2>/dev/null || true\n"
    "        worker_ssh \"docker inspect '$_cid' --format 'status={{.State.Status}} exit={{.State.ExitCode}} started={{.State.StartedAt}} pid={{.State.Pid}}'\" \\\n"
    "            > \"$LOGDIR/worker-final-state.txt\" 2>&1 || true\n"
    "        worker_ssh \"docker logs --tail 200 '$_cid'\" \\\n"
    "            > \"$LOGDIR/worker-final.log\" 2>&1 || true\n"
    "        if worker_ssh \"docker stop -t 60 '$_cid' >/dev/null && docker rm '$_cid' >/dev/null\"; then\n"
    "            log \"paired worker ${_cid:0:12} stopped; state and last 200 lines in $LOGDIR/\"\n"
    "            offline_worker_cid=\"\"\n"
    "            return 0\n"
    "        fi\n"
    "        offline_cleanup_rc=1\n"
    "        log \"paired worker ${_cid:0:12} did NOT stop; it is still on ${WORKER_SSH}\"\n"
    "        log \"the id is retained and this run still owns $_cid\"\n"
    "        return 1\n"
    "    }\n"
    "    # A signal handler that only cleans up turns an interrupt into a\n"
    "    # zero exit, so the caller cannot tell it was interrupted.  Exit as the\n"
    "    # signal would have.\n"
    "    offline_on_signal() {\n"
    "        _sig=\"$1\"; _num=\"$2\"\n"
    "        log \"caught ${_sig}: stopping the paired worker before exiting\"\n"
    "        offline_stop_paired_worker || true\n"
    "        trap - EXIT INT TERM\n"
    "        exit \"$((128 + _num))\"\n"
    "    }\n"
    "    # A cleanup failure at exit must reach the exit status.  The head's own\n"
    "    # status is preserved when it is already nonzero, because that is the\n"
    "    # more specific failure; a zero one is replaced, because a worker still\n"
    "    # holding the box is not a finished run.\n"
    "    offline_on_exit() {\n"
    "        _status=$?\n"
    "        trap - EXIT INT TERM\n"
    "        if ! offline_stop_paired_worker; then\n"
    "            if [ \"$_status\" = 0 ]; then\n"
    "                _status=75\n"
    "                log \"exiting 75: the run finished but the paired worker was not stopped\"\n"
    "            fi\n"
    "        fi\n"
    "        exit \"$_status\"\n"
    "    }\n"
    "    if [ -n \"${OFFLINE_ENTRY:-}\" ]; then\n"
    "        if [ -z \"$offline_worker_cid\" ]; then\n"
    "            die \"docker run started a worker on ${WORKER_SSH} but printed no 64-hex container id; this run cannot stop what it cannot name. Last line was: $(printf '%s' \"$offline_worker_run_out\" | tail -n 1). Check for a container named ${CONTAINER_WORKER} there and remove it by hand\"\n"
    "        fi\n"
    "        log \"offline entry: paired worker is ${offline_worker_cid:0:12} on ${WORKER_SSH}\"\n"
    "        trap offline_on_exit EXIT\n"
    "        trap 'offline_on_signal INT 2' INT\n"
    "        trap 'offline_on_signal TERM 15' TERM\n"
    "    fi\n")

COMPLETE_ANCHOR = """    launch_cluster
    if wait_for_health; then
"""
# The offline head is SUPPOSED to exit: it scores, writes its JSON and stops,
# and never opens $PORT.  wait_for_health (start.sh:1435-1437) treats any head
# exit as a failure, exit 0 included, so without this branch a perfect run is
# reported as "server did not become healthy" and the launcher's exit status
# says nothing about the result.  Wait for the exit instead and judge it on the
# two things the run actually produces: the head's exit code and the artifact.
# post_ready_warmup and on_ready are skipped entirely -- both speak HTTP to a
# server that by design never listens.
COMPLETE_PATCH = """    launch_cluster
    if [ -n "${OFFLINE_ENTRY:-}" ]; then
        log "offline entry: waiting for the head to finish; it never opens ${PORT}"
        docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
        offline_logpid=$!
        offline_rc=0
        docker wait "$CONTAINER_HEAD" >/dev/null 2>&1 || offline_rc=255
        [ "$offline_rc" = 255 ] || offline_rc="$(docker inspect "$CONTAINER_HEAD" --format '{{.State.ExitCode}}' 2>/dev/null || echo 255)"
        kill "$offline_logpid" 2>/dev/null || true
        collect_failure_logs
        offline_stop_paired_worker || true
        if [ "$offline_rc" = "0" ] && [ -s "${OFFLINE_RESULT_HOST}" ]; then
            log "offline run complete: head exit 0, $(wc -c <"${OFFLINE_RESULT_HOST}") bytes in ${OFFLINE_RESULT_HOST}"
            log "sha256 $(sha256sum "${OFFLINE_RESULT_HOST}" | cut -d' ' -f1)"
            if [ "${offline_cleanup_rc:-0}" != "0" ]; then
                die "head exit 0 and the artifact is present, but the paired worker was not stopped; it still holds ${WORKER_SSH}"
            fi
            return
        fi
        if [ -s "${OFFLINE_RESULT_HOST}" ]; then offline_seen=present; else offline_seen=absent; fi
        echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
        tail -n 60 "$LOGDIR/head.log" || true
        die "offline run failed: head exit ${offline_rc}, result ${offline_seen} at ${OFFLINE_RESULT_HOST}; full logs in $LOGDIR/"
    fi
    if wait_for_health; then
"""

PATCHES = [
    ('KV_CACHE_DTYPE forced empty', KV_ANCHOR, KV_PATCH),
    ('head exec swap', EXEC_ANCHOR, EXEC_PATCH),
    ('offline mounts and MODEL_DIR', LAUNCH_ANCHOR, LAUNCH_PATCH),
    ('worker /model bind', WORKER_ANCHOR, WORKER_PATCH),
    ('head docker run mounts', RUN_ANCHOR, RUN_PATCH),
    ('paired worker id at creation', WORKER_ID_ANCHOR, WORKER_ID_PATCH),
    ('paired worker cleanup registration', WORKER_ID_TAIL_ANCHOR, WORKER_ID_TAIL_PATCH),
    ('offline completion branch', COMPLETE_ANCHOR, COMPLETE_PATCH),
]

def apply_patches(text):
    """Return the offline variant of ``text``, or raise ``PatchError``.

    Every anchor must match exactly once.  A recipe update that moves one is a
    launch-stopping error rather than a silent partial patch, because a start.sh
    that took four of eight substitutions launches the served configuration
    under an offline name.
    """
    for label, anchor, replacement in PATCHES:
        found = text.count(anchor)
        if found != 1:
            raise PatchError(
                f'offline patch {label!r}: anchor matched {found} times, '
                f'expected exactly 1.  start.sh has moved; do not launch.')
        text = text.replace(anchor, replacement)

    # The worker's exec must survive verbatim: the swap above is applied to the
    # generated head file at run time, not to this text, but assert it anyway so
    # a future patch that reached into the heredoc is caught here.
    if text.count('exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"') != 2:
        raise PatchError(
            'offline patch: the two inner-script execs are no longer both '
            'present; do not launch.')
    return text


class PatchError(RuntimeError):
    """An anchor did not match exactly once, or an invariant did not hold."""


def main(argv):
    source, dest = Path(argv[1]), Path(argv[2])
    try:
        text = apply_patches(source.read_text())
    except PatchError as exc:
        sys.exit(str(exc))
    dest.write_text(text)
    dest.chmod(0o755)
    print(f'wrote {dest} ({len(PATCHES)} patches applied)')


if __name__ == '__main__':
    main(sys.argv)
