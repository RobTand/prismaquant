#!/usr/bin/env bash
set -euo pipefail

run_root=/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq
log_root="$run_root/logs"
sample_log="$log_root/afast-load-samples.log"
runner_log="$log_root/afast-runner.log"
runner_stop="$run_root/RUNNER_STOP.md"
stage=initializing
mkdir -p "$log_root" "$run_root/history" "$run_root/pilot2-shards" \
    "$run_root/burn-shards" "$run_root/shards"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/w

for stale in "$runner_stop" "$run_root/DSV4_CAMPAIGN_STOP.md"; do
    if [[ -f "$stale" ]]; then
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        mv "$stale" "$run_root/history/$(basename "$stale").$stamp"
    fi
done

sample_state() {
    while true; do
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        load=$(tr '\n' ' ' </proc/loadavg)
        available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
        printf '%s stage=%s loadavg=%s mem_available_bytes=%.0f\n' \
            "$timestamp" "$stage" "$load" "$available"
        sleep 30
    done
}

write_stop() {
    status=$1
    pilot_cells=$(find "$run_root/pilot2-shards" -maxdepth 1 -type f \
        -name 'layer_014_*_K*.pkl' 2>/dev/null | wc -l)
    burn_cells=$(find "$run_root/burn-shards" -maxdepth 1 -type f \
        -name 'layer_*_K*.pkl' 2>/dev/null | wc -l)
    burn_layers=$(find "$run_root/shards" -maxdepth 1 -type f \
        -name 'layer_*.pkl' 2>/dev/null | wc -l)
    tmp="$runner_stop.tmp.$$"
    {
        printf '# DSV4 A-FAST Runner Stop\n\n'
        printf -- '- Stage: %s\n' "$stage"
        printf -- '- Exit status: %s\n' "$status"
        printf -- '- Pilot-2 cells banked: %s/63\n' "$pilot_cells"
        printf -- '- Burn cells banked: %s\n' "$burn_cells"
        printf -- '- Burn layer shards banked: %s/43\n' "$burn_layers"
        printf -- '- Resume command: `tools/run_dsv4_afast_campaign.sh`\n'
        printf -- '- Allocator policy: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`\n\n'
        printf 'Every listed unit is content-keyed and is validated before reuse. '
        printf 'See `%s` and `%s` for the last streamed output and load samples.\n' \
            "$runner_log" "$sample_log"
    } >"$tmp"
    mv "$tmp" "$runner_stop"
}

sampler_pid=
cleanup() {
    status=$?
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
    if [[ $status -ne 0 ]]; then
        write_stop "$status"
    fi
}
trap cleanup EXIT INT TERM

sample_state >>"$sample_log" &
sampler_pid=$!

# The lock file is intentionally supplied from the read-only parent bind;
# flock(2) only needs an open descriptor, not write access to file contents.
exec 9</home/rob/dq-runs/gpu.lock
flock -x 9

printf '%s allocator=%s initial_load=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PYTORCH_CUDA_ALLOC_CONF" \
    "$(tr '\n' ' ' </proc/loadavg)" | tee -a "$runner_log"

stage=shakedown-v2
if python3 /w/tools/dsv4_afast_burn.py shakedown-validate \
    >>"$runner_log" 2>&1; then
    printf '%s shakedown-v2 content-resume PASS\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$runner_log"
else
    before_available=$(awk '/MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
    kill_log="$log_root/afast-shakedown-kill.log"
    : >"$kill_log"
    python3 /w/tools/dsv4_afast_burn.py shakedown-worker \
        >>"$kill_log" 2>&1 &
    shake_pid=$!
    scout_pass_tag=$(python3 -c \
        'from tools.dsv4_afast_burn import BURN_PASS_TAGS; print(BURN_PASS_TAGS["scout"])')
    first_cell="$run_root/burn-shards/layer_000_gate_proj_${scout_pass_tag}_K28.pkl"
    for _ in $(seq 1 1800); do
        if [[ -f "$first_cell" ]]; then
            break
        fi
        if ! kill -0 "$shake_pid" 2>/dev/null; then
            wait "$shake_pid"
            printf 'shakedown worker exited before deliberate kill point\n' >&2
            exit 12
        fi
        sleep 1
    done
    if [[ ! -f "$first_cell" ]]; then
        kill -TERM "$shake_pid" 2>/dev/null || true
        wait "$shake_pid" 2>/dev/null || true
        printf 'shakedown did not persist its first unit within 30 minutes\n' >&2
        exit 13
    fi
    first_cell_sha256=$(sha256sum "$first_cell" | awk '{print $1}')
    kill -TERM "$shake_pid"
    wait "$shake_pid" 2>/dev/null || true
    after_kill=$(awk '/MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)

    stage=shakedown-v2-resume
    python3 /w/tools/dsv4_afast_burn.py shakedown-worker 2>&1 \
        | tee -a "$log_root/afast-shakedown-resume.log" "$runner_log"
    after_resume=$(awk '/MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)

    stage=shakedown-v2-rederive
    python3 /w/tools/dsv4_afast_burn.py shakedown-worker 2>&1 \
        | tee -a "$log_root/afast-shakedown-rederive.log" "$runner_log"
    after_rederive=$(awk '/MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
    python3 /w/tools/dsv4_afast_burn.py shakedown-finalize \
        --before-available "$before_available" \
        --after-kill "$after_kill" \
        --after-resume "$after_resume" \
        --after-rederive "$after_rederive" \
        --first-cell-sha256 "$first_cell_sha256" 2>&1 \
        | tee -a "$runner_log"
fi

stage=pilot2
if python3 -c 'from tools.dsv4_afast_burn import PILOT2_SHARD, _load, _pilot_gate, _verify_pilot2_content; _pilot_gate(); _verify_pilot2_content(_load(PILOT2_SHARD))' \
    >>"$runner_log" 2>&1; then
    printf '%s pilot2 content-resume PASS report preserved\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$runner_log"
else
    python3 /w/tools/dsv4_afast_campaign.py pilot2 2>&1 \
        | tee -a "$log_root/pilot2-resume.log" "$runner_log"
fi
python3 -c 'import json; from pathlib import Path; p=json.loads(Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/pilot2/PILOT2_REPORT.json").read_text()); raise SystemExit(0 if p.get("decision") == "PASS" and p.get("burn_allowed") else 10)'

stage=burn
python3 /w/tools/dsv4_afast_burn.py run 2>&1 \
    | tee -a "$log_root/afast-burn.log" "$runner_log"
python3 -c 'import json; from pathlib import Path; p=json.loads(Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/burn-afast/BURN_REPORT.json").read_text()); raise SystemExit(0 if p.get("merge") == "PASS" and p.get("content_verified_layers") == 43 else 11)'

stage=phase-c-grid
python3 /w/tools/dsv4_afast_allocation_grid.py 2>&1 \
    | tee -a "$log_root/afast-grid.log" "$runner_log"

stage=campaign-report
python3 /w/tools/dsv4_afast_campaign_report.py 2>&1 \
    | tee -a "$log_root/afast-campaign-report.log" "$runner_log"

stage=complete
printf '%s complete report=%s load=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_root/DSV4_CAMPAIGN.md" \
    "$(tr '\n' ' ' </proc/loadavg)" | tee -a "$runner_log"
