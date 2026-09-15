#!/bin/bash
# One row-0055 arm inside an admitted PrismaBuild GPU action on a GB10.
#
#   run_arm.sh ARM SOURCE OUT [-- campaign flags...]
#
# Host side: a 2 s timeline (UTC, cumulative NFS read bytes on /mnt/shared from
# mountstats, GPU power, campaign pids) into OUT/timeline.tsv, started before
# the container and stopped after it by the PID this script started. Container
# side: entry.py runs the campaign under py-spy. The container spec and the
# campaign argv come from row_argv.py, re-pointed at SOURCE and OUT.
set -u
ARM=$1
SOURCE=$2
OUT=$3
shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
python3 "$HERE/row_argv.py" --source "$SOURCE" --out "$OUT" "$@" || exit 3

TIMELINE="$OUT/timeline.tsv"
printf 'unix\tutc\tnfs_normal_read_bytes\tnfs_server_read_bytes\tgpu_power_w\tcampaign_pids\n' > "$TIMELINE"
(
  while true; do
    rb=$(awk '/^device .* mounted on \/mnt\/shared /{m=1;next} /^device /{m=0} m && $1=="bytes:"{print $2"\t"$6; exit}' /proc/self/mountstats)
    w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    pids=$(pgrep -f 'prismaquant\.tessera_campaign --model' | tr '\n' ',')
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%s.%N)" "$(date -u +%T)" "$rb" "$w" "${pids:--}" >> "$TIMELINE"
    sleep 2
  done
) &
SAMPLER=$!

echo "arm=$ARM host=$(hostname) start=$(date +%s.%N) cpus=$(taskset -pc $$ 2>/dev/null)" | tee "$OUT/host.txt"
python3 -m tools.tessera_campaign_container --spec "$(cat "$OUT/spec.json")" -- \
  python3 experiments/stream_row_head_640/entry.py --out "$OUT"
RC=$?
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
echo "arm=$ARM end=$(date +%s.%N) rc=$RC" | tee -a "$OUT/host.txt"
exit $RC
