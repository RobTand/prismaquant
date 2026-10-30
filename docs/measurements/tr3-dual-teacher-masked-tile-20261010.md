# Dual-teacher TR3 procedure for the masked-tile candidate

Part of RobTand/tessera#1176. This doc gives the k2 GPU jobs the runnable
PrismaBuild (PB) procedure for served TR3 on both teachers. The scorer is
`experiments/measure_glm_tr3_vllm.py`. It scores the candidate against both
teachers in one pass when `--teacher2` is set.

Run the client commands from a tessera worktree. Set `TESSERA_CHECKOUT` to
that worktree and run each `pbrun.py` command from it. Point
`PRISMAQUANT_CHECKOUT` at a PrismaQuant checkout at a commit that contains
this doc. Record that commit in the receipt. PB snapshots the PrismaQuant
checkout named by `--cwd`. Execute the scorer from the matching, separately
frozen source checkout described below: PB's per-action closure stamp changes
its snapshot HEAD, which must not invalidate qualification replay.

## Teachers and matched input set

The matched input set is the sealed upstream final panel:

- Panel: `35f0c5c973be614f29db757e9bd4bce407ea218b974a8407ec7e64c571aad72b`.
- Shape: 25 windows, 2048 tokens each, 2047 causal rows per window.
- Panel file: `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/final_panel_handoff.json`.
- Token arrays: the `arrays/` dir beside it (25 token files plus mask).
- Reference: `zai-org/GLM-5.3-Flash-BF16`. Context length is 2048.

Both teachers bind this exact panel (`prismaquant.glm_tr3_raw_teacher/1`):

| Teacher | Manifest | SHA-256 |
| --- | --- | --- |
| `tr3-teacher-04` | `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04/artifact/teacher.json` | `1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae` |
| `tr3-teacher-exl3ref-01` | `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-exl3ref-01/artifact/teacher.json` | `da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b` |

Teacher 1 is the pact-u4 teacher. Teacher 2 is the published EXL3 reference
teacher. The scorer checks both digests before the first forward.

## Candidate

The masked-tile candidate artifact lives in tessera#1176. This doc does not
invent its path. Name it in each command as `$CANDIDATE_DIR`.

Authenticate the candidate copy on the worker host before scoring. Produce a
native digest cache with the existing source-authentication pass (same pattern
as `tr3-candidate-auth-01`). Name it as `$CANDIDATE_DIGEST_CACHE`. The scorer
refuses a stale or incomplete cache instead of rehashing.

## Serve image and topology

Serve the candidate on the measured stock vLLM image with the Tessera plugin:

- Image: `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
- It carries vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929` and Torch `2.13.0+cu130`.
- The checkpoint selects the plugin through its own `quant_method`. Pass `--quantization tessera`.
- Set `TESSERA_SERVE_MODE=resident`. Resident weights are the qualified mode.
- Topology: tensor parallel 2 across **two distinct GB10 hosts**, one CUDA device per host, node ranks 0 and 1. `pbrun --gpu` reserves one GPU, not two local TP ranks. The closed-world ceiling refuses TP above 2.
- Engine profile: eager, one sequence, max length 2049, token batch 2049, no chunked prefill, no prefix caching.
- KV dtype: request `auto`, declare resolved `fp8_ds_mla`. This is the only measured GLM-5.3 pair.
- Logits layout: `vllm_v2_chunk1024`. This is the only measured native GLM-5.3 layout.
- GPU memory fraction: 0.85.

Do not pass `--require-exl3-diag`. It requires a loaded EXL3 worker module.
A Tessera serve has none, so the flag would refuse.

The scorer fails closed on every pin above. A different resolved KV dtype, a
different call layout, or changed bytes refuse with the observed evidence. Use
that evidence to correct the command. Do not override a refusal.

## Configure the two-host reservation and container

There are still two sequential k2 stages: qualification, then full-panel score.
**Each stage needs two concurrent PB actions**, coordinator and headless peer,
on distinct hosts. Keep the same host pair, addresses, port, candidate bytes,
Tessera package, and engine settings for both stages; start a fresh peer for
stage 2. A peer from a completed action cannot be reused.

Set the following deployment parameters in the submitting shell before running
the blocks below (absolute paths; no commas in Docker bind-mount paths):

- `TESSERA_CHECKOUT`: the tessera worktree from which to submit;
  `PRISMAQUANT_CHECKOUT`: clean, committed PrismaQuant source to snapshot;
  `PB_FLEET_TOOLS`: the installed PB client directory containing `pbrun.py` and
  `pbwait.py` (the current source location is `tools/fleet/` in PrismaBuild).
- `PRISMAQUANT_SERVE_SOURCE`: a self-contained, clean, frozen checkout of that
  same source commit, staged at the same absolute path on both workers, with
  its `.git/` directory included (not a linked worktree with an external Git
  directory). `PRISMAQUANT_SERVE_COMMIT`: its full commit SHA. Both stages
  execute this source and verify HEAD/cleanliness inside the container. Their
  `producer_identity` must match exactly; independently stamped PB snapshot
  HEADs do not. The PB snapshot is retained separately at `/workspace/submission`.
- `COORD_HOST_TAG`, `PEER_HOST_TAG`: **different exact hostname tags** advertised
  by the reserved GB10 workers, not just the common `gb10` tag.
- `COORD_IP`, `PEER_IP`: their mutually reachable addresses;
  `COORD_IFACE`, `PEER_IFACE`: the socket network interfaces on those hosts;
  `MASTER_PORT`: an available TCP rendezvous port on the coordinator. Host
  networking below exposes vLLM's other distributed ports too; permit traffic
  between the pair on the trusted serving network. This recipe requests sockets,
  not RoCE. Retain NCCL's own logs before claiming the observed transport.
- `CANDIDATE_DIR`, `CANDIDATE_DIGEST_CACHE`: authenticated candidate and cache,
  present at the **same absolute paths inside and outside the container**.
  Use the same mounted candidate bytes on both hosts and authenticate both
  copies if host-local. The cache is local-stat/path sensitive; produce it in
  this mount namespace on the coordinator, not against a renamed `/model` copy.
- `TESSERA_SERVE_WHEEL`, `TESSERA_WHEEL_SHA256`: the frozen, qualified Tessera
  wheel for the masked-tile arm, staged at the same path on both workers. The
  stock image alone is not the candidate plugin. Build/stage the wheel and any
  native prerequisites before reserving GPUs. Record its digest/source commit.
  The container verifies and installs this wheel offline, with no dependency
  upgrades. Do not mount the submitting worktree over the installed package.
- `COORD_NAMESPACE`, `PEER_NAMESPACE`: D38 namespace descriptor JSON files for
  the exact container command, mounts, interpreter and prerequisite identities;
  `QUAL_COORD_D38`, `QUAL_PEER_D38`, `SCORE_COORD_D38`, `SCORE_PEER_D38`: the
  corresponding full CPU-preflight action keys. Obtain these with PB's existing
  D38 preflight before submission; a scorer argparse lint is not a D38 receipt.
  Include the frozen source commit, wheel digest and per-host network environment
  in that proof. Set and retain `TR3_RUN_ID` when preparing the final preflight
  environment; leave it unchanged through both stages, but use a new ID on rerun.

The pinned image must already be local on both workers. PB's
`--container-image` only checks availability/identity; **it does not launch a
container**. The worker-side `bash -c` below evaluates `$PWD` after PB has
materialized its snapshot, then explicitly runs Docker. The snapshot is mounted
read-only at `/workspace/submission`; the authenticated frozen source is mounted
read-only at `/workspace/prismaquant`, the container working directory and import
root. Candidate/cache and teacher inputs are read-only; receipts are read-write.
Mounting the complete input tree preserves manifest-relative tensor paths. The wheel installation is ephemeral inside each container. Both processes
use the image's `python3`, not an interpreter on the worker host.

```bash
set -euo pipefail
export TESSERA_CHECKOUT=${TESSERA_CHECKOUT:?Set the submitting tessera worktree}
cd "$TESSERA_CHECKOUT"
export PRISMAQUANT_CHECKOUT=${PRISMAQUANT_CHECKOUT:?Set the source checkout}
export PB_FLEET_TOOLS=${PB_FLEET_TOOLS:?Set the PB client tools directory}
export TR3_ARM=${TR3_ARM:-masked-tile}
export SERVE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
export INPUT_ROOT=/mnt/shared/tessera-measurements/glm-canonical-census-20260908
export RECEIPT_ROOT=${RECEIPT_ROOT:-/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/$TR3_ARM/run/tr3}
export TR3_RUN_ID=${TR3_RUN_ID:-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')}
: "${COORD_HOST_TAG:?}" "${PEER_HOST_TAG:?}" "${COORD_IP:?}" "${PEER_IP:?}"
: "${COORD_IFACE:?}" "${PEER_IFACE:?}" "${MASTER_PORT:?}"
: "${CANDIDATE_DIR:?}" "${CANDIDATE_DIGEST_CACHE:?}"
: "${PRISMAQUANT_SERVE_SOURCE:?}" "${PRISMAQUANT_SERVE_COMMIT:?}"
: "${TESSERA_SERVE_WHEEL:?}" "${TESSERA_WHEEL_SHA256:?}"
: "${COORD_NAMESPACE:?}" "${PEER_NAMESPACE:?}"
: "${QUAL_COORD_D38:?}" "${QUAL_PEER_D38:?}" "${SCORE_COORD_D38:?}" "${SCORE_PEER_D38:?}"
test "$COORD_HOST_TAG" != "$PEER_HOST_TAG"
test "$COORD_IP" != "$PEER_IP"
mkdir -p "$RECEIPT_ROOT"

# Arguments: hostname tag, host IP, interface, D38 key, namespace, stage, command...
submit_tr3() {
  local host=$1 ip=$2 iface=$3 d38=$4 namespace=$5 stage=$6
  shift 6
  python3 "$PB_FLEET_TOOLS/pbrun.py" \
    --cwd "$PRISMAQUANT_CHECKOUT" --measurement --host-class gb10 --tag "$host" \
    --cpus 6 --gpu --exclusive --demand mem_gb=96 --gpu-memory-gb 88 \
    --container-image "$SERVE_IMAGE" --max-attempts 1 --timeout-s 1800 --priority 0 \
    --d38-receipt "$d38" --d38-namespace "$namespace" \
    --env OMP_NUM_THREADS=1 --env TESSERA_SERVE_MODE=resident \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env PQ_GOLD_PEER_DRIVER=experiments.measure_glm_tr3_vllm \
    --env NCCL_IB_DISABLE=1 --env NCCL_DEBUG=INFO \
    --env "NCCL_SOCKET_IFNAME=$iface" --env "GLOO_SOCKET_IFNAME=$iface" \
    --env "VLLM_HOST_IP=$ip" --env "TR3_RUN_ID=$TR3_RUN_ID" --env "TR3_STAGE=$stage" \
    --env "SERVE_IMAGE=$SERVE_IMAGE" --env "INPUT_ROOT=$INPUT_ROOT" \
    --env "PRISMAQUANT_SERVE_SOURCE=$PRISMAQUANT_SERVE_SOURCE" --env "PRISMAQUANT_SERVE_COMMIT=$PRISMAQUANT_SERVE_COMMIT" \
    --env "CANDIDATE_DIR=$CANDIDATE_DIR" --env "CANDIDATE_DIGEST_CACHE=$CANDIDATE_DIGEST_CACHE" \
    --env "TESSERA_SERVE_WHEEL=$TESSERA_SERVE_WHEEL" --env "TESSERA_WHEEL_SHA256=$TESSERA_WHEEL_SHA256" \
    --env "RECEIPT_ROOT=$RECEIPT_ROOT" \
    --detach -- bash -c '
      exec docker run --rm --pull=never --gpus all --network host --ipc host \
        --mount "type=bind,src=$PWD,dst=/workspace/submission,readonly" \
        --mount "type=bind,src=$PRISMAQUANT_SERVE_SOURCE,dst=/workspace/prismaquant,readonly" \
        --mount "type=bind,src=$CANDIDATE_DIR,dst=$CANDIDATE_DIR,readonly" \
        --mount "type=bind,src=$CANDIDATE_DIGEST_CACHE,dst=$CANDIDATE_DIGEST_CACHE,readonly" \
        --mount "type=bind,src=$INPUT_ROOT,dst=$INPUT_ROOT,readonly" \
        --mount "type=bind,src=$TESSERA_SERVE_WHEEL,dst=$TESSERA_SERVE_WHEEL,readonly" \
        --mount "type=bind,src=$RECEIPT_ROOT,dst=$RECEIPT_ROOT" \
        --workdir /workspace/prismaquant --env PYTHONPATH=/workspace/prismaquant \
        --env OMP_NUM_THREADS --env TESSERA_SERVE_MODE \
        --env VLLM_ALLOW_INSECURE_SERIALIZATION --env VLLM_WORKER_MULTIPROC_METHOD \
        --env PQ_GOLD_PEER_DRIVER --env NCCL_IB_DISABLE --env NCCL_DEBUG \
        --env NCCL_SOCKET_IFNAME --env GLOO_SOCKET_IFNAME --env VLLM_HOST_IP \
        --env TESSERA_SERVE_WHEEL --env TESSERA_WHEEL_SHA256 --env PRISMAQUANT_SERVE_COMMIT \
        --env TR3_RUN_ID --env TR3_STAGE \
        --entrypoint /bin/bash "$SERVE_IMAGE" -c '\''
          set -euo pipefail
          git config --global --add safe.directory /workspace/prismaquant
          test "$(git rev-parse HEAD)" = "$PRISMAQUANT_SERVE_COMMIT"
          test -z "$(git status --porcelain)"
          printf "%s  %s\n" "$TESSERA_WHEEL_SHA256" "$TESSERA_SERVE_WHEEL" | sha256sum -c -
          python3 -m pip install --no-index --no-deps --force-reinstall "$TESSERA_SERVE_WHEEL"
          exec "$@"
        '\'' tr3-container "$@"
    ' tr3-worker "$@"
}

# Stock CLI spellings of scorer_engine_kwargs(), not scorer argparse flags.
# The driver declaration above lets spawned peers resolve the scorer RPCs.
start_peer() {
  submit_tr3 "$PEER_HOST_TAG" "$PEER_IP" "$PEER_IFACE" "$1" "$PEER_NAMESPACE" "$2" \
    python3 tools/gold_headless_peer.py serve "$CANDIDATE_DIR" --node-rank 1 --headless \
    --tensor-parallel-size 2 --nnodes 2 --master-addr "$COORD_IP" --master-port "$MASTER_PORT" \
    --distributed-executor-backend mp --data-parallel-backend mp \
    --dtype bfloat16 --quantization tessera --kv-cache-dtype auto \
    --gpu-memory-utilization 0.85 --max-model-len 2049 --max-num-batched-tokens 2049 \
    --max-num-seqs 1 --max-logprobs 1 --logprobs-mode raw_logprobs \
    --trust-remote-code --enforce-eager --disable-log-stats --language-model-only \
    --no-enable-prefix-caching --no-enable-chunked-prefill
}

# Wait for rank 0, then reap a peer that may still be waiting in stock vLLM.
# Withdrawal is expected teardown, not a passing peer measurement receipt.
finish_pair() {
  local coord_key peer_key rc=0
  coord_key=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["action_key"])' "$1")
  peer_key=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["action_key"])' "$2")
  python3 "$PB_FLEET_TOOLS/pbwait.py" --endings --until cleaned --wait-s 2100 "$coord_key" > "$1.endings.json" || rc=$?
  python3 "$PB_FLEET_TOOLS/pbrun.py" --withdraw "$peer_key" --reason 'TR3 coordinator ended; reap headless peer' || rc=$?
  # A withdrawn peer returns nonzero. Validate its ending instead of treating
  # every nonzero (including a timeout or unreadable evidence) as expected.
  python3 "$PB_FLEET_TOOLS/pbwait.py" --endings --until cleaned --wait-s 2100 "$peer_key" > "$2.endings.json" || :
  python3 -c '
import json, sys
coordinator, peer = (json.load(open(path)) for path in sys.argv[1:])
if not (coordinator["complete"] and peer["complete"]):
    raise SystemExit("cleanup incomplete; do not start another pair")
ending = peer["endings"][0]
if ending["status"] != "withdrawn" and not (ending["status"] == "executed" and ending["succeeded"]):
    raise SystemExit("unexpected peer ending")
' "$1.endings.json" "$2.endings.json" || rc=$?
  return "$rc"
}
```

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` permits scorer callback transport: use
only the isolated trusted serving network, not an untrusted/public network.
Both environment boundaries are explicit: `pbrun --env` sets the worker action
and `docker --env` forwards it into the container. The unique `TR3_RUN_ID` and
stage distinguish fresh peer actions from previous CAS hits.

Submit the peer detached, then immediately submit the coordinator. Do not use
`--after` or wait for the peer to finish: distributed initialization is their
rendezvous, and the peer must stay alive for every scored window. Reserve both
hosts together before starting, so queue skew does not consume the 1800-second
execution deadline. If submission fails partway, withdraw the surviving action
using its retained key; the PB deadline also bounds orphaned peers.

## Job 1: serve qualification (one window)

This pair serves the candidate and writes the one-window hook qualification.

```bash
start_peer "$QUAL_PEER_D38" qualify > "$RECEIPT_ROOT/qualify-peer-submission.json"
submit_tr3 "$COORD_HOST_TAG" "$COORD_IP" "$COORD_IFACE" "$QUAL_COORD_D38" "$COORD_NAMESPACE" qualify \
  python3 experiments/measure_glm_tr3_vllm.py \
  --model "$CANDIDATE_DIR" \
  --candidate-digest-cache "$CANDIDATE_DIGEST_CACHE" \
  --panel /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/final_panel_handoff.json \
  --arrays-root /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/arrays \
  --teacher /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04/artifact/teacher.json \
  --teacher-sha256 1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae \
  --teacher2 /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-exl3ref-01/artifact/teacher.json \
  --teacher2-sha256 da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b \
  --serve-image localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  --quantization tessera \
  --kv-cache-dtype auto --expected-kv-cache-dtype fp8_ds_mla \
  --logits-layout vllm_v2_chunk1024 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
  --master-addr "$COORD_IP" --master-port "$MASTER_PORT" \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --execution-mode eager \
  --qualify-hook \
  --output "$RECEIPT_ROOT/hook-qualification.json" \
  > "$RECEIPT_ROOT/qualify-coordinator-submission.json"
finish_pair "$RECEIPT_ROOT/qualify-coordinator-submission.json" "$RECEIPT_ROOT/qualify-peer-submission.json"
QUAL_SHA=$(sha256sum "$RECEIPT_ROOT/hook-qualification.json" | cut -d' ' -f1)
```

`$TR3_ARM` defaults to `masked-tile` above. Use the masked-tile arm name
from tessera#1176 when it differs. Keep the qualification and full-panel stages
on separate reservations; release both actions before starting the next pair.

## Job 2: dual-teacher score (full panel)

Run this only after job 1 passes, from the same tessera worktree shell with
its variables/functions still set. It starts a new matching peer, replays the
qualification binding, then scores all 25 windows against both teachers in
one engine load. Do not change the rendezvous/topology between stages.

```bash
cd "$TESSERA_CHECKOUT"
start_peer "$SCORE_PEER_D38" score > "$RECEIPT_ROOT/score-peer-submission.json"
submit_tr3 "$COORD_HOST_TAG" "$COORD_IP" "$COORD_IFACE" "$SCORE_COORD_D38" "$COORD_NAMESPACE" score \
  python3 experiments/measure_glm_tr3_vllm.py \
  --model "$CANDIDATE_DIR" \
  --candidate-digest-cache "$CANDIDATE_DIGEST_CACHE" \
  --panel /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/final_panel_handoff.json \
  --arrays-root /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-inputs-01/arrays \
  --teacher /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-04/artifact/teacher.json \
  --teacher-sha256 1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae \
  --teacher2 /mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-teacher-exl3ref-01/artifact/teacher.json \
  --teacher2-sha256 da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b \
  --serve-image localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  --quantization tessera \
  --kv-cache-dtype auto --expected-kv-cache-dtype fp8_ds_mla \
  --logits-layout vllm_v2_chunk1024 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
  --master-addr "$COORD_IP" --master-port "$MASTER_PORT" \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --execution-mode eager \
  --qualification "$RECEIPT_ROOT/hook-qualification.json" \
  --qualification-sha256 "$QUAL_SHA" \
  --output "$RECEIPT_ROOT/full-vocabulary-kl.json" \
  > "$RECEIPT_ROOT/score-coordinator-submission.json"
finish_pair "$RECEIPT_ROOT/score-coordinator-submission.json" "$RECEIPT_ROOT/score-peer-submission.json"
```

Retain all four PB action keys/generations, CAS receipts and terminal logs
beside the receipt root, including peer startup/errors and transport logs.
The CPU lint exercises the submission/container argv and checks the peer profile
against `headless_peer_argv(scorer_engine_kwargs(...))`. It does **not** establish
GPU qualification, plugin/image compatibility, successful rendezvous, or KL.
Those require the real two-host serving reservations; no measurement is claimed
by this procedure document.

## Receipt path and format

The worker writes the measured JSON receipts to `$RECEIPT_ROOT` on shared
storage. Copy each receipt into this repository at
`docs/measurements/tr3-dual-teacher-masked-tile-20261010/` with the same
file name, and commit the copy. The committed copy under
`docs/measurements` is the receipt of record:

- `docs/measurements/tr3-dual-teacher-masked-tile-20261010/hook-qualification.json`: schema `prismaquant.glm_tr3_hook_qualification/1`, one window.
- `docs/measurements/tr3-dual-teacher-masked-tile-20261010/full-vocabulary-kl.json`: schema `prismaquant.glm_tr3_full_vocabulary_kl/1`, 25 windows, 51175 positions.
- `docs/measurements/tr3-dual-teacher-masked-tile-20261010/full-vocabulary-kl.runtime-<sha256>.json`: the initialized runtime observation, written before scoring.

The full-panel result must carry these fields:

- `runtime_binding.teacher_sha256`: the tr3-teacher-04 sha.
- `runtime_binding.teacher2_sha256`: the tr3-teacher-exl3ref-01 sha.
- `runtime_binding.panel_sha256`: the panel sha.
- `runtime_binding.candidate_identity`: the authenticated candidate checkpoint identity.
- `runtime_binding.serve_image`: the immutable serve image above.
- `runtime_binding.logits_layout`, engine kwargs, and worker observations.
- `second_teacher_full_vocabulary_kl.<teacher2-sha>`: with `estimator`, `per_position_kl`, `teacher_source_execution`, and `summary`.
- `calibration_contract` and `calibration_contract_sha256`.
- `measurement_fidelity`, `summary`, `serve_manifest`, `prompt_alignment`, `rank_calls`.

The comparable ship number is `KL(their teacher || served candidate)`: the
mean of `second_teacher_full_vocabulary_kl`. The gate compares it against the
EXL3 published KL `0.024554564` on this sealed panel. The teacher-1 block is
the pact-u4 number. Neither transfers to another serve, image, or topology.

## CPU verification of this procedure

The launch lint uses shell-boundary test doubles: no PB submission, Docker
entry point, wheel installation, vLLM engine or model forward runs in these
checks. It checks both stages' container/mount/environment argv, distinct-host
placement, matched peer engine options, stable execution source despite different
PB snapshots, and stopping before scoring when peer cleanup is incomplete.

```console
python -m pytest tests/test_tr3_dual_teacher_procedure_2605.py tests/test_glm_tr3_dual_teacher.py tests/test_glm_tr3_full_vocab.py tests/test_glm_tr3_gold_bridge.py tests/test_tr3_uniform_contract.py tests/test_gold_engine_options.py tests/test_gold_headless_peer.py tests/test_doc_claims_driver_truth.py tests/test_docs_entrypoints.py tests/test_docs_staleness.py -q -rs --basetemp="$TMPDIR/pytest"
```

Observed: **293 passed, 2 skipped** on CPU. The two skips require CUDA for
resident teacher windows / `arm_capture`. No GPU serving or TP2 qualification
was performed; the checks do not supply either of the measurement receipts above.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6e07dd584f206cd68ad362bcfb8848ab49f281903716556f7444f8c5b4f97516","size":1432}},"operation":"465a45c38d24e5a60692b7e5db50b92e5db09753346850fe35925fb4efe0ad78","receipt":"271e71509645de54e7fcb04c83e8836cb0e538c6cfc12eb79679e5e25911727c"}]

### PB endings contract repair verification

PrismaBuild `43234d0f`, `tools/fleet/pbwait.py`, emits
`{"schema": "prismabuild.endings.v1", "complete": ..., "endings": [...]}`
for `--endings`. The earlier boundary double incorrectly emitted `actions`,
masking a matching error in `finish_pair()`. Both now use `endings`.

With only the double corrected, the following regression reproduced
`KeyError: 'actions'` (1 failed), before fixing the helper:

```console
python -m pytest tests/test_tr3_dual_teacher_procedure_2605.py::test_documented_jobs_enter_image_and_match_two_host_engine_profile -q --basetemp="$TMPDIR/pytest"
```

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"afcf59135d0447719a0e0cee307654b12f0fdf5ae0b5916b0cd691ad9170cf7f","size":2276}},"operation":"1eb1882e1b6c713cbd51bc0f2c4d969a23fbca8b726f249c939e462ed8c2de13","receipt":"ff465c8da8aa5d19f1ed48c73e7d2bf3245e7de569729176c4ac2a4e06507606"}]

After fixing the helper, the full CPU command above passed again:
**293 passed, 2 skipped** (the same CUDA-only cases). The shell lint now
completes both qualification and scoring pairs using the production response
shape, while still refusing incomplete peer cleanup. No GPU serving was run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b4831644127237db921e98a5f2c87cdfbe0d65fae03678161b35ef8d8ab70b88","size":1431}},"operation":"da81a0997f6a94b16f74aed9df6fac84f38a92b67e5e51a700f4256dd69ca33e","receipt":"b1bffff04b286195cc217331e678c1d2ba04db24398838cbb55dcc8631bc8197"}]
