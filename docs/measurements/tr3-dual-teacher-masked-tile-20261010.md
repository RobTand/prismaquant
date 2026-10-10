# Dual-teacher TR3 procedure for the masked-tile candidate

Part of RobTand/tessera#1176. This doc gives the k2 GPU jobs the runnable
PrismaBuild (PB) procedure for served TR3 on both teachers. The scorer is
`experiments/measure_glm_tr3_vllm.py`. It scores the candidate against both
teachers in one pass when `--teacher2` is set.

Run the client commands from a tessera worktree. Point `PRISMAQUANT_CHECKOUT`
at a PrismaQuant checkout at a commit that contains this doc. Record that
commit in the receipt. PB snapshots that checkout and runs the snapshot.

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
- Topology: tensor parallel 2, one host. The closed-world ceiling refuses TP above 2.
- Engine profile: eager, one sequence, max length 2049, token batch 2049, no chunked prefill, no prefix caching.
- KV dtype: request `auto`, declare resolved `fp8_ds_mla`. This is the only measured GLM-5.3 pair.
- Logits layout: `vllm_v2_chunk1024`. This is the only measured native GLM-5.3 layout.
- GPU memory fraction: 0.85.

Do not pass `--require-exl3-diag`. It requires a loaded EXL3 worker module.
A Tessera serve has none, so the flag would refuse.

The scorer fails closed on every pin above. A different resolved KV dtype, a
different call layout, or changed bytes refuse with the observed evidence. Use
that evidence to correct the command. Do not override a refusal.

## Job 1: serve qualification (one window)

Submit one PB GPU action per reservation. It serves the candidate and writes
the one-window hook qualification. Set the shell variables first.

```bash
export PRISMAQUANT_CHECKOUT=/path/to/prismaquant
export SERVE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
export SERVING_PYTHON=/path/to/serve-image-python
export CANDIDATE_DIR=/path/to/masked-tile-candidate
export CANDIDATE_DIGEST_CACHE=/path/to/masked-tile-digest-cache.json
export RECEIPT_ROOT=/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/$TR3_ARM/run/tr3
mkdir -p $RECEIPT_ROOT
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd $PRISMAQUANT_CHECKOUT --tag gb10 --cpus 6 --gpu \
  --demand mem_gb=96 --gpu-memory-gb 88 \
  --container-image $SERVE_IMAGE --max-attempts 1 --timeout-s 1800 \
  --env OMP_NUM_THREADS=1 --env TESSERA_SERVE_MODE=resident \
  --detach -- $SERVING_PYTHON experiments/measure_glm_tr3_vllm.py \
  --model $CANDIDATE_DIR \
  --candidate-digest-cache $CANDIDATE_DIGEST_CACHE \
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
  --tensor-parallel-size 2 --nnodes 1 \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --execution-mode eager \
  --qualify-hook \
  --output $RECEIPT_ROOT/hook-qualification.json
```

Name `$TR3_ARM` after the masked-tile arm in tessera#1176. Keep the two k2
jobs on separate reservations. Hash the qualification after it completes:

```bash
QUAL_SHA=$(sha256sum $RECEIPT_ROOT/hook-qualification.json | cut -d' ' -f1)
```

## Job 2: dual-teacher score (full panel)

Run this only after job 1 passes. It replays the qualification binding, then
scores all 25 windows against both teachers in one engine load.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd $PRISMAQUANT_CHECKOUT --tag gb10 --cpus 6 --gpu \
  --demand mem_gb=96 --gpu-memory-gb 88 \
  --container-image $SERVE_IMAGE --max-attempts 1 --timeout-s 1800 \
  --env OMP_NUM_THREADS=1 --env TESSERA_SERVE_MODE=resident \
  --detach -- $SERVING_PYTHON experiments/measure_glm_tr3_vllm.py \
  --model $CANDIDATE_DIR \
  --candidate-digest-cache $CANDIDATE_DIGEST_CACHE \
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
  --tensor-parallel-size 2 --nnodes 1 \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --execution-mode eager \
  --qualification $RECEIPT_ROOT/hook-qualification.json \
  --qualification-sha256 $QUAL_SHA \
  --output $RECEIPT_ROOT/full-vocabulary-kl.json
```

Use PB priority 0 for both jobs. Run a CPU dry run first for any changed
entry point (D38). Retain the PB action keys, the CAS receipts, and the
terminal logs beside the receipt root.

Two-host variant: when the reservation spans two hosts, set `--nnodes 2`
with the reservation master address, port, and node rank, and start the peer
with `tools/gold_headless_peer.py` from the same engine kwargs. Single-host
TP 2 above is the default.

## Receipt path and format

Keep every receipt under this doc's scope in `docs/measurements`. The
measured JSON receipts live at `$RECEIPT_ROOT`:

- `$RECEIPT_ROOT/hook-qualification.json`: schema `prismaquant.glm_tr3_hook_qualification/1`, one window.
- `$RECEIPT_ROOT/full-vocabulary-kl.json`: schema `prismaquant.glm_tr3_full_vocabulary_kl/1`, 25 windows, 51175 positions.
- `$RECEIPT_ROOT/full-vocabulary-kl.runtime-<sha256>.json`: the initialized runtime observation, written before scoring.

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
