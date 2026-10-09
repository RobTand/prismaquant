# New-model day-zero intake

The intake tool reads config, source-index metadata, and available safetensors headers.
It never loads tensor payloads on the default central processor path.
A draft does not register a profile, declare an export lane, or qualify a native serving cell.

## Versioned inputs

Use a committed PrismaQuant checkout and a qualified Python environment.
Use one of these source inputs:

- An existing local checkpoint with `config.json` and regular safetensors files.
- An explicit model repository identifier and its full forty-character commit.

The local source can use an index or a single `model.safetensors` file.
The indexed path uses the existing PrismaSnap checkpoint grammar.
The physical check uses the existing export-structure validator.
These owners refuse symlinked files and inconsistent index or header data.
The tool does not add an artifact cache.
Remote acquisition uses Hugging Face `snapshot_download` with an explicit `local_dir`.

## Central processor intake

Submit agent checks through PrismaBuild.
Select an x86 worker and declare zero graphics processors.
Use one native thread per test worker.
Keep validation output under `/tmp`.

Local checkpoint example:

```bash
CLIENT=/home/rob/tmp/pb-submit-celestia-20261003/bin/python
PB=/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py
CHECKOUT=/path/to/committed/prismaquant
PYTHON=/path/to/qualified/python
SOURCE=/path/to/local/checkpoint

fleet-diskcheck --need-gb 0.1 --paths /tmp
"$CLIENT" "$PB" --cwd "$CHECKOUT" --tag x86 \
  --cpus 1 --demand mem_gb=4,gpu=0 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  -- "$PYTHON" tools/day0_model_intake.py \
  --checkpoint "$SOURCE" --output /tmp/model-intake
```

Remote metadata example:

```bash
MODEL_ID=organization/model
REVISION=full-forty-character-model-repository-commit

"$CLIENT" "$PB" --cwd "$CHECKOUT" --tag x86 \
  --cpus 1 --demand mem_gb=4,gpu=0 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  -- "$PYTHON" tools/day0_model_intake.py \
  --model-id "$MODEL_ID" --revision "$REVISION" \
  --download-dir /tmp/model-source --output /tmp/model-intake
```

Use real values for these variables.
Declare source files through the normal PrismaBuild data interface when a worker needs external local inputs.
A temporary path belongs to the worker that executes the action.
Retain its outputs before that worker removes them.

The remote path downloads only `config.json` and `model.safetensors.index.json` for discovery.
It validates these inputs before any full-weight request or runtime launch.
A repository without an index produces a config-only report.
That report states that source metadata is unavailable.
It does not claim that source shapes agree with the config.

## Outputs and refusals

The tool writes these files after successful input checks:

- `structure.draft.json`: a valid `prismaquant.model_structure.v1` document with exact config match fields and observed body-layer prefix.
- `intake.json`: dimensions, detected profile, source scope, header count, profile groups, unsupported kinds, and runtime status.

The draft preserves existing profile naming rules when the registry resolves a registered profile.
It leaves `supported_lanes` empty and removes the preferred lane and default serving profile.
Keep this file outside `model_profiles/specs/` until normal review approves a profile change.
Do not use parsing success as native serving evidence.

`DefaultProfile` remains an unregistered architecture in the report.
An unknown higher-rank parameter or undeclared expert projection has an explicit unsupported-kind entry.
Header geometry is not a live module census.
The report does not qualify singleton, fused, or expert groups for a serving runtime.

The tool refuses these inputs before output publication or runtime launch:

- A remote revision that names a branch or tag instead of a full commit.
- Conflicting local and remote source inputs.
- Missing runtime image or graphics processor selection for an explicit runtime request.
- Invalid config fields, duplicate JSON keys, or inconsistent config dimensions.
- Unsafe source names, missing local shards, inconsistent tensor rosters, or invalid safetensors geometry.
- Source layers that disagree with `num_hidden_layers`.
- Observed embedding or head shapes that disagree with `vocab_size` or `hidden_size`.
- Source-name projections that collide.

Metadata discovery downloads can precede config checks.
An index-only report cannot check absent shard headers or payload bytes.
The complete local path checks every header without reading its payload.

## Later brain floating-point, degree-two check

This path is explicit and separate from central processor intake.
It starts Docker and runs the existing `tools/vllm_prompt_smoke.py` check.
The check creates a vLLM model with brain floating-point weights and tensor-parallel degree two.
It loads the model and generates a short response in eager mode.
It is not a native quantized-kernel qualification or a quality comparison.

Supply these runtime inputs:

- A serving image that contains the model's vLLM implementation and the required libraries.
- The Python executable in that image, if it differs from `/usr/bin/python3`.
- The explicit Docker graphics processor selection.
- Two available devices, or an existing Ray cluster with two compatible ranks.
- The same image, model files, code paths, and config on both ranks for a multi-host Ray check.
- Sufficient admitted memory for model weights, load transients, and the requested context.

The tool does not create a Ray cluster, distribute model files, or plan residency.
Use the existing fleet launch and admission mechanisms for those tasks.
For Ray, pass `--ray-address` and mount the same `/model` and `/source` paths on each rank.
Normal rank-agreement and byte-integrity checks remain mandatory.
No identity seal or new qualification gate is added.

First, submit the same entry point on an x86 worker:

```bash
"$CLIENT" "$PB" --cwd "$CHECKOUT" --tag x86 \
  --cpus 1 --demand mem_gb=4,gpu=0 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  -- "$PYTHON" tools/day0_model_intake.py \
  --checkpoint "$SOURCE" --output /tmp/model-preflight \
  --bf16-tp2-preflight --runtime-image "$IMAGE" --runtime-gpus all
```

The preflight runs the actual prompt-check parser without importing vLLM or acquiring a device.
It publishes the checked arguments and later Docker invocation.
It does not report a runtime pass.

Later, execute the real check with approved graphics processor admission:

```bash
"$PYTHON" tools/day0_model_intake.py \
  --checkpoint "$SOURCE" --output /path/to/check-results \
  --bf16-tp2 --runtime-image "$IMAGE" --runtime-gpus all \
  --runtime-python /usr/bin/python3 --max-model-len 512 \
  --gpu-memory-utilization 0.35
```

Agent batch checks must run inside PrismaBuild admission.
Declare actual memory and device demand for the selected runtime population.
The central processor example does not authorize a graphics processor action.
A remote `--bf16-tp2` request downloads remaining source files only after metadata checks.
The runtime path refuses quantized config or non-brain-floating-point matrix weights.
It writes `bf16-tp2.log`, the prompt result `bf16-tp2.json`, and the observed exit status in `intake.json`.
A runtime failure remains a failure and returns a nonzero exit.
The report never changes `native_serving_qualified` to true.

## Next steps

Review unsupported kinds and the draft through the normal model-profile process.
Use the existing profile validator for registered-profile checks.
Use the serving-runtime census and native correctness checks before any native export promotion.
Retain the source commit, tool commit, PrismaBuild action key, logs, and receipt with each result.
