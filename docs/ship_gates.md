# Run the ship gates

Use `python -m prismaquant.ship_gates --config FILE --output JSON` inside one
PrismaBuild action. Do not submit jobs from a stage. The runner uses existing
producers and replays their evidence with the existing card verifier.

## Prepare the inputs

Export the artifact and open its canonical `shipcard.json` first. Keep job
outputs outside the artifact. Select the artifact with the existing model
profile. Do not copy the architecture roster into the job configuration.

Use a new job result path. Existing results, inputs and artifact files are not
output destinations. Stage result and log paths must be distinct.

The configuration schema is `prismaquant.ship_gates/1`. It has six fields:

- `schema`: the version above.
- `artifact`: the exported checkpoint directory.
- `topology`: explicit `tensor_parallel_size` and `nnodes`, plus the stock
  gold-engine options that the run needs. Multi-node options require explicit
  rank, master address, port and both multiprocessing backend selections.
- `serve_image`: the actual image used for serving. Keep its immutable
  deployment reference with the PrismaBuild submission.
- `inputs`: the files read by the job. Include the scientific configurations,
  teacher payload and metadata, token inputs, and existing evidence inputs.
  PrismaBuild's data manifest must also declare shared-mount data.
- `stages`: an ordered list. Its unique identifiers must equal the card's
  `required_slots`, plus `offline.g3` and `task_suite`.

Every stage has a unique `output` path. The quality stages have `config` paths.
The served gold stages have an `args` array. The runner owns their model,
output, image and topology arguments. Other stages have an `argv` array and
`record` path. A producer may write a slot record to `{output}`, or fill the
canonical card and use `record: "{shipcard}"`. The runner replays the exact
slot before it accepts the stage. It does not trust a process exit alone.

Whole-argument tokens are `{artifact}`, `{shipcard}`, `{output}`, `{image}`,
`{tp}` and `{nnodes}`. They are supported substitutions, not shell expansion.
Other arguments, including JSON strings, stay literal. Use a real deployed
producer. Do not replace a missing producer with a command that sets a pass
flag.

G3 and task configurations belong to `prismaquant.g3_v2` and
`prismaquant.task_suite`. Their schema owner is `prismaquant.quality_stage`.
Each configuration must declare criteria for production. Each criterion names
a metric, `le` or `ge`, and a finite threshold. A successful measurement with
absent criteria is `not_evaluated`, not a passed gate. Offline G3 never fills
`gold.kl`.

Use the existing `model_wikitext_inputs/2` input producer for generic models.
Pass the independent file hash with `--wikitext-inputs-sha256`. Do not normalize
WikiText a second time. Served KL needs the stored teacher and its metadata.
Use `--score-positions all` for the existing gold contract. A final-position
screen cannot close that slot. Keep graph, speculative-decode, calibration,
artifact, runtime and uniform-control correctness checks in force.

## Configure the real producers

Use the artifact's existing lane declaration to select producers. Keep the
configuration with the input evidence. The runner does not supply model names,
images, teacher data, thresholds or serving scripts.

| Stage | Existing producer or verifier | Required evidence |
|---|---|---|
| `offline.g3` | `python -m prismaquant.g3_v2 --config FILE --output JSON` | Bound candidate, paired teacher panel and explicit criteria |
| `task_suite` | `python -m prismaquant.task_suite --config FILE --output JSON` | Real backend, model, tokenizer, tasks, sampling and explicit criteria |
| `native_export.eager` | Lane eager producer; native compressed-tensors uses `validate_native_export --shipcard` | Actual eager generation record |
| `native_export.graph` | Lane graph producer; native compressed-tensors uses `validate_native_export --no-enforce-eager --shipcard` | Actual graph record under that lane's graph contract |
| `ship_gate` | `validate_quantized_model --base-url URL --model-name NAME --artifact-dir DIR --shipcard CARD` | Complete numeric and boundary-check ledger from the bound live endpoint |
| `gold.kl` | Runner-owned `measure_vllm_full_kl` student entry point | Stored teacher, metadata, all-position protocol and observed no-spec execution |
| `gold.ppl` | Runner-owned `measure_vllm_wikitext_ppl` | Generic token payload and its independent file hash |
| `route.sweep`, when declared | `validate_native_export --route-sweep-out`, then `shipcard_cli fill-route-sweep` | One real served sweep per configured rank |
| `route.census`, when declared | Public Tessera census producer, then `shipcard_cli fill-route-census` | Complete census and the exact allocation binding |
| `route.trace`, when declared | Lane trace capture, then `shipcard_cli fill-route-trace` | One actual trace per rank and an explicit rank count |
| `uniform_control`, when required | Installed `tessera.uniform_control verify`, then `shipcard_cli fill-control` | Producer block and the control checkpoint's own served gold record |

Native compressed-tensors and Tessera have different route observations. A
configuration cannot exchange those slots. The runner derives the required
set from the actual card and refuses an omitted lane slot.

A lane producer may need a serving-owner lifecycle driver. Put its actual
command in `argv`; do not copy runtime code into PrismaQuant. An endpoint
must run on the admitted action host under the declared image and artifact.
A serving deployment is an explicit input. An external service's unobserved
state is not evidence.

The generic gold producer-record interchange is `prismaquant.gold_record/1`.
The unchanged live TR3 interchange remains supported. Unknown producer
schemas and a different control path refuse. Keep the byte match, teacher,
calibration, tool and metric comparison contracts unchanged.


## Submit the action

Run the published client's `--help` before a new submission. Perform disk
admission on every eligible host. Retain the complete disk-check JSON for the
checkout, scratch and CAS filesystems.

The following CPU preflight declares eight CPUs and twelve GiB in total. This
covers the controller, sequential stage children and input preparation. Native
threads stay at one. Use an existing client environment and the published
runtime. `CHECKOUT`, `CONFIG`, `RESULT`, `DATA_MANIFEST`, `CLIENT` and
`CPU_PYTHON` are explicit operator paths.

```bash
"$CLIENT" /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd "$CHECKOUT" --tag x86 --cpus 8 --demand mem_gb=12 \
  --max-attempts 1 --timeout-s 1800 --data-manifest "$DATA_MANIFEST" \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env CUDA_VISIBLE_DEVICES= --env TMPDIR=/tmp \
  --detach -- "$CPU_PYTHON" -m prismaquant.ship_gates \
  --config "$CONFIG" --output "$RESULT" --preflight
```

CPU preflight calls the same quality and gold entry points. It reads real
inputs and checks shape, token and configuration contracts. It marks serving
stages `not_run`. It never qualifies a GPU gate.

After that preflight succeeds, the single-host production invocation has the
same configuration and module entry point. This resource declaration reserves
eight CPUs, ninety-six GiB of total host memory and an eighty-four GiB GPU
subset. It is an explicit capacity declaration, not a measured memory claim.
Use it only when the configured workload fits those limits. The serving image
must already exist on the eligible host. `SERVING_PYTHON` names its installed
interpreter; `IMAGE` is an immutable image reference.

```bash
"$CLIENT" /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd "$CHECKOUT" --tag gb10 --cpus 8 --gpu --demand mem_gb=96 \
  --gpu-memory-gb 84 --container-image "$IMAGE" \
  --max-attempts 1 --timeout-s 1800 --data-manifest "$DATA_MANIFEST" \
  --residency stage --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --detach -- "$SERVING_PYTHON" \
  -m prismaquant.ship_gates --config "$CONFIG" --output "$RESULT"
```

This documentation does not authorize a GPU submission. D49 still sets GPU
order. D50 evidence is CPU-only. A single-host declaration does not cover the
current two-Spark target.

For a multi-host run, use published `pbgang.py --manifest M --cwd CHECKOUT`.
Each member needs explicit topology, resource, image and data declarations.
The native gang provides one start barrier. It does not order stages or stop
successful peers. The kernels-owned `rank_window.py`, `managed_window.py` and
`tp2_recipe.py` own rank lifecycle. Their reported recipe covers eager, graph,
KL probe and determinism. Full route, served PPL, offline quality and
one-action coverage are not verified. Do not fork that lifecycle or invent a
second transport. Until its owner supplies the complete stage contract, the
runner refuses multi-host execution. CPU preflight can still inspect it.

## Retain the evidence

Keep the full action key at submission. Start published `pbwait.py --json KEY`
as a supervised completion client. Do not poll with an agent. Inspect the
terminal action exit, logs, result claim, CAS payload and receipt. Keep failed
actions. A submission is not completion.

The runner writes `prismaquant.ship_gates_result/1` atomically after each stage.
It retains failed results, process exits, log paths, file hashes, source and
device population. It prints produced-file hashes in stdout. Retain the files
and compare them with those hashes. PrismaBuild seals input identity, not
arbitrary output files. A successful final result requires every quality
criterion and every required card slot. Publication still uses the existing
`publish_artifact` gate.

Use `--verify-only` to replay existing quality outputs and card evidence
without rerunning measurements. Missing evidence stays a refusal. A CPU
preflight, a test count, or a primitive parity result does not prove serving,
model quality, aggregate reduction parity or multi-host behavior.
