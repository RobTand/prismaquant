# Tessera fleet drivers

The drivers in `tools/tessera_fleet/` submit Tessera exports as admitted Flux
jobs through `prismaquant/flux_jobs.py` and record native Flux job IDs under
`WORKSPACE/.flux-state/`. Workspaces written by the retired execution system
are neither read nor modified.

Run each driver as a module from the repository root:

| Driver | What it submits |
|---|---|
| `python3 -m tools.tessera_fleet.dispatch_model` | A complete serving export, one action per whole layer, then an assembly behind the complete-set barrier. |
| `python3 -m tools.tessera_fleet.dispatch_ladder` | The Tessera rate-band probe, one action per input shard. |
| `python3 -m tools.tessera_fleet.status` | Nothing. It reads a dispatch's recorded keys and collects the native Flux facts for their endings. |
| `python3 tools/render_identity.py` | Nothing. It renders on the local box; see [the cross-box render identity measurement](../measurements/cross-box-render-identity-2026-08-31.md). |

## One sealed workspace per dispatch

Every driver copies the bytes its actions run into a new Git workspace and
names that workspace as each manifest row's `cwd`; the committed snapshot binds the encoder, wrapper and plan bytes. Each
stage keeps its saved submissions and endings under `WORKSPACE/.flux-state/`,
and the immutable payloads under `WORKSPACE/source/`. A shard finished by an
earlier driver generation is not reused here and encodes once more.

## Full-model exports

`dispatch_model` accepts a complete source checkpoint, an explicit plan,
optional static input scales, an immutable encoder commit and a qualified
producer image. The worker derives whole-layer work units from Tessera's
`serving_parts` contract. A one-file checkpoint with 24 contiguous layers
produces 24 encode actions; the caller does not choose partition indices or
counts.

```bash
python3 -m tools.tessera_fleet.dispatch_model \
  --source /mnt/shared/models/LFM2.5-8B-A1B-BF16 \
  --plan /path/to/full-plan.json --input-scales /path/to/scales.safetensors \
  --encoder-checkout /path/to/tessera --encoder-revision FULL_COMMIT_SHA \
  --image REPOSITORY@sha256:DIGEST \
  --out /mnt/shared/models/lfm-mixed-export \
  --workspace /home/rob/tmp/lfm-export-dispatch \
  --prepare-host sparky --prepare-host sparklina
```

The driver targets the pool transport and the producer's whole-layer,
weights-only serving-part interface. It preserves fused groups, expert stacks,
passthrough ownership and the complete explicit plan. The default dense
fallback is E4M3/q1024, and explicit plan entries override it. NVFP4 plan
members require real static scales. No serving gate is bypassed.

**Preparation.** The driver archives the requested encoder commit and copies
the plan, scales and worker (`model_worker.py`, staged as `worker.py`) into
the workspace. One preparation action runs on each `--prepare-host`, pinned by
that host's placement tag. It verifies the exact image and source there and
writes the source identity to `OUT.prepare/HOST.json`. The hosts must agree.
Name every host an encode may land on; the driver does not discover hosts. The
host list is sealed into the export contract.

**Encoding.** Encode actions keep the ordinary class tags, and Flux admission
owns distribution. Source hashes, plan and scales, producer
revision, image and partition domain bind the export contract. Within a
worker, a private locked cache reuses a source hash only while the file's
device, inode, size, mtime and ctime and its expected digest are unchanged.
File identities are checked again after export. This is cooperative
filesystem identity, not hostile-writer immutability.

**Assembly.** Each encode writes a private attempt directory, hashes its
files, and publishes a completed part with a `pb-result.json` record (a
historical file name). The driver admits assembly only after every encode has a
native terminal success and every part record names this export's contract and index. The assembly action
re-hashes each part against that barrier inside its admitted slot, then runs
the producer's complete-set merge. An incomplete part is never exposed as a
loadable model.

Sources and output must be shared paths below `/mnt/shared`, and ancestor
directories must be traversable by Docker's daemon under NFS root squashing.
Containers bind source and sealed code read-only, keep the admitted job's
assigned affinity and resource scope, and use the exact locally installed image digest. The
worker neither pulls images nor schedules anything itself.

Resume a stopped dispatch with the same workspace:

```bash
python3 -m tools.tessera_fleet.dispatch_model \
  --workspace /home/rob/tmp/lfm-export-dispatch --resume
```

A failed or unfinished stage raises and leaves assembly blocked; the stage's
records under `.flux-state/` hold the saved job IDs to inspect. A passing unit test or a successful submission alone is not
evidence of a completed model export.

## Metadata plan and construction-census inputs

Use the model dispatcher to plan export work without a submission:

```bash
python3 -m tools.tessera_fleet.dispatch_model \
  --source /path/to/source --plan /path/to/plan.json \
  --cpus 2 --mem-gb 6 --dry-run
```

Run this command as an admitted Flux job with the project interpreter and CPU demand.
The command reads profile metadata, the source index and tensor headers.
It writes no files and submits no actions. No image or workspace is needed.

The result contains the following fields:
- `wired_architectures` comes from the profile specs' supported lane capabilities.
- `units` records the producer's actual tensor kinds, shapes and source shards.
- `partitions` covers every source tensor once under the producer's ownership rule.
- `construction_census.argv` names Tessera's existing construction-census tool.
- Census execution and runtime qualification remain `not_run`.

Run the generated census command from the Tessera repository inside the target
serving image as an admitted job. Supply its actual `--runtime-image` value.
Use the full source configuration and tokenizer inputs. Do not replace the
configuration with a guessed layer or expert count. This metadata plan does
not replace a census receipt, a native route check or serving qualification.

The plan writer emits the same metadata with `--export-setup-json PATH`.
Metadata can describe a known profile that the lane does not yet declare wired.
The `profile_wired` field reports that distinction; it grants no admission.
Full exports still use the real preparation, encode and assembly stages above.

The old fixed-count shard export driver is removed. Use `dispatch_model`
instead. Physical input shards do not define independent fused-module work.

## Historical ladder probe

`dispatch_ladder` retains its existing rate-band configuration. It copies
`tessera_ladder_probe.py` and encoder sources into its own workspace.
Its `--dry-run` prints rows without a submission. Its defaults do not define
an export partition count.

The status screen exits zero when every recorded action is complete.
It exits four for unfinished work, one for failure and three for absent records.

