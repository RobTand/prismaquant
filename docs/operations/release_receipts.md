# Ingest release receipts

Use the existing shipcard verifier, not a window wrapper's `passed` flag.
This CPU-only command never runs a serve, measures, calls the Hub, copies a
serving manifest or changes the artifact's identity. Run it through the
published PrismaBuild client with honest CPU/memory demand and bounded native
threads. Stage the final LICENSE before creating the authoritative card and
collecting evidence.

## Inputs

1. The artifact's canonical `shipcard.json`, with its exporter-derived build
   anchor and current identity/stat attestation.
2. Optional `--records-dir`: a directory of existing producer-authored slot
   records, one `<slot>.json` each (for example `gold.ppl.json`). Every file
   must name an opened slot and carry the card's exact `model_sha`. Unknown
   filenames, duplicate/nonfinite JSON and mismatched identities refuse.
   Do not hand-author passed flags or change a measurement's identity.
3. Optional `--window`: U4's `prismaquant.pact_u4.arm/1` receipt for that exact
   artifact. The explicit `route_trace_full_config.source_phase` selects its
   matching phase verdict and two distinct raw rank paths. Both raw traces
   are replayed against the artifact's **full** config and installed pinned
   runtime contract. The carried AGREE flag and body-only diagnostic are
   never authority. Both rank paths must be absolute, as U4 emits them.

Example worker command, after the coordinator supplies the finished window
and producer records:

```sh
python -m prismaquant.release_receipts /artifact/shipcard.json \
  --window /window/u4-A16.json --records-dir /window/shipcard-records
```

The default is read-only. Add `--apply` to fill replayable records. All proposed
records are preflighted together with the existing verifier before any write;
global build/identity failures prevent application. Missing **other** slots
can remain while useful evidence is filled. The existing writer rechecks the
artifact's stat fence at each fill. Application is not a multi-slot filesystem
transaction: if an external mutation/IO error interrupts it, inspect the card
and rerun; already-written records are not rolled back.

Exit0 means the resulting candidate card verifies and no supplied output is
unsupported. Exit1 means refusals remain; exit2 means malformed input or IO
failure. `prepared_slots` means replay was attempted, not that a slot passed.
`applied_slots` records completed writes on a normal return. Final `problems`
comes directly from unchanged `shipcard.verify`. Neither success nor a filled
card authorizes upload; the publisher's dry-run and the operator's explicit go
remain mandatory.

## New TR3 Producer Records

The existing `experiments/measure_glm_tr3_vllm.py` accepts an optional
`--gold-record-out /new-window/records/gold.kl.json` on a complete new run.
The output must not already exist or alias the full result or qualification.
It cannot be used with `--qualify-hook`; the ordinary qualified full-panel
or `--qualify-then-score` path still owns measurement and its identity fences.

The producer records all 25 windows / 51,175 positions, full-vocabulary FP64
method, exact panel/teacher calibration digest, clean producer sources,
observed disabled speculation, canonical shipcard artifact identity and the
actual in-process serving manifest with a live engine descendant. The existing
authenticated checkpoint cache still owns weight-content evidence. No raw
retained result is read or upgraded, and no canonical serving-manifest file is
created in the artifact. Ingest the new producer record with the existing
`--records-dir` interface; missing independent slots remain refusals.

The existing `shipcard_cli fill-control` also accepts this exact producer
record as `--control-record`, preserving its metrics, method, source, tool and
no-spec observations. It compares the record's canonical model identity with
the actual control artifact before using the existing control constructor.
Unknown producer schemas and malformed/drifted gold records refuse without
writing. The existing flat gold-result case keeps its original behavior.

## Outputs the Retained U4 Windows Do Not Supply

The retained TR3 result has a real nested serve manifest and full-vocabulary KL vectors,
but is not a gold shipcard record. It does not serialize the shipcard's
artifact SHA or its observed no-spec result. This adapter refuses to invent
those fields, retroactively hash-bind it, or install its manifest as canonical.
Supply an identity-bound record from the measurement producer and its actual
canonical manifest; do not derive PPL/NLL from KL.

The window supplies no graph qualification, PPL/NLL, ship-gate ledger, complete
artifact census or matched-byte control block. Its successful eager commands
are not native-export slot records. The newer 2c phase can supply full-config
rank traces; absence, rank omissions, old histogram-only traces and mismatches
remain refusals. Never substitute body-only traces for priced MTP modules.

## Graph producer records

Graph evidence uses the same `--records-dir` interface. The original equality
measurement producer supplies `native_export.graph.json` alongside its raw
equality receipt. A later smoke must not attach an unrelated equality receipt
because its model has the same `config.json`.

The record binds the receipt's raw-byte digest, actual artifact identity before
and after measurement, producer tool and commit, graph arm and action evidence,
generation output, observed speculation, supplied compilation JSON, resolved
compilation and operator settings, and every rank's image and transport.
An NCCL environment request is not an observed transport.

`serve_scope` names image, model-config digest, Tessera-source digest,
compilation configuration, speculative-token count, maximum model length,
maximum concurrency, tensor-parallel degree, and fabric. The verifier hashes
the real artifact's `config.json`, then calls the lane's shared population,
numerical-equality, and capture/replay verifier. Only source provenance uses the
development seal policy. Byte integrity and the other scope comparisons always
refuse on disagreement.

The currently pinned Tessera verifier lacks the required shared population API.
It therefore cannot qualify a graph slot through this path. A compatible
metadata fixture establishes consumer behavior only. The runtime pin and
independent allocation, export, serving, and publication gates still apply.
Missing original producer records or real release evidence remain refusals.

The compressed-tensors smoke accepts `--graph-receipt`, an explicit
`--compilation-config` JSON object, `--tensor-parallel-size`, and
`--max-num-seqs`. It requires the artifact's existing canonical shipcard so
the current content and stat fences can bind the observed artifact without
another full-weight read. It records its nine-field `serve_scope` and a
separate `graph_smoke_observation` with actual resolved settings, per-rank
observations, artifact identities, and generated text. These later smoke
observations cannot fill the original `graph_observation` producer linkage.
Above one rank, the current native worker observer refuses because it cannot
observe the actual NCCL transport. Changing `NCCL_IB_DISABLE` cannot supply it.

To replay the synthetic numerical tests with separately reviewed metadata,
seal a `git archive` containing that external Tessera commit's `src/tessera`
and `pyproject.toml`. Run `tools/check_graph_receipt_metadata.py` through
PrismaBuild with `--archive`, `--sha256`, and `--commit`, declaring the archive
with `pbrun --requires-files`. The tool verifies the archive bytes, Git commit,
import origin, and shared population API before running the numerical tests.
This test dependency does not adopt that commit as the production runtime pin.

## Uniform control

The lane spec now names the existing external Tessera accountant and
`shipcard_cli fill-control`. Declare the control family/rule and byte slack
before measurement, build the byte-matched arm, serve both under the same
teacher/calibration/runtime protocol, then preserve the accountant's block
and the control's own gold record. This command can ingest the resulting
`uniform_control.json` record without rewriting it. A4/A8 arm labels or
pairwise KL do not establish matched bytes; a missing measurement stays
missing. Neither this adapter nor the lane declaration creates an override.

When the candidate carries the new fidelity, calibration-contract digest or
teacher evidence, the existing uniform comparison requires the control to
carry the exact same corresponding fields. Missing, different or nonfinite
context refuses; structured comparison distinguishes booleans from numbers.
Legacy flat candidates without these optional fields keep their original
comparison. `corpus_sha256` is never relabeled as a combined contract digest.
