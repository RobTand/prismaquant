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

## Uniform control

The lane spec now names the existing external Tessera accountant and
`shipcard_cli fill-control`. Declare the control family/rule and byte slack
before measurement, build the byte-matched arm, serve both under the same
teacher/calibration/runtime protocol, then preserve the accountant's block
and the control's own gold record. This command can ingest the resulting
`uniform_control.json` record without rewriting it. A4/A8 arm labels or
pairwise KL do not establish matched bytes; a missing measurement stays
missing. Neither this adapter nor the lane declaration creates an override.
