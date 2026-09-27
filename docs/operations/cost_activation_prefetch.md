# Batched-cost activation prefetch

`measure_batched_gpu` uses the existing `io_engine` for activation read-ahead.
`PRISMAQUANT_COST_PREFETCH_ACT` remains enabled by default; set it to `0` to use
the existing synchronous path. There is no new executor, cache, worker-count
knob, or fixed byte allowance.

## Budget and lifetime

For each shape group, the caller stats the files selected for each actual
chunk. A verified activation index prices filenames relative to its existing
pinned directory descriptor, matching the loader even if the directory's
pathname is replaced.

For measured chunk byte sizes `s[0] ... s[n-1]`, the activation-stream ceiling
is `max(s[i] + s[i+1])`, treating the absent final successor as zero. This
covers an active chunk and its prefetched successor. It is a byte ceiling,
not a fixed depth: the engine chooses concurrency and may fit additional
smaller chunks within the same measured allowance. The largest measured
chunk supplies the stream's buffer allowance; range-reader entries do not
allocate an additional serialized buffer through the engine.

A chunk is a range-reader entry that calls the existing activation loader.
The engine measures distinct retained tensor storages, including full
storage behind views, and refuses delivery if they exceed the priced charge.
The current chunk remains charged until its input/row-index references are
dropped. Stream closure is guaranteed on success, load failure, and numerical
measurement failure. GPU operands/workspace and the process's other allocations
remain part of the caller's aggregate reservation, not this activation budget.

## Unchanged boundaries

**The `weights_only=False` no-verification arm of `ActivationIndex.load_blob`
is unchanged and outside this migration's scope.** So is that method's
verified descriptor/stat fencing and committed logical-tensor identity replay.
The scheduler neither invents a serialized-file digest nor promotes an
unverified input to verified status. The retained-storage check happens after
decoding; it is not a sandbox or a bound on arbitrary Python allocations during
legacy pickle deserialization. Process admission remains necessary.

No format arithmetic, calibration rows, source identities, cost-row schema,
residency policy, or public measurement arguments change. The alias-aware
architecture freeze detects imported/assigned constructors without growing
its shrink-only allowlist. This legacy batched-cost path is not G2's fresh
Tessera pricing path; this migration establishes no G2 or production GPU
speedup. CPU regressions compare prefetch-on/off rows exactly and replay the
existing producer's verified-activation contract.
