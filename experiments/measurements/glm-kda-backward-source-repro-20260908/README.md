# GLM KDA backward source reproduction — 2026-09-08

The pinned Torch fallback computes overflowing, unused upper-triangular
exponentials before masking them. A complete finite forward can therefore have
NaN input derivatives. The proposed correction masks only strictly upper
triangular exponent arguments before `exp`. This directory is CPU numerical
evidence and a proposed integration contract; no production runtime, profile,
checkpoint, active capture, serving image, or native qualification changed.

## Evidence

The preceding original native diagnostic is retained in
`../glm-joint-original-graph-implementation-20260908/native-negative-audit-diagnostic-01.json`.
PB action `61a7ddd21154a13919849d27f3e6b72b3116807cfb886104545e5770d5df8251`
failed with return code 1 on Sparklina, with scope retirement complete and zero
OOM. Its primary/replay outputs have identical bytes and every observed forward
tensor is finite. The attention output cotangent is finite, but forget-gate
cotangents include 1,514,496 NaNs, q/k cotangents include NaNs, and v cotangents
remain finite. All 8,388,608 input cotangent elements are NaN. This localizes the
native failure to the attention backward region; it does not directly observe
the internal exponent's native values. The raw trace and 17 Netdata samples
from each box are SHA-bound by that audit. No timing improvement is claimed.

`pinned-functions.json` contains the exact `l2norm` and
`chunk_kimi_delta_attention` bodies from the native modeling source SHA256
`2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b`,
lines 416–424 and 482–578. That Transformers source is licensed under Apache
2.0, copyright the HuggingFace team. Decorators are omitted explicitly for this
CPU source reproduction. Native runs must continue through their actual
decorated entrypoints; this extraction is not a replacement kernel.

The complete CPU action
`0bd9ed1ee3a44153d9ec53c888f098037c90c52e11025acf5d9a3dc8b93da88e`
passed with return code 0 on dl380g10, Torch `2.10.0+cpu`, one native thread,
one reserved CPU, 2 GiB reserved memory, no GPU, no skips. The CAS payload,
canonical result claim, source snapshot and closure-only source diff were
independently verified in `cpu-repro-audit.json`. Exact results are in
`cpu-repro-result.json` (SHA256
`166fa2e6f1403a9292a59aa4ec4d76e978498a740931a844cd1db2ab5856faf5`).

For a 64-element gate vector equal to −5, cumulative differences range from
−315 to +315. The original exponent has 1,081 infinities among 4,096 entries;
the causally masked result has no nonfinite entries. Nevertheless, all 64
gate derivatives are NaN. Premasking produces identical result bytes, no
infinities and 64 finite derivatives. An independent analytical causal sum
agrees with maximum absolute derivative error `3.872e-10`.

Six complete-function cases cover sequence lengths 64 and 65 (padding and
multiple chunks), FP32 and BF16 inputs, gates −5 and −0.1, q/k normalization
on and off, and initial/final state on and off. All six original/proposed
outputs match byte-for-byte. All four strong-gate original cases have
nonfinite q/k/g derivatives; the corrected cases are fully finite. The two
finite-gate controls retain every input derivative byte. FP32 corrected
derivatives pass the explicit float64 original-function oracle tolerance
`rtol=3e-4, atol=2e-7`; BF16 oracle errors are recorded, not held to an FP32
tolerance (largest relative L2 error about 0.00206). The float64 oracle changes
the function's intermediate precision explicitly and is not a native runtime.

The first proposal masked the diagonal too. It passed the strong-gate oracle
checks but failed control derivative byte equality because it removed the
existing diagonal cancellation arithmetic. PB failed action
`6219d346dfd3aee00c0bc43185beb018513961aa505151a4d2787849fd0662f6`
and its cleanup/source audit are retained in `cpu-diagonal-mask-negative.json`.
The revised proposal preserves the diagonal. This negative result prevented an
unnecessary change to finite control arithmetic.

Reproduction command, submitted through PB with the resource bounds above:

```sh
/home/rob/venvs/pq-cpu312/bin/python -m experiments.glm_kda_backward_source_repro \
  --out /mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/cpu-kda-source-repro-02/result.json
```

## Proposed correction and compatibility contract — not implemented

Only the exponent expression at original line 527 would change:

```python
# Original
decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
# Proposed
decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(
    mask.triu(diagonal=1).unsqueeze(-1), 0
).exp().float()
```

Both consumers of `decay_mask` remove strictly upper-triangular values after
their per-feature reduction (original lines 528 and 557). For the admitted
finite, nonpositive gate domain, the causal exponent values and diagonal
arithmetic remain unchanged. Upper-triangular exponent values become one
before being discarded. Their mathematical derivative is zero because those
values cannot affect either consumer. This prevents an invalid intermediate;
it neither clamps a cotangent nor replaces the attention algorithm.

The integration should be explicit and opt-in, with these existing owners:

1. `Glm5NextProfile` declares a closed source derivative contract, with the
   original modeling-file SHA, exact function/body SHA, reviewed one-hunk
   transform, corrected SHA, dispatch requirements and qualification receipt.
   A shared default profile method returns no correction. It is not a global
   format/default change or a callback selected by an agent at runtime.
2. A future isolated derivative runtime uses a derived, content-pinned image
   with only that reviewed Transformers expression changed before import.
   Preserve the original decorators and dispatch. Admission must inspect the
   resolved callable and refuse a different hub/FLA implementation; an `eager`
   config selector alone does not prove KDA dispatch. No `__wrapped__` call,
   toy kernel, or runtime monkey patch can satisfy the native gate. The image
   and actual modeling source are distinct from the original native baseline.
3. Extend `joint_aura.source_execution_identity` through the profile contract
   to bind this derivative identity and actual dispatch, alongside its current
   attention/expert selectors. Bind it before `tessera_joint_aura` preparation
   and before `aura_cost` constructs `joint_probe_identity`; existing exact
   preparation/resume comparisons then refuse old derivative statistics and
   mixed runs. Recheck the execution identity at the existing per-layer gate.
   New statistics use a new producer/probe identity and destination. Do not
   repurpose the historical empty-lease source transition or relabel old costs.
4. Keep the canonical capture as the original, immutable producer artifact.
   `tessera_calibration_cache.capture_identity` and `prefetch_capture` retain
   their exact source/runtime/initialization/completeness checks. A separate
   consumer compatibility receipt should bind its original manifest digest,
   original runtime/source, corrected derivative identity, nonpositive gate
   domain, and forward-equivalence evidence. This receipt permits consumption
   only after the capture completes and validates normally; it does not edit
   its identity or declare an in-progress capture complete. A consumer using
   the corrected runtime is not allowed to pretend that runtime produced the
   capture. No new activation or weight cache is needed.
5. The next native gate, only after review and separate authorization, first
   compares corrected layer-0 original-shape forward bytes with the recorded
   original baseline, then checks all input/branch cotangents. A subsequent
   original-shape 72-backward gate must retain baseline/nonfinal/final equality,
   source/metadata ownership, exact routing and cache behavior, finite values,
   bounds, profiles and both-box Netdata. Corrected-runtime results must be
   labelled as such. CPU evidence does not establish native numerical success,
   full-model forward equivalence, memory fit, true Fisher, quality or speed.

The expression's original-shape pairwise tensor is
`[1,64,8,64,64,128]`, 268,435,456 FP32 elements (1 GiB); the new broadcast
mask is only 64×64 bool. No memory-neutrality claim follows from that geometry:
the added operation's forward/backward transient must be measured within the
existing 16 GiB workspace allowance and existing physical/GPU caps. The active
capture process and its files stay under their original execution throughout.
