# Qwen3.8-27B readiness: the budget arithmetic

**Date:** 2026-08-14
**Status:** planning note. Qwen3.8-27B is not released yet; every number below is
computed from the **Qwen3.6-27B** probe as a structural proxy and must be
re-derived against the real checkpoint before it drives anything.

---

## 1. Measured shape of the proxy

From `prod-27b-nvfp4cb-5p5/artifacts/probe.pkl` via the Sensitivity Card:

- **26.047 B quantizable ("body") parameters** across **505 units**, 64 layers
- hidden size 5120

## 2. Body-only disk vs bit rate

`bytes = params * bpp / 8`, over quantizable params only (the reporting
convention: excludes `lm_head`, embeddings, profile-pinned Linears).

| bpp | body GB |
|---|---|
| 3.00 | 9.77 |
| 3.50 | 11.40 |
| 4.00 | 13.02 |
| 4.25 | 13.84 |
| 4.50 | 14.65 |
| 5.00 | 16.28 |
| **5.50** | **17.91** |
| 6.00 | 19.53 |

So the requested **~5.5 bpw "decent sized" quant lands at ~17.9 GB of body** —
comfortable on a 24 GB card, not on a 16 GB one.

## 3. The SUPER SMALL (16 GB card) target is EMBEDDING-BOUND

This is the load-bearing finding. Assume ~12 GB of weights to leave headroom for
KV cache and activations on a 16 GB card.

Embeddings and `lm_head` are **excluded from bpp** by the reporting convention
but they still occupy disk. At vocab ~151.9k x hidden 5120 that is ~0.78 B
parameters *per matrix*:

| embed + lm_head precision | their GB | GB left for body | required body bpp |
|---|---|---|---|
| BF16 | ~3.11 | 8.89 | **2.73** |
| FP8 | ~1.56 | 10.44 | **3.21** |
| 4-bit | ~0.78 | 11.22 | **3.45** |

**At 3.50 bpp the body alone is 11.40 GB — BF16 embeddings would not fit in the
remaining 0.6 GB.** The 12 GB variant is therefore not reachable by pushing body
bpp alone; the embedding/head precision decision comes first and is worth
0.7 bpp of body budget.

Two caveats to settle against the real checkpoint before acting:
- **Is `lm_head` tied to the embedding?** If tied, the table's embedding cost
  halves and 12 GB becomes much easier. Qwen3 dense models at this scale
  historically do **not** tie, but this must be read off the config, not assumed.
- Vocab size is assumed 151,936 (Qwen3 family). Re-measure.

## 4. What the format menu can supply

Sub-4 bpp is CB territory. The registry already carries the rungs:
`NVFP4_CB_K12` … `NVFP4_CB_K24` (group 256, A4). (The signed
`NVFP4_CB_S13`…`S16` rungs were deleted 2026-08-17 — no native kernel serves
their `n_sub = 1` layout.)
`FP8_CB_K28`…`K48` cover the 8-bit-activation CB family.

Two standing constraints apply when picking rungs:
- **Only lower-convex-hull rungs are selectable** at any budget under a
  cost of the form `s*g(K)`; some K values are structurally unreachable no
  matter the budget. Enumerate the hull before promising a specific K.
- **CBL (learned codebooks) is a NULL on Qwen dense.** DSv4's K43 result does
  **not** transfer; Qwen holdout measured ~1.00 across K28-K43. Plan the small
  variant on plain CB rungs, not on a CBL win.

## 5. Readiness checklist

- [ ] Read the real config: vocab, hidden, layers, **tied vs untied `lm_head`**.
- [ ] Register the model structure + serving profile if the arch string differs
      from Qwen3.6 (the plugin path is ~30-200 LoC: `model_profiles/specs/*.json`,
      a `ModelProfile`, a `serving_profile_specs/*.json`, and `pipeline.py`).
- [ ] Run the **forward-fidelity gate** before any probe or cost work. A
      teacher-forced language check is mandatory on a new architecture; vendor
      reference code is the spec.
- [ ] Probe **once**, with `--emit-marginals` on (default), producing a
      Sensitivity Card that serves both the 5.5 bpw and the SUPER SMALL builds.
      Probe calibration size at 27B scale in production has been nsamples=8, not
      the documented 32x1024 — read it off the probe provenance.
- [ ] Decide the embedding/head precision **first** (see §3).
- [ ] Enumerate the selectable CB hull for the chosen budget.
- [ ] Ship gates unchanged: exact full-vocab vLLM KL-vs-BF16 + direct WikiText
      PPL on the served artifact, p99 per-prompt NLL, and publication only via
      `tools/publish_artifact.py`.

## 6. Where the new contract helps

One probe serves both variants. The 5.5 bpw and ~3.4 bpp builds are two
**budgets** against the same Sensitivity Card and two different format menus —
no re-probe, and no rendered menu cache per menu, because weight error is
computed locally from `W` plus each format's own quantizer. The card was
deleted by #3457.
