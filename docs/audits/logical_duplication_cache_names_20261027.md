# PRI-226 C3-02 cache-name consolidation

Status: source complete; independent review and repository-qualified merge pending.
Base main is `05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
Tested source is `5f44bb63a0606843ff5de8156fb2c85e3cb98c1b`.
The one pre-review rebase found that origin/main was unchanged.
Only this audit document follows the tested source.

The cache and native export had independent copies of one ordered name rule.
`name_projection.production_cache_name_candidates` now owns it. Both lookup
loops call the shared owner, and both private helpers are deleted. The helper
also reuses the existing leaf stripper. It yields the original name, optional
trailing `.weight` removal, and optional `model.language_model.` to `model.`
projection of the original. It does not combine the rewrites.

## Contract and callers

At the base, `ProductionWeightCache._name_candidates` at
`prismaquant/production_weight_cache.py:1063-1069` returned a list. Its only
tracked explicit caller was `resolve_key` at :1091, which iterates it.
Export's `_production_cache_name_candidates` at
`prismaquant/export_native_compressed.py:249-255` returned a tuple; its only
tracked explicit caller was `_production_cache_lookup_key` at :267.
The different containers had no observable effect in those consumers.
The new owner returns a tuple. No wrappers or alternate private APIs remain.

Export's lookup is used by expected-key validation at base :298, prefetch at
:1219 and packing at :1318. Those call regions were read directly. PWC's public
resolver also has broader coverage, replay, budget and render consumers; this
slice does not claim all their bodies were re-audited. Explicit call searches
cover tracked production, tests, tools and experiments. External callers or
monkeypatches of removed private helpers were not established.

Each lookup retains its format grammar. PWC tries raw, uppercase, registry
canonical and FP8/MXFP8 aliases. Export's weights-only fallback retains its
uppercase and canonical-first MXFP8 policy. Export first accepts an available
resolver's non-None result, retains fallback on None and propagates failures.
The candidate helper performs no tensor, file or device work. Residency,
prefetch, packed bytes and numerical code stay with their existing owners.

The new tests retain explicit key expectations before any production change.
They cover original/leaf/prefix precedence, the noncomposed miss, empty names,
case, whitespace, format-policy differences, name-before-format precedence,
resolver override/None/error, absent or empty weights and non-string refusal.
An injected shared name rule proves that both real lookup paths consume it.
This is fixture evidence for key selection, not a rendered-artifact or served
qualification.

## Native fleet evidence

All tests and compile checks ran through the published PrismaBuild client on
ordinary CPU admission. `cache_names_jobs.py red|green` in the evidence bundle
records the complete argv, resources, pinned interpreter and 300-second limit.
No host-draining measurement was submitted.

| Check | Action key | Observed result |
| --- | --- | --- |
| Unchanged production plus retained fixtures | `b4c1773452d1d16edcb0edb35acda27643e467fad2a30a23ac81a9b0e6252ccf` | 26 pass; only shared-owner injection fails because the helper is absent |
| Extracted owner, nearby contracts and compile | `53c3f1f3189bd49b0376445b7c305de710bc05b68233c699e3cdf40a2747a386` | 87 pass, zero failures/skips; four touched Python files compile |

The green cohort includes name-projection, architecture, docs staleness and
duplication-baseline tests, plus five existing cache coverage/residency checks.
Fourteen upstream Torch/Python 3.14 warnings remain. The recorded environment
is Python 3.14.4, Torch 2.11.0+cpu and Transformers 5.17.0.
The shrink-only baseline and scanner were not edited; their current guards pass.
No new performance, KL/bpp, GPU or serving result follows.

`verify_cache_names.py` directly rechecks native terminal exits, both log
hashes, every tracked source blob in the sealed snapshots and successful CAS
claim/payload checks. Red source is `b8c72f601499d7372e2b4e7f3ff0b7cce20c91c8`,
with snapshot `ae063227212534b9da58618d0e2a27f732a12557`.
Green snapshot is `c57d6203c9c105168b4794f5c7cffc9ddeabc4b4`.
Each contains only its recorded PB closure metadata beyond the source tree.
The claim checker does not validate worker attestation against the full action
manifest. The initial receipt read reported absence; a later read passed
without test re-execution. Both observations are retained; no cause is claimed.

## Scope and remaining acceptance

The existing dependency-light name-projection module is preferred over a new
cache utility or a heavyweight PWC import in export's weights-only fallback.
The design note records both alternatives and the caller trace. All 49 open
PR file rosters were checked before editing; none touched these three source
files. No existing approved head was edited.

This resolves the source migration for cache-census-3 C3-02 after review and
merge. Other C3 families, earlier census gaps, #1301/#1303 owner outputs and
#1295/#2768 obligations remain open. Archive and deprecated GGUF paths remain
preserved exclusions. Existing scientific/default/pin/ship gates remain in
force. An independent current-main residual census still precedes program
closure; this source unit does not close #3061 or PRI-226.

Build the Lever kept a rerunnable native evidence verifier. Prove It Works
required terminal/source/CAS evidence. Laziness removed both copies without
adding a wrapper or cache mechanism. The decision trail was reviewed inline.
