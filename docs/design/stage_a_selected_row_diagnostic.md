# Stage A fresh selected-row diagnostic

Status: internal dev-only selector and source-owner API wiring, CPU controls
only. Original CUDA material and automatic capture remain refused. This is the
bounded implementation seam for #2010/#2008; neither parent is completed here.

`run_adjoint_capture_core` still receives the full calibration tensor. Its
optional `selected_row_diagnostic` document has exactly these fields:

| Field | Contract |
| --- | --- |
| `schema` | `prismaquant.stage_a.selected_row_diagnostic.v1` |
| `calibration_shape` | Positive `[rows, sequence_length]` of the original draw |
| `calibration_dtype` | Actual input tensor dtype spelling |
| `calibration_tensor_sha256` | SHA256 of the complete contiguous input tensor payload, in that dtype |
| `selected_global_row` | Exact integer index in the original draw |
| `probe_seed` | Exact nonnegative integer equal to `execution.seed_base` |
| `global_token_count` | Original `rows * sequence_length`, never selected-row tokens |
| `vocab_size` | Actual head vocabulary dimension |
| `through` | Exact nonnegative boundary below the model's tail |

The plan must explicitly use one probe and `probe_microbatch=1`. The selector
uses all output logits, temperature 1, Rademacher noise and the existing
`prismaquant.kl_fisher.global_row.v1` domain. Geometry, complete tensor identity,
seed and head width are checked before the core creates output. A cropped draw,
changed dtype, changed row, wrong N, mixed seed or out-of-chain boundary refuses.
Source authority is supplied separately by the reviewed enclosing action;
the geometry document does not authenticate publisher or source material.

The existing layer-major forward captures only the selected sequence. Existing
`tail_logits`, `fisher_probe_scalar`, `SharedStateCotangents` and
`render_free_layer_roll` generate a fresh tail and roll it through the requested
boundary. No recovery, seed, chain split or forward split can accompany this
mode. Full-draw calibration shape and digest remain in both run and boundary
identities. Local output batch 0 maps to the explicitly recorded global row;
Fisher receives that global index and the original full token count. The
diagnostic seals its requested boundary even when it is off the usual stride.
Teacher logits, probe and produced tail/rolled cotangents must be finite. The
receipt records fresh teacher-logit payload SHA256, shape/dtype and probe
geometry, using the existing bounded tensor digest helper (64 MiB chunks).
The enclosing memory envelope must account for that digest scratch and finite
checks alongside logits/autograd, source material, caches and shared states.

The output root gets `selected-row-diagnostic.json` before capture and
`selected-row-diagnostic-receipt.json` only after successful completion. The
receipt schema is distinct, carries `bandable: false` and reports historical
reference comparison as `not_performed`. It contains the retained forward
entries, ordinary checkpoint descriptors and shared-state snapshot bindings.
It writes no campaign adjoint receipt or resumable chain state. Ordinary
capture/resume and the band builder refuse its marker. A failed marked root
cannot be silently reused as a full campaign. CPU fixture parity with a full
draw is selector evidence, not historical-GLM primal or derivative acceptance.

The higher-level API accepts `source_authentication` only with the diagnostic.
It requires the existing `CaptureSourceAuthentication.qualified_original_material`
owner and matching model root, then calls that owner's current device gate
before backend/device allocation or profile/model reads. Profile selection
delegates to the existing owned-config/index seam and the identical owner is
passed to `build_streamed_causal_lm`. The caller retains owner lifetime and
close responsibility. Direct core CUDA diagnostic calls require the same owner
and device gate. CPU fixtures may exercise the core without a model publisher.

An explicitly selected original owner also selects the existing model identity
mechanism's authenticated descriptor intake. Configured legacy source identity/
digest caches refuse before device work, and no legacy proof cache is seeded.
Expected complete-file identity and owned config/index/roster agreement are
distinct from actual source deliveries: the diagnostic receipt retains
`original_source_material` separately with `automatic_capture_qualified: false`.
This identity intake grants no new CUDA or complete-provider authority.

`--selected-row-diagnostic` and `--selected-row-diagnostic-sha256` are paired
intake flags. The spec reader authenticates and decodes the same bound bytes,
rejecting duplicate keys and nonfinite constants. The standalone CLI constructs
no original owner and therefore refuses this operation. A reviewed admitted
enclosing API action is required; no ambient flag upgrades source authority.
No currently supported device gate admits original CUDA loads.

Before actual GLM execution, root must separately approve synthetic CUDA
lifetime qualification, an independently reviewed original a6 publisher/readset
declaration, complete tail/head/shared-state coverage, source/primal compatibility
controls and an overlap-derived finite PB envelope. The target diagnostic is
global row 0, seed 7000, full 512x512 draw (N=262144), boundary 6, BF16 and the
independently bound current original runtime, not a historical corrected
derivative image. Preserve reference/source differences and measure elementwise
and direct-contraction deltas; no tolerance or corrected
price table follows from this interface. The selector's fresh full forward may
also read the prefix: an L05-only four-shard readset cannot authorize it.

## Render-free original context and owned session (#2149)

The current-original entry is `run_original_diagnostic_capture`, not the
pricing wrapper's PREPARED/head/canonical-capture path. Its caller provides the
existing original material owner and independently bound full authority, final
plan and session-preparation receipt. The unchanged original CUDA predicate and
the shared full64/root/runtime/resource admission checks remain fail-closed.
No new provider, source cache, derivative fallback or automatic-capture override
is introduced. Legacy `run_adjoint_capture` joins remain unchanged.

The shared `source_generation` parser owns the closed BASE-plan, preparation,
execution and session-identity schemas. Preparation is explicitly
`prismaquant.original_diagnostic_preparation.v1`, scoped
`render_free_original_source_context`: complete expected source identity,
actual declared source execution, full calibration, resource binding, current
implementation digest and the shared loader's sealed head tensor roster. It
is not pricing PREPARED, priced anchors, a full source initialization witness
or a complete calibration capture. Its digest may enter the existing core's
`prepared_sha256` only in the strict selected-row original context domain.

The hash dependency is acyclic: normalized static authority (excluding only
session/root admission), independent preparation/execution/static source
manifest, BASE plan, existing artifact session, matched root admission, full
authority, then final PB plan/readset. The final enclosing manifest's full
authority/control digests never feed their own session identity. Static source
entries must still agree exactly with the active final manifest.

The explicit CPU metadata command is:

```text
python -m prismaquant.stage_a_selected_row_diagnostic --prepare-original-session \
  --base-plan PATH --base-plan-sha256 SHA \
  --static-authority PATH --static-authority-sha256 SHA \
  --data-manifest-sha256 SHA --allowed-tiers ram,ssd
```

Run it only as a root-approved published PB CPU action. It authenticates the
bound controls, checks the current implementation and decodes the complete
512x512 calibration. `StreamedBoundaryArtifacts.bind` creates its real published
session; `original-diagnostic-prep` completes metadata issuance while the
generation remains pending/running. Both owner and typed receipt explicitly
declare no source/CUDA computation, completed capture or source admission.
No manual UUID, persistent registry, GPU query or model load is needed.

The final plan's independent `original_session_preparation` binding is separate
from its unchanged five-key `original_source` block. The read-only issued-context
reader joins that receipt to the exact BASE/static tuple/session/policy and
delegates namespace checks to the existing artifact owner's
`inspect_published_session`: a genuine running generation, completed metadata
issuer and cold entry directory are required. Policy equality is strict even
in DEV mode. This inspection writes nothing and cannot take over another
capture, resume a failed run, or promote metadata preparation to GPU evidence.

The runtime entry uses the same source/profile/bootstrap/streaming cache and
prefetch owner, validates actual source identity/execution and starts the actual
source initialization audit. The core rebinds that same issued published session
with its exact nonrecursive identity and actual memory callback; it does not
mint another generation or adopt old teacher/cotangent/checkpoint data. The
selected-row marker, non-bandable receipt and one full-N Fisher application
remain unchanged. Metadata preparation and CPU contract tests do not qualify
original source transport, numerical measurements, pricing, wire/native
execution or serving; actual full64, shared SDK4, matched source/root admission,
finite envelope and exact GPU GO are still separate prerequisites.

