# Quantum control-metadata generation seam — 2026-09-21

Branch `fix/quantum-metadata-generation-20260921` (PQ issue #884), base
`8c204ea24a05da1c00dea9bcb978d07bc8bfff5e`. Write ownership:
`prismaquant/joint_layer_quanta.py`, `tools/regenerate_joint_quanta.py`,
`tests/test_quantum_metadata_generation_884.py`, this document and the
`docs/ARCHITECTURE.md` stamp. No dispatcher/controller changes; no PB-side
changes; the Claude-owned bridge files (`tools/dispatch_joint_quanta.py`,
`prismaquant/joint_cost_stage_a.py`, `cost_streaming.py`,
`perturbed_x_cache.py`, `stage_a_produced_output.py`) are untouched.

## Problem

The post-Stage-A quantum binder must publish NEW immutable control files
(records, slice manifests, bound readsets) without moving or repeating
existing scientific artifacts. Two existing fences made that impossible at
the original data root:

1. **First-writer immutability** (`regenerate_joint_quanta._publish` →
   `cost_stage_checkpoint.publish_new_bytes`): beside the pending
   pre-#852 generation, whose slice manifests carry a zero-byte head phase
   row the current producer no longer seals, any same-root regeneration
   publishes *different bytes at the existing manifest paths* and refuses.
   This refusal is correct and stays (proved behaviorally at base
   `8c204ea`: `test_default_layout_publication_refuses_beside_immutable_prior`).
2. **All-or-one-prefix root move** (`_check_authorized_diff`): re-cutting
   under a new `--output-root` drags `output_space` and
   `adjoint.boundary_artifacts` off the original RUN — the records would
   name artifact roots nothing writes to. Fail-closed, also correct.

The missing piece is a seam that separates **where control metadata lives**
from **what the data roots are**.

## Design

One new optional input, one derivation, no new cache or scheduler:

- `layer_quanta(..., metadata_root=None)` — the control root. When absent,
  it is exactly `{output_root}/layer-quanta` and every sealed byte is
  identical to the previous layout (asserted by
  `test_producer_metadata_root_equal_to_default_is_identity`). When given,
  it must be a canonical absolute path (`_canonical_control_root`: absolute,
  normalized, no `..` — the same discipline the bound-readset binders
  enforce per path).
- Data never moves: `output_space` stays `{output_root}/layer-quanta/{qid}`
  and `adjoint.boundary_artifacts` stays
  `{output_root}/layer-quanta/adjoint`, both derived from the data root
  alone. The consumer gate (`joint_cost_quantum.verify_quantum_identity`,
  the `output_space` pin) and stage A's receipt location are untouched.
- Control placement derives from the control root through the same
  relative layout in both branches: `manifests/`, `records/`, and — via the
  single `bound_readset_directory` helper shared by both binders and both
  emitters — `adjoint/bound-readsets/`. One spelling, never two: a bound
  path cannot disagree with the record that names it.
- The slice argv relocates exactly the `--quantum` record path;
  `--output-root` and the entry point stay the data-root bindings. The
  record identity moves with the placement (it is sealed over
  `read_set.manifest_path`), which is the point: a new generation is a new
  identity, never an edit.
- Derivation provenance: `control_metadata_root` appears in the derivation
  block ONLY when an explicit root was given — default-layout derivation
  bytes are unchanged. Nothing validates an exhaustive key set of the
  derivation document (the joiner cites derivation v2 for windows only);
  the field is provenance, not identity ceremony.

CLI (`tools/regenerate_joint_quanta.py`):

- `--metadata-root PATH`: passes the control root to the producer.
  `--records-out` defaults to `{metadata-root}/records` and is also
  optional under `--check-only`; write mode with neither still refuses.
  Publication order is unchanged: manifests first, then bound readsets,
  then hash verification, then records/index/derivation last — crash
  recovery never exposes a record naming missing or different bytes.
  First-writer semantics unchanged: same-bytes replay idempotent,
  differing bytes refuse, `--expect-existing` Gate 1a ALWAYS reproduces
  the default layout at the original root (the seam never weakens it).
- `--expect-existing` + `--metadata-root` is an *authorized relocation
  over an exactly reproduced source generation*: Gate 1a runs unchanged,
  then `_check_authorized_metadata_diff` requires `output_space` and
  `adjoint.boundary_artifacts` byte-equal, the manifest path exactly the
  producer-named metadata path, the manifest digest exactly the hash of
  the newly produced slice wire (the prior slice's own wire is
  digest-verified against the prior record first), and the new identity
  recompute. Combining it with a data-root move refuses by name.
- `--compare-existing DIR`: the mechanical scientific-binding comparison
  for a prior generation that need NOT be reproducible (the campaign's
  pre-#852 pending set). Read-only; separate from Gate 1 (combining the
  two refuses by name).

## What `--compare-existing` certifies — and refuses

The prior records are first validated through the existing
`check_quantum_for_campaign` (stale identity refuses; duplicate quantum
ids refuse). A **pre-bound prior generation** — records carrying
`boundary_readset`/`executable_readset` blocks, which bind actual reads,
not mere filenames — is refused outright; new bindings made in the same
run through the existing binders from the trusted receipt remain allowed.

Per record, identical: campaign block (plan/prepared/parent digests,
scope, roster), `chunks`, `windows`, `read_set` extents, `output_space`,
adjoint minus receipt. Per slice manifest, every top-level field except
`annotations` compares exactly (an unknown scientific field cannot
silently vanish), and `annotations` compares exactly except:

- `phases`: at most the one known pre-#852 zero-byte head row
  (`{"name": "head", "bytes": 0, "cumulative_bytes": 0}`, exact shape)
  may be absent from the new generation; the remaining ordered nonzero
  rows — names, byte counts, cumulative bounds — compare exactly.
- `argv`: only the `--quantum` record-path value may relocate; the
  interpreter, entry point, `--output-root` and every other binding
  compare exactly.

The prior slice's raw wire bytes are digest-verified against the prior
record's sealed `manifest_sha256` before any semantic comparison. The
summary line counts quanta, `windows_total`, placement moves, record-path
relocations, zero-byte head rows dropped and receipts newly bound; any
drift outside the admitted set exits 3 with a named reason.

## What did NOT change

- The strict readset/binding gates: a source-only slice manifest still
  does not qualify a quantum for produced-output strict reads;
  `ExecutableBindingUnsupported` stands (tested);
  render-prerequisite manifests remain sequencing-only with
  `binding: None`.
- The dispatcher, the joiner, the Stage A core, the caches. The joiner's
  scan (`joint_quanta_join._scan_receipts`) reads records from
  `{input_root}/layer-quanta/records` (or flat) and takes every path it
  follows from each record's sealed absolute `output_space`; a generation
  named `…/<gen>/layer-quanta` is therefore joinable via
  `--input-root …/<gen>` with NO joiner change, and cost/status paths
  still resolve under the original RUN. (Verified against
  `joint_quanta_join.py:151-170,467-500,656-699`.)
- No live publication, no GPU work, no model hashing: the campaign
  evidence is `--check-only`.

## Campaign commands (reviewable, root executes)

The exact original inputs (PANEL/RUN per the takeover handover; plan
`0b2cc006…`, prepared `962207a3…`, parent `71fd8f56…` — the ordered
manifest `43f40d18…` is NOT the parent):

```bash
# A. current-producer reproduction at the data root + scientific
#    comparison against the untouched pending generation (writes nothing):
tools/regenerate_joint_quanta.py \
  --plan  $PANEL/$RUN.plan.json --plan-sha256 0b2cc006…\
  --prepared $PANEL/$RUN/prepare/prepared.json --prepared-sha256 962207a3…\
  --parent-manifest $PANEL/stage-a-recovery-20260920/parent.run.json.gz \
  --parent-manifest-sha256 71fd8f56…\
  --derivation $PANEL/stage-a-recovery-20260920/records/derivation.json \
  --partition  $PANEL/layer-quanta/window-partition.json \
  --compare-existing $PANEL/stage-a-recovery-20260920/records \
  --check-only

# B. same, naming the NEW metadata generation namespace (still writes
#    nothing; proves the placement derivation):
#    ... + --metadata-root <reviewed-generation-dir>/layer-quanta

# C. when Stage A completes and a real receipt exists (the publication
#    root runs; NOT run by this task):
tools/regenerate_joint_quanta.py \
  --plan … --plan-sha256 … --prepared … --prepared-sha256 … \
  --parent-manifest … --parent-manifest-sha256 … \
  --derivation … --partition … \
  --metadata-root $PANEL/stage-a-recovery-20260920/metadata-generations/<gen>/layer-quanta \
  --adjoint-receipt $PANEL/$RUN/layer-quanta/adjoint/adjoint-capture.json \
  [--boundary-readsets]
```

A and B exit 0 with the mechanical comparison summary (45/45 quanta, 360
windows, control placement/zero-head/receipt counts); C publishes the new
immutable generation with first-writer semantics while the pending
generation and every data artifact stay untouched.

## Gates

`tests/test_quantum_metadata_generation_884.py` (tiny CPU fixtures built
only by the real producer, the real CLI and the real receipt writers):
the behavioral regression at base, producer seam and identity anchor,
CLI roundtrip/replay/refusals, relocation gate, comparison certification
and its refusal matrix (identity, duplicates, pre-bound priors, unknown
manifest fields, phase order/bounds, argv bindings, unverified bytes,
membership, record drift), receipt + boundary/executable readset binding
under the namespace with the consumer identity gate, and the dispatcher's
unchanged refusal. Adjacent suites
(`test_joint_layer_quanta.py`, `test_quantum_launch_contract_838.py`,
`test_quantum_boundary_readset.py`, `test_quantum_executable_readset.py`)
run unchanged against the seam-free default layout.
