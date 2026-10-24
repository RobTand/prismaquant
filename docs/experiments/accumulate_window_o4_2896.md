# #2896 O4 — portable current-main arm-A reproduction recipe

This restores the #1406 experiment's preparation-to-Arm-A route, not a new
experiment or a measurement receipt. The restored instrument consumes the exact
final prepared plan. CPU checks cannot establish numerical equivalence, speed,
residency, power, or substitution acceptance. The GPU commands below are a recipe
for separately authorized execution; remaining C/D measurements are excluded.

## Instrument interface

```bash
python -m experiments.accumulate_window_research \
  --plan "$FINAL_PLAN" --prepared "$PREPARED" \
  --prepared-sha256 "$PREPARED_SHA" --reference-cost "$REFERENCE_COST" \
  --out "$RESEARCH_OUT" --arms A --window 1
```

It must consume the final plan/prepared pair unchanged: no temporary relocated
plan, backend substitution after prepare, output-root rewrite, or implicit
C/D defaults. A missing instrument or incompatible API is a blocker, not a
reason to weaken the recipe.

## Inputs and deployment configuration

Run from the repository root in the project's known-good environment. Supply
locations through the existing sourced deployment environment configuration,
visible at the **same paths inside the configured execution containers**.
Use the existing campaign spec mechanisms
for image, code mount, PB executable, demand, resource/progress limits and
container environment; do not inherit an old fleet's paths or placement.
Keep model and original inputs immutable. Record checkout commit, installed
Tessera seal, image identity, runtime/arithmetic settings, and launch receipts.

The shell fragments form a staged Bash recipe (`set -euo pipefail`); commands
that read/write large artifacts or use CUDA belong in the approved deployment,
not an arbitrary host Python. The input JSON files below are reviewed,
**fully populated current-schema configurations**, not minimal new schemas:

| Variable | Configured input |
| --- | --- |
| `MODEL` | Original complete model checkpoint |
| `REFERENCE_COST` | Original published joint-cost pickle, not a new merged anchor cost |
| `CALIBRATION_INPUT` | Original exact calibration safetensors |
| `O4_ROOT` | New, nonexistent output directory on the configured storage |
| `BODY_PLAN` | Bound body plan for the original draw and authenticated canonical body capture |
| `BODY_PREPARED` | Genuine completed body prepare with last-layer `source_execution` expert dispatch |
| `BOUNDARIES` | Genuine exact body boundary manifest, including every original row entering the last body layer |
| `MTP_OFFLOAD_FOLDER` | Writable scratch directory for final-hidden source offload, visible inside the configured execution container |
| `CAMPAIGN_SPEC` | Current dispatch spec for the new MTP census/capture, same original R1024 candidate/recipe menu |
| `JOINT_TEMPLATE` | Complete current joint-plan configuration preserving original MTP window policy, resources, source prefetch and objective |

Do not synthesize the body prerequisites from per-Linear X/H, copy one row's
metadata 512 times, use a prefix-only qualification as a full capture, or use
MTP prepared data in place of the body prepared dispatch. If these prerequisites
are absent or incompatible, obtain them through the established body capture /
prepare workflow first. This document does not authorize that work or invent a
new body pipeline. Bind their actual files, draw, source and sessions; paths
alone are not proof. Final-hidden checks calibration and authenticates the
canonical capture; capture checks final-hidden source and expert projection.

```bash
set -euo pipefail
: "${MODEL:?}" "${REFERENCE_COST:?}" "${CALIBRATION_INPUT:?}" "${O4_ROOT:?}"
: "${BODY_PLAN:?}" "${BODY_PREPARED:?}" "${BOUNDARIES:?}" "${MTP_OFFLOAD_FOLDER:?}"
: "${CAMPAIGN_SPEC:?}" "${JOINT_TEMPLATE:?}"
export MODEL REFERENCE_COST CALIBRATION_INPUT O4_ROOT
export BODY_PLAN BODY_PREPARED BOUNDARIES MTP_OFFLOAD_FOLDER CAMPAIGN_SPEC JOINT_TEMPLATE
# Portable SHA-256: no dependency on sha256sum versus shasum spelling.
sha() { python - "$1" <<'PY'
import hashlib, sys
with open(sys.argv[1], 'rb') as f:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(8 << 20), b''):
        h.update(block)
print(h.hexdigest())
PY
}
# Verify BEFORE any unpickle, including comparison helpers and campaign readers.
REFERENCE_SHA=2e3dee29a974ad4b5e5de9ac5eb4116bf213e73ca25e1ba94eb6daec08443103
CALIBRATION_SHA=9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329
test "$(sha "$REFERENCE_COST")" = "$REFERENCE_SHA"
test "$(sha "$CALIBRATION_INPUT")" = "$CALIBRATION_SHA"
test ! -e "$O4_ROOT"
mkdir -p "$O4_ROOT"
export PRISMAQUANT_DEV_MODE=0
# New cache from the identity CLI; never copy a stale prepared identity file.
python -m prismaquant.tessera_joint_aura identity \
  --model "$MODEL" --out "$O4_ROOT/source-digest-cache.json"
```

The digest above authenticates the reference bytes, not the safety of an
arbitrary pickle. Use only the trusted original artifact, protect it against
concurrent modification, and recheck immediately before its consumer. Preserve
calibration **512 samples × 512 tokens, seed 0**, no retokenization/redraw;
preserve **4 probes, `seed_base: 7000`**, full token scope and original candidate
wires/recipes. A new source digest cache is the plan's `source_digest_cache`,
not an invented streamed `source_identity_cache`: dispatch's similarly named
option accepts the existing streamed identity proof schema, not this digest
cache interchangeably.

## Regenerate incompatible MTP artifacts, do not edit their seals

A historical MTP capture may have an incompatible builder/initialization
contract on current main. Regenerate it using the three actual CLI phases
(`projection`, not `project`). Do not patch its manifest, transplant a matching
builder digest, or reuse an unrelated final-hidden/source session. Use a fresh
phase directory for every attempt; the projection and final-hidden publishers
are create-only. These are separately authorized producer operations:

```bash
BODY_PLAN_SHA=$(sha "$BODY_PLAN")
BODY_PREPARED_SHA=$(sha "$BODY_PREPARED")
BOUNDARIES_SHA=$(sha "$BOUNDARIES")
python -m tools.glm_mtp_capture --phase projection \
  --plan "$BODY_PLAN" --plan-sha256 "$BODY_PLAN_SHA" \
  --out "$O4_ROOT/mtp-projection"
python -m tools.glm_mtp_capture --phase final-hidden \
  --plan "$BODY_PLAN" --plan-sha256 "$BODY_PLAN_SHA" \
  --boundaries "$BOUNDARIES" --boundaries-sha256 "$BOUNDARIES_SHA" \
  --out "$O4_ROOT/final-hidden" --offload-folder "$MTP_OFFLOAD_FOLDER"
python -m tools.glm_mtp_capture --phase capture \
  --plan "$BODY_PLAN" --plan-sha256 "$BODY_PLAN_SHA" \
  --prepared "$BODY_PREPARED" --prepared-sha256 "$BODY_PREPARED_SHA" \
  --final-hidden "$O4_ROOT/final-hidden/manifest.json" \
  --final-hidden-sha256 "$(sha "$O4_ROOT/final-hidden/manifest.json")" \
  --expert-projection "$O4_ROOT/mtp-projection/mtp-projection.json" \
  --expert-projection-sha256 "$(sha "$O4_ROOT/mtp-projection/mtp-projection.json")" \
  --out "$O4_ROOT/mtp-capture" --census-out "$O4_ROOT/mtp-census.json"
```

Check the produced census has **867 units and 3 anchor groups**, the original
unit/group membership, model/draw identities, and no missing routed experts.
Keep the original R1024 wire bytes and candidate roster as immutable donors.
Regeneration is not permission to choose new formats, prune candidates, sample
experts, reduce calibration, or replace wires with newly convenient ones.

## Rebuild a genuinely completed anchor campaign on builder mismatch

Configure `CAMPAIGN_SPEC` for this census, capture, model, exact draw and original
menu; its model paths must agree with the producer metadata. Put the actual
producer census in the new campaign workspace. The manifest must be generated
by `plan`, never hand-edited after generation. For ordinary compatible seed
reuse, the supported `--seed-checkpoint` plus `--seed-wire-dir` (or
`--seed-workspace`) gates may be used with the original artifacts; refusal of
a seed's builder is not permission to transplant its digests. On mismatch,
execute a new current-builder campaign and retain old wires as immutable
comparison/donor evidence. If the required original wire identity cannot be
preserved through the supported producer/seed gates, stop and disclose the
blocker rather than claiming the reproduction.

```bash
WORKSPACE="$O4_ROOT/anchor-campaign"
export WORKSPACE
mkdir "$WORKSPACE"
cp "$O4_ROOT/mtp-census.json" "$WORKSPACE/census.json"
python -m tools.dispatch_tessera_campaign plan \
  --spec "$CAMPAIGN_SPEC" --workspace "$WORKSPACE" \
  --calibration-cache "$O4_ROOT/mtp-capture/manifest.json"
python -m tools.dispatch_tessera_campaign check \
  --workspace "$WORKSPACE" --spec "$CAMPAIGN_SPEC"
# Separately authorized GPU campaign: submit waits, but inspect receipts too.
python -m tools.dispatch_tessera_campaign submit \
  --workspace "$WORKSPACE" --spec "$CAMPAIGN_SPEC"
# Only after every planned row has genuine successful completion receipts:
python -m tools.dispatch_tessera_campaign merge \
  --workspace "$WORKSPACE" --out "$O4_ROOT/anchors.pkl"
```

For a streaming campaign, additionally supply `plan --source-identity-cache`
with an authentic current `prismaquant.streamed_model.identity_cache.v1` proof
for this source/capture. That proof is a distinct prerequisite; do not rename
the digest cache to satisfy it. Plan/check success, a submitted job, files left
by a failed row, or a manually constructed `receipts.json` is not execution
success. Merge consumes successful campaign receipts, complete row costs,
checkpoint envelopes and wires. Keep the generated `plan.json`, `receipts.json`,
merged `anchors.pkl` and `anchors.anchors.json` together. The final input loader
requires the exact complete census roster/groups and campaign fanout.

## Freeze the final torch plan, then prepare and run that same plan

Construct a **new configuration**, not a patched producer manifest. The following
fills location/digest bindings in the reviewed complete template; all other
execution policy fields remain the template's original settings. Review the
result before prepare. `inputs` bindings have exactly `path` and `sha256`;
the required counts are separate integers. Never put example or old digest
strings into a binding for new bytes.

```bash
FINAL_PLAN="$O4_ROOT/final-plan.json"
export FINAL_PLAN
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['O4_ROOT']).resolve()
workspace = Path(os.environ['WORKSPACE']).resolve()
def bound(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}
p = json.loads(Path(os.environ['JOINT_TEMPLATE']).read_text())
p['model'] = str(Path(os.environ['MODEL']).resolve())
p['output_root'] = str(root / 'joint')  # nonexistent before prepare
p['source_digest_cache'] = bound(root / 'source-digest-cache.json')
p['calibration_input'] = bound(os.environ['CALIBRATION_INPUT'])
p['canonical_capture'] = bound(root / 'mtp-capture/manifest.json')
p['inputs'].update({
    'census': bound(workspace / 'census.json'),
    'campaign_plan': bound(workspace / 'plan.json'),
    'campaign_receipts': bound(workspace / 'receipts.json'),
    'merged_cost': bound(root / 'anchors.pkl'),
    'merged_checkpoint': bound(root / 'anchors.anchors.json'),
    'required_source_units': 867, 'required_campaign_groups': 3,
})
p['source_scope'] = 'mtp'
# Default is strict; an independently reviewed allowance belongs in the final
# configuration BEFORE hashing/preparing, following the exact schema below.
p.pop('historical_encoder_reuse', None)
e = p['execution']
e.update(projection_backend={'name': 'torch'}, n_calib_samples=512,
         calib_seqlen=512, n_probes=4, seed_base=7000)
# Original seed-0 draw is authenticated by calibration_input and capture,
# not replaced by seed_base (which controls probes).
assert p['calibration_input']['sha256'] == '9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329'
assert not Path(p['output_root']).exists()
with Path(os.environ['FINAL_PLAN']).open('x') as f:
    json.dump(p, f, indent=2, sort_keys=True)
    f.write('\n')
PY
FINAL_PLAN_SHA=$(sha "$FINAL_PLAN")
export PRISMAQUANT_DEV_MODE=0
python -m prismaquant.tessera_joint_aura prepare \
  --plan "$FINAL_PLAN" --plan-sha256 "$FINAL_PLAN_SHA"
PREPARED="$O4_ROOT/joint/prepare/prepared.json"
PREPARED_SHA=$(sha "$PREPARED")
# Do not resume an old tree or alter plan/backend/output_root after prepare.
test "$(sha "$FINAL_PLAN")" = "$FINAL_PLAN_SHA"
test "$(sha "$REFERENCE_COST")" = "$REFERENCE_SHA"
RESEARCH_OUT="$O4_ROOT/joint"
python -m experiments.accumulate_window_research \
  --plan "$FINAL_PLAN" --prepared "$PREPARED" \
  --prepared-sha256 "$PREPARED_SHA" --reference-cost "$REFERENCE_COST" \
  --out "$RESEARCH_OUT" --arms A --window 1
```

The last command is the instrumented **run**, not a second uninstrumented full
cost pass. It must call the production run with the same final plan and bound
prepared completion and preserve `PRISMAQUANT_DEV_MODE=0`. The corresponding
production interface is `python -m prismaquant.tessera_joint_aura run --plan
"$FINAL_PLAN" --plan-sha256 "$FINAL_PLAN_SHA" --prepared "$PREPARED"
--prepared-sha256 "$PREPARED_SHA"`; do not execute that separately into the same
tree. If a template carries a source identity proof, qualification window,
source transition, boundary storage, final-hidden reference or other bound
inputs, review those too: no stale path/session may survive configuration.
This recipe is not a complete generic joint-plan generator.

### Historical encoder reuse is an explicit allowance, not verification

Default: omit `historical_encoder_reuse`. Independently assess applicability
before retaining it from a template. It cannot repair an incompatible MTP
builder, missing receipts, changed weights/H/calibration/recipe, or wire bytes.
Only if the actual checkpoint records the exact historical seal below and
independent evidence supports that specific reuse may the final plan carry:

```json
{
  "historical_encoder_reuse": {
    "schema": "prismaquant.tessera_joint_aura.historical_encoder_reuse.v1",
    "allowlist": [{
      "encoder_source_sha256": "da5805bc707c041efaeb27c5083fb7809fd8cdc51d501f41103b791270b569cb",
      "reason": "REPLACE with independently reviewed applicability reason",
      "evidence": "REPLACE with durable provenance/evidence reference",
      "recorded_by": "REPLACE with actual reviewer identity",
      "recorded_unix": 0
    }]
  }
}
```

This illustrates the **exact closed schema**, not a ready-to-use permission.
Replace the metadata including the actual finite nonnegative recording time;
reason/evidence/author must be nonempty factual strings. No extra fields,
wildcards, prefixes, duplicates or empty allowlist. Disclose **both observed
recorded and current installed encoder seals**, the allowance entry and
`unverified_encoder_reuse` status in the report. A differing historical seal is
not re-derived by the current package; do not describe it as current
verification. Equality needs no allowance. See the architecture's historical
encoder reuse contract and `normalize_historical_encoder_reuse` /
`resolve_encoder_source_reuse` in `tessera_joint_aura`.

## Strict acceptance and evidence to retain

Before accepting A, derive the expected `(qname, format)` key set **independently
from window 1's admitted candidate specs**, not from returned terms, an
intersection, all payload keys, or an arbitrary 710-key subset. Require:

- Exactly **710** selected terms, with actual keys equal to the independently
  selected set: missing, extra and same-count wrong-window terms all refuse.
- Compare each A `total` to the reference cell's
  `signed_components_per_probe[0]["total"]`. Require bitwise equality for all
  **710/710**, including signed zero; numeric `==`, tolerances, aggregate
  totals, or equal predicted dloss are not substitutes. Report `compared=710`,
  `bitwise_equal=710`, `max_rel=0`; those summary numbers alone cannot establish
  the independent key-set gate.
- Preserve the original 867-unit / 3-group universe, R1024 candidate/wire
  identities, 512×512 seed-0 tokens and four probe seeds 7000–7003. Window 1 /
  probe 0 is this screen's measured subset, not a new smaller calibration or a
  claim that all four probes were reproduced.
- Retain original reference/token digests, generated producer outputs and
  receipts, final plan bytes/digest, prepared bytes/digest, launch environment,
  independent key roster and comparison report. Record failures as failures.
  Do not relabel a successful prepare, CPU preflight or campaign check as A's
  numerical result.

No C/D experiments, GPU measurement, numerical/performance acceptance or
production claim was performed for this change. Unavailable original inputs
prevent executing the recipe; planning is not artifact production.

## Inspected contracts

- [Agent rules](../../AGENTS.md), [design guidelines](../design/design_guidelines.md),
  [architecture](../ARCHITECTURE.md) (historical encoder reuse and joint input contracts).
- `tools/glm_mtp_capture.py`: `projection`, `final-hidden`, `capture` arguments
  and authenticated source / prepared-dispatch prerequisites.
- `tools/dispatch_tessera_campaign.py`: `plan`, `check`, `submit`, `merge`,
  seed admission and successful campaign receipt requirements.
- `prismaquant/tessera_joint_aura.py`: identity CLI, bound plan grammar,
  complete campaign intake, prepared binding and historical encoder reuse.
- `tests/test_accumulate_window_research.py`: independent 710-key / bitwise
  probe-zero gates and same-final-plan launcher contract. Its historical fixture
  documents the old implementation; tests exercise the restored module.

## Portable validation evidence

The required adjacent CPU suite plus installed-callback and recipe-argument
regressions passed: **145 passed, 3 skipped** (CPU only). The callback test runs
`compute_mtp_cost` through the installed research backward, real statistics lease,
prewarmed torch projection and report writer on a tiny MTP model. Its 710 synthetic
unchanged-weight aliases prove successful report/exit behavior and rejection of
value and signed-zero mismatches, not the historical population's numerical
match. GPU telemetry is stubbed. The recipe test parses all three documented MTP
commands with the current CLI, including the configured offload directory; it
does not produce artifacts.

No GPU run, full source authentication, regeneration, prepare, Arm-A numerical
comparison or performance measurement was performed. This proves the portable
instrumentation contract, not W2's actual 710-term match.

Command: `python -m pytest tests/test_accumulate_window_research.py tests/test_tessera_joint_encoder_reuse.py tests/test_tessera_joint_aura.py tests/test_stageb_prepared_inputs_bridge.py tests/test_accumulate_window_integration.py tests/test_accumulate_window_o4_arguments.py -q --basetemp="$TMPDIR/pytest"`

Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"95e2ac45b717253ecd5382eab799640a95983b48a2bfd1581211dd2cc2d59bd8","size":994}},"operation":"d262074980d56d66edfa921d5926764fd622ddb1d326c2dab0cf310e0a6f2508","receipt":"0160a7b52044bc9a647aa8990a09f86b26a8f6aeba73d4cc1ed4702590731a53"}
