# Streamed gold companion — 2026-09-08

The first release objective is one measured bitrate-matched GLM Tessera/EXL3
comparison. This instrument supplies a shared authenticated teacher; it does
not establish either engine's superiority. Six-variant allocation is deferred.

The BF16 source exceeds the resident memory of one or two Sparks. The existing
streamed teacher already builds the fixed 8×512 seed-42 WikiText training draw
and all-position top-K=8192 distributions with coverage and forward-fidelity
gates. The opt-in companion preserves those results and adds FP32 full-vocabulary
log-softmax rows at the final context position, from the **same forward**.
No second teacher forward, fitting capture or residency subsystem is introduced.

Default v1 payload/meta/evidence semantics remain unchanged. V2 retains the
existing fields and adds `final_logprobs` (tensor `[8,V]`, or null without the
companion), observed `source_execution`, `producer_identity`,
`wikitext_inputs_sha256`, normalized original `model_identity`, and
`fit_overlap_status: "unverified"`. Semantic
descriptors bind the final tensor as well as all three existing tensors. Final
rows must be finite, nonpositive, full-vocabulary FP32 log probabilities and
normalize within the existing 1e-6 mass tolerance. Top-K coverage/fidelity still
run before writing and on replay; no threshold was relaxed or newly fitted.

V2 source evidence includes the existing full checkpoint identity, observed
dispatch/derivative identity, the existing gold tool source closure and a seal
of the complete installed PrismaQuant package. These are observations, not
changes to the frozen pricing package. The builder compares checkpoint identity,
tokenizer bytes, input-file bytes, declared derivative and producer identity
before/after the forward. Non-null GLM derivative policy must match an observed
v2 derivative and its image-build SHA; original/null must observe no derivative.
The existing binder also validates the actual corrected runtime code and image.

The model-bound WikiText reader is owned separately by Raman. Its explicit
SHA-bound input preserves corpus/revision/sampling settings while binding GLM's
tokenizer/model family/vocabulary instead of the legacy DSv4 token constants.
The builder routes to that reader only with `--wikitext-inputs-sha256`; the
default old reader remains unchanged. Teacher calibration validation already
accepts model-specific tokenizer/token SHA and totals; its fixed gold scoring
contract is unchanged. The training draw's overlap with fitting is not yet
established by this payload. A separate overlap audit must carry any stronger
claim; full-vocabulary KL does not imply held-out evaluation.

```text
python tools/build_streamed_full_kl_teacher.py \
  --model ORIGINAL_BF16_SOURCE --identity-cache COMPLETE_SOURCE_IDENTITY \
  --wikitext-inputs MODEL_BOUND_GOLD_INPUTS --wikitext-inputs-sha256 INPUT_SHA256 \
  --source-derivative-json DERIVATIVE_POLICY --source-derivative-sha256 POLICY_SHA256 \
  --include-final-logprobs --offload-folder ADMITTED_OFFLOAD_DIRECTORY \
  --output NEW_TEACHER_PT --meta-output NEW_TEACHER_JSON
```

`DERIVATIVE_POLICY` uses the existing closed GLM contract and image-build
path/SHA binding; JSON null explicitly requests original execution. It does not
modify model code. The explicit policy path selects eager attention. Source
identity, tokenizer, draw and runtime evidence must correspond to the chosen
reference. This is a command template with required bound inputs, not a
launched measurement. The teacher runs through PB in the known-good container.
PB owns admission and placement; vLLM students use Rob's direct-run exemption.

V2 student replay requires `--teacher-meta` before loading. It compares the
candidate's actual tokenizer identity with the teacher and compares normalized
model family/vocabulary with the original input model domain, permitting
quantization-config additions. The streamer's staged text-only runtime config
remains independently bound; it is not mistaken for the full outer GLM config. Use `--score-positions final` for the companion
or `--score-positions all` for existing top-K scoring. Requesting final from a
v2 payload with a null companion refuses. V1 all-position payloads retain their
historical routing. Final v2 results retain `teacher_evidence`, explicit
`score_positions: "final"` and the actual position count.

The companion is engine-independent tensor data. An EXL3 consumer must use
these exact token IDs and vocabulary order, normalize full logits identically,
and retain teacher/source/input hashes before its number is paired with vLLM.
This change adds no EXL3 runtime consumer. Both arms still need actual served
quality, measured artifact bytes/quantizable denominator, device fit and
performance on the same declared hardware/workload. No GPU teacher or native
full-model measurement was launched for this change.

Issue #437 scopes the instrument work. The four new public-option/reduction/
routing regressions failed on unchanged tools in PB action
`7d493666fdd0cf303ed3bfdca047fd4f7e1674f282603eab733f04bec58be0a1` (exit 1).
The initial implementation passed 132 tests, zero skips, 56 Torch deprecation
warnings, in 40.78 seconds in action
`1bd94ecd595b8c2930d6a9c81e751fe6c6fd488ba1ef7da1c6835f81adffea59` (exit 0).
Both ran on dl380g10 CPU, Python 3.12/Torch 2.10 with four workers and native
threads bounded to one; reservations were 6 and 12 GiB respectively. The
attached audits verify actual terminals/logs and source bundles; the positive
audit also verifies canonical CAS receipt and result bytes. The failed action
has no successful CAS receipt. Additional review-driven identity checks and
shared-reader integration require final validation after this initial result.

The existing generic teacher-payload contract file was also collected explicitly
from `archive/gridbook_lane_2026-09-02/tests/test_full_kl_teacher_payload.py`;
it imports the live tools directly. PB action
`69a5591d1e0565b28d077ed486484c6d8588869212716bc66b91533b44e2a1d7`
passed all 44 tests with zero skips, including the SciPy statistic comparison,
in 41.12 seconds. DL CPU4/native1 used 6,220,759,040 peak cgroup bytes under
12 GiB admission. Its CAS receipt, actual result and source bundle are verified
in `legacy-contract-audit.json`. The separate archived full-KL math file with a
path-based collection skip was not counted as validation.


The model-bound reader implementation is integrated from `1a41c04c`, with
canonical cached Salesforce/WikiText fingerprint fix `ddb9e2af`. The accepted
materialization locator is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/model-bound-wikitext-inputs-01/glm-wikitext-inputs.v2.json`,
SHA `eb7c36a6211796bdaf523c20ca3583cbe4794e9583a11f3ec5f124641b374652`.
Raman owns its PB materialization and source/CAS audits. Its full-KL tensor SHA
is `d6549876d0bdef6ab5c5882003447ab4cd1df9bb05f1b4190ae4ed6294f595aa`.
The fixed input is not disjoint from fitting: the overlap audit reports gold
window 5 tokens `[280:512]` equal original fit window 176 tokens `[0:232]`,
a 232-token exact substring. No entire 512-token gold window is identical.
The test PPL prefix's longest exact substring is 12 tokens. These are exact
substring findings, not a statistical independence claim. Audit
`overlap-audit.json` beside the input has SHA
`30a0261a304c16a7e914b293bd06b5749290711ab147eefb59752a5d7b35048e`.
The requested fixed draw is retained without resampling. V2's conservative
`fit_overlap_status: "unverified"` is not a no-overlap assertion; the separate
factual audit carries the stronger observed overlap finding.

Final combined validation action
`08b37f82d47e69e3006b4beb62bfbd12a027c2742bbab39ba85c3b7acc5bcec8`
compiled all nine touched tools/test files and passed 186 tests, zero skips,
56 Torch deprecation warnings, in 42.18 seconds. It covers the final teacher,
reader/PPL integration, topology/provenance and architecture/staleness checks.
DL CPU4/native1 peak cgroup memory was 3,256,373,248 bytes under 12 GiB admission.
Canonical CAS receipt, actual result bytes and source bundle were verified;
the tested snapshot differs from implementation head `8c82e9d53` only by the
PB-generated closure. No `prismaquant/` bytes differ from frozen `21ae7d28`.
The attached `final-integration-audit.json` contains the exact identities.
