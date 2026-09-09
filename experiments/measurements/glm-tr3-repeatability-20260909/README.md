# GLM EXL3 repeatability diagnosis — 2026-09-09

This experiment investigates repeated native results before using the EXL3
checkpoint as a quality comparator. It does not quantize EXL3, change the
teacher, tune allocation on the final panel, or qualify a serving release.

## Observed native results

Four repetitions of the same `final-0000` window in one engine produced mean
full-vocabulary KL values of 0.04676430184428184, 0.026519084908803215,
0.047837852412688134 and 0.026992125752628725. Each repetition supplied all
2047 prediction positions. The native prompt-logprob alignment check passed
against each repetition's own output. This establishes repeat variation; it
does not establish which native operation causes it.

The first diagnostic result is retained at
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-exl3-repeatability-01/repeatability.json`,
SHA256 `40704ded216ab089808422bb4b582ae721d4dedb65fb540a4b1caa6e69da7af6`.

The third diagnostic hashes complete observed boundary tensors, alongside
first-token hashes. Its four repetitions contain 522 ordered boundary events
per rank; repeated calls to the same module are separate events. Comparing
repetition 0 with 1 first differs at the layer-7 MLP output. Comparing 0 with
2 or 3 first differs at the layer-3 MLP output on both ranks. Earlier observed
full tensors agree in those comparisons. Equal gate logits do not establish
equal top-k selections or equal route tables.

That result is retained at
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-exl3-repeatability-03/repeatability.json`,
SHA256 `3ffd86be50092261d31cca61877822cb007d327207b41d2d52d4d605e41090a6`.
Its source is `632e2324349bb122cbf25514ee6ecdee8126c91e`; the head exited 0.
Neither diagnostic artifact is a hook qualification or a complete-panel score.

## Bounded grouped-down replay

The scorer's opt-in `--diagnostic-grouped-down-replay` requires
`--diagnostic-repeat-first-window 4 --qualify-hook --diagnostic-layers
--diagnostic-full-boundaries --require-exl3-diag`. It wraps the already-loaded
EXL3 grouped-down extension selected by the worker, recording the extension,
EXL3 caller and replay helper file hashes. It never imports an absent native
plugin to make the diagnostic available.

The replay fires once at `model.layers.3.mlp`, with 2048 input rows, while the
real capture owner's `window_id` is armed. It makes eight calls using the
original live arguments and fresh zeroed output buffers. All passed tensor
bytes, including the input activation and pointer/routing/segment tables, are
hashed before and after. Pointed-to weight bodies are not hashed. The original
served output is retained and checked separately; replay outputs do not replace
it. The coordinator requires a first-request record from every TP rank and
includes the records in diagnostic progress and completion artifacts.

This is an intrusive correctness diagnostic with synchronization and host
hashing. It is unsuitable for speed measurements. Eight identical calls do
not establish determinism; differences require checking input stability,
finiteness and output preservation before interpretation. Gather and gate/up
are outside this replay's scope. Native execution of this new replay remains
pending while the pricing campaign occupies the GPUs.

## CPU integration evidence

PrismaBuild action
`8199c7fa47d39482885afa2fa69b08544ddf7dcfe33f0789813aa60186cec725`
reproduced missing replay records on both ranks with the original helper:
it checked an `armed` attribute that the real `PromptLogitsCapture` lacks.
The run also caught one invalid flag combination in a new negative test.
The remaining 103 tests passed. The helper now uses the real `window_id`
contract, and that test's flags were corrected.

Action `a20af22facf3d8e0b6f62d541a163a5e4e9d691f6799952c8d0372979cec4b70`
then passed all 106 scorer CPU tests, with no skips, on DL380 using four
workers and native threads bounded to one. The command was
`/home/rob/venvs/pq-cpu312/bin/python -m pytest -q -n 4
tests/test_glm_tr3_full_vocab.py` in the admitted checkout. The actual terminal,
cleanup, CAS result and tested file bytes were independently checked and
retained under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/tr3-replay-preparation-04/`.
This run used replay helper commit `9bff2b5b24`; it is integration evidence,
not native-kernel evidence or validation of subsequent helper changes.

The final integration adopted replay helper `680a3cafe4` as `bd5a6abc3`.
PrismaBuild action
`5b6df91e1f4ae3d7b0d75965459df05c68e4ff4f95ce33c9c447724fb9f68f6f`
compiled the scorer, helper and their two test modules, then passed 147 CPU
tests with one skip in 8.73s using the same worker/thread limits. The skipped
helper check reads the scorer from a developer-specific absolute path that is
absent on DL380. The two integration tests instantiate and exercise the real
capture owner from the admitted checkout and both passed. Actual source and
CAS checks are in `root-integration-final-cas-source.json` under the preparation
directory above. The red snapshot was also authenticated against the original
delivered helper; `root-integration-red-audit.json` records that negative result.
No native replay result exists at this checkpoint.
