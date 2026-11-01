# Native export tri-state consolidation, PRI-226

This bounded configuration family is implemented against source base
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`. Tested implementation source is
`ba44f86967532c3013049b882d2a2f4c8314e2b9`. Independent review and merge are
pending. Native issue #1295 and the whole-repository program remain open.

## Contract and complete bounded callers

The old `run-pipeline.sh:2732-2749` independently parsed two environment
settings with identical lexical and refusal policies. The shared owner is
now `prismaquant/run-pipeline.sh:2722`, `append_export_tristate_arg`.

| Setting | Existing default and observable log | Shared caller and output |
| --- | --- | --- |
| `EXPORT_GPTQ` | `run-pipeline.sh:717` uses `:=auto`; `:1092` prints the effective value | `:2744` appends `--gptq` or `--no-gptq` |
| `EXPORT_SCALE_SWEEP` | `run-pipeline.sh:718` uses `:=auto`; `:1092` prints the effective value | `:2745` appends `--scale-sweep` or `--no-scale-sweep` |

These are the complete direct callers of the new private helper. The frozen
old path contains both defaults and the complete native argument block.
Unset and empty settings become `auto`. The parser accepts exactly
`0,false,False,FALSE,no,No,NO` and `1,true,True,TRUE,yes,Yes,YES`. Auto appends
nothing. Invalid input emits the original setting-specific stderr line and
exits 2. GPTQ runs first, including when both values are invalid.

The parser mutates the existing `EXPORT_ARGS` array after its native command,
paths, device, shard size, activation cache and pin arguments. Production cache
options remain later in the same path. It launches no process. The Tessera and
retained GGUF branches exit before this block; bytes before and after the
native assembly block match the base exactly.

The bare exporter's `--gptq` and `--scale-sweep` options remain argparse
`BooleanOptionalAction` inputs with `None` defaults at
`export_native_compressed.py:5104` and `:5121`. Their CLI grammar and downstream
auto policy remain separate from environment-token parsing. No exporter
algorithm or default is consolidated through the shell helper.

## Design and characterization

An explicit local helper keeps both environment-name/option mappings visible.
A declarative loop would require indirect variable lookup and a second mapping
encoding. The helper shares only the proven lexical policy.

`tests/fixtures/pre226_native_export_args.sh` is an independent copy of the
base defaults and argument block. Its bytes were checked against the recorded
base through `export-flags-oracle-source-check.json`. The live path does not
import the fixture. `tests/test_native_export_tristate.py` compares exact
stdout, stderr, exit status and NUL-delimited arguments with that fixture,
then separately checks expected argument tokens and error messages.

The cases cover all fourteen boolean spellings for each setting, auto, empty,
unset, nine invalid spellings, mixed values, two-invalid first refusal and
second-setting refusal. Paths with spaces/newlines and literal shell syntax
remain single arguments. A spy in the real argument block proves that both
settings delegate, in order, with their original value and option suffix.
The checks execute the actual shell section. They do not execute a model
pipeline or prove that an entire run reaches this section.

## Fleet evidence

All execution used the published PrismaBuild client and the pinned project
CPU interpreter on dl380g10. This was ordinary CPU test admission, not an
idle-host measurement.

- Red action `126351f4aba1eac2f18c3c05d85bf46b68bcb65b77145cc2bbf8bcaeb4d06e9f`
  tests source `f6331f77e4`. All 58 characterization cases pass. Only the shared
  parser control fails because the owner is absent, yielding exit 1.
- Green action `9407a01f261898f3347fee60e842d36b9c69261c15f5994dfaf267c4d36169fb`
  tests source `ba44f86967532c3013049b882d2a2f4c8314e2b9`. All 106 cases pass,
  with no skips or failures. Fourteen upstream Torch/Python 3.14 deprecation
  warnings remain. The new Python test compiles and the full pipeline passes
  `bash -n` in that same admitted action.
- The green selection includes the new tests, pipeline defaults, executed
  shell gates, prepriced override wiring, architecture and docs staleness.
  Native terminal status and log hashes are checked. Both retained source
  snapshots match every tracked file at their recorded commit. The successful
  result's CAS claim and payload checks pass. The checker does not validate
  worker attestation.
- The first untagged submission was refused before publication because the
  interpreter is absent on the submitting host. The retained submission names
  the x86 worker class that owns the recorded interpreter. No fleet action
  was duplicated by that refusal.

The retained dispatcher is `export_flags_jobs.py`; the verifier is
`verify_export_flags.py`. Evidence, commands, native endings, action records,
terminal logs, source checks and CAS checks are under
`/mnt/shared/paperclip-runs/PRI-226/` and in the attached evidence bundle.
Only audit documentation changes after the tested implementation commit.

## Program disposition

This closes the duplicated tri-state-policy candidate from `config-census-2`
subject to independent review and merge. The shard-size default candidate,
other shell/configuration areas and all unread census areas remain open.
Existing #1301/#1303 numerical and identity owners are untouched. The
shrink-only duplication baseline is unchanged. No new scanner or exemption
is introduced. The fixture preserves old behavior as a test oracle.

This is argument-parser consolidation. No model, calibration, numerical,
artifact-byte, residency, throughput, KL, bpp or serving result is claimed.
Existing cache/prefetch, vanilla-vLLM compressed-tensors and pinned Tessera
plugin gates remain in their owners. No default/menu/pin or ship-gate decision
is requested. The new source retains ordinary package provenance; no
historical source seal is rewritten or newly admitted.

PRs #3065, #3126 and #3129 remain at their separate approved heads. This slice
does not change PRI-228 membership or establish merged proof for any of them.
Program closure still requires the independent current-main residual census.
