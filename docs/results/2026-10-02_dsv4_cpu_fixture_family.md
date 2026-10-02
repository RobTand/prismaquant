# DSv4 CPU fixture/import-state family

Scope: PQ #1990/#1992 and #1957/#1958/#1959/#1969–#1976/#1978/#1984/#1985.
Base: `6707ee494a504195babd355e77d76343233d738d` (merged PR #1993).
This is test infrastructure, not production, GPU/model initialization, numerical,
serving, performance, runtime-pin, export or release qualification.

## Cause and repair boundary

The real `test_vl_wrapper_config_reproduces_issue_12` asks
`AutoModelForCausalLM.from_config` to reject an unsupported `Ovis2Config`.
Transformers 5.16.1 formats its supported configuration list using the lazy
mapping (`auto_factory.py:250–252`, `keys:618–624`), importing native
`transformers.models.deepseek_v4`. Restoring `OVERRIDE_ERRORS` cannot undo
`sys.modules` or lazy mapping imports. A subsequent real vendored registration
correctly refuses the native package. No production guard is defective here.

First separate the real predecessor/session owner from MXFP4-specific
assertions in `tests/test_layer_streaming_mxfp4_isolation.py`. Reuse the existing
`test_own_process_isolation._session` and genuine
`own_process_samples/sample_dsv4_native_guard.py`. Then mark only the actual
vendored consumers of the four newly repaired files with `own_process`:
the existing harness groups selected node IDs per module, runs a child only
in a mixed-file session, and propagates each actual outcome. Lightweight
profile/rename/Fisher/walk tests remain shared. No namespace deletion, fake
package/profile, registration suppression, skipped target, assertion or width
change is used. The source-backed fake-trace ratchet gets the same boundary;
its existing source-file requirement and refusal remain unchanged.

## Inventory and closure contract

| Issue | Actual node (under `tests/`) | Historical batch / guard | Acceptance |
|---|---|---|---|
| 1957 | `test_model_walk.py::test_dsv4_real_cpu_walk_discovers_and_decides_wo_a` | b00007 / VendoredOverrideError at `_shrunken_dsv4` registration | Controlled RED and mixed-order GREEN, unchanged walk claims |
| 1958 | `test_model_walk.py::test_dsv4_walk_fails_without_the_profile_rules` | b00007 / same registrar | Same order proof, unchanged wo_a/bmm refusal |
| 1959 | `test_grouped_linear_fisher.py::test_real_dsv4_wo_a_gets_a_priced_probe_row` | b00008 / same registrar | Same order proof, unchanged measured toy Fisher row |
| 1969 | `test_layer_streaming_mxfp4.py::test_batched_decode_matches_reference_across_chunks` | b00010 / DeadVendoredOverrideError during checkpoint profile detection | Already repaired by PR1993; adopt full-file proof |
| 1970 | `test_layer_streaming_mxfp4.py::test_declared_expert_dtype_populates_mxfp4_names` | b00010 / same registry guard | Already repaired; adopt full-file proof |
| 1971 | `test_layer_streaming_mxfp4.py::test_declared_non_e8m0_scale_fmt_raises` | b00010 / same registry guard | Already repaired; adopt full-file proof |
| 1972 | `test_layer_streaming_mxfp4.py::test_e8m0_ff_scale_yields_all_nan_block` | b00010 / same registry guard | Already repaired; adopt full-file proof |
| 1973 | `test_layer_streaming_mxfp4.py::test_layer_cache_estimate_matches_real_resident_bytes[packed_dtype0]` | b00010 / same registry guard | Already repaired; adopt exact dtype node proof |
| 1974 | `test_layer_streaming_mxfp4.py::test_layer_cache_estimate_matches_real_resident_bytes[packed_dtype1]` | b00010 / same registry guard | Already repaired; adopt exact dtype node proof |
| 1975 | `test_layer_streaming_mxfp4.py::test_missing_scale_fmt_is_not_fatal` | b00010 / same registry guard | Already repaired; adopt full-file proof |
| 1976 | `test_layer_streaming_mxfp4.py::test_mxfp4_decode_bit_exact_vs_scalar_reference` | b00010 / same registry guard | Already repaired; adopt unchanged bit-exact proof |
| 1978 | `test_layer_streaming_mxfp4.py::test_real_dsv4_layout_routed_nibbles_and_shared_block_fp8` | b00010 / same registry guard | Already repaired; adopt unchanged layout proof |
| 1984 | `test_dsv4_layer_streaming_rename.py::test_csa_compressor_returns_indices_not_a_gather` | b00012 / VendoredOverrideError at real registration | Controlled RED and mixed-order GREEN, unchanged sentinel/layout checks |
| 1985 | `test_dsv4_layer_streaming_rename.py::test_indexer_pooling_carries_the_coff_overlap_widening` | b00012 / same registrar | Same order proof, unchanged original widening dimensions |
| 1990 | `test_deepseek_v4_profile.py::test_rope_axis_mapping_matches_the_vendored_definition` | b00017 / profile's real registrar | Controlled RED and mixed-order GREEN, unchanged rope equality |
| 1992 | Import-order isolation umbrella | b00017 / same as 1990 | Deterministic predecessor identified, genuine guards retained, combined qualification |

Every implicated historical shard lists the wrapper module before the failing
DSv4 file. This identifies an actual preceding import trigger, not an assertion
that no other predecessor can import DSv4. Full historical shard order was not
re-executed. Logs: `/home/rob/pbmergeq/prismaquant/batches/b00007/candidate.log`
(lines 606–726), b00008 (624/651), b00010 (501–1414), b00012 (531–666),
b00017 (526–579). No missing log recovery or shared recursive scan was needed.

## Exact evidence and commands

Evidence root:
`/home/rob/tmp/claude-campaign-20260926/tmp/native-pq-dsv4-family/`.
Public issue/ownership refreshes, node inventory and claims are recorded there.
Final source/terminal/receipt/CAS reconciliation belongs in the native writer
report, not in a fabricated pre-completion qualification claim.

Controlled RED action:
`3eb1797f1f7b55870409f4588b1a7e29d73877c9fd622ec9d9b7a4b8966cb6f5`.
Four outer regression groups fail: six actual inner DSv4 nodes raise genuine
`VendoredOverrideError`; each group's wrapper and native-guard controls pass.
Command is published `pbrun.py --cwd CHECKOUT --anywhere --cpus 2
--demand mem_gb=4 --priority -10 --timeout-s 600 --wait-s 600`, with action-owned
campaign `TMPDIR` and OMP/MKL/OpenBLAS threads=1. The admitted interpreter is
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`; it verifies installed
Git commits and RECORD bytes via `pbtest_pins.verify_install`, then calls real
pytest on `tests/test_dsv4_fixture_isolation.py`. PB commit
`95a59051d48cda82eea7927f31870c6c862d7174` (44 verified files), Tessera commit
`b40c93cb73745097e57a1ba4cf5b9eee166c759a` (102 verified files).
The earlier action `6ab6eea97eebbb7f9dc4d75a8da6a78db7b347464b2f67c76b9ce0727e8b54d3`
failed before pytest because the caller omitted checkout `tools` from
`sys.path`; it collected zero tests and is NOT RED evidence. Only the caller
import path was corrected; no package repair or bypass was made.

Reuse PR1993 action
`86df5b74035900e0726e91f2fe74b9298dce0cf61126944624c7b6eb419aad4b`:
32 passed, zero failed/skipped, three fixture files compiled. Its full-file
mixed-order regression actually checks all 20 original MXFP4 cases plus the
wrapper/native controls. Nine newly inventoried siblings share the 13-failure
historical registry guard; their numerical assertions and target file are
unchanged here. Source/receipt mapping is in the immutable takeover bank and
`cpu-boundary-next/dsv4-rebased-primary.json`. Do not count these nine as new
implementation, nor reopen already-closed #1977/#1979/#1980/#1981. The shared
session-helper refactor is separately checked by one final full-file regression,
not another duplicate direct qualification of the unchanged MXFP4 file.

Final gate: one combined targeted/regression/own-process/duplication/IO-site/doc
ratchet action, installed-pin checks, actual `pbtest_outcomes.main`, and compile
of the six touched Python test files. Record exact collection, outcomes, skips,
terminal/exit, source snapshot and CAS in the writer report. A preexisting
source-backed fake-trace skip cannot be called a passed ratchet; its required
input is `/home/rob/dq-runs/dsv4-flash-0731/source/config.json`.

## Holds and authority

Local edit-time analyzer diagnostics on unchanged pytest/torch/Transformers
imports reflect a different local analysis environment from the admitted
interpreter. They are retained, not suppressed, not claimed clean, and not a
waiver of required CI. New fixture/session imports were resolved before RED.
No lint-clean claim or host/shared package repair is made.

Astra owns independent exact-head review, acceptance, required CI, central
integration, queue, merge and cleanup. Only fully proven issue criteria receive
closing references after the combined gate. Production/serving/research
umbrellas, parked model-init grant, historical fe02 freshness failure, GPU,
performance/repricing and release remain held and outside this PR.
