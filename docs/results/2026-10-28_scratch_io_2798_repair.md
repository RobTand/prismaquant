# Explicit scratch-I/O declaration repair, 2026-10-28

## Scope and source

PRI-43 is a repair child of canonical PRI-318 / PrismaQuant #2798, not a second
implementation or closure of that obligation. It adopts PR #2925 at exact head
`da05a6cd033edb90a4f4efa6cf40f23bd76818c3` onto current main
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0` in isolated branch
`prismaquant-2798-pri43`. The original `tl/prismabuild-8f32e84c30cc-wp1` is not
edited. Original research, implementation and repair-1 commits are
`4a886ca0b371fd43639eede8c6e7660cff4ef687`,
`625caf74e79edbc579826d54c56162455ce9509b`, and the adopted head.

Source links and receipts remain on PR #2925. The original finite-float age
repair and its `10**309` / `10**400` regressions are retained. The two historical
placement failures and later reported independent 295-test pass remain distinct
historical evidence; neither is relabeled as a fresh gate result. Inherited
repair/review counters are not reset; this repair has no native reviewer bounce.

The production patch remains the inherited shared sealed-spec projection:
explicit, default-off traffic is forwarded verbatim, without a second source
read, inferred traffic, omitted capacities, merged roots or placement logic.
The only new executable-source difference from the adopted head is an acceptance
fixture root correction. Its temporary-directory roots were under `/tmp`, which
the existing compilation-cache contract always treats as overlay even when
mounted. Synthetic `/scratch/fixture-2798/root-*` roots exercise the intended
capacity-only baseline and multi-pair refusal without weakening that contract.
The tests only seal requests; no directory is created at those synthetic roots.

## Local validation

Environment: Python 3.14.4, pytest 9.1.1, torch 2.11.0+cpu, transformers 5.17.0,
tessera-quant 0.1.0. Installed PB SDK 0.1.0 has direct-url source commit
`027103d9a8417e06c7f13356e58779a313cd7088`; this is identified installed SDK
coverage, not a claim to have tested current fleet runtime. Interpreter:
`/home/rob/venvs/pq-pin-aaa41700-pb20261016-git/bin/python`.

All pytest commands use `-c pytest.ini`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`CUDA_VISIBLE_DEVICES=''`, and task-owned `--basetemp` directories.
The harness Git wrapper removes inherited author/committer environment variables;
the unchanged Git-identity fixtures therefore need `GIT_TEMPLATE_DIR` pointing
to a task-only Git init template with a configured synthetic user identity.
No global Git configuration, launcher, permission or production code is changed.

1. Current-main baseline, before adopting the patch:
   `python -m pytest -c pytest.ini tests/test_dispatch_joint_quanta.py -q -k 'test_two_stage_a_rows_whose_windows_exceed_a_box_are_not_both_placed or test_a_stage_b_row_charges_spill_cotangent_and_spool_to_the_box' --basetemp="$PAPERCLIP_TASK_SCRATCH_DIR/baseline-placement"`:
   **2 passed, 115 deselected**, 2.80 s. Both placement tests are unchanged.
2. First adopted dispatcher/container run: **215 passed, 5 failed**, 5.11 s.
   Three failures came from the explicit-cache `/tmp` fixture roots; two came
   from Git identity being removed by the harness. Both placement cases passed.
   An auxiliary diagnostic first encountered a tests import-path collision,
   then confirmed the Git identity error with `PYTHONPATH=tests`.
3. With corrected synthetic roots and the task-only Git template:
   `python -m pytest -c pytest.ini tests/test_dispatch_joint_quanta.py tests/test_tessera_campaign_container.py -q --basetemp="$PAPERCLIP_TASK_SCRATCH_DIR/dispatcher-container-repaired"`:
   **220 passed**, 4.21 s. This includes all 43 #2798 acceptance cases, five
   float-age boundary cases, and both previously disclosed placement cases.
4. `python -m pytest -c pytest.ini tests/test_joint_quantum_row.py tests/test_architecture_doc.py tests/test_io_site_freeze.py tests/test_duplication_baseline.py -q --basetemp="$PAPERCLIP_TASK_SCRATCH_DIR/guards"`:
   **42 passed**, 14 existing Torch/Python deprecation warnings, 55.51 s.
5. `node .github/scripts/linked_issue.test.cjs`: **64 passed**, no failures or
   skips. This tests local policy code, not a fresh native `linked issue` status.

## Remaining gates and limits

The compliant replacement uses **Refs #2798**, never a closing reference. Local
evidence does not close the parent, satisfy D52, or replace native exact-head
review. Worker 13's configured independent review is additional to the required
cross-family D52 approval of the exact published head. Manager routing is needed
if that capability is unavailable. No worker merge is authorized.

Current workflow source describes a composed `pb-tests` pre-merge check and
skips the hosted CPU suite on PR events. Fresh required GitHub checks must be
enforced on the integrated SHA through the Mergify queue, not inferred from
historical or local receipts. PrismaBuild is for admitted GPU, benchmark and
large/long-running compute, not a substitute CI system. No new fleet action or
CAS receipt is claimed by these fast local checks.

There is no GPU, throughput/rate, serving, profile-freshness, fleet readiness,
deployment or numerical qualification. Explicit opt-in remains disabled by
default; multi-root targets remain unsupported for I/O opt-in. No shared
baseline/census/changelog is rewritten.
