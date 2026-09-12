"""Demonstrate that each admission check actually fails when it is broken.

Not a test and not collected as one: this is the evidence behind "a check that
cannot fail is not a check". For each guard in `admit_fixed_resources` it
edits that guard out of the source, runs the tests that claim to cover it, and
requires them to FAIL. A guard whose removal leaves its tests green would be a
vacuous check and is reported as such.

Run it under PrismaBuild like any other test execution, from the repository
root: `pbrun.py --cwd <checkout> --cpus 2 --demand mem_gb=4 --priority -10
--tag dl380g10 -- <interpreter> tools/demonstrate_admission_checks.py`. It
mutates only the snapshot it runs in and restores the file after every case,
including on failure.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "prismaquant" / "runtime_provenance.py"
ADMISSION = "tests/test_runtime_fixed_resource_admission.py"

CASES = [
    ("the gate reads `derived` instead of its own recomputation",
     "        recomputed = verdict.recomputed_terms[term]",
     '        recomputed = report["derived"]["terms"][term]',
     [f"{ADMISSION}::test_the_table_is_checked_against_the_recomputation_and_not_against_derived"]),

    ("the declared-versus-recomputed comparison is dropped",
     "        elif recomputed != value:",
     "        elif False:",
     [f"{ADMISSION}::test_a_declared_fixed_scratch_that_differs_refuses",
      f"{ADMISSION}::test_the_table_is_checked_against_the_recomputation_and_not_against_derived"]),

    ("an unrecomputable term is compared as a number instead of blocking",
     "        if recomputed is None:\n            refusals.append(f\"no {term} is recomputable, so this table's fixed {field} \"\n                            f\"({value}) has no evidence\")",
     "        if False:\n            pass",
     [f"{ADMISSION}::test_an_unrecomputable_term_blocks_rather_than_disagreeing"]),

    ("the report is consumed without the independently supplied run identity",
     "    verdict = consume_full_engine_resource_report(reference, root=root,\n"
     "                                                  expected_run_identity=expected)",
     "    verdict = consume_full_engine_resource_report(reference, root=root)",
     [f"{ADMISSION}::test_a_stale_source_configuration_or_runtime_refuses",
      f"{ADMISSION}::test_a_foreign_device_refuses",
      f"{ADMISSION}::test_a_relation_that_names_another_full_engine_run_refuses"]),

    ("the gate stops refusing a synthetic capture on its own authority",
     "    if verdict.fixture_provenance is not None:",
     "    if False:",
     [f"{ADMISSION}::test_a_synthetic_capture_refuses_on_this_gate_s_own_authority"]),

    ("the supported execution boundary is not checked against the table",
     "        if declared_execution[key] != supported:",
     "        if False:",
     [f"{ADMISSION}::test_an_unsupported_boundary_refuses"]),

    ("a batch larger than one request is projected",
     "    if context.batch_size != 1:",
     "    if False:",
     [f"{ADMISSION}::test_an_unsupported_boundary_refuses"]),

    ("one measured assignment is taken to cover every priced alternative",
     "    if alternatives:",
     "    if False:",
     [f"{ADMISSION}::test_a_second_priced_format_for_one_unit_refuses"]),

    ("the census is not compared with the roster this table prices",
     "            if census_units != table_units:",
     "            if False:",
     [f"{ADMISSION}::test_a_census_that_omits_a_priced_unit_refuses"]),

    ("a unit may carry more than one selected row",
     "        if unit in selected:",
     "        if False:",
     [f"{ADMISSION}::test_two_selected_rows_for_one_unit_refuse"]),

    ("a selected row need not name a format this table prices",
     '        if "format" not in row:',
     "        if False:",
     [f"{ADMISSION}::test_a_selected_row_that_names_no_format_refuses",
      f"{ADMISSION}::test_a_selected_row_this_table_does_not_price_refuses"]),

    ("the selected rows need not cover the census",
     "    if census_units is not None and sorted(selected) != census_units:",
     "    if False:",
     [f"{ADMISSION}::test_a_selected_row_omitted_for_a_census_unit_refuses"]),

    ("the workload's calibration is not bound to the table's",
     "    elif (not isinstance(calibration, Mapping)\n"
     "            or calibration.get(\"sha256\") != context.calibration_sha256):",
     "    elif False:",
     [f"{ADMISSION}::test_a_stale_calibration_refuses"]),

    ("an inline resource claim is accepted as a report reference",
     "    if not isinstance(claim, Mapping) or set(claim) != set(FIXED_RESOURCE_REPORT_REFERENCE):",
     "    if False:",
     [f"{ADMISSION}::test_the_receipt_must_reference_a_report_rather_than_claim_a_result",
      "tests/test_runtime_provenance.py::test_relation_never_admits_unproved_fixed_resources",
      "tests/test_full_engine_resource_report.py::test_the_admission_gate_is_the_only_reader_of_this_recomputed_partition"]),
]


def main() -> int:
    original = SOURCE.read_text(encoding="utf-8")
    vacuous, broken = [], []
    for name, old, new, tests in CASES:
        if old not in original:
            broken.append(f"{name}: the guard text is not in the source any more")
            continue
        SOURCE.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", *tests],
                                    cwd=ROOT, capture_output=True, text=True)
        finally:
            SOURCE.write_text(original, encoding="utf-8")
        tail = result.stdout.strip().splitlines()[-1:] or ["<no output>"]
        verdict = "TESTS FAIL (the check is real)" if result.returncode else "TESTS PASS -- VACUOUS"
        print(f"[{verdict}] {name}\n    rc={result.returncode} {tail[0]}", flush=True)
        if not result.returncode:
            vacuous.append(name)
    for message in broken:
        print(f"[COULD NOT MUTATE] {message}", flush=True)
    if vacuous or broken:
        print(f"\n{len(vacuous)} vacuous, {len(broken)} unmutatable of {len(CASES)}")
        return 1
    print(f"\nall {len(CASES)} guards demonstrated: removing each one fails its tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
