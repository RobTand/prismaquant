"""Check the historical evidence for the retention decisions in issue 2806."""
import os
from pathlib import Path
import subprocess
import sys


def test_retention_evidence_matches_supplied_baseline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    selectors = [
        "tests/test_stage_settings_guard.py",
        "tests/test_prismaquant_export_native_compressed.py::TestProductionCacheExportPath",
        "tests/test_retired_codebook_refusals_1304.py",
        "tests/test_own_process_isolation.py",
        "tests/test_own_process_child_exit_audit.py",
        "tests/test_own_process_measurement.py",
        "tests/test_io_engine_sealed_memory.py",
        "tests/test_reclaim_readings_gb10.py",
        "tests/test_tessera_row_stream.py",
    ]
    result = subprocess.run(
        [sys.executable, "tools/coverage_containment.py", "query",
         "docs/measurements/2806-test-line-ranges-afac07dfbcfe.tsv.gz",
         *selectors, "--lines"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    out = Path(os.environ.get("TL3B_OUT", tmp_path))
    (out / "2806-retention-query.tsv").write_text(result.stdout)
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    counts = {row[1]: int(row[2]) for row in rows if row[0] == "lines"}
    aura = selectors[0] + "::test_aura_streaming_and_checkpoint_directory_are_reuse_identity"
    for stage in ("aura-cost", "aura-hybrid-cost"):
        assert counts[f"{aura}[streaming-{stage}]"] == 96
        assert counts[f"{aura}[checkpoint-directory-{stage}]"] == 94
    cache = selectors[1] + "::test_"
    for method, count in (
        ("mxfp8_scale_sweep", 194), ("fp8_scale_sweep", 195),
        ("fp8_gptq", 196), ("mxfp8_e4m3_gptq", 195),
    ):
        assert counts[cache + method + "_cache_defers_to_export_recompute"] == count
    assert [
        "contained-by", cache + "fp8_scale_sweep_cache_defers_to_export_recompute",
        cache + "fp8_gptq_cache_defers_to_export_recompute", "196",
    ] in rows
    assert counts[selectors[2] + "::test_retired_refusal_is_not_a_key_error"] == 6
    for selector, count in zip(selectors[3:6], (22, 8, 13)):
        assert sum(row[0] == "not-measured" and row[1].startswith(selector + "::")
                   for row in rows) == count
    for selector in selectors[6:8]:
        assert ["unknown", selector] in rows
    assert sum(node.startswith(selectors[8] + "::") for node in counts) == 23
