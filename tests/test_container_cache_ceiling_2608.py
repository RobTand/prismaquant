"""A measured ceiling needs complete GPU workload evidence (PQ #2608)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant import container_cache_peak as peak_mod


def _growth_workload(root: Path, total: int = 600000):
    def run():
        path = root / "workload.bin"
        written = 0
        with open(path, "wb") as handle:
            while written < total:
                handle.write(b"\x5a" * 50000)
                handle.flush()
                written += 50000
                time.sleep(0.06)
        path.unlink()

    return run


def _gpu_workload() -> dict:
    """Synthetic producer-shaped results, not evidence of a GPU run."""
    return {"quantiser": "fp4_e2m1/g16", "shape": [64, 256], "seed": 2463,
            "device": "cuda:0", "quantise_s": 0.1, "output_mean": 0.0,
            "kda_probe": {"status": "ran", "sha256": "0" * 64,
                          "shape": [1, 128, 2, 128], "probe_s": 0.1,
                          "compiled": sorted(
                              f"{name}_{mode}"
                              for name in ("gram_fwd", "gram_bwd_rows", "gram_bwd_cols")
                              for mode in ("inclusive", "strict"))}}


def _evidence_fixture() -> dict:
    """Serialized parser input only. No workload or GPU executes here."""
    samples = [
        {"index": index, "unix": 100.0 + index * 0.25, "phase": phase,
         "scan_duration_s": 0.01, "allocated_bytes": size,
         "apparent_bytes": size, "files": 1, "dirs": 1, "errors": []}
        for index, (phase, size) in enumerate(
            [("initial", 512), ("periodic", 1024), ("final", 512)])
    ]
    return {"schema": "prismaquant.container_cache_row.v1",
            "failure": None, "workload": _gpu_workload(),
            "measurement": {"sample_interval_s": 0.25, "samples": samples,
                            "sample_count": 3, "gaps": [], "errors": [],
                            "peak_allocated_bytes": 1024, "peak_sample_index": 1,
                            "final_allocated_bytes": 512,
                            "row_started_unix": 100.02, "row_finished_unix": 100.49,
                            "child_exitcode": 0, "valid": True,
                            "incomplete_scan": False}}


def _quantum_fixture():
    receipt = _evidence_fixture()
    receipt["schema"] = "prismaquant.container_cache_quantum_row.v1"
    receipt["device"] = "cuda"
    receipt["workload"] = {
        "row": "stage_a_capture_plus_stage_b_quantum", "layer": 1,
        "units": ["model.layers.1.proj"], "formats": ["BF16"],
        "n_probes": 2, "seed_base": 7000, "checkpoints": [0, 1, 2],
        "windows": 1, "served_compile": receipt["workload"]}
    return receipt



def _complete_receipt(tmp_path: Path) -> dict:
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = peak_mod.measure_around(
        _growth_workload(cache), cache, interval_s=0.02)
    assert receipt["measurement"]["valid"], receipt["measurement"]["errors"]
    receipt["workload"] = _gpu_workload()
    receipt["schema"] = "prismaquant.container_cache_row.v1"
    receipt["power"] = {"summary": {"sample_count": 3},
                        "samples": [10.0, 11.0, 10.5], "times": [1.0, 2.0, 3.0],
                        "error": None}
    return receipt


def test_ceiling_refuses_receipt_without_gpu_workload(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["workload"] = {}
    with pytest.raises(ValueError, match="GPU workload"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_ceiling_refuses_reservation_without_peaks(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["measurement"] = {"samples": [], "sample_count": 0,
                              "peak_allocated_bytes": 0,
                              "peak_sample_index": None,
                              "errors": [], "gaps": [],
                              "incomplete_scan": False, "valid": False,
                              "child_exitcode": 0}
    with pytest.raises(ValueError, match="peak"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_ceiling_refuses_partial_evidence_with_missing_peak(tmp_path):
    receipt = _complete_receipt(tmp_path)
    receipt["measurement"] = dict(receipt["measurement"])
    receipt["measurement"]["peak_allocated_bytes"] = 0
    receipt["measurement"]["peak_sample_index"] = None
    receipt["measurement"]["valid"] = False
    with pytest.raises(ValueError, match="peak"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_killed_child_refuses_ceiling_and_names_coverage_gap(tmp_path):
    (tmp_path / "ballast.bin").write_bytes(b"\x5a" * 100000)
    sampler = peak_mod.CachePeakSampler(tmp_path, interval_s=0.05)
    sampler.__enter__()
    sampler._process.kill()
    sampler._process.join(timeout=5)
    sampler.stop()
    result = sampler.result()
    assert not result["valid"]
    assert result["incomplete_scan"]
    receipt = {"measurement": result, "workload": _gpu_workload(),
               "schema": "prismaquant.container_cache_row.v1",
               "device": "cuda", "failure": None}
    with pytest.raises(ValueError, match="cover the complete row"):
        peak_mod.derive_cache_ceiling_from_receipt(
            receipt, headroom_bytes=4341760)


def test_complete_evidence_matches_direct_derivation(tmp_path):
    receipt = _complete_receipt(tmp_path)
    headroom = 4341760
    assert (peak_mod.derive_cache_ceiling_from_receipt(
        receipt, headroom_bytes=headroom)
        == peak_mod.derive_cache_ceiling(
            receipt["measurement"]["peak_allocated_bytes"],
            headroom_bytes=headroom))


@pytest.mark.parametrize("workload", [
    {"device": "cpu", "reservation_bytes": 2147483648},
    {"reservation_bytes": 2147483648}, {"device": "cuda:0"},
    {"kda_probe": {"status": "ran"}},
])
def test_cuda_label_does_not_replace_workload_results(workload):
    receipt = _evidence_fixture()
    receipt["device"] = "cuda"
    receipt["workload"] = workload
    with pytest.raises(ValueError, match="GPU workload"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("path", [
    "quantiser", "shape", "seed", "quantise_s", "output_mean", "device",
    "kda_probe", "kda_probe.status", "kda_probe.sha256", "kda_probe.shape",
    "kda_probe.probe_s", "kda_probe.compiled",
])
def test_missing_workload_result_names_the_field(path):
    receipt = _evidence_fixture()
    target = receipt["workload"]
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    del target[keys[-1]]
    with pytest.raises(ValueError, match=keys[-1]):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("path,value", [
    ("device", "cpu"), ("device", "cuda_not_a_device"),
    ("quantise_s", float("nan")), ("output_mean", float("inf")),
    ("seed", True), ("shape", [64, 0]),
    ("kda_probe.status", "skipped_no_cuda"),
    ("kda_probe.sha256", "not-a-digest"), ("kda_probe.shape", []),
    ("kda_probe.probe_s", -1), ("kda_probe.compiled", []),
    ("kda_probe.compiled", ["gram_fwd_strict"]),
])
def test_malformed_workload_result_refuses(path, value):
    receipt = _evidence_fixture()
    target = receipt["workload"]
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match=keys[-1]):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field", [
    "index", "unix", "phase", "scan_duration_s", "allocated_bytes",
    "apparent_bytes", "files", "dirs", "errors",
])
def test_missing_peak_sample_field_refuses_despite_valid_summary(field):
    receipt = _evidence_fixture()
    del receipt["measurement"]["samples"][1][field]
    with pytest.raises(ValueError, match=field):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field,value", [
    ("index", 2), ("index", True), ("unix", float("nan")),
    ("unix", 99.0), ("unix", 101.0), ("scan_duration_s", -0.01),
    ("scan_duration_s", 1.0), ("phase", "final"),
    ("allocated_bytes", "1024"), ("allocated_bytes", -1),
    ("allocated_bytes", True), ("apparent_bytes", None),
    ("files", 1.5), ("dirs", False), ("errors", ["scan failed"]),
])
def test_malformed_sample_refuses_despite_valid_summary(field, value):
    receipt = _evidence_fixture()
    receipt["measurement"]["samples"][1][field] = value
    with pytest.raises(ValueError, match=field):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field", [
    "sample_interval_s", "sample_count", "row_started_unix",
    "row_finished_unix", "child_exitcode", "gaps", "errors",
    "valid", "incomplete_scan", "peak_allocated_bytes",
    "peak_sample_index", "final_allocated_bytes",
])
def test_missing_measurement_field_refuses(field):
    receipt = _evidence_fixture()
    del receipt["measurement"][field]
    with pytest.raises(ValueError, match=field):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field,value", [
    ("samples", [{}]), ("samples", [None]),
    ("sample_count", 1), ("sample_count", True),
    ("sample_interval_s", 0), ("sample_interval_s", float("inf")),
    ("row_started_unix", 100.0), ("row_finished_unix", 100.6),
    ("row_finished_unix", 99.0), ("child_exitcode", False),
    ("gaps", [{}]), ("errors", ""), ("valid", 1),
    ("incomplete_scan", 0), ("peak_allocated_bytes", 2048),
    ("peak_allocated_bytes", 512), ("peak_sample_index", 0),
    ("peak_sample_index", True), ("final_allocated_bytes", 1024),
])
def test_inconsistent_measurement_refuses(field, value):
    receipt = _evidence_fixture()
    receipt["measurement"][field] = value
    with pytest.raises(ValueError):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


def test_recorded_gap_cannot_be_hidden_by_empty_summary():
    receipt = _evidence_fixture()
    receipt["measurement"]["sample_interval_s"] = 0.1
    with pytest.raises(ValueError, match="gap"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("boundary", [0, -1])
def test_lifecycle_boundary_phase_is_required(boundary):
    receipt = _evidence_fixture()
    receipt["measurement"]["samples"][boundary]["phase"] = "periodic"
    with pytest.raises(ValueError, match="phase"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


def test_quantum_requires_its_own_results_not_only_served_compile():
    receipt = _quantum_fixture()
    receipt["workload"] = {"served_compile": receipt["workload"]["served_compile"]}
    with pytest.raises(ValueError, match="GPU workload.*row"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


def test_quantum_cpu_request_disagrees_with_cuda_compile_results():
    receipt = _quantum_fixture()
    receipt["device"] = "cpu"
    with pytest.raises(ValueError, match="GPU workload.*device"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


def test_quantum_checkpoint_must_cover_declared_layer():
    receipt = _quantum_fixture()
    receipt["workload"]["checkpoints"] = [0, 2]
    with pytest.raises(ValueError, match="no usable checkpoint"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


def test_distinct_cuda_indices_are_permitted_with_complete_results():
    receipt = _quantum_fixture()
    receipt["device"] = "cuda:1"
    receipt["workload"]["served_compile"]["device"] = "cuda:0"
    headroom = 4341760
    assert (peak_mod.derive_cache_ceiling_from_receipt(
        receipt, headroom_bytes=headroom)
        == peak_mod.derive_cache_ceiling(1024, headroom_bytes=headroom))


@pytest.mark.parametrize("make_receipt", [_evidence_fixture, _quantum_fixture])
@pytest.mark.parametrize("headroom", [0, peak_mod.GIB_BYTES - 1024, peak_mod.GIB_BYTES - 1023])
def test_complete_serialized_results_preserve_formula(make_receipt, headroom):
    receipt = make_receipt()
    assert peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=headroom) == (
        peak_mod.derive_cache_ceiling(1024, headroom_bytes=headroom))


def test_equal_maxima_keep_sampler_first_peak_index():
    receipt = _evidence_fixture()
    receipt["measurement"]["samples"][2]["allocated_bytes"] = 1024
    receipt["measurement"]["final_allocated_bytes"] = 1024
    assert peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)["peak_bytes"] == 1024
    receipt["measurement"]["peak_sample_index"] = 2
    with pytest.raises(ValueError, match="peak_sample_index"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field", ["row", "layer", "units", "formats", "n_probes", "seed_base", "checkpoints", "windows", "served_compile"])
def test_quantum_missing_completion_result_names_field(field):
    receipt = _quantum_fixture()
    del receipt["workload"][field]
    with pytest.raises(ValueError, match=field):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("field,value", [
    ("units", []), ("units", ["model.layers.1.proj", "model.layers.1.proj"]),
    ("formats", [None]), ("layer", True), ("n_probes", 0),
    ("seed_base", -1), ("checkpoints", [2, 0]), ("windows", 0),
    ("served_compile", {"device": "cuda"}),
])
def test_quantum_malformed_completion_result_refuses(field, value):
    receipt = _quantum_fixture()
    receipt["workload"][field] = value
    with pytest.raises(ValueError):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)


@pytest.mark.parametrize("schema", [None, "unknown"])
def test_unknown_workload_producer_refuses(schema):
    receipt = _evidence_fixture()
    receipt["schema"] = schema
    with pytest.raises(ValueError, match="schema"):
        peak_mod.derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)

