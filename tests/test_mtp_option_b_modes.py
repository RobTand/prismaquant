"""Option B must refuse incompatible searches, not silently publish a frontier."""
from __future__ import annotations

import pytest

from prismaquant import allocator, prefill_frontier
from test_allocator_measured_runtime_cli import _main_fixture, admit_synthetic_table
from test_mtp_option_b_2532 import _tiny_case
from test_pact_allocator_replay import _fixture as _pact_fixture
from test_shape_runtime_prices import checker_sdk_fixture  # noqa: F401


@pytest.mark.parametrize("mode", ["pact", "measured_runtime"])
def test_option_b_refuses_frontier_without_publishing(tmp_path, monkeypatch, mode):
    runtime = tmp_path
    if mode == "pact":
        argv = _pact_fixture(runtime, monkeypatch).argv
        own = []
        reason = "PACT shape-time search"
    else:
        admit_synthetic_table(monkeypatch)
        _, argv = _main_fixture(runtime, shape=(256, 256))
        # Frontier mode owns the SLO grid, not the single-solve SLO.
        argv = argv[1:argv.index("--slo-prefill-p95-ttft-ms")]
        own = ["--slo-grid", "auto"]
        reason = "measured-runtime sweep"
    mtp = tmp_path / "mtp"
    mtp.mkdir()
    model, _, _, _, _, cost, constants = _tiny_case(mtp)
    output = tmp_path / "frontier.json"
    candidates = tmp_path / "candidates"
    with pytest.raises(SystemExit, match=f"--mtp-option-b-dir.*{reason}"):
        prefill_frontier.main([
            *own, "--output", str(output), "--", *argv,
            "--model-override", str(model), "--target-disk-gb", "1",
            "--artifact-overhead-reserve-bytes", "512",
            "--mtp-joint-cost", str(cost), "--mtp-serve-constants", str(constants),
            "--mtp-byte-budget", "1000000", "--mtp-option-b-dir", str(candidates),
        ])
    assert not output.exists()
    assert not output.with_name(output.name + ".assignments").exists()
    assert not candidates.exists()
    assert not (runtime / "layer.json").exists()
    assert not (runtime / "pareto.csv").exists()


@pytest.mark.parametrize("mode", ["pact", "measured_runtime"])
def test_option_b_mode_refusal_precedes_input_loading(tmp_path, mode):
    # Missing inputs are intentional: incompatible modes must refuse before
    # source partitioning or file reads, even without the frontier CLI wrapper.
    extra = (["--pact-shape-table", str(tmp_path / "shape-table.json"),
              "--pact-regime", "2048", "--pact-tensor-parallel", "1"]
             if mode == "pact" else [
                 "--measured-runtime-table", str(tmp_path / "runtime.json"),
                 "--measured-runtime-context", str(tmp_path / "context.json")])

    def unexpected_sweep(_ctx):
        pytest.fail("incompatible sweep callback executed")

    existing = set(tmp_path.iterdir())
    with pytest.raises(SystemExit, match="--mtp-option-b-dir.*not supported"):
        allocator.main([
            "--layer-config", str(tmp_path / "layer.json"),
            "--pareto-csv", str(tmp_path / "pareto.csv"),
            "--probe", str(tmp_path / "probe.pkl"),
            "--costs", str(tmp_path / "costs.pkl"),
            "--target-disk-gb", "1", "--artifact-overhead-reserve-bytes", "512",
            "--mtp-joint-cost", str(tmp_path / "mtp.pkl"),
            "--mtp-byte-budget", "1000000",
            "--mtp-serve-constants", str(tmp_path / "constants.json"),
            "--mtp-option-b-dir", str(tmp_path / "candidates"), *extra,
        ], measured_runtime_sweep=unexpected_sweep)
    assert set(tmp_path.iterdir()) == existing
