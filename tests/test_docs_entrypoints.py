from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_deleted_prismascout_l3_entrypoint_is_not_live_documented():
    live_paths = [
        ROOT / "docs" / "archive" / "propagated_cost.md",
        ROOT / "examples" / "launchers" / "run-perturbed-x-smoke.sh",
        ROOT
        / "examples"
        / "launchers"
        / "launch-qwen3p6-35b-a3b-prismascout-knee.sh",
        ROOT / "examples" / "launchers" / "bench-l3-speed-kit.sh",
    ]

    for path in live_paths:
        text = path.read_text()
        assert "prismaquant.iterate_perturbed_allocation" not in text
        assert "from prismaquant.iterate_perturbed_allocation" not in text


def test_readme_shipping_menu_matches_pipeline_default():
    readme = (ROOT / "README.md").read_text()
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()

    assert 'FORMATS:=NVFP4,FP8_DYNAMIC,BF16' in script
    assert "export FORMATS=NVFP4,FP8_DYNAMIC,BF16" in readme
    assert "MXFP8_E4M3,BF16" not in readme

