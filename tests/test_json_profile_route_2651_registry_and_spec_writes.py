import json
from functools import partial

from prismaquant import digests
from prismaquant.artifact_registry import ArtifactRecord, ArtifactRegistry
from prismaquant.pipeline import PipelineSpec, write_pipeline_spec
from prismaquant.prepriced_cost import _write_report


def test_listed_calls_use_direct_ascii_indent2_lax(tmp_path, monkeypatch):
    registry = ArtifactRegistry(tmp_path / "registry.json")
    record = ArtifactRecord(
        record_id="test", model_path="model", artifact_path=None,
        layer_config_sha="a" * 64, layer_config_path=None, target_bpp=4.5,
        achieved_bpp=4.5, format_histogram={}, ppl_wikitext=None,
        ppl_mmlu_acc=None, end_kl=float("inf"), eval_meta={}, created_at="",
        notes="caf\u00e9",
    )
    spec = PipelineSpec(id="caf\u00e9", stages=())
    spec_path = tmp_path / "spec.json"
    source = tmp_path / "cost.pkl"
    source.touch()
    report = {"path": str(source), "note": "caf\u00e9", "cost": float("nan")}
    report_path = tmp_path / "receipt.json"
    real_text = digests.JsonProfile.text

    def mark(profile, value):
        text = real_text(profile, value)
        return " " + text if profile is digests.DIRECT_ASCII_INDENT2_LAX else text

    for path, write, payload in (
        (registry.path, partial(registry._write, [record]), {"records": [record.to_dict()]}),
        (spec_path, partial(write_pipeline_spec, spec, spec_path), spec.to_dict()),
        (report_path, partial(_write_report, report_path, report), report),
    ):
        expected = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
        write()
        assert path.read_bytes() == expected, f"Keep the original bytes in {path.name}."
        with monkeypatch.context() as patch:
            patch.setattr(digests.JsonProfile, "text", mark)
            write()
        assert path.read_bytes() == b" " + expected, f"Publish the profile text in {path.name}."
