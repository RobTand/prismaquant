from __future__ import annotations

from prismaquant.artifact_registry import (
    ArtifactRecord,
    ArtifactRegistry,
    layer_config_sha256,
)


def _record(
    record_id: str,
    layer_config_sha: str,
    *,
    model_path: str = "/hfcache/Qwen3-4B",
    ppl_wikitext: float = 10.0,
    ppl_mmlu_acc: float = 0.50,
    end_kl: float = 1.0,
) -> ArtifactRecord:
    return ArtifactRecord(
        record_id=record_id,
        model_path=model_path,
        artifact_path=None,
        layer_config_sha=layer_config_sha,
        layer_config_path=None,
        target_bpp=4.5,
        achieved_bpp=4.49,
        format_histogram={"NVFP4": 58, "MXFP8": 181},
        ppl_wikitext=ppl_wikitext,
        ppl_mmlu_acc=ppl_mmlu_acc,
        end_kl=end_kl,
        eval_meta={
            "model_sha": "a" * 64,
            "model_identity_scope": "local_content",
            "eval_split": "test",
            "metric_era": "2026-06-12-test-split",
            "wikitext_split": "test",
            "n_wikitext_tokens": 2048,
            "n_mmlu_questions": 200,
            "calib_seqlen": 512,
            "calib_n_samples": 8,
            "calib_split": "test",
            "calib_seed": 42,
            "wikitext_max_window_tokens": 2048,
            "device": "cpu",
            "dtype": "fp32",
        },
        created_at="2026-10-17T00:00:00+00:00",
    )


def test_registry_round_trip(tmp_path):
    layer_config = {
        "model.layers.0.mlp.down_proj": {
            "bits": 4,
            "group_size": 16,
            "data_type": "nv_fp",
            "act_bits": 4,
            "act_data_type": "nv_fp4_with_static_gs",
        }
    }
    config_path = tmp_path / "layer_config.json"
    config_path.write_text(__import__("json").dumps(layer_config))

    registry_path = tmp_path / "registry.json"
    registry = ArtifactRegistry(registry_path)
    registry.add(_record("abc123", layer_config_sha256(layer_config)))

    reloaded = ArtifactRegistry(registry_path)
    found = reloaded.find_by_layer_config(layer_config)
    assert found is not None
    assert found.record_id == "abc123"
    assert reloaded.find_by_layer_config(str(config_path)).record_id == "abc123"


def test_registry_compare_passes_when_candidate_better(tmp_path):
    registry = ArtifactRegistry(tmp_path / "registry.json")
    baseline_sha = layer_config_sha256({"a": "BF16"})
    candidate_sha = layer_config_sha256({"a": "NVFP4"})
    registry.add(
        _record(
            "baseline",
            baseline_sha,
            ppl_wikitext=10.0,
            ppl_mmlu_acc=0.500,
            end_kl=1.0,
        )
    )
    registry.add(
        _record(
            "candidate",
            candidate_sha,
            ppl_wikitext=10.04,
            ppl_mmlu_acc=0.498,
            end_kl=0.9,
        )
    )

    result = registry.compare("candidate", "baseline")
    assert result["pass"] is True
    assert result["metrics"]["end_kl"]["passed"] is True


def test_registry_compare_fails_on_ppl_regression(tmp_path):
    registry = ArtifactRegistry(tmp_path / "registry.json")
    registry.add(
        _record(
            "baseline",
            layer_config_sha256({"a": "BF16"}),
            ppl_wikitext=10.0,
            ppl_mmlu_acc=0.500,
            end_kl=1.0,
        )
    )
    registry.add(
        _record(
            "candidate",
            layer_config_sha256({"a": "NVFP4"}),
            ppl_wikitext=10.5,
            ppl_mmlu_acc=0.500,
            end_kl=0.9,
        )
    )

    result = registry.compare("candidate", "baseline")
    assert result["pass"] is False
    assert result["metrics"]["ppl_wikitext"]["passed"] is False
