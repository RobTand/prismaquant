"""Read-only external cost intake; workloads run through PrismaBuild."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


@pytest.fixture
def bf16_source(tmp_path):
    import torch
    from safetensors.torch import save_file
    from prismaquant.streamed_model_identity import source_model_decode_contract

    root = tmp_path / "exact source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    save_file({"model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4, dtype=torch.bfloat16)},
              str(root / "model.safetensors"))
    assert source_model_decode_contract(root) is None
    return str(root)


def _api():
    from prismaquant import prepriced_cost
    return prepriced_cost


def _payload(model, mode="local"):
    return {
        "costs": {"model.layers.0.self_attn.o_proj": {
            "BF16": {"predicted_dloss": 0.0},
        }},
        "formats": ["BF16"],
        "provenance": {"cost_mode": mode},
        "meta": {"model": model},
    }


def _write(tmp_path, payload):
    path = tmp_path / "external table with spaces.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def _validate(model, path, **kwargs):
    return _api().validate_prepriced_cost(
        path, cost_mode=kwargs.pop("cost_mode", "local"),
        model=model, **kwargs)


def _tessera_payload(model):
    from prismaquant.cost_currency import RENDER_SCORE_COST_MODE
    from prismaquant.tessera_campaign_table import (CURRENCY)
    payload = _payload(model, RENDER_SCORE_COST_MODE)
    fmt = "TESSERA_E4M3_K1_R1024"
    payload["formats"].append(fmt)
    payload["costs"]["model.layers.0.self_attn.o_proj"][fmt] = {
        "output_mse": 1e-4, "output_mse_measured": True,
        "currency": CURRENCY,
        "hessian_identity": {
            "supplied": True, "text_sha": "fixture-draw",
            "token_count": 4, "kwarg": "hessian",
        },
    }
    return payload


@pytest.mark.parametrize("mode", ["local", "production-render-score", "aura", "production-render"])
def test_valid_input_reports_exact_bytes_without_modifying_file(tmp_path, mode, bf16_source):
    from prismaquant.cost_currency import COST_MODE_OBJECTIVE_CURRENCY
    path = _write(tmp_path, _payload(bf16_source, mode))
    original = path.read_bytes()
    report = _validate(bf16_source, path, cost_mode=mode)
    assert report["path"] == str(path.resolve())
    assert report["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["cost_mode"] == mode
    assert report["currency"]["expected_currency"] == COST_MODE_OBJECTIVE_CURRENCY[mode]
    assert report["formats"] == ["BF16"]
    assert report["usable_rows"] == 1
    assert report["model_binding"]["kind"] == "exact_model_reference"
    assert report["model_binding"]["checkpoint_content_attested"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("condition", ["missing", "directory", "invalid-pickle", "schema", "empty", "errors-only"])
def test_invalid_input_refuses_with_path(tmp_path, condition, bf16_source):
    path = _write(tmp_path, _payload(bf16_source))
    if condition == "missing":
        path = tmp_path / "missing.pkl"
    elif condition == "directory":
        path = tmp_path
    elif condition == "invalid-pickle":
        path.write_bytes(b"this is not a pickle")
    elif condition == "schema":
        path.write_bytes(pickle.dumps({"costs": {"unit": {"BF16": {}}}}))
    elif condition == "empty":
        payload = _payload(bf16_source)
        payload["costs"] = {}
        path.write_bytes(pickle.dumps(payload))
    elif condition == "errors-only":
        payload = _payload(bf16_source)
        payload["costs"] = {"unit": {"BF16": {"error": "not measured"}}}
        path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match=path.name):
        _validate(bf16_source, path)


@pytest.mark.parametrize("stamped,requested", [(None, "local"), ("unowned-mode", "local"), ("aura", "local"), ("local", "unowned-mode"), ("production-render", "production-render-score")])
def test_mode_must_be_explicit_known_and_exact(tmp_path, stamped, requested, bf16_source):
    payload = _payload(bf16_source, stamped)
    if stamped is None:
        payload.pop("provenance")
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="cost_mode|COST_MODE"):
        _validate(bf16_source, path, cost_mode=requested)


@pytest.mark.parametrize("location", ["meta", "provenance", "shard", "nested-shard", "baseline-meta", "baseline-shard"])
def test_owned_model_reference_locations_are_reported(tmp_path, location, bf16_source):
    payload = _payload(bf16_source)
    model = payload["meta"].pop("model")
    if location == "shard":
        payload["meta"]["shards"] = [{"model": model}]
    elif location == "nested-shard":
        payload["meta"]["shards"] = [{"incremental_shard": {"model": model}}]
    elif location == "baseline-meta":
        payload["meta"]["baseline_meta"] = {"model": model}
    elif location == "baseline-shard":
        payload["meta"]["baseline_meta"] = {"shards": [{"model": model}]}
    else:
        payload[location]["model"] = model
    report = _validate(bf16_source, _write(tmp_path, payload))
    assert report["model_binding"]["model"] == model
    assert report["model_binding"]["fields"]


@pytest.mark.parametrize("condition", ["missing", "different", "basename", "conflict", "malformed"])
def test_model_reference_missing_or_conflicting_refuses(tmp_path, condition, bf16_source):
    payload = _payload(bf16_source)
    if condition == "missing":
        payload["meta"].pop("model")
    elif condition == "different":
        payload["meta"]["model"] = "/other/model"
    elif condition == "basename":
        payload["meta"]["model"] = "exact source"
    elif condition == "conflict":
        payload["provenance"]["model"] = "/other/model"
    else:
        payload["meta"]["model"] = {"source": bf16_source}
    with pytest.raises(ValueError, match="model"):
        _validate(bf16_source, _write(tmp_path, payload))


def test_format_report_uses_actual_rows_without_fabricating_missing_roster(tmp_path, bf16_source):
    payload = _payload(bf16_source)
    payload.pop("formats")
    payload["costs"]["unit"] = {"NVFP4": {"output_mse": 1e-3}}
    report = _validate(bf16_source, _write(tmp_path, payload))
    assert report["formats"] == ["BF16", "NVFP4"]
    assert report["declared_formats"] == []


def test_tessera_currency_and_hessian_owner_reports_are_preserved(tmp_path, bf16_source):
    from prismaquant.cost_currency import require_run_currency
    from prismaquant.tessera_menu import assert_uniform_hessian_identity
    payload = _tessera_payload(bf16_source)
    report = _validate(bf16_source, _write(tmp_path, payload), cost_mode=payload["provenance"]["cost_mode"])
    assert report["currency"] == require_run_currency(payload)
    assert report["tessera_hessian_identity"] == assert_uniform_hessian_identity(payload["costs"])


def test_tessera_currency_mismatch_refuses_even_with_matching_run_mode(tmp_path, bf16_source):
    payload = _tessera_payload(bf16_source)
    payload["provenance"]["cost_mode"] = "aura"
    with pytest.raises(ValueError, match="currency|aura"):
        _validate(bf16_source, _write(tmp_path, payload), cost_mode="aura")


def test_mixed_tessera_hessian_identities_refuse(tmp_path, bf16_source):
    payload = _tessera_payload(bf16_source)
    other = copy.deepcopy(payload["costs"]["model.layers.0.self_attn.o_proj"])
    other["TESSERA_E4M3_K1_R1024"]["hessian_identity"]["text_sha"] = "other-draw"
    payload["costs"]["other.unit"] = other
    with pytest.raises(ValueError, match="Hessian identities"):
        _validate(bf16_source, _write(tmp_path, payload), cost_mode=payload["provenance"]["cost_mode"])


def test_unstamped_tessera_hessian_remains_no_claim_not_matching_identity(tmp_path, bf16_source):
    payload = _tessera_payload(bf16_source)
    payload["costs"]["model.layers.0.self_attn.o_proj"]["TESSERA_E4M3_K1_R1024"].pop("hessian_identity")
    report = _validate(bf16_source, _write(tmp_path, payload), cost_mode=payload["provenance"]["cost_mode"])
    assert report["tessera_hessian_identity"]["unstamped_rows"] == 1
    assert report["tessera_hessian_identity"]["supplied"] is None


@pytest.mark.parametrize("manifest_valid", [True, False])
def test_research_assembly_cannot_be_accepted_by_pipeline_override(tmp_path, manifest_valid, bf16_source):
    from prismaquant.research_cost_acceptance import RESEARCH_COST_MANIFEST_SCHEMA, RESEARCH_COST_PROVENANCE
    payload = _payload(bf16_source)
    payload["provenance"]["cost_provenance"] = RESEARCH_COST_PROVENANCE
    if manifest_valid:
        payload["provenance"]["research_cost_manifest"] = {
            "schema": RESEARCH_COST_MANIFEST_SCHEMA, "assembled_row_count": len(payload["costs"]),
        }
    with pytest.raises(ValueError, match="research"):
        _validate(bf16_source, _write(tmp_path, payload))


def test_expected_sha_refuses_changed_bytes(tmp_path, bf16_source):
    path = _write(tmp_path, _payload(bf16_source))
    with pytest.raises(ValueError, match="sha256"):
        _validate(bf16_source, path, expected_sha256="0" * 64)


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "prismaquant.prepriced_cost", *map(str, args)],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False,
        timeout=60)


def test_cli_report_and_reverification_bind_original_input_bytes(tmp_path, bf16_source):
    path = _write(tmp_path, _payload(bf16_source))
    report = tmp_path / "local receipt.json"
    before = path.read_bytes()
    result = _cli("--path", path, "--cost-mode", "local", "--model", bf16_source, "--report", report)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["sha256"] == hashlib.sha256(before).hexdigest()
    assert _cli("--verify-report", report).returncode == 0
    path.write_bytes(before + b"changed after preflight")
    refused = _cli("--verify-report", report)
    assert refused.returncode != 0
    assert "sha256" in refused.stderr


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_cli_report_never_overwrites_supplied_input(tmp_path, alias, bf16_source):
    path = _write(tmp_path, _payload(bf16_source))
    before = path.read_bytes()
    report = path if alias == "same" else tmp_path / "alias.json"
    if alias == "symlink":
        report.symlink_to(path)
    elif alias == "hardlink":
        report.hardlink_to(path)
    result = _cli("--path", path, "--cost-mode", "local", "--model", bf16_source, "--report", report)
    assert result.returncode != 0
    assert "report" in result.stderr
    assert path.read_bytes() == before


def test_verification_rejects_malformed_receipt(tmp_path):
    report = tmp_path / "receipt.json"
    report.write_text(json.dumps({"path": "relative.pkl", "sha256": "missing"}))
    with pytest.raises(ValueError, match="report"):
        _api().verify_prepriced_cost_report(report)


@pytest.mark.parametrize("alias", ["file", "parent"])
def test_reverification_refuses_original_input_symlink_retarget(tmp_path, alias, bf16_source):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    source = _write(original, _payload(bf16_source))
    other = _write(replacement, _payload(bf16_source, "aura"))
    link = tmp_path / "input-link"
    if alias == "file":
        link.symlink_to(source)
        supplied = link
    else:
        link.symlink_to(original, target_is_directory=True)
        supplied = link / source.name
    report = _validate(bf16_source, supplied)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(report))
    link.unlink()
    link.symlink_to(other if alias == "file" else replacement,
                    target_is_directory=alias == "parent")
    # The old resolved file is unchanged, but the driver's original --costs
    # argument now consumes different bytes. Hashing only report.path misses it.
    with pytest.raises(ValueError, match="path|sha256"):
        _api().verify_prepriced_cost_report(receipt)
