from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.streamed_model_identity import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant import union_production_cache as union
from prismaquant.format_registry import RetiredFormatError


CODE_IDENTITY = {
    "git_commit": "1" * 40,
    "producer_source_sha256": "2" * 64,
}


def _source_identity(*, shard_sha: str = "3" * 64) -> dict:
    value_bearing = {
        "config": {"model_type": "test", "hidden_size": 4},
        "weight_map": {
            "model.a.weight": "model.safetensors",
            "model.b.weight": "model.safetensors",
        },
        "shards": [{
            "path": "/source/model.safetensors",
            "size": 128,
            "sha256": shard_sha,
        }],
        "checkpoint_weight_map": {
            "model.a.weight": "model.safetensors",
            "model.b.weight": "model.safetensors",
        },
    }
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": "/source",
        "resolved_commit": "source-commit",
        "content_sha256": canonical_json_sha256(
            value_bearing, where="test source identity"
        ),
        **value_bearing,
    }


def _metadata(
    entries: dict[tuple[str, str], torch.Tensor],
    assignment: dict[str, str],
    *,
    calib_hash: str = "4" * 32,
    levers: dict | None = None,
    extra: dict | None = None,
    render_scope: str = "assignment",
    requested_formats: list[str] | None = None,
) -> tuple[dict, dict]:
    levers = dict(levers or {"gptq": True, "joint_scale_opt": True})
    records = {
        f"{qname}|{fmt}": {
            "schema": "test.render_score.v1",
            "score": float(index + 1),
        }
        for index, (qname, fmt) in enumerate(sorted(entries))
    }
    formats = requested_formats or sorted({
        fmt for fmt in assignment.values() if fmt != "BF16"
    })
    metadata = {
        "render_scope": render_scope,
        "render_retention": "materialized",
        "requested_formats": formats,
        "requested_entries": len(entries),
        "streaming": True,
        "calib_hash": calib_hash,
        "format_plan_identity_sha256": "5" * 64,
        "render_mechanism_order": [{
            "name": "gptq",
            "operation": "gptq",
            "scope": "linear",
            "gate_metric": "output_mse",
        }],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": len(records),
            "records": records,
        },
    }
    metadata.update(extra or {})
    return metadata, levers


def _make_shard(
    root: Path,
    *,
    shard_id: str,
    entries: dict[tuple[str, str], torch.Tensor],
    assignment: dict[str, str],
    source_identity: dict | None = None,
    settings: dict | None = None,
    code_identity: dict | None = None,
    calib_hash: str = "4" * 32,
    levers: dict | None = None,
    extra_metadata: dict | None = None,
    failed: dict | None = None,
    coverage: dict | None = None,
    render_scope: str = "assignment",
    requested_formats: list[str] | None = None,
    witnesses: dict | None = None,
) -> Path:
    """Build and manifest one shard bundle.

    ``witnesses`` maps ``(qname, fmt)`` to a fused joint global. A witness for
    an entry the shard holds is bound to the file written for it, as its fill
    would; one for another entry names a shard this bundle does not have.
    """
    from prismaquant.file_identity import file_stat_signature
    from prismaquant.production_weight_cache import (
        NVFP4_JOINT_GLOBAL_WITNESS_KEY,
        NVFP4JointGlobalWitness,
        PublishedShard,
        nvfp4_joint_global_witness_section,
    )

    bundle = root / shard_id
    weights = bundle / "weights"
    weights.mkdir(parents=True)
    cache_weights = {}
    activation_max_abs = {}
    for index, ((qname, fmt), tensor) in enumerate(sorted(entries.items())):
        filename = f"render-{index}.pt"
        torch.save(tensor, weights / filename)
        cache_weights[(qname, fmt)] = filename
        activation_max_abs[qname] = float(index + 2)
    extra = dict(extra_metadata or {})
    if witnesses:
        bound = {}
        for key, joint_global in witnesses.items():
            filename = cache_weights.get(key, "elsewhere.pt")
            signature = (
                file_stat_signature((weights / filename).lstat())
                if key in cache_weights else (0, 0, 0, 0, 0)
            )
            bound[key] = NVFP4JointGlobalWitness(
                joint_global, PublishedShard(filename, signature))
        extra[NVFP4_JOINT_GLOBAL_WITNESS_KEY] = (
            nvfp4_joint_global_witness_section(bound))
    metadata, resolved_levers = _metadata(
        entries,
        assignment,
        calib_hash=calib_hash,
        levers=levers,
        extra=extra,
        render_scope=render_scope,
        requested_formats=requested_formats,
    )
    cache = ProductionWeightCache(
        weights=cache_weights,
        levers=resolved_levers,
        activation_max_abs=activation_max_abs,
        failed=failed or {},
        cache_dir="weights",
        metadata=metadata,
    )
    cache_path = bundle / "cache.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = bundle / "shard_manifest.json"
    coverage_args = (
        {"assignment": assignment}
        if coverage is None
        else {"coverage": coverage}
    )
    union.create_shard_manifest(
        cache_path=cache_path,
        cache_dir=weights,
        manifest_path=manifest,
        shard_id=shard_id,
        source_model_identity=source_identity or _source_identity(),
        settings=settings or {
            "schema": "test.union_settings.v1",
            "max_act_rows": 512,
            "dtype": "bf16",
        },
        code_identity=code_identity or CODE_IDENTITY,
        **coverage_args,
    )
    return manifest


@pytest.fixture(autouse=True)
def _stable_current_code(monkeypatch):
    monkeypatch.setattr(
        union, "_current_code_identity", lambda: dict(CODE_IDENTITY)
    )


def test_exact_union_builds_portable_verifiable_bundle(tmp_path):
    assignment = {
        "model.a": "NVFP4",
        "model.b": "FP8_E4M3",
        "model.fixed": "BF16",
    }
    shard_a = _make_shard(
        tmp_path,
        shard_id="sparky",
        entries={
            ("model.a", "NVFP4"): torch.arange(8, dtype=torch.bfloat16),
        },
        assignment=assignment,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="sparklina",
        entries={
            ("model.b", "FP8_E4M3"): torch.arange(6, dtype=torch.bfloat16),
        },
        assignment=assignment,
    )

    output = tmp_path / "union"
    union.union_shard_manifests(
        [shard_a, shard_b], assignment=assignment, output_bundle=output
    )
    payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment
    )

    assert payload["entries"] == 2
    assert set(cache.weights) == {
        ("model.a", "NVFP4"),
        ("model.b", "FP8_E4M3"),
    }
    assert cache.verify_files()["missing"] == []
    assert cache.metadata["requested_entries"] == 2
    assert cache.metadata["render_scores"]["entries"] == 2
    assert cache.metadata["exact_union"]["schema"] == union.UNION_METADATA_SCHEMA
    assert sorted(path.name for path in (output / "weights").iterdir()) == sorted(
        Path(value).name for value in cache.weights.values()
    )

    # Input order cannot change the canonical semantic cache.
    output_2 = tmp_path / "union-2"
    union.union_shard_manifests(
        [shard_b, shard_a], assignment=assignment, output_bundle=output_2
    )
    _payload_2, cache_2 = union.verify_union_manifest(
        output_2 / "union_manifest.json", assignment=assignment
    )
    cache.relocate("weights")
    cache_2.relocate("weights")
    assert cache.__dict__ == cache_2.__dict__


def test_union_rejects_overlapping_cache_keys(tmp_path):
    assignment = {"model.a": "NVFP4"}
    entries = {
        ("model.a", "NVFP4"): torch.arange(4, dtype=torch.bfloat16),
    }
    shard_a = _make_shard(
        tmp_path, shard_id="a", entries=entries, assignment=assignment
    )
    shard_b = _make_shard(
        tmp_path, shard_id="b", entries=entries, assignment=assignment
    )

    with pytest.raises(ValueError, match="shards overlap"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=assignment,
            output_bundle=tmp_path / "union",
        )


@pytest.mark.parametrize("identity_axis", [
    "source", "calibration", "code", "settings", "render",
])
def test_union_rejects_every_shared_identity_axis(tmp_path, identity_axis):
    assignment = {"model.a": "NVFP4", "model.b": "FP8_E4M3"}
    common = {
        "source_identity": _source_identity(),
        "settings": {"schema": "test.settings.v1", "max_act_rows": 512},
        "code_identity": CODE_IDENTITY,
        "calib_hash": "4" * 32,
        "levers": {"gptq": True},
    }
    changed = copy.deepcopy(common)
    if identity_axis == "source":
        changed["source_identity"] = _source_identity(shard_sha="6" * 64)
    elif identity_axis == "calibration":
        changed["calib_hash"] = "7" * 32
    elif identity_axis == "code":
        changed["code_identity"] = {
            "git_commit": "8" * 40,
            "producer_source_sha256": "9" * 64,
        }
    elif identity_axis == "settings":
        changed["settings"]["max_act_rows"] = 1024
    else:
        changed["levers"] = {"gptq": False}

    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
        **common,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.ones(5)},
        assignment=assignment,
        **changed,
    )

    with pytest.raises(ValueError, match="campaign.*identity differs"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=assignment,
            output_bundle=tmp_path / "union",
            require_current_code=False,
        )


def test_union_rejects_missing_or_modified_backing_file(tmp_path):
    assignment = {"model.a": "NVFP4", "model.b": "FP8_E4M3"}
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.ones(5)},
        assignment=assignment,
    )
    payload = json.loads(shard_b.read_text())["payload"]
    backing = shard_b.parent / payload["cache_dir"] / payload["backing_files"][0]["path"]
    backing.unlink()

    with pytest.raises(ValueError, match="backing file is missing"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=assignment,
            output_bundle=tmp_path / "union",
        )


def test_union_rejects_incomplete_assignment_coverage(tmp_path):
    assignment = {
        "model.a": "NVFP4",
        "model.b": "FP8_E4M3",
        "model.c": "NVFP4",
    }
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.ones(5)},
        assignment=assignment,
    )

    with pytest.raises(ValueError, match="missing assignment entries"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=assignment,
            output_bundle=tmp_path / "union",
        )


def test_manifest_rejects_failed_or_out_of_assignment_cache(tmp_path):
    assignment = {"model.a": "NVFP4"}
    with pytest.raises(ValueError, match="outside the exact assignment"):
        _make_shard(
            tmp_path,
            shard_id="extra",
            entries={("model.extra", "NVFP4"): torch.ones(3)},
            assignment=assignment,
        )

    with pytest.raises(ValueError, match="failed renders"):
        _make_shard(
            tmp_path,
            shard_id="failed",
            entries={("model.a", "NVFP4"): torch.ones(3)},
            assignment=assignment,
            failed={("model.a", "NVFP4"): "render failed"},
        )


def test_union_rejects_unknown_differing_metadata(tmp_path):
    assignment = {"model.a": "NVFP4", "model.b": "FP8_E4M3"}
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
        extra_metadata={"future_render_contract": {"version": 1}},
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.ones(5)},
        assignment=assignment,
        extra_metadata={"future_render_contract": {"version": 2}},
    )

    with pytest.raises(ValueError, match="future_render_contract.*differs"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=assignment,
            output_bundle=tmp_path / "union",
        )


def test_union_refuses_a_retired_codebook_rung(tmp_path):
    # The retired codebook lane, archived 2026-09-25, #1304: a stale shard
    # that names a codebook rung refuses instead of being merged or dropped.
    assignment = {"model.a": "NVFP4_CB_K16"}
    with pytest.raises(RetiredFormatError, match="gridbook_lane"):
        _make_shard(
            tmp_path,
            shard_id="cb",
            entries={("model.a", "NVFP4_CB_K16"): torch.ones(3)},
            assignment=assignment,
        )


def test_union_and_verify_cli_round_trip(tmp_path, capsys):
    assignment = {"model.a": "NVFP4", "model.b": "FP8_E4M3"}
    assignment_path = tmp_path / "layer_config.json"
    assignment_path.write_text(json.dumps(assignment), encoding="utf-8")
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.ones(5)},
        assignment=assignment,
    )
    output = tmp_path / "cli-union"

    assert union.main([
        "union",
        "--manifest", str(shard_a),
        "--manifest", str(shard_b),
        "--assignment", str(assignment_path),
        "--output-dir", str(output),
    ]) == 0
    assert union.main([
        "verify",
        "--manifest", str(output / "union_manifest.json"),
        "--assignment", str(assignment_path),
    ]) == 0
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [summary["entries"] for summary in summaries] == [2, 2]


def _write_stripe_plan(root: Path, stripes: list[list[str]]) -> Path:
    rows = []
    for index, names in enumerate(stripes):
        path = root / f"stripe-{index:02d}.qnames.txt"
        path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
        rows.append({
            "index": index,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qnames": len(names),
            "groups": [f"test:{index}"],
            "estimated_work": len(names) * 10,
            "parameters": len(names) * 5,
        })
    plan = {
        "schema": "prismaquant.production_cache_stripe_plan.v1",
        "profile": "test",
        "probe_sha256": "a" * 64,
        "formats": ["NVFP4", "FP8_E4M3"],
        "n_stripes": len(rows),
        "qnames": sum(len(names) for names in stripes),
        "stripes": rows,
    }
    path = root / "stripe-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_format_menu_stripe_plan_union_is_exact_cartesian_product(tmp_path):
    plan_path = _write_stripe_plan(tmp_path, [["model.a"], ["model.b"]])
    coverage = union.load_stripe_plan_coverage(plan_path)
    assignment = {"model.a": "NVFP4", "model.b": "NVFP4"}
    shard_a = _make_shard(
        tmp_path,
        shard_id="menu-a",
        entries={
            ("model.a", "FP8_E4M3"): torch.ones(2),
            ("model.a", "NVFP4"): torch.ones(3),
        },
        assignment=assignment,
        coverage=coverage,
        render_scope="format-menu",
        requested_formats=["FP8_E4M3", "NVFP4"],
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="menu-b",
        entries={
            ("model.b", "FP8_E4M3"): torch.ones(4),
            ("model.b", "NVFP4"): torch.ones(5),
        },
        assignment=assignment,
        coverage=coverage,
        render_scope="format-menu",
        requested_formats=["FP8_E4M3", "NVFP4"],
    )
    output = tmp_path / "menu-union"
    union.union_shard_manifests(
        [shard_a, shard_b], coverage=coverage, output_bundle=output
    )
    payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", coverage=coverage
    )
    assert payload["coverage_mode"] == "format-menu"
    assert payload["entries"] == 4
    assert set(cache.weights) == {
        (name, fmt)
        for name in ("model.a", "model.b")
        for fmt in ("FP8_E4M3", "NVFP4")
    }


def test_format_menu_union_rejects_one_missing_pair(tmp_path):
    coverage = union.format_menu_coverage(
        ["model.a", "model.b"], ["NVFP4", "FP8_E4M3"]
    )
    assignment = {"model.a": "NVFP4", "model.b": "NVFP4"}
    common = {
        "assignment": assignment,
        "coverage": coverage,
        "render_scope": "format-menu",
        "requested_formats": ["FP8_E4M3", "NVFP4"],
    }
    shard_a = _make_shard(
        tmp_path,
        shard_id="missing-a",
        entries={
            ("model.a", "FP8_E4M3"): torch.ones(2),
            ("model.a", "NVFP4"): torch.ones(3),
        },
        **common,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="missing-b",
        entries={("model.b", "NVFP4"): torch.ones(5)},
        **common,
    )
    with pytest.raises(ValueError, match="missing expected format-menu coverage"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            coverage=coverage,
            output_bundle=tmp_path / "incomplete-menu",
        )


def test_stripe_plan_coverage_rejects_tampered_qname_file(tmp_path):
    plan_path = _write_stripe_plan(tmp_path, [["model.a"], ["model.b"]])
    (tmp_path / "stripe-01.qnames.txt").write_text(
        "model.changed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="qname file SHA-256 mismatch"):
        union.load_stripe_plan_coverage(plan_path)


def test_campaign_scale_coverage_is_exactly_505_by_two():
    coverage = union.format_menu_coverage(
        [f"model.linear.{index}" for index in range(505)],
        ["NVFP4", "FP8_E4M3"],
    )
    assert len(coverage["qnames"]) == 505
    assert len(coverage["pairs"]) == 1010


def test_union_preserves_complete_mtp_provenance_from_owning_shard(tmp_path):
    assignment = {"model.body": "NVFP4", "mtp.fc": "NVFP4"}
    body = _make_shard(
        tmp_path,
        shard_id="body",
        entries={("model.body", "NVFP4"): torch.ones(2)},
        assignment=assignment,
    )
    mtp_metadata = {
        "schema": union.MTP_RENDER_METADATA_SCHEMA,
        "scope": "assignment",
        "entries": 1,
        "qnames": ["mtp.fc"],
        "formats": ["NVFP4"],
        "formats_by_qname": {"mtp.fc": ["NVFP4"]},
        "source_prefix": "model.mtp.",
        "source_tensor_count": 7,
        "activation_rows": {"mtp.fc": 128},
        "max_act_rows": 512,
    }
    mtp = _make_shard(
        tmp_path,
        shard_id="mtp",
        entries={("mtp.fc", "NVFP4"): torch.ones(3)},
        assignment=assignment,
        extra_metadata={"mtp_render": mtp_metadata},
    )
    output = tmp_path / "mtp-union"
    union.union_shard_manifests(
        [body, mtp], assignment=assignment, output_bundle=output
    )
    _payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment
    )
    assert cache.metadata["mtp_render"] == mtp_metadata


def test_format_menu_cli_uses_verified_stripe_plan(tmp_path, capsys):
    plan_path = _write_stripe_plan(tmp_path, [["model.a"], ["model.b"]])
    coverage = union.load_stripe_plan_coverage(plan_path)
    assignment = {"model.a": "NVFP4", "model.b": "NVFP4"}
    common = {
        "assignment": assignment,
        "coverage": coverage,
        "render_scope": "format-menu",
        "requested_formats": ["FP8_E4M3", "NVFP4"],
    }
    shard_a = _make_shard(
        tmp_path,
        shard_id="cli-menu-a",
        entries={("model.a", fmt): torch.ones(2) for fmt in coverage["formats"]},
        **common,
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="cli-menu-b",
        entries={("model.b", fmt): torch.ones(3) for fmt in coverage["formats"]},
        **common,
    )
    output = tmp_path / "cli-menu-union"
    assert union.main([
        "union",
        "--manifest", str(shard_a),
        "--manifest", str(shard_b),
        "--stripe-plan", str(plan_path),
        "--output-dir", str(output),
    ]) == 0
    assert union.main([
        "verify",
        "--manifest", str(output / "union_manifest.json"),
        "--stripe-plan", str(plan_path),
    ]) == 0
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["coverage_mode"] for row in summaries] == [
        "format-menu", "format-menu"
    ]


def _bound_witnesses(cache_dir: Path, cache) -> dict:
    """Each witness's joint global, after requiring it to bind its shard here."""
    from prismaquant.production_weight_cache import (
        nvfp4_joint_global_witness_problem,
        nvfp4_joint_global_witnesses,
    )

    bound = {}
    for key, witness in nvfp4_joint_global_witnesses(
        cache.metadata, where="test"
    ).items():
        served = cache.weights[key]
        assert nvfp4_joint_global_witness_problem(
            witness, served=served, path=cache_dir / served) is None, key
        bound[key] = witness.joint_global
    return bound


def test_union_merges_each_shards_own_joint_global_witnesses(tmp_path):
    assignment = {"model.a": "NVFP4", "model.c": "NVFP4"}
    shard_a = _make_shard(
        tmp_path, shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
        witnesses={("model.a", "NVFP4"): 0.25},
    )
    shard_c = _make_shard(
        tmp_path, shard_id="c",
        entries={("model.c", "NVFP4"): torch.ones(5)},
        assignment=assignment,
        witnesses={("model.c", "NVFP4"): 0.5},
    )

    output = tmp_path / "union"
    union.union_shard_manifests(
        [shard_a, shard_c], assignment=assignment, output_bundle=output)
    _payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment)

    # Each witness now binds the union's own verified copy of its shard.
    assert _bound_witnesses(output / "weights", cache) == {
        ("model.a", "NVFP4"): 0.25,
        ("model.c", "NVFP4"): 0.5,
    }


@pytest.mark.parametrize("change", ["chmod", "copy"])
def test_union_export_refusal_names_rebuild_instead_of_unscoped_resume(change, tmp_path):
    assignment = {"model.a": "NVFP4"}
    manifest = _make_shard(
        tmp_path, shard_id="a", assignment=assignment,
        entries={("model.a", "NVFP4"): torch.ones(3)},
        witnesses={("model.a", "NVFP4"): 0.25},
    )
    output = tmp_path / "union"
    union.union_shard_manifests(
        [manifest], assignment=assignment, output_bundle=output)
    _payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment)
    assert not (output / "render_identity.json").exists()
    assert not (output / "weights/render_identity.json").exists()
    shard = output / "weights" / cache.weights[("model.a", "NVFP4")]
    before = shard.read_bytes()
    if change == "copy":
        copied = shard.with_name(shard.name + ".copy")
        shutil.copy2(shard, copied)
        os.replace(copied, shard)
    else:
        mode = shard.stat().st_mode & 0o777
        shard.chmod(0o600 if mode != 0o600 else 0o644)
    assert shard.read_bytes() == before
    cache.get("model.a", "NVFP4")
    with pytest.raises(RuntimeError) as refusal:
        cache.witnessed_nvfp4_joint_global("model.a", "NVFP4", where="export")
    message = str(refusal.value)
    assert "Only resident-fill directories support removing the affected shard" in message
    assert "Rebuild an exact union from its verified input stripes at the final export location" in message
    assert "For a copied cache, export from the original directory" in message


def test_union_refuses_a_shard_witnessing_another_shards_entry(tmp_path):
    assignment = {"model.a": "NVFP4", "model.c": "NVFP4"}
    legacy = _make_shard(
        tmp_path, shard_id="legacy",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
    )
    lender = _make_shard(
        tmp_path, shard_id="lender",
        entries={("model.c", "NVFP4"): torch.ones(5)},
        assignment=assignment,
        witnesses={("model.a", "NVFP4"): 0.25},
    )

    with pytest.raises(ValueError, match="entries it does not hold"):
        union.union_shard_manifests(
            [legacy, lender], assignment=assignment,
            output_bundle=tmp_path / "union")


def test_shard_manifest_refuses_a_witness_its_hashed_file_breaks(tmp_path):
    assignment = {"model.a": "NVFP4"}
    manifest = _make_shard(
        tmp_path, shard_id="a",
        entries={("model.a", "NVFP4"): torch.ones(3)},
        assignment=assignment,
        witnesses={("model.a", "NVFP4"): 0.25},
    )
    shard = manifest.parent / "weights" / "render-0.pt"
    copied = shard.with_name("render-0.pt.copy")
    shutil.copy2(shard, copied)
    os.replace(copied, shard)

    with pytest.raises(ValueError, match="does not match its published file identity"):
        union.create_shard_manifest(
            cache_path=manifest.parent / "cache.pkl",
            cache_dir=manifest.parent / "weights",
            manifest_path=manifest.parent / "again.json",
            shard_id="a",
            source_model_identity=_source_identity(),
            settings={
                "schema": "test.union_settings.v1",
                "max_act_rows": 512,
                "dtype": "bf16",
            },
            code_identity=CODE_IDENTITY,
            assignment=assignment,
        )


@pytest.mark.parametrize("operation", ["manifest", "verify"])
def test_union_boundary_names_a_server_local_device_mismatch(
    operation, tmp_path, monkeypatch,
):
    from prismaquant import streamed_model_identity as identity

    assignment = {"model.a": "NVFP4", "model.c": "NVFP4"}
    manifests = [
        _make_shard(
            tmp_path, shard_id=label,
            entries={(qname, "NVFP4"): torch.ones(3)},
            assignment=assignment, witnesses={(qname, "NVFP4"): 0.25},
        )
        for label, qname in [("a", "model.a"), ("c", "model.c")]
    ]
    output = tmp_path / "union"
    union.union_shard_manifests(
        manifests, assignment=assignment, output_bundle=output)
    original_fingerprint = identity.stat_fingerprint

    def another_device_view(path, value):
        observed = original_fingerprint(path, value)
        return {**observed, "device": observed["device"] + 1}

    monkeypatch.setattr(identity, "stat_fingerprint", another_device_view)
    monkeypatch.setattr(identity, "_mount_filesystem_type", lambda _path: "zfs")

    def check_boundary():
        if operation == "verify":
            return union.verify_union_manifest(
                output / "union_manifest.json", assignment=assignment)
        return union.create_shard_manifest(
            cache_path=manifests[0].parent / "cache.pkl",
            cache_dir=manifests[0].parent / "weights",
            manifest_path=manifests[0].parent / "again.json",
            shard_id="a", source_model_identity=_source_identity(),
            settings={"schema": "test.union_settings.v1", "max_act_rows": 512, "dtype": "bf16"},
            code_identity=CODE_IDENTITY, assignment=assignment,
        )

    with pytest.raises(ValueError, match="device-only mismatch.*NFS client mount"):
        check_boundary()
    # Identical recorded fields through an NFS client retain their admission.
    monkeypatch.setattr(identity, "_mount_filesystem_type", lambda _path: "nfs4")
    check_boundary()


def test_union_rebinds_witnesses_of_transferred_shards_to_its_copies(tmp_path, monkeypatch):
    """A transferred shard's files are other files; its manifest's SHA-256,
    taken while its witnesses bound them, carries the witnesses to the
    union's copies. A copy of the union bundle binds none of them."""
    assignment = {"model.a": "NVFP4", "model.c": "NVFP4"}
    built = tmp_path / "built"
    built.mkdir()
    manifests = [
        _make_shard(
            built, shard_id=shard_id,
            entries={(qname, "NVFP4"): torch.ones(size)},
            assignment=assignment,
            witnesses={(qname, "NVFP4"): value},
        )
        for shard_id, qname, size, value in (
            ("a", "model.a", 3, 0.25), ("c", "model.c", 5, 0.5))
    ]
    transferred = tmp_path / "transferred"
    shutil.copytree(built, transferred)

    output = tmp_path / "union"
    union.union_shard_manifests(
        [transferred / path.relative_to(built) for path in manifests],
        assignment=assignment, output_bundle=output)
    _payload, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment)
    assert _bound_witnesses(output / "weights", cache) == {
        ("model.a", "NVFP4"): 0.25,
        ("model.c", "NVFP4"): 0.5,
    }

    from prismaquant.production_weight_cache import (
        nvfp4_joint_global_witness_problem,
        nvfp4_joint_global_witnesses,
    )

    copied = tmp_path / "copied-union"
    shutil.copytree(output, copied)
    copy_cache = union._load_cache(copied / "production_weight_cache.pkl")
    copy_cache.relocate(copied / "weights")
    for key, witness in nvfp4_joint_global_witnesses(
        copy_cache.metadata, where="test"
    ).items():
        served = copy_cache.weights[key]
        assert "does not match its published file identity" in (
            nvfp4_joint_global_witness_problem(
                witness, served=served, path=copied / "weights" / served))

    import prismaquant.export_native_compressed as exporter
    from prismaquant.native_render_levers import NativeRenderLevers

    monkeypatch.setattr(exporter, "_PRODUCTION_WEIGHT_CACHE", copy_cache)
    monkeypatch.setattr(exporter, "_RENDER_LEVERS", NativeRenderLevers())

    def no_unwitnessed_repack(*args, **kwargs):
        raise AssertionError("a copied union reached the codec before witness refusal")

    monkeypatch.setattr(
        exporter._native_render_codec, "quantize_dequantize_nvfp4", no_unwitnessed_repack)
    with pytest.raises(RuntimeError, match="does not match its published file identity"):
        exporter._pack_production_cached_2d(
            "model.a", "NVFP4", nvfp4_global_real_override=torch.tensor(0.25),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="rebuild the union.*input stripe bundles"):
        union.verify_union_manifest(copied / "union_manifest.json", assignment=assignment)

    # The portable input stripes can build a new union at the final location
    # without re-rendering any weights. Its new copies carry bound witnesses.
    rebuilt = tmp_path / "export-union"
    union.union_shard_manifests(
        [transferred / path.relative_to(built) for path in manifests],
        assignment=assignment, output_bundle=rebuilt,
    )
    _payload, rebuilt_cache = union.verify_union_manifest(
        rebuilt / "union_manifest.json", assignment=assignment,
    )
    assert _bound_witnesses(rebuilt / "weights", rebuilt_cache) == {
        ("model.a", "NVFP4"): 0.25,
        ("model.c", "NVFP4"): 0.5,
    }
