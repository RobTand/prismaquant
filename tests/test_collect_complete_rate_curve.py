"""The complete-curve collector seals producer evidence, not loose scalars."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import pytest

from experiments.collect_complete_rate_curve import (
    CAMPAIGN_SCHEMA,
    PLAN_SCHEMA,
    SEMANTIC_ACTIVATION_BY_ROUTE,
    _validate_campaign_menu,
    _expected_row,
    _verify_wire,
    collect,
    main,
    sha256,
)
from experiments.sparse_rate_adaptive import validate_curve
from prismaquant.cost_stage_checkpoint import (
    _load_unit,
    prepare_journal,
    unit_path,
    write_unit,
)
from prismaquant.tessera_campaign import CURRENCY, CampaignAnchor
from tessera.cached_unit import ENCODING_INPUT_SCHEMA
from tessera.encoder_identity import encoder_fixture_id


SHA = "a" * 64
QNAME = "model.layers.3.down_proj"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@dataclass
class Fixture:
    plan_path: Path
    checkpoint: Path
    cache: Path
    out: Path
    plan: dict
    identity: dict
    identity_sha256: str
    state: dict

    def rewrite_state(self) -> None:
        write_unit(
            self.checkpoint.with_name(self.checkpoint.name + ".parts"),
            stage="Tessera campaign",
            qname=QNAME,
            identity_sha256=self.identity_sha256,
            state=self.state,
        )


def make_fixture(tmp_path: Path, family: str, rates: list[int]) -> Fixture:
    from prismaquant import tessera_campaign

    plan_path = tmp_path / "measurement-plan.json"
    checkpoint = tmp_path / "cost.anchors.json"
    cache = tmp_path / "cache"
    (cache / "wire").mkdir(parents=True)

    activation = {
        "TESSERA_BF16_K1": "bf16_unquantized",
        "TESSERA_E2M1_K2": "e2m1_group16_ue4m3_static",
        "TESSERA_E4M3_K1": "fp8_per_token_dynamic",
    }[family]
    body = "tcq" if family == "TESSERA_E2M1_K2" and rates == [896] else "window"
    load_contract = {
        "schema": "prismaquant.streaming_initialization.v1",
        "source_map_sha256": "b" * 64,
        "state_sha256": "c" * 64,
    }
    census = tmp_path / "census.json"
    write_json(census, {
        "model": "fixture-model",
        "attention_implementation": "eager",
        "nsamples": 4,
        "seed": 7,
        "seqlen": 8,
        "model_load_contract": load_contract,
        "unit_shapes": {QNAME: [2, 3]},
    })
    calibration_receipt = {"fit_ids_sha256": "d" * 64, "fit_tokens": 32}
    capture = tmp_path / "capture.json"
    write_json(capture, {"identity": {
        "calibration": calibration_receipt,
        "census_sha256": sha256(census),
        "model_load_contract": load_contract,
    }})
    pin = tmp_path / "pin.json"
    write_json(pin, {"sources": {"encoder": {"commit": "fixture-encoder"}}})
    calibration = {
        "attention_implementation": "eager",
        "capture_manifest": str(capture),
        "capture_manifest_sha256": sha256(capture),
        "census": str(census),
        "census_sha256": sha256(census),
        "hessian": "require",
        "max_act_rows": 4,
        "nsamples": 4,
        "seed": 7,
        "seqlen": 8,
        "streaming": "selected-tensors-v1",
        "tp_degree": 1,
    }
    restriction = {
        "schema": "prismaquant.tessera_campaign_family_restriction.v1",
        "dense": [family],
        "routed_moe": ["TESSERA_E4M3_K1"],
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "frozen_before_measurement",
        "measurement_kind": "measured",
        "currency": CURRENCY,
        "curve_id": f"fixture-{family.lower()}",
        "qname": QNAME,
        "qshape": [2, 3],
        "family": family,
        "activation_contract": activation,
        "menu_mode": "research",
        "legal_rates": rates,
        "audit_regions": {"focus": rates[1:-1]},
        "source_identity": {
            "model": "fixture-model",
            "campaign_source": {
                "base_commit": "fixture",
                "campaign_module_sha256": sha256(Path(tessera_campaign.__file__)),
            },
            "producer": {
                "commit": "fixture-encoder",
                "pin": str(pin),
                "pin_sha256": sha256(pin),
                "source_sha256": "e" * 64,
            },
            "weight_identity": load_contract,
        },
        "calibration_identity": calibration,
        "recipe_identity": {
            "grid": {"TESSERA_BF16_K1": "BF16", "TESSERA_E2M1_K2": "E2M1",
                     "TESSERA_E4M3_K1": "E4M3"}[family],
            "body": body,
            "scale_plane": "lut16" if family == "TESSERA_E2M1_K2" else "channel",
            "activation_contract": activation,
            **({"terminal": body == "tcq"}
               if family == "TESSERA_E2M1_K2" else {}),
        },
        "family_restriction": restriction,
    }
    write_json(plan_path, plan)

    weight = {"algorithm": "sha256.dtype_shape_contiguous.v1", "dtype": "torch.bfloat16",
              "sha256": "f" * 64, "shape": [2, 3]}
    hessian = {"sha256": "1" * 64, "shape": [3, 3]}
    outside_rates = (
        [895] if family == "TESSERA_E2M1_K2" and rates == [896]
        else [rates[0] - 1, rates[-1] + 1]
    )
    unit = {
        "weight": weight,
        "hessian": hessian,
        "input_global_scale": 0.5,
        "menu": [f"{family}_R{rate}" for rate in (*outside_rates, *rates)],
    }
    identity = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "currency": CURRENCY,
        "encoder_source_sha256": "e" * 64,
        "calibration": calibration_receipt,
        "settings": {
            "model": "fixture-model", "family_restriction": restriction,
            "menu_mode": "research", "attention_implementation": "eager",
            "hessian": "require", "max_act_rows": 4, "nsamples": 4, "seed": 7,
            "seqlen": 8, "tp_degree": 1, "streaming": True,
            "streaming_capture_policy": "legacy", "exhaustive_rate_grid": True,
            "rate_band": f"{rates[0]},{rates[-1]}",
            "anchor_budget": len(rates), "max_rounds": 1,
        },
        "family_restriction": {"policy": restriction, "structure_by_unit": {QNAME: "dense"}},
        "units": {QNAME: unit},
    }
    _journal, identity_sha256, _resumed = prepare_journal(
        checkpoint.with_name(checkpoint.name + ".parts"),
        manifest_path=checkpoint,
        stage="Tessera campaign",
        resume=True,
        identity=identity,
        qnames=[QNAME],
    )
    anchors, records = [], {}
    for offset, rate in enumerate(rates):
        typed = _expected_row(plan, rate)
        blob = f"wire-{family}-{rate}".encode()
        filename = f"{QNAME.replace('.', '__')}__{typed['format_name']}.tessera"
        wire_path = cache / "wire" / filename
        wire_path.write_bytes(blob)
        anchor = CampaignAnchor(
            qname=QNAME, family=family, format_name=typed["format_name"],
            body_rate_q256=rate, dloss=1.0 + offset, dloss_stderr=0.0,
            memory_bytes=len(blob), bits_per_param=3.0,
            activation_contract=typed["producer_activation"],
            activation_quantized=typed["activation_quantized"], wire_bytes=len(blob),
            seconds=0.1, hessian_applied=typed["hessian_applied"],
            input_global_scale=(0.5 if typed["uses_static_scale"] else None),
        )
        anchors.append(dict(vars(anchor)))
        records[typed["format_name"]] = {
            "file": filename, "blob_sha256": sha256(wire_path), "blob_bytes": len(blob),
            "identity": {
                "schema": ENCODING_INPUT_SCHEMA, "unit": QNAME, "source": weight,
                "encoder_source_sha256": identity["encoder_source_sha256"],
                "encoder_fixture_id": encoder_fixture_id().hex(),
                "recipe": typed["wire_recipe"],
                "calibration": ({"hessian": hessian} if typed["hessian_applied"] else None),
            },
        }
    state = {"anchors": anchors, "wire_records": records}
    result = Fixture(plan_path, checkpoint, cache, tmp_path / "curve.json", plan,
                     identity, identity_sha256, state)
    result.rewrite_state()
    return result


@pytest.mark.parametrize("family,rates", [
    ("TESSERA_BF16_K1", [832, 960, 1088]),
    ("TESSERA_E2M1_K2", [832, 864, 895]),
    ("TESSERA_E4M3_K1", [832, 1024, 1088]),
])
def test_typed_family_curves_are_accepted_by_frozen_harness(tmp_path, family, rates):
    fixture = make_fixture(tmp_path, family, rates)
    curve = collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)
    assert validate_curve(curve) == (tuple(rates), tuple(curve["values"]))
    assert curve["measurement_kind"] == "measured"
    assert curve["audit_regions"] == fixture.plan["audit_regions"]


def test_e2m1_terminal_is_a_separate_typed_tcq_point(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_E2M1_K2", [896])
    curve = collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)
    assert curve["rates"] == [896]
    assert fixture.state["wire_records"]["TESSERA_E2M1_K2_R896"]["identity"]["recipe"]["body"] == "tcq"


@pytest.mark.parametrize("mutation,match", [
    ("malformed", "not a Tessera format"),
    ("wrong_family", "outside the plan"),
    ("duplicate", "repeats"),
    ("missing_in_band", "in-band menu differs"),
    ("extra_in_band", "in-band menu differs"),
])
def test_campaign_menu_rejects_invalid_or_wrong_in_band_rosters(tmp_path, mutation, match):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    menu = list(fixture.identity["units"][QNAME]["menu"])
    if mutation == "malformed":
        menu.append("BF16")
    elif mutation == "wrong_family":
        menu.append("TESSERA_E4M3_K1_R512")
    elif mutation == "duplicate":
        menu.append(menu[0])
    elif mutation == "missing_in_band":
        menu.remove("TESSERA_BF16_K1_R960")
    else:
        menu.append("TESSERA_BF16_K1_R900")
    with pytest.raises(RuntimeError, match=match):
        _validate_campaign_menu(
            menu,
            settings=fixture.identity["settings"],
            plan=fixture.plan,
            where=fixture.checkpoint,
        )


@pytest.mark.parametrize("field,value,match", [
    ("exhaustive_rate_grid", False, "not exhaustive"),
    ("rate_band", "831,1088", "rate band differs"),
    ("anchor_budget", 2, "anchor budget differs"),
    ("max_rounds", 2, "exactly one round"),
])
def test_campaign_menu_requires_the_recorded_exhaustive_band_contract(
    tmp_path, field, value, match
):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    settings = dict(fixture.identity["settings"])
    settings[field] = value
    with pytest.raises(RuntimeError, match=match):
        _validate_campaign_menu(
            fixture.identity["units"][QNAME]["menu"],
            settings=settings,
            plan=fixture.plan,
            where=fixture.checkpoint,
        )


@pytest.mark.parametrize("field,value", [
    ("family", "TESSERA_E4M3_K1"),
    ("activation_contract", "fp8_e4m3"),
    ("activation_quantized", 1),
    ("hessian_applied", 1),
])
def test_typed_anchor_fields_cannot_be_forged(tmp_path, field, value):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    fixture.state["anchors"][0][field] = value
    fixture.rewrite_state()
    with pytest.raises(RuntimeError):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def test_manifest_and_canonical_shard_identity_are_required(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    manifest = json.loads(fixture.checkpoint.read_text())
    manifest["identity"]["settings"]["seed"] = 8
    write_json(fixture.checkpoint, manifest)
    with pytest.raises(RuntimeError, match="identity digest"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def test_canonical_loader_refuses_a_rebound_unit_shard(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    shard = unit_path(fixture.checkpoint.with_name(fixture.checkpoint.name + ".parts"), QNAME)
    envelope = pickle.loads(shard.read_bytes())
    envelope["identity_sha256"] = "0" * 64
    shard.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(RuntimeError, match="identity_sha256"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


@pytest.mark.parametrize("mutation", ["recipe", "source", "schema", "fixture"])
def test_wire_producer_identity_is_authoritative(tmp_path, mutation):
    fixture = make_fixture(tmp_path, "TESSERA_E4M3_K1", [832, 960, 1088])
    record = fixture.state["wire_records"]["TESSERA_E4M3_K1_R832"]["identity"]
    if mutation == "recipe":
        record["recipe"]["q256"] += 1
    elif mutation == "source":
        record["source"]["sha256"] = "0" * 64
    elif mutation == "schema":
        record["schema"] = "invented"
    else:
        record["encoder_fixture_id"] = "0" * 64
    fixture.rewrite_state()
    with pytest.raises(RuntimeError, match="producer identity"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


@pytest.mark.parametrize("value", [True, 0, float("nan"), float("inf")])
def test_measured_values_must_be_positive_finite_numbers(tmp_path, value):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    fixture.state["anchors"][0]["dloss"] = value
    fixture.rewrite_state()
    with pytest.raises(RuntimeError, match="invalid measured value"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def test_wire_basename_size_hash_and_symlink_are_checked(tmp_path):
    for case in ("basename", "size", "hash", "symlink"):
        root = tmp_path / case
        root.mkdir()
        fixture = make_fixture(root, "TESSERA_BF16_K1", [832, 960, 1088])
        anchor = CampaignAnchor(**fixture.state["anchors"][0])
        record = fixture.state["wire_records"][anchor.format_name]
        path = fixture.cache / "wire" / record["file"]
        if case == "basename":
            record["file"] = "../escape.tessera"
            fixture.rewrite_state()
        elif case == "size":
            record["blob_bytes"] += 1
            fixture.rewrite_state()
        elif case == "hash":
            path.write_bytes(b"changed")
        else:
            target = root / "elsewhere"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        with pytest.raises(RuntimeError):
            collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def test_plan_references_and_campaign_calibration_are_bound(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_BF16_K1", [832, 960, 1088])
    capture = Path(fixture.plan["calibration_identity"]["capture_manifest"])
    payload = json.loads(capture.read_text())
    payload["identity"]["calibration"]["fit_tokens"] += 1
    write_json(capture, payload)
    fixture.plan["calibration_identity"]["capture_manifest_sha256"] = sha256(capture)
    write_json(fixture.plan_path, fixture.plan)
    with pytest.raises(RuntimeError, match="calibration identity"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def _write_audit_region_correction(fixture: Fixture, **changes) -> Path:
    correction = {
        "schema": "prismaquant.complete_rate_audit_region_correction.v1",
        "original_plan_path": str(fixture.plan_path.resolve()),
        "original_plan_sha256": sha256(fixture.plan_path),
        "corrected_audit_regions": {
            label: [rate for rate in region if rate in fixture.plan["legal_rates"]]
            for label, region in fixture.plan["audit_regions"].items()
        },
    }
    correction.update(changes)
    path = fixture.plan_path.with_name("audit-region-correction.json")
    write_json(path, correction)
    return path


def test_audit_region_correction_binds_original_plan_and_retains_empty_labels(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_E4M3_K1", [832])
    fixture.plan["audit_regions"] = {
        "e4m3_around_1024": list(range(1016, 1033)),
        "declared_empty": [],
    }
    write_json(fixture.plan_path, fixture.plan)
    original_bytes = fixture.plan_path.read_bytes()
    correction = _write_audit_region_correction(fixture)

    assert main([
        "--plan", str(fixture.plan_path),
        "--audit-region-correction", str(correction),
        "--checkpoint", str(fixture.checkpoint),
        "--cache-dir", str(fixture.cache),
        "--out", str(fixture.out),
    ]) == 0

    curve = json.loads(fixture.out.read_text())
    assert fixture.plan_path.read_bytes() == original_bytes
    assert curve["measurement_plan"] == {
        "path": str(fixture.plan_path.resolve()),
        "sha256": sha256(fixture.plan_path),
    }
    assert curve["audit_regions"] == {
        "e4m3_around_1024": [],
        "declared_empty": [],
    }
    assert curve["audit_region_correction"] == {
        "path": str(correction.resolve()),
        "sha256": sha256(correction),
    }
    assert validate_curve(curve, min_points=1) == ((832,), (1.0,))


@pytest.mark.parametrize("mutation,match", [
    ({"schema": "invented"}, "unsupported audit-region correction"),
    ({"original_plan_sha256": "0" * 64}, "original plan bytes"),
    ({"corrected_audit_regions": {"e4m3_around_1024": [832]}}, "exact intersection"),
    ({"extra": True}, "exactly"),
])
def test_audit_region_correction_is_a_closed_exact_intersection(tmp_path, mutation, match):
    fixture = make_fixture(tmp_path, "TESSERA_E4M3_K1", [1088])
    fixture.plan["audit_regions"] = {"e4m3_around_1024": list(range(1016, 1033))}
    write_json(fixture.plan_path, fixture.plan)
    correction = _write_audit_region_correction(fixture, **mutation)
    with pytest.raises(RuntimeError, match=match):
        collect(
            fixture.plan_path,
            fixture.checkpoint,
            fixture.cache,
            fixture.out,
            audit_region_correction=correction,
        )


def test_out_of_roster_audit_region_still_fails_without_a_correction(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_E4M3_K1", [832])
    fixture.plan["audit_regions"] = {"e4m3_around_1024": list(range(1016, 1033))}
    write_json(fixture.plan_path, fixture.plan)
    with pytest.raises(RuntimeError, match="subsets of legal_rates"):
        collect(fixture.plan_path, fixture.checkpoint, fixture.cache, fixture.out)


def test_correction_plan_path_and_curve_reference_are_rechecked(tmp_path):
    fixture = make_fixture(tmp_path, "TESSERA_E4M3_K1", [832])
    fixture.plan["audit_regions"] = {"e4m3_around_1024": list(range(1016, 1033))}
    write_json(fixture.plan_path, fixture.plan)
    correction = _write_audit_region_correction(fixture)
    payload = json.loads(correction.read_text())
    payload["original_plan_path"] = str(fixture.checkpoint.resolve())
    write_json(correction, payload)
    with pytest.raises(RuntimeError, match="plan path differs"):
        collect(
            fixture.plan_path,
            fixture.checkpoint,
            fixture.cache,
            fixture.out,
            audit_region_correction=correction,
        )

    correction = _write_audit_region_correction(fixture)
    curve = collect(
        fixture.plan_path,
        fixture.checkpoint,
        fixture.cache,
        fixture.out,
        audit_region_correction=correction,
    )
    curve["audit_region_correction"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reference differs from its bytes"):
        validate_curve(curve, min_points=1)


HISTORICAL = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "first-proof-anchor-preparation-05/workspace/rows/row-0110/cost.anchors.json"
)

CURRENT_COMPLETE_BF16 = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "sparse-rate-20260911/complete-down-curve-01/"
    "dense_l10_shared_down_bf16.anchors.json"
)


def test_current_complete_bf16_metadata_has_the_frozen_in_band_menu():
    if not CURRENT_COMPLETE_BF16.is_file():
        pytest.skip("shared complete-curve checkpoint is unavailable")
    plan_path = CURRENT_COMPLETE_BF16.with_name(
        "dense_l10_shared_down_bf16.measurement-plan.json"
    )
    manifest = json.loads(CURRENT_COMPLETE_BF16.read_text())
    plan = json.loads(plan_path.read_text())
    identity = manifest["identity"]
    menu = identity["units"][plan["qname"]]["menu"]
    assert len(menu) > len(plan["legal_rates"])
    _validate_campaign_menu(
        menu,
        settings=identity["settings"],
        plan=plan,
        where=CURRENT_COMPLETE_BF16,
    )


def test_historical_l3_journal_uses_the_same_typed_receipt_semantics():
    if not HISTORICAL.is_file():
        pytest.skip("shared historical journal is unavailable")
    manifest = json.loads(HISTORICAL.read_text())
    qname = "model.language_model.layers.3.mlp.shared_experts.down_proj"
    shard = unit_path(HISTORICAL.with_name(HISTORICAL.name + ".parts"), qname)
    state = _load_unit(shard, stage="Tessera campaign", qname=qname,
                       identity_sha256=manifest["identity_sha256"])
    unit = manifest["identity"]["units"][qname]
    wire_dir = HISTORICAL.parent / "cache" / "wire"
    cases = {
        "TESSERA_BF16_K1_R832": ("TESSERA_BF16_K1", "bf16_unquantized", False),
        "TESSERA_E2M1_K2_R896": ("TESSERA_E2M1_K2", "e2m1_group16_ue4m3_static", True),
        "TESSERA_E4M3_K1_R832": ("TESSERA_E4M3_K1", "fp8_per_token_dynamic", False),
    }
    anchors = {row["format_name"]: CampaignAnchor(**row) for row in state["anchors"]}
    for name, (family, activation, terminal) in cases.items():
        anchor = anchors[name]
        plan = {"curve_id": "historical-contract", "qname": qname, "family": family,
                "activation_contract": activation,
                "calibration_identity": {"hessian": "require"},
                "recipe_identity": {
                    "grid": {"TESSERA_BF16_K1": "BF16", "TESSERA_E2M1_K2": "E2M1",
                             "TESSERA_E4M3_K1": "E4M3"}[family],
                    "body": "tcq" if terminal else "window",
                    "scale_plane": "lut16" if family == "TESSERA_E2M1_K2" else "channel",
                    "activation_contract": activation,
                    **({"terminal": terminal} if family == "TESSERA_E2M1_K2" else {}),
                }}
        expected = _expected_row(plan, anchor.body_rate_q256)
        assert SEMANTIC_ACTIVATION_BY_ROUTE
        _verify_wire(anchor=anchor, record=state["wire_records"][name], expected=expected,
                     qname=qname, unit=unit, identity=manifest["identity"], wire_dir=wire_dir)
