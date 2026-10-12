"""GLM scoped attention census and H capture (PQ #2579, part of #1842)."""
import argparse
import hashlib
import importlib.metadata
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from prismaquant import glm_mtp_capture
from prismaquant import tessera_calibration_cache as cc
from prismaquant import tessera_campaign as tc
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.perturbed_x_cache import activation_cache_filename


KDA_LAYERS = (0, 1)
MLA_LAYERS = (2, 3)
KDA_LEAVES = ("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj",
              "forget_gate.f_b_proj", "g_a_proj", "g_b_proj", "o_proj")
MLA_LEAVES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
INDEXER_LEAVES = ("indexer.wq_b", "indexer.wk", "indexer.weights_proj")


def _prefix(layer):
    return f"model.language_model.layers.{layer}.self_attn."


def _attention_units():
    """Every KDA, MLA and kv_b_proj Linear of the scope, with tiny shapes."""
    units = {}
    for layer in KDA_LAYERS:
        for index, leaf in enumerate(KDA_LEAVES):
            units[_prefix(layer) + leaf] = [8, 4 + (index % 3)]
    for layer in MLA_LAYERS:
        for index, leaf in enumerate(MLA_LEAVES):
            units[_prefix(layer) + leaf] = [8, 5 + (index % 2)]
    return units


def _contract():
    return {"schema": "prismaquant.pretrained_initialization.v1",
            "scope": "checkpoint_missing_state", "status": "completed",
            "transformers_version": importlib.metadata.version("transformers")}


def _runtime():
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers")}


def test_scoped_roster_lists_every_kda_mla_and_kv_b_unit():
    """The pinned-roster-only roster is the attention scope: all KDA and MLA
    Linears, kv_b_proj among them, and the DSA indexer stays pinned."""
    units = _attention_units()
    indexer = [_prefix(layer) + leaf for layer in MLA_LAYERS for leaf in INDEXER_LEAVES]
    body = ["model.language_model.layers.0.mlp.down_proj"]
    roster = tc.campaign_roster(body + sorted(units) + indexer, Glm5NextProfile(),
                                allow_pinned=tc.GLM_ATTENTION_ALLOW_PINNED,
                                pinned_roster_only=True)
    assert set(roster.dense) == set(units)
    assert set(roster.lifted) == set(units)
    assert set(indexer) <= set(roster.pinned)


def _base_census(model):
    """A body census over the 512 by 512 draw with a two-unit row pair."""
    return {"model": str(model), "nsamples": 512, "seqlen": 512, "seed": 0,
            "layer_stride": 1, "text_sha256": "corpus", "fit_ids_sha256": "ids",
            "counts": {"body_a": 262144, "body_b": 13}}


def _write_base(tmp_path, model):
    base = _base_census(model)
    path = tmp_path / "body-census.json"
    path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return base, ref


def _attention_census(base, ref, units):
    counts = {name: 16 for name in units}
    maxima = {name: 1.0 for name in units}
    return glm_mtp_capture.attention_census(
        base_census=base, base_census_ref=ref, units=units, counts=counts,
        max_abs=maxima, groups={f"u:{name}": [name] for name in units},
        model_load_contract=_contract(), attention_implementation="eager",
        capture_runtime=_runtime(), expert_projection=None), counts


def test_derived_census_names_base_draw_and_full_attention_scope(tmp_path):
    """The derived census links the base draw by digest and covers exactly the
    attention units, while its row pair stays the base draw's."""
    units = _attention_units()
    base, ref = _write_base(tmp_path, tmp_path / "source")
    census, counts = _attention_census(base, ref, units)
    assert census["attention_extension"]["base_census"] == ref
    assert census["attention_extension"]["base_census"]["sha256"] == ref["sha256"]
    assert set(census["counts"]) == set(units)
    assert (census["nsamples"], census["seqlen"]) == (512, 512)
    assert tc.census_token_counts(census, {}) == (262144, 13)
    assert tc.census_token_counts(census, counts) == (262144, 13)
    with pytest.raises(RuntimeError, match="exactly its units"):
        glm_mtp_capture.attention_census(
            base_census=base, base_census_ref=ref, units=units,
            counts={"body_a": 1}, max_abs={name: 1.0 for name in units},
            groups={}, model_load_contract=_contract(),
            attention_implementation="eager", capture_runtime=_runtime())


def test_body_census_without_flags_stays_byte_identical():
    """A census built without the scoped flags carries no new key, so the body
    output stays byte identical."""
    args = argparse.Namespace(model="m", nsamples=4, seqlen=8, seed=0, layer_stride=1)
    payload = tc.calibration_census(
        {"a": 16}, {"a": 1.0}, args=args, groups={"u:a": ["a"]},
        dense_targets=["a"], expert_targets=[], shapes={"a": (8, 4)},
        identity={"text_sha256": "t", "fit_ids_sha256": "f"})
    assert "attention_extension" not in payload
    assert "pinned_roster" not in payload
    assert set(payload) == {"schema", "model", "nsamples", "seqlen", "seed",
                            "layer_stride", "text_sha256", "fit_ids_sha256",
                            "counts", "max_abs", "unit_shapes", "anchor_groups",
                            "dense_targets", "expert_targets", "expert_projection"}
    assert tc._derived_census_base(payload) is None


def test_attention_capture_holds_one_h_row_per_unit(tmp_path):
    """The published capture seals one H table per listed attention unit, each
    measured over the base draw the census names."""
    torch.manual_seed(0)
    units = _attention_units()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"attention census fixture")
    contract = _contract()
    body_counts = {"body_a": 262144, "body_b": 13}
    body_acts = {"body_a": torch.randn(8, 3, dtype=torch.float32),
                 "body_b": torch.randn(8, 3, dtype=torch.float32)}
    body_max = {name: float(value.abs().max().item())
                for name, value in body_acts.items()}
    body_census = dict(_base_census(source), counts=dict(body_counts),
                       max_abs=dict(body_max),
                       unit_shapes={"body_a": [4, 3], "body_b": [4, 3]},
                       model_load_contract=contract,
                       attention_implementation="eager", capture_runtime=_runtime())
    body_path = tmp_path / "body-census.json"
    body_path.write_text(json.dumps(body_census, indent=2, sort_keys=True) + "\n")
    body_identity = cc.capture_identity(
        str(body_path), calibration={"fit_ids_sha256": "ids"}, max_act_rows=8,
        model_load_contract=contract, attention_implementation="eager")
    body_hessians = {name: (value.T @ value).to(torch.float32)
                     for name, value in body_acts.items()}
    body_record = cc.publish_capture(
        tmp_path / "body-capture", census_path=str(body_path),
        identity=body_identity, acts=body_acts, hessians=body_hessians,
        counts=body_counts, maxima=body_max)
    ref = {"path": str(body_path),
           "sha256": hashlib.sha256(body_path.read_bytes()).hexdigest()}
    roster = tc.campaign_roster(
        sorted(units), Glm5NextProfile(), allow_pinned=tc.GLM_ATTENTION_ALLOW_PINNED,
        pinned_roster_only=True)
    lift = argparse.Namespace(allow_pinned=tc.GLM_ATTENTION_ALLOW_PINNED,
                              pinned_roster_only=True)
    census = glm_mtp_capture.attention_census(
        base_census=body_census, base_census_ref=ref, units=units,
        counts={name: 16 for name in units},
        max_abs={name: 2.0 for name in units},
        groups={f"u:{name}": [name] for name in units},
        model_load_contract=contract, attention_implementation="eager",
        capture_runtime=_runtime(), expert_projection=None,
        pinned_roster=tc.pinned_roster_block(lift, roster))
    rows, hessians, maxima = {}, {}, {}
    for name, shape in units.items():
        value = torch.randn(16, shape[1], dtype=torch.float32)
        rows[name] = value
        hessians[name] = (value.T @ value).to(torch.float32)
        maxima[name] = float(value.abs().max().item())
    census["max_abs"] = dict(maxima)
    census_path = tmp_path / "attention-census.json"
    with cc.CaptureSourceAuthentication(
            str(source), body_identity, {},
            manifest_sha256=body_record["sha256"]) as owner:
        identity, census_sha256, sealed = glm_mtp_capture.publish_attention_capture(
            tmp_path / "attention-capture", census=census, census_path=census_path,
            source_authentication=owner,
            calibration={"text_sha256": "corpus", "fit_ids_sha256": "ids"},
            max_act_rows=32, rows=rows, hessians=hessians,
            counts={name: 16 for name in units}, max_abs=maxima,
            completed_contract=contract)
    assert identity["units"] == {name: list(shape) for name, shape in units.items()}
    assert census_sha256 == hashlib.sha256(census_path.read_bytes()).hexdigest()
    assert census["pinned_roster"]["pinned_roster_only"] is True
    manifest = cc.require_capture_contract(sealed["path"], sealed["sha256"])
    assert set(manifest["entries"]) == set(units)
    assert manifest["identity"] == identity
    for name, shape in units.items():
        stored = torch.load(tmp_path / "attention-capture" / "inputs"
                            / activation_cache_filename(name), weights_only=True)
        assert list(stored["hessian"].shape) == [shape[1], shape[1]]
        assert torch.equal(stored["hessian"], hessians[name])
