import hashlib
import json
import pickle
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator, footprint, format_registry, model_profiles, shipcard
from prismaquant import tessera_plan_writer, validate_assignments_kl as validator
from prismaquant.export_native_compressed import _bf16_passthrough_for_assignment
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.model_profiles import DefaultProfile


FIXTURE = json.loads(
    '{"assignments":{"fixed_bf16":{"lm_head":"BF16","model.layers.0.mlp.down_proj":"NVFP4"},"selected_nvfp4":{"lm_head":"NVFP4","model.layers.0.mlp.down_proj":"NVFP4"}},"cap":{"assignment":{"model.layers.0.mlp.down_proj":"BF16"},"budget_bytes":12,"directory":"artifact","extra_file":["nested/extra.bin","x",1],"files":{"nested/sidecar.json":["s",5],"weights.bin":["w",7]},"selection_non_tensor_reserve_bytes":4,"selection_tensor_payload_bytes":8,"where":"trial cap"},"mtp_payload":{"params":{"mtp.layers.0.mlp.down_proj":256},"source_dtype":{"mtp.layers.0.mlp.down_proj":"bfloat16"}},"mtp_resident_format":"FP8_E4M3","profile":{"pinned_names":["lm_head"],"source_passthrough_prefixes":["mtp.","model.visual."]},"source_manifest":{"lm_head":1024,"model.embed_tokens":1024,"model.layers.0.mlp.down_proj":512,"mtp.layers.0.mlp.down_proj":512},"source_regime":"bf16","source_total_bytes":3072,"stats":{"lm_head":{"in_features":16,"n_params":512,"out_features":32},"model.layers.0.mlp.down_proj":{"in_features":16,"n_params":256,"out_features":16},"mtp.layers.0.mlp.down_proj":{"in_features":16,"n_params":256,"out_features":16}}}'
)
HEAD = "lm_head"
BODY = "model.layers.0.mlp.down_proj"
MTP = "mtp.layers.0.mlp.down_proj"
SPECS = {name: format_registry.get_format(name) for name in ("NVFP4", "BF16", "FP8_E4M3")}


class TrialProfile(DefaultProfile):
    def pinned_names(self):
        return tuple(FIXTURE["profile"]["pinned_names"])

    def is_pinned_name(self, name):
        return name in self.pinned_names()

    def source_passthrough_prefixes(self):
        return tuple(FIXTURE["profile"]["source_passthrough_prefixes"])

    def lm_head_name(self):
        return HEAD


def _digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


def test_recorded_head_accounting_bytes():
    assert _digest(FIXTURE) == "03170ae04b8b2a08f060f1426ab90b5325bda178fea49e7ef3ec1e808bc430cb"
    assignments = FIXTURE["assignments"]
    stats = FIXTURE["stats"]
    manifest = FIXTURE["source_manifest"]
    payloads = {
        case: footprint.assignment_artifact_bytes(
            assignment, stats, source_total_bytes=FIXTURE["source_total_bytes"],
            source_manifest=manifest, regime=FIXTURE["source_regime"],
        ) for case, assignment in assignments.items()
    }
    assert _digest(payloads) == "5f1c4ff6d6f40fd34350cbec9efb036d070d9e17ddc9815e13a2a776ee742d64"
    profile = TrialProfile()
    legacy = {
        case: validator._assignment_bpp_details(stats, assignment, SPECS, profile=profile)
        for case, assignment in assignments.items()
    }
    assert _digest(legacy) == "3e799a9fba95ce86383f214ace521f67ac6d2d3a810e109d6a80f4df65299a95"
    native = {
        case: sorted(_bf16_passthrough_for_assignment(None, profile, metadata))
        for case, metadata in {
            "immutable": {},
            "fixed": {"lm_head_mode": "fixed", "lm_head_format": "FP8_E4M3"},
            "dp_bf16": {"lm_head_mode": "dp", "lm_head_format": "BF16"},
        }.items()
    }
    assert _digest(native) == "84af05933aef8103a932f3dc80f3cf738941e7a595e477731f9aa7a1376c173d"
    selected = payloads["selected_nvfp4"]
    assert selected["reencoded_source_bytes"] == 1536
    assert selected["floor_bytes"] == 1536
    assert selected["body_quant_bytes"] == 448
    mtp_args = (
        selected["artifact_payload_bytes"], FIXTURE["mtp_payload"],
        SPECS[FIXTURE["mtp_resident_format"]].memory_bytes_for_shape((16, 16)),
    )
    mtp = footprint.mtp_selection_rebased_bytes(
        *mtp_args, context="trial MTP", manifest=manifest,
        assigned_names=assignments["selected_nvfp4"],
    )
    assert _digest(mtp) == "d4c2ae2a36bda1e2c753c6aa78e3bcdc25919eafc5973d5c2d34a5a479323f49"
    with pytest.raises(ValueError) as exc:
        footprint.mtp_selection_rebased_bytes(
            *mtp_args, context="trial MTP", manifest=manifest, assigned_names=[MTP],
        )
    error_type = type(exc.value)
    refusal = {"type": f"{error_type.__module__}.{error_type.__name__}", "message": str(exc.value)}
    assert _digest(refusal) == "90c9483ea449c24af13f850d0576646d321d5850e3e4810a3a51fbfc00970200"


@pytest.mark.parametrize("head_format,head_bytes,artifact_bytes", [
    ("NVFP4", 296, 1792), ("BF16", 1024, 2520),
])
def test_selectable_head_population_follows_the_allocation(
    tmp_path, monkeypatch, head_format, head_bytes, artifact_bytes,
):
    class Profile(TrialProfile):
        def pinned_names(self):
            return (*super().pinned_names(), "model.norm")

    profile = Profile()
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _: profile)
    model = tmp_path / "model"
    model.mkdir()
    stats = FIXTURE["stats"]
    save_file({
        name + ".weight": torch.zeros(
            (stats[name]["out_features"], stats[name]["in_features"])
            if name in stats else (32, 16), dtype=torch.bfloat16,
        ) for name in FIXTURE["source_manifest"]
    }, model / "model.safetensors")
    probe = tmp_path / "probe.pkl"
    probe.write_bytes(pickle.dumps({
        "stats": {name: dict(stats[name], h_trace=1.0) for name in (HEAD, BODY)}
        | {"model.norm": dict(stats[BODY], h_trace=1.0)},
        "meta": {"model": str(model)},
    }))
    costs = tmp_path / "costs.pkl"
    costs.write_bytes(pickle.dumps({
        "costs": {name: {fmt: {
            "weight_mse": 0.001, "output_mse": 0.001, "output_mse_measured": True,
            "predicted_dloss": 0.001,
        } for fmt in ("NVFP4", "BF16")} for name in (HEAD, BODY)},
        "meta": {"formats": ["NVFP4", "BF16"]},
    }))

    def solve(stats, candidates, target_bits, format_specs, format_rank, bit_precision, **kw):
        assignment = {BODY: "NVFP4", HEAD: head_format}
        bits = sum(8 * next(c.memory_bytes for c in candidates[name] if c.fmt == fmt)
                   for name, fmt in assignment.items())
        return assignment, bits / 768

    monkeypatch.setattr(allocator, "solve_with_promotion", solve)
    monkeypatch.setattr(allocator, "_select_mtp", lambda *a, **kw: (
        SimpleNamespace(payload=FIXTURE["mtp_payload"], path="trial", sha256="trial"),
        {"assignment": {MTP: "FP8_E4M3"}, "rung": "FP8_E4M3", "resident_bytes": 320,
         "byte_budget": 320, "selection": {"regime": "trial"}},
    ))
    config_path = tmp_path / "layer_config.json"
    pareto = tmp_path / "pareto"
    monkeypatch.setattr(sys, "argv", [
        "allocator", "--probe", str(probe), "--costs", str(costs),
        "--layer-config", str(config_path), "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--pareto-output-dir", str(pareto), "--allow-default-profile",
        "--allow-pinned", HEAD, "--formats", "NVFP4,BF16", "--pareto-targets", "16,17",
        "--target-disk-gb", "0.00001", "--artifact-overhead-reserve-bytes", "10",
        "--mtp-joint-cost", "trial", "--mtp-byte-budget", "320",
    ])
    allocator.main()
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["predicted_floor_gb"] * footprint.GB == pytest.approx(1344)
    config = json.loads(config_path.read_text())
    meta = config[LAYER_CONFIG_META_KEY]
    assert meta["whole_artifact_budget"]["selection_tensor_payload_bytes"] == artifact_bytes
    population = meta.get("quantizable_population")
    assert population is not None, "The recipe must disclose the quantizable population."
    assert population["body"] == {"params": 256, "tensor_payload_bytes": 152, "bpp": 4.75}
    assert population["head"] == {
        "mode": "dp", "params": 512, "tensor_payload_bytes": head_bytes,
        "formats": {HEAD: head_format},
    }
    assert population["selectable"] == {
        "params": 768, "tensor_payload_bytes": 152 + head_bytes,
        "bpp": (152 + head_bytes) * 8 / 768,
    }
    manifest = json.loads((pareto / "manifest.json").read_text())
    candidate_path = manifest["candidates"][0]["path"]
    assignment = validator.load_assignment_json(candidate_path)
    candidate = json.loads(open(candidate_path).read())
    assert footprint.whole_artifact_budget_from_assignment_payload(
        candidate, assignment=assignment, where="trial validation",
    ) == meta["whole_artifact_budget"]
    disclose = getattr(validator, "_resolved_quantizable_population", None)
    assert disclose is not None
    assert disclose(candidate, assignment, where="trial validation") == population
    with pytest.raises(ValueError, match="population"):
        disclose(candidate, {**assignment, HEAD: "FP8_E4M3"}, where="trial validation")

    tensor = BODY + ".weight"
    plan, provenance = tessera_plan_writer.build_serving_plan(
        {BODY: "BF16", LAYER_CONFIG_META_KEY: meta}, {tensor: (16, 16)},
        cover="as-allocated", allow_disagreement=False, with_control=False,
        surface=SimpleNamespace(body_layer=lambda _: 0, units_from_plan=lambda *_: [],
                                selection_requirement=lambda _: {}),
    )
    assert plan == {tensor: "BF16"}
    assert provenance["totals"]["quantized_params"] == 0
    assert provenance["prismaquant_meta"]["quantizable_population"] == population
    card = shipcard.build_shipcard(model, build={
        "layer_config": str(config_path), "layer_config_sha": shipcard.file_sha256(config_path),
    })
    assert card["build"]["quantizable_population"] == population
