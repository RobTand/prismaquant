import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator, footprint, format_registry, lane_shipcard, model_profiles, shipcard
from prismaquant import tessera_plan_writer, validate_assignments_kl as validator
from prismaquant.export_native_compressed import _bf16_passthrough_for_assignment
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.model_profiles import DefaultProfile
from tools.tessera_fleet import dispatch_model as dispatch


FIXTURE = json.loads('''{"assignments":{"fixed_bf16":{"lm_head":"BF16","model.layers.0.mlp.down_proj":"NVFP4"},"selected_nvfp4":{"lm_head":"NVFP4","model.layers.0.mlp.down_proj":"NVFP4"}},"cap":{"assignment":{"model.layers.0.mlp.down_proj":"BF16"},"budget_bytes":12,"directory":"artifact","extra_file":["nested/extra.bin","x",1],"files":{"nested/sidecar.json":["s",5],"weights.bin":["w",7]},"selection_non_tensor_reserve_bytes":4,"selection_tensor_payload_bytes":8,"where":"trial cap"},"mtp_payload":{"params":{"mtp.layers.0.mlp.down_proj":256},"source_dtype":{"mtp.layers.0.mlp.down_proj":"bfloat16"}},"mtp_resident_format":"FP8_E4M3","profile":{"pinned_names":["lm_head"],"source_passthrough_prefixes":["mtp.","model.visual."]},"source_manifest":{"lm_head":1024,"model.embed_tokens":1024,"model.layers.0.mlp.down_proj":512,"mtp.layers.0.mlp.down_proj":512},"source_regime":"bf16","source_total_bytes":3072,"stats":{"lm_head":{"in_features":16,"n_params":512,"out_features":32},"model.layers.0.mlp.down_proj":{"in_features":16,"n_params":256,"out_features":16},"mtp.layers.0.mlp.down_proj":{"in_features":16,"n_params":256,"out_features":16}}}''')
CAP = FIXTURE['cap']


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _stamp(budget):
    return footprint.whole_artifact_budget_stamp(
        budget_bytes=budget, selection_assignment=CAP['assignment'],
        selection_tensor_payload_bytes=CAP['selection_tensor_payload_bytes'],
        selection_non_tensor_reserve_bytes=CAP['selection_non_tensor_reserve_bytes'],
    )


def _artifact(root):
    for name, (value, count) in CAP['files'].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value * count)
    return root


def _build(tmp_path, budget):
    allocation = tmp_path / 'artifacts' / 'layer_config.json'
    allocation.parent.mkdir(exist_ok=True)
    allocation.write_text(json.dumps({**CAP['assignment'], '__prismaquant__': {
        'whole_artifact_budget': _stamp(budget),
    }}))
    plan_assignment = tmp_path / 'plan_assignment.json'
    plan_assignment.write_text('{}')
    build = tmp_path / 'tessera_build.json'
    build.write_text(json.dumps({
        'layer_config': str(allocation), 'layer_config_sha': shipcard.file_sha256(allocation),
        'plan_assignment': str(plan_assignment),
        'plan_assignment_sha256': shipcard.file_sha256(plan_assignment),
    }))
    return build, shipcard.file_sha256(build)


def test_budget_stamp_and_final_gate_keep_recorded_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = _artifact(Path(CAP['directory']))
    stamp = _stamp(CAP['budget_bytes'])
    assert _digest(FIXTURE) == '03170ae04b8b2a08f060f1426ab90b5325bda178fea49e7ef3ec1e808bc430cb'
    assert _digest(stamp) == 'dd1b62a6ab9c7208df08e05a321e990a9ca5f9cc69a608aca4fd574d954a38b7'
    payload = {'whole_artifact_budget': stamp}
    kwargs = {'assignment': CAP['assignment'], 'where': CAP['where']}
    result = footprint.enforce_whole_artifact_budget(artifact, payload, **kwargs)
    assert _digest(result) == 'f6e4f0f0a900775d7ccfc8fa000c10fe5f7998a3d09df2220596a04285da7a5e'
    name, value, count = CAP['extra_file']
    (artifact / name).write_text(value * count)
    with pytest.raises(RuntimeError) as refusal:
        footprint.enforce_whole_artifact_budget(artifact, payload, **kwargs)
    error = type(refusal.value)
    result = {'type': f'{error.__module__}.{error.__name__}', 'message': str(refusal.value),
              'actual_bytes': footprint.recursive_regular_file_bytes(artifact)}
    assert _digest(result) == '2a3fb5501a1b14f522dd1ae2c2682e64c0337033f2561552f9f0472ecc5ec466'


def _open_card(artifact, build, digest, *flags):
    try:
        return lane_shipcard.main(['open', '--lane', 'tessera', '--artifact', str(artifact),
                                   '--build-json', str(build), '--build-sha256', digest, *flags])
    except SystemExit as exc:
        return exc.code


def test_tessera_open_counts_the_card_before_publication(tmp_path, capsys):
    artifact = _artifact(tmp_path / 'artifact')
    card = artifact / shipcard.SHIPCARD_FILENAME
    build, digest = _build(tmp_path, CAP['budget_bytes'])
    assert _open_card(artifact, build, digest) == 2
    assert 'exceeding the hard whole-artifact budget' in capsys.readouterr().err
    assert not card.exists()
    assert footprint.recursive_regular_file_bytes(artifact) == CAP['budget_bytes']

    cap = CAP['budget_bytes'] + shipcard.SHIPCARD_RESERVED_BYTES
    build, digest = _build(tmp_path, cap)
    assert _open_card(artifact, build, digest) == 0
    assert footprint.recursive_regular_file_bytes(artifact) == cap
    assert shipcard.load_shipcard(card)['build']['layer_config_sha'] == json.loads(
        build.read_bytes())['layer_config_sha']

    existing = card.read_bytes()
    (artifact / 'extra.json').write_text('x')
    assert _open_card(artifact, build, digest, '--keep-existing') == 2
    assert 'by 1B' in capsys.readouterr().err
    assert card.read_bytes() == existing


def test_tessera_driver_passes_the_producer_build_digest_before_completion():
    driver = (Path(__file__).parents[1] / 'prismaquant' / 'run-pipeline.sh').read_text()
    export = driver.split('python3 -m tessera.export_serving', 1)[1].split('\n  exit 0', 1)[0]
    producer, opening = export.split('python3 -m prismaquant.lane_shipcard open', 1)
    assert '--priced-inputs-sha256 "$TESSERA_BUILD_SHA256"' in producer
    assert '--build-sha256 "$TESSERA_BUILD_SHA256"' in opening.split('; then', 1)[0]
    assert 'echo "[pipeline] done."' not in producer
    assert 'echo "[pipeline] done."' in opening


def test_tessera_driver_refuses_a_budgeted_export_without_the_build_digest(tmp_path):
    artifact = _artifact(tmp_path / 'exported')
    build, digest = _build(tmp_path, CAP['budget_bytes'] + shipcard.SHIPCARD_RESERVED_BYTES)
    assert _open_card(artifact, build, digest) == 0
    root = Path(__file__).resolve().parents[1]
    driver = (root / 'prismaquant' / 'run-pipeline.sh').read_text()
    completion = driver.split('2>&1 | tee "${WORK_DIR}/logs/export.log"', 1)[1]
    completion = completion.split('  # Print a shell-safe recipe', 1)[0]
    result = subprocess.run(
        ['bash', '-c', 'set -eu\npython3() { "$PYTEST_PYTHON" "$@"; }\n' + completion],
        cwd=root, env={'WORK_DIR': str(tmp_path),
                       'PYTEST_PYTHON': sys.executable, 'TESSERA_BUILD_JSON': str(build),
                       'TESSERA_BUILD_SHA256': ''},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'requires the preflight build digest' in result.stderr
    assert '[pipeline] done.' not in result.stdout


def test_tessera_open_rejects_allocation_drift(tmp_path, capsys):
    artifact = _artifact(tmp_path / 'artifact')
    build, digest = _build(tmp_path, CAP['budget_bytes'] + shipcard.SHIPCARD_RESERVED_BYTES)
    allocation = Path(json.loads(build.read_bytes())['layer_config'])
    allocation.write_text(json.dumps(CAP['assignment']))
    assert _open_card(artifact, build, digest) == 2
    assert 'allocation changed' in capsys.readouterr().err
    assert not (artifact / shipcard.SHIPCARD_FILENAME).exists()


def test_tessera_merge_gates_release_record(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path / 'artifact')
    record_path = artifact / 'pb-result.json'
    record = {'contract': 'test-merge', 'index': None}
    record_path.write_text(json.dumps(record))
    budget = CAP['budget_bytes'] + record_path.stat().st_size
    build, digest = _build(tmp_path, budget)
    state = tmp_path / dispatch.common.STATE
    state.mkdir()
    spec = {'out': str(artifact), 'count': 1, 'contract': record['contract'],
            'build_json': str(build), 'build_sha256': digest}
    (state / 'job.json').write_text(json.dumps(spec))
    monkeypatch.setattr(dispatch, 'model_export_row', lambda *args, **kwargs: {})
    monkeypatch.setattr(dispatch.common, 'run_stage', lambda *args, **kwargs: None)
    monkeypatch.setattr(dispatch, 'sealed_stage_source', lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(dispatch, 'saved_stage', lambda *args: [])
    monkeypatch.setattr(dispatch, 'part_records', lambda spec: [])
    monkeypatch.setattr(dispatch.common.campaign_rows, 'validate_row', lambda row: None)

    release = state / 'export-result.json'
    argv = ['--workspace', str(tmp_path), '--resume']
    name, value, count = CAP['extra_file']
    extra = artifact / name
    extra.write_text(value * count)
    with pytest.raises(RuntimeError, match='exceeding the hard whole-artifact budget.*by 1B'):
        dispatch.main(argv)
    assert not release.exists()
    extra.unlink()
    assert dispatch.main(argv) == 0
    assert json.loads(release.read_text()) == record


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


def test_recorded_head_accounting_bytes():
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
