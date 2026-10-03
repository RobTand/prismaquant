"""A genuine T-16 wire cannot authenticate incompatible scoring metadata (#2109)."""
import json
import pickle
import shutil
from types import SimpleNamespace

import pytest

from test_tessera_campaign_resume import (
    UNIT, _forbid_reencode, _main_fixture, _priced_file_digests,
)


FMT = "TESSERA_BF16_K1_R1024"


def t16_campaign_fixture(monkeypatch, path):
    campaign, checkpoint, argv, model, inputs = _main_fixture(monkeypatch, path, priced=True)
    inputs["menu"] = [SimpleNamespace(format_name=FMT, family="TESSERA_BF16_K1",
        body_rate_q256=1024, bpp=4.0)]
    return campaign, checkpoint, argv, model, inputs


@pytest.fixture(scope="module")
def completed_t16_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-t16-publication")
    with pytest.MonkeyPatch.context() as patch:
        campaign, _checkpoint, argv, _model, _inputs = t16_campaign_fixture(patch, root)
        # Real encode, PWC and bounded ordered publication, then read-back wire
        # receipt and durable journal. This is CPU producer evidence only.
        assert campaign.main([*argv, "--publication-overlap-bytes", str(1024 ** 2)]) == 0
    payload = pickle.loads((root / "cost.pkl").read_bytes())
    assert payload["costs"][UNIT][FMT]["activation_contract"] == "bfloat16"
    assert payload["costs"][UNIT][FMT]["activation_quantized"] is False
    return root, _priced_file_digests(root), payload["costs"]


@pytest.mark.parametrize("dev_mode", ["0", "1"])
@pytest.mark.parametrize("seed", [False, True], ids=["resume", "seed"])
@pytest.mark.parametrize("mutation", [None, "contract", "quantized", "observation_type"])
def test_t16_produced_wire_does_not_admit_incompatible_activation_metadata(
        completed_t16_campaign, monkeypatch, tmp_path, dev_mode, seed, mutation):
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path, write_unit

    root, digests, initial_costs = completed_t16_campaign
    assert _priced_file_digests(root) == digests

    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    campaign, checkpoint, argv, _model, _inputs = t16_campaign_fixture(monkeypatch, tmp_path)
    manifest = json.loads(checkpoint.read_text())
    parts = checkpoint.with_name(checkpoint.name + ".parts")
    state = _load_unit(unit_path(parts, UNIT), stage="Tessera campaign", qname=UNIT,
        identity_sha256=manifest["identity_sha256"])
    anchor = state["anchors"][0]
    assert anchor["format_name"] == FMT and state["wire_records"][FMT]["blob_bytes"] > 0
    if mutation == "contract":
        anchor["activation_contract"] = "fp8_e4m3"
    elif mutation == "quantized":
        anchor["activation_quantized"] = True
    elif mutation == "observation_type":
        anchor["activation_quantized"] = "false"
    if mutation is not None:
        # Publish a valid checksummed envelope; the defect is a scientific
        # mismatch, not a corrupt pickle/digest or fabricated native wire.
        write_unit(parts, stage="Tessera campaign", qname=UNIT,
            identity_sha256=manifest["identity_sha256"], state=state)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    _forbid_reencode(monkeypatch, campaign)
    output = tmp_path / "consumed.pkl"
    argv[argv.index("--out") + 1] = str(output)
    new_cache = tmp_path / "seed-cache"
    if seed:
        argv[argv.index("--checkpoint") + 1] = str(tmp_path / "seed.anchors.json")
        argv[argv.index("--cache-dir") + 1] = str(new_cache)
        argv += ["--seed-checkpoint", str(checkpoint)]
    if mutation is None:
        assert campaign.main(argv) == 0
        assert pickle.loads(output.read_bytes())["costs"] == initial_costs
    else:
        with pytest.raises(campaign.ActivationScaleContractError, match="activation (contract|observation)"):
            campaign.main(argv)
        assert not output.exists()
        if seed:
            assert not list((new_cache / "wire").glob("*.tessera"))
    assert _priced_file_digests(root) == digests


def test_quantizing_route_may_observe_unchanged_actual_rows(monkeypatch, tmp_path):
    campaign, _checkpoint, argv, _model, inputs = _main_fixture(monkeypatch, tmp_path, priced=True)
    inputs["rows"].zero_()
    assert campaign.main(argv) == 0
    payload = pickle.loads((tmp_path / "cost.pkl").read_bytes())
    row = payload["costs"][UNIT]["TESSERA_E4M3_K1_R1024"]
    assert row["activation_quantized"] is False
    _forbid_reencode(monkeypatch, campaign)
    argv[argv.index("--out") + 1] = str(tmp_path / "unchanged-a8.pkl")
    assert campaign.main(argv) == 0
    assert pickle.loads((tmp_path / "unchanged-a8.pkl").read_bytes())["costs"] == payload["costs"]
