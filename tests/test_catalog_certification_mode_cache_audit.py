"""Synthetic catalog metadata must not reuse dev admission as certification."""
from __future__ import annotations

import pytest

from prismaquant.joint_catalog_extension import verify_catalog_pair
from tests.test_joint_catalog_extension import _pair
from tests.test_joint_quanta_join import campaign, probe  # noqa: F401


@pytest.mark.parametrize("dev_warm", [False, True], ids=["fresh-certified", "dev-warmed"])
def test_missing_encoder_proof_refuses_in_current_certified_mode(
    tmp_path, monkeypatch, campaign, probe, dev_warm,
):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    added = {"TESSERA_E4M3_K1_R1152": sorted(campaign["roster"])}
    inputs, _, _ = _pair(tmp_path, campaign, probe, added=added, proof=False)
    if dev_warm:
        assert verify_catalog_pair(inputs)["added_cells"] == len(campaign["roster"])
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="names no encoder source proof"):
        verify_catalog_pair(inputs)


def test_proved_catalog_remains_admitted_after_dev_to_certified_transition(
    tmp_path, monkeypatch, campaign, probe,
):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    inputs, _, _ = _pair(tmp_path, campaign, probe, proof=True)
    before = verify_catalog_pair(inputs)
    assert before["added_cells"] == len(campaign["roster"])
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    assert verify_catalog_pair(inputs) == before
