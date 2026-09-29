"""Joined two-capture Hessian checks through both readers (issue #1273).

``tessera_joint_aura.walk_one`` and ``tessera_anchored_surface`` verified
every measured row's Hessian fields against ``provenance.hessian``. On a
table joined from two content-equal captures (PQ #1270) the overlay rows
carry another seal, and both readers refused rows priced under an identical
H. Both readers now admit seals from the uniform-table verdict
(``joint_catalog_extension.admitted_hessian_capture_seals``) while every
other field -- and every mixed-content refusal -- keeps its original text.

Fixtures reuse the #1270 joined capture builders. The ACCEPT tests fail on
the pre-change tree (no shared admission owner; the readers refuse the
overlay seal); the MIXED tests refuse with the original texts on both trees
(RED on the mutated driver: if new code admitted differing content, or
changed a text, these fail).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))


def _joined_fixture():
    from tests.test_hessian_identity_content_equal import _joined

    return _joined


def _bind(path, payload):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_admission_accepts_two_content_equal_captures(tmp_path):
    from prismaquant.joint_catalog_extension import admitted_hessian_capture_seals

    costs, references, panel, overlay = _joined_fixture()(tmp_path)
    admitted = admitted_hessian_capture_seals(costs, references=references)
    # Only the mapped non-primary seal is admitted; panel rows pass against
    # the provenance by equality, so the primary needs no admission.
    assert admitted == frozenset({overlay["capture_sha256"]})
    assert panel["capture_sha256"] not in admitted


def test_admission_admits_nothing_on_a_single_seal_table(tmp_path):
    from prismaquant.joint_catalog_extension import admitted_hessian_capture_seals
    from tests.test_hessian_identity_content_equal import ADDED_FMT  # noqa

    costs, references, panel, _overlay = _joined_fixture()(tmp_path)
    single = {name: {fmt: row for fmt, row in rows.items() if fmt != ADDED_FMT}
              for name, rows in costs.items()}
    assert admitted_hessian_capture_seals(single) == frozenset()


def test_admission_refuses_mixed_content(tmp_path):
    import torch
    from prismaquant.joint_catalog_extension import admitted_hessian_capture_seals
    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    shared_h = torch.eye(2) * 3
    costs, references, _panel, _overlay = _joined_fixture()(
        tmp_path, override={UNITS[0]: shared_h * 7})
    with pytest.raises(ValueError, match="mixes Hessian identities"):
        admitted_hessian_capture_seals(costs, references=references)


def _aura_hessian_cases():
    from prismaquant import tessera_joint_aura as aura

    return aura._verify_measured_hessian_identity


def test_aura_verifier_admits_overlay_seal(tmp_path):
    verify = _aura_hessian_cases()
    costs, references, panel, overlay = _joined_fixture()(tmp_path)
    from prismaquant.joint_catalog_extension import admitted_hessian_capture_seals

    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    name = UNITS[0]
    admitted = admitted_hessian_capture_seals(
        {name: costs[name]}, references=references)
    row = {"hessian_identity": dict(costs[name][ADDED_FMT]["hessian_identity"],
                                    applied=False)}  # anchor carries False
    anchor = {"hessian_applied": False}
    provenance = dict(panel, capture_sha256=panel["capture_sha256"])
    verify(name, row, anchor, provenance, admitted)


def test_aura_verifier_refuses_mixed_content_with_original_text(tmp_path):
    import torch

    verify = _aura_hessian_cases()
    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    shared_h = torch.eye(2) * 3
    costs, references, panel, _overlay = _joined_fixture()(
        tmp_path, override={UNITS[0]: shared_h * 7})
    name = UNITS[0]
    row = {"hessian_identity": dict(costs[name][ADDED_FMT]["hessian_identity"],
                                    applied=False)}  # anchor carries False
    anchor = {"hessian_applied": False}
    provenance = dict(panel, capture_sha256=panel["capture_sha256"])
    with pytest.raises(ValueError, match=r"measured H capture_sha256"):
        verify(name, row, anchor, provenance, frozenset())


def test_surface_verifier_admits_overlay_seal(tmp_path):
    from prismaquant import tessera_anchored_surface as surface
    from prismaquant.joint_catalog_extension import admitted_hessian_capture_seals
    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    costs, references, panel, overlay = _joined_fixture()(tmp_path)
    name = UNITS[0]
    admitted = admitted_hessian_capture_seals(
        {name: costs[name]}, references=references)
    hessian = dict(costs[name][ADDED_FMT]["hessian_identity"])
    provenance = dict(panel, capture_sha256=panel["capture_sha256"])
    surface._verify_measured_hessian(hessian, None, provenance, admitted)


def test_aura_verifier_refuses_single_seal_mismatch_with_original_text():
    verify = _aura_hessian_cases()
    row = {"hessian_identity": {
        "applied": False, "supplied": True, "capture_sha256": "0" * 64,
        "text_sha256": "1" * 64, "fit_ids_sha256": "2" * 64, "fit_tokens": 8}}
    provenance = {"supplied": True, "capture_sha256": "f" * 64,
                  "text_sha256": "1" * 64, "fit_ids_sha256": "2" * 64,
                  "fit_tokens": 8}
    with pytest.raises(ValueError, match=r"measured H capture_sha256"):
        verify("u", row, {"hessian_applied": False}, provenance, frozenset())


def test_surface_verifier_refuses_single_seal_mismatch_with_original_text(tmp_path):
    from prismaquant import tessera_anchored_surface as surface
    from prismaquant.tessera_anchored_surface import ReplayError

    hessian = {
        "applied": None, "supplied": True, "capture_sha256": "0" * 64,
        "text_sha256": "1" * 64, "fit_ids_sha256": "2" * 64, "fit_tokens": 8}
    provenance = {"supplied": True, "capture_sha256": "f" * 64,
                  "text_sha256": "1" * 64, "fit_ids_sha256": "2" * 64,
                  "fit_tokens": 8}
    with pytest.raises(ReplayError, match=r"Hessian capture_sha256 mismatch"):
        surface._verify_measured_hessian(hessian, None, provenance, frozenset())
    import torch

    from prismaquant import tessera_anchored_surface as surface
    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    shared_h = torch.eye(2) * 3
    costs, references, panel, _overlay = _joined_fixture()(
        tmp_path, override={UNITS[0]: shared_h * 7})
    name = UNITS[0]
    hessian = dict(costs[name][ADDED_FMT]["hessian_identity"])
    provenance = dict(panel, capture_sha256=panel["capture_sha256"])
    from prismaquant.tessera_anchored_surface import ReplayError

    with pytest.raises(ReplayError, match=r"Hessian capture_sha256 mismatch"):
        surface._verify_measured_hessian(hessian, None, provenance, frozenset())


def test_surface_verifier_refuses_mixed_content_with_original_text(tmp_path):
    import torch

    from prismaquant import tessera_anchored_surface as surface
    from tests.test_hessian_identity_content_equal import ADDED_FMT, UNITS  # noqa

    shared_h = torch.eye(2) * 3
    costs, references, panel, _overlay = _joined_fixture()(
        tmp_path, override={UNITS[0]: shared_h * 7})
    name = UNITS[0]
    hessian = dict(costs[name][ADDED_FMT]["hessian_identity"])
    provenance = dict(panel, capture_sha256=panel["capture_sha256"])
    from prismaquant.tessera_anchored_surface import ReplayError

    with pytest.raises(ReplayError, match=r"Hessian capture_sha256 mismatch"):
        surface._verify_measured_hessian(hessian, None, provenance, frozenset())


def _workspaces(tmp_path, units, *, override=None):
    from tests.test_joint_catalog_extension import (
        _canonical_capture,
        _workspace_hessian,
    )

    shared = _canonical_capture(tmp_path / "canonical", units)
    panel_row, panel = _workspace_hessian(tmp_path / "panel", shared, units)
    overlay_row, overlay = _workspace_hessian(
        tmp_path / "overlay", shared, units[1:], census_copy=True,
        override=override)
    assert panel_row["capture_sha256"] != overlay_row["capture_sha256"]
    cost = _bind(tmp_path / "overlay-cost.pkl",
                 pickle.dumps({"provenance": {"hessian": overlay}}))
    catalog = _bind(tmp_path / "catalog.json", {"cost": cost})
    plan = _bind(tmp_path / "plan.json", {"inputs": {"candidate_overlay": catalog}})
    extension = _bind(tmp_path / "extension.json",
                      {"inputs": {"extended_plan": plan}})
    return panel_row, panel, overlay_row, overlay, extension


def test_aura_accepts_joined_table_end_to_end(tmp_path):
    import pickle

    from prismaquant import tessera_joint_aura as bridge
    from tests.test_tessera_joint_aura import bind, fixture

    config, names, fmt, _payload, _states = fixture(tmp_path)
    panel_row, panel, overlay_row, overlay, extension = _workspaces(
        tmp_path, names)
    cost_path = tmp_path / "merged" / "cost.pkl"
    data = pickle.loads(cost_path.read_bytes())
    data["costs"][names[0]][fmt]["hessian_identity"] = dict(
        panel_row, applied=False)
    data["costs"][names[1]][fmt]["hessian_identity"] = dict(
        overlay_row, applied=False)
    data["provenance"]["hessian"] = dict(panel_row)
    data["provenance"]["catalog_extension"] = extension
    cost_path.write_bytes(pickle.dumps(data))
    config["merged_cost"] = bind(cost_path)
    out = bridge.load_measured_anchor_input(config, verify_payloads=False)
    assert sorted(out.formats_by_qname) == sorted(names)


def test_aura_refuses_mixed_content_end_to_end_with_original_text(tmp_path):
    import pickle
    import torch

    from prismaquant import tessera_joint_aura as bridge
    from tests.test_tessera_joint_aura import bind, fixture

    config, names, fmt, _payload, _states = fixture(tmp_path)
    shared_h = torch.eye(2) * 3
    panel_row, panel, overlay_row, overlay, extension = _workspaces(
        tmp_path, names, override={names[1]: shared_h * 7})
    cost_path = tmp_path / "merged" / "cost.pkl"
    data = pickle.loads(cost_path.read_bytes())
    data["costs"][names[0]][fmt]["hessian_identity"] = dict(
        panel_row, applied=False)
    data["costs"][names[1]][fmt]["hessian_identity"] = dict(
        overlay_row, applied=False)
    data["provenance"]["hessian"] = dict(panel_row)
    data["provenance"]["catalog_extension"] = extension
    cost_path.write_bytes(pickle.dumps(data))
    config["merged_cost"] = bind(cost_path)
    with pytest.raises(Exception, match=r"measured H capture_sha256"):
        bridge.load_measured_anchor_input(config, verify_payloads=False)


PANEL_FMT = "TESSERA_E4M3_K1_R1024"
OVERLAY_FMT = "TESSERA_E2M1_K2_R896"


def _surface_joined(tmp_path, *, override=None):
    """make_campaign shapes with two real-format rows under two seals."""
    import pickle

    from prismaquant.cost_stage_checkpoint import unit_path, write_unit
    from tests.test_joint_catalog_extension import (
        _canonical_capture,
        _workspace_hessian,
    )
    from tests.test_tessera_anchored_surface import make_campaign, seal

    units = ["h", "p1"]
    canon = _canonical_capture(tmp_path / "canonical", units)
    panel_row, panel = _workspace_hessian(tmp_path / "panel", canon, units)
    overlay_row, overlay = _workspace_hessian(
        tmp_path / "overlay", canon, units[1:], census_copy=True,
        override=override)
    cost = _bind(tmp_path / "overlay-cost.pkl",
                 pickle.dumps({"provenance": {"hessian": overlay}}))
    catalog = _bind(tmp_path / "catalog.json", {"cost": cost})
    plan_chain = _bind(tmp_path / "plan.json", {"inputs": {"candidate_overlay": catalog}})
    extension = _bind(tmp_path / "extension.json",
                      {"inputs": {"extended_plan": plan_chain}})

    cost_path, checkpoint, plan, _values = make_campaign(tmp_path)
    parts = checkpoint.with_name(checkpoint.name + ".parts")
    payload = pickle.loads(cost_path.read_bytes())
    manifest = json.loads(checkpoint.read_text())
    for unit in units:
        manifest["identity"]["units"][unit]["menu"].append(
            PANEL_FMT if unit == "h" else OVERLAY_FMT)
    manifest["identity_sha256"] = seal(manifest["identity"])
    checkpoint.write_text(json.dumps(manifest))
    seal_value = manifest["identity_sha256"]
    for unit, ident in (("h", panel_row), ("p1", overlay_row)):
        real_fmt = PANEL_FMT if unit == "h" else OVERLAY_FMT
        base_row = dict(payload["costs"][unit]["F_R1"])
        base_row["hessian_identity"] = dict(ident)
        payload["costs"][unit][real_fmt] = base_row
        envelope = pickle.loads(unit_path(parts, unit).read_bytes())
        state = pickle.loads(envelope["payload"])
        base_anchor = next(a for a in state["anchors"] if a["format_name"] == "F_R1")
        state["anchors"].append(dict(base_anchor, format_name=real_fmt))
        state["wire_records"][real_fmt] = dict(state["wire_records"]["F_R1"])
        write_unit(parts, stage="Tessera campaign", qname=unit,
                   identity_sha256=seal_value, state=state)
    payload["provenance"]["hessian"] = dict(panel_row)
    payload["provenance"]["catalog_extension"] = extension
    cost_path.write_bytes(pickle.dumps(payload))
    plan["input"]["payload_sha256"] = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    plan["input"]["checkpoint_identity_sha256"] = manifest["identity_sha256"]
    return cost_path, checkpoint, plan


def test_surface_accepts_joined_table_end_to_end(tmp_path):
    from prismaquant import tessera_anchored_surface as surface

    cost_path, checkpoint, plan = _surface_joined(tmp_path)
    measurements, _identity, _groups = surface.load_campaign_measurements(
        cost_path, checkpoint, plan)
    assert measurements[("h", PANEL_FMT)]["value"] == measurements[("h", "F_R1")]["value"]


def test_surface_refuses_mixed_content_end_to_end_with_original_text(tmp_path):
    import torch

    from prismaquant import tessera_anchored_surface as surface

    shared_h = torch.eye(2) * 3
    cost_path, checkpoint, plan = _surface_joined(
        tmp_path, override={"p1": shared_h * 7})
    with pytest.raises(surface.ReplayError, match=r"Hessian capture_sha256 mismatch"):
        surface.load_campaign_measurements(cost_path, checkpoint, plan)
