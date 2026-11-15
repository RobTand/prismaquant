"""Joined Hessian admission and live AURA readers preserve source identity."""

from __future__ import annotations

import hashlib
import json
import pickle
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
    data["provenance"]["hessian"] = dict(panel)  # with capture_path: references open it
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
    data["provenance"]["hessian"] = dict(panel)  # with capture_path: references open it
    data["provenance"]["catalog_extension"] = extension
    cost_path.write_bytes(pickle.dumps(data))
    config["merged_cost"] = bind(cost_path)
    with pytest.raises(Exception, match=r"measured H capture_sha256"):
        bridge.load_measured_anchor_input(config, verify_payloads=False)
