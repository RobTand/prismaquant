"""The allocator's MTP stamper records the bound joint-cost sha (PQ #1665)."""
import hashlib
import json
import os
import pickle
from types import SimpleNamespace

from prismaquant.allocator import _select_mtp, _stamp_mtp_selection
from prismaquant.glm_mtp_selection import load_mtp_cost

_MTP_UNIT = "model.language_model.layers.45.mlp.experts.0.down_proj"
_MTP_FORMAT = "TESSERA_BF16_K1_R1024"


def _stamp(cost, record=None):
    layer_cfg = {"__prismaquant__": {}}
    if record is None:
        record = {
            "assignment": {_MTP_UNIT: _MTP_FORMAT},
            "rung": "g0=" + _MTP_FORMAT,
            "resident_bytes": 1,
            "byte_budget": 2,
            "selection": {"regime": "test"},
        }
    _stamp_mtp_selection(cost, layer_cfg, {}, record)
    return layer_cfg["__prismaquant__"]["mtp_selection"]


def _write_cost(path, label):
    from test_glm_mtp_selection import _payload

    payload = {**_payload(), "audit_input": label}
    path.write_bytes(pickle.dumps(payload))
    return load_mtp_cost(path)


def test_stamp_mtp_selection_stamps_joint_cost_sha(tmp_path):
    cost = tmp_path / "merged-cost.pkl"
    source = _write_cost(cost, "bound-mtp-cost-bytes")
    sel = _stamp(source)
    assert sel["mtp_joint_cost_sha256"] == hashlib.sha256(cost.read_bytes()).hexdigest()
    assert sel["cost_path"] == str(cost)


def test_stamp_mtp_selection_cost_sha_tracks_cost_bytes(tmp_path):
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    a = _write_cost(first, "first-cost")
    b = _write_cost(second, "second-cost")
    assert _stamp(a)["mtp_joint_cost_sha256"] != _stamp(b)["mtp_joint_cost_sha256"]


def test_replacement_after_real_selection_cannot_change_the_stamped_cost(tmp_path):
    from test_glm_mtp_selection import CONSTANTS

    path = tmp_path / "cost.pkl"
    original = _write_cost(path, "selected")
    constants = tmp_path / "constants.json"
    constants.write_text(json.dumps(CONSTANTS))
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({group: "BF16" for group in original.payload["groups"]}))
    args = SimpleNamespace(
        mtp_joint_cost=str(path), mtp_byte_budget=10**9,
        mtp_serve_constants=str(constants), mtp_acceptance_points=None,
        mtp_fixed_formats=str(fixed), mtp_formats=None,
    )
    selected_cost, record = _select_mtp(args)
    assert set(record["assignment"].values()) == {"BF16"}
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(pickle.dumps(original.payload, protocol=5))
    # The fixture's initial default protocol may already be 5 on new Python.
    if replacement.read_bytes() == path.read_bytes():
        replacement.write_bytes(pickle.dumps(original.payload, protocol=4))
    assert replacement.read_bytes() != path.read_bytes()
    os.replace(replacement, path)
    stamped = _stamp(selected_cost, record)
    assert stamped["mtp_joint_cost_sha256"] == original.sha256
    assert stamped["mtp_joint_cost_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    assert stamped["cost_path"] == str(path)
