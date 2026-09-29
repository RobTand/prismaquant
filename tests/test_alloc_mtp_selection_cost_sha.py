"""The allocator's MTP stamper records the bound joint-cost sha (PQ #1665)."""
import hashlib
from types import SimpleNamespace

from prismaquant.allocator import _stamp_mtp_selection

_MTP_UNIT = "model.language_model.layers.45.mlp.experts.0.down_proj"
_MTP_FORMAT = "TESSERA_BF16_K1_R1024"


def _stamp(cost_path, record=None):
    args = SimpleNamespace(mtp_joint_cost=str(cost_path))
    layer_cfg = {"__prismaquant__": {}}
    if record is None:
        record = {
            "assignment": {_MTP_UNIT: _MTP_FORMAT},
            "rung": "g0=" + _MTP_FORMAT,
            "resident_bytes": 1,
            "byte_budget": 2,
            "selection": {"regime": "test"},
        }
    _stamp_mtp_selection(args, layer_cfg, {}, record)
    return layer_cfg["__prismaquant__"]["mtp_selection"]


def test_stamp_mtp_selection_stamps_joint_cost_sha(tmp_path):
    cost = tmp_path / "merged-cost.pkl"
    cost.write_bytes(b"bound-mtp-cost-bytes")
    sel = _stamp(cost)
    assert sel["mtp_joint_cost_sha256"] == hashlib.sha256(b"bound-mtp-cost-bytes").hexdigest()
    assert sel["cost_path"] == str(cost)


def test_stamp_mtp_selection_cost_sha_tracks_cost_bytes(tmp_path):
    first = tmp_path / "first.pkl"
    first.write_bytes(b"first-cost")
    second = tmp_path / "second.pkl"
    second.write_bytes(b"second-cost")
    assert _stamp(first)["mtp_joint_cost_sha256"] != _stamp(second)["mtp_joint_cost_sha256"]
