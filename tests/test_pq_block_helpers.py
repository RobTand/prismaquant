"""Preserve byte recipes and refusal behavior in the research helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tessera.alphabet import E4M3_GRID
from tessera.export import encode_linear_planes, served_recipe
from tessera.fused import pack_fused
from tessera.unit_artifact import parse_unit_artifact

from prismaquant.digests import bytes_sha256hex, file_sha256hex
from prismaquant.tensor_digests import tensor_sha256
from tools import pq_block_concentration as concentration
from tools import pq_block_feasibility_intake as intake
from tools import pq_block_reference_wire as wire
from tools import pq_block_trial as trial
from tools.pq_block_trial_math import write_trial_json

ROOT = Path(__file__).resolve().parents[1]
QNAME = "model.language_model.layers.0.mlp.down_proj"


def make_synthetic_parents(directory):
    weight = ((torch.arange(128 * 32).reshape(128, 32) % 31) - 15).float() / 32
    parents, entries = [], []
    for rung in (768, 1024):
        recipe = served_recipe(E4M3_GRID, rung, "dense")
        exported, _, _ = encode_linear_planes(
            weight, grid=E4M3_GRID, q256=rung, name=f"TESSERA_E4M3_K1_R{rung}",
            verify=True, body=recipe.body, span=recipe.span,
            scale_plane=recipe.scale_plane, window_bits=recipe.window_bits,
            window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
            channel_sigma=recipe.channel_sigma)
        path = directory / f"parent-R{rung}.tessera"
        path.write_bytes(exported.blob)
        parents.append(parse_unit_artifact(exported.blob, device="cpu"))
        entries.append({"rung": rung, "path": str(path), "bytes": len(exported.blob),
                        "sha256": bytes_sha256hex(exported.blob)})
    return parents, entries


@pytest.fixture(scope="module")
def synthetic_parents(tmp_path_factory):
    return make_synthetic_parents(tmp_path_factory.mktemp("research-parents"))


def test_trial_json_preserves_order_unicode_and_newline(tmp_path):
    document = {"z": "é", "a": [1, -0.0, None]}
    path = tmp_path / "result.json"
    write_trial_json(path, document)
    assert path.read_bytes() == b'{\n  "z": "\\u00e9",\n  "a": [\n    1,\n    -0.0,\n    null\n  ]\n}\n'


def test_trial_json_refuses_nan_before_write(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"previous bytes")
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        write_trial_json(path, {"value": float("nan")})
    assert path.read_bytes() == b"previous bytes"


def test_intake_file_stamp_uses_complete_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    payload = bytes(range(256)) * 4097
    path.write_bytes(payload)
    assert intake.own_file(path) == {
        "path": str(path), "bytes": len(payload), "sha256": bytes_sha256hex(payload)}
    with pytest.raises(FileNotFoundError):
        intake.own_file(tmp_path / "absent")


def test_trial_refusal_class_and_message():
    with pytest.raises(trial.TrialRefused, match="block rows must be a positive multiple of 8"):
        trial._geometry("7x2")
    with pytest.raises(wire.BlockWireFormatError, match="pack needs at least one parent"):
        wire.fixed_bytes([], 8, 2)


def test_wire_exact_roundtrip_checksum_and_refusals(synthetic_parents):
    parents, _ = synthetic_parents
    tags = (np.arange(32).reshape(2, 16) % 2).astype(np.int64)
    blob, breakdown = wire.pack_projection(parents, tags, 64, 2)
    decoded = wire.decode_projection(blob)
    from tessera.stock import materialize_stock, stock_dequant
    rendered = [stock_dequant(materialize_stock(p.unit, p.forests, p.code)) for p in parents]
    expected = torch.empty_like(rendered[0])
    for i in range(2):
        for j in range(16):
            expected[i * 64:(i + 1) * 64, j * 2:(j + 1) * 2] = rendered[tags[i, j]][
                i * 64:(i + 1) * 64, j * 2:(j + 1) * 2]
    assert torch.equal(decoded, expected)
    assert blob[-32:] == bytes.fromhex(bytes_sha256hex(blob[:-32]))
    assert breakdown["checksum"] == blob[-32:].hex()
    corrupt = bytearray(blob)
    corrupt[-1] ^= 1
    with pytest.raises(wire.BlockWireFormatError, match="checksum mismatch: the blob is corrupt"):
        wire.decode_projection(corrupt)
    with pytest.raises(wire.BlockWireFormatError, match="overflow: packed content needs"):
        wire.pack_projection(parents, tags, 64, 2, target_bytes=len(blob) - 1)


def test_schedule_actual_cli_raw_wrapper_and_bad_digest(tmp_path, synthetic_parents):
    parents, entries = synthetic_parents
    tags = (np.arange(32).reshape(2, 16) % 2).astype(np.int64)
    blob, breakdown = wire.pack_projection(parents, tags, 64, 2)
    bank = tmp_path / "bank.json"
    selection = tmp_path / "selection.npy"
    packed = tmp_path / "packed.bin"
    breakdown_path = tmp_path / "breakdown.json"
    output = tmp_path / "schedule.json"
    np.save(selection, tags)
    packed.write_bytes(blob)
    write_trial_json(bank, {"schema": "prismaquant.block_trial_parent_bank.v1",
                            "qname": QNAME, "parents": entries})
    command = [sys.executable, str(ROOT / "tools/pq_block_schedule_cost.py"),
               "--parent-bank", str(bank), "--selection", str(selection),
               "--breakdown", str(breakdown_path), "--output", str(output),
               "--packed-blob", str(packed)]
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    for document in (breakdown, {
            "schema": "prismaquant.block_trial_breakdown.v1", "wire": breakdown,
            "candidate_order": [768, 1024], "files": {
                "selection": {"sha256": file_sha256hex(selection)},
                "packed_blob": {"sha256": bytes_sha256hex(blob)}}}):
        write_trial_json(breakdown_path, document)
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(output.read_bytes())
        assert summary["stored_bytes"]["total_bytes"] == len(blob)
        assert output.read_bytes().endswith(b"\n")
    document["files"]["selection"]["sha256"] = "0" * 64
    write_trial_json(breakdown_path, document)
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert "selection bytes fail their producer stamp" in result.stderr


def test_concentration_tensor_and_member_digests(tmp_path, synthetic_parents):
    _, entries = synthetic_parents
    member = Path(entries[1]["path"]).read_bytes()
    blob = pack_fused([("down_proj", 128, member)])
    tensor = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
    key = QNAME + ".wire"
    shard = "weights.safetensors"
    save_file({key: tensor}, str(tmp_path / shard))
    rendered, stamp = concentration.decode_existing(tmp_path, {key: shard}, QNAME)
    assert rendered.shape == (128, 32)
    assert stamp["container_sha256"] == tensor_sha256(tensor) == bytes_sha256hex(blob)
    assert stamp["member_sha256"] == bytes_sha256hex(member)
