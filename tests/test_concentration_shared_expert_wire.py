from hashlib import sha256

import numpy as np
import torch
from safetensors.torch import save_file
from tessera.alphabet import E4M3_GRID
from tessera.export import encode_linear_planes, served_recipe
from tessera.fused import pack_fused
from tessera.stock import materialize_stock, stock_dequant
from tessera.unit_artifact import parse_unit_artifact

from tools import pq_block_concentration as concentration


def test_concentration_decodes_the_shared_expert_member_from_fused_wire(tmp_path):
    weight = ((torch.arange(128 * 32).reshape(128, 32) % 31) - 15).float() / 32
    recipe = served_recipe(E4M3_GRID, 1024, "dense")
    members, expected = [], {}
    for role, values in (("gate_proj", weight), ("up_proj", -weight)):
        exported, _, _ = encode_linear_planes(
            values, grid=E4M3_GRID, q256=1024, name="TESSERA_E4M3_K1_R1024",
            verify=True, body=recipe.body, span=recipe.span,
            scale_plane=recipe.scale_plane, window_bits=recipe.window_bits,
            window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
            channel_sigma=recipe.channel_sigma)
        parsed = parse_unit_artifact(exported.blob, device="cpu")
        stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
        expected[role] = stock_dequant(stock), parsed.manifest.branch
        members.append((role, weight.shape[0], exported.blob))
    assert not torch.equal(expected["gate_proj"][0], expected["up_proj"][0])

    blob = pack_fused(members)
    tensor = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
    prefix = "model.layers.0.mlp.shared_experts"
    key, shard = prefix + ".gate_up_proj.wire_bytes", "shared.safetensors"
    save_file({key: tensor}, str(tmp_path / shard))
    for role, _, member in members:
        target, branch = expected[role]
        rendered, stamp = concentration.decode_existing(tmp_path, {key: shard}, prefix + "." + role)
        torch.testing.assert_close(rendered, target, rtol=0, atol=0)
        assert stamp == {
            "tensor_key": key, "shard": shard,
            "container_bytes": len(blob), "member_bytes": len(member),
            "container_sha256": sha256(blob).hexdigest(), "member_sha256": sha256(member).hexdigest(),
            "stored_root_q256": branch.root_q256, "stored_unit_name": branch.unit_id,
        }
