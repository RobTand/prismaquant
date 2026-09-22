"""Tiny real-CUDA Stage B activation check; execute only in an admitted action."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prismaquant import format_registry
from prismaquant.joint_cost_quantum import bind_joint_served_quantizer
from prismaquant.nvfp4_activation_contract import ServedQuantizerUnboundError
from prismaquant.perturbed_x_cache import _activation_qdq

fmt = "TESSERA_E2M1_K2_R896"
spec = format_registry.get_format(fmt)
name = "qualification.a4"
x = torch.linspace(-10, 10, 4096, device="cuda", dtype=torch.bfloat16).reshape(4, 1024)
try:
    _activation_qdq(x, spec, {name: 12.0}, name)
except ServedQuantizerUnboundError:
    pass
else:
    raise AssertionError("fresh Stage B process unexpectedly had bound arithmetic")
identity = bind_joint_served_quantizer({name: [fmt]})
y = _activation_qdq(x, spec, {name: 12.0}, name)
torch.cuda.synchronize()
assert y.shape == x.shape and y.dtype == x.dtype
assert torch.isfinite(y).all().item() and not torch.equal(x, y)
assert bind_joint_served_quantizer({name: [fmt]}) == identity
print(json.dumps({"status": "passed", "served_quantizer": identity,
                  "shape": list(x.shape), "changed_elements": int((x != y).sum().item()),
                  "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()}, sort_keys=True))
