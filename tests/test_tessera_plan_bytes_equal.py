"""The acceptance fixture for prismaquant #1587: same allocation, same bytes.

The old export path ran two Tessera ``experiments/`` scripts
(``plan_from_layer_config.py --prismaquant`` then
``export_tessera_serving.py``); the new one writes the plan with
``prismaquant.tessera_plan_writer`` and exports through the supported entry
point ``python -m tessera.export_serving`` (RobTand/tessera#687).  This test
exports the same tiny GLM-shaped allocation through BOTH paths and requires
the plan documents and the exported bytes to be equal.

It runs wherever a Tessera checkout provides both paths -- true of any
checkout at tessera#687 or later, where the experiments/ entry points remain
as shims -- and skips otherwise (no ``TESSERA_REPO``, or a pin that predates
the package exporter, such as 38e96012).  At the pin bump to the #687 commit
this is the test that runs un-skipped in CI.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

PQ_ROOT = Path(__file__).resolve().parents[1]

#: GLM-5.3 names: the v44 construction census covers Glm5Next dense units
#: (``language_model.model.layers.*.mlp.{down,gate,up}_proj`` and
#: ``self_attn.{q,k,v,o}_proj``) and unpacked routed experts
#: (``...mlp.experts.*.{down,gate,up}_proj``); anything else the exporter
#: refuses as uncensused -- including Qwen3 names, which is why this
#: fixture is GLM-shaped, not Qwen3-shaped.
L0 = "model.language_model.layers.0"
UNITS = {
    f"{L0}.self_attn.q_proj": (32, 32),
    f"{L0}.self_attn.k_proj": (32, 32),
    f"{L0}.self_attn.v_proj": (32, 32),
    f"{L0}.self_attn.o_proj": (32, 32),
    f"{L0}.mlp.gate_proj": (64, 32),
    f"{L0}.mlp.up_proj": (64, 32),
    f"{L0}.mlp.down_proj": (32, 64),
    "model.language_model.layers.1.self_attn.o_proj": (32, 32),
    "model.language_model.layers.1.mlp.gate_proj": (64, 32),
    "model.language_model.layers.1.mlp.up_proj": (64, 32),
    "model.language_model.layers.1.mlp.down_proj": (32, 64),
}
#: The MoE half: one unpacked two-expert stack on layer 10 (past GLM's
#: first-3-dense prefix), the layout GLM-5.3 exports through
#: (``<moe>.experts`` keyed ``{grid, q256, source_layout}``).  Gate/up
#: (64, 32) and down (32, 64) satisfy the E4M3 tile rule (rows % 32,
#: cols % 16); both experts are uniform, as the exporter requires.  The
#: router (``mlp.gate``) is unallocated and stays BF16 -- the runtime
#: gives it no quantized route.
L1MOE = "model.language_model.layers.10.mlp"
MOE_EXPERTS = (0, 1)
MOE_UNITS = {
    f"{L1MOE}.experts.{e}.{proj}": shape
    for e in MOE_EXPERTS
    for proj, shape in (("gate_proj", (64, 32)),
                        ("up_proj", (64, 32)),
                        ("down_proj", (32, 64)))
}
MOE_ROUTER = f"{L1MOE}.gate"
MOE_RUNG = {"tessera_format": "TESSERA_E4M3_K1_R1024"}


def _repo():
    repo = os.environ.get("TESSERA_REPO")
    if not repo:
        pytest.skip("requires TESSERA_REPO to name a Tessera checkout")
    root = Path(repo).resolve()
    for needed in ("experiments/plan_from_layer_config.py",
                   "experiments/export_tessera_serving.py",
                   "src/tessera/export_serving.py"):
        if not (root / needed).is_file():
            pytest.skip(f"checkout carries no {needed} (pre-#687 pin)")
    return root


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for unit, (rows, cols) in {**UNITS, **MOE_UNITS}.items():
        tensors[unit + ".weight"] = torch.randn(
            rows, cols, generator=generator).bfloat16()
    tensors[MOE_ROUTER + ".weight"] = torch.randn(
        len(MOE_EXPERTS), 32, generator=generator).bfloat16()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 11,
        "num_experts": len(MOE_EXPERTS)}))
    return path


def _assignment(path: Path) -> Path:
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    other = {"tessera_format": "TESSERA_E4M3_K1_R896"}
    # q/k/v/o stay unallocated (present in the checkpoint, planned as
    # nothing): the producer's fused_module fuses q/k/v into a qkv_proj the
    # v44 census does not offer for Glm5 (separate q/k/v only), so no qkv
    # allocation can export on this tree -- Tessera #706, not worked around
    # here -- and o_proj is never_offered (vLLM builds GLM o at BF16 by
    # design).  The fused path is exercised by gate/up instead, agreeing on
    # one rung per group: a disagreeing fused group is refused (or demoted
    # with --allow-fused-disagreement), which would test the refusal path
    # instead of the bytes path.  Rung variety comes from layer 1 at R896.
    L1 = "model.language_model.layers.1"
    allocation = {
        f"{L0}.mlp.gate_proj": fmt,
        f"{L0}.mlp.up_proj": fmt,
        f"{L1}.mlp.gate_proj": other,
        f"{L1}.mlp.up_proj": other,
        f"{L0}.mlp.down_proj": "BF16",
        f"{L1}.mlp.down_proj": "BF16",
        # One exact rung for the whole stack: the producer serves it whole.
        **{unit: dict(MOE_RUNG) for unit in MOE_UNITS},
    }
    path.write_text(json.dumps(allocation, indent=2, sort_keys=True))
    return path


def _run(argv, *, cwd, env):
    result = subprocess.run(
        [sys.executable, *argv], cwd=str(cwd), env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _tree_digests(root: Path):
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = \
                hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_the_supported_path_exports_the_same_bytes(tmp_path):
    repo = _repo()
    src = _checkpoint(tmp_path / "src")
    assignment = _assignment(tmp_path / "layer_config.json")

    # Old path: the two experiments/ scripts the arm used to call.
    old_plan = tmp_path / "plan-old.json"
    _run([str(repo / "experiments/plan_from_layer_config.py"),
          str(assignment), str(src), str(old_plan),
          "--cover", "as-allocated", "--prismaquant", str(PQ_ROOT)],
         cwd=repo, env=dict(os.environ))
    out_old = tmp_path / "exported-old"
    _run([str(repo / "experiments/export_tessera_serving.py"),
          str(src), str(out_old), "--plan-json", str(old_plan),
          "--device", "cpu", "--no-verify"],
         cwd=repo, env=dict(os.environ))

    # New path: PrismaQuant writes the plan; the package module exports.
    new_plan = tmp_path / "plan-new.json"
    env = dict(os.environ, PYTHONPATH=str(PQ_ROOT))
    _run(["-m", "prismaquant.tessera_plan_writer",
          str(assignment), str(src), str(new_plan), "--cover", "as-allocated"],
         cwd=PQ_ROOT, env=env)
    out_new = tmp_path / "exported-new"
    _run(["-m", "tessera.export_serving",
          str(src), str(out_new), "--plan-json", str(new_plan),
          "--device", "cpu", "--no-verify"],
         cwd=repo, env=dict(os.environ,
                            PYTHONPATH=str(repo / "src")))

    assert json.loads(new_plan.read_text()) == json.loads(old_plan.read_text())
    # The MoE path ran: the stack is planned as one unit, not completed BF16.
    stack = f"{L1MOE}.experts"
    entry = json.loads(new_plan.read_text())[stack]
    assert entry == {"grid": "E4M3", "q256": 1024,
                      "source_layout": "unpacked_per_expert"}, entry

    old_files, new_files = _tree_digests(out_old), _tree_digests(out_new)
    assert sorted(old_files) == sorted(new_files)
    for name in sorted(old_files):
        if name.endswith(".safetensors"):
            assert old_files[name] == new_files[name], name
    # Manifests may embed their own absolute paths; compare them with both
    # output roots neutralised so only real content differences can fail.
    # The exporter also stamps its wall clock and the --plan-json path.
    def _neutral(root, plan, name):
        text = (root / name).read_text()
        text = text.replace(str(root), "<out>").replace(str(plan), "<plan>")
        return re.sub(r'"written":\s*"[^"]*"', '"written":"<t>"', text)
    for name in sorted(old_files):
        if name.endswith(".json"):
            assert (_neutral(out_old, old_plan, name)
                    == _neutral(out_new, new_plan, name)), name
