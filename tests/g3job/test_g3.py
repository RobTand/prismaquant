"""CPU tests for the G3 offline decoded-forward evaluator (g3_lib.py), run through PB x86.

(a) the substitution swaps exactly the Tessera-quantized tensors: the priced-Linear -> slice map
    is the streamed loader's own per-expert packing, the rows partition every Tessera tensor of a
    layer, nothing else in the layer changes, and the manifest covers every wire the panel
    forward executes;
(b) the hash gate refuses corrupted, truncated and wrong-unit blobs and a perturbed decode, on
    real T8R wires, and leaves the layer bitwise unchanged when it refuses;
(c) the W+A QDQ is the runtime_contract.json sm_121 TESSERA_E4M3_K1 contract
    (fp8_per_token_dynamic) as the pinned Tessera test restates the kernel, on that test's own
    input classes, with the served TP2 slicing, and the hooks put it exactly where it belongs.
"""
from __future__ import annotations

import ast
import json
import os
import re
import struct
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.fleet_data

HERE = Path(__file__).resolve().parents[2] / "tools" / "g3job"
sys.path.insert(0, str(HERE))
PIN = Path(os.environ.get("TESSERA_PIN", "/mnt/shared/tessera-pins/b40c93cb73745097e57a1ba4cf5b9eee166c759a"))
PQ = Path(os.environ.get("PQ_SRC", "/mnt/shared/tessera-measurements/surrogate-diag-20260929/src/pq-7882eda3"))
MANIFEST = Path(os.environ.get("G3_MANIFEST",
                               "/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3/unit_manifest_v2.json"))
SOURCE = Path("/mnt/shared/models/GLM-5.3-Flash-BF16")
PAYLOAD_SHA = "6f158c986b2aa21f5e41e9960b0403b25efc9ae6f44d7b93b6d294869fd27341"

import g3_lib as L  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", os.environ.get("OMP_NUM_THREADS", "1"))))


# ============================================================== (c) the activation contract
def _pinned_namespace():
    """The pinned test's input classes and `_kernel_arithmetic`, executed from its own source."""
    route = ast.parse((PIN / "tests" / "test_serving_fp8_route.py").read_text())
    fp8_max = None
    for node in route.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FP8_MAX" for t in node.targets):
            fp8_max = ast.literal_eval(node.value)
    assert fp8_max == 448.0
    tree = ast.parse((PIN / "tests" / "test_native_fp8_quant.py").read_text())
    want = {"MIN_SCALE", "_e4m3_positive_values", "_bf16_exact", "_random", "_tiny_amax", "_outliers",
            "_powers_of_two", "MIDPOINT_SCALES", "_midpoints", "CLASSES", "_kernel_arithmetic"}
    body = []
    for node in tree.body:
        names = set()
        if isinstance(node, ast.FunctionDef):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {getattr(t, "id", None) for t in node.targets}
        if names & want:
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    ns = {"torch": torch, "FP8_MAX": fp8_max}
    exec(compile(mod, str(PIN / "tests" / "test_native_fp8_quant.py"), "exec"), ns)
    assert want <= set(ns), sorted(want - set(ns))
    return ns


NS = _pinned_namespace()


def test_contract_table_names_fp8_per_token_dynamic_for_e4m3_on_sm121():
    contract = json.loads((PIN / "src" / "tessera" / "serving" / "runtime_contract.json").read_text())
    executes = contract["lane_eligibility"]["platforms"]["sm_121"]["executes"]
    assert executes["TESSERA_E4M3_K1"] == "fp8_per_token_dynamic"
    native = (PIN / "src" / "tessera" / "serving" / "native_ops.py").read_text()
    assert "dynamic_per_token_scaled_fp8_quant" in native
    print("CONTRACT sm_121 executes", executes)


def test_min_scale_is_the_pinned_floor():
    assert L.MIN_SCALE == NS["MIN_SCALE"]
    assert L.FP8_MAX == 448.0


@pytest.mark.parametrize("name", sorted(NS["CLASSES"]))
def test_qdq_is_the_pinned_kernel_arithmetic(name):
    x = NS["CLASSES"][name]()
    q_k, s_k = NS["_kernel_arithmetic"](x)
    q, s = L.fp8_per_token_dynamic(x)
    codes_differ = int((q.view(torch.uint8) != q_k.view(torch.uint8)).sum())
    scales_differ = int((s != s_k).sum())
    deq = L.qdq_rows(x, 1)
    ref = (q_k.float() * s_k).to(x.dtype)
    deq_differ = int((deq.view(torch.int16) != ref.view(torch.int16)).sum())
    print(f"QDQ {name} shape={tuple(x.shape)} codes_differ={codes_differ} scales_differ={scales_differ} "
          f"deq_differ={deq_differ}")
    assert codes_differ == 0 and scales_differ == 0 and deq_differ == 0


def _deq_kernel(x2d):
    q, s = NS["_kernel_arithmetic"](x2d)
    return (q.float() * s).to(x2d.dtype)


def test_tp_slices_quantize_each_contiguous_half_with_its_own_scale():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(9, 4096, generator=g).to(torch.bfloat16)
    x[:, :2048] *= 1000.0       # halves of very different magnitude: one scale would crush the small half
    ref = torch.cat([_deq_kernel(x[:, :2048]), _deq_kernel(x[:, 2048:])], dim=1)
    got = L.qdq_rows(x, 2)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert not torch.equal(L.qdq_rows(x, 1), got)
    x3 = x.reshape(3, 3, 4096)
    assert torch.equal(L.qdq_rows(x3, 2).reshape(9, 4096).view(torch.int16), ref.view(torch.int16))
    with pytest.raises(ValueError):
        L.qdq_rows(torch.zeros(2, 7, dtype=torch.bfloat16), 2)


# ============================================================== tiny GLM-5.3 MLP modules
def _cfg():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    cfg = Glm5NextTextConfig(hidden_size=64, moe_intermediate_size=32, intermediate_size=96,
                             n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
                             n_group=1, topk_group=1)
    cfg._experts_implementation = "eager"
    return cfg


class _Layer(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


def _randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _n, p in [*module.named_parameters(), *module.named_buffers()]:
            p.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
    return module


def _moe_layer(seed=1):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE
    return _Layer(_randomize(Glm5NextTextMoE(_cfg()).to(torch.bfloat16), seed)).eval()


def _dense_layer(seed=2):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP
    return _Layer(_randomize(Glm5NextTextMLP(_cfg()).to(torch.bfloat16), seed)).eval()


def _spec_order():
    spec = json.loads((PQ / "prismaquant" / "model_profiles" / "specs" / "glm5_next.json").read_text())
    return tuple(spec["packed_experts"]["projection_splits"]["gate_up_proj"])


def _per_expert_source(layer, L_idx, seed=3):
    """Checkpoint-style per-expert tensors, packed into the live layer by the streamed loader's
    own bridge (prismaquant.layer_streaming._pack_per_expert_into_packed, PQ 7882eda3), with
    the projection order the GLM spec declares."""
    sys.path.insert(0, str(PQ))
    from prismaquant.layer_streaming import _pack_per_expert_into_packed
    ex = layer.mlp.experts
    E, twoI, H = ex.gate_up_proj.shape
    I = twoI // 2
    g = torch.Generator().manual_seed(seed)
    prefix = f"model.language_model.layers.{L_idx}.mlp.experts"
    src = {}
    for e in range(E):
        for role, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            src[f"{prefix}.{e}.{role}.weight"] = torch.randn(shape, generator=g).to(torch.bfloat16)
    out = dict(src)
    order = _spec_order()
    pat = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")
    n = _pack_per_expert_into_packed(
        out, is_per_expert=lambda name: bool(pat.match(name)),
        parent_for_projection=lambda p: "gate_up_proj" if p in order else ("down_proj" if p == "down_proj" else None),
        projection_names_for=lambda param: order if param == "gate_up_proj" else (param,),
        live_param_shape={f"{prefix}.gate_up_proj": (E, twoI, H), f"{prefix}.down_proj": (E, H, I)}.get)
    assert n == 2 and set(out) == {f"{prefix}.gate_up_proj", f"{prefix}.down_proj"}
    with torch.no_grad():
        ex.gate_up_proj.copy_(out[f"{prefix}.gate_up_proj"])
        ex.down_proj.copy_(out[f"{prefix}.down_proj"])
    return src


def _rows(layer, L_idx, src=None):
    """Manifest-shaped rows for every Tessera unit of one layer, with source identities taken
    from the checkpoint-style tensors (routed) or the live Linear (shared/dense)."""
    rows = []
    mlp = layer.mlp
    base = f"model.language_model.layers.{L_idx}.mlp"
    if hasattr(mlp, "experts"):
        E = mlp.experts.gate_up_proj.shape[0]
        for e in range(E):
            for role in ("gate_proj", "up_proj", "down_proj"):
                q = f"{base}.experts.{e}.{role}"
                rows.append({"qname": q, "kind": "routed", "role": role, "expert": e,
                             "source_sha256": L.tensor_sha256(src[q + ".weight"])})
        for role in ("gate_proj", "up_proj", "down_proj"):
            rows.append({"qname": f"{base}.shared_experts.{role}", "kind": "shared", "role": role, "expert": None,
                         "source_sha256": L.tensor_sha256(getattr(mlp.shared_experts, role).weight)})
    else:
        for role in ("gate_proj", "up_proj", "down_proj"):
            rows.append({"qname": f"{base}.{role}", "kind": "dense", "role": role, "expert": None,
                         "source_sha256": L.tensor_sha256(getattr(mlp, role).weight)})
    return rows


def _snapshot(layer):
    return {n: t.detach().clone() for n, t in [*layer.named_parameters(), *layer.named_buffers()]}


def _replacements(layer, rows, seed):
    g = torch.Generator().manual_seed(seed)
    reps = {}
    for r in rows:
        v = L.unit_view(layer, r)
        reps[r["qname"]] = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        r["rendered_sha256"] = L.tensor_sha256(reps[r["qname"]])
    return reps


TESSERA_PARAMS = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj",
                  "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
                  "mlp.shared_experts.down_proj.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}


def test_unit_view_is_the_streamed_loaders_packing():
    layer = _moe_layer()
    src = _per_expert_source(layer, 5)
    assert _spec_order() == ("gate_proj", "up_proj")
    for r in _rows(layer, 5, src):
        if r["kind"] == "routed":
            assert torch.equal(L.unit_view(layer, r), src[r["qname"] + ".weight"]), r["qname"]
            assert L.tensor_sha256(L.unit_view(layer, r)) == r["source_sha256"]


@pytest.mark.parametrize("kind", ["moe", "dense"])
@pytest.mark.parametrize("subset", ["all", "every_third"])
def test_substitution_swaps_exactly_the_rows(kind, subset):
    layer = _moe_layer() if kind == "moe" else _dense_layer()
    src = _per_expert_source(layer, 7) if kind == "moe" else None
    rows = _rows(layer, 7, src)
    before = _snapshot(layer)
    reps = _replacements(layer, rows, seed=11)
    chosen = rows if subset == "all" else rows[::3]
    tensors = {r["qname"]: reps[r["qname"]] for r in chosen}
    # coverage: every element of every Tessera tensor belongs to exactly one row
    params = dict(layer.named_parameters())
    marks = {n: torch.zeros(p.shape, dtype=torch.int32) for n, p in params.items() if n in TESSERA_PARAMS}
    for r in rows:
        v = L.unit_view(layer, r)
        for n, p in params.items():
            if n in marks and v.untyped_storage().data_ptr() == p.untyped_storage().data_ptr():
                off = v.storage_offset() - p.storage_offset()
                flat = marks[n].view(-1)
                idx = torch.arange(v.numel()).reshape(v.shape)
                # strided view -> flat indices inside the parameter
                strides = torch.tensor(v.stride())
                coords = torch.stack(torch.unravel_index(idx.reshape(-1), v.shape), dim=1)
                flat.index_add_(0, off + (coords * strides).sum(dim=1), torch.ones(v.numel(), dtype=torch.int32))
    for n, m in marks.items():
        assert int(m.min()) == 1 and int(m.max()) == 1, f"{n}: rows do not partition the tensor"
    assert set(marks) == (TESSERA_PARAMS & set(params))
    res = L.substitute(layer, rows, tensors)
    assert res["copied"] == len(chosen)
    after = _snapshot(layer)
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert changed <= TESSERA_PARAMS
    for r in rows:
        v = L.unit_view(layer, r)
        if r["qname"] in tensors:
            assert torch.equal(v, tensors[r["qname"]])
        else:
            want = src[r["qname"] + ".weight"] if r["kind"] == "routed" else None
            if want is not None:
                assert torch.equal(v, want)
    for n in before:
        if n not in TESSERA_PARAMS:
            assert torch.equal(before[n], after[n]), n
    print(f"SUBSTITUTE {kind}/{subset}: rows={len(rows)} copied={res['copied']} checked={res['checked']} "
          f"changed_params={sorted(changed)}")


def test_source_gate_refuses_a_wrong_map_and_copies_nothing():
    layer = _moe_layer()
    src = _per_expert_source(layer, 9)
    rows = _rows(layer, 9, src)
    reps = _replacements(layer, rows, seed=12)
    swapped = []
    for r in rows:
        r2 = dict(r)
        if r["kind"] == "routed" and r["role"] in ("gate_proj", "up_proj"):
            r2["role"] = "up_proj" if r["role"] == "gate_proj" else "gate_proj"
        swapped.append(r2)
    before = _snapshot(layer)
    with pytest.raises(L.HashGateError):
        L.substitute(layer, swapped, reps)
    after = _snapshot(layer)
    assert all(torch.equal(before[n], after[n]) for n in before)


# ============================================================== the W+A hooks
def test_wa_hooks_quantize_exactly_the_tessera_gemm_inputs_and_remove_cleanly():
    layer = _moe_layer(seed=21)
    _per_expert_source(layer, 4, seed=22)
    mlp = layer.mlp
    x = (torch.randn(1, 6, 64, generator=torch.Generator().manual_seed(23))).to(torch.bfloat16)
    with torch.no_grad():
        plain = mlp(x)
    seen = {}

    def rec_pre(name):
        def f(_m, args):
            seen.setdefault(name + ":in", []).append(args[0].detach().clone())
        return f

    def rec_post(name):
        def f(_m, args, _out):
            seen.setdefault(name + ":executed", []).append(args[0].detach().clone())
        return f

    remove = L.install_wa_hooks(layer, tp=2)
    handles = []
    for name, mod in (("experts", mlp.experts), ("shared.gate", mlp.shared_experts.gate_proj),
                      ("shared.up", mlp.shared_experts.up_proj), ("shared.down", mlp.shared_experts.down_proj)):
        handles.append(mod.register_forward_pre_hook(rec_pre(name), prepend=True))
        handles.append(mod.register_forward_hook(rec_post(name)))
    gate_calls = []
    wrapped = mlp.experts._apply_gate

    def spy(gate_up):
        out = wrapped(gate_up)
        gate_calls.append((gate_up.detach().clone(), out.detach().clone()))
        return out

    mlp.experts._apply_gate = spy
    with torch.no_grad():
        hooked = mlp(x)
    mlp.experts._apply_gate = wrapped
    for h in handles:
        h.remove()
    # column-parallel inputs: whole-row per-token QDQ
    for name in ("experts", "shared.gate", "shared.up"):
        for raw, ex in zip(seen[name + ":in"], seen[name + ":executed"]):
            assert torch.equal(ex, L.qdq_rows(raw, 1)), name
            assert not torch.equal(ex, raw)
    # row-parallel inputs: per contiguous half
    for raw, ex in zip(seen["shared.down:in"], seen["shared.down:executed"]):
        assert torch.equal(ex, L.qdq_rows(raw, 2))
    # routed down input: the gate output, QDQ per half, against the class's own _apply_gate
    assert gate_calls
    cls_gate = type(mlp.experts)._apply_gate
    for gu, out in gate_calls:
        assert torch.equal(out, L.qdq_rows(cls_gate(mlp.experts, gu), 2))
    # only Tessera GEMMs carry hooks: router and everything else untouched
    hooked_mods = {n for n, m in layer.named_modules() if m._forward_pre_hooks}
    assert hooked_mods == {"mlp.experts", "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
                           "mlp.shared_experts.down_proj"}, hooked_mods
    assert "_apply_gate" in vars(mlp.experts)
    remove()
    assert not any(m._forward_pre_hooks for m in layer.modules())
    assert "_apply_gate" not in vars(mlp.experts)
    with torch.no_grad():
        again = mlp(x)
    assert torch.equal(again, plain)
    assert not torch.equal(hooked, plain)


def test_wa_hooks_on_a_dense_layer():
    layer = _dense_layer(seed=31)
    mlp = layer.mlp
    x = torch.randn(1, 5, 64, generator=torch.Generator().manual_seed(32)).to(torch.bfloat16)
    with torch.no_grad():
        plain = mlp(x)
    seen = {}
    remove = L.install_wa_hooks(layer, tp=2)
    hs = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(mlp, name)
        hs.append(mod.register_forward_pre_hook(lambda _m, a, n=name: seen.__setitem__(n + ":in", a[0].clone()),
                                                prepend=True))
        hs.append(mod.register_forward_hook(lambda _m, a, _o, n=name: seen.__setitem__(n + ":ex", a[0].clone())))
    with torch.no_grad():
        mlp(x)
    for h in hs:
        h.remove()
    assert torch.equal(seen["gate_proj:ex"], L.qdq_rows(seen["gate_proj:in"], 1))
    assert torch.equal(seen["up_proj:ex"], L.qdq_rows(seen["up_proj:in"], 1))
    assert torch.equal(seen["down_proj:ex"], L.qdq_rows(seen["down_proj:in"], 2))
    remove()
    with torch.no_grad():
        assert torch.equal(mlp(x), plain)


# ============================================================== (b) the hash gate, real wires
def _manifest():
    if not MANIFEST.exists():
        pytest.fail(f"manifest {MANIFEST} not built yet")
    return json.loads(MANIFEST.read_text())


_M = None


def M():
    global _M
    if _M is None:
        _M = _manifest()
        _M["by_q"] = {r["qname"]: r for r in _M["rows"]}
    return _M


def _root(arm, loc):
    """The directory a manifest wire location is read from (v2 rows carry an explicit root)."""
    key = loc.get("root", arm)
    return {"t8r": M()["t8r_export"], "a8": M()["a8_export"], "wirecache": M().get("wire_cache")}[key]


def _blob(row, arm):
    loc = row[arm + "_wire"]
    root = _root(arm, loc)
    off, n = L.member_location(loc)
    return L.read_range(os.path.join(root, loc["shard"]), off, n)


def _source_tensor(qname):
    idx = json.loads((SOURCE / "model.safetensors.index.json").read_text())["weight_map"]
    name = qname + ".weight"
    path = SOURCE / idx[name]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    a, b = meta["data_offsets"]
    raw = L.read_range(str(path), 8 + n + a, b - a)
    assert meta["dtype"] == "BF16"
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])


def _decoder():
    from tessera.unit_artifact import read_unit_artifact
    return read_unit_artifact


def test_manifest_is_complete_and_bound_to_the_payload():
    m = M()
    assert m["payload_sha256"] == PAYLOAD_SHA
    assert m["problems"] == []
    assert m["n_rows"] == 36423 == len(m["rows"])
    per_layer = {int(k): v for k, v in m["rows_per_layer"].items()}
    assert per_layer == {**{i: 3 for i in range(3)}, **{i: 867 for i in range(3, 45)}}
    assert set(m["fused_member_names"]) == {"gate_proj", "up_proj", "down_proj"}
    for r in m["rows"]:
        for arm in ("t8r", "a8"):
            assert re.fullmatch(r"[0-9a-f]{64}", r[arm + "_rendered_sha256"])
            assert r[arm + "_member_bytes_equals_priced"] is True
        assert re.fullmatch(r"[0-9a-f]{64}", r["source_sha256"])
    print("MANIFEST counts", m["counts_kind_format"])


@pytest.mark.parametrize("arm", ["t8r", "a8"])
def test_manifest_covers_every_wire_the_panel_forward_executes(arm):
    m = M()
    root = Path(m[arm + "_export"])
    wm = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    wires = {n for n in wm if n.endswith(".wire") or n.endswith(".wire_bytes")}
    used = {r[arm + "_wire"]["tensor"] for r in m["rows"]}
    assert used <= wires, sorted(used - wires)[:5]
    extra = sorted(wires - used)
    body = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
    executed_extra = [n for n in extra if body.match(n) and int(body.match(n).group(1)) < 45]
    print(f"WIRES {arm}: export={len(wires)} manifest={len(used)} extra={len(extra)} "
          f"extra_in_body={len(executed_extra)} extra_names={json.dumps(extra[:60])}")
    assert executed_extra == []


def test_framing_restatement_agrees_with_tessera_parse_fused():
    from tessera.fused import parse_fused
    m = M()
    for q in ("model.language_model.layers.10.mlp.shared_experts.gate_proj",
              "model.language_model.layers.0.mlp.gate_proj",
              "model.language_model.layers.5.mlp.experts.7.down_proj"):
        loc = m["by_q"][q]["t8r_wire"]
        data = L.read_range(os.path.join(m["t8r_export"], loc["shard"]), loc["offset"], loc["length"])
        mine = L.unwrap_members(data)
        theirs = {mm.name: mm.blob for mm in parse_fused(data)}
        assert mine == theirs, q
        off, n = L.member_location(loc)
        assert data[off - loc["offset"]: off - loc["offset"] + n] == theirs[loc["member"]]


def _flip(blob, at):
    b = bytearray(blob)
    b[at] ^= 0x5A
    return bytes(b)


def test_hash_gate_accepts_the_right_blob_and_refuses_every_corruption():
    decode = _decoder()
    m = M()
    q = "model.language_model.layers.5.mlp.experts.0.gate_proj"
    q_other = "model.language_model.layers.5.mlp.experts.1.gate_proj"
    row, other = m["by_q"][q], m["by_q"][q_other]
    blob = _blob(row, "t8r")
    good = L.decode_gated(blob, row["t8r_rendered_sha256"], q, decoder=decode)
    outcomes = {}
    # a valid blob of another unit decodes cleanly and is refused by the gate alone
    wrong = decode(_blob(other, "t8r"), device="cpu").to(torch.bfloat16)
    assert wrong.shape == good.shape
    with pytest.raises(L.HashGateError):
        L.check_identity(wrong, row["t8r_rendered_sha256"], q)
    outcomes["wrong_unit"] = "HashGateError"
    # one element nudged by one bf16 ulp
    nudged = good.clone()
    nudged.view(torch.int16).view(-1)[12345] ^= 1      # one bf16 ulp in one element
    assert not torch.equal(nudged, good)
    with pytest.raises(L.HashGateError):
        L.check_identity(nudged, row["t8r_rendered_sha256"], q)
    outcomes["one_element"] = "HashGateError"
    # corrupted and truncated blobs: refused by the decoder or by the gate, never accepted
    for label, bad in (("flip_head", _flip(blob, 3)), ("flip_mid", _flip(blob, len(blob) // 2)),
                       ("flip_last", _flip(blob, len(blob) - 1)), ("truncated", blob[:-17]),
                       ("empty", b"")):
        try:
            L.decode_gated(bad, row["t8r_rendered_sha256"], q, decoder=decode)
        except L.HashGateError:
            outcomes[label] = "HashGateError"
        except Exception as exc:  # the decoder's own refusal
            outcomes[label] = type(exc).__name__
        else:
            pytest.fail(f"{label}: a corrupted blob was accepted")
    print("HASH-GATE outcomes", outcomes)


def test_hash_gate_on_a_real_shape_layer_copies_only_verified_bytes():
    """One real GLM-5.3 routed expert (layer 5, expert 0) installed from the BF16 source into a
    packed slot: the source gate proves the slot, the right decode is copied, a wrong-unit decode
    is refused with the slot bitwise unchanged."""
    decode = _decoder()
    m = M()
    base = "model.language_model.layers.5.mlp.experts.0"
    srcs = {role: _source_tensor(f"{base}.{role}") for role in ("gate_proj", "up_proj", "down_proj")}
    I, H = srcs["gate_proj"].shape

    class Ex(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.cat([srcs["gate_proj"], srcs["up_proj"]])[None].clone(),
                                                   requires_grad=False)
            self.down_proj = torch.nn.Parameter(srcs["down_proj"][None].clone(), requires_grad=False)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Ex()

    layer = _Layer(Mlp())
    rows = []
    for role in ("gate_proj", "up_proj", "down_proj"):
        r = dict(m["by_q"][f"{base}.{role}"])
        assert r["expert"] == 0 and r["kind"] == "routed"
        r["rendered_sha256"] = r["t8r_rendered_sha256"]
        rows.append(r)
    before = _snapshot(layer)
    res = L.substitute(layer, rows, {}, check_rendered=False)
    assert res == {"checked": 3, "copied": 0}
    decoded = {r["qname"]: L.decode_gated(_blob(r, "t8r"), r["t8r_rendered_sha256"], r["qname"], decoder=decode)
               for r in rows}
    bad = dict(decoded)
    bad[rows[0]["qname"]] = decoded[rows[1]["qname"]]     # up's decode offered as gate's
    with pytest.raises(L.HashGateError):
        L.substitute(layer, rows, bad)
    after = _snapshot(layer)
    assert all(torch.equal(before[n], after[n]) for n in before)
    res = L.substitute(layer, rows, decoded)
    assert res == {"checked": 6, "copied": 3}
    for r in rows:
        assert torch.equal(L.unit_view(layer, r), decoded[r["qname"]])
    rel = {r["role"]: float((decoded[r["qname"]].float() - srcs[r["role"]].float()).norm()
                            / srcs[r["role"]].float().norm()) for r in rows}
    print(f"REAL-LAYER L5e0 {m['by_q'][base + '.gate_proj']['t8r_format']} I={I} H={H} decoded_rel_err={rel}")


# ============================================================== (d) the pre-registered law-fit pick
LAWFIT_CSV = Path("/mnt/shared/tessera-measurements/surrogate-diag-20260929/out9/lawfit_total_pick.csv")
LAWFIT_CSV_SHA = "2a1c7ddabcb2b14175fdad1e80359758bace0a43fa4ca7f491c651e63e6e2b4a"
LAWFIT_JSON = Path("/mnt/shared/tessera-measurements/surrogate-diag-20260929/out9/lawfit.json")


def test_lawfit_manifest_is_exactly_the_preregistered_pick():
    import csv
    import hashlib
    m = M()
    assert m["schema"] == "surrogate-diag.g3.unit_manifest.v2"
    assert m["lawfit_csv_sha256"] == LAWFIT_CSV_SHA
    assert hashlib.sha256(LAWFIT_CSV.read_bytes()).hexdigest() == LAWFIT_CSV_SHA
    pick = {r["qname"]: r["format"] for r in csv.DictReader(LAWFIT_CSV.open(newline=""))}
    assert len(pick) == len(m["rows"]) == 36423
    roots = {"t8r": 0, "a8": 0, "wirecache": 0}
    for r in m["rows"]:
        assert r["lawfit_format"] == pick[r["qname"]]
        assert re.fullmatch(r"[0-9a-f]{64}", r["lawfit_rendered_sha256"])
        assert r["lawfit_member_bytes_equals_priced"] is True
        loc = r["lawfit_wire"]
        roots[loc["root"]] += 1
        if loc["root"] == "t8r":
            assert r["lawfit_format"] == r["t8r_format"]
            assert r["lawfit_rendered_sha256"] == r["t8r_rendered_sha256"]
            assert {k: v for k, v in loc.items() if k != "root"} == r["t8r_wire"]
        elif loc["root"] == "a8":
            assert r["lawfit_format"] == r["a8_format"] != r["t8r_format"]
            assert r["lawfit_rendered_sha256"] == r["a8_rendered_sha256"]
            assert {k: v for k, v in loc.items() if k != "root"} == r["a8_wire"]
        else:
            assert r["lawfit_format"] not in (r["t8r_format"], r["a8_format"])
            assert loc["shard"] == r["qname"].replace(".", "__") + "__" + r["lawfit_format"] + ".tessera"
            assert loc["offset"] == 0 and loc["member"] is None and loc["length"] == loc["member_bytes"]
    assert roots == m["lawfit_roots"]
    fit = json.loads(LAWFIT_JSON.read_text())["d_candidate"]
    assert fit["csv_sha256"] == LAWFIT_CSV_SHA
    total = sum(r["lawfit_priced_wire_bytes"] for r in m["rows"])
    print(f"LAWFIT roots={roots} priced_wire_bytes={total} dp_wire_bytes={fit['wire_bytes']:.0f} "
          f"budget={fit['budget']:.0f} wirecache_units={len(m['lawfit_wirecache_units'])}")
    assert total == int(fit["wire_bytes"])


def _lawfit_probe_rows():
    """One law-fit row per (root, kind, role) class, deterministic: the first in manifest order."""
    seen = {}
    for r in M()["rows"]:
        key = (r["lawfit_wire"]["root"], r["kind"], r["role"])
        seen.setdefault(key, r)
    return seen


def test_lawfit_rows_decode_to_the_priced_rendered_identity_through_the_runners_reader():
    import g3_offline_decoded_kl as R
    assert R.ARMS["lawfit_w"] == ("lawfit", False) and R.ARMS["lawfit_wa"] == ("lawfit", True)
    decode = _decoder()
    m = M()
    probes = _lawfit_probe_rows()
    assert {k[0] for k in probes} == {"t8r", "a8", "wirecache"}
    wc = {k for k in probes if k[0] == "wirecache"}
    assert {(k[1], k[2]) for k in wc} >= {("routed", "down_proj"), ("shared", "gate_proj"),
                                          ("shared", "up_proj"), ("shared", "down_proj"),
                                          ("dense", "down_proj")}, sorted(wc)
    reader = R.WireReader("lawfit", {}, {"t8r": m["t8r_export"], "a8": m["a8_export"],
                                         "wirecache": m["wire_cache"]})
    for key, r in sorted(probes.items(), key=lambda kv: str(kv[0])):
        blob = reader._read(r)
        assert blob == _blob(r, "lawfit")
        assert len(blob) == r["lawfit_priced_wire_bytes"]
        dec = L.decode_gated(blob, r["lawfit_rendered_sha256"], r["qname"], decoder=decode)
        print(f"LAWFIT-DECODE {key} {r['qname']} {r['lawfit_format']} {tuple(dec.shape)} ok")
    # a wire-cache blob of another format of the same Linear is refused by the gate
    k = next(k for k in sorted(wc) if k[1] == "routed")
    r = probes[k]
    other = m["by_q"][r["qname"]]
    wrong = decode(_blob(other, "t8r"), device="cpu").to(torch.bfloat16)
    with pytest.raises(L.HashGateError):
        L.check_identity(wrong, r["lawfit_rendered_sha256"], r["qname"])


# ============================================================== (e) the multi-arm single pass
PASS_ARMS = ["null", "a8_w", "a8_wa", "lawfit_w", "lawfit_wa", "t8r_w", "t8r_wa"]


def test_multi_plan_decodes_each_distinct_rendered_identity_once_per_layer():
    import g3_offline_decoded_kl as R
    by_layer = {}
    for r in M()["rows"]:
        by_layer.setdefault(int(r["layer"]), []).append(r)
    plan, totals = R.plan_arms(PASS_ARMS, by_layer)
    print("MULTI-PLAN", json.dumps(totals))
    assert totals["a8_w"] == {"decoded": 36423, "carried": 0}
    assert totals["lawfit_w"]["decoded"] == 20001 and totals["t8r_w"]["decoded"] == 9573
    for a in ("a8_wa", "lawfit_wa", "t8r_wa"):
        assert totals[a] == {"decoded": 0, "carried": 36423}
    assert sum(t["decoded"] for t in totals.values()) == 65997
    # every arm ends each layer with exactly its own rendered identity on every row
    for layer, steps in plan.items():
        rows = by_layer[layer]
        installed = [None] * len(rows)
        for k, arm in enumerate(PASS_ARMS):
            key = R.ARMS[arm][0]
            for i in steps[k]["decode"]:
                installed[i] = rows[i][key + "_rendered_sha256"]
            if key is not None:
                assert installed == [r[key + "_rendered_sha256"] for r in rows], (layer, arm)
    with pytest.raises(SystemExit):
        R.plan_arms(["a8_w", "null"], by_layer)
    with pytest.raises(SystemExit):
        R.plan_arms(["null", "a8_w", "a8_w"], by_layer)


def test_multi_installer_puts_each_arms_weights_on_every_row_and_gates_every_decode(monkeypatch):
    import g3_offline_decoded_kl as R
    layer = _moe_layer()
    src = _per_expert_source(layer, 4)
    rows = _rows(layer, 4, src)
    g = torch.Generator().manual_seed(21)
    table = {}
    for i, r in enumerate(rows):
        v = L.unit_view(layer, r)
        a8 = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        t8 = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        own = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        lf = (a8, t8, own)[i % 3]            # law-fit shares with A8, with T8R, or is its own
        for key, t in (("a8", a8), ("t8r", t8), ("lawfit", lf)):
            blob = f"{key}|{r['qname']}".encode()
            table[blob] = t
            r[key + "_rendered_sha256"] = L.tensor_sha256(t)
            r[key + "_wire"] = {"root": key, "shard": "x", "offset": 0, "length": len(blob),
                                "member": None, "member_bytes": len(blob)}
    monkeypatch.setattr(R.WireReader, "_read", lambda self, row: f"{self.arm}|{row['qname']}".encode())
    decoder = lambda blob, device="cpu": table[blob].clone()
    by_layer = {4: rows}
    plan, totals = R.plan_arms(PASS_ARMS, by_layer)
    inst = R.MultiInstaller(None, PASS_ARMS, by_layer, plan, {"a8": "/", "t8r": "/", "lawfit": "/"},
                            decoder, 2)
    views = [L.unit_view(layer, r) for r in rows]
    inst.source_gate(4, views)
    for k, arm in enumerate(PASS_ARMS):
        rec = inst.install(4, k, views)
        key = R.ARMS[arm][0]
        for r, v in zip(rows, views):
            want = src[r["qname"] + ".weight"] if key is None and r["kind"] == "routed" else (
                None if key is None else table[f"{key}|{r['qname']}".encode()])
            if want is not None:
                assert torch.equal(v, want), (arm, r["qname"])
            if key is not None:
                assert L.tensor_sha256(v) == r[key + "_rendered_sha256"], (arm, r["qname"])
        assert rec["decoded"] == len(plan[4][k]["decode"])
    c = inst.counts
    assert c["a8_w"]["rendered_verified"] == len(rows) and c["a8_wa"]["rendered_carried"] == len(rows)
    assert c["lawfit_w"]["rendered_verified"] == sum(1 for i in range(len(rows)) if i % 3 != 0)
    # after the law-fit arm, T8R re-decodes the rows where the law-fit pick is not T8R's (A8's or its own)
    assert c["t8r_w"]["rendered_verified"] == sum(1 for i in range(len(rows)) if i % 3 != 1)
    assert all(c[a]["source_verified"] == len(rows) for a in PASS_ARMS)
    print("MULTI-INSTALL", json.dumps(c))
    # a decode that does not hash to its arm's rendered identity aborts the install
    table[f"a8|{rows[5]['qname']}".encode()] = table[f"t8r|{rows[5]['qname']}".encode()]
    inst2 = R.MultiInstaller(None, PASS_ARMS, by_layer, plan, {"a8": "/", "t8r": "/", "lawfit": "/"},
                             decoder, 2)
    layer2 = _moe_layer()
    _per_expert_source(layer2, 4)
    views2 = [L.unit_view(layer2, r) for r in rows]
    inst2.source_gate(4, views2)
    inst2.install(4, 0, views2)
    with pytest.raises(L.HashGateError):
        inst2.install(4, 1, views2)


@pytest.mark.parametrize("layer", [5, 6])
def test_multi_layer_visit_runs_every_arm_once_profiles_only_the_first_quantized_install(tmp_path, layer):
    """The per-layer body of the multi-arm pass (visit_layer_arms), on CPU, with the run's seven arms.

    Regression: on the profile layer the first quantized arm's install is profiled and the arms after
    it must still install, forward and unhook (mp1 150a7959 crashed there with
    AttributeError("'bool' object has no attribute '__exit__'")).
    """
    import g3_offline_decoded_kl as R
    arms = ["null", "a8_w", "a8_wa", "lawfit_w", "lawfit_wa", "t8r_w", "t8r_wa"]
    events = []

    class Inst:
        def install(self, layer_, k, views):
            events.append(("install", arms[k]))
            torch.ones(64, 64) @ torch.ones(64, 64)      # some CPU work inside the profiled region
            return {"arm": arms[k], "decoded": 0, "install_s": 0.0}

    hooked = []

    def wa_hooks(mod):
        hooked.append(True)
        events.append(("hook", len(hooked)))

        def remove():
            hooked.pop()
            events.append(("unhook", len(hooked)))
        return remove

    inputs = [torch.zeros(3), torch.zeros(3)]

    def forward_batch(tokens):
        events.append(("fwd", len(hooked)))

    done = []
    recs = R.visit_layer_arms(layer, torch.nn.Linear(2, 2), arms, Inst(), [], inputs, forward_batch,
                              profile_layer=5, out=tmp_path, wa_hooks=wa_hooks, sync=lambda: events.append(("sync",)),
                              after_arm=done.append)
    assert [r["arm"] for r in recs] == arms and done == list(range(len(arms)))
    assert [e[1] for e in events if e[0] == "install"] == arms
    assert all(isinstance(r["forward_s"], float) for r in recs)
    # each arm: install, [hook], fwd x len(inputs), sync, [unhook]; hooks exactly on the W+A arms
    per_arm, cur = [], None
    for e in events:
        if e[0] == "install":
            cur = [e]
            per_arm.append(cur)
        else:
            cur.append(e)
    for arm, seq in zip(arms, per_arm):
        wa = R.ARMS[arm][1]
        want = ([("install", arm)] + ([("hook", 1)] if wa else []) + [("fwd", 1 if wa else 0)] * len(inputs)
                + [("sync",)] + ([("unhook", 0)] if wa else []))
        assert seq == want, (arm, seq)
    assert not hooked
    traces = sorted(p.name for p in tmp_path.iterdir())
    profiled = [r["arm"] for r in recs if "profile_trace" in r]
    if layer == 5:
        assert profiled == ["a8_w"] and traces == ["install-L5-a8_w.trace.json"]
        assert "profile_top" in recs[1] and all("profile_top" not in r for r in recs[2:])
    else:
        assert profiled == [] and traces == []


class _FakeCtx:
    """The runner-context surface SourcePrefetchKeeper touches, with a scripted memory-refusal count."""

    def __init__(self, refuse=0, decline=0):
        import threading
        self._inflight, self._inflight_lock = {}, threading.Lock()
        self.cached = set()
        self.layer_cache = type("C", (), {"peek": lambda _s, L: L in self.cached})()
        self.prefetch_memory_skips = 0
        self.refuse, self.decline = refuse, decline      # scheduler refusals; worker declines after submit
        self.scheduled, self.released = [], []

    def schedule_prefetch(self, L):
        if self.refuse > 0:
            self.refuse -= 1
            self.prefetch_memory_skips += 1
            return None
        from concurrent.futures import Future
        f = Future()
        self._inflight[L] = f
        self.scheduled.append(L)
        return f

    def settle(self):
        """What the runner's worker does at its start: decline (drop its own in-flight entry) or read."""
        if self.decline > 0 and self._inflight:
            self.decline -= 1
            self.prefetch_memory_skips += 1
            self._inflight.clear()

    def release_completed_layer(self, L):
        self.released.append(L)


def _keeper(ctx, **kw):
    import g3_offline_decoded_kl as R
    calls = {"empty": 0, "sleep": 0}

    def empty():
        calls["empty"] += 1

    def sleep(s):
        calls["sleep"] += 1
        ctx.settle()
    return R.SourcePrefetchKeeper(ctx, 45, empty_cache=empty, mem_available=lambda: 12.5, sleep=sleep, **kw), calls


def test_source_keeper_records_the_runners_schedule_and_does_nothing_else():
    ctx = _FakeCtx()
    ctx.schedule_prefetch(8)                     # the runner scheduled 8 when it installed 7
    k, calls = _keeper(ctx)
    rec = {}
    assert k.ensure(7, "runner", rec, schedule=False)
    for where in ("visit_start", "after_null", "visit_end"):
        assert k.ensure(7, where, rec, wait=(where == "visit_end"))
    sp = rec["src_prefetch"]
    assert sp["by"] == "runner" and sp["attempts"] == 0 and calls == {"empty": 0, "sleep": 0}
    assert ctx.scheduled == [8]


def test_source_keeper_retries_a_refused_schedule_at_each_arm_boundary():
    ctx = _FakeCtx(refuse=3)                     # the runner's own attempt and the next two are refused
    ctx.refuse -= 1                               # (the runner's attempt at install consumed one)
    k, calls = _keeper(ctx)
    rec = {}
    assert not k.ensure(20, "runner", rec, schedule=False)
    assert not k.ensure(20, "visit_start", rec)
    assert not k.ensure(20, "after_null", rec)
    assert k.ensure(20, "after_a8_w", rec)
    sp = rec["src_prefetch"]
    assert sp["by"] == "after_a8_w" and sp["attempts"] == 3 and ctx.scheduled == [21]
    assert calls["empty"] == 3 and set(sp["mem_available_at"]) == {"visit_start", "after_null", "after_a8_w"}
    assert sp["skips_after"] - sp["skips_before"] == 2
    assert k.ensure(20, "visit_end", rec, wait=True) and sp["by"] == "after_a8_w"


def test_source_keeper_reschedules_after_the_worker_declines():
    ctx = _FakeCtx(decline=1)                    # scheduled, then the worker's own memory check declines
    k, _calls = _keeper(ctx)
    rec = {}
    assert not k.ensure(3, "visit_start", rec)
    assert rec["src_prefetch"]["by"] is None and ctx.scheduled == [4]
    assert k.ensure(3, "after_null", rec)
    assert rec["src_prefetch"]["by"] == "after_null" and ctx.scheduled == [4, 4]


def test_source_keeper_waits_then_fails_closed_and_needs_nothing_after_the_last_layer():
    ctx = _FakeCtx(refuse=10 ** 6)
    k, calls = _keeper(ctx, wait_s=0.05, poll_s=0.0)
    rec = {}
    assert not k.ensure(30, "visit_end", rec, wait=True)
    assert rec["src_prefetch"]["by"] is None and rec["src_prefetch"]["attempts"] >= 1 and not ctx.scheduled
    rec = {}
    assert k.ensure(44, "visit_start", rec) and rec["src_prefetch"]["by"] == "none-needed"
    assert k.release(-1) is None
    ctx.cached.add(3)
    assert k.release(3) is True and k.release(4) is False
    assert ctx.released == [3, 4]


def test_multi_installer_holds_at_most_one_layer_of_wires_per_arm(monkeypatch):
    import g3_offline_decoded_kl as R
    layers = {3: _moe_layer(seed=1), 4: _moe_layer(seed=2), 6: _moe_layer(seed=3)}
    by_layer, table = {}, {}
    g = torch.Generator().manual_seed(5)
    for Li, layer in layers.items():
        src = _per_expert_source(layer, Li)
        rows = _rows(layer, Li, src)
        for i, r in enumerate(rows):
            v = L.unit_view(layer, r)
            a8 = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
            t8 = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
            lf = t8 if (Li == 4 or i % 2) else a8       # layer 4: law-fit = T8R everywhere, so T8R decodes nothing there
            for key, t in (("a8", a8), ("t8r", t8), ("lawfit", lf)):
                blob = f"{key}|{r['qname']}".encode()
                table[blob] = t
                r[key + "_rendered_sha256"] = L.tensor_sha256(t)
                r[key + "_wire"] = {"root": key, "shard": "x", "offset": 0, "length": len(blob),
                                    "member": None, "member_bytes": len(blob)}
        by_layer[Li] = rows
    reads = []

    def _read(self, row):
        reads.append((self.arm, row["qname"].split(".layers.")[1].split(".")[0]))
        return f"{self.arm}|{row['qname']}".encode()
    monkeypatch.setattr(R.WireReader, "_read", _read)
    decoder = lambda blob, device="cpu": table[blob].clone()
    plan, _totals = R.plan_arms(PASS_ARMS, by_layer)
    inst = R.MultiInstaller(None, PASS_ARMS, by_layer, plan, {"a8": "/", "t8r": "/", "lawfit": "/"}, decoder, 2)
    assert not any(inst.pending_layers().values())
    inst.prefetch_first()
    first = inst.pending_layers()
    assert first["a8_w"] == [3] and first["lawfit_w"] == [3] and first["t8r_w"] == [3]
    for Li in (3, 4, 6):
        views = [L.unit_view(layers[Li], r) for r in by_layer[Li]]
        inst.source_gate(Li, views)
        for k, arm in enumerate(PASS_ARMS):
            inst.install(Li, k, views)
            pend = inst.pending_layers()
            for a, ls in pend.items():
                assert len(ls) <= 1, (Li, arm, pend)          # one layer of wire bytes per arm, at most
                if ls:
                    assert ls[0] >= Li, (Li, arm, pend)
            if R.ARMS[arm][0] is not None and arm in inst.readers:
                r = inst.readers[arm]
                nxt = r.next_layer(Li)
                assert pend[arm] == ([nxt] if nxt is not None else []), (Li, arm, pend)
            key = R.ARMS[arm][0]
            if key is not None:
                assert all(L.tensor_sha256(v) == r_[key + "_rendered_sha256"] for v, r_ in zip(views, by_layer[Li]))
    assert not any(inst.pending_layers().values())
    assert ("t8r", "4") not in reads                          # T8R had nothing to decode on layer 4
    assert inst.counts["t8r_w"]["rendered_carried"] >= len(by_layer[4])


def test_multi_visit_layer_wires_keeper_installer_and_arms(tmp_path):
    """The multi-arm visitor end to end on CPU with fakes: release L-1, keep L+1 in flight,
    gate the source once, run all arms, record and log the layer."""
    import g3_offline_decoded_kl as R
    events = []

    class Inst:
        def source_gate(self, layer, views):
            events.append(("gate", layer, len(views)))
            return {"source_gate_s": 0.0, "source_hashes": len(views)}

        def install(self, layer, k, views):
            events.append(("install", PASS_ARMS[k]))
            return {"arm": PASS_ARMS[k], "decoded": 0, "install_s": 0.0}

        def pending_layers(self):
            return {"a8_w": []}

    class Teachers:
        started = []

        def start(self, i):
            self.started.append(i)

    ctx = _FakeCtx(refuse=1)
    keeper, _calls = _keeper(ctx)
    runner = type("Rn", (), {"layers": [torch.nn.Linear(2, 2) for _ in range(45)]})()
    layer_records = []
    fwd = []
    rec = R.multi_visit_layer(
        9, lambda t: fwd.append(1), None, runner=runner, inst=Inst(), keeper=keeper,
        by_layer={9: [{"qname": "a"}, {"qname": "b"}]}, arms=PASS_ARMS, inputs=[torch.zeros(2)] * 3,
        teachers=Teachers(), last=44, profile_layer=5, out=tmp_path, layer_records=layer_records,
        unit_view=lambda mod, r: r["qname"], wa_hooks=lambda m: (lambda: None), sync=lambda: None,
        max_allocated=lambda: 123)
    assert layer_records == [rec] and rec["layer"] == 9 and rec["units"] == 2 and rec["cuda_max_allocated"] == 123
    assert ctx.released == [8] and ctx.scheduled == [10] and rec["released_prev_was_cached"] is False
    assert rec["src_prefetch"]["by"] == "after_null" and rec["src_prefetch"]["attempts"] == 2
    assert {"runner", "visit_start", "after_null"} <= set(rec["src_prefetch"]["mem_available_at"])
    assert rec["cuda_memory_reserved"] is None
    assert [e for e in events if e[0] == "gate"] == [("gate", 9, 2)]
    assert [e[1] for e in events if e[0] == "install"] == PASS_ARMS
    assert len(fwd) == 3 * len(PASS_ARMS) and not Teachers.started
    rec = R.multi_visit_layer(
        44, lambda t: None, None, runner=runner, inst=Inst(), keeper=keeper,
        by_layer={44: [{"qname": "a"}]}, arms=PASS_ARMS, inputs=[torch.zeros(2)],
        teachers=Teachers(), last=44, profile_layer=5, out=tmp_path, layer_records=layer_records,
        unit_view=lambda mod, r: r["qname"], wa_hooks=lambda m: (lambda: None), sync=lambda: None,
        max_allocated=lambda: 1)
    assert Teachers.started == [0] and rec["src_prefetch"]["by"] == "none-needed" and ctx.released == [8, 43]


# ============================================================== (f) placement arms (codec-decomp 2026-09-30)
PLACEMENT_ARMS = ["a8_routed_only_w", "a8_routed_only_wa", "a8_shared_src_w", "a8_shared_src_wa", "a8_w", "a8_wa"]


def test_wa_hooks_skip_passthrough_units_and_nothing_else():
    layer = _moe_layer(seed=41)
    _per_expert_source(layer, 4, seed=42)
    mlp = layer.mlp
    x = torch.randn(1, 6, 64, generator=torch.Generator().manual_seed(43)).to(torch.bfloat16)
    with torch.no_grad():
        plain_shared = mlp.shared_experts(x)
    remove = L.install_wa_hooks(layer, tp=2, skip={"shared"})
    hooked = {n for n, m in layer.named_modules() if m._forward_pre_hooks}
    assert hooked == {"mlp.experts"}, hooked                 # routed input still quantized
    assert "_apply_gate" in vars(mlp.experts)                 # routed down input still quantized
    with torch.no_grad():
        assert torch.equal(mlp.shared_experts(x), plain_shared)   # passthrough: BF16 in, no QDQ
    remove()
    assert not any(m._forward_pre_hooks for m in layer.modules()) and "_apply_gate" not in vars(mlp.experts)
    # a dense layer kept as passthrough gets no hooks at all
    dense = _dense_layer(seed=44)
    with torch.no_grad():
        plain = dense.mlp(x)
    remove = L.install_wa_hooks(dense, tp=2, skip={"shared", "dense"})
    assert not any(m._forward_pre_hooks for m in dense.modules())
    with torch.no_grad():
        assert torch.equal(dense.mlp(x), plain)
    remove()
    # without skip the dense layer is hooked (the kinds are independent)
    remove = L.install_wa_hooks(dense, tp=2, skip={"shared"})
    assert {n for n, m in dense.named_modules() if m._forward_pre_hooks} == {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    remove()
    with pytest.raises(ValueError):
        L.install_wa_hooks(layer, tp=2, skip={"attention"})
    with pytest.raises(ValueError):
        L.install_wa_hooks(layer, tp=2, skip={("routed", "down_proj")})


def test_placement_plan_on_the_real_manifest():
    import g3_offline_decoded_kl as R
    by_layer = {}
    for r in M()["rows"]:
        by_layer.setdefault(int(r["layer"]), []).append(r)
    n_shared = sum(1 for r in M()["rows"] if r["kind"] == "shared")
    n_dense = sum(1 for r in M()["rows"] if r["kind"] == "dense")
    n_routed = sum(1 for r in M()["rows"] if r["kind"] == "routed")
    assert (n_routed, n_shared, n_dense) == (36288, 126, 9)
    plan, totals = R.plan_arms(PLACEMENT_ARMS, by_layer)
    print("PLACEMENT-PLAN", json.dumps(totals))
    assert totals["a8_routed_only_w"] == {"decoded": n_routed, "carried": 0, "source_kept": n_shared + n_dense}
    assert totals["a8_routed_only_wa"] == {"decoded": 0, "carried": n_routed, "source_kept": n_shared + n_dense}
    assert totals["a8_shared_src_w"] == {"decoded": n_dense, "carried": n_routed, "source_kept": n_shared}
    assert totals["a8_shared_src_wa"] == {"decoded": 0, "carried": n_routed + n_dense, "source_kept": n_shared}
    assert totals["a8_w"] == {"decoded": n_shared, "carried": n_routed + n_dense}
    assert totals["a8_wa"] == {"decoded": 0, "carried": 36423}
    # replaying the plan leaves every arm on exactly its own identities (source for kept kinds)
    for layer, steps in plan.items():
        rows = by_layer[layer]
        installed = [None] * len(rows)
        for k, arm in enumerate(PLACEMENT_ARMS):
            for i in steps[k]["decode"]:
                installed[i] = rows[i]["a8_rendered_sha256"]
            assert installed == R.arm_wants(arm, rows), (layer, arm)
            for i in steps[k]["decode"]:
                assert rows[i]["kind"] not in R.SRC_KINDS.get(arm, ()), (layer, arm)
    # an order that would need a source slice back is refused
    with pytest.raises(SystemExit):
        R.plan_arms(["a8_w", "a8_shared_src_w"], by_layer)
    with pytest.raises(SystemExit):
        R.plan_arms(["a8_shared_src_w", "a8_routed_only_w"], by_layer)


def test_placement_installer_keeps_source_rows_bitwise_and_counts_them(monkeypatch):
    import g3_offline_decoded_kl as R
    layer = _moe_layer(seed=51)
    src = _per_expert_source(layer, 4, seed=52)
    rows = _rows(layer, 4, src)
    g = torch.Generator().manual_seed(53)
    table = {}
    for r in rows:
        v = L.unit_view(layer, r)
        t = (torch.randn(v.shape, generator=g) * 0.01).to(v.dtype)
        blob = f"a8|{r['qname']}".encode()
        table[blob] = t
        r["a8_rendered_sha256"] = L.tensor_sha256(t)
        r["a8_wire"] = {"root": "a8", "shard": "x", "offset": 0, "length": len(blob), "member": None,
                        "member_bytes": len(blob)}
    monkeypatch.setattr(R.WireReader, "_read", lambda self, row: f"{self.arm}|{row['qname']}".encode())
    decoder = lambda blob, device="cpu": table[blob].clone()
    by_layer = {4: rows}
    arms = ["a8_routed_only_w", "a8_routed_only_wa", "a8_shared_src_w", "a8_w"]
    plan, _ = R.plan_arms(arms, by_layer)
    inst = R.MultiInstaller(None, arms, by_layer, plan, {"a8": "/"}, decoder, 2)
    views = [L.unit_view(layer, r) for r in rows]
    shared_src = {r["qname"]: L.unit_view(layer, r).clone() for r in rows if r["kind"] == "shared"}
    inst.source_gate(4, views)
    for k, arm in enumerate(arms):
        rec = inst.install(4, k, views)
        for r, v in zip(rows, views):
            if r["kind"] in R.SRC_KINDS.get(arm, ()):
                assert torch.equal(v, shared_src[r["qname"]]), (arm, r["qname"])   # untouched source
            else:
                assert L.tensor_sha256(v) == r["a8_rendered_sha256"], (arm, r["qname"])
        n_sh = sum(1 for r in rows if r["kind"] == "shared")
        assert rec.get("source_kept", 0) == (n_sh if R.SRC_KINDS.get(arm) else 0)
    c = inst.counts
    assert c["a8_routed_only_w"]["source_kept"] == 3 and c["a8_routed_only_w"]["copied"] == len(rows) - 3
    assert c["a8_shared_src_w"]["copied"] == 0 and c["a8_w"]["copied"] == 3 and "source_kept" not in c["a8_w"]


def test_visit_layer_arms_passes_the_passthrough_kinds_to_the_wa_hooks():
    import g3_offline_decoded_kl as R
    arms = ["a8_routed_only_w", "a8_routed_only_wa", "a8_shared_src_wa", "a8_wa"]
    calls = []

    class Inst:
        def install(self, layer_, k, views):
            return {"arm": arms[k], "decoded": 0, "install_s": 0.0}

    def wa_hooks(mod, skip=None):
        calls.append(skip)
        return lambda: None

    R.visit_layer_arms(7, object(), arms, Inst(), [], [None], lambda t: None, profile_layer=-1,
                       out=Path("."), wa_hooks=wa_hooks, sync=lambda: None)
    assert calls == [frozenset({"shared", "dense"}), frozenset({"shared"}), None]


# ============================================================== (g) manifest v3 picks and the EXL3 reference (2026-09-30)
def test_wa_hooks_skip_the_routed_stack_and_single_roles():
    layer = _moe_layer(seed=61)
    _per_expert_source(layer, 4, seed=62)
    mlp = layer.mlp
    remove = L.install_wa_hooks(layer, tp=2, skip={"routed"})
    hooked = {n for n, m in layer.named_modules() if m._forward_pre_hooks}
    assert hooked == {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj"}
    assert "_apply_gate" not in vars(mlp.experts)                # T16 routed: no A-side QDQ at all
    remove()
    assert not any(m._forward_pre_hooks for m in layer.modules())
    remove = L.install_wa_hooks(layer, tp=2, skip={("shared", "down_proj")})
    hooked = {n for n, m in layer.named_modules() if m._forward_pre_hooks}
    assert hooked == {"mlp.experts", "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"}
    assert "_apply_gate" in vars(mlp.experts)
    remove()
    dense = _dense_layer(seed=64)
    remove = L.install_wa_hooks(dense, tp=2, skip={("dense", "gate_proj"), ("dense", "up_proj")})
    assert {n for n, m in dense.named_modules() if m._forward_pre_hooks} == {"mlp.down_proj"}
    remove()
    remove = L.install_wa_hooks(layer, tp=2, skip={"routed", "shared"})
    assert not any(m._forward_pre_hooks for m in layer.modules()) and "_apply_gate" not in vars(mlp.experts)
    remove()


def _v3_rows(layer_idx=5, experts=2):
    """Synthetic v3 rows: routed E4M3/T16/EXL3 formats, shared E4M3/T16 formats."""
    rows = []
    def fmts(q, kind):
        f = {"TESSERA_E4M3_K1_R1024": {"rendered_sha256": "e8" + q, "rendered_shape": [16, 16], "contract": "fp8_per_token_dynamic",
                                       "wire": {"root": "a8", "shard": "s", "offset": 0, "length": 1, "member_bytes": 1}},
             "TESSERA_BF16_K1_R1024": {"rendered_sha256": "b16" + q, "rendered_shape": [16, 16], "contract": "bf16_unquantized",
                                       "wire": {"root": "wirecache", "shard": "w", "offset": 0, "length": 1, "member_bytes": 1}}}
        if kind == "routed":
            f["EXL3"] = {"rendered_sha256": "x3" + q, "rendered_shape": [128, 256], "contract": "exl3_weight_only",
                         "wire": {"root": "exl3", "ranges": [["a", 0, 1]], "member_bytes": 1, "wire_sha256": "z"}}
        return f
    for e in range(experts):
        for role in ("gate_proj", "up_proj", "down_proj"):
            q = f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.{role}"
            rows.append({"qname": q, "kind": "routed", "role": role, "expert": e, "layer": layer_idx,
                         "formats": fmts(q, "routed")})
    for role in ("gate_proj", "up_proj", "down_proj"):
        q = f"model.language_model.layers.{layer_idx}.mlp.shared_experts.{role}"
        rows.append({"qname": q, "kind": "shared", "role": role, "expert": None, "layer": layer_idx,
                     "formats": fmts(q, "shared")})
    return rows


def _pick_manifest(rows, picks):
    for r in rows:
        r["pick"] = {n: f(r) for n, f in picks.items()}
    return {"schema": "surrogate-diag.g3.unit_manifest.v3", "picks": sorted(picks), "rows": rows}


def test_register_picks_materialises_each_rows_identity_and_contract(monkeypatch):
    import g3_offline_decoded_kl as R
    monkeypatch.setattr(R, "ARMS", dict(R.ARMS))
    rows = _v3_rows()
    m = _pick_manifest(rows, {
        "zx": lambda r: "EXL3" if r["kind"] == "routed" else "SOURCE",
        "zs": lambda r: "TESSERA_E4M3_K1_R1024" if r["kind"] == "routed" or r["role"] == "down_proj" else "SOURCE"})
    assert R.register_picks(m) == ["zs", "zx"]
    assert R.ARMS["zx_w"] == ("pick_zx", False) and R.ARMS["zs_wa"] == ("pick_zs", True)
    r0, sh_gate, sh_down = rows[0], rows[-3], rows[-1]
    assert r0["pick_zx_format"] == "EXL3" and r0["pick_zx_rendered_sha256"] == "x3" + r0["qname"]
    assert r0["pick_zx_rendered_shape"] == [128, 256] and r0["pick_zx_wire"]["root"] == "exl3"
    assert sh_gate["pick_zx_contract"] == "source_bf16" and "pick_zx_rendered_sha256" not in sh_gate
    assert R.arm_wants("zx_w", rows) == ["x3" + r["qname"] for r in rows[:-3]] + [None] * 3
    assert R.arm_wants("zs_wa", rows)[-3:] == [None, None, "e8" + sh_down["qname"]]
    # W+A hooks: E4M3 rows quantized; SOURCE projections skipped one role at a time
    assert R.hook_skip("zs_wa", rows) == frozenset({("shared", "gate_proj"), ("shared", "up_proj")})
    with pytest.raises(SystemExit):
        R.hook_skip("zx_wa", rows)                     # EXL3 has no G3 activation model: w arm only
    with pytest.raises(SystemExit):
        R.register_picks(m)                            # names collide on a second registration


def test_hook_skip_takes_the_routed_stack_as_one_unit(monkeypatch):
    import g3_offline_decoded_kl as R
    monkeypatch.setattr(R, "ARMS", dict(R.ARMS))
    rows = _v3_rows()
    m = _pick_manifest(rows, {
        "t16": lambda r: "TESSERA_BF16_K1_R1024" if r["kind"] == "routed" else "TESSERA_E4M3_K1_R1024",
        "mix": lambda r: "TESSERA_BF16_K1_R1024" if r["kind"] == "routed" and r["expert"] == 0 else
               ("TESSERA_E4M3_K1_R1024" if r["kind"] == "routed" else "TESSERA_BF16_K1_R1024")})
    R.register_picks(m)
    assert R.hook_skip("t16_wa", rows) == frozenset({"routed"})
    with pytest.raises(SystemExit):
        R.hook_skip("mix_wa", rows)                    # one contract per routed stack per layer
    assert R.hook_skip("a8_wa", rows) == frozenset() and R.hook_skip("a8_shared_src_wa", rows) == frozenset({"shared"})
    # legacy placement arms keep their kind-level skip
    assert R.hook_skip("a8_routed_only_wa", []) == frozenset({"shared", "dense"})


def test_pick_plan_orders_decodes_and_keeps_source_rows(monkeypatch):
    import g3_offline_decoded_kl as R
    monkeypatch.setattr(R, "ARMS", dict(R.ARMS))
    rows = _v3_rows()
    E, T = "TESSERA_E4M3_K1_R1024", "TESSERA_BF16_K1_R1024"
    m = _pick_manifest(rows, {
        "exl3": lambda r: "EXL3" if r["kind"] == "routed" else "SOURCE",
        "t16r": lambda r: T if r["kind"] == "routed" else "SOURCE",
        "a8s16k": lambda r: T if r["kind"] == "routed" else ("SOURCE" if r["role"] != "down_proj" else E),
        "a8s": lambda r: E if r["kind"] == "routed" or r["role"] == "down_proj" else "SOURCE"})
    R.register_picks(m)
    arms = ["exl3_w", "t16r_w", "a8s16k_wa", "a8s_w", "a8s_wa"]
    plan, totals = R.plan_arms(arms, {5: rows})
    n_r = sum(1 for r in rows if r["kind"] == "routed")
    assert totals["exl3_w"] == {"decoded": n_r, "carried": 0, "source_kept": 3}
    assert totals["t16r_w"] == {"decoded": n_r, "carried": 0, "source_kept": 3}
    assert totals["a8s16k_wa"] == {"decoded": 1, "carried": n_r, "source_kept": 2}
    assert totals["a8s_w"] == {"decoded": n_r, "carried": 1, "source_kept": 2}
    assert totals["a8s_wa"] == {"decoded": 0, "carried": n_r + 1, "source_kept": 2}
    with pytest.raises(SystemExit):
        R.plan_arms(["a8s_w", "exl3_w"], {5: rows})     # exl3 needs the shared down back at source


def test_exl3_wire_is_read_from_its_ranges_and_gated_on_the_prepass_framing(monkeypatch):
    import hashlib
    import g3_offline_decoded_kl as R
    store = {("/e/a", 10, 3): b"abc", ("/e/b", 0, 2): b"de", ("/e/a", 99, 4): b"fghi"}
    monkeypatch.setattr(R.L, "read_range", lambda path, off, n: store[(path, off, n)])
    loc = {"root": "exl3", "ranges": [["a", 10, 3], ["b", 0, 2], ["a", 99, 4]], "member_bytes": 9,
           "wire_sha256": hashlib.sha256(b"abcdefghi").hexdigest()}
    row = {"qname": "q", "pick_x_wire": loc}
    rd = R.WireReader("pick_x", {}, {"exl3": "/e"})
    assert rd._read(row) == b"abcdefghi"
    row["pick_x_wire"] = dict(loc, wire_sha256="0" * 64)
    with pytest.raises(L.HashGateError):
        rd._read(row)
    row["pick_x_wire"] = dict(loc, member_bytes=8)
    with pytest.raises(L.HashGateError):
        rd._read(row)
    with pytest.raises(SystemExit):
        R.WireReader("pick_x", {}, {"exl3": None})._read({"qname": "q", "pick_x_wire": loc})


def test_installer_decodes_exl3_rows_with_exl3_torch_and_others_with_the_tessera_reader():
    import g3_offline_decoded_kl as R
    from g3_pq_policy import g3_exl3 as X
    g = torch.Generator().manual_seed(71)
    in_f, out_f = 256, 128
    suh = (torch.randn(in_f, generator=g) * 0.1).half()
    svh = (torch.randn(out_f, generator=g) * 0.1).half()
    tr = torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 64), generator=g, dtype=torch.int32).to(torch.int16)
    mcg = torch.tensor([0x4BAC23ED], dtype=torch.int32)
    blob = X.pack_wire(suh.numpy().tobytes(), svh.numpy().tobytes(), tr.numpy().tobytes(), mcg.numpy().tobytes())
    inst = R.MultiInstaller.__new__(R.MultiInstaller)
    inst.decoder = lambda b, device="cpu": ("tessera", b)
    row = {"pick_x_format": "EXL3", "pick_x_rendered_shape": [out_f, in_f]}
    got = inst.decode_unit("pick_x", row, blob, "cpu")
    want = X.effective_weight(tr, suh, svh)
    assert got.shape == (out_f, in_f) and torch.equal(got, want)
    assert inst.decode_unit("pick_x", {"pick_x_format": "TESSERA_E4M3_K1_R1024"}, b"w", "cpu") == ("tessera", b"w")
    assert inst.decode_unit("a8", {}, b"w", "cpu") == ("tessera", b"w")
    with pytest.raises(ValueError):
        inst.decode_unit("pick_x", row, blob[:-1], "cpu")        # a wire of the wrong length is refused


def test_visit_layer_arms_passes_a_picks_per_layer_skip(monkeypatch):
    import g3_offline_decoded_kl as R
    monkeypatch.setattr(R, "ARMS", dict(R.ARMS))
    rows = _v3_rows()
    m = _pick_manifest(rows, {"p": lambda r: "TESSERA_BF16_K1_R1024" if r["kind"] == "routed" else
                              ("SOURCE" if r["role"] == "down_proj" else "TESSERA_E4M3_K1_R1024")})
    R.register_picks(m)
    arms = ["p_w", "p_wa", "a8_wa"]
    calls = []

    class Inst:
        def install(self, layer_, k, views):
            return {"arm": arms[k], "decoded": 0, "install_s": 0.0}

    def wa_hooks(mod, skip=None):
        calls.append(skip)
        return lambda: None

    recs = R.visit_layer_arms(5, object(), arms, Inst(), [], [None], lambda t: None, profile_layer=-1,
                              out=Path("."), wa_hooks=wa_hooks, sync=lambda: None, rows=rows)
    assert calls == [frozenset({"routed", ("shared", "down_proj")}), None]
    assert recs[1]["hook_skip"] == sorted(map(str, {"routed", ("shared", "down_proj")}))
