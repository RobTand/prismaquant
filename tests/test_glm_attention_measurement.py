"""Full-draw collector measurement on CPU projection paths, NOT GLM kernel parity."""
import argparse
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors.torch import save_file

from prismaquant import glm_mtp_capture
from prismaquant import tessera_calibration_cache as cc
from prismaquant import tessera_campaign as tc
from prismaquant.calibration_data import load_calibration_input
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.perturbed_x_cache import activation_cache_filename
from prismaquant.tessera_hessian import calibration_identity, token_ids_sha256
from test_glm_attention_census import KDA_LEAVES, MLA_LEAVES, _contract, _prefix, _runtime


class _KDA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "b_proj", "o_proj"):
            setattr(self, name, torch.nn.Linear(4, 4, bias=False))
        self.forget_gate = torch.nn.Module()
        self.forget_gate.f_a_proj = torch.nn.Linear(4, 2, bias=False)
        self.forget_gate.f_b_proj = torch.nn.Linear(2, 4, bias=False)
        self.g_a_proj = torch.nn.Linear(4, 2, bias=False)
        self.g_b_proj = torch.nn.Linear(2, 4, bias=False)

    def forward(self, x):
        forget = self.forget_gate.f_b_proj(self.forget_gate.f_a_proj(x).tanh()).sigmoid()
        gate = self.g_b_proj(self.g_a_proj(x).tanh()).sigmoid()
        mixed = self.q_proj(x) * self.k_proj(x) + self.v_proj(x)
        return self.o_proj(mixed * self.b_proj(x).sigmoid() * forget * gate)


class _MLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_a_proj = torch.nn.Linear(4, 2, bias=False)
        self.q_b_proj = torch.nn.Linear(2, 4, bias=False)
        self.kv_a_proj_with_mqa = torch.nn.Linear(4, 3, bias=False)
        self.kv_b_proj = torch.nn.Linear(2, 8, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        q = self.q_b_proj(self.q_a_proj(x))
        compressed, rotary = self.kv_a_proj_with_mqa(x).split((2, 1), dim=-1)
        k, v = self.kv_b_proj(compressed).chunk(2, dim=-1)
        return self.o_proj((q * k + rotary).sigmoid() * v)


class _ProjectionPath(torch.nn.Module):
    """Toy token-local mixing; deliberately no upstream attention kernel or model."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList()
        for kind in (_KDA, _KDA, _MLA, _MLA):
            layer = torch.nn.Module()
            layer.self_attn = kind()
            self.model.language_model.layers.append(layer)
        self.batches = []
        with torch.no_grad():
            for index, module in enumerate(m for m in self.modules() if isinstance(m, torch.nn.Linear)):
                values = torch.arange(module.weight.numel()).reshape_as(module.weight)
                module.weight.copy_(((values + index) % 7 + 1) / 32)

    def forward(self, ids):
        self.batches.append(ids.detach().clone())
        x = torch.stack((ids.float() / 256, (ids % 13).float() / 13,
                         (ids % 7).float() / 7, torch.ones_like(ids).float()), dim=-1)
        for layer in self.model.language_model.layers:
            x = x + layer.self_attn(x) / 8
        return x


@pytest.fixture
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.usefixtures("one_cpu_thread")
def test_full_draw_attention_measurement_and_publication(tmp_path):
    ids = (torch.arange(512 * 512, dtype=torch.int64) % 251).reshape(512, 512)
    ids[-1, -1] = 4095  # A late outlier makes a prefix-only maximum observably wrong.
    calibration = calibration_identity("synthetic projection-path fixture", [ids], fit_tokens=ids.numel())
    draw_path = tmp_path / "draw.safetensors"
    provenance = dict(calibration, nsamples=512, seqlen=512)
    save_file({"calibration_ids": ids}, str(draw_path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    ids, calibration["calibration_input"] = load_calibration_input(
        draw_path, expected_sha256=hashlib.sha256(draw_path.read_bytes()).hexdigest(),
        n_samples=512, seqlen=512)
    model = _ProjectionPath().eval()
    targets = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    assert set(targets) == {_prefix(i) + leaf for i in range(4)
                            for leaf in (KDA_LEAVES if i < 2 else MLA_LEAVES)}
    units = {name: list(module.weight.shape) for name, module in targets.items()}

    # Independent reference pass: record ALL inputs once, then do full FP64 Grams.
    # No collector callback, row_consumer, per-batch Gram, or capped-row oracle.
    reference = {}
    def record(name):
        def hook(_module, args):
            reference[name] = args[0].detach().reshape(-1, args[0].shape[-1]).clone()
        return hook
    handles = [module.register_forward_pre_hook(record(name)) for name, module in targets.items()]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for handle in handles:
            handle.remove()
    assert set(reference) == set(units)
    expected_h = {name: x.double().T @ x.double() for name, x in reference.items()}
    expected_max = {name: x.abs().max().item() for name, x in reference.items()}
    model.batches.clear()
    rows, hessians, counts, maxima = tc._collect_activations(
        model, targets, ids.split(16), max_rows=257, device="cpu", want_hessian=True)
    assert set(rows) == set(hessians) == set(counts) == set(maxima) == set(units)
    for name, x in reference.items():
        assert x.shape == (262144, units[name][1])
        assert counts[name] == 262144
        assert rows[name].shape == (257, units[name][1])
        torch.testing.assert_close(rows[name], x[:257], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(hessians[name].double(), expected_h[name], rtol=2e-5, atol=1e-3)
        assert maxima[name] == pytest.approx(expected_max[name], rel=3e-6, abs=1e-7)
        assert maxima[name] > rows[name].abs().max().item()

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    save_file(model.state_dict(), str(source / "model.safetensors"))
    contract = _contract()
    first = next(iter(targets))
    body = tc.calibration_census(
        {"body": ids.numel()}, {"body": expected_max[first]},
        args=argparse.Namespace(model=str(source), nsamples=512, seqlen=512, seed=0, layer_stride=1),
        groups={"u:body": ["body"]}, dense_targets=["body"], expert_targets=[],
        shapes={"body": units[first]}, identity=calibration, model_load_contract=contract,
        attention_implementation="eager", capture_runtime=_runtime())
    assert len(model.batches) == 32
    assert all(batch.shape == (16, body["seqlen"]) for batch in model.batches)
    observed = torch.cat(model.batches)
    assert observed.shape == (body["nsamples"], body["seqlen"])
    assert torch.equal(observed, ids)
    assert token_ids_sha256(model.batches) == body["fit_ids_sha256"]
    body_path = tmp_path / "body-census.json"
    body_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    ref = {"path": str(body_path), "sha256": hashlib.sha256(body_path.read_bytes()).hexdigest()}
    body_identity = cc.capture_identity(
        body_path, calibration=calibration, max_act_rows=257,
        model_load_contract=contract, attention_implementation="eager")
    body_receipt = cc.publish_capture(
        tmp_path / "body-capture", census_path=body_path, identity=body_identity,
        acts={"body": rows[first]}, hessians={"body": hessians[first]},
        counts={"body": counts[first]}, maxima={"body": maxima[first]})
    roster = tc.campaign_roster(
        sorted(units), Glm5NextProfile(), allow_pinned=tc.GLM_ATTENTION_ALLOW_PINNED,
        pinned_roster_only=True)
    lift = argparse.Namespace(allow_pinned=tc.GLM_ATTENTION_ALLOW_PINNED, pinned_roster_only=True)
    census = glm_mtp_capture.attention_census(
        base_census=body, base_census_ref=ref, units=units, counts=counts, max_abs=maxima,
        groups={f"u:{name}": [name] for name in units}, model_load_contract=contract,
        attention_implementation="eager", capture_runtime=_runtime(),
        pinned_roster=tc.pinned_roster_block(lift, roster))
    tc.require_census_draw(census, calibration, where="full immutable attention draw")
    assert census["calibration_input"] == calibration["calibration_input"]
    root, census_path = tmp_path / "attention-capture", tmp_path / "attention-census.json"
    with cc.CaptureSourceAuthentication(
            str(source), body_identity, {}, manifest_sha256=body_receipt["sha256"]) as owner:
        identity, digest, sealed = glm_mtp_capture.publish_attention_capture(
            root, census=census, census_path=census_path, source_authentication=owner,
            calibration=calibration, max_act_rows=257, rows=rows, hessians=hessians,
            counts=counts, max_abs=maxima, completed_contract=contract)
    assert digest == hashlib.sha256(census_path.read_bytes()).hexdigest()
    assert census["counts"] == counts and census["max_abs"] == maxima
    assert census["attention_extension"]["base_census"] == ref
    assert census["fit_ids_sha256"] == token_ids_sha256(model.batches)
    manifest = cc.require_capture_contract(sealed["path"], sealed["sha256"])
    assert manifest["identity"] == identity
    assert identity["units"] == units and set(manifest["entries"]) == set(units)
    for name, shape in units.items():
        stored = torch.load(root / "inputs" / activation_cache_filename(name), weights_only=True)
        assert list(stored["hessian"].shape) == [shape[1], shape[1]]
        assert stored["count"] == 262144 and stored["inputs"].shape == (257, shape[1])
        torch.testing.assert_close(stored["hessian"].double(), expected_h[name], rtol=2e-5, atol=1e-3)
        torch.testing.assert_close(stored["inputs"], rows[name], rtol=0, atol=0)
        assert stored["max_abs"] == maxima[name]
