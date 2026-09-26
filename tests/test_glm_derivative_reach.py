"""The GLM KDA correction reaches only the classes that can run it (PQ #1341).

The corrected runtime refuses an unbound model only when the model holds an
instance of a class whose code can run the corrected expression. The reach is
derived from the pinned source, not declared: every definition the correction
changed, closed under the names that call, subclass or decorate with it.

The GLM MTP layer is built from this file's attention, MoE and norms and has no
KDA module. On the corrected source it computes the same bytes as on the stock
source and runs with no derivative identity. A KDA module beside it, or a
mutation that lets the correction reach its attention, is refused again.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from prismaquant import genuine_weight_initialization  # noqa: E402
from prismaquant import glm_mtp  # noqa: E402
from prismaquant import glm_source_derivative as derivative  # noqa: E402
from prismaquant.model_profiles.glm5_next import Glm5NextProfile  # noqa: E402
from tests.test_glm_mtp_layer import _randomize, _text_config  # noqa: E402

NAME = "transformers.models.glm5_next.modeling_glm5_next"
PROFILE = Glm5NextProfile()
MTP_CLASSES = {"Glm5NextTextAttention", "Glm5NextTextIndexer", "Glm5NextTextMoE",
               "Glm5NextTextExperts", "Glm5NextTextTopkRouter", "Glm5NextTextMLP",
               "Glm5NextTextRMSNorm"}


def _stock_raw():
    raw = Path(importlib.import_module(NAME).__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == derivative.ORIGINAL_MODELING_SHA256
    return raw


def _load_as_modeling(path, monkeypatch):
    """Execute ``path`` as the GLM modeling module, for this test only."""
    package = importlib.import_module("transformers.models.glm5_next")
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, NAME, module)
    monkeypatch.setattr(package, "modeling_glm5_next", module)
    spec.loader.exec_module(module)
    return module


def _mtp_layer(seed=20261341):
    with genuine_weight_initialization():
        layer = glm_mtp.Glm5NextMtpLayer(_text_config())
    _randomize(layer, seed)
    return layer.to(torch.float32).eval()


def _mtp_inputs(rows=8):
    generator = torch.Generator().manual_seed(1341)
    embeds = torch.randn(1, rows, 64, generator=generator)
    previous = torch.randn(1, rows, 64, generator=generator)
    return embeds, previous, torch.arange(rows).unsqueeze(0)


def test_the_pinned_correction_changes_only_the_kda_chunk():
    stock = _stock_raw()
    corrected = derivative.corrected_source(stock)
    before, after = derivative._definitions(stock), derivative._definitions(corrected)
    changed = {key for key in before if before[key][0] != after[key][0]}
    assert changed == {"chunk_kimi_delta_attention"}
    assert set(before) == set(after)
    reach = derivative.correction_reach(stock, corrected)
    assert "Glm5NextTextLinearAttention" in reach
    assert derivative._MODULE_LEVEL not in reach
    assert not reach & MTP_CLASSES
    assert derivative.original_source(corrected) == stock


def test_the_mtp_layer_computes_the_same_bytes_on_the_corrected_source(tmp_path, monkeypatch):
    stock_layer = _mtp_layer()
    inputs = _mtp_inputs()
    with torch.no_grad():
        expected = stock_layer(*inputs)

    path = tmp_path / "modeling_glm5_next.py"
    path.write_bytes(derivative.corrected_source(_stock_raw()))
    corrected = _load_as_modeling(path, monkeypatch)
    layer = _mtp_layer()
    assert type(layer.self_attn) is corrected.Glm5NextTextAttention
    assert type(layer.mlp) is corrected.Glm5NextTextMoE
    used = {type(m).__name__ for m in layer.modules() if type(m).__module__ == NAME}
    assert used and used <= MTP_CLASSES
    layer.load_state_dict(stock_layer.state_dict())
    with torch.no_grad():
        assert torch.equal(layer(*inputs), expected)

    model = glm_mtp.MtpCheckpointModel(layer)
    assert derivative.bind_source_derivative(model, PROFILE, None) is None
    assert derivative.source_derivative_identity(model) is None

    with torch.device("meta"):
        kda = corrected.Glm5NextTextLinearAttention(_text_config(), 0)
    body = torch.nn.ModuleDict({"mtp": model, "kda": kda})
    with pytest.raises(ValueError, match="explicit derivative binding"):
        derivative.bind_source_derivative(body, PROFILE, None)


def test_a_correction_that_reaches_the_mtp_attention_refuses_it(tmp_path, monkeypatch):
    """Mutation: the stock attention now names the corrected function."""
    header = "class Glm5NextTextAttention(nn.Module):"
    stock = _stock_raw().decode()
    assert stock.count(header) == 1
    mutated = stock.replace(header, "@(lambda cls, _=chunk_kimi_delta_attention: cls)\n" + header).encode()
    corrected = mutated.replace(derivative.ORIGINAL_EXPRESSION.encode(),
                                derivative.CORRECTED_EXPRESSION.encode(), 1)
    monkeypatch.setattr(derivative, "ORIGINAL_MODELING_SHA256", hashlib.sha256(mutated).hexdigest())
    monkeypatch.setattr(derivative, "CORRECTED_MODELING_SHA256", hashlib.sha256(corrected).hexdigest())
    assert "Glm5NextTextAttention" in derivative.correction_reach(mutated, corrected)

    path = tmp_path / "modeling_glm5_next.py"
    path.write_bytes(corrected)
    _load_as_modeling(path, monkeypatch)
    model = glm_mtp.MtpCheckpointModel(_mtp_layer())
    with pytest.raises(ValueError, match=r"explicit derivative binding \(Glm5NextTextAttention reaches"):
        derivative.bind_source_derivative(model, PROFILE, None)


def test_a_correction_beyond_the_reviewed_expression_refuses_every_model(tmp_path, monkeypatch):
    """Mutation: the corrected file also changes code outside the reviewed expression."""
    changed = derivative.corrected_source(_stock_raw()) + b"\n_UNREVIEWED = 1\n"
    monkeypatch.setattr(derivative, "CORRECTED_MODELING_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="more than the reviewed expression"):
        derivative.original_source(changed)

    path = tmp_path / "modeling_glm5_next.py"
    path.write_bytes(changed)
    _load_as_modeling(path, monkeypatch)
    model = glm_mtp.MtpCheckpointModel(_mtp_layer())
    with pytest.raises(ValueError, match="explicit derivative binding; its reach cannot be derived"):
        derivative.bind_source_derivative(model, PROFILE, None)


@pytest.mark.parametrize("site", ["module_level", "subclass", "self_method"])
def test_reach_follows_module_code_subclasses_and_methods(site):
    original = (
        "def chunk_kimi_delta_attention(g):\n"
        "    return g.exp()\n"
        "\n"
        "class Kda:\n"
        "    def forward(self, g):\n"
        "        return chunk_kimi_delta_attention(g)\n"
        "\n"
        "class Plain:\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    extra = {
        "module_level": "ALIAS = chunk_kimi_delta_attention\n",
        "subclass": "class Child(Kda):\n    pass\n",
        "self_method": "class Helper:\n    def run(self, g):\n        return chunk_kimi_delta_attention(g)\n"
                       "    def forward(self, g):\n        return self.run(g)\n",
    }[site]
    before = original + extra
    after = before.replace("return g.exp()", "return g.clamp(max=0).exp()")
    reach = derivative.correction_reach(before.encode(), after.encode())
    if site == "module_level":
        assert reach == {derivative._MODULE_LEVEL}
    else:
        assert "Kda" in reach and "Plain" not in reach
        assert ("Child" if site == "subclass" else "Helper") in reach
