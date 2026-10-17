"""The real discovery CLI can use its documented materialized-load dtype."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture
def checkpoint(tmp_path):
    from transformers import AutoConfig, AutoModelForCausalLM
    from prismaquant import genuine_weight_initialization

    config = AutoConfig.for_model(
        "qwen3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
        attention_dropout=0.0, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    with genuine_weight_initialization():
        model = AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    source = tmp_path / "source"
    model.save_pretrained(source)
    return source


@pytest.mark.parametrize("dtype,wanted", [
    pytest.param(None, "torch.bfloat16", id="default"),
    pytest.param("bf16", "torch.bfloat16", id="advertised_bf16"),
    pytest.param("bfloat16", "torch.bfloat16", id="canonical_bf16"),
    pytest.param("float32", "torch.float32", id="canonical_fp32"),
    pytest.param("fp16", "torch.float16", id="short_fp16"),
    pytest.param("float16", "torch.float16", id="canonical_fp16"),
    pytest.param("half", "torch.float16", id="torch_half"),
    pytest.param("fp32", "torch.float32", id="short_fp32"),
    pytest.param("float", "torch.float32", id="torch_float"),
    pytest.param("double", "torch.float64", id="torch_double"),
    pytest.param("float64", "torch.float64", id="canonical_fp64"),
])
def test_actual_materialized_model_walk_dtype(checkpoint, tmp_path, dtype, wanted):
    from prismaquant.model_walk import main

    output = tmp_path / "walk.json"
    args = ["--model", str(checkpoint), "--materialize", "--execution", "real",
            "--output", str(output), "--rules", "profile"]
    if dtype is not None:
        args += ["--dtype", dtype]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["context"]["materialized"] is True
    assert report["provenance"]["execution"] == "real"
    assert report["gate"]["refused"] is False
    parameters = [node for node in report["walk"]["nodes"] if node["kind"] == "parameter"]
    assert parameters
    assert all(node["dtype"] == wanted for node in parameters)
