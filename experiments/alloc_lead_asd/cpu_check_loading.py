"""Qualify the staged-buffer model loader against normal tiny Qwen3 loading."""
from __future__ import annotations

from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from experiments.alloc_lead_asd.run_pair import load_source_model


torch.set_num_threads(1)
torch.manual_seed(1962)
config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)
source = AutoModelForCausalLM.from_config(deepcopy(config), attn_implementation="sdpa")
state = {name: value.detach().to(torch.bfloat16).clone()
         for name, value in source.state_dict().items() if name != "lm_head.weight"}
ids = torch.tensor([[1, 4, 16, 32, 7]])
for dtype in (torch.float32, torch.bfloat16):
    reference = AutoModelForCausalLM.from_config(
        deepcopy(config), torch_dtype=dtype, attn_implementation="sdpa").eval()
    reference.load_state_dict(state, strict=False)
    reference.tie_weights()
    actual = load_source_model(deepcopy(config), state, dtype=dtype, device="cpu")
    assert actual.lm_head.weight is actual.model.embed_tokens.weight
    for name, value in actual.named_parameters():
        torch.testing.assert_close(value, state[name].to(dtype), rtol=0, atol=0)
    expected_buffers = dict(reference.named_buffers())
    for name, value in actual.named_buffers():
        assert not value.is_meta
        assert value.dtype == expected_buffers[name].dtype
        torch.testing.assert_close(value, expected_buffers[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(actual(ids, use_cache=False).logits,
                                   reference(ids, use_cache=False).logits, rtol=0, atol=0)
    print(f"PASS {dtype}: checkpoint values, tied embeddings, nonpersistent buffers, exact logits")
try:
    load_source_model(deepcopy(config), {k: v for k, v in state.items()
                                        if not k.endswith('q_proj.weight')},
                      dtype=torch.float32, device="cpu")
except RuntimeError as error:
    assert "source state coverage differs" in str(error)
    print("PASS missing checkpoint parameter refused")
else:
    raise AssertionError("missing checkpoint state was accepted")
