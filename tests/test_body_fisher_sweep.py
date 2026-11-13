"""Exercise the production no-grad body forward against a reference forward."""
from __future__ import annotations

import hashlib
import struct

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from prismaquant import incremental_probe as probe, probe_statistics
from prismaquant.streaming_model import _build_streaming_context


def _forward(checkpoint, tmp_path, ids, *, act, rows_limit, tag=""):
    context = _build_streaming_context(str(checkpoint), device=torch.device("cpu"),
        dtype=torch.float32, offload_folder=str(tmp_path / f"offload{tag}"),
        layers_per_shard=1, max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0, attn_implementation="eager")
    act_rows = probe._ActRowCache(str(act), rows_limit)
    try:
        stats, _routing, _embeds, _final = probe._probe_forward(
            context, calib=ids, scope=".*", exclude="(?!)", act_rows=act_rows,
            prefetch_lookahead=1)
        act_rows.drain()
        assert all(parameter.is_meta for layer in context.layers
                   for parameter in layer.parameters())
    finally:
        context.shutdown()
    return stats


def test_real_streamed_body_rows_match_the_reference_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ACT_CACHE_WORKERS", "1")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    torch.manual_seed(123)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval()
    checkpoint = tmp_path / "model"
    reference.save_pretrained(checkpoint)
    ids = torch.tensor([[1, 3, 7, 11], [2, 5, 13, 17]], dtype=torch.int64)
    tracked = [name for name, module in reference.named_modules()
               if isinstance(module, torch.nn.Linear)]
    # Body Linear inputs of a reference forward: the oracle for the act/ rows
    # that the no-grad probe forward writes.
    body_inputs = {}
    input_hooks = [module.register_forward_hook(
        lambda _m, inp, _o, _name=name: body_inputs.__setitem__(
            _name, inp[0].detach().reshape(-1, inp[0].size(-1))))
        for name, module in reference.named_modules()
        if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)]
    try:
        with torch.no_grad():
            hidden = reference.model(input_ids=ids, use_cache=False).last_hidden_state
    finally:
        for handle in input_hooks:
            handle.remove()

    act = tmp_path / "activations"
    actual = _forward(checkpoint, tmp_path, ids, act=act, rows_limit=ids.numel())
    assert len(body_inputs) == 14
    for name, rows in body_inputs.items():
        blob = torch.load(act / (name.replace(".", "__") + ".pt"), weights_only=False)
        torch.testing.assert_close(blob["inputs"], rows[blob["row_indices"]])
    # lm_head row = sample * T + token over the scored (non-last) positions.
    head = torch.load(act / "lm_head.pt", weights_only=False)
    assert head["row_indices"].numel() == ids.size(0) * (ids.size(1) - 1)
    assert not bool((head["row_indices"] % ids.size(1) == ids.size(1) - 1).any())
    torch.testing.assert_close(
        head["inputs"],
        hidden.reshape(-1, config.hidden_size)[head["row_indices"]])
    assert set(actual) == set(tracked)
    for name in tracked:
        weight = reference.get_submodule(name).weight.detach()
        expected_tokens = ids.size(0) * (ids.size(1) - 1) if name == "lm_head" else ids.numel()
        assert actual[name]["n_tokens_seen"] == expected_tokens, name
        assert actual[name]["n_params"] == weight.numel(), name
        torch.testing.assert_close(
            torch.tensor(actual[name]["w_norm_sq"]), weight.float().square().sum(),
            msg=name)
        torch.testing.assert_close(
            torch.tensor(actual[name]["w_max_abs"]), weight.float().abs().max(),
            msg=name)


class _GroupedLinear(torch.nn.Linear):
    """A declared grouped class: `[G*R, D]` weight consumed per group."""

    def __init__(self, d_in, out_features, n_groups):
        super().__init__(d_in, out_features, bias=False)
        self.n_groups = n_groups


def test_a_grouped_linear_row_carries_num_groups_not_num_experts():
    """`num_groups` -- never `num_experts`, which downstream reads as a
    packed expert stack (`_shape_from_stats`, `_stats_indicates_packed_expert`)."""
    from prismaquant.probe_inventory import ProbeUnit

    class _Profile:
        def probe_grouped_module_class_names(self):
            return ("_GroupedLinear",)

    groups, rows_per_group, d_in = 2, 3, 4
    model = torch.nn.Module()
    model.grouped = _GroupedLinear(d_in, groups * rows_per_group, groups)
    unit = ProbeUnit("grouped", "grouped_linear", ("grouped.weight",))
    entry = probe._probe_unit_rows(model, _Profile(), unit, {})["grouped"]
    assert entry["num_groups"] == groups
    assert "num_experts" not in entry
    assert entry["in_features"] == d_in
    assert entry["out_features"] == groups * rows_per_group
    assert entry["n_params"] == groups * rows_per_group * d_in


def _scalar_priority(row: int) -> int:
    """Independent reference: hashlib key and integer fmix32, no torch."""
    def mix(x):
        x ^= x >> 16
        x = (x * 0x85EBCA6B) & 0xFFFFFFFF
        x ^= x >> 13
        x = (x * 0xC2B2AE35) & 0xFFFFFFFF
        return x ^ (x >> 16)
    digest = hashlib.blake2b(probe_statistics.ACT_ROW_SELECTION.encode(),
                             digest_size=8).digest()
    k0, k1 = struct.unpack("<II", digest)
    return mix(mix(row ^ k0) ^ k1)


def test_slice_selection_merged_by_top_k_equals_the_monolithic_selection():
    rows = torch.arange(1000)
    flat = torch.randn(1000, 4, generator=torch.Generator().manual_seed(5))
    expected = sorted(sorted(range(1000), key=_scalar_priority)[:37])
    whole_x, whole_rows = {}, {}
    probe_statistics.keep_act_rows(whole_x, whole_rows, "q", flat, rows, 37)
    assert whole_rows["q"][0].tolist() == expected
    assert torch.equal(whole_x["q"][0], flat[expected])
    merged_x, merged_rows = {}, {}
    for part in (slice(600, 1000), slice(0, 250), slice(250, 600)):
        part_x, part_rows = {}, {}
        probe_statistics.keep_act_rows(part_x, part_rows, "q", flat[part], rows[part], 37)
        probe_statistics.keep_act_rows(merged_x, merged_rows, "q",
                                       part_x["q"][0], part_rows["q"][0], 37)
    assert merged_rows["q"][0].tolist() == expected
    assert torch.equal(merged_x["q"][0], whole_x["q"][0])


def test_act_cache_is_byte_identical_under_different_global_rng_states(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ACT_CACHE_WORKERS", "1")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    checkpoint = tmp_path / "model"
    Qwen3ForCausalLM(config).save_pretrained(checkpoint)
    # 2 x 160 tokens: lm_head sees 318 rows, more than its 256-row limit.
    ids = torch.randint(1, 32, (2, 160), generator=torch.Generator().manual_seed(3))
    acts = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        act = tmp_path / f"act{seed}"
        _forward(checkpoint, tmp_path, ids, act=act, rows_limit=37, tag=str(seed))
        acts.append({p.name: p.read_bytes() for p in sorted(act.glob("*.pt"))})
    assert len(acts[0]) == 15 and acts[0] == acts[1]
    head = torch.load(tmp_path / "act1" / "lm_head.pt", weights_only=False)
    rows = head["row_indices"]
    assert head["row_selection"] == probe_statistics.ACT_ROW_SELECTION
    assert rows.numel() == 256 and torch.equal(rows, rows.unique())
    assert not bool((rows % 160 == 159).any())
