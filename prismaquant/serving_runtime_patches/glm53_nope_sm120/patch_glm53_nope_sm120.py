#!/usr/bin/env python3
"""Make GLM-5.3 NoPE MLA (qk_rope_head_dim=0) serve on FLASHINFER_MLA_SPARSE_SM120.

Applied on top of eugr/spark-vllm@sha256:0afec8d4... (vLLM 0.28.1rc1.dev397,
flashinfer 0.6.18) for GB10 / sm121.

This is NOT the MiaAI-Lab 576-pad overlay. That overlay targets an older vLLM
base whose flashinfer had no native no-rope sparse-MLA model type, so it padded
the query into the 576-wide GLM_NSA geometry and trimmed the candidate set to
2048. This image's flashinfer already instantiates _MODEL_TYPE_GLM53_NOPE at
d_qk=512 / topk 2176 (flashinfer/mla/_sparse_mla_sm120.py:392-419,
_sparse_mla_sm120_plan.py:173), and vLLM already routes GLM to
kv_scale_format="arbitrary_fp32" (flashinfer_mla_sparse_sm120.py:26-29), so the
native route is taken and only the two gaps below remain.

Gap 1 (the measured blocker): the SM120 impl inherits do_kv_cache_update from
AttentionImpl (vllm/v1/attention/backend.py:1056), which calls
concat_and_cache_mla with the model's own k_pe. With qk_rope_head_dim=0 the
CUDA kernel refuses -- "pe_dim must be 64 for fp8_ds_mla" (cache_kernels.cu).
The packed record is 656 B (512 fp8 latent + 16 B scales + 128 B rope) for both
DSV3_2/GLM_NSA and GLM53_NOPE, so writing a 64-wide zero rope block fills the
region the no-rope reader ignores.

Gap 2: the native no-rope decode kernel takes a per-query-token active top-k
length and rejects zero-length rows, and the page table is the kpool buffer
width (2176), not index_topk (2048). The SM100 sibling in this same image
already does this (flashinfer_mla_sparse.py:378, 405-427, 458-476); the SM120
impl still passes seq_lens=None and sparse_mla_top_k=attn_metadata.topk_tokens.
This mirrors the SM100 handling, gated on is_nope_mla so the DSv3.2 path on
SM120 is byte-identical to before.
"""

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
target = SITE / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
source = target.read_text()
applied = []


def replace_once(tag: str, old: str, new: str) -> None:
    global source
    n = source.count(old)
    if n != 1:
        raise SystemExit(f"[{tag}] expected exactly 1 target, found {n}: {old!r}")
    source = source.replace(old, new)
    applied.append(tag)


# --- 1. is_nope_mla flag -----------------------------------------------------
replace_once(
    "init_is_nope_mla",
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        from vllm.config import get_current_vllm_config\n",
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        # Native no-rope MLA (GLM-5.3): flashinfer resolves d_qk=512 +\n"
    "        # arbitrary_fp32 to _MODEL_TYPE_GLM53_NOPE, which needs a per-query\n"
    "        # active top-k length and a zero rope block in the packed record.\n"
    "        self.is_nope_mla = self.qk_rope_head_dim == 0\n"
    "        from vllm.config import get_current_vllm_config\n",
)

# --- 2. valid top-k counts + page-table capacity -----------------------------
replace_once(
    "forward_valid_counts",
    "        topk_indices_physical = cast(\n"
    "            torch.Tensor,\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "            ),\n"
    "        )\n",
    "        extra_kwargs: dict[str, torch.Tensor] = {}\n"
    "        empty_rows: torch.Tensor | None = None\n"
    "        if self.is_nope_mla:\n"
    "            topk_indices_physical, topk_lengths = cast(\n"
    "                tuple[torch.Tensor, torch.Tensor],\n"
    "                triton_convert_req_index_to_global_index(\n"
    "                    attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                    attn_metadata.block_table,\n"
    "                    topk_indices,\n"
    "                    BLOCK_SIZE=attn_metadata.block_size,\n"
    "                    NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "                    return_valid_counts=True,\n"
    "                ),\n"
    "            )\n"
    "            # The page table is the kpool buffer width (index_topk widened by\n"
    "            # kpool-1 and rounded to a multiple of 128), not index_topk; the\n"
    "            # kernel treats sparse_mla_top_k as capacity and bounds each row\n"
    "            # by its own valid count.\n"
    "            sparse_topk_capacity = topk_indices_physical.shape[1]\n"
    "            empty_rows = topk_lengths == 0\n"
    "            topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
    "                empty_rows, 0\n"
    "            )\n"
    "            seq_lens_arg: torch.Tensor | None = topk_lengths.clamp(min=1)\n"
    '            extra_kwargs["sparse_mla_top_k_lens"] = seq_lens_arg\n'
    "        else:\n"
    "            topk_indices_physical = cast(\n"
    "                torch.Tensor,\n"
    "                triton_convert_req_index_to_global_index(\n"
    "                    attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                    attn_metadata.block_table,\n"
    "                    topk_indices,\n"
    "                    BLOCK_SIZE=attn_metadata.block_size,\n"
    "                    NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "                ),\n"
    "            )\n"
    "            sparse_topk_capacity = attn_metadata.topk_tokens\n"
    "            seq_lens_arg = None\n",
)

replace_once(
    "forward_seq_lens",
    "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
    "            seq_lens=seq_lens_arg,\n            max_seq_len=sparse_topk_capacity,\n",
)

replace_once(
    "forward_top_k",
    "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
    "            sparse_mla_top_k=sparse_topk_capacity,\n",
)

replace_once(
    "forward_extra_kwargs",
    "            kv_scale_format=self.kv_scale_format,\n        )\n"
    "        return out.squeeze(1), None\n",
    "            kv_scale_format=self.kv_scale_format,\n"
    "            **extra_kwargs,\n        )\n"
    "        out = out.squeeze(1)\n"
    "        if empty_rows is not None:\n"
    "            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
    "        return out, None\n"
    "\n"
    "    def do_kv_cache_update(\n"
    "        self,\n"
    "        kv_c_normed: torch.Tensor,\n"
    "        k_pe: torch.Tensor,\n"
    "        kv_cache: torch.Tensor,\n"
    "        slot_mapping: torch.Tensor,\n"
    "        kv_cache_dtype: str,\n"
    "        k_scale: torch.Tensor,\n"
    "    ) -> None:\n"
    "        if self.is_nope_mla:\n"
    "            # concat_and_cache_mla hardcodes pe_dim == 64 for the packed\n"
    "            # fp8_ds_mla record. The no-rope reader ignores the 128 B rope\n"
    "            # block, so write it as zeros.\n"
    "            k_pe = kv_c_normed.new_zeros((kv_c_normed.shape[0], 1, 64))\n"
    "        super().do_kv_cache_update(\n"
    "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
    "        )\n",
)

target.write_text(source)
print("PATCHED flashinfer_mla_sparse_sm120.py:", ", ".join(applied))
