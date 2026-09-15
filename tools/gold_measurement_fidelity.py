"""What a gold number actually measured, stated as a field rather than implied.

Three instruments in this repository produce something called "KL", and they
are not the same estimator:

* `experiments/measure_glm_tr3_vllm.py` scores the **full** vocabulary
  (GLM-5.3: 154,880 columns) at every causal position of a sealed panel.
* `tools/measure_vllm_full_kl.py` defaults to `--score-positions final`, which
  IS full-vocabulary but only at the window-final context; under
  `--score-positions all` it is **top-K plus one tail bucket** at every
  position, K = `--prompt-top-k` (default 1024). "full" in that file's name
  means all-positions, not all-columns.
* `/home/rob/dq-runs/kl_tool.py` is top-20 on a 913-token corpus.

A reader holding one result JSON cannot tell which of those produced it, and
the three differ by more than the deltas they are used to argue about. So each
gold result carries a structured `measurement_fidelity` block naming its own
estimator. It is derived from the arguments the run actually used -- never
typed by the caller -- so it cannot describe a different run than the one it
travels with.

`vocabulary: "full"` means every column of the model's vocabulary entered the
sum. `"top_k_plus_tail"` means K columns entered exactly and the remaining
teacher mass entered as a single bucket (`measure_vllm_full_kl._position_kl`),
which is a LOWER bound on the full-vocabulary divergence, not an estimate of
it: two arms are comparable to each other at equal K, and neither is
comparable to a full-vocabulary number.
"""
from __future__ import annotations

from typing import Any

FIDELITY_SCHEMA = "prismaquant.gold_measurement_fidelity/1"


def full_kl_fidelity(
    *,
    score_positions: str,
    prompt_top_k: int | None,
    vocab_size: int | None,
) -> dict[str, Any]:
    """The estimator `tools/measure_vllm_full_kl.py` just ran.

    `score_positions` is the tool's own argument, so the two spellings below
    are the only two that exist; anything else is refused rather than
    described, because an unrecognised mode would be labelled with whichever
    branch this function guessed.
    """
    if score_positions not in {"final", "all"}:
        raise ValueError(
            "score_positions must be 'final' or 'all'; a gold result cannot "
            f"carry a fidelity label for an unknown estimator {score_positions!r}"
        )
    if vocab_size is not None and (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise ValueError("vocab_size must be a positive integer when known")

    if score_positions == "final":
        # The engine is asked for every column (`logprobs=-1`, and
        # `_logprob_vector` refuses a short row), at one position per window.
        return {
            "schema": FIDELITY_SCHEMA,
            "instrument": "tools/measure_vllm_full_kl.py",
            "positions": "window_final_context_only",
            "vocabulary": "full",
            "top_k": None,
            "tail_bucket": False,
            "vocab_size": vocab_size,
            "comparable_with": "another full-vocabulary final-position result",
        }
    if (
        isinstance(prompt_top_k, bool)
        or not isinstance(prompt_top_k, int)
        or prompt_top_k <= 0
    ):
        raise ValueError(
            "all-position scoring requires a positive --prompt-top-k: the "
            "support width IS the fidelity, so it cannot be omitted"
        )
    return {
        "schema": FIDELITY_SCHEMA,
        "instrument": "tools/measure_vllm_full_kl.py",
        "positions": "all_prompt_positions",
        "vocabulary": "top_k_plus_tail",
        "top_k": int(prompt_top_k),
        "tail_bucket": True,
        "vocab_size": vocab_size,
        "comparable_with": (
            f"another all-position result at top_k={int(prompt_top_k)}; NOT "
            "comparable with a full-vocabulary number"
        ),
    }


def wikitext_ppl_fidelity(
    *,
    n_tokens_scored: int,
    seqlen: int,
    n_chunks: int,
) -> dict[str, Any]:
    """The estimator `tools/measure_vllm_wikitext_ppl.py` just ran.

    PPL has no support width to state -- every scored position uses the target
    token's own logprob -- so what distinguishes two PPL numbers is the window
    geometry, which is what this records.
    """
    for name, value in (
        ("n_tokens_scored", n_tokens_scored),
        ("seqlen", seqlen),
        ("n_chunks", n_chunks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "schema": FIDELITY_SCHEMA,
        "instrument": "tools/measure_vllm_wikitext_ppl.py",
        "positions": "within_each_chunk_positions_1_through_N_minus_1",
        "vocabulary": "target_token_logprob_only",
        "top_k": None,
        "tail_bucket": False,
        "n_tokens_scored": int(n_tokens_scored),
        "seqlen": int(seqlen),
        "n_chunks": int(n_chunks),
        "comparable_with": (
            f"another PPL over {int(n_chunks)} x {int(seqlen)}-token windows "
            "on the same corpus"
        ),
    }
