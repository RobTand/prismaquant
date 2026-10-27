#!/usr/bin/env python3
"""Quality benchmark for the v22/v23 MiniMax-M2.7 artifact.

Runs:
  1. Perplexity on the embedded illustrative text (no spec-decode per
     `validator_spec_decode_caveat` memory — echo+logprobs returns the
     DRAFT model's NLL with spec-decode on).
  2. Sample generations on agentic / math / code prompts to verify
     coherent output across the cal-mix domains.

Talks to vLLM via the OpenAI-compatible /v1/completions endpoint
served by launch-minimax-v21-vllm-serve.sh.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismaquant.validate_quantized_model import scored_echo_logprobs

VLLM = "http://localhost:8000/v1"
MODEL = "minimax-m2.7-prismaquant"


def perplexity_on_text(text: str, max_tokens: int = 1024) -> tuple[float, int]:
    """Tokenize via vLLM's echo-back loglikelihood mode."""
    r = requests.post(
        f"{VLLM}/completions",
        json={
            "model": MODEL,
            "prompt": text[:max_tokens * 4],  # rough char->tok upper bound
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
            "temperature": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    out = r.json()
    if not out.get("choices"):
        raise RuntimeError(f"empty choices: {out}")
    scored = scored_echo_logprobs(
        out["choices"][0]["logprobs"]["token_logprobs"],
        where="perplexity response",
    )
    avg_nll = -sum(scored) / len(scored)
    import math
    return math.exp(avg_nll), len(scored)


def sample_generations(prompts: list[str], max_tokens: int = 256) -> list[str]:
    out: list[str] = []
    for p in prompts:
        r = requests.post(
            f"{VLLM}/completions",
            json={
                "model": MODEL,
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            timeout=300,
        )
        r.raise_for_status()
        out.append(r.json()["choices"][0]["text"])
    return out


SMOKE_TEXTS = [
    # Wikitext-style snippet for perplexity (~500 tokens worth)
    """The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.""",
]

PROMPT_AGENTIC = "You are a helpful agent. The user wants to send a calendar invite for a 30-minute meeting tomorrow at 2pm. List the steps you would take in order, briefly:"
PROMPT_MATH = "Solve step by step: A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance and average speed?"
PROMPT_CODE = "Write a Python function `is_palindrome(s: str) -> bool` that returns True if `s` is a palindrome ignoring case and non-alphanumeric characters."


def main() -> int:
    print(f"[bench] hitting {VLLM} model={MODEL}")
    # Warm up
    try:
        r = requests.get(f"{VLLM}/models", timeout=30)
        r.raise_for_status()
        print(f"[bench] models: {[m['id'] for m in r.json()['data']]}")
    except Exception as e:
        print(f"[bench] FAILED to reach vLLM: {e!r}")
        return 1

    print("\n[1] Perplexity on wikitext-style sample ...")
    t0 = time.time()
    try:
        ppl, n_tok = perplexity_on_text(SMOKE_TEXTS[0])
    except Exception as e:
        print(f"[bench] FAILED perplexity response: {e!r}")
        return 1
    print(f"  ppl={ppl:.3f} on {n_tok} tokens ({time.time()-t0:.1f}s)")

    print("\n[2] Sample generations:")
    try:
        samples = sample_generations(
            [PROMPT_AGENTIC, PROMPT_MATH, PROMPT_CODE],
            max_tokens=256,
        )
        for label, text in zip(["AGENTIC", "MATH", "CODE"], samples):
            print(f"\n--- {label} ---")
            print(text.strip()[:600])
            print("---")
    except Exception as e:
        print(f"[bench] FAILED generation response: {e!r}")
        return 1

    # This smoke report applies no quality acceptance threshold.
    print(f"\n[summary] ppl={ppl:.3f}  generations={len(samples)} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
