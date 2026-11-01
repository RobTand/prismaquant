#!/usr/bin/env python3
"""Throughput benchmark for the v22/v23 MiniMax-M2.7 artifact.

Drives a few prompt sizes through vLLM's nonstreaming /v1/completions endpoint.
Reports whole-request latency and completion tokens per elapsed second,
including prefill and HTTP overhead. It does not isolate decode throughput.

Run after `launch-minimax-v21-vllm-serve.sh`.
"""
from __future__ import annotations

import json
import sys
import time

import requests

VLLM = "http://localhost:8000/v1"
MODEL = "minimax-m2.7-prismaquant"


def time_one(prompt: str, max_tokens: int) -> dict:
    t0 = time.time()
    r = requests.post(
        f"{VLLM}/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=600,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    out = r.json()
    usage = out.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", max_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": elapsed,
        "request_completion_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
    }


PROMPTS = [
    ("short",     "What is the capital of France?",                                        128),
    ("medium",    "Write a 200-word summary of the French Revolution, focusing on its causes and key figures.", 256),
    ("long-pref", " ".join(["The history of mathematics extends as far back as ancient times."] * 50) + " Continue the essay.", 256),
]


def main() -> int:
    print(f"[bench] hitting {VLLM} model={MODEL}")
    try:
        r = requests.get(f"{VLLM}/models", timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[bench] FAILED to reach vLLM: {e!r}")
        return 1

    # Warm up
    print("[bench] warmup ...")
    time_one("Hello.", 32)

    print("\n[bench] runs:")
    for label, prompt, max_tok in PROMPTS:
        m = time_one(prompt, max_tok)
        print(
            f"  {label:12s} "
            f"prompt={m['prompt_tokens']:4d} "
            f"completion={m['completion_tokens']:4d} "
            f"wall={m['wall_s']:6.2f}s "
            f"request completion={m['request_completion_tok_s']:5.2f} tok/s"
        )

    print("\n[bench] reference: old custom low-bit baseline = 3.78 tok/s "
          "(session_2026_04_24_3stream_win memory)")
    print("[bench] compare that reference only if its timing includes the same request work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
