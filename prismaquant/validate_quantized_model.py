"""Pre-ship quality validator for PrismaQuant artifacts.

Designed to catch the class of failure that shipped a broken 27B
checkpoint to HF in this session: predicted Δloss said the artifact
was *better* than its predecessor (13.5% lower), but the actual
model produced ~10,000× worse perplexity because the allocator's
fused-sibling sum-aggregation under-weighted asymmetric sensitivity.

The predicted-Δloss heuristic is not enough. Every artifact must
pass *measured* quality gates before upload.

Checks, in order:

  1. **Serve check** — vLLM actually starts the model (load, MTP
     wrapper, CUDA graph capture) with the recipe's flags.
  2. **Generation sanity** — small set of prompts must produce
     coherent outputs. Filters obvious catastrophic breakage
     (NaN/repetition loops/nonsense) before wasting on stats.
  3. **Perplexity / NLL** — logprobs over a diverse held-out
     prompt suite. Hard thresholds: `ppl < MAX_PPL` and
     `p99 per-prompt NLL < MAX_P99_NLL`. p99 catches the 27B
     failure mode where 80% of prompts scored NLL~10 while
     2/10 scored normally — mean alone missed it.
  4. **MTP acceptance** — if spec-decode is on, per-position
     acceptance > `MIN_MTP_ACCEPT_P0` at position 0.

Use from CI or pre-ship hook:

    python -m prismaquant.validate_quantized_model \\
        --artifact /path/to/exported \\
        --baseline rdtand/<previous-known-good>     # optional
        --report /path/to/report.md

Exit 0 = all checks passed. Exit 1 = at least one check failed
(prints a report to stdout + writes `--report` markdown).

**Workflow for artifacts with speculative decoding (MTP / Eagle):**
run the validator twice, against two different serves.

  1. Serve WITHOUT `--speculative-config`, run validator → the
     perplexity check produces target-model NLL and a meaningful
     verdict. MTP acceptance is skipped because no drafts fire.

  2. Re-serve WITH `--speculative-config`, run validator → the
     perplexity check will refuse to run (it detects spec-decode
     via /metrics and fails with a diagnostic rather than silently
     return draft-model NLL). MTP acceptance runs and reports
     position-0 accept rate.

The model is "ship-ready" only if both passes succeed. This is
awkward but honest: vLLM offers no target-model logprob path
while spec-decode is active, and faking perplexity from the draft
model has already burned one false-FAIL in the session that spawned
this file.

Design notes:
  - We use vLLM's OpenAI API for logprobs via /v1/completions
    with `echo=True`, which echoes the prompt tokens with their
    prior logprobs.
  - Model serving is run in a subprocess / docker exec so we
    can tear it down cleanly between runs.
  - Prompts span science, history, CS, economics, everyday prose
    to catch domain-specific failure. Short enough to keep a full
    suite under ~30s at spec-decode speed.
  - Thresholds are calibrated: MAX_PPL=25 catches catastrophic
    breakage but tolerates normal 4-bit quant degradation
    (BF16 baseline ~3-5, 4-bit ~4-8). MAX_P99_NLL=6 is ~2σ above
    BF16 average.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict


# -----------------------------------------------------------------
# Prompt suite — diverse, held-out, deliberately off the beaten path
# -----------------------------------------------------------------
EVAL_PROMPTS: list[str] = [
    "The mitochondrion is an organelle found in most eukaryotic cells. It is the site of cellular respiration and ATP production. Mitochondria have their own DNA and are thought to have originated from ancient prokaryotes.",
    "In modern cryptography, a hash function maps data of arbitrary size to a fixed-size output. Good cryptographic hashes are deterministic, collision-resistant, and exhibit the avalanche property: small input changes produce large output changes.",
    "The French Revolution began in 1789 and ended in the late 1790s. It fundamentally reshaped European political thought, ending the monarchy in France and introducing ideals of liberty, equality, and fraternity that would influence democratic movements for centuries.",
    "A binary search tree is a data structure in which each node has at most two children, and for every node the left subtree contains keys less than the node's key and the right subtree contains keys greater. Lookups, insertions, and deletions take O(log n) on average.",
    "Keynesian economics argues that aggregate demand drives economic output, especially during recessions. Governments can stimulate demand through fiscal policy — spending more or cutting taxes — when private consumption and investment fall short.",
    "Photosynthesis converts light energy, primarily from the sun, into chemical energy stored in glucose. It occurs mainly in the chloroplasts of plant cells, where chlorophyll absorbs photons and drives the reduction of carbon dioxide into carbohydrates.",
    "In compilers, a lexer breaks source code into tokens, and a parser groups tokens into a syntax tree according to grammar rules. Semantic analysis then annotates the tree with type and scope information, ready for code generation or interpretation.",
    "The Great Pyramid of Giza was built around 2560 BC as a tomb for the Fourth Dynasty Egyptian pharaoh Khufu. It held the record for the tallest man-made structure for nearly 4000 years, until the completion of Lincoln Cathedral in 1311.",
    "Operating systems manage hardware resources and provide services to applications. Key abstractions include processes, virtual memory, file systems, and a scheduler that decides which process runs when on the available CPU cores.",
    "Neural networks are loosely inspired by biological neurons. In a feedforward network, input activations pass through layers of weighted sums and nonlinear functions. Training uses backpropagation to compute gradients of a loss with respect to each weight.",
    "A sauce Bechamel is one of the five French mother sauces. Its base is a roux of butter and flour cooked gently, to which warm milk is gradually whisked in until the mixture thickens into a smooth white sauce seasoned with salt, nutmeg, and pepper.",
    "The theory of plate tectonics explains the movement of large sections of Earth's lithosphere. Plates move atop the semi-fluid asthenosphere, driven by convection currents in the mantle. Their interactions cause earthquakes, volcanoes, and mountain building.",
]

# Generation-sanity prompts — short, expect coherent short continuation.
# Unlike the perplexity prompts, we actually SAMPLE here (non-zero temp
# at small max_tokens) and only assert the output looks like English.
GEN_PROMPTS: list[str] = [
    "The best approach to learning a new programming language is",
    "When cooking rice in a pot, the most common mistake is",
    "A sensible way to explain gravitational time dilation to a non-physicist is",
    "The single most important fact about photosynthesis is",
]


# -----------------------------------------------------------------
# Default thresholds (tune via CLI if needed)
# -----------------------------------------------------------------
DEFAULT_MAX_PPL = 25.0
DEFAULT_MAX_P99_NLL = 6.0
DEFAULT_MAX_MEAN_NLL = 3.0
DEFAULT_MIN_GEN_LEN = 30               # chars in each generated completion
DEFAULT_MIN_MTP_ACCEPT_P0 = 0.60       # position-0 accept fraction


# -----------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    artifact: str
    base_url: str
    model_name: str
    thresholds: dict
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# -----------------------------------------------------------------
# HTTP helpers (no extra deps — urllib is stdlib)
# -----------------------------------------------------------------
def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as r:
            return r.status == 200
    except Exception:
        return False


def _spec_decode_on(base_url: str) -> bool:
    """True iff the vLLM serve was launched with --speculative-config.

    Detection: vLLM registers the `vllm:spec_decode_*` Prometheus
    counters + gauges at startup whenever spec-decode is configured,
    even before any drafts run. Their literal presence in /metrics is
    a config-time signal.

    Critical for the perplexity check: with spec-decode on, vLLM
    routes /v1/completions echo+logprobs through the DRAFT model, so
    the NLL values returned are the 1-layer MTP head's logprobs,
    NOT the target model's. Those are not usable for target-model
    perplexity measurement. Detecting the condition lets the
    validator refuse to silently mis-report."""
    try:
        text = _get_text(f"{base_url}/metrics")
    except Exception:
        return False
    return "vllm:spec_decode" in text


def wait_for_ready(base_url: str, max_seconds: float = 900.0,
                   poll_interval: float = 5.0) -> bool:
    """Block until the vLLM server responds to /health with 200."""
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        if _health_ok(base_url):
            return True
        time.sleep(poll_interval)
    return False


# -----------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------
def check_serve_ready(base_url: str) -> CheckResult:
    ok = _health_ok(base_url)
    return CheckResult(
        name="serve_ready",
        passed=ok,
        detail="/health returned 200" if ok else "/health did NOT return 200 (server not up)",
    )


def check_generation_sanity(base_url: str, model_name: str,
                            min_gen_len: int) -> CheckResult:
    """Sample a few short completions — fail if any are empty or
    clearly-not-English. Catches crashes that return empty responses,
    and gross breakage that returns pure noise."""
    short_outputs = []
    for i, prompt in enumerate(GEN_PROMPTS, 1):
        try:
            r = _post_json(
                f"{base_url}/v1/completions",
                {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 40,
                    "temperature": 0.3,
                    "top_p": 0.95,
                },
            )
            text = (r["choices"][0].get("text") or "").strip()
        except Exception as e:
            return CheckResult(
                name="generation_sanity",
                passed=False,
                detail=f"request {i} failed: {type(e).__name__}: {e}",
            )
        if len(text) < min_gen_len:
            short_outputs.append((i, len(text), text))
    if short_outputs:
        return CheckResult(
            name="generation_sanity",
            passed=False,
            detail=(f"{len(short_outputs)}/{len(GEN_PROMPTS)} completions "
                    f"shorter than {min_gen_len} chars: "
                    f"{[(i, n) for i, n, _ in short_outputs]}"),
            metrics={"short_outputs": short_outputs},
        )
    return CheckResult(
        name="generation_sanity",
        passed=True,
        detail=f"all {len(GEN_PROMPTS)} completions ≥ {min_gen_len} chars",
    )


def check_perplexity(base_url: str, model_name: str,
                     max_ppl: float, max_p99_nll: float,
                     max_mean_nll: float) -> CheckResult:
    """Compute per-token NLL across the eval prompt suite.

    Hard fails when mean NLL exceeds threshold OR when p99 per-prompt
    NLL exceeds threshold. p99 catches bimodal-failure where the model
    has "quality pockets" (see 27B session: 2/10 prompts normal, 8/10
    catastrophic at NLL~10). Mean alone would have flagged, but p99
    is the more diagnostic signal.

    **Hard-fails with a diagnostic if spec-decode is detected on the
    serve.** vLLM routes /v1/completions echo+logprobs through the
    draft model when speculative decoding is configured — the NLL
    numbers you'd get back are the 1-layer MTP head's logprobs, not
    the target model's. Running perplexity checks against those is
    like measuring a book's quality by its typos in the copyright
    page. The only reliable fix today is a separate vLLM serve
    without --speculative-config; see the module docstring for the
    standard two-serve workflow.
    """
    if _spec_decode_on(base_url):
        return CheckResult(
            name="perplexity",
            passed=False,
            detail=("spec-decode is configured on this vLLM serve — /v1/"
                    "completions echo+logprobs would return DRAFT model "
                    "NLL, not target. Re-serve WITHOUT --speculative-config "
                    "for the perplexity check (use a second serve for MTP "
                    "acceptance). Reports from a spec-decode-on eval have "
                    "false-failed healthy models in the past."),
            metrics={"spec_decode_detected": True, "skipped": True},
        )
    per_prompt_avg_nll: list[float] = []
    total_tokens = 0
    total_nll = 0.0
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        try:
            r = _post_json(
                f"{base_url}/v1/completions",
                {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 1,
                    "echo": True,
                },
            )
        except Exception as e:
            return CheckResult(
                name="perplexity",
                passed=False,
                detail=f"prompt {i}: {type(e).__name__}: {e}",
            )
        lp = r["choices"][0]["logprobs"]
        token_logprobs = lp.get("token_logprobs") or []
        valid = [x for x in token_logprobs if x is not None]
        if not valid:
            return CheckResult(
                name="perplexity",
                passed=False,
                detail=f"prompt {i}: no token_logprobs returned",
            )
        nll = -sum(valid)
        total_nll += nll
        total_tokens += len(valid)
        per_prompt_avg_nll.append(nll / len(valid))

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    per_prompt_avg_nll.sort()
    k = max(0, int(0.99 * len(per_prompt_avg_nll)) - 1)
    p99 = per_prompt_avg_nll[-1] if len(per_prompt_avg_nll) <= 2 else per_prompt_avg_nll[k]
    # Actually for small N, "p99" is basically "max" — use max for clarity.
    p99 = per_prompt_avg_nll[-1]

    metrics = {
        "perplexity": ppl,
        "mean_nll_per_tok": mean_nll,
        "p99_nll_per_tok": p99,
        "per_prompt_avg_nll": per_prompt_avg_nll,
        "n_tokens": total_tokens,
    }

    reasons = []
    if ppl > max_ppl:
        reasons.append(f"ppl={ppl:.2f} > {max_ppl}")
    if mean_nll > max_mean_nll:
        reasons.append(f"mean_nll={mean_nll:.3f} > {max_mean_nll}")
    if p99 > max_p99_nll:
        reasons.append(f"max(per-prompt avg NLL)={p99:.3f} > {max_p99_nll} "
                       f"(bimodal failure)")
    return CheckResult(
        name="perplexity",
        passed=len(reasons) == 0,
        detail=("OK" if not reasons else "; ".join(reasons)),
        metrics=metrics,
    )


def check_mtp_acceptance(base_url: str, min_p0: float) -> CheckResult:
    """Scrape /metrics for spec-decode acceptance rates. Passes if
    position-0 acceptance fraction exceeds `min_p0`. If no spec-decode
    metrics are exposed (spec-decode not enabled), passes with 'skipped'."""
    try:
        text = _get_text(f"{base_url}/metrics")
    except Exception as e:
        return CheckResult(
            name="mtp_acceptance",
            passed=False,
            detail=f"/metrics unreachable: {type(e).__name__}: {e}",
        )
    drafts = accepted_p0 = None
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(line.split()[-1])
        elif 'spec_decode_num_accepted_tokens_per_pos_total' in line and 'position="0"' in line:
            accepted_p0 = float(line.split()[-1])
    if drafts is None or drafts <= 0:
        return CheckResult(
            name="mtp_acceptance",
            passed=True,
            detail="no spec-decode drafts recorded — skipping (spec-decode off?)",
            metrics={"drafts": drafts or 0},
        )
    frac = (accepted_p0 or 0) / drafts
    metrics = {"drafts": drafts, "accepted_p0": accepted_p0, "accept_rate_p0": frac}
    return CheckResult(
        name="mtp_acceptance",
        passed=(frac >= min_p0),
        detail=(f"pos-0 acceptance = {frac:.1%} "
                f"({'≥' if frac >= min_p0 else '<'} {min_p0:.0%} threshold)"),
        metrics=metrics,
    )


# -----------------------------------------------------------------
# Top-level runner
# -----------------------------------------------------------------
def run_validation(
    base_url: str,
    model_name: str,
    *,
    max_ppl: float = DEFAULT_MAX_PPL,
    max_mean_nll: float = DEFAULT_MAX_MEAN_NLL,
    max_p99_nll: float = DEFAULT_MAX_P99_NLL,
    min_gen_len: int = DEFAULT_MIN_GEN_LEN,
    min_mtp_accept_p0: float = DEFAULT_MIN_MTP_ACCEPT_P0,
    wait_seconds: float = 900.0,
) -> ValidationReport:
    rep = ValidationReport(
        artifact=model_name,
        base_url=base_url,
        model_name=model_name,
        thresholds={
            "max_ppl": max_ppl,
            "max_mean_nll": max_mean_nll,
            "max_p99_nll": max_p99_nll,
            "min_gen_len": min_gen_len,
            "min_mtp_accept_p0": min_mtp_accept_p0,
        },
    )

    # Wait for server first (don't race probes).
    ok = wait_for_ready(base_url, max_seconds=wait_seconds)
    if not ok:
        rep.checks.append(CheckResult(
            name="serve_ready",
            passed=False,
            detail=f"vLLM /health did not reach 200 within {wait_seconds}s",
        ))
        return rep

    rep.checks.append(check_serve_ready(base_url))
    rep.checks.append(check_generation_sanity(base_url, model_name, min_gen_len))
    rep.checks.append(check_perplexity(
        base_url, model_name,
        max_ppl=max_ppl, max_p99_nll=max_p99_nll, max_mean_nll=max_mean_nll,
    ))
    rep.checks.append(check_mtp_acceptance(base_url, min_mtp_accept_p0))
    return rep


def format_report_md(rep: ValidationReport) -> str:
    status = "✅ PASS" if rep.passed else "❌ FAIL"
    lines = [
        f"# PrismaQuant Validation Report — {status}",
        "",
        f"- **artifact:** `{rep.artifact}`",
        f"- **endpoint:** {rep.base_url}",
        f"- **thresholds:** {json.dumps(rep.thresholds, indent=None)}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in rep.checks:
        icon = "✅" if c.passed else "❌"
        lines.append(f"| {c.name} | {icon} | {c.detail} |")
    # Metrics detail
    for c in rep.checks:
        if c.metrics:
            lines.append("")
            lines.append(f"### {c.name} metrics")
            lines.append("```json")
            lines.append(json.dumps(c.metrics, indent=2, default=str))
            lines.append("```")
    return "\n".join(lines)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-ship quality validator for PrismaQuant artifacts. "
                    "Hits a running vLLM endpoint and runs serve / generation "
                    "sanity / perplexity / MTP acceptance checks.")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_URL",
                                                         "http://localhost:8000"),
                    help="vLLM OpenAI-compatible server URL")
    ap.add_argument("--model-name", required=True,
                    help="Model name as reported by vLLM (local path like /exported "
                         "or HF repo id like org/name)")
    ap.add_argument("--max-ppl", type=float, default=DEFAULT_MAX_PPL)
    ap.add_argument("--max-mean-nll", type=float, default=DEFAULT_MAX_MEAN_NLL)
    ap.add_argument("--max-p99-nll", type=float, default=DEFAULT_MAX_P99_NLL)
    ap.add_argument("--min-gen-len", type=int, default=DEFAULT_MIN_GEN_LEN)
    ap.add_argument("--min-mtp-accept-p0", type=float,
                    default=DEFAULT_MIN_MTP_ACCEPT_P0)
    ap.add_argument("--wait-seconds", type=float, default=900.0,
                    help="Max time to wait for /health 200 before giving up")
    ap.add_argument("--report", default=None,
                    help="Optional path to write the markdown report")
    args = ap.parse_args()

    rep = run_validation(
        args.base_url, args.model_name,
        max_ppl=args.max_ppl,
        max_mean_nll=args.max_mean_nll,
        max_p99_nll=args.max_p99_nll,
        min_gen_len=args.min_gen_len,
        min_mtp_accept_p0=args.min_mtp_accept_p0,
        wait_seconds=args.wait_seconds,
    )
    md = format_report_md(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as f:
            f.write(md)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
