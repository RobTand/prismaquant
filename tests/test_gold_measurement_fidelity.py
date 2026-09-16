"""A gold result must say which estimator produced it.

Three instruments in this repository produce a number called "KL" and they are
not the same estimator (full vocabulary at one position; top-K plus a tail
bucket at every position; top-20 on a different corpus). The differences exceed
the deltas the numbers are used to argue about, so the label is a structured
field derived from the run's own arguments rather than prose a reader supplies.
"""
from __future__ import annotations

import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools.gold_measurement_fidelity import (
    FIDELITY_SCHEMA,
    full_kl_fidelity,
    wikitext_ppl_fidelity,
)


def test_final_scoring_is_labelled_full_vocabulary_with_no_tail_bucket():
    """`--score-positions final` asks the engine for every column."""
    fidelity = full_kl_fidelity(
        score_positions="final", prompt_top_k=None, vocab_size=151936)
    assert fidelity["schema"] == FIDELITY_SCHEMA
    assert fidelity["vocabulary"] == "full"
    assert fidelity["positions"] == "window_final_context_only"
    assert fidelity["top_k"] is None
    assert fidelity["tail_bucket"] is False
    assert fidelity["vocab_size"] == 151936


def test_all_position_scoring_is_labelled_top_k_plus_tail_not_full():
    """The tool's name says "full_kl"; under `--score-positions all` "full"
    means all-POSITIONS. Recording it as full-vocabulary would be the exact
    confound the field exists to prevent."""
    fidelity = full_kl_fidelity(
        score_positions="all", prompt_top_k=1024, vocab_size=151936)
    assert fidelity["vocabulary"] == "top_k_plus_tail"
    assert fidelity["positions"] == "all_prompt_positions"
    assert fidelity["top_k"] == 1024
    assert fidelity["tail_bucket"] is True
    assert "NOT comparable with a full-vocabulary number" in (
        fidelity["comparable_with"])


def test_the_label_follows_the_tools_real_defaults_not_a_typed_copy(
    monkeypatch,
):
    """Read the defaults off the TOOL's own parser.

    An earlier version of this test built its own `ArgumentParser` with the
    same two defaults typed in, which would keep passing if the tool's default
    K became 2048 -- a test that certifies nothing. Driving the real CLI to the
    point where it has parsed, then reading the namespace, is what actually
    pins them.
    """
    import sys as _sys

    import tools.measure_vllm_full_kl as tool

    captured = {}

    class Reached(Exception):
        """Not a RuntimeError: `main()` catches those and converts them into
        `parser.error()`, i.e. SystemExit, which would swallow the sentinel."""

    def stop(args):
        captured.update(vars(args))
        raise Reached

    monkeypatch.setattr(tool, "_resolve_serve_image", stop)
    monkeypatch.setattr(_sys, "argv", [
        "measure_vllm_full_kl", "--mode", "teacher", "--model", "artifact",
        "--output", "result.json"])
    with pytest.raises(Reached):
        tool.main()

    assert captured["score_positions"] == "final"
    assert captured["prompt_top_k"] == 1024

    # The default invocation is therefore full-vocabulary at one position...
    default = full_kl_fidelity(
        score_positions=captured["score_positions"],
        prompt_top_k=captured["prompt_top_k"],
        vocab_size=None,
    )
    assert default["vocabulary"] == "full"
    # ...and asking for every position makes it top-K at that same K.
    all_positions = full_kl_fidelity(
        score_positions="all",
        prompt_top_k=captured["prompt_top_k"],
        vocab_size=None,
    )
    assert all_positions["vocabulary"] == "top_k_plus_tail"
    assert all_positions["top_k"] == captured["prompt_top_k"]


@pytest.mark.parametrize("score_positions", ["", "full", "ALL", "top_k", None])
def test_an_unknown_estimator_refuses_rather_than_being_described(
    score_positions,
):
    with pytest.raises(ValueError, match="score_positions"):
        full_kl_fidelity(score_positions=score_positions, prompt_top_k=8,
                         vocab_size=10)


@pytest.mark.parametrize("top_k", [0, -1, None, True, 4.0])
def test_all_position_scoring_without_a_valid_support_width_refuses(top_k):
    """The support width IS the fidelity: a labelled top-K result whose K is
    unknown says nothing."""
    with pytest.raises(ValueError):
        full_kl_fidelity(score_positions="all", prompt_top_k=top_k,
                         vocab_size=10)


def test_ppl_fidelity_records_the_window_geometry_that_distinguishes_two_runs():
    fidelity = wikitext_ppl_fidelity(
        n_tokens_scored=8176, seqlen=512, n_chunks=16)
    assert fidelity["vocabulary"] == "target_token_logprob_only"
    assert (fidelity["n_tokens_scored"], fidelity["seqlen"],
            fidelity["n_chunks"]) == (8176, 512, 16)
    assert "16 x 512-token windows" in fidelity["comparable_with"]


@pytest.mark.parametrize("fields", [
    {"n_tokens_scored": 0}, {"seqlen": 0}, {"n_chunks": 0},
    {"n_tokens_scored": True}, {"seqlen": 512.0},
])
def test_invalid_ppl_geometry_refuses(fields):
    arguments = {"n_tokens_scored": 8176, "seqlen": 512, "n_chunks": 16}
    arguments.update(fields)
    with pytest.raises(ValueError):
        wikitext_ppl_fidelity(**arguments)
