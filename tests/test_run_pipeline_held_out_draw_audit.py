"""Check the pipeline shell contract without a model load."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "prismaquant/run-pipeline.sh"


def section(start, end):
    source = SCRIPT.read_text()
    first = source.index(start)
    return source[first:source.index(end, first)]


def stage_request():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS

    return {source: "unchanged" for keys in STAGE_SETTINGS_KEYS.values()
            for _, source in keys}


def old_artifact(path, stage, settings, content):
    from prismaquant.pipeline import check_stage_settings, stage_settings_document

    rc, messages = check_stage_settings(path, stage, stage_settings_document(settings))
    assert rc == 0, messages
    path.write_text(json.dumps(content))


def run_shell(tmp_path, body, values):
    env = dict(os.environ, **{key: str(value) for key, value in values.items()})
    env["AUDIT_PYTHON"] = sys.executable
    env["AUDIT_CALLS"] = str(tmp_path / "calls.txt")
    wrapper = r'''
set -euo pipefail
python3() {
  if [[ "${1:-}" == "-m" && ( "${2:-}" == "prismaquant.validate_assignments_kl" || "${2:-}" == "prismaquant.aura_cost" ) ]]; then
    printf '%s\n' "$*" >> "$AUDIT_CALLS"
    return 0
  fi
  "$AUDIT_PYTHON" "$@"
}
'''
    return subprocess.run(["bash", "-c", wrapper + body], env=env,
                          text=True, capture_output=True)



@pytest.mark.parametrize("cost_samples", [32, 64])
def test_pipeline_held_out_skip_covers_aura_cost_draw(tmp_path, cost_samples):
    import torch
    from prismaquant.calibration_data import (
        CalibrationRequest, load_calibration, load_validation_calibration,
    )
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.validate_assignments_kl import _assert_calibration_disjoint

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([
                [int(token) for token in text.split()]
            ]))

    corpus = tmp_path / "calibration.jsonl"
    corpus.write_text("".join(json.dumps({
        "text": " ".join(str(1000 * row + i) for i in range(16))
    }) + "\n" for row in range(100)))
    defaults = section(': "${VALIDATED_FRONTIER_NSAMPLES:', "# Imatrix (column-weight)")
    result = run_shell(tmp_path, defaults +
                       '\nprintf "%s %s %s" "$VALIDATED_FRONTIER_NSAMPLES" '
                       '"$VALIDATED_FRONTIER_SEQLEN" "$VALIDATED_FRONTIER_CALIB_SKIP_FIRST"', {
        "NSAMPLES": 32, "SEQLEN": 16, "DATASET": corpus,
        "VALIDATED_FRONTIER_DATASET": corpus, "AURA_COST_NSAMPLES": cost_samples,
        "AURA_COST_SEQLEN": 16, "AURA_EXPERT_SEQLEN": 16, "COST_MODE": "aura",
        "SELECTION_MODE": "surrogate", "VALIDATED_FRONTIER_MATERIALIZATION": "inplace",
        "TARGET_DISK_GB": "", "MODEL_PATH": "unused",
    })
    assert result.returncode == 0, result.stderr
    nsamples, seqlen, skip = map(int, result.stdout.split())
    tokenizer = Tokenizer()
    cost_ids = load_calibration(tokenizer, str(corpus), cost_samples, seqlen)
    draw = load_validation_calibration(tokenizer, CalibrationRequest(
        source=str(corpus), n_samples=nsamples, seqlen=seqlen,
        repeats=1, seed=42, skip_first=skip, split="train", repeat_seed_stride=1,
    ))
    selected = draw.selection_repeats[0]
    cost_rows = {tuple(row.tolist()) for row in cost_ids}
    overlap = cost_rows & {tuple(row.tolist()) for row in selected}
    upstream = {"cost": {calibration_data_hash(cost_ids)}}
    selection = [calibration_data_hash(selected)]
    assert not overlap, f"Selection shares {len(overlap)} windows with cost generation."
    _assert_calibration_disjoint(selection, upstream)


def local_draw(tmp_path):
    import torch

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[int(token) for token in text.split()]]))

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": " ".join(str(row * 100 + i)
        for i in range(32))}) + "\n" for row in range(300)))
    return corpus, Tokenizer()


@pytest.mark.parametrize("draws, expected", [
    ([(32, 16, 42), (64, 16, 42), (16, 16, 42)], 64),
    ([(32, 16, 42), (8, 16, 42), (96, 16, 42)], 96),
    ([(32, 16, 42), (80, 16, 42)], 80),
])
def test_skip_covers_each_active_draw(tmp_path, draws, expected):
    from prismaquant.pipeline import held_out_calibration_skip

    corpus, tokenizer = local_draw(tmp_path)
    assert held_out_calibration_skip(
        source=str(corpus), validation_source=str(corpus), validation_seqlen=16,
        draws=draws, requested_skip=None, enforce=True, tokenizer=tokenizer,
    ) == expected


@pytest.mark.parametrize("change", ["length", "seed", "small_skip"])
def test_measured_positions_extend_the_prefix(tmp_path, change):
    from prismaquant.calibration_data import load_calibration
    from prismaquant.pipeline import held_out_calibration_skip

    corpus, tokenizer = local_draw(tmp_path)
    draws = [(32, 8 if change == "length" else 16, 1 if change == "seed" else 42)]
    skip = held_out_calibration_skip(
        source=str(corpus), validation_source=str(corpus), validation_seqlen=16,
        draws=draws, requested_skip=16 if change == "small_skip" else None,
        enforce=True, tokenizer=tokenizer, validation_samples=1,
    )
    consumed, selected = [], []
    count, length, seed = draws[0]
    load_calibration(tokenizer, str(corpus), count, length, calib_seed=seed,
                     source_documents=consumed)
    load_calibration(tokenizer, str(corpus), skip + 1, 16, source_documents=selected)
    assert min(selected[skip]) > max(document for row in consumed for document in row)


def test_different_corpora_have_no_shared_position_domain(tmp_path):
    from prismaquant.pipeline import held_out_calibration_skip

    corpus, tokenizer = local_draw(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text('{"text": "different"}\n')
    with pytest.raises(ValueError, match="source-position domain"):
        held_out_calibration_skip(
            source=str(corpus), validation_source=str(other), validation_seqlen=16,
            draws=[(32, 16, 42)], requested_skip=None, enforce=True, tokenizer=tokenizer,
        )


def test_corpus_alias_keeps_complete_skip(tmp_path):
    from prismaquant.pipeline import held_out_calibration_skip

    corpus, tokenizer = local_draw(tmp_path)
    alias = tmp_path / "alias.jsonl"
    alias.write_bytes(corpus.read_bytes())
    assert held_out_calibration_skip(
        source=str(corpus), validation_source=str(alias), validation_seqlen=16,
        draws=[(64, 16, 42)], requested_skip=None, enforce=True, tokenizer=tokenizer,
    ) == 64


def test_insufficient_safe_windows_refuse(tmp_path):
    from prismaquant.pipeline import held_out_calibration_skip

    corpus, tokenizer = local_draw(tmp_path)
    with pytest.raises(ValueError, match="cannot supply"):
        held_out_calibration_skip(
            source=str(corpus), validation_source=str(corpus), validation_seqlen=16,
            draws=[(300, 16, 42)], requested_skip=None, enforce=True, tokenizer=tokenizer,
        )
