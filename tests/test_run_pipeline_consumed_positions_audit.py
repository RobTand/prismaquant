"""Exercise the real CPU draws at the pipeline held-out boundary."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'prismaquant/run-pipeline.sh'


class Tokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        import torch
        return SimpleNamespace(input_ids=torch.tensor([[int(token) for token in text.split()]]))


def corpus_file(tmp_path, lengths):
    path = tmp_path / 'corpus.jsonl'
    path.write_text(''.join(json.dumps({'text': ' '.join(str(row * 10000 + index + 1)
        for index in range(length))}) + '\n' for row, length in enumerate(lengths)))
    return path


def shell_skip(tmp_path, corpus, large_cost):
    source = SCRIPT.read_text()
    start = source.index(': "${VALIDATED_FRONTIER_NSAMPLES:')
    block = source[start:source.index('# Imatrix (column-weight)', start)]
    wrapper = r'''
set -euo pipefail
python3() {
  if [[ "${1:-}" == "-m" && "${2:-}" == "prismaquant.pipeline" ]]; then
    "$AUDIT_PYTHON" -c '
import runpy, sys, torch, transformers
from types import SimpleNamespace
class Tokenizer:
    eos_token_id = 0
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[int(token) for token in text.split()]]))
transformers.AutoTokenizer.from_pretrained = lambda *args, **kwargs: Tokenizer()
sys.argv = ["prismaquant.pipeline", *sys.argv[3:]]
runpy.run_module("prismaquant.pipeline", run_name="__main__")
' "$@"
    return $?
  fi
  "$AUDIT_PYTHON" "$@"
}
'''
    env = dict(os.environ, AUDIT_PYTHON=sys.executable, MODEL_PATH='unused',
               NSAMPLES='32', SEQLEN='1024', DATASET=str(corpus),
               VALIDATED_FRONTIER_DATASET=str(corpus), COST_MODE='aura',
               SELECTION_MODE='validated-surrogate', VALIDATED_FRONTIER_MATERIALIZATION='inplace',
               TARGET_DISK_GB='', AURA_COST_NSAMPLES='64' if large_cost else '8',
               AURA_COST_SEQLEN='1024' if large_cost else '128',
               AURA_EXPERT_SEQLEN='1024' if large_cost else '512')
    result = subprocess.run(['bash', '-c', wrapper + block +
        '\nprintf "RESULT_SKIP=%s\\n" "$VALIDATED_FRONTIER_CALIB_SKIP_FIRST"'],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return int(re.search(r'RESULT_SKIP=(\d+)', result.stdout).group(1))


@pytest.mark.parametrize('large_cost', [False, True])
def test_default_and_larger_cost_draws_are_held_out(tmp_path, large_cost):
    import torch
    from prismaquant.calibration_data import CalibrationRequest, load_calibration, load_validation_calibration
    from prismaquant.perturbed_x_cache import calibration_data_hash

    corpus = corpus_file(tmp_path, [2048] * 120)
    skip = shell_skip(tmp_path, corpus, large_cost)
    tokenizer = Tokenizer()
    draws = [(32, 1024), (64, 1024) if large_cost else (8, 128),
             (16, 1024) if large_cost else (16, 512)]
    fit = [load_calibration(tokenizer, str(corpus), n, length) for n, length in draws]
    request = lambda count: CalibrationRequest(str(corpus), 32, 1024, 1, 42, count, 'train', 1)
    selection = load_validation_calibration(tokenizer, request(skip)).selection_repeats[0]
    baseline = load_validation_calibration(tokenizer, request(32)).selection_repeats[0]
    selected_documents = set((selection // 10000).flatten().tolist())
    fit_documents = set().union(*(set((ids // 10000).flatten().tolist()) for ids in fit))
    assert not selected_documents & fit_documents
    assert skip == (64 if large_cost else 32)
    assert calibration_data_hash(baseline) == "fcf298d73e62a86a0da585ba0f99ca3a"
    assert torch.equal(selection, baseline) is (not large_cost)
    print('WINDOWS', json.dumps({'large_cost': large_cost, 'skip': skip,
          'selection_hash': calibration_data_hash(selection),
          'base_selection_hash': calibration_data_hash(baseline),
          'shared_documents': len(selected_documents & fit_documents)}))


@pytest.mark.parametrize("seed", [42, 7])
def test_source_documents_preserve_tokens_and_original_indices(tmp_path, seed):
    import torch
    from prismaquant.calibration_data import load_calibration

    corpus = corpus_file(tmp_path, [32] * 20)
    documents = []
    tracked = load_calibration(Tokenizer(), str(corpus), 8, 16,
                               calib_seed=seed, source_documents=documents)
    original = load_calibration(Tokenizer(), str(corpus), 8, 16, calib_seed=seed)
    assert torch.equal(tracked, original)
    assert documents == [frozenset((row // 10000).tolist()) for row in tracked]
    if seed != 42:
        assert documents != [frozenset({index}) for index in range(8)]


def test_packed_windows_report_each_document_and_preserve_tokens(tmp_path):
    import torch
    from prismaquant.calibration_data import load_calibration

    corpus = corpus_file(tmp_path, [20, 3, 5, 7, 2, 4, 6, 8, 1, 9])
    documents = []
    tracked = load_calibration(Tokenizer(), str(corpus), 4, 16, source_documents=documents)
    original = load_calibration(Tokenizer(), str(corpus), 4, 16)
    assert torch.equal(tracked, original)
    assert documents == [frozenset({0}), frozenset({1, 2, 3}),
                         frozenset({3, 4, 5, 6}), frozenset({6, 7, 8, 9})]


def test_mixed_lengths_use_consumed_documents_instead_of_max_count(tmp_path):
    from prismaquant.calibration_data import load_calibration
    from prismaquant.pipeline import held_out_calibration_skip

    corpus = corpus_file(tmp_path, [3, 128, 2048] * 140)
    draws = [(32, 1024, 42), (120, 128, 42), (16, 512, 42)]
    skip = held_out_calibration_skip(source=str(corpus), validation_source=str(corpus),
        validation_seqlen=1024, draws=draws, requested_skip=None, enforce=True,
        tokenizer=Tokenizer(), validation_samples=32)
    assert skip == 60
    consumed = []
    for count, length, seed in draws:
        load_calibration(Tokenizer(), str(corpus), count, length,
                         calib_seed=seed, source_documents=consumed)
    selected = []
    load_calibration(Tokenizer(), str(corpus), skip + 32, 1024, source_documents=selected)
    assert min(document for row in selected[skip:] for document in row) > max(
        document for row in consumed for document in row)


@pytest.mark.parametrize("source", ["nonlocal_dataset", "corpus.csv"])
def test_unmeasured_loader_sources_refuse_before_loading(tmp_path, source, monkeypatch):
    from prismaquant.calibration_data import load_calibration

    if source.endswith(".csv"):
        path = tmp_path / source
        path.write_text("text\nunsupported\n")
        source = str(path)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda *a, **k: pytest.fail("unproven source loaded")))
    with pytest.raises(ValueError, match="source positions|held-out positions"):
        load_calibration(Tokenizer(), source, 1, 16, source_documents=[])


@pytest.mark.parametrize("kind", ["jsonl", "txt"])
def test_skipped_records_keep_original_source_indices(tmp_path, kind):
    from prismaquant.calibration_data import load_calibration

    corpus = tmp_path / ("source." + kind)
    text = " ".join(str(index + 1) for index in range(32))
    corpus.write_text(("{}\n\n" + json.dumps({"text": text}) + "\n")
                      if kind == "jsonl" else "\n\n" + text + "\n")
    documents = []
    load_calibration(Tokenizer(), str(corpus), 1, 16, source_documents=documents)
    assert documents == [frozenset({2})]
