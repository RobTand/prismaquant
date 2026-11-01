"""Real expert CLI, source delivery, KL and journals on a small CPU model.

CUDA admission, tokenizer and calibration sampling are explicit doubles.
The legacy journal fixture removes only the new input before journal intake.
"""
from copy import deepcopy
import math
import pickle

import pytest
import torch

from prismaquant import cost_stage_checkpoint as journal
from prismaquant import expert_empirical_cost as expert
from prismaquant import memory_management
from prismaquant.prepriced_cost import validate_native_cost_source_for_model
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.native_source_contract_fixtures import (
    producer_cli_inputs, producer_cpu_guard, producer_source,
)


def _arguments(tmp_path, source, driver):
    args = ['--model', str(source), '--output', str(tmp_path / 'cost.pkl'),
            '--device', 'cpu', '--dataset', 'synthetic-token-draw',
            '--formats', 'FP8_E4M3,BF16', '--max-units', '1']
    if driver != 'resident':
        args += ['--streaming', '--checkpoint-dir', str(tmp_path / 'journal')]
    return args


def _setup(tmp_path, monkeypatch, driver, native=True):
    source, _expected = producer_source(tmp_path, native=native)
    producer_cli_inputs(monkeypatch)
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(memory_management, 'CaptureMemoryGuard', lambda *_args: guard)
    monkeypatch.setenv('PRISMAQUANT_EXPERT_EVAL_DRIVER', driver)
    return source, _arguments(tmp_path, source, driver)


@pytest.mark.parametrize('driver', ['resident', 'window', 'forked'])
@pytest.mark.parametrize('native', [True, False], ids=['native', 'bf16'])
def test_actual_expert_main_binds_only_fresh_rows(tmp_path, monkeypatch, driver, native):
    if not native and driver == 'resident':
        pytest.importorskip('accelerate', reason='the unchanged BF16 expert HF device_map needs accelerate')
    source, args = _setup(tmp_path, monkeypatch, driver, native)
    inherited = {'predicted_dloss': 0.375, 'cost_source': 'legacy-control'}
    base = {'costs': {'inherited': {'FP8_E4M3': inherited}}, 'stats': {}}
    base_path = tmp_path / 'base.pkl'
    base_path.write_bytes(pickle.dumps(base))
    before = base_path.read_bytes()
    args += ['--backfill-base', str(base_path)]
    assert expert.main(args) == 0
    payload = pickle.loads((tmp_path / 'cost.pkl').read_bytes())
    assert payload['costs']['inherited']['FP8_E4M3'] == inherited
    assert base_path.read_bytes() == before
    fresh = {name: rows for name, rows in payload['costs'].items() if name != 'inherited'}
    assert fresh
    for rows in fresh.values():
        assert math.isfinite(rows['FP8_E4M3']['predicted_dloss'])
        for row in rows.values():
            if native:
                assert row.get('source_binding') == build_source_model_binding(source, dtype=torch.bfloat16)
            else:
                assert 'source_binding' not in row
    if native:
        with pytest.raises(ValueError, match='native.*source binding'):
            validate_native_cost_source_for_model(payload, model=str(source), where='mixed fresh/inherited')
        payload['costs'] = fresh
        validate_native_cost_source_for_model(payload, model=str(source), where='fresh expert cells')


@pytest.mark.parametrize('driver', ['window', 'forked'])
@pytest.mark.parametrize('mode', ['0', '1'], ids=['certified', 'dev'])
def test_actual_expert_main_refuses_unbound_journal_before_measurement(
        tmp_path, monkeypatch, driver, mode):
    source, args = _setup(tmp_path, monkeypatch, driver)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    actual = journal.prepare_journal

    class StopAfterJournal(Exception):
        pass

    def legacy(*args, **kwargs):
        kwargs['identity'] = deepcopy(kwargs['identity'])
        kwargs['identity']['extra'].pop('source_binding', None)
        actual(*args, **kwargs)
        raise StopAfterJournal

    monkeypatch.setattr(journal, 'prepare_journal', legacy)
    with pytest.raises(StopAfterJournal):
        expert.main(args)

    def resume(*args, **kwargs):
        actual(*args, **kwargs)
        raise StopAfterJournal

    monkeypatch.setattr(journal, 'prepare_journal', resume)
    with pytest.raises(RuntimeError, match='identity'):
        expert.main(args + ['--resume'])
