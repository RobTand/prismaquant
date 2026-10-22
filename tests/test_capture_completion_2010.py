"""Monolith publication must follow the recording owner's byte-verifying close."""
from __future__ import annotations

import errno
import json
import os
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import tessera_calibration_cache as store
from prismaquant import tessera_campaign as campaign
from test_capture_layer_chain import _Chain


def _assert_descriptors_closed(source):
    assert source._closed
    assert source._files
    for state in source._files.values():
        with pytest.raises(OSError) as error:
            os.fstat(state['fd'])
        assert error.value.errno == errno.EBADF


@pytest.fixture
def monolith(tmp_path, monkeypatch):
    # Reuse only the small census/unit fixture, not a chain quantum or its prep.
    capture = _Chain(tmp_path, prepare=False)
    shard = capture.source / 'model.safetensors'
    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    save_file({'weight': weight}, str(shard))
    tokens, text = [torch.arange(8).reshape(1, 8)], 'completion boundary fixture'
    census = json.loads(capture.census.read_text())
    census.update(campaign.th.calibration_identity(text, tokens, fit_tokens=7))
    capture.census.write_text(json.dumps(census))
    source = store.record_capture_source(capture.census, model=capture.source)
    args = SimpleNamespace(
        model=str(capture.source), calibration_census=str(capture.census),
        cache_dir=str(tmp_path / 'telemetry'), capture_calibration_out=str(tmp_path / 'capture'),
        seed=7, nsamples=1, seqlen=8, max_act_rows=2, tp_degree=1)
    Path(args.cache_dir).mkdir()
    manifest = tmp_path / 'capture' / 'capture_manifest.json'
    events, snapshots = [], {}

    # Keep the real writer and source proof. The menu is unrelated to this
    # completion boundary (and may require a different installed Tessera).
    monkeypatch.setattr(campaign, 'expand_menus_for_targets',
                        lambda weights, targets, **kw: {name: [object()] for name in targets})

    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 3, bias=False)
    model.b = torch.nn.Linear(2, 3, bias=False)

    def forward(batch):
        with source.safe_open(safe_open, shard, framework='pt', device='cpu') as handle:
            assert torch.equal(handle.get_tensor('weight'), weight)

    def visit_layer_batches(tokens, visit):
        for layer in range(2):
            visit(layer, forward)

    runner = SimpleNamespace(
        model=model, device=torch.device('cpu'), num_layers=2,
        layer_index_for_qname=lambda name: {'a': 0, 'b': 1}[name],
        visit_layer_batches=visit_layer_batches,
        context=SimpleNamespace(
            source_authentication=source,
            begin_source_initialization_audit=lambda: None,
            source_initialization_contract=lambda: capture.contract))

    def collect(model, names, tokens, max_rows, device, *, forward_batch, **kw):
        assert kw['want_hessian']
        forward_batch(tokens[0])
        return ({name: capture.acts[name][:min(capture.counts[name], max_rows)].clone()
                 for name in names},
                {name: capture.hessians[name].clone() for name in names},
                {name: capture.counts[name] for name in names},
                {name: capture.maxima[name] for name in names})

    monkeypatch.setattr(campaign, '_collect_activations', collect)

    authenticate = source.authenticate_complete_source
    recorded_files = source.recorded_source_files
    verify_content = source._verify_held_content
    close = source.close
    finish = store.CaptureWriter.finish

    def authenticate_complete_source():
        snapshots['receipt'] = authenticate()
        events.append('authenticated')
        return snapshots['receipt']

    def recorded_source_files():
        snapshots['files'] = recorded_files()
        events.append('snapshotted')
        return snapshots['files']

    def verify(name, state):
        assert not manifest.exists()
        verify_content(name, state)
        events.append(('verified', name))

    def close_source():
        close()
        _assert_descriptors_closed(source)
        events.append('closed')

    def finish_capture(writer, **kwargs):
        events.append('finish')
        receipt = finish(writer, **kwargs)
        assert manifest.is_file()
        events.append('published')
        return receipt

    monkeypatch.setattr(source, 'authenticate_complete_source', authenticate_complete_source)
    monkeypatch.setattr(source, 'recorded_source_files', recorded_source_files)
    monkeypatch.setattr(source, '_verify_held_content', verify)
    monkeypatch.setattr(source, 'close', close_source)
    monkeypatch.setattr(store.CaptureWriter, 'finish', finish_capture)

    def run():
        return campaign._run_streamed_calibration(
            args, runner, None, mode='attested',
            population=SimpleNamespace(members=[], declared=False),
            dense_targets=['a', 'b'], expert_targets=[], scope_groups={},
            tokens=tokens, corpus_text=text, census=census,
            context_by_unit={}, attention_implementation='eager',
            capture_runtime=census['capture_runtime'])

    try:
        yield SimpleNamespace(run=run, source=source, shard=shard, manifest=manifest,
                              events=events, snapshots=snapshots, args=args)
    finally:
        # A red regression on the old completion path must not leak the owner;
        # close itself releases every descriptor even when content verification refuses.
        if not source._closed:
            monkeypatch.setattr(source, '_verify_held_content', verify_content)
            with suppress(RuntimeError):
                close()
        _assert_descriptors_closed(source)


def test_monolith_verifies_and_closes_source_before_publishing(monolith):
    assert monolith.run() == 0
    events = monolith.events
    verified = [event[1] for event in events if isinstance(event, tuple)]
    assert set(verified) == set(monolith.snapshots['files'])
    assert len(verified) == len(set(verified))
    assert events[:2] == ['authenticated', 'snapshotted']
    assert events[-3:] == ['closed', 'finish', 'published']
    _assert_descriptors_closed(monolith.source)

    manifest = json.loads(monolith.manifest.read_text())
    assert manifest['status'] == 'complete'
    assert manifest['identity']['source_files'] == monolith.snapshots['files']
    receipt_path = Path(monolith.args.cache_dir) / 'capture-source-authentication.json'
    assert json.loads(receipt_path.read_text()) == monolith.snapshots['receipt']
    assert set(manifest['entries']) == {'a', 'b'}


def test_monolith_same_signature_mutation_refuses_without_manifest(monolith, monkeypatch):
    # POSIX ctime cannot be restored with utime. Model a same-signature source
    # by holding just that component constant; retain the inode/size/mtime fences.
    signature = store.file_stat_signature
    monkeypatch.setattr(store, 'file_stat_signature', lambda value: (*signature(value)[:-1], 0))
    snapshot_files = monolith.source.recorded_source_files

    def mutate_after_snapshot():
        files = snapshot_files()
        before = monolith.shard.stat()
        raw = bytearray(monolith.shard.read_bytes())
        raw[-1] ^= 1
        with monolith.shard.open('r+b') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.utime(monolith.shard, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert store.file_stat_signature(monolith.shard.stat()) == store.file_stat_signature(before)
        monolith.source.require_unchanged()  # Stat-only checks deliberately still pass.
        monolith.events.append('mutated')
        return files

    monkeypatch.setattr(monolith.source, 'recorded_source_files', mutate_after_snapshot)
    with pytest.raises(RuntimeError, match='authenticated source changed during consumption: model.safetensors'):
        monolith.run()
    assert monolith.events[:3] == ['authenticated', 'snapshotted', 'mutated']
    assert 'finish' not in monolith.events
    assert not monolith.manifest.exists()
    # Per-unit journal entries may survive for replay, but never a complete capture.
    assert list((monolith.manifest.parent / 'inputs').glob('*.pt'))
    _assert_descriptors_closed(monolith.source)
