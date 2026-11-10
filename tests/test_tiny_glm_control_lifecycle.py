"""Live experiment orchestration with real private CPU publication, not GLM qualification."""
import contextlib
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from experiments.alloc_lead_asd import tiny_glm_control as control
from prismaquant.stage_a_produced_output import (
    BoundaryProducedPublication, build_boundary_template, publish_document,
)
from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA
from produced_native_fixtures import selected_publications


@pytest.mark.parametrize('outcome', ['success', 'capture-error', 'late-error'])
def test_main_uses_capture_owner_finalization_and_preserves_failure(tmp_path, monkeypatch, outcome):
    output = tmp_path / 'result'
    record = tmp_path / 'record'; record.mkdir()
    declaration = build_boundary_template(output_prefix=output,
        artifact_max_bytes=1 << 24, group_size=2, max_entry_tensor_bytes=1024,
        concurrent_groups=8)
    binding = publish_document(tmp_path / 'declaration.json', declaration)
    producer = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': '123',
        'record_dir': str(record), 'launch_sha256': 'a' * 64,
        'argv': ['fixture', '--produced-declaration', binding['path'],
                 '--produced-declaration-sha256', binding['sha256']]}
    publication = BoundaryProducedPublication(declaration=declaration, binding=binding, producer=producer)
    selected_publications(monkeypatch)
    monkeypatch.setattr(BoundaryProducedPublication, 'bind_from_admitted_owner', classmethod(lambda cls: publication))
    ids = torch.arange(128).reshape(4, 32)
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    payloads = {'model.safetensors': save({'w': torch.tensor([2.0])}),
        'tokens.safetensors': save({'calibration_ids': ids}), 'config.json': b'{}',
        'source.json': json.dumps({'model_identity': {'content_sha256': 'b' * 64},
            'tokens_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}).encode()}
    entries = []
    for name, raw in payloads.items():
        path = inputs / name; path.write_bytes(raw)
        entries.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps({'entries': entries}))
    monkeypatch.setattr(control, 'activate_staged_tier_policy', lambda _: None)
    monkeypatch.setattr(control, 'bind_residency_manifest', lambda _: None)
    monkeypatch.setattr(control, 'read_staged_whole_file', lambda path, *a, **k: path.read_bytes())
    monkeypatch.setattr(control, 'residency_report', lambda: {'scope': 'private unstaged CPU fixture'})
    monkeypatch.setattr(control, 'dependency_identity', lambda: {'scope': 'unqualified fixture'})
    monkeypatch.setattr(control, 'commit', lambda *a: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(control, 'model_and_runner', lambda *a: (None, None, {}))
    monkeypatch.setattr(control, 'profiled', lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(control, 'direct', lambda *a: {'rows': [],
        'gradients': {(layer, probe): torch.ones(4) for layer in range(2) for probe in range(2)}})
    monkeypatch.setattr(control, 'components', lambda *a: {})
    expected = [torch.tensor([3.0, 7.0]), torch.tensor([11.0, 19.0])]
    references = []
    original = RuntimeError('original ' + outcome)

    def capture(*args):
        owner = args[-1]
        storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
            'directory': str(output / 'boundaries'), 'max_resident_bytes': 1 << 20,
            'max_auxiliary_bytes': 1 << 20, 'max_artifact_bytes': 1 << 24, 'prefetch_batches': 2})
        storage.bind({'fixture': 'live tiny-control lifecycle'}, n_probes=1, published=True)
        storage.bind_produced_output(owner, group_size=2, n_batches=2, max_entry_tensor_bytes=1024)
        with storage:
            for batch, tensor in enumerate(expected):
                references.append(storage.write(tensor, batch_index=batch, boundary_index=0))
            with storage.prefetch(references) as window:
                assert all(torch.equal(storage.get(window, ref), wanted)
                           for ref, wanted in zip(references, expected))
            if outcome == 'capture-error':
                raise original
        return {'rows': [], 'receipt': {'scope': 'private exact tensors, not GLM'},
            'rolled': {(layer, batch, probe): torch.ones(1)
                       for layer in range(2) for batch in range(4) for probe in range(2)}}

    monkeypatch.setattr(control, 'capture', capture)
    def statistics(*args):
        if outcome == 'late-error':
            raise original
        return {'scope': 'not measured'}
    monkeypatch.setattr(control, 'statistics', statistics)
    arguments = ['--manifest', str(manifest), '--manifest-sha256', hashlib.sha256(manifest.read_bytes()).hexdigest(),
        '--output', str(output), '--device', 'cpu', '--only-dtype', 'float32',
        '--produced-declaration', binding['path'], '--produced-declaration-sha256', binding['sha256']]
    if outcome == 'success':
        control.main(arguments)
        assert json.loads((output / 'result.json').read_text())['complete'] is True
    else:
        with pytest.raises(RuntimeError) as error:
            control.main(arguments)
        assert error.value is original
        assert not (output / 'result.json').exists()
    assert len(references) == 2
    index = record / 'produced' / 'index.json'
    assert index.exists() == (outcome != 'capture-error')
    assert len(publication._groups) == 1
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record, read_exact_entry_tensors
    actual = read_exact_entry_tensors([exact_entry_record(ref) for ref in references],
                                     expected_session=publication.session)
    assert all(torch.equal(actual[ref.name], wanted) for ref, wanted in zip(references, expected))


def test_one_dtype_is_required_before_any_input_or_publication(tmp_path):
    with pytest.raises(SystemExit) as error:
        control.main(['--manifest', '/absent', '--manifest-sha256', 'a' * 64,
                      '--output', str(tmp_path / 'absent')])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
