"""Historical committed activations retain exact lazy and prefetched custody."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prismaquant.io_engine import EntryError
from prismaquant.measure_quant_cost import ActivationIndex, _activation_chunk_stream


@pytest.fixture
def committed_activation(tmp_path):
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'retained_sample_activation_reader.json').read_text())
    tensors = {name: torch.tensor(values, dtype=(torch.float32 if name == 'inputs'
                                                 else torch.int64))
               for name, values in fixture['tensors'].items()}
    directory = tmp_path / 'activations'
    directory.mkdir()
    path = directory / fixture['filename']
    torch.save({**tensors, 'name': fixture['qname'],
                'sample_parallel_merge': fixture['stamp']}, path)
    return fixture, directory, path, tensors['inputs']


def test_verified_activation_index_refuses_post_validation_replace(
        tmp_path, committed_activation):
    fixture, directory, path, expected = committed_activation
    qname = fixture['qname']
    index = ActivationIndex(directory, {qname: {}},
                            verification_contract=fixture['verification_contract'])
    try:
        assert torch.equal(index.load(qname), expected)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        payload['inputs'][0, 0] += 1.0
        replacement = tmp_path / 'replacement' / path.name
        replacement.parent.mkdir()
        torch.save(payload, replacement)
        replacement.replace(path)
        with pytest.raises(ValueError, match='differs from commit'):
            index.load_blob(qname)
    finally:
        index.close()


def test_prefetched_verified_activation_keeps_directory_and_identity_fences(
        tmp_path, committed_activation):
    fixture, directory, path, expected = committed_activation
    qname = fixture['qname']
    index = ActivationIndex(directory, {qname: {}},
                            verification_contract=fixture['verification_contract'])
    try:
        moved = tmp_path / 'moved-verified-cache'
        directory.rename(moved)
        directory.mkdir()
        path.write_bytes(b'foreign pathname contents')
        committed_path = moved / path.name
        assert index.prefetch_bytes(qname) == committed_path.stat().st_size
        chunks = [[(qname, None)]]
        with _activation_chunk_stream(index, chunks) as stream:
            got = stream.take(0)[0].value[0][0]
            assert torch.equal(got, expected)
            del got
            stream.release()
        payload = torch.load(committed_path, map_location='cpu', weights_only=False)
        payload['inputs'][0, 0] += 1.0
        torch.save(payload, committed_path)
        with _activation_chunk_stream(index, chunks) as stream:
            with pytest.raises(EntryError, match='differs from commit'):
                stream.take(0)
    finally:
        index.close()
