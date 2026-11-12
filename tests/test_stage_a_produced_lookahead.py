"""Read lookahead restores existing publications through the IO owner."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from prismaquant import io_engine
from produced_native_fixtures import owner
from test_stage_a_produced_boundary_chain import GROUP_SIZE, write_group


@pytest.mark.parametrize('then', [False, True])
def test_each_window_and_next_pass_read_exact_bytes_after_local_eviction(tmp_path, monkeypatch, then):
    from prismaquant.streamed_boundary_artifacts import prefetched_boundary_batches
    count = GROUP_SIZE * 6
    storage, pub = owner(tmp_path, n_batches=count, group_size=GROUP_SIZE,
        tensor_bytes=32, spool_bytes=2 * count * (32 + 65536))
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    with storage:
        refs = write_group(storage, count=count)
        batches = [SimpleNamespace(activations_cpu={0: ref}) for ref in refs]
        spool = storage._local_output_spool
        spool.join_copies()
        for ref in refs:
            storage.release_produced_group(ref)
        assert all(row['released'] for row in spool._groups.values())
        before = dict(pub._groups)
        copied = []
        actual_copy = io_engine.publish_copy
        def copy(source, destination, **kwargs):
            result = actual_copy(source, destination, **kwargs)
            copied.append((source, destination))
            return result
        monkeypatch.setattr(io_engine, 'publish_copy', copy)
        for _ in range(2):
            seen = []
            with prefetched_boundary_batches(storage, batches, 0,
                    then=(0, None) if then else None) as windows:
                for index, _, tensor, _ in windows:
                    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32) + index)
                    seen.append(index)
            assert seen == list(range(count))
        assert copied and all(Path(dst).is_relative_to(spool.root) for _, dst in copied)
        assert pub._groups == before and len(before) == 6
        assert storage.telemetry['resident_tensor_bytes'] == 0
