"""A native split producer prewrites and publishes only its whole-group range."""
import pytest
from produced_native_fixtures import owner
from test_stage_a_produced_boundary_chain import GROUP_SIZE as GROUP, write_group, read_group


def _owner(tmp_path, monkeypatch, batch_range):
    storage, pub = owner(tmp_path, group_size=GROUP, tensor_bytes=32,
        n_batches=3 * GROUP, batch_range=batch_range)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: None)
    return storage, pub


def test_a_ranged_owner_never_prewrites_the_next_ranges_group(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, (GROUP, 2 * GROUP))
    with storage:
        refs = write_group(storage, first=GROUP)
        read_group(storage, refs, first=GROUP)
    assert [key[3] for key in storage._produced_groups] == [1]
    assert len(pub._prewrites) == len(pub._groups) == 1
    assert len(next(iter(storage._produced_groups.values()))['references']) == GROUP


def test_a_ranged_owner_refuses_an_entry_outside_its_range(tmp_path, monkeypatch):
    storage, pub = _owner(tmp_path, monkeypatch, (GROUP, 2 * GROUP))
    with storage:
        with pytest.raises(RuntimeError, match='outside.*produced range'):
            write_group(storage, count=1, first=2 * GROUP)
    assert not storage._produced_groups and not pub._prewrites


@pytest.mark.parametrize('batch_range', [(1, GROUP), (0, GROUP + 1), (GROUP, GROUP)])
def test_a_range_is_whole_groups(tmp_path, monkeypatch, batch_range):
    with pytest.raises(ValueError, match='whole groups'):
        _owner(tmp_path, monkeypatch, batch_range)
