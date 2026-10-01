"""The projected-unit byte check compares on the device, one sync per pass (PQ #1935).

``_checked_projected_units`` read each unit's source tensor and copied the
live view to the host (``.cpu()``) to compare it there: one device sync and
one pageable allocation per unit, thousands per GLM-5.3 layer, with the GPU
idle at ~15 W. It now stages the source bytes to the live tensor's device and
reads every unit's verdict with one sync. The refusal is unchanged: a byte
mismatch anywhere refuses with ``PROJECTED_BYTES_REFUSAL``, naming each unit.

``_reference_check_unit`` and ``_reference_checked_units`` are the old
pair verbatim, from ``prismaquant/tessera_campaign.py`` at f5869e0c; only
their names (and the one call between them) changed.
"""
from __future__ import annotations

import pytest
import torch

from cuda_sync_sites import sync_sites
from prismaquant import tessera_campaign as campaign
from prismaquant.tessera_campaign import (
    PROJECTED_BYTES_REFUSAL, _measured_projected_units, _read_projected_unit)


def _reference_check_unit(name, unit, *, live, model_path, source,
                          release_source_pages=False, source_authentication=None):
    """Read one unit's source tensor and compare it byte-for-byte with ``live``.

    Returns ``None`` when the two are equal, else the mismatch description the
    refusal names. The bytes read are the shard the producer hashed, through
    the owner the snapshot read it with. With ``release_source_pages`` the
    unit's own consumed span is advised as soon as it compares equal. Nothing
    here reserves memory: the serial caller reserves around it.
    """
    import torch

    weight, release = _read_projected_unit(
        name, unit, model_path=model_path, source=source,
        release_source_pages=release_source_pages,
        source_authentication=source_authentication)
    live = live.detach().cpu()
    if live.dtype != weight.dtype or not torch.equal(live, weight):
        return (f"{name} (live {tuple(live.shape)} {live.dtype} vs source "
                f"{unit['source_tensor']} {tuple(weight.shape)} {weight.dtype})")
    del weight, live
    release()
    return None


def _reference_checked_units(bound, *, weights, model_path, source,
                             measured=None, resource_check=None,
                             release_source_pages=False, source_authentication=None) -> dict[str, dict]:
    """The producer's unit records for the units this run prices, bytes checked.

    Each unit's source tensor is read from the shard the producer hashed and
    compared byte-for-byte with the live view this run prices, so the exporter
    cannot encode bytes this table did not price (PrismaQuant #183).  Only the
    ``measured`` units are read: a shard cannot check a tensor it never loaded,
    and claiming it had would be the assertion the check exists to replace.

    This is the serial pass the load-all head and the census run. The stream
    head snapshots no projected unit, so it has nothing to compare: its readers
    price the source tensor itself (``_read_projected_unit``, PQ #1654).
    """
    projected: dict[str, dict] = {}
    mismatched: list[str] = []
    for name, unit in _measured_projected_units(bound, measured).items():
        if resource_check is not None:
            resource_check(f'before_source_projection_check:{name}')
        mismatch = _reference_check_unit(
            name, unit, live=weights[name], model_path=model_path, source=source,
            release_source_pages=release_source_pages,
            source_authentication=source_authentication)
        if mismatch is None:
            projected[name] = unit
        else:
            mismatched.append(mismatch)
        if resource_check is not None:
            resource_check(f'after_source_projection_check:{name}')
    if mismatched:
        raise RuntimeError(PROJECTED_BYTES_REFUSAL.format(units=", ".join(mismatched)))
    return projected


ROWS, COLS = 4, 8


def _devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _source(tmp_path, count, dtype=torch.bfloat16):
    """``count`` units in two shards, and live views equal to them."""
    from safetensors.torch import save_file

    root = tmp_path / "source"
    root.mkdir()
    gen = torch.Generator().manual_seed(1935)
    tensors = {f"t{i}": torch.randn(ROWS, COLS, generator=gen).to(dtype) for i in range(count)}
    files = {}
    for shard in (0, 1):
        names = [n for i, n in enumerate(tensors) if i % 2 == shard]
        save_file({n: tensors[n] for n in names}, str(root / f"shard{shard}.safetensors"))
        files.update({n: f"shard{shard}.safetensors" for n in names})
    bound = {"s0": {}, "s1": {}}
    for i, name in enumerate(tensors):
        bound[f"s{i % 2}"][f"u{i}"] = {"source_tensor": name, "rows": ROWS, "cols": COLS}
    source = {"tensors": files}
    live = {f"u{i}": tensors[f"t{i}"].clone() for i in range(count)}
    return root, source, bound, live


def _flip_one_byte(tensor, row, col):
    tensor.view(torch.int16)[row, col] ^= 1


def _description(name, tensor_name, live, dtype=torch.bfloat16, shape=(ROWS, COLS)):
    return (f"{name} (live {tuple(live.shape)} {live.dtype} vs source "
            f"{tensor_name} {shape} {dtype})")


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("release", [False, True])
def test_equal_units_pass_and_return_their_records(tmp_path, device, release):
    root, source, bound, live = _source(tmp_path, 6)
    weights = {name: t.to(device) for name, t in live.items()}
    got = campaign._checked_projected_units(bound, weights=weights, model_path=root,
                                            source=source, release_source_pages=release)
    assert got == _measured_projected_units(bound)
    assert got == _reference_checked_units(bound, weights=weights, model_path=root,
                                           source=source, release_source_pages=release)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("release", [False, True])
def test_a_one_byte_mismatch_refuses_and_names_the_unit(tmp_path, device, release):
    root, source, bound, live = _source(tmp_path, 6)
    _flip_one_byte(live["u3"], 2, 5)
    weights = {name: t.to(device) for name, t in live.items()}
    kwargs = dict(weights=weights, model_path=root, source=source,
                  release_source_pages=release)
    with pytest.raises(RuntimeError) as new:
        campaign._checked_projected_units(bound, **kwargs)
    assert str(new.value) == PROJECTED_BYTES_REFUSAL.format(
        units=_description("u3", "t3", weights["u3"]))
    with pytest.raises(RuntimeError) as old:
        _reference_checked_units(bound, **kwargs)
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("device", _devices())
def test_every_mismatch_is_named_in_check_order(tmp_path, device):
    """Two byte flips, a dtype change and a shape change, refused together."""
    root, source, bound, live = _source(tmp_path, 8)
    _flip_one_byte(live["u1"], 0, 0)
    _flip_one_byte(live["u6"], 3, 7)
    live["u2"] = live["u2"].to(torch.float32)
    live["u4"] = live["u4"][:2]
    weights = {name: t.to(device) for name, t in live.items()}
    kwargs = dict(weights=weights, model_path=root, source=source)
    with pytest.raises(RuntimeError) as new:
        campaign._checked_projected_units(bound, **kwargs)
    with pytest.raises(RuntimeError) as old:
        _reference_checked_units(bound, **kwargs)
    assert str(new.value) == str(old.value)
    # s0 holds the even units and s1 the odd ones; the check reads s0 first.
    text = str(new.value)
    assert "u2 (live (4, 8) torch.float32" in text
    assert "u4 (live (2, 8) torch.bfloat16" in text
    assert text.index("u2 (") < text.index("u4 (") < text.index("u6 (") < text.index("u1 (")


def test_the_single_unit_check_keeps_its_contract(tmp_path):
    root, source, bound, live = _source(tmp_path, 2)
    unit = bound["s0"]["u0"]
    assert campaign._check_projected_unit("u0", unit, live=live["u0"], model_path=root,
                                          source=source) is None
    _flip_one_byte(live["u0"], 1, 1)
    assert campaign._check_projected_unit(
        "u0", unit, live=live["u0"], model_path=root, source=source) == _reference_check_unit(
        "u0", unit, live=live["u0"], model_path=root, source=source)


@pytest.mark.parametrize("device", _devices())
def test_the_mismatch_test_bites_when_the_comparison_is_stubbed(tmp_path, device, monkeypatch):
    """Mutate the comparison, not the fixture: with it stubbed, nothing refuses."""
    root, source, bound, live = _source(tmp_path, 6)
    _flip_one_byte(live["u3"], 2, 5)
    weights = {name: t.to(device) for name, t in live.items()}
    kwargs = dict(weights=weights, model_path=root, source=source)
    with pytest.raises(RuntimeError):
        campaign._checked_projected_units(bound, **kwargs)
    monkeypatch.setattr(campaign, "_device_differs",
                        lambda live, staged: torch.zeros((), dtype=torch.bool, device=live.device))
    monkeypatch.setattr(campaign, "_host_differs", lambda live, weight: False)
    assert campaign._checked_projected_units(bound, **kwargs) == _measured_projected_units(bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="counts CUDA host syncs")
def test_the_check_syncs_once_per_pass_not_once_per_unit(tmp_path):
    count = 48
    root, source, bound, live = _source(tmp_path, count)
    weights = {name: t.cuda() for name, t in live.items()}
    kwargs = dict(weights=weights, model_path=root, source=source)
    campaign._checked_projected_units(bound, **kwargs)  # warm the pinned staging cache
    old = sync_sites(lambda: _reference_checked_units(bound, **kwargs))
    assert len(old) >= count, old  # the instrument sees the per-unit loop
    new = sync_sites(lambda: campaign._checked_projected_units(bound, **kwargs))
    assert len(new) <= 1, new
