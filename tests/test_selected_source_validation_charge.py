"""A selected row's source-validation charge follows the reads the row makes.

RobTand/prismaquant#1491: a four-layer GLM-5.3 shared-expert row selected
134 MB of source weight and was charged 84 GiB, 55.2 GiB of it a
``source_validation_bytes`` term that summed every tensor of every selected
layer. Two runs of that row peaked at 11.1 and 15.2 GiB on the box.

What the row actually reads, and when:

* The source hash runs in ``source_preparation``. The first payload read of a
  file hashes the whole file, once per row, a block at a time with each
  block's pages released (``tessera_calibration_cache.sha256`` with
  ``release_read_pages``). So its charge is one hash window per distinct file,
  not per layer.
* ``resident_anchors`` and ``stream_projection`` read only the projection's
  byte check: selected source tensors, whose pages are released after the
  loop. So that charge is the selected tensors' stored bytes, not the layer's.

The checkpoint here is real (safetensors shards and an index); only the model
profile and the declared class are stood in, as in
``tests/test_streaming_buffer_precision.py``.
"""
import json

import torch
from safetensors.torch import save_file


LAYERS = 4
SELECTED = [8, 16]  # shared-expert-like [out, in] BF16 weight, selected
UNSELECTED = [64, 64]  # routed-expert-like BF16 weight, never selected


def _checkpoint(tmp_path):
    """Four decoder layers over two shards, each layer a small selected
    weight beside a large unselected one, the shape of a GLM-5.3 MoE layer."""
    shards = {'model-00001-of-00002.safetensors': {}, 'model-00002-of-00002.safetensors': {}}
    names = sorted(shards)
    for layer in range(LAYERS):
        shard = shards[names[layer // 2]]
        shard[f'layers.{layer}.shared.weight'] = torch.ones(SELECTED, dtype=torch.bfloat16)
        shard[f'layers.{layer}.routed.weight'] = torch.ones(UNSELECTED, dtype=torch.bfloat16)
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(tmp_path / name))
        weight_map.update(dict.fromkeys(tensors, name))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata={}, weight_map=weight_map)))
    (tmp_path / 'config.json').write_text(json.dumps(dict(model_type='llama',
        hidden_size=16, num_hidden_layers=LAYERS, num_attention_heads=1,
        num_key_value_heads=1, intermediate_size=16, vocab_size=8)))


def _plan(tmp_path, monkeypatch, policy):
    from prismaquant import autoscale, model_profiles, streaming_model
    from prismaquant.model_profiles import DefaultProfile
    profile = DefaultProfile()
    monkeypatch.setattr(profile, 'body_layer_prefix', lambda: 'layers')
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: profile)
    monkeypatch.setattr(streaming_model, '_resolve_declared_model_cls', lambda *_: torch.nn.Module)
    shapes = {f'layers.{layer}.shared': SELECTED for layer in range(LAYERS)}
    return autoscale.selected_anchor_resources(tmp_path, unit_shapes=shapes,
        counts=dict.fromkeys(shapes, 1), max_act_rows=1, cache_slots=2,
        prefetch_workers=1, headroom_gb=0, source_snapshot_policy=policy,
        capture_load_policy=dict(schema='prismaquant.verified_activation_load.v1',
                                 max_buffer_bytes=4096, max_scratch_bytes=1024**2))


def test_selected_row_is_not_charged_whole_layers_it_never_reads(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    plan = _plan(tmp_path, monkeypatch, 'selected-tensors-v1')
    anchors = plan['phases']['resident_anchors']
    selected_bytes = LAYERS * SELECTED[0] * SELECTED[1] * 2
    layer_bytes = LAYERS * (SELECTED[0] * SELECTED[1] + UNSELECTED[0] * UNSELECTED[1]) * 2
    widest_weight = SELECTED[0] * SELECTED[1] * 4
    # The RED line on main: it charges every tensor of the four layers.
    assert anchors['source_validation_bytes'] < layer_bytes + widest_weight, (
        'selected row charged the whole-layer raw-page allowance '
        f"({anchors['source_validation_bytes']} B for {selected_bytes} B selected)")

    from prismaquant.tessera_calibration_cache import SOURCE_HASH_BLOCK_BYTES
    hash_window = 2 * SOURCE_HASH_BLOCK_BYTES
    # The projection's page window (the selected tensors), its two per-unit
    # copies, and one serial hash window.
    assert anchors['source_validation_bytes'] == selected_bytes + widest_weight + hash_window
    # The stream head's projection phase charges the same term.
    assert plan['stream_phases']['stream_projection']['source_validation_bytes'] == (
        anchors['source_validation_bytes'])
    # The hash is charged where it runs: one window for each distinct file
    # the selected tensors live in, two here.
    preparation = plan['phases']['source_preparation']
    assert preparation['source_authentication_window_bytes'] == 2 * hash_window
    assert plan['memory_bytes'] == max(sum(p.values()) for p in plan['phases'].values())


def test_whole_layer_snapshot_keeps_its_whole_layer_charge(tmp_path, monkeypatch):
    """Under whole-layer-v1 the row reads every tensor of its layers, so the
    projection window stays the layer; only the hash window is added."""
    _checkpoint(tmp_path)
    plan = _plan(tmp_path, monkeypatch, 'whole-layer-v1')
    from prismaquant.tessera_calibration_cache import SOURCE_HASH_BLOCK_BYTES
    hash_window = 2 * SOURCE_HASH_BLOCK_BYTES
    layer_bytes = LAYERS * (SELECTED[0] * SELECTED[1] + UNSELECTED[0] * UNSELECTED[1]) * 2
    widest_weight = SELECTED[0] * SELECTED[1] * 4
    anchors = plan['phases']['resident_anchors']
    assert anchors['source_validation_bytes'] == layer_bytes + widest_weight + hash_window
    assert plan['phases']['source_preparation']['source_authentication_window_bytes'] == (
        2 * hash_window)


def test_the_hash_window_is_the_block_the_guarded_hash_releases(tmp_path, monkeypatch):
    """The admission constant and the hash's release step are one number:
    each advice covers exactly the block just digested, so a hash never holds
    more than one block of read pages."""
    import os
    from prismaquant import tessera_calibration_cache as cc
    block = cc.SOURCE_HASH_BLOCK_BYTES
    path = tmp_path / 'source.bin'
    path.write_bytes(b'z' * (2 * block + 5))
    advice = []
    monkeypatch.setattr(os, 'posix_fadvise',
                        lambda _fd, start, size, _mode: advice.append((start, size)))
    cc.sha256(path, release_read_pages=True)
    page = os.sysconf('SC_PAGE_SIZE')
    assert advice[:2] == [(0, block), (block, block)]
    assert all(size <= block for _start, size in advice)
    assert advice[-1][0] + advice[-1][1] == (2 * block + 5) // page * page
