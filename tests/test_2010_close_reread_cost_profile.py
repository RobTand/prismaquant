"""Temporary cost profile for the PQ #2010 close reread (deleted before merge).

It times one record/read/close capture with the close reread on and off.
It prints admission hash time, close reread time, and capture wall delta.
"""
import json
import time
from pathlib import Path

import torch


def _source(tmp_path, shards=4, size_mb=64):
    from safetensors.torch import save_file
    source = tmp_path / 'source'
    source.mkdir()
    names = []
    cols = 256
    rows = (size_mb * 1024 * 1024) // (4 * cols)
    for i in range(shards):
        name = f'model-{i:05d}.safetensors'
        save_file({'w': torch.randn(rows, cols)}, str(source / name))
        names.append(name)
    path = tmp_path / 'census.json'
    path.write_text(json.dumps({'model': str(source)}))
    return source, path, names


def _capture(source, census, names):
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    start = time.perf_counter()
    owner = cc.record_capture_source(census, model=source)
    for name in names:
        with owner.safe_open(safe_open, source / name, framework='pt') as reader:
            reader.get_tensor('w')
    owner.close()
    return time.perf_counter() - start


def test_close_reread_cost_profile(tmp_path, monkeypatch):
    import os
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc
    source, census, names = _source(tmp_path)
    total_mb = sum((source / n).stat().st_size for n in names) / 1024**2

    real_verify = cc.CaptureSourceAuthentication._verify_held_content
    shard = source / names[0]
    t0 = time.perf_counter()
    owner = cc.record_capture_source(census, model=source)
    with owner.safe_open(safe_open, shard, framework='pt') as reader:
        reader.get_tensor('w')
    t_admit = time.perf_counter() - t0
    t0 = time.perf_counter()
    real_verify(owner, shard.name, owner._files[shard.name])
    t_reread = time.perf_counter() - t0
    owner.close()

    # Warm run first so both arms meet hot page cache, as in production.
    _capture(source, census, names)
    monkeypatch.setattr(cc.CaptureSourceAuthentication, '_verify_held_content',
        lambda self, n, s: None)
    off = min(_capture(source, census, names) for _ in range(3))
    monkeypatch.undo()
    on = min(_capture(source, census, names) for _ in range(3))
    overhead_pct = 100.0 * (on - off) / off
    print(f'[pq2010-cost] files={len(names)} total_mb={total_mb:.1f} '
          f'admit_one_shard_s={t_admit:.3f} reread_one_shard_s={t_reread:.3f} '
          f'capture_off_s={off:.3f} capture_on_s={on:.3f} overhead_pct={overhead_pct:.1f} '
          f'loadavg={os.getloadavg()}')
