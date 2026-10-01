"""Streamed calibration capture as retryable layer-chain quanta (PQ #1885)."""
import json
from pathlib import Path

import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import capture  # noqa: F401  (fixture)


def test_writer_finish_publishes_journal_completed_records(capture, tmp_path):  # noqa: F811
    """A later process publishes the units an earlier one journalled.

    Each chain quantum writes only its own layers' units, so the process that
    finishes the capture holds none of them in memory. ``finish`` must
    publish the journal's completed records, not only this process's.
    """
    _root, path, census, identity, acts, hessians, monolith = capture
    root = tmp_path/'chain'
    first = cc.CaptureWriter(root, census_path=path, identity=identity)
    first.write(acts={'a': acts['a']}, hessians={'a': hessians['a']},
                counts={'a': census['counts']['a']}, maxima={'a': census['max_abs']['a']})
    del first
    second = cc.CaptureWriter(root, census_path=path, identity=identity)
    second.write(acts={'b': acts['b']}, hessians={'b': hessians['b']},
                 counts={'b': census['counts']['b']}, maxima={'b': census['max_abs']['b']})
    record = second.finish(model_load_contract=identity['model_load_contract'])
    published = json.loads(Path(record['path']).read_text())
    assert set(published['entries']) == {'a', 'b'}
    assert published == json.loads(Path(monolith['path']).read_text())
    for name in ('a', 'b'):
        entry = torch.load(root/published['entries'][name]['path'], weights_only=True)
        assert torch.equal(entry['inputs'], acts[name])
        assert torch.equal(entry['hessian'], hessians[name])
