"""A research figure must plot the prediction bytes its audit actually binds."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest



FIXTURES = Path(__file__).parent / "fixture-data" / "family-transfer-figure"


class _ReachedPlotBoundary(Exception):
    pass


@pytest.fixture
def figure(monkeypatch):
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = None
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    source = Path(__file__).resolve().parents[1] / "experiments" / "plot_sparse_rate_family_transfer.py"
    spec = importlib.util.spec_from_file_location("_figure_seal_contract", source)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "matplotlib", matplotlib)
        imports.setitem(sys.modules, "matplotlib.pyplot", pyplot)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tamper", [False, True], ids=["retained-seal", "changed-prediction"])
def test_figure_binds_exact_prediction_body_before_plotting(tmp_path, monkeypatch, figure, tamper):
    artifact = tmp_path / "artifacts"
    shutil.copytree(FIXTURES, artifact)
    path = artifact / figure.FILES["e4_seal"]
    raw = path.read_bytes()
    retained = json.loads(raw)
    if tamper:
        changed = deepcopy(retained)
        row = changed["predictions"][1]
        assert row["rate"] not in (
            changed["predictions"][0]["rate"], changed["predictions"][-1]["rate"])
        row["affine_bf_from_paired_endpoints"] *= 1.5
        assert changed["seal_sha256"] == retained["seal_sha256"]
        path.write_text(json.dumps(changed, sort_keys=True, indent=2) + "\n")
        assert hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(raw).digest()
    else:
        assert path.read_bytes() == raw

    seen = []

    def plotting(*args, **kwargs):
        seen.append("plot")
        raise _ReachedPlotBoundary

    monkeypatch.setattr(figure.plt, "subplots", plotting)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "plot_sparse_rate_family_transfer.py", "--artifact-dir", str(artifact),
        "--out", str(out),
    ])
    if tamper:
        with pytest.raises(ValueError, match="seal.*(digest|hash|bytes)"):
            figure.main()
        assert seen == []
    else:
        with pytest.raises(_ReachedPlotBoundary):
            figure.main()
        assert seen == ["plot"]
    assert not out.exists()
