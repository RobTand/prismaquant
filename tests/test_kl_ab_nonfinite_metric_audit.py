"""Real compare/CLI finite-input admission, with actual fingerprint replay."""
from __future__ import annotations

import json

import pytest

from tests.test_gold_comparison_population_audit import _record
from tools import kl_ab


@pytest.mark.parametrize("entrypoint", ["compare", "cli"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "positive_infinity", "negative_infinity"])
def test_explicit_nonfinite_gold_metric_never_reports_a_successful_delta(
    tmp_path, capsys, entrypoint, invalid,
):
    a = _record(tmp_path, tag="a", metric="kl_confident_mean", seqlen=4)
    b = _record(tmp_path, tag="b", metric="kl_confident_mean", seqlen=4)
    b["kl_confident_mean"] = invalid
    if entrypoint == "compare":
        code, lines = kl_ab.compare(a, b, metric="kl_confident_mean")
        text = "\n".join(lines)
    else:
        paths = [tmp_path / "a.json", tmp_path / "b.json"]
        for path, record in zip(paths, (a, b)):
            path.write_text(json.dumps(record, allow_nan=True))
        before = [path.read_bytes() for path in paths]
        code = kl_ab.main([*(str(path) for path in paths), "--metric", "kl_confident_mean"])
        text = capsys.readouterr().out
        assert [path.read_bytes() for path in paths] == before
    assert "INVALID ATTESTATION" not in text, text
    assert code == 3, text
    assert "delta (" not in text, text


def test_finite_zero_metric_remains_a_valid_comparison(tmp_path):
    a = _record(tmp_path, tag="a", metric="kl_confident_mean", seqlen=4)
    b = _record(tmp_path, tag="b", metric="kl_confident_mean", seqlen=4)
    b["kl_confident_mean"] = 0.0
    code, lines = kl_ab.compare(a, b, metric="kl_confident_mean")
    assert code == 0, "\n".join(lines)
    assert "delta (" in "\n".join(lines)


def test_automatic_metric_selection_already_avoids_a_nonfinite_preferred_value(tmp_path, capsys):
    a = _record(tmp_path, tag="a", metric="kl_confident_mean", seqlen=4)
    b = _record(tmp_path, tag="b", metric="kl_confident_mean", seqlen=4)
    b["kl_confident_mean"] = float("nan")
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, record in zip(paths, (a, b)):
        path.write_text(json.dumps(record, allow_nan=True))
    assert kl_ab.main([*(str(path) for path in paths)]) == 0
    assert "metric: kl_mean" in capsys.readouterr().out
