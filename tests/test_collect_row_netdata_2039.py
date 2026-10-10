"""PQ #2039: the both-Spark Netdata collector returns the pair or refuses.

``tools/collect_row_netdata.py`` is the one collector of both Sparks' series
(PQ #2463). The paired copy measure calls ``collect_window`` instead of copying
its loop, so the pair contract is pinned here: a document names both hosts and
every required context on each, a host missing a context refuses, and a window
whose samples fall outside the requested range refuses rather than passing.
"""
import pytest

from tools import collect_row_netdata as collector

AFTER, BEFORE = 1000.0, 1030.0


def _charts(contexts=None):
    contexts = collector.REQUIRED_CONTEXTS if contexts is None else contexts
    return {"charts": {f"chart.{context}": {"context": context} for context in sorted(contexts)}}


def _window(*, stamps=(1001.0, 1010.0, 1020.0)):
    return {"labels": ["time", "dim"], "data": [[stamp, 1.5] for stamp in stamps]}


def _fake_fetch(monkeypatch, *, charts=None, window=None, calls=None):
    def fetch(host, endpoint):
        if calls is not None:
            calls.append((host, endpoint.split("?")[0]))
        if endpoint == "charts":
            return (charts or {}).get(host, _charts())
        return window if window is not None else _window()

    monkeypatch.setattr(collector, "fetch", fetch)


def test_the_document_holds_both_hosts_and_every_required_context(monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls=calls)
    document = collector.collect_window(AFTER, BEFORE)
    assert document["schema"] == "prismaquant.row_netdata.v1"
    assert (document["after"], document["before"]) == (AFTER, BEFORE)
    assert sorted(document["hosts"]) == sorted(collector.HOSTS) == ["sparklina", "sparky"]
    for host, row in document["hosts"].items():
        contexts = {chart.split(".", 1)[1] for chart in row["charts"]}
        assert contexts == collector.REQUIRED_CONTEXTS
        assert all(series["points"] == 3 for series in row["series"].values())
    # Each host answered a chart listing and one data window per required chart.
    for host in collector.HOSTS:
        assert (host, "charts") in calls
        assert sum(1 for h, endpoint in calls if h == host and endpoint == "data") == 4


def test_a_host_missing_a_required_context_refuses_the_whole_pair(monkeypatch):
    short = _charts(collector.REQUIRED_CONTEXTS - {"system.io"})
    _fake_fetch(monkeypatch, charts={"sparky": short})
    with pytest.raises(SystemExit, match="required netdata contexts missing on sparky"):
        collector.collect_window(AFTER, BEFORE)


def test_a_sample_outside_the_requested_window_refuses(monkeypatch):
    _fake_fetch(monkeypatch, window=_window(stamps=(1001.0, 1050.0)))
    with pytest.raises(RuntimeError, match="outside its requested window"):
        collector.collect_window(AFTER, BEFORE)


def test_a_refused_window_names_its_host_chart_and_stamps(monkeypatch):
    _fake_fetch(monkeypatch, window=_window(stamps=(999.0, 1010.0)))
    with pytest.raises(RuntimeError) as error:
        collector.collect_window(AFTER, BEFORE)
    text = str(error.value)
    assert "sparklina chart." in text and "asked 1000.0..1030.0" in text
    assert "samples 999.0..1010.0" in text


def test_a_window_with_no_measured_dimension_refuses(monkeypatch):
    _fake_fetch(monkeypatch, window={"labels": ["time", "dim"], "data": [[1001.0, None]]})
    with pytest.raises(RuntimeError, match="no measured samples"):
        collector.collect_window(AFTER, BEFORE)


def test_the_command_line_refuses_an_empty_window(capsys):
    with pytest.raises(SystemExit):
        collector.main(["--after", "5", "--before", "5"])
    assert "need --after < --before" in capsys.readouterr().err
