"""CPU-only collector tests: no SSH, HTTP or GPU access."""
import json
import subprocess
import urllib.parse

import pytest

from tools import collect_row_netdata as collector


@pytest.fixture
def fake_pair(monkeypatch):
    calls = []
    hosts = ("worker-a", "worker-b")
    monkeypatch.setattr(collector, "HOSTS", hosts)

    def fetch(host, endpoint):
        calls.append((host, endpoint))
        if endpoint == "charts":
            return {"charts": {context: {"context": context}
                               for context in collector.REQUIRED_CONTEXTS}}
        query = urllib.parse.parse_qs(endpoint.split("?", 1)[1])
        assert float(query["after"][0]) == 100 and float(query["before"][0]) == 110
        assert query["options"] == ["seconds"]
        return {"labels": ["time", "value"], "data": [[100, 1.0], [110, 2.0]]}

    monkeypatch.setattr(collector, "fetch", fetch)
    return hosts, calls, fetch


def test_collect_window_returns_both_complete_validated_hosts(fake_pair):
    hosts, calls, _ = fake_pair
    result = collector.collect_window(100, 110)
    assert result["schema"] == "prismaquant.row_netdata.v1"
    assert list(result["hosts"]) == list(hosts)
    for body in result["hosts"].values():
        assert set(body["charts"]) == collector.REQUIRED_CONTEXTS
        assert all(series["points"] == 2 for series in body["series"].values())
    assert len(calls) == 2 * (1 + len(collector.REQUIRED_CONTEXTS))


def test_cli_serializes_the_same_window(fake_pair, capsys):
    assert collector.main(["--after", "100", "--before", "110"]) == 0
    assert len(json.loads(capsys.readouterr().out)["hosts"]) == 2


@pytest.mark.parametrize("after,before", [(110, 100), (100, 100),
                                            (float("nan"), 110), (100, float("inf"))])
def test_invalid_bounds_refuse_before_transport(after, before, monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid window reached transport")

    monkeypatch.setattr(collector, "fetch", forbidden)
    with pytest.raises(ValueError, match="finite after < before"):
        collector.collect_window(after, before)
    with pytest.raises(SystemExit) as error:
        collector.main(["--after", str(after), "--before", str(before)])
    assert error.value.code == 2


def test_missing_context_on_second_host_never_returns_a_partial_pair(fake_pair, monkeypatch):
    hosts, _, original = fake_pair

    def fetch(host, endpoint):
        if host == hosts[1] and endpoint == "charts":
            return {"charts": {}}
        return original(host, endpoint)

    monkeypatch.setattr(collector, "fetch", fetch)
    with pytest.raises(SystemExit, match="required netdata contexts missing on worker-b"):
        collector.collect_window(100, 110)


@pytest.mark.parametrize("bad", [
    None, {"labels": ["time", "x"], "data": None},
    {"labels": ["time", "x"], "data": [[99, 1]]},
    {"labels": ["time", "x"], "data": [[100, float("nan")]]},
    {"labels": ["time", "x"], "data": [[100, None]]},
    {"labels": ["time", "x"], "data": [["invalid", 1]]},
])
def test_bad_telemetry_keeps_validator_guard_and_adds_host_chart_context(
        bad, fake_pair, monkeypatch):
    _, _, original = fake_pair

    def fetch(host, endpoint):
        return original(host, endpoint) if endpoint == "charts" else bad

    monkeypatch.setattr(collector, "fetch", fetch)
    with pytest.raises(RuntimeError, match=r"worker-a .*asked 100\.\.110"):
        collector.collect_window(100, 110)


def test_transport_failure_on_second_host_is_not_a_partial_success(fake_pair, monkeypatch):
    hosts, _, original = fake_pair

    def fetch(host, endpoint):
        if host == hosts[1]:
            raise subprocess.TimeoutExpired("transport", 15)
        return original(host, endpoint)

    monkeypatch.setattr(collector, "fetch", fetch)
    with pytest.raises(subprocess.TimeoutExpired):
        collector.collect_window(100, 110)
