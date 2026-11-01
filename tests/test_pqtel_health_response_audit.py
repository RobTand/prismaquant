"""Real health CLI/report; only external host and HTTP observations are doubled."""

import builtins
import io
import json
import os
from types import SimpleNamespace

import pytest

from pqtel import cli, health


def _host_observations(monkeypatch, body, status=200):
    real_open = builtins.open
    real_listdir = os.listdir
    real_stat = os.stat
    real_statvfs = os.statvfs

    def run(command, **kwargs):
        if command[0] == "loginctl":
            return SimpleNamespace(stdout="Linger=yes\n", returncode=0)
        assert command[0] == "systemctl", command
        if "show" in command:
            output = "Restart=always\nOOMScoreAdjust=-500\nOOMPolicy=continue\n"
        elif "is-enabled" in command:
            output = "disabled\n"
        else:
            output = "active\n"
        return SimpleNamespace(stdout=output, returncode=0)

    def open_observation(path, *args, **kwargs):
        if path == "/proc/sys/kernel/perf_event_paranoid":
            return io.StringIO("2\n")
        if path == "/proc/driver/nvidia/params":
            return io.StringIO("RmProfilingAdminOnly: 0\n")
        return real_open(path, *args, **kwargs)

    def listdir(path):
        if path in (health.CSV_DIR, health.OLD_VITALS_DIR):
            return []
        return real_listdir(path)

    def stat(path, *args, **kwargs):
        if path == health.GUARD_FLAG:
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    def statvfs(path):
        if path == "/home/rob":
            return SimpleNamespace(f_bavail=50, f_blocks=100, f_frsize=2**30)
        return real_statvfs(path)

    class Response(io.BytesIO):
        pass

    def urlopen(url, timeout):
        assert url == health.NETDATA_LOCAL + "/api/v1/info"
        response = Response(body)
        response.status = status
        return response

    monkeypatch.setattr(health.subprocess, "run", run)
    monkeypatch.setattr(health.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(builtins, "open", open_observation)
    monkeypatch.setattr(health.os, "listdir", listdir)
    monkeypatch.setattr(health.os, "stat", stat)
    monkeypatch.setattr(health.os, "statvfs", statvfs)


def _report(monkeypatch, capsys, body, status=200):
    _host_observations(monkeypatch, body, status)
    assert cli.main(["health", "--json", "--local-only"]) == 0
    boxes = json.loads(capsys.readouterr().out)
    assert len(boxes) == 1
    checks = {row["check"]: row for row in boxes[0]["checks"]}
    # This observation occurs after Netdata intake in the real run().
    assert checks["free disk"]["status"] == health.OK
    assert "50.0%" in checks["free disk"]["detail"]
    return checks["netdata reachable"]


@pytest.mark.parametrize("body", [b"{not-json", b"[]"], ids=["invalid-json", "array"])
def test_bad_http_200_body_preserves_health_report(monkeypatch, capsys, body):
    row = _report(monkeypatch, capsys, body)
    assert row["check"] == "netdata reachable"
    assert health.NETDATA_LOCAL in row["detail"]


def test_valid_netdata_info_is_reported(monkeypatch, capsys):
    row = _report(
        monkeypatch, capsys,
        b'{"version":"audit-version","mirrored_hosts":["audit-host"]}',
    )
    assert row["status"] == health.OK
    assert "audit-version" in row["detail"]
    assert "audit-host" in row["detail"]


def test_unavailable_netdata_preserves_health_report(monkeypatch, capsys):
    row = _report(monkeypatch, capsys, b"unavailable", status=503)
    assert row["status"] == health.DEFECT


def test_invalid_local_info_is_an_invalid_observation(monkeypatch, capsys):
    row = _report(monkeypatch, capsys, b"[]")
    assert row["status"] == health.DEFECT
    assert "invalid" in row["detail"].lower()


def test_invalid_remote_info_is_unverified(monkeypatch):
    remote = "http://audit.invalid:19999"

    def urlopen(url, timeout):
        assert url == remote + "/api/v1/info"
        response = io.BytesIO(b"{not-json")
        response.status = 200
        return response

    monkeypatch.setattr(health.urllib.request, "urlopen", urlopen)
    report = health.Report()
    box = report.box("audit peer (remote)")
    assert health.check_netdata_reachable(report, box, remote, "netdata reachable") is None
    row = box["checks"][0]
    assert row["status"] == health.UNVERIFIED
    assert "invalid" in row["detail"].lower()
    assert report.render()[2] == 1


@pytest.mark.parametrize("body", [b"{}", b'{"mirrored_hosts":null}', b'{"mirrored_hosts":[]}'])
def test_mapping_without_optional_netdata_metadata_keeps_existing_defaults(monkeypatch, capsys, body):
    row = _report(monkeypatch, capsys, body)
    assert row["status"] == health.OK
    assert "netdata ?, host ?" in row["detail"]


@pytest.mark.parametrize(
    "hosts", [{"host": "audit-host"}, "audit-host", [123]],
    ids=["object", "string", "nonstring-entry"],
)
def test_malformed_mirrored_hosts_is_an_invalid_observation(monkeypatch, capsys, hosts):
    row = _report(monkeypatch, capsys, json.dumps({"mirrored_hosts": hosts}).encode())
    assert row["status"] == health.DEFECT
    assert "invalid" in row["detail"].lower()


def test_normal_multiple_mirrored_hosts_keeps_first_host(monkeypatch, capsys):
    row = _report(
        monkeypatch, capsys,
        b'{"version":"audit-version","mirrored_hosts":["audit-first","audit-second"]}',
    )
    assert row["status"] == health.OK
    assert "host audit-first" in row["detail"]
    assert "audit-second" not in row["detail"]
