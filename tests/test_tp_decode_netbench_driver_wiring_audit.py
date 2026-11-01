"""Actual CLI driver with CPU tensor/collective and external-binary adapters.

The adapters expose Python wiring only. Their discarded scratch outputs are
synthetic and establish no netbench, network, serving or performance result.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools import tp_decode_feasibility as tool


class _CollectiveAdapter:
    ReduceOp = SimpleNamespace(MIN="min")

    def __init__(self):
        self.reductions = 0
        self.broadcasts = []
        self.closed = False

    def all_reduce(self, tensor, op=None):
        self.reductions += 1

    def barrier(self):
        pass

    def broadcast(self, tensor, src):
        self.broadcasts.append(src)
        if src == 1:
            tensor.fill_(6.25)
        # Match the synchronous public API: mutate the supplied tensor.
        return None

    def destroy_process_group(self):
        self.closed = True


class _ServerAdapter:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True


def _run_driver(monkeypatch, capsys, *, expose_torch, iperf_available):
    collective = _CollectiveAdapter()
    initialized = []

    def initialize(rank, world_size, init_method):
        initialized.append((rank, world_size, init_method))
        return torch, collective

    monkeypatch.setattr(tool, "_init_pg", initialize)
    if expose_torch:
        # Isolate the independent bandwidth receive issue from the missing
        # local torch binding. No production initialization result is forged.
        monkeypatch.setattr(tool, "torch", torch, raising=False)
    real_run, real_popen = subprocess.run, subprocess.Popen
    servers = []

    def run(argv, *args, **kwargs):
        if argv == ["which", "iperf3"]:
            return subprocess.CompletedProcess(argv, 0 if iperf_available else 1)
        return real_run(argv, *args, **kwargs)

    def popen(argv, *args, **kwargs):
        if argv[:2] == ["iperf3", "-s"]:
            server = _ServerAdapter()
            servers.append(server)
            return server
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    # The real scratch validator refuses /tmp. This owned non-/tmp fixture
    # uses its actual accepted path contract and is always removed on exit.
    with tempfile.TemporaryDirectory(prefix="pq-tp-driver-audit-", dir="/var/tmp") as scratch:
        output = Path(scratch) / "driver.json"
        argv = ["--mode", "netbench", "--rank", "0", "--peer", "127.0.0.1",
                "--sizes-bytes", "2", "--warmup", "200", "--iters", "2000",
                "--iperf3-time", "1", "--json-out", str(output)]
        assert tool.main(argv) == 0
        payload = json.loads(output.read_bytes())
        assert initialized == [(0, 2, "tcp://127.0.0.1:29517")]
        assert collective.reductions >= 2200
        assert collective.closed
        assert payload["results"][0]["bytes"] == 2
        assert payload["results"][0]["n"] == 2000
        return payload, collective, servers, capsys.readouterr().out


def test_netbench_uses_the_torch_module_returned_by_its_device_owner(monkeypatch, capsys):
    payload, _, _, _ = _run_driver(monkeypatch, capsys, expose_torch=False, iperf_available=False)
    assert payload["iperf3"] is None


def test_declared_device_adapter_reaches_normal_driver_completion(monkeypatch, capsys):
    payload, _, _, output = _run_driver(monkeypatch, capsys, expose_torch=True, iperf_available=False)
    assert payload["iperf3"] is None
    assert "iperf3 not present on both boxes" in output


def test_bandwidth_receive_reads_the_tensor_mutated_by_broadcast(monkeypatch, capsys):
    payload, collective, servers, output = _run_driver(
        monkeypatch, capsys, expose_torch=True, iperf_available=True)
    assert collective.broadcasts == [0, 1]
    assert servers and all(server.terminated for server in servers)
    assert payload["iperf3"] == {"ran": True, "gbits_s": 6.25, "seconds": 1}, output


def test_rank_one_cli_uses_initialized_torch_and_sends_its_bandwidth_value(monkeypatch, capsys):
    class SendingAdapter(_CollectiveAdapter):
        def broadcast(self, tensor, src):
            self.broadcasts.append((src, float(tensor.item())))
            return None

    collective = SendingAdapter()
    initialized = []

    def initialize(rank, world_size, init_method):
        initialized.append((rank, world_size, init_method))
        return torch, collective

    commands = []

    def external(argv, *args, **kwargs):
        commands.append(argv)
        if argv == ["which", "iperf3"]:
            return subprocess.CompletedProcess(argv, 0)
        assert argv[:2] == ["iperf3", "-c"]
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({
            "end": {"sum_received": {"bits_per_second": 10.5e9}}}))

    monkeypatch.setattr(tool, "_init_pg", initialize)
    monkeypatch.delattr(tool, "torch", raising=False)
    monkeypatch.setattr(subprocess, "run", external)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("rank one started a server"))
    monkeypatch.setattr(tool, "_write_json", lambda *a, **k: pytest.fail("rank one published rank zero's report"))
    assert tool.main(["--mode", "netbench", "--rank", "1", "--peer", "127.0.0.1",
                      "--sizes-bytes", "2", "--warmup", "200", "--iters", "2000",
                      "--iperf3-time", "1"]) == 0
    assert initialized == [(1, 2, "tcp://127.0.0.1:29517")]
    assert collective.reductions == 2201 and collective.closed
    assert collective.broadcasts == [(0, 0.0), (1, 10.5)]
    assert commands == [["which", "iperf3"],
                        ["iperf3", "-c", "127.0.0.1", "-p", "29518", "-t", "1", "-J"]]
    assert "MEASURED 10.50" not in capsys.readouterr().out
