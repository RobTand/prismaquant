"""The production Gateway consumes a selected native result and admits only the reviewed source and invocation.

The native owner's terminal authority (saved job, launch, runtime, publication, stdout announcement) is a
controlled double here: ``native_results.read_selected_result`` returns the SelectedResult the real reader
would. Every other layer is real: the selection parser, the Gateway mapping, the invocation parser, the
reviewed-source match, completion and counters binding. Authority refusals on real native records (failed,
cancelled, replaced, foreign source, changed publication bytes) are proved in the admitted native job, not here.
These tests establish no genuine GPU pilot qualification.
"""
from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path

import pytest

from test_dispatch_joint_quanta import campaign, records_dir
from test_joint_dispatch_pilot import _argv, _published_pilots, _ready, _pilot_spec
from tools import dispatch_joint_quanta as dispatch

@pytest.fixture(autouse=True)
def _refuse_fleet_spec_reads(monkeypatch, _pilot_spec):
    """Keep the production reader, using the existing portable pilot spec.

    A fleet mount on the verification worker must not conceal a missing
    fixture. Guard reads as well as replacing the dispatcher's spec root.
    """
    def guard(reader, *, opens=False):
        def portable_read(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            reads = not opens or 'r' in mode or '+' in mode
            if reads and not isinstance(path, int):
                absolute = os.path.abspath(os.fsdecode(path))
                assert absolute != '/mnt/shared' and not absolute.startswith('/mnt/shared/'), (
                    'pilot read escaped its hermetic fixture')
            return reader(path, *args, **kwargs)
        return portable_read

    for name in ('read_text', 'read_bytes', 'open'):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name), opens=name == 'open'))
    monkeypatch.setattr(builtins, 'open', guard(builtins.open, opens=True))
    monkeypatch.setattr(io, 'open', guard(io.open, opens=True))


@pytest.mark.parametrize('path', [
    '/mnt/shared', '/mnt/shared/fixture.json', '/mnt//shared/fixture.json',
    '/mnt/shared/../shared/fixture.json',
])
@pytest.mark.parametrize('reader', ['read_text', 'read_bytes', 'path_open', 'builtin_open', 'io_open'])
def test_shared_read_guard_refuses_every_reader(path, reader):
    with pytest.raises(AssertionError, match='pilot read escaped its hermetic fixture'):
        if reader == 'path_open':
            Path(path).open('rb')
        elif reader == 'builtin_open':
            builtins.open(path, 'rb')
        elif reader == 'io_open':
            io.open(path, 'rb')
        else:
            getattr(Path(path), reader)()


class _ProductionPilotGateway(dispatch.FakeGateway):
    """Fixture submit/terminal behaviour with the production pilot reader."""
    read_verified_result = dispatch.Gateway.read_verified_result


FAULTS = [None, "commit", "tree", "entry", "plan", "environment", "wrapper", "legacy",
          "failed", "cancelled", "replaced"]


def _install_native_reader(monkeypatch, canned, fault):
    """Make the real Gateway read ``canned`` results through a double of the native terminal authority."""
    from prismaquant import flux_jobs, native_results

    def read(selection, *, artifact_limits=None):
        result = canned.results[flux_jobs.job_id_number(selection["job_id"])]
        if fault == "failed":
            raise native_results.ProducerFailed(flux_jobs.ExecutionResult(
                "failed", selection["job_id"], Path(selection["record_dir"]), 1))
        if fault == "cancelled":
            raise native_results.ProducerFailed(flux_jobs.ExecutionResult(
                "canceled", selection["job_id"], Path(selection["record_dir"]), None))
        if fault == "replaced":
            raise native_results.ResultUnavailable("publication file identity changed")
        assert set(artifact_limits) == {"completion", "counters"}
        launch = {"argv": list(result["argv"]), "source": dict(result["source"]),
                  "resources": dict(result["resources"]), "afterok": []}
        return native_results.SelectedResult(
            {}, {"completion": result["payload"], "counters": result["counters_bytes"]}, {}, launch,
            flux_jobs.ExecutionResult("succeeded", selection["job_id"], Path(selection["record_dir"]), 0))

    monkeypatch.setattr(native_results, "read_selected_result", read)


def _mutate(canned, fault):
    for result in canned.results.values():
        command = result["argv"]
        if fault in ("commit", "tree"):
            result["source"][fault] = "8" * 40
        elif fault == "entry":
            command[command.index("prismaquant.joint_cost_quantum")] = "foreign.module"
        elif fault == "plan":
            command[command.index("--plan-sha256") + 1] = "e" * 64
        elif fault == "environment":
            spec = json.loads(command[4])
            spec["env"]["PYTHONPATH"] = "/foreign/imports"
            command[4] = json.dumps(spec)
        elif fault == "wrapper":
            command[:3] = ["/bin/echo", "unrelated", "result"]
        elif fault == "legacy":
            result["payload"] = b'{"passed": true, "units_done": 1}\n'


#: What the production Gateway path refuses each fault for. A control that broke for another reason, such as
#: sealing or setup, would still say "refused"; it would not say this.
_REFUSALS = {
    "commit": "no unique independently reviewed contract",
    "tree": "no unique independently reviewed contract",
    "entry": "is not the supported containerized quantum entry",
    "plan": "sealed invocation differs from the proposed row",
    "environment": "no unique independently reviewed contract",
    "wrapper": "is not the supported containerized quantum entry",
    "legacy": "carries 0 quantum completion announcements",
    "failed": "native pilot result refused",
    "cancelled": "native pilot result refused",
    "replaced": "native pilot result refused",
}


@pytest.mark.parametrize("fault", FAULTS)
def test_production_gateway_admits_only_the_reviewed_source_and_invocation(
        tmp_path, campaign, records_dir, monkeypatch, capsys, fault):
    receipt = _ready(tmp_path, campaign, records_dir)
    canned = dispatch.FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=canned)
    _mutate(canned, fault)
    _install_native_reader(monkeypatch, canned, fault)
    capsys.readouterr()
    output = tmp_path / "consumer-dry-run"
    result = dispatch.main(_argv(records_dir, output, receipt, extra=[*args, "--dry-run"]),
                           _gateway=_ProductionPilotGateway())
    captured = capsys.readouterr()
    assert not output.exists()
    assert result == (0 if fault is None else 3)
    if fault is not None:
        assert "refused" in captured.err
        assert _REFUSALS[fault] in captured.err
        assert not captured.out
    else:
        rows = json.loads(captured.out)["rows"]
        assert len(rows) == len(canned.results) == 3
        evidence = [row["pilot_admission"]["result"] for row in rows]
        assert {item["job_id"] for item in evidence} == {result["job_id"] for result in canned.results.values()}
        assert all(len(item["publication_sha256"]) == 64 for item in evidence)
        assert all(item["source"] == {"commit": "7" * 40, "tree": "6" * 40} for item in evidence)


def test_a_changed_counter_byte_is_not_the_published_counters(
        tmp_path, campaign, records_dir, monkeypatch, capsys):
    """The counters the operator supplies must be exactly the counters the native result published."""
    receipt = _ready(tmp_path, campaign, records_dir)
    canned = dispatch.FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=canned)
    for result in canned.results.values():
        result["counters_bytes"] = result["counters_bytes"] + b" "
    _install_native_reader(monkeypatch, canned, None)
    capsys.readouterr()
    assert dispatch.main(_argv(records_dir, tmp_path / "out", receipt, extra=[*args, "--dry-run"]),
                         _gateway=_ProductionPilotGateway()) == 3
    assert "differ from the counters the native result published" in capsys.readouterr().err


def test_production_gateway_reads_a_consumer_copy_not_the_producer_path(
        tmp_path, campaign, records_dir, monkeypatch, capsys):
    receipt = _ready(tmp_path, campaign, records_dir)
    canned = dispatch.FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=canned)
    for result in canned.results.values():
        completion = json.loads(result["payload"])
        completion["quantum_record"]["path"] = "/workspace/producer-only/quantum.json"
        assert not Path(completion["quantum_record"]["path"]).exists()
        result["payload"] = json.dumps(completion).encode() + b"\n"
    _install_native_reader(monkeypatch, canned, None)
    capsys.readouterr()
    output = tmp_path / "consumer-dry-run"
    assert dispatch.main(_argv(records_dir, output, receipt, extra=[*args, "--dry-run"]),
                         _gateway=_ProductionPilotGateway()) == 0
    assert not output.exists()
    assert len(json.loads(capsys.readouterr().out)["rows"]) == 3
