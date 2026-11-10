"""g3_submit: a native submission over a staged binding that fails closed at the real profile limit."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import g3_submit

BINDING = {"schema": "prismaquant.staged_launch.v1", "path": "/mnt/shared/m.json", "sha256": "a" * 64,
           "read_contract_path": "/mnt/shared/c.json", "read_contract_sha256": "b" * 64, "allowed_tiers": "ssd"}
LAUNCH = ["--", "--arm", "a8_w", "--output-root", "/mnt/shared/out"]


def arguments(tmp_path, *extra):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(BINDING))
    return ["--record-dir", str(tmp_path / "record"), "--staged-launch", str(binding),
            "--read-plan", "/mnt/shared/plan.json", "--read-plan-sha256", "c" * 64, *extra, *LAUNCH]


def test_a_gpu_request_is_refused_at_the_native_profile_with_its_real_text(tmp_path, capsys, monkeypatch):
    from prismaquant import flux_jobs
    monkeypatch.setattr(flux_jobs, "submit_job", lambda *a, **k: pytest.fail("a refused request submitted"))
    assert g3_submit.main(arguments(tmp_path, "--cpus", "4", "--mem-gb", "16", "--gpu-memory-gb", "80")) == 2
    assert capsys.readouterr().err.strip() == (
        "G3 submission refused at the native profile: gpus must be an integer in [0, 0] "
        "for pq.flux.stock-slot.v3")


def test_an_oversized_memory_request_is_refused_with_its_real_text(tmp_path, capsys, monkeypatch):
    from prismaquant import flux_jobs
    monkeypatch.setattr(flux_jobs, "submit_job", lambda *a, **k: pytest.fail("a refused request submitted"))
    assert g3_submit.main(arguments(tmp_path, "--cpus", "4", "--mem-gb", "104")) == 2
    assert capsys.readouterr().err.strip() == (
        "G3 submission refused at the native profile: memory_gib must be an integer in [1, 16] "
        "for pq.flux.stock-slot.v3")


def test_a_cpu_request_submits_once_over_the_staged_binding(tmp_path, capsys, monkeypatch):
    from prismaquant import flux_jobs
    seen = []

    def submit(argv, **kwargs):
        seen.append((argv, kwargs))
        return flux_jobs.FluxSubmission("fSAVED1", kwargs["record_dir"], "d" * 64)

    monkeypatch.setattr(flux_jobs, "submit_job", submit)
    assert g3_submit.main(arguments(tmp_path, "--cpus", "2", "--mem-gb", "8", "--walltime-s", "600")) == 0
    assert capsys.readouterr().out.strip() == "fSAVED1"
    (argv, kwargs), = seen
    assert argv[:3] == ["/usr/bin/python3", "-s", "tools/g3job/g3_progress_launch.py"]
    assert argv[3:] == ["--arm", "a8_w", "--output-root", "/mnt/shared/out",
                        "--read-plan", "/mnt/shared/plan.json", "--read-plan-sha256", "c" * 64]
    assert kwargs["staged_input"] == BINDING and kwargs["record_dir"] == tmp_path / "record"
    assert kwargs["resources"] == flux_jobs.FluxResources(cores=2, memory_gib=8, gpus=0, walltime_s=600)
    assert kwargs["checkout"] == Path(g3_submit.ROOT)
    assert "pbrun" not in json.dumps(argv)


def test_an_unknown_submission_is_reported_and_never_retried(tmp_path, capsys, monkeypatch):
    from prismaquant import flux_jobs
    calls = []

    def submit(*args, **kwargs):
        calls.append(1)
        raise flux_jobs.SubmissionUnknown("submission unknown in record; do not resubmit")

    monkeypatch.setattr(flux_jobs, "submit_job", submit)
    assert g3_submit.main(arguments(tmp_path)) == 3
    assert calls == [1] and "do not resubmit" in capsys.readouterr().err


def test_resume_reconnects_to_the_saved_id_and_never_submits(tmp_path, capsys, monkeypatch):
    from prismaquant import flux_jobs
    monkeypatch.setattr(flux_jobs, "submit_job", lambda *a, **k: pytest.fail("resume submitted"))
    monkeypatch.setattr(flux_jobs, "load_submission", lambda record: SimpleNamespace(job_id="fSAVED1"))
    assert g3_submit.main(["--record-dir", str(tmp_path / "record"), "--resume"]) == 0
    assert capsys.readouterr().out.strip() == "fSAVED1"


def test_a_submission_without_its_staged_binding_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        g3_submit.main(["--record-dir", str(tmp_path / "record"), *LAUNCH])
