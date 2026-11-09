"""A canonical native admission and launch stub for container adapter tests.

The adapter authenticates a running native Flux job before it creates a
container. Unit tests cannot be inside one, so they pass this exact admission
record (the shape ``native_admission`` returns) and stub only the owned-container
run. The real admission and the real container run are proven in admitted jobs.
"""
from __future__ import annotations

ADMISSION = {
    "schema": "prismaquant.campaign_container_admission.v1",
    "job_id": "f" + "A" * 10,
    "launch_sha256": "1" * 64,
    "source": {"checkout": "/worker/snapshot", "commit": "2" * 40, "tree": "3" * 40},
    "record_dir": "/mnt/shared/records/job",
    "affinity": [21, 22, 23, 30],
    "cores": 4, "memory_gib": 16, "gpus": 0, "walltime_s": 1800,
    "profile": {"profile": "pq.flux.fv1-model.perf-memory.v2",
                "config_sha256": "4" * 64, "model_sha256": "5" * 64},
}
CIDFILE = "/mnt/shared/records/job/container.cid"


def with_admission(**changes) -> dict:
    return {**ADMISSION, **changes}


def stub_launch(monkeypatch, runner, executed: list) -> None:
    """Replace admission and the owned run; record the exact argv ``main`` would run."""
    monkeypatch.setattr(runner, "native_admission", lambda environ: ADMISSION)

    def run(spec, command, *, cwd, image_id, content_sha256, admission, record_dir,
            environ, bounded=False, checkout_commit=None, image=None):
        executed.append(runner.podman_command(
            spec, command, cwd=cwd, image_id=image_id, admission=admission,
            cidfile=CIDFILE, content_sha256=content_sha256, environ=environ,
            bounded=bounded, checkout_commit=checkout_commit))
        return 0

    monkeypatch.setattr(runner, "run_owned_container", run)
