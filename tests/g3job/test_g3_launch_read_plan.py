"""The container uses the complete read plan bound to the staged manifest and refuses altered inputs."""
import gzip
import hashlib
import json

import pytest

import g3_residency as R
from staged_native_fixtures import bind, native_staged_state, publish_inputs  # noqa: F401

pytestmark = pytest.mark.usefixtures("native_staged_state")


def staged_plan(tmp_path, monkeypatch, *, drop_entry=False):
    from g3_pq_policy import residency_map as G3_MAPS
    files = []
    for name, raw in (("source", b"source"), ("teacher", b"teacher")):
        path = tmp_path / name
        path.write_bytes(raw)
        files.append(path)
    binding = publish_inputs(tmp_path, files)
    bind(monkeypatch, binding)
    G3_MAPS.reset_residency_resolver_for_tests()
    entries = [{"path": str(p), "offset": 0, "bytes": len(p.read_bytes()), "sha256": None} for p in files]
    if drop_entry:
        entries = entries[:1]
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v2", "mount_prefix": str(tmp_path),
        "produced_by": {"tool": "g3-plan-regression"}, "annotations": {}, "entries": entries,
        "entry_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
        "read_plan": {"phases": [
            {"name": "setup", "entry_indices": [0], "bytes": 6, "cumulative_bytes": 6},
            {"name": "teachers", "entry_indices": [len(entries) - 1], "bytes": 6, "cumulative_bytes": 12},
        ], "read_bytes": 12}}
    return manifest, G3_MAPS


@pytest.mark.parametrize("corrupt", [False, True])
def test_launch_plan_preserves_phases_and_checks_its_own_digest(tmp_path, monkeypatch, corrupt):
    manifest, maps = staged_plan(tmp_path, monkeypatch)
    raw = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "plan.json.gz"
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]) if corrupt else raw)
    try:
        if corrupt:
            from g3_pq_policy.staged_lease import ReadsetUnbound
            with pytest.raises(ReadsetUnbound, match="does not hash"):
                R.launch_read_plan(str(path), digest)
        else:
            assert R.launch_read_plan(str(path), digest) == manifest
    finally:
        maps.reset_residency_resolver_for_tests()


def test_a_plan_that_does_not_equal_the_staged_extents_is_refused(tmp_path, monkeypatch):
    manifest, maps = staged_plan(tmp_path, monkeypatch, drop_entry=True)
    raw = json.dumps(manifest).encode()
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    try:
        from g3_pq_policy.staged_lease import ReadsetUnbound
        with pytest.raises(ReadsetUnbound, match="does not equal the read plan extents"):
            R.launch_read_plan(str(path), hashlib.sha256(raw).hexdigest())
    finally:
        maps.reset_residency_resolver_for_tests()


def test_a_plan_with_the_wrong_schema_is_refused(tmp_path, monkeypatch):
    manifest, maps = staged_plan(tmp_path, monkeypatch)
    manifest["schema"] = "other"
    raw = json.dumps(manifest).encode()
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    try:
        from g3_pq_policy.staged_lease import ReadsetUnbound
        with pytest.raises(ReadsetUnbound, match="schema or phases are unsupported"):
            R.launch_read_plan(str(path), hashlib.sha256(raw).hexdigest())
    finally:
        maps.reset_residency_resolver_for_tests()
