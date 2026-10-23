"""The container uses the complete claimed plan and refuses altered metadata."""
import gzip
import hashlib
import json

import pytest

import g3_residency as R


@pytest.mark.parametrize("corrupt", [False, True])
def test_launch_plan_preserves_phases_and_checks_its_own_digest(tmp_path, monkeypatch, corrupt):
    import prismabuild.client as sdk
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v2",
        "mount_prefix": "/mnt/shared",
        "produced_by": {"tool": "g3-plan-regression"},
        "annotations": {},
        "entries": [{"path": "/mnt/shared/source", "offset": 0, "bytes": 6, "sha256": None}],
        "entry_count": 1,
        "total_bytes": 6,
        "read_plan": {"phases": [
            {"name": "setup", "entry_indices": [0], "bytes": 6, "cumulative_bytes": 6},
            {"name": "teachers", "entry_indices": [0], "bytes": 6, "cumulative_bytes": 12},
        ], "read_bytes": 12},
    }
    raw = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    cas = tmp_path / "cas"
    blob = cas / "blobs" / digest[:2] / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]) if corrupt else raw)
    action_key = "a" * 64
    context = {"action_key": action_key, "queue_root": str(tmp_path / "queue")}
    row = {"action_key": action_key, "cas_root": str(cas),
           "residency": {"manifest_sha256": digest, "manifest_bytes": len(raw)}}
    monkeypatch.setattr(sdk, "injected_context", lambda **kwargs: {"ok": True, "ctx": context})
    monkeypatch.setattr(sdk, "read_claimed_record", lambda *args: row)
    if corrupt:
        from g3_pq_policy.staged_lease import ReadsetUnbound
        with pytest.raises(ReadsetUnbound, match="does not hash"):
            R.launch_read_plan()
    else:
        assert R.launch_read_plan() == manifest
