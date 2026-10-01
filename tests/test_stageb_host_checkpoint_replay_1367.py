"""Host replay contract, distinct from funded production profiling acceptance."""
from __future__ import annotations

import json

import pytest

from experiments.stageb_checkpoint_host_replay import replay
from prismaquant import aura_cost


def _source(tmp_path):
    root = tmp_path / "published"
    root.mkdir()
    units = []
    for index in range(5):
        name = f"unit-{index}"
        aura_cost._write_aura_unit_checkpoint(
            root, qname=name, identity_sha256="a" * 64,
            state={"row": {"s2": float(index), "probe": [1.0, 2.0]},
                   "identity": {"calibration": "unchanged"}})
        units.append({"qname": name,
                      "file": str(aura_cost._aura_unit_checkpoint_path(root, name).relative_to(root))})
    (root / "manifest.json").write_text(json.dumps({"units": units, "identity_sha256": "a" * 64}))
    return root


def test_host_replay_checks_bytes_owner_frontier_and_resume(tmp_path):
    source = _source(tmp_path)
    result = replay(source, tmp_path / "fresh", expected_units=5,
                    host_windows=2, budget_bytes=32 << 20, max_jobs=2)
    assert result["units"] == result["durable_resume_skipped"] == 5
    assert result["encoder_calls_shared_io"] == result["encoder_calls_consumer"] == 5
    assert result["baseline_candidate_digests_equal"]
    assert result["historical_file_digest_matches"] == 5
    assert result["source_unchanged"]
    assert result["host_partitions"] == [2, 3]
    assert result["publication"]["charged_bytes"] == 0
    assert not result["original_resolved_window_membership_replayed"]
    assert not result["gpu_overlap_or_speedup_claim"]
    with pytest.raises(FileExistsError):
        replay(source, tmp_path / "fresh", expected_units=5,
               host_windows=2, budget_bytes=32 << 20, max_jobs=2)


def test_host_replay_cannot_extend_published_namespace(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="published source"):
        replay(source, source / "after", expected_units=5,
               host_windows=2, budget_bytes=32 << 20, max_jobs=2)
    assert not (source / "after").exists()


def test_host_replay_refuses_incomplete_census(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="census"):
        replay(source, tmp_path / "fresh", expected_units=6,
               host_windows=2, budget_bytes=32 << 20, max_jobs=2)
    assert not (tmp_path / "fresh" / "synchronous").exists()


def test_source_allowance_precedes_any_decode(monkeypatch, tmp_path):
    import experiments.stageb_checkpoint_host_replay as host

    source = _source(tmp_path)
    path = aura_cost._aura_unit_checkpoint_path(source, "unit-0")
    # The established loader accepts trailing bytes; this remains a valid
    # envelope, but cannot be constructed inside the requested staging slot.
    with path.open("ab") as handle:
        handle.write(b"x" * (128 << 10))

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("source decoding preceded its construction allowance")

    monkeypatch.setattr(host.pickle, "loads", unexpected_decode)
    with pytest.raises(ValueError, match="source construction allowance"):
        replay(source, tmp_path / "fresh", expected_units=5,
               host_windows=2, budget_bytes=8 << 20, max_jobs=2)


@pytest.mark.parametrize("payload", [
    b"\x80\x04]r\xff\xff\xff\x7f.",  # Sparse memo can allocate enormous tables.
    b"\x80\x04cbuiltins\nlist\n)R.",  # Constructor execution is not builtin decoding.
])
def test_source_refuses_unbounded_pickle_before_state_decode(monkeypatch, tmp_path, payload):
    import hashlib
    import pickle
    import experiments.stageb_checkpoint_host_replay as host

    source = _source(tmp_path)
    path = aura_cost._aura_unit_checkpoint_path(source, "unit-0")
    loads = pickle.loads
    envelope = loads(path.read_bytes())
    envelope.update(payload=payload, payload_sha256=hashlib.sha256(payload).hexdigest())
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))

    def refuse_unsafe_decode(data, *args, **kwargs):
        if data == payload:
            raise AssertionError("unbounded pickle reached state decoder")
        return loads(data, *args, **kwargs)

    monkeypatch.setattr(host.pickle, "loads", refuse_unsafe_decode)
    with pytest.raises(ValueError, match="bounded builtin pickle"):
        replay(source, tmp_path / "fresh", expected_units=5,
               host_windows=2, budget_bytes=32 << 20, max_jobs=2)
