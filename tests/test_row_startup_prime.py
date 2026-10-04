"""Verified, reserved reads before admit stay inside the existing window (#1743)."""
from collections import Counter
import threading
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc
from prismaquant.tessera_row_stream import RowStream
from test_tessera_row_stream import (COLUMNS, LOAD_POLICY, OUTPUT_FEATURES,
                                     UNITS, stream_fixture)


def _stream(monkeypatch, tmp_path, *, check=None):
    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    manifest = state["manifest"]
    stream = RowStream(capture_path=manifest, expected_sha256=cc.sha256(manifest),
        expected_identity=state["canonical"], census=state["census"], names=UNITS,
        policy=LOAD_POLICY,
        weights={name: torch.ones(OUTPUT_FEATURES, COLUMNS, dtype=torch.bfloat16)
                 for name in UNITS},
        hessian_identity=state["calibration"],
        bind=lambda name, **kwargs: (None, {"weight": name}),
        threads=1, batch_size=1, device="cpu", memo_capacity=1, resource_check=check)
    return stream, state


def test_prime_starts_a_reserved_read_without_collecting_or_rereading(monkeypatch, tmp_path):
    checks, reads = [], Counter()
    started = threading.Event()
    original = cc._verified_capture_entry

    def verified(path, name, **kwargs):
        assert checks and checks[0][1] > 0
        reads[name] += 1
        started.set()
        return original(path, name, **kwargs)

    monkeypatch.setattr(cc, "_verified_capture_entry", verified)
    stream, state = _stream(monkeypatch, tmp_path,
                            check=lambda label, **kw: checks.append((label, kw.get("reserve_bytes", 0))))
    try:
        stream.prime([UNITS[0]])
        assert started.wait(5), "the first verified read still waits for admit"
        assert not stream._first and not stream._live
        stream.plan([[name] for name in UNITS])
        for i in range(len(UNITS)):
            stream.admit(i)
        stream.finish()
        assert reads == Counter(dict.fromkeys(UNITS, 1))
        assert stream.stats["rereads"] == 0
        peak = stream.stats["peak_resident_units"]
        assert peak is not None and peak <= 2
        census = state["census"]
        assert isinstance(census, dict)
        assert stream.observed_counts() == census["counts"]
        execution = {}
        cc.prefetch_capture(state["manifest"], expected_identity=state["canonical"],
            census=state["census"], names=UNITS, device="cpu",
            verified_load_policy=LOAD_POLICY, load_execution=execution)
        assert stream.load_execution() == execution
    finally:
        stream.close()


def test_prime_checks_the_memory_budget_before_any_submission(monkeypatch, tmp_path):
    reads = []
    monkeypatch.setattr(cc, "_verified_capture_entry", lambda *a, **k: reads.append(a))

    def refuse(label, **kwargs):
        assert kwargs["reserve_bytes"] > 0
        raise RuntimeError("test row memory budget refused")

    stream, _ = _stream(monkeypatch, tmp_path, check=refuse)
    try:
        with pytest.raises(RuntimeError, match="test row memory budget refused"):
            stream.prime([UNITS[0]])
        assert not reads and not stream._inflight
    finally:
        stream.close()


def test_prime_is_one_batch_and_must_match_the_first_plan(monkeypatch, tmp_path):
    stream, _ = _stream(monkeypatch, tmp_path)
    try:
        with pytest.raises(ValueError, match="one batch"):
            stream.prime(UNITS[:2])
        with pytest.raises(ValueError, match="unknown"):
            stream.prime(["not-in-this-capture"])
        stream.prime([UNITS[0]])
        with pytest.raises(RuntimeError, match="first planned batch"):
            stream.plan([[UNITS[1]], [UNITS[0]]])
    finally:
        stream.close()


def test_a_new_head_starts_first_reads_before_run_identity(monkeypatch, tmp_path):
    campaign, argv, _state = stream_fixture(monkeypatch, tmp_path)
    started = threading.Event()
    original_load = cc._verified_capture_entry
    original_identity = campaign._campaign_checkpoint_identity

    def verified(*args, **kwargs):
        started.set()
        return original_load(*args, **kwargs)

    def identity(*args, **kwargs):
        assert started.wait(5), "a new row still starts first reads only at admit"
        return original_identity(*args, **kwargs)

    monkeypatch.setattr(cc, "_verified_capture_entry", verified)
    monkeypatch.setattr(campaign, "_campaign_checkpoint_identity", identity)
    assert campaign.main(argv) == 0


def test_first_admit_reserves_the_primed_promise_and_the_ahead_window(monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    original = cc._verified_capture_entry
    checks = []

    def verified(path, name, **kwargs):
        if name == UNITS[0]:
            started.set()
            assert release.wait(5)
        return original(path, name, **kwargs)

    monkeypatch.setattr(cc, "_verified_capture_entry", verified)
    stream, _ = _stream(monkeypatch, tmp_path)

    def check(label, **kwargs):
        if label == "before_row_stream_admit:0":
            checks.append(kwargs["reserve_bytes"])
            release.set()

    stream._resource_check = check
    try:
        stream.prime([UNITS[0]])
        assert started.wait(5)
        stream.plan([[name] for name in UNITS])
        expected = sum(stream.reader_reserve_bytes(name) for name in UNITS[:2])
        stream.admit(0)
        assert checks == [expected]
    finally:
        release.set()
        stream.close()


@pytest.mark.parametrize("existing", ["checkpoint", "parts", "stream-journal"])
def test_existing_journal_keeps_the_before_read_resume_gate(tmp_path, existing):
    from prismaquant import tessera_campaign as campaign

    checkpoint = tmp_path / "row.pkl"
    paths = {"checkpoint": checkpoint,
             "parts": checkpoint.with_name(checkpoint.name + ".parts"),
             "stream-journal": checkpoint.with_name(checkpoint.name + campaign.STREAM_JOURNAL_SUFFIX)}
    paths[existing].mkdir()
    calls = []
    getattr(campaign, "_prime_first_anchor_batch")(SimpleNamespace(prime=calls.append),
        args=SimpleNamespace(seed_checkpoint=[], finalize_checkpoint=False),
        checkpoint=checkpoint, targets=[], menus={}, weights={}, profile=None,
        expert_members=None, encode_structure={}, projected_units={}, audit_units=set(),
        route_cache={})
    assert not calls


def test_ambiguous_journal_access_is_not_treated_as_a_new_row(monkeypatch, tmp_path):
    from prismaquant import tessera_campaign as campaign

    checkpoint = tmp_path / "row.pkl"
    stat = type(checkpoint).stat

    def unreadable(path, **kwargs):
        if path == checkpoint:
            raise PermissionError("prime path unreadable")
        return stat(path, **kwargs)

    monkeypatch.setattr(type(checkpoint), "stat", unreadable)
    calls = []
    with pytest.raises(PermissionError, match="prime path unreadable"):
        getattr(campaign, "_prime_first_anchor_batch")(SimpleNamespace(prime=calls.append),
            args=SimpleNamespace(seed_checkpoint=[], finalize_checkpoint=False),
            checkpoint=checkpoint, targets=[], menus={}, weights={}, profile=None,
            expert_members=None, encode_structure={}, projected_units={}, audit_units=set(),
            route_cache={})
    assert not calls


def test_partitioned_head_keeps_full_group_planning_before_any_early_read(monkeypatch, tmp_path):
    from prismaquant import tessera_campaign as campaign

    checkpoint = tmp_path / "partition.pkl"

    def premature_stat(*args, **kwargs):
        raise AssertionError("partition subset must not plan or read an early batch")

    monkeypatch.setattr(type(checkpoint), "stat", premature_stat)
    calls = []
    getattr(campaign, "_prime_first_anchor_batch")(SimpleNamespace(prime=calls.append),
        args=SimpleNamespace(), checkpoint=checkpoint, targets=[], menus={}, weights={},
        profile=None, expert_members=None, encode_structure={}, projected_units={},
        audit_units=set(), route_cache={}, partitioned=True)
    assert not calls
