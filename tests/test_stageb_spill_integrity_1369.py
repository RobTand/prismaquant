"""PQ #1369: reject changed spill operands before the IO engine delivers them."""
from __future__ import annotations

from contextlib import contextmanager
import threading

import pytest
import torch
from torch import nn

import prismaquant.joint_replay_spill as spill
from test_stageb_one_pass_spill import _spill_root


@contextmanager
def _captured(tmp_path, monkeypatch, *, threads=False, probes=2, arena_blocks=None):
    # The x86 PB worker's older kernel does not report STATX_DIOALIGN.
    # Model a 4 KiB reported grid; keep the real local file and O_DIRECT
    # syscalls. Grid discovery/refusal is covered by the scratch tests.
    import prismaquant.perturbed_x_cache as cache
    monkeypatch.setattr(cache, "_direct_io_block", lambda fd: 4096)
    modules = {"a": nn.Linear(8, 8, bias=False)}
    geometry = spill.spill_geometry(
        modules, [("a",)], pending={"a"}, batch_tokens=[4, 4],
        n_probes=probes, element_size=2, experts_per_token=None)
    with spill.StageBReplaySpill(
            root=_spill_root(tmp_path, needs_direct_io=False),
            max_bytes=1 << 20, geometry=geometry,
            window_names=[("a",)], n_probes=probes, dtype=torch.bfloat16,
            device="cpu", threads=threads) as session:
        if arena_blocks is not None:
            session.arena_bytes = arena_blocks * session._block
        for probe in range(probes):
            session._probe = probe
            session._records_seen = 0
            window = session._windows[0]
            window.entry_cursor = {}
            window.record_cursor = 0
            window.g_logical = {}
            session._start_arenas()
            for invocation in range(2):
                x = (torch.arange(32).reshape(4, 8) + invocation * 64).to(torch.bfloat16)
                g = torch.full_like(x, probe + invocation + 1)
                session._record("a", x, g)
                session._end_batch()
            session._end_capture()
            session._captured += 1
        yield session


def _payload_range(session, kind, probe):
    window = session._windows[0]
    if kind == "x":
        entry = window.entries["a"][0]
        offset = session._physical(window.x_runs["a"], window.x_starts["a"],
                                   entry.logical, entry.nbytes)
        return offset, entry.nbytes
    record = window.records[0]
    offset = session._physical(window.g_runs[("a", probe)],
                               window.g_starts[("a", probe)], record[3], record[4])
    return offset, record[4]


def _error_chain(error):
    result = []
    while error is not None:
        result.append(str(error))
        error = error.__cause__ or error.__context__
    return "\n".join(result)


@pytest.mark.parametrize("kind,probe", [("x", 0), ("g", 0), ("g", 1)])
@pytest.mark.parametrize("threads", [False, True])
def test_disk_bit_flip_is_rejected_and_restored_chunk_is_bitwise(
        tmp_path, monkeypatch, kind, probe, threads):
    with _captured(tmp_path, monkeypatch, threads=threads) as session:
        window = session._windows[0]
        item = window.plan[0]
        original, _ = session._read_chunk(window, probe, item)
        low, nbytes = _payload_range(session, kind, probe)
        block = session._block
        start = low - low % block
        sector = spill._aligned_buffer(block, block, False)
        view = memoryview(sector.numpy())
        session._scratch.read_into(start, [view])
        sector[low - start] ^= 1
        session._scratch.write(start, [view], call_bytes=block)

        stream = session._open_replay_stream()
        assert stream is not None
        with pytest.raises(Exception) as caught:
            # The stream enforces capture/probe order; consume the prefix
            # before asking for probe 1. Read-ahead may fail the stream early.
            for reading_probe in range(probe + 1):
                for chunk in range(len(window.plan)):
                    stream.take((0, reading_probe, chunk))
                    stream.release()
        text = _error_chain(caught.value)
        assert "checksum mismatch" in text, text
        assert "window 0" in text and f"probe {probe}" in text
        assert f"[{low}, {low + nbytes})" in text
        session.close_replay_stream()

        sector[low - start] ^= 1
        session._scratch.write(start, [view], call_bytes=block)
        restored, _ = session._read_chunk(window, probe, item)
        used = item[1][3]
        assert torch.equal(restored[:used], original[:used])


def test_same_length_wrong_offset_is_rejected(tmp_path, monkeypatch):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        window = session._windows[0]
        low, _ = _payload_range(session, "x", 0)
        other = window.entries["a"][1]
        wrong = session._physical(window.x_runs["a"], window.x_starts["a"],
                                  other.logical, other.nbytes)
        real_read = session._scratch.read_into

        def misdirected(offset, views):
            # Copy a different in-bounds slot into the expected destination.
            # The spill's length, alignment and layout gates cannot detect it.
            got = real_read(offset, views)
            if offset == low - low % session._block:
                real_read(wrong - wrong % session._block,
                          [views[0][:session._block]])
            return got

        monkeypatch.setattr(session._scratch, "read_into", misdirected)
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            session._read_chunk(window, 0, window.plan[0])


def test_verification_runs_in_engine_reader_not_consumer(tmp_path, monkeypatch):
    import xxhash

    with _captured(tmp_path, monkeypatch) as session:
        consumer = threading.get_ident()
        observed = []
        real_hash = xxhash.xxh3_64_intdigest

        def traced_hash(payload):
            observed.append(threading.get_ident())
            return real_hash(payload)

        monkeypatch.setattr(xxhash, "xxh3_64_intdigest", traced_hash)
        stream = session._open_replay_stream()
        assert stream is not None
        for probe in range(2):
            for chunk in range(len(session._windows[0].plan)):
                stream.take((0, probe, chunk))
                stream.release()
        assert observed, "spill chunks were delivered without verifying their checksums"
        assert all(thread != consumer for thread in observed)
        telemetry = session.telemetry
        assert telemetry["checksum_algorithm"] == "xxh3_64"
        # Two X slots written once; four G slots across two probes.
        assert telemetry["checksum_bytes_written"] == 6 * 64
        # Every probe reads both its X and G slots.
        assert telemetry["checksum_bytes_verified"] == 8 * 64
        assert telemetry["checksums_verified"] == 8
        assert telemetry["checksum_write_cpu_s"] >= 0
        assert telemetry["checksum_read_cpu_s"] >= 0


@pytest.mark.parametrize("threads", [False, True])
def test_verified_replay_preserves_live_operands(tmp_path, monkeypatch, threads):
    class Lease:
        modules = {"a": nn.Linear(8, 8, bias=False)}

        def __init__(self):
            self.observed = []

        def _observe_invocation(self, name, weight, x, gradient):
            assert name == "a" and weight is self.modules[name].weight
            self.observed.append((x.clone(), gradient.clone()))

    with _captured(tmp_path, monkeypatch, threads=threads) as session:
        for probe in range(2):
            lease = Lease()
            session.replay(0, probe, lease)
            assert len(lease.observed) == 2
            for invocation, (x, gradient) in enumerate(lease.observed):
                expected_x = (torch.arange(32).reshape(4, 8) + invocation * 64).to(
                    torch.bfloat16)
                expected_g = torch.full_like(expected_x, probe + invocation + 1)
                assert torch.equal(x, expected_x)
                assert torch.equal(gradient, expected_g)


def test_transient_read_ahead_corruption_is_terminal_without_reread(tmp_path, monkeypatch):
    from prismaquant import io_engine
    from test_io_engine import _quiet

    with _captured(tmp_path, monkeypatch) as session:
        low, _ = _payload_range(session, "g", 1)
        real_read = session._scratch.read_into
        victim_reads = []

        def transient(offset, views):
            got = real_read(offset, views)
            if offset <= low < offset + got:
                victim_reads.append(offset)
                if len(victim_reads) == 1:
                    # Only this undemanded read is wrong. A second disk read
                    # would succeed, so retrying would hide the integrity fault.
                    views[0][low - offset] ^= 1
            return got

        monkeypatch.setattr(session._scratch, "read_into", transient)
        stream = session._open_replay_stream()
        assert stream is not None
        _quiet(stream)
        assert len(victim_reads) == 1
        assert not stream._demanded
        stream.take((0, 0, 0))
        stream.release()
        with pytest.raises(io_engine.EntryError, match="checksum mismatch") as caught:
            stream.take((0, 1, 0))
        assert "window 0, probe 1" in str(caught.value)
        assert len(victim_reads) == 1, "an integrity failure must not reread the spill"
        assert stream.counters["rereads"] == 0


def test_checksum_writes_run_on_existing_writer_thread(tmp_path, monkeypatch):
    import xxhash

    consumer = threading.get_ident()
    observed = []
    real_hash = xxhash.xxh3_64_intdigest

    def traced_hash(payload):
        observed.append(threading.get_ident())
        return real_hash(payload)

    monkeypatch.setattr(xxhash, "xxh3_64_intdigest", traced_hash)
    with _captured(tmp_path, monkeypatch, threads=True):
        assert len(observed) == 6
        assert all(thread != consumer for thread in observed)


@pytest.mark.parametrize("threads", [False, True])
def test_checksums_survive_arena_rollover_between_input_and_gradient(
        tmp_path, monkeypatch, threads):
    # One tensor envelope fits an arena. Staging G must flush its X, and
    # staging the next X must flush the preceding G's checksum slot.
    with _captured(tmp_path, monkeypatch, threads=threads, arena_blocks=1) as session:
        stream = session._open_replay_stream()
        assert stream is not None
        for probe in range(2):
            stream.take((0, probe, 0))
            stream.release()
        assert session.telemetry["checksums_verified"] == 8
