"""One telemetry owner reads what the replaced readers read (PQ #1299).

``prismaquant.io_spans`` owns the ``/proc`` readers and the one sampler
thread. The migrating PR ran each replaced reader next to the owner on the
texts below; the values those readers returned are written out here (PQ
#1330), so the owner, and each call site now built on it, is checked against
them: the same integers, and floats that are bit-identical (``float.hex``).
"""
from __future__ import annotations

import io
import subprocess
import time

import pytest

from prismaquant import io_spans
from prismaquant.io_spans import (
    GpuPowerSampler, MemAvailableFloor, PeriodicSampler, counter_delta,
    mem_available_bytes, read_meminfo, read_proc_io, read_proc_status)

MEMINFO = """\
MemTotal:       127535056 kB
MemFree:         3325224 kB
MemAvailable:   115340268 kB
Buffers:            7688 kB
Cached:         109316412 kB
SwapCached:         1216 kB
SwapTotal:      16777212 kB
SwapFree:       16613372 kB
HugePages_Total:       0
Hugepagesize:       2048 kB
"""

STATUS = """\
Name:\tpython3
State:\tS (sleeping)
Tgid:\t4242
Uid:\t1000\t1000\t1000\t1000
VmPeak:\t 8412344 kB
VmHWM:\t 2097153 kB
VmRSS:\t 1048577 kB
VmSwap:\t       0 kB
Threads:\t12
Cpus_allowed_list:\t0-19
"""

PROC_IO = """\
rchar: 123456789
wchar: 42
syscr: 1000
syscw: 7
read_bytes: 4096
write_bytes: 0
cancelled_write_bytes: 0
"""


# What the replaced readers returned on the texts above: kB fields in bytes,
# unitless counts as they are, and the old sites' own float expressions.
MEMINFO_BYTES = {
    "MemTotal": 127535056 * 1024, "MemFree": 3325224 * 1024,
    "MemAvailable": 115340268 * 1024, "Buffers": 7688 * 1024,
    "Cached": 109316412 * 1024, "SwapCached": 1216 * 1024,
    "SwapTotal": 16777212 * 1024, "SwapFree": 16613372 * 1024,
    "HugePages_Total": 0, "Hugepagesize": 2048 * 1024,
}
AVAILABLE = 115340268 * 1024
PROC_IO_COUNTERS = {"rchar": 123456789, "wchar": 42, "syscr": 1000, "syscw": 7,
                    "read_bytes": 4096, "write_bytes": 0, "cancelled_write_bytes": 0}


@pytest.fixture
def meminfo(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO)
    return path


@pytest.fixture
def owner_reads(monkeypatch, meminfo):
    """Point the owner's meminfo reader at the text, as every site sees it."""
    monkeypatch.setattr(io_spans, "read_meminfo",
                        lambda path=None: io_spans._read_kb_table(meminfo))
    return meminfo


# -- the owner's readers -----------------------------------------------------

def test_read_meminfo_returns_the_replaced_parses(meminfo):
    assert read_meminfo(meminfo) == MEMINFO_BYTES
    assert list(read_meminfo(meminfo)) == list(MEMINFO_BYTES)
    assert mem_available_bytes(meminfo) == AVAILABLE


def test_missing_mem_available_refuses_with_the_stage_b_text(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 1 kB\n")
    with pytest.raises(RuntimeError, match="/proc/meminfo has no MemAvailable"):
        mem_available_bytes(path)
    with pytest.raises(OSError):
        mem_available_bytes(tmp_path / "absent")


def test_read_proc_status_keeps_integer_fields_only(tmp_path):
    path = tmp_path / "status"
    path.write_text(STATUS)
    status = read_proc_status(path)
    assert "Name" not in status and "State" not in status
    assert "Cpus_allowed_list" not in status
    assert status["VmHWM"] == 2097153 * 1024 and status["Threads"] == 12


def test_read_proc_io_returns_the_tool_readers_counters(tmp_path):
    path = tmp_path / "io"
    path.write_text(PROC_IO)
    before = read_proc_io(path)
    assert before == PROC_IO_COUNTERS
    path.write_text(PROC_IO.replace("123456789", "123460000"))
    assert counter_delta(read_proc_io(path), before) == {
        **{key: 0 for key in PROC_IO_COUNTERS}, "rchar": 3211}


# -- the call sites now built on the owner ------------------------------------

def test_aura_free_gib_is_bit_identical(owner_reads):
    from prismaquant import aura_cost

    assert aura_cost._free_gib().hex() == (115340268 / (1024 ** 2)).hex()


def test_expert_swap_reading_is_bit_identical(owner_reads):
    new = io_spans.read_meminfo()
    assert (new["MemAvailable"] / 1024 ** 3).hex() == (115340268 / 1048576).hex()
    assert ((new["SwapTotal"] - new["SwapFree"]) / 1024 ** 3).hex() == (
        (16777212 - 16613372) / 1048576).hex()


def test_host_memory_wrappers_return_the_old_bytes(owner_reads):
    from prismaquant import autoscale, export_native_compressed, memory_management
    from prismaquant import source_prefetch

    assert memory_management._host_memory_info() == (AVAILABLE, 127535056 * 1024)
    assert source_prefetch._available_memory_bytes() == AVAILABLE
    assert export_native_compressed._host_mem_available_bytes() == AVAILABLE
    # psutil.virtual_memory().available is MemAvailable in bytes on Linux.
    assert autoscale._available_ram_bytes() == AVAILABLE


def test_proc_status_wrapper_returns_the_old_kilobytes(monkeypatch, tmp_path):
    from prismaquant import incremental_probe

    path = tmp_path / "status"
    path.write_text(STATUS)
    monkeypatch.setattr(io_spans, "read_proc_status",
                        lambda path_=None: io_spans._read_kb_table(path))
    assert incremental_probe._read_proc_status_kb("VmHWM", "VmRSS", "VmSwap", "VmAbsent") == {
        "VmHWM": 2097153, "VmRSS": 1048577, "VmSwap": 0, "VmAbsent": 0}


def test_wrapper_fallbacks_hold_when_meminfo_is_unreadable(monkeypatch):
    from prismaquant import aura_cost, export_native_compressed, memory_management

    def unreadable(path=None):
        raise OSError("no /proc")

    monkeypatch.setattr(io_spans, "read_meminfo", unreadable)
    assert memory_management._host_memory_info() is None
    assert export_native_compressed._host_mem_available_bytes() == 1 << 30
    monkeypatch.setattr(aura_cost.torch.cuda, "mem_get_info", lambda: (3 * 1024 ** 3, 0))
    assert aura_cost._free_gib() == 3.0


# -- the sampler thread ------------------------------------------------------

def test_periodic_sampler_ticks_until_false_or_stop():
    ticks = []

    def tick():
        ticks.append(time.monotonic())
        return len(ticks) < 3

    sampler = PeriodicSampler(tick, interval_s=0.001, name="test-ticks").start()
    sampler.join(timeout=5)
    assert len(ticks) == 3 and not sampler._thread.is_alive()

    seen = []
    with PeriodicSampler(lambda: seen.append(1), interval_s=0.001, name="test-stop"):
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.001)
    assert seen


def test_wait_first_sampler_skips_the_reading_at_start():
    seen = []
    sampler = PeriodicSampler(lambda: seen.append(1), interval_s=60, name="test-wait",
                              tick_first=False).start()
    sampler.stop(timeout=5)
    assert seen == [] and not sampler._thread.is_alive()


def test_stop_before_start_is_a_no_op():
    PeriodicSampler(lambda: None, interval_s=1, name="test-unstarted").stop()


def test_mem_available_floor_reads_at_entry_every_interval_and_exit(monkeypatch):
    readings = iter([900, 500, 700] + [800] * 10_000)
    monkeypatch.setattr(io_spans, "mem_available_bytes", lambda: next(readings))
    with MemAvailableFloor(0.0005, name="test-floor") as floor:
        deadline = time.monotonic() + 5
        while floor.samples < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
    assert floor.first["bytes"] == 900
    assert floor.minimum["bytes"] == 500
    assert floor.samples >= 4


class _FakeNvidiaSmi:
    def __init__(self, text):
        self.stdout = io.StringIO(text)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_gpu_power_sampler_block_is_unchanged(monkeypatch):
    launched = []

    def popen(argv, **kwargs):
        launched.append((argv, kwargs))
        return _FakeNvidiaSmi("12.5\nnot-a-number\n30.25, extra\n")

    monkeypatch.setattr(subprocess, "Popen", popen)
    sampler = GpuPowerSampler().start()
    sampler._sampler.join(timeout=5)
    assert launched[0][0] == ["nvidia-smi", "--query-gpu=power.draw",
                              "--format=csv,noheader,nounits", "-l", "1"]
    assert launched[0][1]["stderr"] is subprocess.DEVNULL
    assert sampler.stop() == {
        "sample_count": 2, "interval_s": 1.0, "gpu_joules": 42.75,
        "gpu_power_w_p50": 30.25, "gpu_power_w_p95": 12.5, "gpu_power_w_max": 30.25}


def test_gpu_power_sampler_records_a_failed_launch(monkeypatch):
    def popen(*_args, **_kwargs):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert GpuPowerSampler().start().stop() == {
        "sample_count": 0, "interval_s": 1.0, "gpu_joules": None,
        "gpu_power_w_p50": None, "gpu_power_w_p95": None, "gpu_power_w_max": None,
        "sampler_error": "sampler launch failed: no nvidia-smi"}
