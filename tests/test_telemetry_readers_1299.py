"""One telemetry owner reads what the replaced readers read (PQ #1299).

``prismaquant.io_spans`` owns the ``/proc`` readers and the one sampler
thread. Each ``_old_*`` function below is a replaced reader, copied verbatim
except that its path is a parameter, and each test proves that the owner, or
the call site now built on it, returns the same value: the same integers, and
floats that are bit-identical (``float.hex``). A kB-to-GiB division and a
bytes-to-GiB division are the same rational number, so they round to the same
float.
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


# -- the replaced readers, verbatim but for the path ------------------------

def _old_measure_quant_cost_read_meminfo(path):
    info = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.partition(":")
            parts = v.strip().split()
            if parts:
                info[k] = int(parts[0]) * 1024  # kB → bytes
    return info


def _old_host_memory_info_proc(path):  # memory_management, /proc branch
    values = {}
    with open(path) as f:
        for line in f:
            key, rest = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    if "MemAvailable" in values and "MemTotal" in values:
        return values["MemAvailable"], values["MemTotal"]
    return None


def _old_stage_b_mem_available_bytes(path):
    with open(path) as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def _old_aura_free_gib(path):
    with open(path) as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    raise AssertionError("golden text has MemAvailable")


def _old_expert_avail_swap_gb(path):
    vals = {}
    with open(path) as fh:
        for line in fh:
            key = line.split(":", 1)[0]
            if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                vals[key] = int(line.split()[1])
    swap = (vals.get("SwapTotal", 0) - vals.get("SwapFree", 0)) / 1048576
    return vals.get("MemAvailable", 0) / 1048576, swap


def _old_read_proc_status_kb(path, *keys):  # incremental_probe
    out = {k: 0 for k in keys}
    try:
        with open(path) as f:
            for line in f:
                k, _, rest = line.partition(":")
                k = k.strip()
                if k in out:
                    out[k] = int(rest.strip().split()[0])
    except Exception:
        pass
    return out


def _old_tools_proc_io(path):  # profile_stage_b_head / staged_* benches
    values = {}
    with open(path) as handle:
        for line in handle:
            key, _, value = line.partition(":")
            values[key.strip()] = int(value)
    return values


def _old_io_delta(before, after):  # staged_exact_read_bench
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


# -- fixtures ----------------------------------------------------------------

@pytest.fixture(params=["golden", "live"])
def meminfo(request, tmp_path):
    """The golden text, and a snapshot of this box's own ``/proc/meminfo``."""
    path = tmp_path / "meminfo"
    if request.param == "golden":
        path.write_text(MEMINFO)
    else:
        try:
            path.write_text(open("/proc/meminfo").read())
        except OSError:
            pytest.skip("no /proc/meminfo on this host")
    return path


@pytest.fixture
def owner_reads(monkeypatch, meminfo):
    """Point the owner's meminfo reader at the snapshot, as every site sees it."""
    monkeypatch.setattr(io_spans, "read_meminfo",
                        lambda path=None: io_spans._read_kb_table(meminfo))
    return meminfo


def _same_float(a, b):
    return float(a).hex() == float(b).hex()


# -- the owner's readers -----------------------------------------------------

def test_read_meminfo_equals_every_replaced_parse(meminfo):
    new = read_meminfo(meminfo)
    old = _old_measure_quant_cost_read_meminfo(meminfo)
    kb_keys = [line.split(":")[0] for line in meminfo.read_text().splitlines()
               if line.rstrip().endswith(" kB")]
    assert kb_keys and {k: new[k] for k in kb_keys} == {k: old[k] for k in kb_keys}
    assert (new["MemAvailable"], new["MemTotal"]) == _old_host_memory_info_proc(meminfo)
    assert mem_available_bytes(meminfo) == _old_stage_b_mem_available_bytes(meminfo)


def test_unitless_fields_stay_counts(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO)
    assert read_meminfo(path)["HugePages_Total"] == 0
    assert read_meminfo(path)["Hugepagesize"] == 2048 * 1024


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
    keys = ("VmHWM", "VmRSS", "VmSwap", "VmAbsent")
    assert ({k: status.get(k, 0) // 1024 for k in keys}
            == _old_read_proc_status_kb(path, *keys))


def test_read_proc_io_equals_the_tool_readers(tmp_path):
    path = tmp_path / "io"
    path.write_text(PROC_IO)
    assert read_proc_io(path) == _old_tools_proc_io(path)
    before = read_proc_io(path)
    path.write_text(PROC_IO.replace("123456789", "123460000"))
    after = read_proc_io(path)
    assert counter_delta(after, before) == _old_io_delta(before, after)


# -- the call sites now built on the owner ------------------------------------

def test_aura_free_gib_is_bit_identical(owner_reads):
    from prismaquant import aura_cost

    assert _same_float(aura_cost._free_gib(), _old_aura_free_gib(owner_reads))


def test_expert_swap_reading_is_bit_identical(owner_reads):
    new = io_spans.read_meminfo()
    avail, swap = _old_expert_avail_swap_gb(owner_reads)
    assert _same_float(new.get("MemAvailable", 0) / 1024 ** 3, avail)
    assert _same_float((new.get("SwapTotal", 0) - new.get("SwapFree", 0)) / 1024 ** 3, swap)


def test_host_memory_wrappers_return_the_old_bytes(owner_reads):
    from prismaquant import autoscale, export_native_compressed, memory_management
    from prismaquant import source_prefetch

    available = _old_stage_b_mem_available_bytes(owner_reads)
    assert memory_management._host_memory_info() == _old_host_memory_info_proc(owner_reads)
    assert source_prefetch._available_memory_bytes() == available
    assert export_native_compressed._host_mem_available_bytes() == available
    # psutil.virtual_memory().available is MemAvailable in bytes on Linux.
    assert autoscale._available_ram_bytes() == available


def test_proc_status_wrapper_returns_the_old_kilobytes(monkeypatch, tmp_path):
    from prismaquant import incremental_probe

    path = tmp_path / "status"
    path.write_text(STATUS)
    monkeypatch.setattr(io_spans, "read_proc_status",
                        lambda path_=None: io_spans._read_kb_table(path))
    keys = ("VmHWM", "VmRSS", "VmSwap")
    assert incremental_probe._read_proc_status_kb(*keys) == _old_read_proc_status_kb(path, *keys)


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
