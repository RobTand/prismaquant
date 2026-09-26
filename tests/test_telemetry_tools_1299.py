"""The bench tools read ``/proc`` and sample through ``io_spans`` (PQ #1299).

The migrating PR ran each parser a tool carried before it moved onto
``prismaquant.io_spans`` next to the tool on the texts below, and required
the same value, key order included. Those values are frozen in
``fixtures/telemetry_tools_1299.json`` (PQ #1330), so each tool is now
checked against its frozen row.
"""
from __future__ import annotations

import io
import statistics
import subprocess

import pytest

from prismaquant import io_spans
from prismaquant.io_spans import GpuPowerSampler, nfs_read_bytes
from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("telemetry_tools_1299")

MOUNTSTATS = """\
device rootfs mounted on / with fstype rootfs
device proc mounted on /proc with fstype proc
device tmpfs mounted on /ram/prewarm with fstype tmpfs
device 192.168.1.10:/tank/shared mounted on /mnt/shared with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmin=3
\tage:\t86034
\timpl_id:\tname='',domain='',date='0,0'
\tcaps:\tcaps=0x3ffbffff,wtmult=512,dtsize=1048576,bsize=0,namlen=255
\tnfsv4:\tbm0=0xfdffbfff,bm1=0xf9be3e,bm2=0x68800,acl=0x3,sessions,pnfs=not configured
\tsec:\tflavor=1,pseudoflavor=1
\tevents:\t1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
\tbytes:\t812345678 2000 0 0 790000000 1800 193000 6
\tRPC iostats version: 1.1  p/v: 100003/4 (nfs)
\txprt:\ttcp 0 1 2 3 4 5 6 7 8 9 10
\tper-op statistics
\t        NULL: 1 1 0 44 24 0 0 0 0
\t        READ: 193000 193000 0 26248000 790024000 12 40011 45123 0
\t       WRITE: 0 0 0 0 0 0 0 0 0
\t     GETATTR: 7 7 0 1000 1500 0 5 6 0
\t      LOOKUP: 3 3 0 400 600 1 2 3 1
\t      ACCESS: 2 2 0 300 200 0 1 1
\t        OPEN: 9 9 0 1800 2700 0 11 12 0
device 192.168.1.10:/stage mounted on /stage/prewarm with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2
\tbytes:\t7 8 9 10 11 12 13 14
\tper-op statistics
\t        READ: 4 4 0 480 400 0 2 3 0
\t       CLOSE: 5 5 0 600 500 0 1 2 0
"""


@pytest.fixture
def mountstats(tmp_path, monkeypatch):
    path = tmp_path / "mountstats"
    path.write_text(MOUNTSTATS)
    real = io_spans.read_mountstats
    monkeypatch.setattr(io_spans, "read_mountstats", lambda _path=None: real(path))
    return path


def test_nfs_read_bytes_is_the_old_mount_bytes(mountstats):
    assert nfs_read_bytes(mountstats) == {
        "/mnt/shared": (812345678, 790000000), "/stage/prewarm": (7, 11)}


def test_profile_head_mount_ops_is_unchanged(mountstats):
    from tools import profile_stage_b_head as head

    for mount in ("/mnt/shared", "/stage/prewarm", "/ram/prewarm", "/absent"):
        GOLDEN.value(head.mount_ops(mount))
    assert head.mount_ops()["READ"] == 193000


def test_staged_exact_mountstats_is_unchanged(mountstats):
    from tools import staged_exact_read_bench as bench

    new = bench.mountstats()
    GOLDEN.value(new)
    assert list(new) == ["/ram/prewarm", "/mnt/shared", "/stage/prewarm"]
    assert new["/ram/prewarm"] == {"bytes": None, "ops": {}}
    assert new["/mnt/shared"]["ops"]["ACCESS"] == [2, 2, 0, 300, 200, 0, 1, 1]


def test_live_reader_mountstats_sees_the_nfs_mounts(mountstats):
    """The old parser tested ``" on "`` against split tokens: always ``{}``."""
    from tools import live_reader_qualify as probe

    before = probe._mountstats()
    assert before == {
        "/mnt/shared": {"client_read": 812345678, "server_read": 790000000},
        "/stage/prewarm": {"client_read": 7, "server_read": 11}}
    after = {**before, "/mnt/shared": {"client_read": 812345678 + 4096,
                                       "server_read": 790000000}}
    assert probe.eval_pool_delta(before, before, "/mnt/shared") == (True, 0)
    assert probe.eval_pool_delta(before, after, "/mnt/shared") == (True, 4096)


def test_staged_exact_mountstats_is_empty_when_unreadable(tmp_path, monkeypatch):
    from tools import staged_exact_read_bench as bench

    real = io_spans.read_mountstats
    monkeypatch.setattr(io_spans, "read_mountstats",
                        lambda _path=None: real(tmp_path / "absent"))
    assert bench.mountstats() == {}


STATUS = """\
Name:\tpython3
VmPeak:\t 9133212 kB
VmSize:\t 9005456 kB
VmHWM:\t 1234567 kB
VmRSS:\t  987654 kB
RssAnon:\t  800000 kB
Threads:\t14
"""


@pytest.fixture
def status(tmp_path, monkeypatch):
    path = tmp_path / "status"
    path.write_text(STATUS)
    real = io_spans.read_proc_status
    monkeypatch.setattr(io_spans, "read_proc_status", lambda _path=None: real(path))
    return path


def test_checkpoint_probe_gib_is_bit_identical(status):
    from tools import checkpoint_parse_probe as probe

    for field in ("VmRSS:", "VmHWM:"):
        GOLDEN.value(probe._gib(field).hex())
    with pytest.raises(RuntimeError, match="no VmSwap: in /proc/self/status"):
        probe._gib("VmSwap:")


def test_chain_roll_memory_watch_block_is_unchanged(status, monkeypatch):
    from tools import chain_roll_bench as bench

    readings = iter([100 << 20, 60 << 20, 80 << 20, 90 << 20, 90 << 20, 90 << 20])
    monkeypatch.setattr(io_spans, "mem_available_bytes", lambda _path=None: next(readings))

    class _Cuda:
        @staticmethod
        def max_memory_reserved():
            return 7

        @staticmethod
        def max_memory_allocated():
            return 5

    class _Torch:
        cuda = _Cuda

    watch = bench._MemoryWatch(_Torch)
    watch._floor._sampler.stop()
    block = watch.stop()
    GOLDEN.value(block["process"])
    assert block["mem_available_baseline_bytes"] == 100 << 20
    assert block["mem_available_min_bytes"] == 60 << 20
    assert block["mem_available_drop_bytes"] == 40 << 20
    assert block["cuda_max_reserved_bytes"] == 7 and block["cuda_max_allocated_bytes"] == 5


MEMINFO = "MemTotal:  127535056 kB\nMemAvailable:  115340268 kB\nHugePages_Total:  0\n"
MEMORY_STAT = "anon 4096\nfile 8192\nfile_dirty 0\nfile_writeback 0\nactive_file 4096\n" \
              "inactive_file 4096\npgscan 11\npgsteal 10\nworkingset_refault_file 3\n"


@pytest.mark.parametrize("stat_text", [MEMORY_STAT, MEMORY_STAT + "torn\n"],
                         ids=["whole", "malformed"])
def test_unit_journal_memory_sample_is_unchanged(tmp_path, monkeypatch, stat_text):
    from tools import unit_journal_bench as bench

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)
    real = io_spans.read_meminfo
    monkeypatch.setattr(io_spans, "read_meminfo", lambda _path=None: real(meminfo))
    group = tmp_path / "group"
    group.mkdir()
    (group / "memory.current").write_text("12288\n")
    (group / "memory.max").write_text("max\n")
    (group / "memory.stat").write_text(stat_text)
    (group / "memory.pressure").write_text(
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=17\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=5\n")
    for where in (group, None):
        new = bench._memory_sample(where)
        assert new.pop("unix") > 0
        GOLDEN.value(new)


def test_memcg_peaks_reads_the_same_fields(tmp_path):
    from tools.staged_exact_read_bench import MemcgPeaks

    stat = tmp_path / "memory.stat"
    stat.write_text(MEMORY_STAT)
    peaks = MemcgPeaks()
    peaks.path = stat
    old = {}
    for line in stat.read_text().splitlines():
        key, _, value = line.partition(" ")
        if key in MemcgPeaks.FIELDS:
            old[key] = int(value)
    assert peaks._read() == old
    with peaks:
        pass
    assert peaks.first == old and peaks._sampler is not None


class _FakeNvidiaSmi:
    def __init__(self, text):
        self.stdout = io.StringIO(text)

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize(("interval", "loop"), [
    (1.0, ["-l", "1"]), (2, ["-l", "2"]), (0.5, ["-lms", "500"]), (0.1, ["-lms", "100"])])
def test_gpu_power_sampler_loop_argument(monkeypatch, interval, loop):
    launched = []

    def popen(argv, **kwargs):
        launched.append(argv)
        return _FakeNvidiaSmi("")

    monkeypatch.setattr(subprocess, "Popen", popen)
    GpuPowerSampler(interval).start().stop()
    assert launched == [["nvidia-smi", "--query-gpu=power.draw",
                         "--format=csv,noheader,nounits", *loop]]


def test_power_windows_are_unchanged(monkeypatch):
    from tools import pwc_window_load_bench as pwc
    from tools import render_window_bench as rwb

    monkeypatch.setattr(subprocess, "Popen",
                        lambda *_a, **_k: _FakeNvidiaSmi("30.25\n[N/A]\n12.5\n41.0\n"))
    sampler = GpuPowerSampler(0.1).start()
    sampler._sampler.join(timeout=5)
    sampler.stop()
    assert sampler.samples == [30.25, 12.5, 41.0] and len(sampler.times) == 3
    sampler.times = [100.0, 100.1, 100.2]
    pairs = list(zip(sampler.times, sampler.samples))
    for start, end in ((100.0, 100.2), (100.05, 100.2), (100.1, 100.1), (101.0, 102.0)):
        GOLDEN.value(rwb._power_summary(sampler.watts_between(start, end)))
        values = [w for when, w in pairs if start <= when <= end]
        assert pwc._power_between(sampler, start, end) == (
            statistics.fmean(values) if values else None, len(values))


@pytest.mark.parametrize("name", [
    "chain_roll_bench", "checkpoint_parse_probe", "nvfp4_served_qdq_bench", "profile_stage_b_head", "pwc_window_load_bench",
    "render_window_bench", "stage_fed_demonstration", "staged_exact_read_bench",
    "staged_read_stream_ab", "unit_journal_bench"])
def test_migrated_tool_imports(name):
    import importlib

    importlib.import_module(f"tools.{name}")
