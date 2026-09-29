"""Overhead of the exposed-wait instrumentation on a ReadStream fixture (#1292).

Runs unchanged on a checkout without the change (no wait_sink attribute: the
"before" arm) and with it (the "after" arm attaches the real ledger sink).
Reports wall per take (median of REPEATS) plus a cProfile of the drain.
"""
import cProfile, hashlib, io, os, pstats, statistics, sys, tempfile, time
sys.path.insert(0, os.getcwd())
os.environ.pop("PRISMABUILD_RESIDENCY_MAP", None)
from prismaquant import io_engine

GROUPS, PER, SIZE, REPEATS = 96, 8, 65536, 9

def decode(raw, receipt, staged):
    return bytes(raw), {"staged": staged}

def main():
    base = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or os.getcwd(), prefix="ewbench-")
    entries = []
    for g in range(GROUPS):
        for i in range(PER):
            data = os.urandom(SIZE)
            p = os.path.join(base, f"g{g}e{i}.bin")
            open(p, "wb").write(data)
            entries.append(io_engine.ReadEntry(
                key=f"g{g}e{i}", path=p, size=SIZE, limit=SIZE, held_bytes=SIZE,
                expected_sha256=hashlib.sha256(data).hexdigest(), decoder=decode, group=g))
    have_sink = hasattr(io_engine.ReadStream, "wait_sink") or "wait_sink" in io_engine.ReadStream.__init__.__code__.co_names or True
    ledger = None
    try:
        from prismaquant.io_spans import ExposedWaitLedger
        ledger = ExposedWaitLedger()
    except ImportError:
        pass
    arm = "after" if ledger is not None else "before"
    def drain():
        budget = io_engine.FixedBudget(buffer_bytes=SIZE * PER * 4, headroom=SIZE * PER * 4)
        t0 = time.perf_counter()
        with io_engine.read_stream(entries, budget=budget) as s:
            if ledger is not None:
                s.wait_sink = ledger.sink
            for g in range(GROUPS):
                s.take(g)
        return time.perf_counter() - t0
    drain()  # warm the page cache and pool
    walls = [drain() for _ in range(REPEATS)]
    med = statistics.median(walls)
    print(f"ARM {arm} groups={GROUPS} per_group={PER} bytes={GROUPS*PER*SIZE}")
    print(f"ARM {arm} wall_s median={med:.6f} min={min(walls):.6f} max={max(walls):.6f}"
          f" per_take_us={med/GROUPS*1e6:.1f}")
    pr = cProfile.Profile(); pr.enable(); drain(); pr.disable()
    out = io.StringIO()
    st = pstats.Stats(pr, stream=out).sort_stats("cumulative")
    st.print_stats(r"io_engine|io_spans", 12)
    print(out.getvalue()[:3500])
    if ledger is not None:
        print("ledger intervals", len(ledger.snapshot().get("intervals", [])) if isinstance(ledger.snapshot(), dict) else "?")
    import shutil; shutil.rmtree(base, ignore_errors=True)

main()
