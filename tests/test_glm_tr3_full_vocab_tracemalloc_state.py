"""Exercise the real single-buffer memory gate inside a caller's trace session.

The gate under repair lives in test_glm_tr3_full_vocab.py. Delegate to it rather
than duplicating its loader calls, fixture size, or one-/two-window bounds.
"""
import tracemalloc

import pytest

import test_glm_tr3_full_vocab as gate


@pytest.mark.parametrize("allocation_state", ["retained", "historical"])
def test_single_buffer_gate_isolates_existing_traces(tmp_path, monkeypatch, allocation_state):
    already_tracing = tracemalloc.is_tracing()
    traceback_limit = tracemalloc.get_traceback_limit() if already_tracing else 1
    try:
        if not already_tracing:
            gate.test_teacher_window_single_read_holds_one_window_on_the_host(tmp_path)
            assert not tracemalloc.is_tracing(), "memory gate left tracing enabled"
            tracemalloc.start()

        # More than the gate's ~4 MiB window. The retained case requires
        # subtracting current traced bytes; the released case requires ignoring
        # a historical peak. Keep the retained object alive across both loads.
        unrelated = bytearray(16 * 1024 * 1024)
        if allocation_state == "historical":
            del unrelated

        gate.test_teacher_window_single_read_holds_one_window_on_the_host(tmp_path)
        assert tracemalloc.is_tracing(), "memory gate stopped its caller's tracing"

        # A numerically correct but two-copy loader must still fail this gate.
        # The first invocation above always exercises the production loader.
        def two_copy_load(path, descriptor):
            return gate.np.load(gate.io.BytesIO(path.read_bytes()), allow_pickle=False)

        monkeypatch.setattr(gate.served, "load_teacher_window", two_copy_load)
        with pytest.raises(AssertionError):
            gate.test_teacher_window_single_read_holds_one_window_on_the_host(tmp_path)
        assert tracemalloc.is_tracing(), "failing memory gate stopped its caller's tracing"
    finally:
        if not already_tracing:
            tracemalloc.stop()
        elif not tracemalloc.is_tracing():
            # Best-effort cleanup after a broken gate: restarting cannot restore
            # the caller's lost traces, and must not count as preservation.
            tracemalloc.start(traceback_limit)
