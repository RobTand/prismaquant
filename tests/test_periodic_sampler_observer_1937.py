"""CPU sampler lifecycle contract; never launch observer CLI/host probes."""
import ast
from pathlib import Path
import threading

import pytest

from prismaquant.io_spans import PeriodicSampler


def test_sampler_exposes_native_id_and_liveness_without_thread_internals():
    ready = threading.Event()
    native_ids = []

    def tick():
        native_ids.append(threading.get_native_id())
        ready.set()

    sampler = PeriodicSampler(tick, interval_s=30, name='observer-native-id')
    assert sampler.native_id is None
    assert not sampler.is_alive()
    sampler.start()
    try:
        assert ready.wait(5)
        assert sampler.is_alive()
        assert sampler.native_id == native_ids[0]
    finally:
        sampler.stop(timeout=5)
    assert not sampler.is_alive()
    assert len(native_ids) == 1, 'default stop must not add a new final tick'


@pytest.mark.parametrize('keep_running', [True, False])
def test_requested_final_tick_runs_once_on_stop_not_callback_completion(keep_running):
    ready = threading.Event()
    ticks = []

    def tick():
        ticks.append(threading.get_native_id())
        ready.set()
        return keep_running

    sampler = PeriodicSampler(tick, interval_s=30, name='observer-final-tick',
                              tick_last=True).start()
    try:
        assert ready.wait(5)
        if not keep_running:
            sampler.join(timeout=5)
            assert not sampler.is_alive()
    finally:
        sampler.stop(timeout=5)
    assert not sampler.is_alive()
    assert len(ticks) == (2 if keep_running else 1)
    assert all(native == sampler.native_id for native in ticks)
    sampler.stop(timeout=5)
    assert len(ticks) == (2 if keep_running else 1), 'repeated stop must not repeat final tick'


def test_observer_uses_shared_sampler_with_thirty_second_tick_and_final_tick():
    source = Path(__file__).resolve().parents[1] / 'tools/pq_row_profile_observer.py'
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'PeriodicSampler']
    assert len(calls) == 1, 'observer must use the existing sampler, not a bare thread'
    keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    assert keywords['interval_s'] == 30
    assert keywords['tick_last'] is True
    assert keywords['name'] == 'row-netdata'
