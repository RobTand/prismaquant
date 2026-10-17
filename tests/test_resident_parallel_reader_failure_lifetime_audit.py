"""Actual resident failure must join its launched intra-layer readers.

A wrapper at the real
safetensors get_tensor boundary injects one OSError and holds a sibling
reader before its real payload access. Both are released/joined by fixture
teardown. This is ownership evidence, not a workload or latency claim.
"""
from __future__ import annotations

import threading

import pytest

from prismaquant import layer_streaming as decoder
from prismaquant import streaming_model as owner
from tests.test_resident_source_decode_audit import (
    _check_values, _cpu_guard, _load, _source,
)
import torch


@pytest.mark.parametrize("failure", [None, "read", "submit"],
                         ids=["successful-read", "failed-first-reader", "failed-second-submit"])
def test_resident_joins_parallel_readers_before_return(tmp_path, monkeypatch, failure):
    root, expected = _source(tmp_path, native=False)
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "2")
    monkeypatch.setenv("PRISMAQUANT_RELEASE_SOURCE_PAGES", "0")
    guard = _cpu_guard(tmp_path, monkeypatch)
    keys, contexts = {}, []
    blocked = threading.Event()
    first_attempted = threading.Event()
    release = threading.Event()
    blocked_exited = threading.Event()
    done = threading.Event()
    outcome = {}
    build = owner._build_streaming_context
    safe_open = decoder.safe_open

    def observe_context(*args, **kwargs):
        context = build(*args, **kwargs)
        selected = decoder.select_source_tensors(
            context.weight_shard, context.weight_ckpt,
            (f"{context.layers_prefix}0.",))
        assert len(selected) == 1
        pairs = next(iter(selected.values()))
        assert len(pairs) >= decoder._LAYER_READ_MIN_TENSORS
        chunks = decoder._split_pairs(pairs, 2)
        assert len(chunks) == 2
        keys.update(first=chunks[0][0][1], blocked=chunks[1][0][1])
        contexts.append(context)
        return context

    class Reader:
        def __init__(self, context):
            self.context = context
            self.held = False

        def __enter__(self):
            self.file = self.context.__enter__()
            return self

        def __getattr__(self, name):
            return getattr(self.file, name)

        def get_tensor(self, name):
            if failure == "submit" and name == keys.get("first"):
                self.held = True
                blocked.set()
                first_attempted.set()
                assert release.wait(10), "fixture did not release its held reader"
            elif name == keys.get("blocked"):
                self.held = True
                blocked.set()
                assert release.wait(10), "fixture did not release its held reader"
            elif name == keys.get("first"):
                assert blocked.wait(10), "fixture sibling reader did not enter"
                first_attempted.set()
                if failure == "read":
                    raise OSError("injected resident reader failure")
            return self.file.get_tensor(name)

        def __exit__(self, *args):
            try:
                return self.context.__exit__(*args)
            finally:
                if self.held:
                    blocked_exited.set()

    def load():
        try:
            outcome["model"] = _load(root, tmp_path, guard)
        except BaseException as error:
            outcome["error"] = error
        finally:
            done.set()

    monkeypatch.setattr(owner, "_build_streaming_context", observe_context)
    monkeypatch.setattr(decoder, "safe_open", lambda *args, **kwargs: Reader(safe_open(*args, **kwargs)))
    if failure == "submit":
        pool = decoder._layer_read_pool(2)
        submit = pool.submit
        submissions = []

        def fail_second_submit(*args, **kwargs):
            submissions.append(True)
            if len(submissions) == 2:
                assert blocked.wait(10), "first submitted reader did not enter"
                raise OSError("injected resident submission failure")
            return submit(*args, **kwargs)

        monkeypatch.setattr(pool, "submit", fail_second_submit)
    consumer = threading.Thread(target=load, name="resident-lifetime-audit")
    consumer.start()
    returned_while_blocked = False
    try:
        assert blocked.wait(10), "real resident loader never launched the held reader"
        assert first_attempted.wait(10), "real resident loader never attempted the first reader"
        returned_while_blocked = done.wait(2)
    finally:
        release.set()
        consumer.join(15)
        assert not consumer.is_alive(), "fixture resident caller did not finish"
        assert blocked_exited.wait(10), "fixture reader did not close"

    assert contexts
    assert contexts[0].prefetch_pool._shutdown
    assert all(not thread.is_alive() for thread in contexts[0].prefetch_pool._threads)
    if failure is not None:
        assert isinstance(outcome.get("error"), OSError), repr(outcome)
        message = ("injected resident reader failure" if failure == "read"
                   else "injected resident submission failure")
        assert message in str(outcome["error"])
        assert all(value.is_meta for value in contexts[0].model.parameters())
    else:
        assert "error" not in outcome, repr(outcome)
        _check_values(outcome["model"], expected, torch.bfloat16, False)
    assert not returned_while_blocked, (
        "Resident caller returned while a sibling of its failed intra-layer "
        "read was still held before payload access")
