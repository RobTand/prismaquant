"""Real retained-window publication barriers; CPU correctness, not timing."""
from __future__ import annotations

import pickle
import shutil
import threading
from pathlib import Path

import pytest

import prismaquant.aura_cost as aura
import prismaquant.joint_cost_quantum as quantum
import prismaquant.joint_statistics_replay as replay_module
from test_stageb_one_pass_spill import campaign, _quantum


BUDGET = 64 << 20


def _journal(campaign):
    path = Path(campaign.records[1]["output_space"]["checkpoint_dir"])
    assert path.is_relative_to(campaign.root), "only remove this fixture's own journal"
    return path


def _clear(campaign):
    path = _journal(campaign)
    if path.exists():
        shutil.rmtree(path)


def _envelopes(campaign):
    return {path.name: path.read_bytes() for path in (_journal(campaign) / "units").glob("*.pkl")}


def _enable(monkeypatch, budget=BUDGET, *, max_jobs=None):
    original = quantum.run_layer_quantum_core

    def launch(*args, **kwargs):
        kwargs["execution"] = {
            **kwargs["execution"], "checkpoint_publication_budget_bytes": budget}
        if max_jobs is not None:
            kwargs["execution"]["checkpoint_publication_max_jobs"] = max_jobs
        return original(*args, **kwargs)

    monkeypatch.setattr(quantum, "run_layer_quantum_core", launch)


def test_next_real_window_replays_before_held_serialization(campaign, monkeypatch):
    assert campaign.device.type == "cpu", "submit this CPU fixture on x86"
    assert len(campaign.preflight[1]) > 1
    _clear(campaign)
    _enable(monkeypatch)
    writer_started = threading.Event()
    release_writer = threading.Event()
    second_replay = threading.Event()
    seen = {"window": None, "durable": 0, "advanced": False, "worker": None}
    consumer = threading.get_ident()
    encode = aura._encode_aura_unit_checkpoint
    observe = replay_module.observe_and_project_retained_windows
    commit = quantum.QuantumProgress.commit

    def held_encode(**kwargs):
        if not writer_started.is_set():
            seen["worker"] = threading.get_ident()
            writer_started.set()
            assert release_writer.wait(30), "test encoder was not released"
        return encode(**kwargs)

    def progress_commit(self):
        seen["durable"] = self.units()
        return commit(self)

    def instrument(*args, **kwargs):
        before, backward = kwargs["before_window"], kwargs["backward"]

        def before_window(index, names):
            seen["window"] = index
            return before(index, names)

        def replay(**call):
            result = backward(**call)
            if seen["window"] == 1:
                second_replay.set()
            return result

        kwargs.update(before_window=before_window, backward=replay)
        return observe(*args, **kwargs)

    def control():
        try:
            if writer_started.wait(20):
                seen["advanced"] = second_replay.wait(10)
                seen["before_ack"] = seen["durable"]
        finally:
            release_writer.set()

    monkeypatch.setattr(aura, "_encode_aura_unit_checkpoint", held_encode)
    monkeypatch.setattr(quantum.QuantumProgress, "commit", progress_commit)
    monkeypatch.setattr(replay_module, "observe_and_project_retained_windows", instrument)
    controller = threading.Thread(target=control, name="checkpoint-test-control")
    controller.start()
    try:
        payload, state = _quantum(campaign, monkeypatch, layer=1)
    finally:
        release_writer.set()
        controller.join(30)
    assert not controller.is_alive()
    assert payload is not None, getattr(state, "error", None)
    assert seen["advanced"], "next retained replay waited for held unit serialization"
    assert seen["before_ack"] == 0, "unacknowledged units became durable progress"
    assert seen["worker"] != consumer, "durable publication ran on the consumer"
    assert seen["durable"] == len(payload["costs"])


@pytest.mark.parametrize("max_jobs", [None, 1, 4])
def test_sync_async_envelopes_and_complete_resume_are_exact(campaign, monkeypatch, max_jobs):
    assert campaign.device.type == "cpu"
    _clear(campaign)
    synchronous, state = _quantum(campaign, monkeypatch, layer=1)
    assert synchronous is not None, getattr(state, "error", None)
    expected = _envelopes(campaign)
    _clear(campaign)
    _enable(monkeypatch, max_jobs=max_jobs)
    asynchronous, state = _quantum(campaign, monkeypatch, layer=1)
    assert asynchronous is not None, getattr(state, "error", None)
    assert _envelopes(campaign) == expected
    assert asynchronous["costs"] == synchronous["costs"]

    observe = replay_module.observe_and_project_retained_windows
    final_backwards = []

    def complete_resume(*args, **kwargs):
        assert kwargs["completed_names"] == set(asynchronous["costs"])
        backward = kwargs["backward"]

        def record_backward(**call):
            final_backwards.append((call["probe_index"], call["final"], call["lease"]))
            return backward(**call)

        kwargs["backward"] = record_backward
        return observe(*args, **kwargs)

    monkeypatch.setattr(replay_module, "observe_and_project_retained_windows", complete_resume)
    resumed, state = _quantum(campaign, monkeypatch, layer=1, resume=True)
    assert resumed is not None, getattr(state, "error", None)
    assert resumed["costs"] == asynchronous["costs"]
    assert _envelopes(campaign) == expected
    assert final_backwards == [(i, True, None) for i in range(3)]


def test_fifo_failure_preserves_real_prefix_and_partial_resume(campaign, monkeypatch):
    assert campaign.device.type == "cpu"
    _clear(campaign)
    baseline, state = _quantum(campaign, monkeypatch, layer=1)
    assert baseline is not None, getattr(state, "error", None)
    expected = _envelopes(campaign)
    _clear(campaign)
    _enable(monkeypatch)
    atomic = aura.atomic_write_bytes
    attempts = []

    def fail_second(path, body):
        if Path(path).suffix == ".pkl":
            attempts.append(pickle.loads(body)["qname"])
            if len(attempts) == 2:
                raise OSError("injected second unit durability failure")
        atomic(path, body)

    monkeypatch.setattr(aura, "atomic_write_bytes", fail_second)
    payload, failed = _quantum(campaign, monkeypatch, layer=1)
    assert payload is None and hasattr(failed, "error")
    assert len(attempts) == 2, "FIFO publication continued behind failure"
    prefix = _envelopes(campaign)
    assert len(prefix) == 1
    assert all(expected[key] == value for key, value in prefix.items())

    monkeypatch.setattr(aura, "atomic_write_bytes", atomic)
    observe = replay_module.observe_and_project_retained_windows
    restored = []

    def partial_resume(*args, **kwargs):
        restored.append(set(kwargs["completed_names"]))
        return observe(*args, **kwargs)

    monkeypatch.setattr(replay_module, "observe_and_project_retained_windows", partial_resume)
    payload, state = _quantum(campaign, monkeypatch, layer=1, resume=True)
    assert payload is not None, getattr(state, "error", None)
    assert restored == [{attempts[0]}]
    assert _envelopes(campaign) == expected
    assert payload["costs"] == baseline["costs"]


def test_final_write_failure_flushes_before_unload_or_return(campaign, monkeypatch):
    assert campaign.device.type == "cpu"
    _clear(campaign)
    entered, release, replay_finished = (threading.Event() for _ in range(3))
    seen = {"tail": None, "failed": False, "premature_unload": False,
            "replay_while_held": False}
    core = quantum.run_layer_quantum_core
    atomic = aura.atomic_write_bytes
    observe = replay_module.observe_and_project_retained_windows

    def launch(*args, **kwargs):
        seen["tail"] = kwargs["resolved_windows"][-1]["names"][-1]
        context = args[0].context
        unload = context.unload

        def checked_unload(layer):
            if layer == 1 and entered.is_set() and not seen["failed"]:
                seen["premature_unload"] = True
            return unload(layer)

        context.unload = checked_unload
        kwargs["execution"] = {
            **kwargs["execution"], "checkpoint_publication_budget_bytes": BUDGET}
        return core(*args, **kwargs)

    def fail_tail(path, body):
        if Path(path).suffix == ".pkl" and pickle.loads(body)["qname"] == seen["tail"]:
            entered.set()
            assert release.wait(30)
            seen["failed"] = True
            raise OSError("injected final unit durability failure")
        atomic(path, body)

    def finished(*args, **kwargs):
        result = observe(*args, **kwargs)
        replay_finished.set()
        return result

    def control():
        try:
            if entered.wait(20):
                seen["replay_while_held"] = replay_finished.wait(10)
        finally:
            release.set()

    monkeypatch.setattr(quantum, "run_layer_quantum_core", launch)
    monkeypatch.setattr(aura, "atomic_write_bytes", fail_tail)
    monkeypatch.setattr(replay_module, "observe_and_project_retained_windows", finished)
    controller = threading.Thread(target=control, name="checkpoint-tail-control")
    controller.start()
    try:
        payload, state = _quantum(campaign, monkeypatch, layer=1)
    finally:
        release.set()
        controller.join(30)
    assert not controller.is_alive()
    assert payload is None and hasattr(state, "error")
    assert seen["failed"] and seen["replay_while_held"]
    assert not seen["premature_unload"]
    assert len(_envelopes(campaign)) == sum(len(w.original_full_target_names)
                                         for w in campaign.preflight[1]) - 1
