"""Overlap the producer's exact cold source seal, never replace it (PQ #1742)."""
import ast
import inspect
from concurrent.futures import Future
from contextlib import ExitStack
from functools import lru_cache
from types import SimpleNamespace

import pytest

from prismaquant import io_engine
from prismaquant import tessera_campaign as campaign


class ControlledSubmit:
    def __init__(self):
        self.future = Future()
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return self.future

    def run(self):
        fn, args = self.calls[0]
        try:
            self.future.set_result(fn(*args))
        except BaseException as error:
            self.future.set_exception(error)


def _start(monkeypatch, *, census=False, capture=False):
    submit = ControlledSubmit()
    monkeypatch.setattr(io_engine, "ENGINE", submit)
    scope = ExitStack()
    args = SimpleNamespace(census_out="census.json" if census else None,
                           capture_calibration_out="capture" if capture else None)
    ahead = campaign._start_encoder_source_seal_ahead(args, scope)
    return ahead, submit, scope


def test_start_submits_without_waiting_and_uses_the_same_producer_seal(monkeypatch):
    calls = []
    digest = "f" * 64

    @lru_cache(maxsize=1)
    def source_seal():
        calls.append("producer")
        return digest

    monkeypatch.setattr(campaign, "_checkpoint_identity_api",
                        lambda: SimpleNamespace(encoder_source_sha256=source_seal))
    ahead, submit, scope = _start(monkeypatch)
    assert ahead is not None
    assert len(submit.calls) == 1
    assert not submit.future.done()  # startup has not blocked on the seal
    assert calls == []
    submit.run()
    assert ahead.wait() == digest
    # The unchanged run-identity call and producer bind now see their own cache.
    assert campaign._checkpoint_identity_api().encoder_source_sha256() == digest
    assert calls == ["producer"]
    scope.close()


@pytest.mark.parametrize("census,capture", [(True, False), (False, True), (True, True)])
def test_nonpricing_heads_do_not_start_an_unused_source_hash(monkeypatch, census, capture):
    ahead, submit, scope = _start(monkeypatch, census=census, capture=capture)
    assert ahead is None
    assert submit.calls == []
    scope.close()


def test_source_seal_failure_reaches_identity_and_cleanup_does_not_mask_it(monkeypatch):
    failure = ValueError("producer source seal failed")

    def fail():
        raise failure

    monkeypatch.setattr(campaign, "_checkpoint_identity_api",
                        lambda: SimpleNamespace(encoder_source_sha256=fail))
    ahead, submit, scope = _start(monkeypatch)
    assert ahead is not None
    submit.run()
    with pytest.raises(ValueError, match="producer source seal failed") as seen:
        ahead.wait()
    assert seen.value is failure
    scope.close()  # drain, without replacing an earlier error during unwind


def test_pricing_scopes_have_no_shared_future(monkeypatch):
    monkeypatch.setattr(campaign, "_checkpoint_identity_api",
                        lambda: SimpleNamespace(encoder_source_sha256=lambda: "a" * 64))
    first, a, scope_a = _start(monkeypatch)
    assert first is not None
    a.run()
    second, b, scope_b = _start(monkeypatch)
    assert second is not None
    assert a.future is not b.future
    b.run()
    assert first.wait() == second.wait() == "a" * 64
    scope_a.close()
    scope_b.close()


def test_main_starts_before_model_and_joins_before_identity_uses():
    tree = ast.parse(inspect.getsource(campaign._main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def named(name):
        return [node.lineno for node in calls
                if isinstance(node.func, ast.Name) and node.func.id == name]

    start, = named("_start_encoder_source_seal_ahead")
    join, = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
             and node.func.attr == "wait" and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "encoder_seal_ahead"]
    assert start < min(named("build_streamed_causal_lm")) < join
    assert join < min(named("_campaign_identity_metadata_plan"))
    assert join < min(named("_campaign_bound_identities"))
    assert join < min(named("_campaign_checkpoint_identity"))
    # Still ask the producer in the existing identity, not a new digest/field.
    identity = ast.parse(inspect.getsource(campaign._campaign_checkpoint_identity))
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "encoder_source_sha256"
               for node in ast.walk(identity)) == 1
