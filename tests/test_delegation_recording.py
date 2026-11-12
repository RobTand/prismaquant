"""Pin observable recorder behavior against a frozen pre-consolidation path."""
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["frozen", "shared"])
def record_calls(request):
    if request.param == "frozen":
        from tests.fixtures.pre226_unary_recorder import _spy
        return _spy
    from tests.delegation_recording import record_unary_calls
    return record_unary_calls


def test_records_before_delegate_with_argument_and_result_identity(record_calls, monkeypatch):
    argument = []
    result = object()

    def original(value):
        assert calls == [argument]
        assert calls[0] is value is argument
        value.append("mutated")
        return result

    target = SimpleNamespace(owner=original)
    with monkeypatch.context() as patch:
        calls = record_calls(patch, target, "owner")
        assert target.owner(argument) is result
        assert calls[0] is argument
        assert calls == [["mutated"]]
    assert target.owner is original


def test_keeps_captured_callable_when_target_is_rebound(record_calls, monkeypatch):
    result = object()
    target = SimpleNamespace(owner=lambda value: result)
    calls = record_calls(monkeypatch, target, "owner")
    recorder = target.owner
    target.owner = lambda value: pytest.fail("rebound target was called")
    argument = object()
    assert recorder(argument) is result
    assert calls == [argument]


@pytest.mark.parametrize("present", [False, True])
def test_absent_or_none_target_returns_none_and_undo_restores(record_calls, monkeypatch, present):
    target = SimpleNamespace(**({"owner": None} if present else {}))
    argument = object()
    with monkeypatch.context() as patch:
        calls = record_calls(patch, target, "owner")
        assert target.owner(argument) is None
        assert calls[0] is argument
    assert hasattr(target, "owner") is present
    if present:
        assert target.owner is None


def test_records_argument_and_propagates_exact_exception(record_calls, monkeypatch):
    error = RuntimeError("delegate failed")
    argument = object()

    def original(value):
        assert calls[0] is value is argument
        raise error

    target = SimpleNamespace(owner=original)
    calls = record_calls(monkeypatch, target, "owner")
    with pytest.raises(RuntimeError) as caught:
        target.owner(argument)
    assert caught.value is error
    assert calls == [argument]


def test_nested_recorders_keep_lists_and_restore_each_layer(record_calls, monkeypatch):
    result = object()
    target = SimpleNamespace(owner=lambda value: result)
    original = target.owner
    argument = object()
    with monkeypatch.context() as outer_patch:
        outer_calls = record_calls(outer_patch, target, "owner")
        outer = target.owner
        with monkeypatch.context() as inner_patch:
            inner_calls = record_calls(inner_patch, target, "owner")
            assert inner_calls is not outer_calls
            assert target.owner(argument) is result
            assert inner_calls == outer_calls == [argument]
        assert target.owner is outer
        assert target.owner(argument) is result
        assert inner_calls == [argument]
        assert outer_calls == [argument, argument]
    assert target.owner is original
