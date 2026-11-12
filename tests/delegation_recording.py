"""Call-through recording for unary owner-routing tests."""


def record_unary_calls(monkeypatch, module, name):
    """Record arguments before calling the target captured at installation."""
    calls = []
    original = getattr(module, name, None)

    def record(value):
        calls.append(value)
        return original(value) if original is not None else None

    monkeypatch.setattr(module, name, record, raising=False)
    return calls
