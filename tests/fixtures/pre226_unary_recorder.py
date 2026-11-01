"""Frozen test_census_compat_routing_1703.py:20-30 at 05b62b474279.

Independent old-path oracle for PRI-226 test-plumbing consolidation.
"""


def _spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name, None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(module, name, stand_in, raising=False)
    return calls
