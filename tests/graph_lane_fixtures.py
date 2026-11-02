"""The synthetic-graph-lane fixture (moved out of ``tests/conftest.py``, #3215).

A helper module, not a test module. A test with producer-shaped synthetic
graph records imports ``synthetic_graph_lane`` from here, so only those tests
depend on ``prismaquant.tessera_lane`` through it.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def synthetic_graph_lane(monkeypatch):
    """Isolate unrelated card policies from external numerical qualification.

    Requested explicitly by tests with producer-shaped synthetic graph records.
    Real numerical and unavailable-verifier tests do not request this fixture.
    """
    from prismaquant import tessera_lane

    monkeypatch.setattr(tessera_lane, "graph_equality_verify", lambda receipt, scope: None)
