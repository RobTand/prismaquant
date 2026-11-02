"""The legacy-capture-mechanism fixture (moved out of ``tests/conftest.py``, #3215).

A helper module, not a test module. Only the positive mechanism tests import
``legacy_capture_mechanism``, so only they depend on
``prismaquant.tessera_calibration_cache`` through it.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def legacy_capture_mechanism(monkeypatch):
    """Exercise explicit legacy capture mechanics, without provider qualification.

    Only positive mechanism tests request this fixture. Production automatic
    admission remains covered without it by the independent refusal matrix.
    This controlled fixture does not establish immutable original material.
    """
    from prismaquant import tessera_calibration_cache as capture

    monkeypatch.setattr(capture, "require_automatic_capture_source_recording",
                        lambda: None)
