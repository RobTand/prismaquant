"""Pluggable per-expert projection names on the model profile.

A profile returns its per-expert module attribute names; the default is
('w1','w2','w3') so Qwen3/Qwen3.5 behavior is unchanged.
"""
from __future__ import annotations

from prismaquant.model_profiles.base import ModelProfile


class _Profile(ModelProfile):
    name = "test-default"

    def __init__(self, proj_names):
        self._proj = tuple(proj_names)

    def matches(self, *a, **k):
        return False

    def structure_spec(self):
        return None

    def unpacked_expert_projection_names(self):
        return self._proj


def test_default_profile_projection_names_are_w1_w2_w3():
    prof = _Profile(("w1", "w2", "w3"))
    assert prof.unpacked_expert_projection_names() == ("w1", "w2", "w3")


def test_base_profile_default_unpacked_names():
    # The base ModelProfile (no override) defaults to the Qwen standard.
    class _Bare(ModelProfile):
        name = "bare"

        def matches(self, *a, **k):
            return False

        def structure_spec(self):
            return None

    assert _Bare().unpacked_expert_projection_names() == ("w1", "w2", "w3")
