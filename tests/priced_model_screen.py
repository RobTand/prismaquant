"""The explicit model binding a CPU screen declares, in one place.

Importing ``priced_model_screen`` into a test module makes it an autouse fixture
FOR THAT MODULE ONLY.  That scoping is the point: a repository-wide default
would mask a missing production binding, and a test that exercises the
production seam must see the production behaviour -- including the refusal --
not a fixture's stand-in for it.  Only tests whose subject IS the Torch
arithmetic declare the model here; everything else binds, or fails to bind,
exactly as production does.
"""
from __future__ import annotations

import pytest


def bind_priced_model_screen():
    from prismaquant import nvfp4_activation_contract as owner

    owner._reset_served_quantizer_identity_for_tests()
    owner.bind_served_quantizer_identity(
        identity=owner.ServedQuantizerIdentity(
            backend=owner.SERVED_QUANTIZER_BACKEND_MODEL),
        require=False,
        context="pytest (CPU screen)")


@pytest.fixture(autouse=True)
def priced_model_screen():
    from prismaquant import nvfp4_activation_contract as owner

    bind_priced_model_screen()
    try:
        yield
    finally:
        owner._reset_served_quantizer_identity_for_tests()
