"""Development identity drift does not waive authenticated acquisition controls."""
from __future__ import annotations


def test_seal_lint_scans_acquisition_input_owner():
    import test_no_new_seals as lint
    path = "prismaquant/tessera_acquisition_inputs.py"
    sources = lint._campaign_sources()
    assert path in sources, "the authenticated acquisition input owner is outside the seal ratchet"
    sources[path] += '\n\ndef injected_identity_wall(recorded_identity, running_identity):\n    if recorded_identity != running_identity:\n        raise ValueError("recorded producer drift")\n'
    assert any("injected_identity_wall" in problem for problem in lint.violations(sources))
