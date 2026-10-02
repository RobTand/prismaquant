"""Verify diagnostic-only backward policy restores process flags on failure."""
from __future__ import annotations

import torch

from experiments.alloc_lead_asd.a_side_diag import backward_policy


original = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
try:
    for enabled, warn_only in ((False, False), (True, True), (True, False)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        try:
            with backward_policy(True):
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise LookupError("simulated backward refusal")
        except LookupError:
            pass
        assert (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled()) == (enabled, warn_only)
    print("PASS: strict backward policy restores all prior flags after an exception")
finally:
    torch.use_deterministic_algorithms(original[0], warn_only=original[1])
