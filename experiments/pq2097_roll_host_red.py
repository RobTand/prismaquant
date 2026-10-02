"""Run the allocation oracle on the original allocating core, without an API RED."""
import sys

import pytest

from prismaquant.joint_adjoint_checkpoints import _RollPipeline


original = _RollPipeline.__init__


def accepting_research_keyword(self, *args, reuse_host_buffers=False, **kwargs):
    # Only adapt the constructor signature. The old copy/delivery core is real.
    return original(self, *args, **kwargs)


_RollPipeline.__init__ = accepting_research_keyword
sys.exit(pytest.main([
    "-q", "-p", "no:cacheprovider",
    "tests/test_roll_host_reuse_2097.py::test_exact_rows_order_and_two_bank_bound[True]",
]))
