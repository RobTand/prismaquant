"""SHA-256 validators share the digests owner (issue #1385).

Five validators spelled their own hex check; each now calls
``digests.is_sha256hex`` with its raise type and message text unchanged.
(Four sibling sites already delegated.) This GoldenTable pins the shared
predicate over str/non-str, uppercase, 63/65-char and non-hex inputs, plus
each site's exact refusal. One noted delta: joint_cost_read_schedule
accepted only exact-``str`` (``type() is``) where the owner accepts
subclasses -- the owner's documented semantics govern.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from prismaquant.digests import is_sha256hex  # noqa: E402
from prismaquant.joint_cost_read_schedule import _sha as schedule_sha  # noqa: E402
from prismaquant.measured_runtime_prices import (  # noqa: E402
    RuntimePriceError,
    _sha as prices_sha,
)
from prismaquant.native_operator_panel import _sha as panel_sha  # noqa: E402
from prismaquant.stage_a_chain_seed import (  # noqa: E402
    ChainSeedRefused,
    _pinned,
)


@pytest.mark.parametrize(("value", "want"), [
    ("0" * 64, True),
    ("abcdef0123456789" * 4, True),
    ("", False),
    (None, False),
    (123, False),
    (b"0" * 64, False),
    ("A" * 64, False),
    ("0" * 63, False),
    ("0" * 65, False),
    ("g" * 64, False),
    ("0" * 63 + " ", False),
])
def test_owner_golden_table(value, want):
    assert is_sha256hex(value) is want


def test_schedule_refusal():
    with pytest.raises(Exception, match="must be a lowercase SHA-256 digest"):
        schedule_sha("xyz", "w")
    assert schedule_sha("0" * 64, "w") == "0" * 64


def test_prices_refusal():
    with pytest.raises(RuntimePriceError, match="expected lowercase SHA-256"):
        prices_sha("xyz", "w")
    assert prices_sha("0" * 64, "w") == "0" * 64


def test_panel_refusal():
    with pytest.raises(ValueError, match="lowercase SHA256 required"):
        panel_sha("xyz", "n")
    assert panel_sha("0" * 64, "n") == "0" * 64


def test_pinned_refusal():
    with pytest.raises(ChainSeedRefused, match="is not a"):
        _pinned({"path": "p", "sha256": "xyz"}, "w")
    assert _pinned({"path": "p", "sha256": "0" * 64}, "w") == {
        "path": "p", "sha256": "0" * 64}
