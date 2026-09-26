"""The Tessera native-extension row has one spelling (PQ #1302).

``tessera_runtime_contract.TesseraNativeExtension`` and
``tessera_serving_runtime_pin.TesseraServingNativeExtension`` each rendered the
same four-field contract row. The row now has one owner,
``tessera_serving_runtime_pin.native_extension_contract_row``, which is
stdlib-only so the pin keeps importing nothing from the package. Each class's
``as_contract_row`` is that function. The outcomes below were frozen from the
two pre-consolidation methods (``tests/golden_table.py``).
"""
from __future__ import annotations

import pytest

from prismaquant import tessera_runtime_contract as contract
from prismaquant import tessera_serving_runtime_pin as pin
from prismaquant.lane_eligibility import LaneClaim
from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("module_pairs_1302")

_WHEN_UNAVAILABLE = [
    {},
    {"resident": {"status": "substituted", "decoder": "torch_materialize_stock"},
     "streamed": {"status": "refused", "decoder": None}},
    # Unsorted modes and extra per-mode fields: the row sorts modes and keeps
    # only status and decoder.
    {"streamed": {"status": "refused", "decoder": None, "note": "dropped"},
     "resident": {"decoder": "d", "status": "substituted"}},
    # A mode missing a field is a KeyError, not a silent default.
    {"resident": {"status": "substituted"}},
]


def _dev_row(when_unavailable):
    return contract.TesseraNativeExtension(
        module_name_prefix="tessera_nvfp4_",
        filename_glob="tessera_nvfp4_*.so",
        match="basename_fnmatch",
        source="csrc/nvfp4",
        loaded_by="tessera.serving.nvfp4",
        routes=("r1",),
        when_unavailable=when_unavailable,
        lane=LaneClaim(extension="tessera_nvfp4_", decoder="d"),
    )


def _pin_row(when_unavailable):
    return pin.TesseraServingNativeExtension(
        module_name_prefix="tessera_nvfp4_",
        filename_glob="tessera_nvfp4_*.so",
        match="basename_fnmatch",
        when_unavailable=when_unavailable,
    )


@pytest.mark.parametrize("build", [_dev_row, _pin_row], ids=["contract", "pin"])
@pytest.mark.parametrize("index", range(len(_WHEN_UNAVAILABLE)))
def test_native_extension_contract_row_is_unchanged(build, index):
    GOLDEN.call(lambda: build(_WHEN_UNAVAILABLE[index]).as_contract_row())


def test_both_classes_bind_the_one_owner():
    owner = pin.native_extension_contract_row
    assert contract.TesseraNativeExtension.as_contract_row is owner
    assert pin.TesseraServingNativeExtension.as_contract_row is owner
