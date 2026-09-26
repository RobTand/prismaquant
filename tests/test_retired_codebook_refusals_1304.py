"""Every reader of a persisted format name refuses a retired codebook rung (#1304).

The Gridbook codebook lane (``NVFP4_CB_K*`` / ``FP8_CB_K*``) was retired on
2026-09-02, and its format, cost and render code was archived on 2026-09-25 at
``archive/gridbook_lane_2026-09-02``. A ``cost.pkl``, ``layer_config.json``,
``selection.json`` or cache manifest written before that can still name one of
those rungs. Each reader below must refuse such a name with
``RetiredFormatError`` pointing at the archive. It must never skip the row, and
it must never re-render the unit to BF16.

One row per reader. A reader added later that resolves format names belongs in
this table.
"""
from __future__ import annotations

import pytest

from prismaquant import footprint
from prismaquant import format_registry as fr

RETIRED = ("NVFP4_CB_K16", "FP8_CB_K28", "nvfp4_cb_k1", "FP8_CB_K48")
STATS = {
    "model.layers.0.mlp.down_proj": {
        "n_params": 64 * 256,
        "in_features": 256,
        "out_features": 64,
    },
}


def _registry(name, _tmp_path):
    fr.get_format(name)


def _footprint_assignment_bytes(name, _tmp_path):
    footprint.assignment_artifact_bytes(
        {"model.layers.0.mlp.down_proj": name},
        STATS,
        source_total_bytes=2 * 64 * 256,
        source_manifest=None,
        regime="bf16",
    )


def _footprint_tensor_payload(name, _tmp_path):
    footprint.format_tensor_payload_breakdown(
        name, (64, 256), qname="model.layers.0.mlp.down_proj"
    )


def _export_coercion(name, tmp_path):
    from prismaquant.export_native_compressed import (
        _coerce_runtime_legal_assignment,
    )

    _coerce_runtime_legal_assignment(
        str(tmp_path), {"model.layers.0.mlp.down_proj": name}
    )


READERS = [
    pytest.param(_registry, id="format_registry.get_format"),
    pytest.param(
        _footprint_assignment_bytes, id="footprint.assignment_artifact_bytes"
    ),
    pytest.param(
        _footprint_tensor_payload,
        id="footprint.format_tensor_payload_breakdown",
    ),
    pytest.param(
        _export_coercion,
        id="export_native_compressed._coerce_runtime_legal_assignment",
    ),
]


@pytest.mark.parametrize("name", RETIRED)
@pytest.mark.parametrize("reader", READERS)
def test_reader_refuses_a_retired_codebook_rung(reader, name, tmp_path):
    with pytest.raises(fr.RetiredFormatError) as info:
        reader(name, tmp_path)
    assert fr.RETIRED_CODEBOOK_ARCHIVE in str(info.value)


def test_retired_refusal_is_not_a_key_error():
    # Readers that skip unknown formats catch KeyError. A retired rung must
    # not be skippable that way.
    assert not issubclass(fr.RetiredFormatError, KeyError)
    assert issubclass(fr.RetiredFormatError, ValueError)


@pytest.mark.parametrize(
    "name", ("TESSERA_NVFP4_K1_R4", "NVFP4", "FP8_DYNAMIC", "NVFP4_CB", "CB_K16")
)
def test_non_codebook_names_do_not_hit_the_retired_refusal(name):
    assert not fr.RETIRED_CODEBOOK_FORMAT_RE.fullmatch(name)
