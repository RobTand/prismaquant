"""Every reader of a persisted format name refuses a retired codebook rung (#1304).

The Gridbook codebook lane (``NVFP4_CB_K*`` / ``FP8_CB_K*``) was retired on
2026-09-02, and its format, cost and render code was archived on 2026-09-25 at
``archive/gridbook_lane_2026-09-02``. A ``cost.pkl``, ``layer_config.json``,
``selection.json`` or cache manifest written before that can still name one of
those rungs. Each reader below must refuse such a name with
``RetiredFormatError`` pointing at the archive. It must never skip the row, and
it must never re-render the unit to BF16.

One row per reader. A reader added later that resolves format names belongs in
this table. Readers that need a model or a render to reach their format check
carry their refusal test beside their other tests: the cache fill and lookup
(``test_production_weight_cache``), the render cost
(``test_production_render_cost``), the frontier selector CLI
(``test_select_validated_frontier``), the union cache, the weight session and
the perturbed-activation cache.

The second half of the file is the other direction: no Tessera rung reaches a
retired codebook path, and the codebook-only paths named in #1304 are gone.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import pickle
import re

import pytest

from prismaquant import footprint
from prismaquant import format_registry as fr
from prismaquant import schemas

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


def _codebook_entry(name):
    match = re.fullmatch(r"(NVFP4|FP8)_CB_K(\d+)", name.upper())
    return {"data_type": f"{match.group(1).lower()}_cb", "bits": 4,
            "cb_k": int(match.group(2))}


def _layer_config_string(name, _tmp_path):
    from prismaquant.layer_config import canonicalize_format

    canonicalize_format(name)


def _layer_config_scheme(name, _tmp_path):
    from prismaquant.layer_config import canonicalize_format

    canonicalize_format(_codebook_entry(name))


def _cost_payload_menu(name, _tmp_path):
    schemas.validate_cost_payload({"costs": {}, "formats": ["NVFP4", name]})


def _cost_payload_row(name, _tmp_path):
    schemas.validate_cost_payload({
        "costs": {"model.layers.0.mlp.down_proj": {
            "NVFP4": {"predicted_dloss": 1.0},
            name: {"predicted_dloss": 0.5},
        }},
        "formats": ["NVFP4"],
    })


def _cost_shard_merge(name, tmp_path):
    from prismaquant.incremental_measure_quant_cost import merge_cost_pickles

    shard = tmp_path / "shard.pkl"
    shard.write_bytes(pickle.dumps({
        "costs": {"model.layers.0.mlp.down_proj": {
            name: {"weight_mse": 1.0}}},
        "formats": [name],
        "meta": {},
    }))
    merge_cost_pickles([shard], tmp_path / "cost.pkl")
    assert not (tmp_path / "cost.pkl").exists()


def _selection_layer_config(name, _tmp_path):
    from prismaquant.select_validated_frontier import (
        _layer_config_from_assignment,
    )

    _layer_config_from_assignment({"model.layers.0.mlp.down_proj": name})


def _cache_key(name, _tmp_path):
    from prismaquant.production_weight_cache import _is_cb_format_name

    _is_cb_format_name(name)


def _reprice(name, _tmp_path):
    from prismaquant.per_row_pricing import reprice_assignment

    reprice_assignment(
        {"model.layers.0.mlp.down_proj": name},
        STATS,
        {"model.layers.0.mlp.down_proj": {name: {"predicted_dloss": 1.0}}},
        activation_pricing=None,
    )


def _aqua_activation_formats(name, _tmp_path):
    from prismaquant.aqua_activation_cost import required_activation_formats

    required_activation_formats(
        [name], shape=(64, 256), device="cpu", executes_all=True, patterns=())


READERS = [
    pytest.param(_registry, id="format_registry.get_format"),
    pytest.param(_layer_config_string, id="layer_config.canonicalize_format[str]"),
    pytest.param(
        _layer_config_scheme, id="layer_config.canonicalize_format[scheme]"),
    pytest.param(_cost_payload_menu, id="schemas.validate_cost_payload[formats]"),
    pytest.param(_cost_payload_row, id="schemas.validate_cost_payload[costs]"),
    pytest.param(
        _cost_shard_merge, id="incremental_measure_quant_cost.merge_cost_pickles"),
    pytest.param(
        _selection_layer_config,
        id="select_validated_frontier._layer_config_from_assignment",
    ),
    pytest.param(_cache_key, id="production_weight_cache._is_cb_format_name"),
    pytest.param(_reprice, id="per_row_pricing.reprice_assignment"),
    pytest.param(
        _aqua_activation_formats,
        id="aqua_activation_cost.required_activation_formats",
    ),
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


TESSERA = (
    "TESSERA_E2M1_K2_R896",
    "TESSERA_E2M1_K1_R256",
    "TESSERA_E4M3_K1_R512",
    "TESSERA_E4M3_K1_R1024",
)
#: The lane-spec activation globs the retired codebook lane declared.
CODEBOOK_ACTIVATION_GLOBS = ("NVFP4_CB_*", "FP8_CB_*", "*_CB_K*")


@pytest.mark.parametrize("name", TESSERA)
def test_a_tessera_rung_never_reaches_a_codebook_path(name):
    # The name gate every reader above goes through lets it pass.
    assert not fr.RETIRED_CODEBOOK_FORMAT_RE.fullmatch(name)
    assert not schemas.RETIRED_CODEBOOK_NAME_RE.fullmatch(name)
    assert schemas.refuse_retired_codebook_format(name) is None
    # It resolves to a live family, and the footprint prices it without a
    # sidecar (the codebook branch was the only sidecar producer).
    spec = fr.get_format(name)
    assert spec.family not in {"nvfp4_cb", "fp8_cb"}
    row = footprint.format_tensor_payload_breakdown(
        name, (64, 256), qname="model.layers.0.mlp.down_proj")
    assert row["format"] == name
    assert row["tensor_payload_bytes"] > 0
    assert row["sidecar_payload_bytes"] == 0
    assert row["sidecar_identity_key"] is None
    # The AQUA A-side selects formats by lane-spec glob. A codebook glob
    # cannot select a Tessera rung.
    assert not any(
        fnmatch.fnmatchcase(name, pat) for pat in CODEBOOK_ACTIVATION_GLOBS)


def test_the_codebook_only_paths_are_gone():
    from prismaquant import expert_empirical_cost

    # The K-rung ladder in the packed-expert empirical cost.
    ladder = [n for n in dir(expert_empirical_cost)
              if re.search(r"ladder|_cb_|^cb_|codebook", n, re.IGNORECASE)]
    assert ladder == []
    # The min-chain identity check lived in the archived cb_minchain module.
    assert not hasattr(schemas, "validate_chain_identity")
    # The footprint's codebook branches and their sidecar helpers.
    assert [n for n in dir(footprint)
            if re.search(r"(^|_)cb(_|$)|codebook|lattice", n, re.IGNORECASE)] == []
    for module in ("cb_minchain", "nvfp4_cb_formats", "nvfp4_cb_footprint",
                   "cb_ladder_cross_family", "routed_moe_codebooks"):
        assert importlib.util.find_spec(f"prismaquant.{module}") is None, module
