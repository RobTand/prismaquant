"""Version dispatch belongs to the canonical producer, not a second reader grammar."""
from __future__ import annotations

import copy
import json

import pytest

from test_rung_allowability import BUILD, FAMILY, _formats, _load, publication


def _versioned_publication(root, table_schema="fleet.rung_allowability.v2"):
    path = root / FAMILY / BUILD["id"] / "v0001.json"
    table = json.loads(path.read_text())
    table["schema"] = table_schema
    if table_schema.endswith(".v2"):
        entry = _formats()[FAMILY]
        recipe = copy.deepcopy(entry["allowable_rungs"]["wire"])
        for row in table["rungs"]:
            if row["measurement_status"] != "measured":
                row["measurements"] = []
                continue
            for measurement in row["measurements"]:
                geometry = measurement["geometry"]
                geometry.update(body_kind="window", decoder_kind="fused_window",
                    decoder_owner="tessera.routed_fused", execution_scope="raw_packed_window",
                    word_ring={"kind": "staged", "owner": "tessera.routed_fused"})
                geometry["shared_memory"]["kind"] = "used"
                geometry["register_pressure"]["compiler"] = "cuda_cuobjdump"
                geometry["recipe"] = copy.deepcopy(recipe)
            row["quality"]["scope"] = {"owner": "tessera.export.encode_linear",
                "format": FAMILY, "grid": entry["grid"],
                "arity": entry["allowable_rungs"]["code_arity"], "rung": row["rung"],
                "recipe": copy.deepcopy(recipe),
                "kernel_kinds": sorted({m["kernel_kind"] for m in row["measurements"]})}
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    index["schema"] = "fleet.rung_allowability.index.v2"
    index["formats"][FAMILY]["kernel_builds"][BUILD["id"]]["versions"]["1"]["table_schema"] = table_schema
    index_path.write_text(json.dumps(index))
    path.write_text(json.dumps(table))
    return path, table


@pytest.mark.parametrize("table_schema", ["fleet.rung_allowability.v1", "fleet.rung_allowability.v2"])
def test_versioned_index_dispatches_v1_and_v2_window_tables(publication, table_schema):
    _versioned_publication(publication, table_schema)
    admitted = _load(publication)
    assert admitted.allows(1024)
    assert not admitted.allows(1025)
    assert not admitted.allows(1026)
    assert admitted.provenance()["schema"] == table_schema


@pytest.mark.parametrize("field", ["body_kind", "decoder_kind", "decoder_owner", "execution_scope", "word_ring"])
def test_v2_missing_body_and_execution_facts_refuse_through_producer(publication, field):
    path, table = _versioned_publication(publication)
    del table["rungs"][0]["measurements"][0]["geometry"][field]
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="schema|body|decoder|missing"):
        _load(publication)


def test_v2_window_zero_width_remains_a_producer_refusal(publication):
    path, table = _versioned_publication(publication)
    table["rungs"][0]["measurements"][0]["geometry"]["decode_width"]["window_bits"] = 0
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="WINDOW|window|decode|schema"):
        _load(publication)


def _native_tcq_publication(publication):

    _path, table = _versioned_publication(publication)
    family = "TESSERA_E2M1_K2"
    entry = copy.deepcopy(_formats()[family])
    step = entry["reader_rate_step_q256"]
    stamp = next(stamp for stamp in entry["attested_wire"] if stamp["q256"] == 896)
    recipe = {key: value for key, value in stamp.items() if key != "q256"}
    table["format"] = family
    table["scope"].update(rung_min=896, rung_max=896, grid_step_q256=step)
    table["rungs"] = table["rungs"][:1]
    table["rungs"][0]["rung"] = 896
    geometry = table["rungs"][0]["measurements"][0]["geometry"]
    geometry.update(body_kind="tcq", decoder_kind="native_tcq",
        decoder_owner="tessera.kernel_a4", execution_scope="native_tcq_decode_gemm",
        word_ring={"kind": "none", "owner": "tessera.kernel_a4"})
    geometry["decode_width"] = {"window_bits": 0, "word_stages": None, "value_bits": 4,
        "arity": 2, "run_widths": [7], "memory": 6, "span": 2,
        "history_lookup_bits": 7, "label_lut_entries": 128,
        "block_m": 64, "block_n": 64, "block_k": 128, "mma_k": 64, "scale_group": 16}
    # Actual seven-plane native o_proj census retained by the corrected
    # producer fixture from cff24a0e7d05; not the former two-plane proxy.
    geometry["alignment"] = {"kind": "tcq_planes",
        "owner": "tessera.compact_prep.prepare_span2_compact", "slot_words": None,
        "plane_shapes": {"select": [528392], "label": [1048576], "point": [6291456],
            "nibbles": [524288], "lut_bytes": [16], "label_lut": [128], "code_nibbles": [256]},
        "plane_bytes": {"select": 528392, "label": 1048576, "point": 6291456,
            "nibbles": 524288, "lut_bytes": 16, "label_lut": 512, "code_nibbles": 256},
        "plane_element_bytes": {"select": 1, "label": 1, "point": 1, "nibbles": 1,
            "lut_bytes": 1, "label_lut": 4, "code_nibbles": 1}}
    geometry["shared_memory"] = {"kind": "used", "requested_bytes": 12800,
        "available_bytes": 101376, "fits": True}
    geometry["register_pressure"] = {"compiler": "triton_compiled_kernel", "REG": 196,
        "SPILLS": 0, "STACK": None, "LOCAL": None,
        "SHARED": geometry["shared_memory"]["requested_bytes"],
        "compiler_symbol": "_a4_span2_gemm_kernel"}
    geometry["recipe"] = copy.deepcopy(recipe)
    table["rungs"][0]["quality"]["scope"] = {"owner": "tessera.export.encode_linear",
        "format": family, "grid": entry["grid"], "arity": 2, "rung": 896,
        "recipe": copy.deepcopy(recipe), "kernel_kinds": ["dense"]}
    target = publication / family / BUILD["id"] / "v0001.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(table))
    index = {"schema": "fleet.rung_allowability.index.v2", "formats": {family: {
        "kernel_builds": {BUILD["id"]: {"current_version": 1, "versions": {"1": {
            "path": str(target.relative_to(publication)), "table_schema": table["schema"],
            "table_status": table["table_status"]}}}}}}}
    (publication / "index.json").write_text(json.dumps(index))
    # A test-only v11 rule over this already published census rung. It does not
    # widen the production pin or assert a new serving cell or wire.
    entry["allowable_rungs"] = {"rule": "window_rate_set", "code_arity": 2,
        "range_q256": [896, 896], "step_q256": step, "run_tables": [[7]],
        "excluded_run_tables": [], "excluded_q256": [],
        "wire": {key: value for key, value in stamp.items() if key != "q256"},
        "evidence": ["docs/measurements/fixture.json"]}
    return target, table, entry


def test_v2_native_tcq_uses_explicit_non_window_facts(publication):
    from prismaquant.rung_allowability import load_rung_allowability
    _path, _table, entry = _native_tcq_publication(publication)
    admitted = load_rung_allowability(publication, format_entry=entry, expected_kernel_build=BUILD)
    assert admitted.allows(896)
    assert admitted.provenance()["schema"] == "fleet.rung_allowability.v2"


@pytest.mark.parametrize("plane", ["select", "label", "point", "nibbles", "lut_bytes",
                                  "label_lut", "code_nibbles"])
def test_v2_native_plane_census_is_owned_and_mandatory(publication, plane):
    from prismaquant.rung_allowability import load_rung_allowability
    path, table, entry = _native_tcq_publication(publication)
    alignment = table["rungs"][0]["measurements"][0]["geometry"]["alignment"]
    for field in ("plane_shapes", "plane_bytes", "plane_element_bytes"):
        del alignment[field][plane]
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="plane census|plane"):
        load_rung_allowability(publication, format_entry=entry, expected_kernel_build=BUILD)


@pytest.mark.parametrize("mutation", ["bytes", "element_width"])
def test_v2_native_plane_bytes_are_refused_by_producer_when_inconsistent(publication, mutation):
    from prismaquant.rung_allowability import load_rung_allowability
    path, table, entry = _native_tcq_publication(publication)
    alignment = table["rungs"][0]["measurements"][0]["geometry"]["alignment"]
    if mutation == "bytes":
        alignment["plane_bytes"]["label_lut"] += 1
    else:
        alignment["plane_element_bytes"]["label_lut"] = 1
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="byte|width"):
        load_rung_allowability(publication, format_entry=entry, expected_kernel_build=BUILD)


@pytest.mark.parametrize("mutation", ["missing", "format", "grid", "arity", "rung",
                                    "recipe", "structure"])
def test_v2_quality_scope_refusal_is_delegated_to_producer(publication, mutation):
    path, table = _versioned_publication(publication)
    quality = table["rungs"][0]["quality"]
    if mutation == "missing":
        del quality["scope"]
    elif mutation == "format":
        quality["scope"]["format"] = "TESSERA_BF16_K1"
    elif mutation == "grid":
        quality["scope"]["grid"] = "BF16"
    elif mutation == "arity":
        quality["scope"]["arity"] = 2
    elif mutation == "rung":
        quality["scope"]["rung"] = 1025
    elif mutation == "recipe":
        quality["scope"]["recipe"]["body"] = "tcq"
    else:
        quality["scope"]["kernel_kinds"] = ["routed"]
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="quality|recipe|scope|arity"):
        _load(publication)


def test_v2_zero_width_point_is_explicit_not_a_positive_placeholder(publication):
    from prismaquant import rung_allowability as reader
    _path, table, _entry = _native_tcq_publication(publication)
    row = table["rungs"][0]
    row["rung"] = 128
    table["scope"].update(rung_min=128, rung_max=128)
    measurement = row["measurements"][0]
    identity = {"cell_id": "fixture:routed:gate_up:M1", "kernel_kind": "routed",
                "shape_id": "gate_up", "M": 1}
    measurement.update(identity)
    table["scope"]["required_cells"] = [identity]
    row["quality"]["scope"].update(rung=128, kernel_kinds=["routed"])
    geometry = measurement["geometry"]
    geometry["decode_width"]["run_widths"] = [1]
    geometry["alignment"]["plane_shapes"]["point"] = [0]
    geometry["alignment"]["plane_bytes"]["point"] = 0
    geometry["alignment"]["plane_shapes"]["code_nibbles"] = [4]
    geometry["alignment"]["plane_bytes"]["code_nibbles"] = 4
    api = reader._producer_api()
    api.validate_table(table)
    assert api.admit_rung(table, format=table["format"], kernel_build_id=BUILD["id"],
                          rung=128)["status"] == "allow"
    # Metadata-only owner control, not a new serving/reader-grid admission.
    geometry["decode_width"]["run_widths"] = [7]
    geometry["alignment"]["plane_shapes"]["code_nibbles"] = [256]
    geometry["alignment"]["plane_bytes"]["code_nibbles"] = 256
    with pytest.raises(ValueError, match="POINT|zero-width"):
        api.validate_table(table)
