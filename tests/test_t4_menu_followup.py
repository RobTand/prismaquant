"""Offline T-4 consumer integration, not speed/numerical/serving qualification.

The immutable canonical metadata owner and synthetic table are test inputs.
No installed/runtime pin is changed, and no private table, API or GPU is used.
In particular, a producer performance verdict is not a native/serving licence.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from prismaquant import rung_allowability as allowability
from prismaquant.tessera_formats import get_tessera_family

FIXTURE = Path(__file__).parent / "fixtures" / "rung_allowability_v3" / "t4_consumer"
FAMILY = "TESSERA_E2M1_K2"


@pytest.fixture
def publication(tmp_path, monkeypatch):
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    # Freeze the *owner*, not a consumer-side rate roster or class formula.
    source = FIXTURE / "producer.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "c2a8fa35aee64bd4effd4cd24ee1e8b6ed208afc30cf71fe2a81f283074a6a3c"
    )
    assert provenance["producer"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256((FIXTURE / "synthetic_t4_scope.json").read_bytes()).hexdigest() == (
        provenance["artifacts"]["synthetic_t4_scope.json"]["sha256"]
    )
    spec = importlib.util.spec_from_file_location("_t4_canonical_test_input", source)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    monkeypatch.setattr(allowability, "_producer_api", lambda: producer)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    table = json.loads((FIXTURE / "synthetic_t4_scope.json").read_text())
    producer.validate_table(table)
    root = tmp_path / "publication"
    root.mkdir()
    path = root / "table.json"
    path.write_text(json.dumps(table))
    index = {
        "schema": "fleet.rung_allowability.index.v2",
        "formats": {FAMILY: {"kernel_builds": {table["kernel_build"]["id"]: {
            "current_version": table["table_version"],
            "versions": {str(table["table_version"]): {
                "path": path.name, "table_schema": table["schema"],
                "table_status": table["table_status"],
            }},
        }}}},
    }
    (root / "index.json").write_text(json.dumps(index))
    return root, path, table, producer


def _entry():
    from prismaquant.lane_eligibility import load_published_formats
    from prismaquant.tessera_runtime_contract import contract_path
    # Real canonical family metadata: deliberately do NOT invent a WINDOW
    # allowable_rungs rule for a native-TCQ family just to make loading work.
    return copy.deepcopy(load_published_formats(contract_path=contract_path())[FAMILY])


def _load(publication, *, entry=None):
    root, _, table, _ = publication
    return allowability.load_rung_allowability(
        root, format_entry=_entry() if entry is None else entry,
        expected_kernel_build=table["kernel_build"],
    )


def _case(publication, kind):
    _, _, table, producer = publication
    published = producer.publication_scope(table)
    (rung,) = published["qualified_menu"][kind]
    row = next(row for row in table["rungs"] if row["rung"] == rung)
    cell = next(cell for cell in row["measurements"] if cell["kernel_kind"] == kind)
    evidence = cell["evidence"]
    scope = {"kernel_kind": kind, "rows": evidence["rows"],
             "columns": evidence["columns"], "m": cell["M"],
             "routing": evidence["routing"],
             "activation_contract": table["kernel_build"]["activation_contract"],
             "recipe": cell["geometry"]["recipe"]}
    return rung, cell, scope


@pytest.mark.parametrize("kind", ["dense", "routed"])
def test_t4_canonical_metadata_reaches_existing_consumer(publication, kind):
    from prismaquant.allocator_candidates import candidate_rung_admission
    rung, cell, scope = _case(publication, kind)
    owner = _load(publication)
    assert owner.allows(rung, scope=scope), owner.refusal(rung, scope=scope)
    admission = candidate_rung_admission(
        f"{FAMILY}_R{rung}", target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_scope=scope,
    )
    assert admission is not None and admission.measured_allowable
    assert admission.admits("research"), admission.detail
    assert not admission.admits("attested"), "performance is not native qualification"
    identity = owner.class_identity(rung, cell)
    assert identity == publication[3].geometry_class_identity(publication[2], rung, cell)
    assert identity["arity"] == get_tessera_family(FAMILY).arity
    assert identity["kind"] == "pure"
    assert get_tessera_family(FAMILY).root_rate(rung).denominator == 1
    speed = owner.canonical_time(rung, cell_id=cell["cell_id"])
    assert speed["value_kind"] == "measured"
    assert speed["measurement"] == cell
    assert not speed["numerical_qualification_inherited"]
    assert not speed["native_qualification_inherited"]
    assert not speed["serving_qualification_inherited"]
    # The table deliberately contains no scientific price. Weight error must
    # not become a fabricated KL value, nor may pure paired-code rates need
    # unrelated scalar-bit chord anchors.
    assert owner.chord_cost(f"{FAMILY}_R{rung}", costs={}, unit="synthetic-unit") is None


def test_t4_pact_menu_keeps_operator_price_and_exact_byte_budget(publication):
    from dataclasses import replace
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    from prismaquant import lane_eligibility as lane, shape_runtime_prices as prices, pact_hull
    from test_shape_runtime_prices import _payload, _doc, _row, SCOPE, COMMIT, SHA, SPAN2

    rung, cell, scope = _case(publication, "dense")
    owner = _load(publication)
    assert owner.allows(rung, scope=scope)
    rows, columns = scope["rows"], scope["columns"]
    name = f"{FAMILY}_R{rung}"
    stats = {"u": {"in_features": columns, "out_features": rows,
                   "n_params": rows * columns, "unit_structure": "dense"}}
    # Synthetic caller-owned prices, not measured KL or operator costs.
    costs = {"u": {name: {"predicted_dloss": 1.0}}}
    built = candidates.build_candidates(
        stats, costs, [registry.get_format(name)], target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_m=cell["M"],
    ).candidates
    assert [option.fmt for option in built.get("u", [])] == [name]
    payload = _payload()
    eligibility = lane._parse_table(
        payload["lane_eligibility"], payload["formats"], "", COMMIT, SHA,
        native_extensions=payload["native_extensions"],
    )
    doc = _doc([_row("dense", f"{rows}x{columns}", FAMILY, rung,
                     cell["M"], SPAN2, (0.023, 0.024, 0.025))],
               regimes=(cell["M"],), tensor_parallel=1)
    table = prices.admit_shape_table(prices.parse_shape_table(doc),
        scope=replace(SCOPE, tensor_parallel=1), eligibility=eligibility)
    key = ("u", name)
    pricing = prices.build_shape_runtime_resources(
        table, built, option_members={key: {"u": name}},
        member_shapes={"u": (rows, columns)}, member_structure={"u": "dense"},
        regime_m=cell["M"], published_formats={row["family"]: row for row in payload["formats"]},
        rung_allowability={FAMILY: owner},
    )
    # An operator's price must not become the 10us raw-kernel fixture time.
    assert pricing.resources[key].prefill_ms == pytest.approx(0.024)
    assert pricing.prefill[key].source != "canonical_class"
    option = built["u"][0]
    assert isinstance(option.memory_bytes, int) and option.memory_bytes > 0
    exact_bytes, _, _ = candidates.serialized_candidate_payload(
        registry.get_format(name), (rows, columns), qname="u")
    assert option.memory_bytes == exact_bytes
    menu = pricing.time_candidates(built)
    times = {k: resource.prefill_ms for k, resource in pricing.resources.items()}
    hull = pact_hull.dichotomic_lower_hull(menu, times, max_memory_bytes=option.memory_bytes)
    assert hull.vertices and all(dict(point.assignment) == {"u": name} for point in hull.vertices)
    assert all(point.memory_bytes == option.memory_bytes for point in hull.vertices)
    with pytest.raises(pact_hull.PactHullError, match="no assignment fits"):
        pact_hull.dichotomic_lower_hull(menu, times, max_memory_bytes=option.memory_bytes - 1)


@pytest.mark.parametrize("kind", ["dense", "routed"])
@pytest.mark.parametrize("missing", ["scope", "activation", "recipe", "speed", "scale", "hold", "native"])
def test_t4_positive_control_then_missing_evidence_still_refuses(publication, kind, missing):
    from prismaquant.allocator_candidates import candidate_rung_admission
    rung, cell, scope = _case(publication, kind)
    owner = _load(publication)
    assert owner.allows(rung, scope=scope), owner.refusal(rung, scope=scope)
    bad_scope = dict(scope)
    if missing == "scope":
        bad_scope.pop("columns")
    elif missing == "activation":
        bad_scope["activation_contract"] = "unmeasured-activation"
    elif missing == "recipe":
        bad_scope["recipe"] = dict(scope["recipe"], seed=123)
    else:
        table = copy.deepcopy(publication[2])
        row = next(row for row in table["rungs"] if row["rung"] == rung)
        target = next(item for item in row["measurements"] if item["cell_id"] == cell["cell_id"])
        if missing == "speed":
            target.update(kernel_time_us=None)
        elif missing == "scale":
            target["geometry"]["decode_width"].pop("scale_group")
        elif missing == "native":
            table["kernel_build"].setdefault("metadata", {})["serving_qualified"] = False
        else:
            row["anomaly_flags"] = row["quality"]["anomaly_flags"] = ["reader_bounds_hold"]
        table["geometry_classes"] = publication[3].measured_geometry_classes(table)
        publication[1].write_text(json.dumps(table))
        if missing in ("speed", "scale"):
            reason = "measured kernel time/path" if missing == "speed" else "native TCQ launch geometry"
            with pytest.raises(ValueError, match=reason):
                _load(publication)
            return  # required raw evidence is checked at load time
        owner = _load(publication)
    reason = {
        "scope": "unmeasured_shape_or_M_scope",
        "activation": "unmeasured_activation_scope",
        "recipe": "unmeasured_recipe_scope",
        "hold": "recorded_correctness_hold",
        "native": "kernel_not_serving_qualified",
    }[missing]
    assert reason in owner.refusal(rung, scope=bad_scope)
    admission = candidate_rung_admission(
        f"{FAMILY}_R{rung}", target_profile="research",
        rung_allowability={FAMILY: owner}, allowability_scope=bad_scope,
    )
    assert admission is not None
    assert not admission.measured_allowable
    assert not admission.admits("research")


@pytest.mark.parametrize("kind", ["dense", "routed"])
def test_t4_capacity_is_not_menu_admission(publication, kind):
    rung, _, scope = _case(publication, kind)
    owner = _load(publication)
    assert owner.allows(rung, scope=scope), owner.refusal(rung, scope=scope)
    family = get_tessera_family(FAMILY)
    low, high = family.mathematical_q256_bounds
    published = set(publication[3].performant_rungs(FAMILY, kind))
    unapproved = [q for q in range(low, high + 1) if q % 256 == 0 and q not in published]
    assert unapproved
    for q in unapproved:
        assert not owner.allows(q, scope=scope)
        assert "performance_admission_not_established" in owner.refusal(q, scope=scope)
