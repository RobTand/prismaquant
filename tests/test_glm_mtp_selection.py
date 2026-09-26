"""The GLM MTP layer's rungs are chosen under a byte sub-budget (PQ #1346, M6 of #1271).

Layer 45 is priced on its own objective, the MTP head's self-KL, in a table
that the body's join refuses. The allocator's MTP path reads that table,
builds one rung per uniform group (the routed stack, the shared expert), and
hands the menu to the canon selector, ``mtp_rung_selection.select_rung``, with
the declared sub-budget as its memory gate. With no served acceptance points
the selector is degenerate and returns the lowest-E rung within the budget.

Rows here are synthetic but complete joint-AURA entries; these tests make no
serving-quality claim.
"""
from __future__ import annotations

import json
import pickle
import sys

import pytest

torch = pytest.importorskip("torch")

from prismaquant import format_registry as fr  # noqa: E402
from prismaquant import joint_aura as joint  # noqa: E402
from prismaquant import mtp_rung_selection as canon  # noqa: E402
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA  # noqa: E402
from prismaquant.glm_mtp import mtp_objective_identity, with_mtp_objective  # noqa: E402

PREFIX = "model.language_model.layers.45.mlp."
ROUTED = tuple(f"{PREFIX}experts.{e}.{p}" for e in range(2) for p in ("gate_proj", "up_proj", "down_proj"))
SHARED = tuple(f"{PREFIX}shared_experts.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
R1024, R832, E4M3_SHARED = "TESSERA_E4M3_K1_R1024", "TESSERA_E4M3_K1_R832", "TESSERA_E4M3_K1_R1024"
PARAMS = 64 * 128


def _probe(*, objective=True, seed_base=7000):
    source_content = {
        "config": {"fixture": "mtp selection"},
        "weight_map": {"fixture.weight": "fixture.safetensors"},
        "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "a" * 64}],
    }
    source_model = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic",
        "resolved_commit": None, **source_content,
        "content_sha256": joint.identity_sha256(source_content),
    }
    probe = {
        "schema": "prismaquant.joint_aura.probes.v2", "seed_base": seed_base,
        "n_probes": 4, "calibration_sha256": "c" * 64,
        "producer_source_sha256": "d" * 64, "source_model": source_model,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "temperature": 1.0, "arithmetic": joint.arithmetic_identity(torch.float32),
    }
    if objective:
        probe = with_mtp_objective(probe, mtp_objective_identity(mtp_layer=45, sequence_length=512,
                                                                 n_sequences=512))
    return probe


def _row(name, fmt, signed, probe):
    operator = {
        "schema": "prismaquant.joint_aura.operator.v2", "qname": name,
        "format": fmt, "probe_identity_sha256": joint.identity_sha256(probe),
        "source_weight": {"content_sha256": joint.identity_sha256({"source": name}),
                          "shape": [64, 128], "dtype": "torch.bfloat16", "logical_bytes": 2 * PARAMS},
        "rendered_weight": {"content_sha256": joint.identity_sha256({"render": name, "format": fmt}),
                            "shape": [64, 128], "dtype": "torch.float32", "logical_bytes": 4 * PARAMS},
        "activation": joint.activation_identity(fr.get_format("FP8_E4M3"), {}, name),
        "arithmetic": probe["arithmetic"],
    }
    return joint.make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[dict(weight=float(x), activation=0.0, mixed=0.0, total=float(x))
                           for x in signed])


def _payload(*, probe=None, drop=None):
    """Routed rows at two rates, shared rows at one Tessera rate; BF16 source."""
    probe = probe or _probe()
    costs, wire = {}, {}
    for name in ROUTED:
        costs[name] = {R1024: _row(name, R1024, [0.010, 0.012, 0.011, 0.009], probe),
                       R832: _row(name, R832, [0.030, 0.028, 0.031, 0.029], probe)}
        wire[name] = {R1024: 4 * PARAMS // 8 + 16, R832: 13 * PARAMS // 32 + 16}
    for name in SHARED:
        costs[name] = {E4M3_SHARED: _row(name, E4M3_SHARED, [0.020, 0.021, 0.019, 0.020], probe)}
        wire[name] = {E4M3_SHARED: 4 * PARAMS // 8 + 16}
    if drop is not None:
        name, fmt = drop
        del costs[name][fmt]
        del wire[name][fmt]
    return {
        "schema": "prismaquant.glm_mtp_cost.v1",
        "mtp_layer": 45,
        "groups": {"routed": list(ROUTED), "shared": list(SHARED)},
        "params": {name: PARAMS for name in ROUTED + SHARED},
        "source_dtype": {name: "bfloat16" for name in ROUTED + SHARED},
        "costs": costs,
        "wire_bytes": wire,
    }


def _bytes(fmt_routed, fmt_shared):
    payload = _payload()
    routed = sum(payload["wire_bytes"][n][fmt_routed] for n in ROUTED)
    shared = (2 * PARAMS * len(SHARED) if fmt_shared == "BF16"
              else sum(payload["wire_bytes"][n][fmt_shared] for n in SHARED))
    return routed + shared


# A cost side that varies (about 5% of the cycle across the menu), so the
# degenerate branch is reached only for want of acceptance data.
CONSTANTS = {"t_ms": 40.0, "d0_ms": 6.0, "c_ms_per_bit": 0.2, "source": "fixture"}


def test_group_product_menu_sums_rows_per_uniform_group():
    rows = {"a": {"X": (1.0, 10), "Y": (3.0, 4)}, "b": {"X": (2.0, 10), "Y": (5.0, 4)},
            "c": {"P": (0.5, 8)}}
    menu, incomplete = canon.group_product_menu({"g": ("a", "b"), "h": ("c",)}, rows,
                                                params={"a": 8, "b": 8, "c": 8})
    by_name = {r.name: r for r in menu}
    assert set(by_name) == {"g=X|h=P", "g=Y|h=P"}
    assert by_name["g=X|h=P"].E == pytest.approx(3.5)
    assert by_name["g=X|h=P"].resident_bytes == 28
    assert by_name["g=X|h=P"].bits == pytest.approx(8 * 28 / 24)
    assert by_name["g=Y|h=P"].E == pytest.approx(8.5)
    assert incomplete == {}


def test_a_rung_one_unit_lacks_is_not_offered_to_its_group():
    rows = {"a": {"X": (1.0, 10), "Y": (3.0, 4)}, "b": {"X": (2.0, 10)}}
    menu, incomplete = canon.group_product_menu({"g": ("a", "b")}, rows, params={"a": 8, "b": 8})
    assert [r.name for r in menu] == ["g=X"]
    assert incomplete == {"g": {"Y": ["b"]}}


def test_lowest_E_within_the_sub_budget_without_acceptance_points():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    full = _bytes(R1024, "BF16")
    result = select_mtp_rungs(_payload(), byte_budget=full, constants=CONSTANTS)
    assert result["rung"] == f"routed={R1024}|shared=BF16"
    assert result["resident_bytes"] == full
    assert result["selection"]["regime"] == "degenerate"
    assert result["selection"]["degenerate_reason"] == "insufficient_acceptance_data"
    assert {result["assignment"][n] for n in ROUTED} == {R1024}
    assert {result["assignment"][n] for n in SHARED} == {"BF16"}

    tight = _bytes(R1024, E4M3_SHARED)
    result = select_mtp_rungs(_payload(), byte_budget=tight, constants=CONSTANTS)
    assert result["rung"] == f"routed={R1024}|shared={E4M3_SHARED}"

    tighter = _bytes(R832, E4M3_SHARED)
    result = select_mtp_rungs(_payload(), byte_budget=tighter, constants=CONSTANTS)
    assert result["rung"] == f"routed={R832}|shared={E4M3_SHARED}"
    with pytest.raises(ValueError, match="no rung fits"):
        select_mtp_rungs(_payload(), byte_budget=tighter - 1, constants=CONSTANTS)


def test_bf16_is_offered_only_for_a_bf16_source():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    payload = _payload()
    payload["source_dtype"][SHARED[0]] = "float8_e4m3fn"
    result = select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS)
    assert all("shared=BF16" not in row["name"] for row in result["selection"]["menu"])


def test_a_body_probe_identity_is_refused():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    with pytest.raises(ValueError, match="MTP objective"):
        select_mtp_rungs(_payload(probe=_probe(objective=False)), byte_budget=10**12,
                         constants=CONSTANTS)


def test_mixed_probe_identities_are_refused():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    payload = _payload()
    other = _payload(probe=_probe(seed_base=7100))
    payload["costs"][ROUTED[0]][R1024] = other["costs"][ROUTED[0]][R1024]
    with pytest.raises(ValueError, match="one probe identity"):
        select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS)


def test_a_group_rung_missing_for_one_unit_is_recorded_and_not_offered():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    result = select_mtp_rungs(_payload(drop=(ROUTED[3], R832)), byte_budget=10**12,
                              constants=CONSTANTS)
    assert all(f"routed={R832}" not in row["name"] for row in result["selection"]["menu"])
    assert result["incomplete_rungs"] == {"routed": {R832: [ROUTED[3]]}}


def test_an_unattested_rung_is_left_off_the_menu_and_recorded():
    from prismaquant.glm_mtp_selection import select_mtp_rungs

    def eligible(unit, rung):
        return not (rung == R1024 and unit in ROUTED)

    result = select_mtp_rungs(_payload(), byte_budget=10**12, constants=CONSTANTS,
                              eligible=eligible)
    assert result["rung"] == f"routed={R832}|shared=BF16"
    assert result["unattested_rungs"] == {R1024: len(ROUTED)}
    assert all(f"routed={R1024}" not in row["name"] for row in result["selection"]["menu"])


def _write_payload(tmp_path, payload):
    path = tmp_path / "mtp-cost.pkl"
    path.write_bytes(pickle.dumps(payload))
    constants = tmp_path / "serve-constants.json"
    constants.write_text(json.dumps(CONSTANTS))
    return path, constants


def test_allocator_stamps_the_mtp_selection_outside_body_bpp(tmp_path, monkeypatch):
    from tests.test_allocator_output_pin_1304 import _stock_inputs

    from prismaquant import allocator

    from prismaquant import format_registry

    monkeypatch.setattr(format_registry, "format_is_producer_eligible",
                        lambda name, **_: name != R832)
    argv = _stock_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    body = json.loads((tmp_path / "layer_config.json").read_text())

    payload, constants = _write_payload(tmp_path, _payload())
    budget = _bytes(R1024, "BF16")
    out = tmp_path / "with-mtp.json"
    argv = [*argv[:argv.index("--layer-config")], "--layer-config", str(out),
            *argv[argv.index("--layer-config") + 2:],
            "--mtp-joint-cost", str(payload), "--mtp-byte-budget", str(budget),
            "--mtp-serve-constants", str(constants)]
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    got = json.loads(out.read_text())

    meta_key = allocator.LAYER_CONFIG_META_KEY
    record = got[meta_key]["mtp_selection"]
    assert record["rung"] == f"routed={R1024}|shared=BF16"
    assert record["byte_budget"] == budget
    assert record["resident_bytes"] == budget
    assert record["objective"] == "mtp_head_self_kl"
    assert record["unattested_rungs"] == {R832: len(ROUTED)}
    for name in ROUTED:
        assert got[name] == fr.get_format(R1024).autoround_config()
    for name in SHARED:
        assert got[name] == fr.get_format("BF16").autoround_config()
    body_names = {k for k in body if k != meta_key}
    assert body_names == {k for k in got if k != meta_key} - set(ROUTED) - set(SHARED)
    for field in ("assignment_payload_bits_total", "achieved_bits"):
        if field in body[meta_key]:
            assert got[meta_key][field] == body[meta_key][field]


def test_allocator_refuses_an_mtp_name_the_body_assigned(tmp_path, monkeypatch):
    from tests.test_allocator_output_pin_1304 import _stock_inputs

    from prismaquant import allocator

    argv = _stock_inputs(tmp_path)
    payload = _payload()
    clash = "model.layers.0.mlp.down_proj"
    payload["groups"]["shared"] = [*payload["groups"]["shared"], clash]
    payload["params"][clash] = PARAMS
    payload["source_dtype"][clash] = "bfloat16"
    probe = payload["costs"][SHARED[0]][E4M3_SHARED]["probe_identity"]
    payload["costs"][clash] = {E4M3_SHARED: _row(clash, E4M3_SHARED, [0.02, 0.02, 0.02, 0.02], probe)}
    payload["wire_bytes"][clash] = dict(payload["wire_bytes"][SHARED[0]])
    path, constants = _write_payload(tmp_path, payload)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv, "--mtp-joint-cost", str(path),
                                      "--mtp-byte-budget", str(10**12),
                                      "--mtp-serve-constants", str(constants)])
    with pytest.raises(SystemExit, match="MTP"):
        allocator.main()
