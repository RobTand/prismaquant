"""The assignment tools publish the same JSON bytes through the bound profile.

The literals below come from the unedited tools. Patches replace the model and
the journals, so that each ``main`` runs on a CPU and the bytes do not depend on
paths. Each run works in its own directory and uses relative paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prismaquant.digests import JsonProfile, bytes_sha256hex as sha  # noqa: E402
from tools import build_sensitivity_ablation_assignments as ablation  # noqa: E402
from tools import compare_joint_layer_gate as gate  # noqa: E402
from tools import make_uniform_assignment as uniform  # noqa: E402

NAME = "direct-ascii-indent2-lax"
EDGE = {"text": "caf\u00e9 \u2603 \U0001f600", "ints": {10: "ten", 2: "two"},
        "mixed": {"b": [1, 2.5, [], {}, None], "a": {}}, "nan": float("nan"),
        "inf": float("inf"), "nested": [[1, [2]], {"z": {"y": []}}]}
NAN = EDGE["nan"]
MIXED = {"a": "x", 1: "y"}
QNAMES = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.caf\u00e9"]


def _ablation(monkeypatch, root, bad):
    (root / "out").mkdir(parents=True)
    monkeypatch.chdir(root)
    summary = {"formats": bad or EDGE, "predicted_dloss_saved": 0.0, "non_bf16_count": 2}
    monkeypatch.setattr(ablation, "load_assignment", lambda path: dict.fromkeys(QNAMES, "NVFP4"))
    monkeypatch.setattr(ablation, "_load_pickle_mapping", lambda path, key: {})
    monkeypatch.setattr(ablation, "_load_costs", lambda path: {})
    monkeypatch.setattr(ablation, "_bpp", lambda stats, assignment: 2.5)
    monkeypatch.setattr(ablation, "_summarize_names", lambda names, **kw: summary)
    ablation.main(["--base-assignment", "base-caf\u00e9.json", "--probe", "probe.pkl",
                   "--costs", "costs.pkl", "--output-dir", "out",
                   "--categories", "self_attn,caf\u00e9", "--top-groups-per-category", "0"])
    return {"overlay": Path("out/assignments/promote_all_self_attn_to_bf16.json"),
            "manifest": Path("out/manifest.json")}


def _uniform(monkeypatch, root, bad):
    (root / "out").mkdir(parents=True)
    monkeypatch.chdir(root)
    result = uniform.UniformAssignment(
        assignment={"a.q": "NVFP4", "b.caf\u00e9": "BF16"}, payload_bytes=8,
        excluded=bad or {"caf\u00e9": "tied (detail)"}, demoted_units=("u\u2603",),
        unit_count=float("inf"), params_by_format={"BF16": 4, "NVFP4": NAN})
    profile = SimpleNamespace(name="p\u00e9", lm_head_name=lambda: "lm_head",
                              build_model_graph=lambda model: None)
    monkeypatch.setattr("prismaquant.model_profiles.registry.detect_profile", lambda model: profile)
    monkeypatch.setattr("prismaquant.tied_embeddings.lm_head_is_tied_alias", lambda *a, **k: False)
    monkeypatch.setattr(uniform, "resolve_target_profile", lambda *a, **k: "vllm_packed_moe")
    monkeypatch.setattr(uniform, "_meta_skeleton", lambda path: None)
    monkeypatch.setattr(uniform, "_scan_source_dtype_manifest", lambda *a: {})
    monkeypatch.setattr(uniform, "build_uniform_assignment", lambda *a, **k: result)
    monkeypatch.setattr(uniform, "assert_assignment_legal", lambda *a, **k: None)
    monkeypatch.setattr(uniform, "layer_config_payload", lambda res, meta=None: EDGE)
    uniform.main(["--model", "model-caf\u00e9", "--format", "NVFP4", "--out", "out/layer_config.json",
                  "--report", "out/report.json"])
    return {"layer_config": Path("out/layer_config.json"), "report": Path("out/report.json")}


def _gate(monkeypatch, root, bad):
    (root / "out").mkdir(parents=True)
    monkeypatch.chdir(root)
    verdict = {"layer": 44, "verdict": "match", "matched": 1, "units_shared": 1, "units_single": 1,
               "units_quantum": 1, "pending": [], "differed": [], "differed_paths": bad or EDGE}
    monkeypatch.setattr(gate, "_journal_identity", lambda path: {"identity_sha256": "0" * 64, "units": {"u"}})
    monkeypatch.setattr(gate, "compare_layer", lambda *a, **k: verdict)
    gate.main(["--single-run", "single-caf\u00e9", "--quantum", "quantum", "--layers", "44",
               "--json", "out/gate.json"])
    return {"gate": Path("out/gate.json")}


RUNNERS = {"ablation": _ablation, "uniform": _uniform, "gate": _gate}
EXPECTED = {
    "overlay": (
        b'{\n  "model.layers.0.self_attn.caf\\u00e9": "BF16",\n  "model.layers.0.self_attn.q_proj": "BF16"\n}\n',
        "963ecb5d18c4fed7cdbdeba79d2bff64dff68892e88264412127c30552bc9951"),
    "manifest": (
        b'{\n  "base_assignment": "base-caf\\u00e9.json",\n  "base_bpp": 2.5,\n  "categories": [\n    "self_attn",\n    "caf\\u00e9"\n  ],\n  "costs": "costs.pkl",\n  "probe": "probe.pkl",\n  "top_groups_per_category": 0,\n  "variants": [\n    {\n      "bpp_after_promotion": 2.5,\n      "bpp_base": 2.5,\n      "bpp_delta": 0.0,\n      "category": "baseline",\n      "entry_count": 0,\n      "formats": {},\n      "kind": "baseline",\n      "label": "baseline",\n      "missing_cost_count": 0,\n      "missing_cost_sample": [],\n      "non_bf16_count": 0,\n      "params": 0,\n      "path": "out/assignments/baseline.json",\n      "predicted_dloss_saved": 0.0\n    },\n    {\n      "bpp_after_promotion": 2.5,\n      "bpp_base": 2.5,\n      "bpp_delta": 0.0,\n      "category": "self_attn",\n      "formats": {\n        "inf": Infinity,\n        "ints": {\n          "2": "two",\n          "10": "ten"\n        },\n        "mixed": {\n          "a": {},\n          "b": [\n            1,\n            2.5,\n            [],\n            {},\n            null\n          ]\n        },\n        "nan": NaN,\n        "nested": [\n          [\n            1,\n            [\n              2\n            ]\n          ],\n          {\n            "z": {\n              "y": []\n            }\n          }\n        ],\n        "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00"\n      },\n      "kind": "category",\n      "label": "promote_all_self_attn_to_bf16",\n      "non_bf16_count": 2,\n      "path": "out/assignments/promote_all_self_attn_to_bf16.json",\n      "predicted_dloss_saved": 0.0\n    }\n  ]\n}\n',
        "beacdfef244119243ee424bd0303a4bd8221f39112b81aa5a845267fac484abd"),
    "layer_config": (
        b'{\n  "inf": Infinity,\n  "ints": {\n    "2": "two",\n    "10": "ten"\n  },\n  "mixed": {\n    "a": {},\n    "b": [\n      1,\n      2.5,\n      [],\n      {},\n      null\n    ]\n  },\n  "nan": NaN,\n  "nested": [\n    [\n      1,\n      [\n        2\n      ]\n    ],\n    {\n      "z": {\n        "y": []\n      }\n    }\n  ],\n  "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00"\n}\n',
        "54b01efda01850eef78b7c90d82ad79646cab46e42a518d945c8dcec901ba220"),
    "report": (
        b'{\n  "achieved_bits": NaN,\n  "assigned": 2,\n  "demoted_units": [\n    "u\\u2603"\n  ],\n  "excluded": {\n    "caf\\u00e9": "tied (detail)"\n  },\n  "format_counts": {\n    "BF16": 1,\n    "NVFP4": 1\n  },\n  "meta": {\n    "achieved_bits": NaN,\n    "fallback_format": "BF16",\n    "generator": "tools/make_uniform_assignment.py",\n    "model_path": "model-caf\\u00e9",\n    "model_profile": "p\\u00e9",\n    "note": "single-format ISOLATION build (format_choice_4p5 \\u00a75 Stage 2 endpoint) \\u2014 not a shippable menu",\n    "schema": "prismaquant.layer_config_meta.v1",\n    "target_bits": NaN,\n    "target_profile": "vllm_packed_moe",\n    "target_profile_default": "vllm_packed_moe",\n    "target_profile_requested": null,\n    "uniform_format": "NVFP4"\n  },\n  "params_by_format": {\n    "BF16": 4,\n    "NVFP4": NaN\n  },\n  "unit_count": Infinity\n}\n',
        "b647b725e7c8aaac5cd46b42cb9d353482b6e54db21206d3d98a52e0aa2460c9"),
    "gate": (
        b'{\n  "all_match": true,\n  "layers": [\n    {\n      "differed": [],\n      "differed_paths": {\n        "inf": Infinity,\n        "ints": {\n          "2": "two",\n          "10": "ten"\n        },\n        "mixed": {\n          "a": {},\n          "b": [\n            1,\n            2.5,\n            [],\n            {},\n            null\n          ]\n        },\n        "nan": NaN,\n        "nested": [\n          [\n            1,\n            [\n              2\n            ]\n          ],\n          {\n            "z": {\n              "y": []\n            }\n          }\n        ],\n        "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00"\n      },\n      "layer": 44,\n      "matched": 1,\n      "pending": [],\n      "units_quantum": 1,\n      "units_shared": 1,\n      "units_single": 1,\n      "verdict": "match"\n    }\n  ],\n  "quantum": "quantum",\n  "schema": "prismaquant.joint_layer_gate_comparison.v1",\n  "single_run": "single-caf\\u00e9"\n}\n',
        "b98310d1e75f3d3fb7f5ab6ad1245089ce90ecc6e8552d6c181cbcf9035fb9b1"),
}
MSG = "'<' not supported between instances of 'int' and 'str'"
REFUSALS = {"ablation": (MSG, ["baseline.json", "promote_all_self_attn_to_bf16.json"]),
            "uniform": (MSG, ["layer_config.json"]), "gate": (MSG, [])}


def test_assignments_bytes(tmp_path, monkeypatch):
    for label, run in RUNNERS.items():
        for key, path in run(monkeypatch, tmp_path / label, None).items():
            raw = path.read_bytes()
            assert raw == EXPECTED[key][0], repr(raw)
            assert sha(raw) == EXPECTED[key][1]


@pytest.mark.parametrize("label", sorted(RUNNERS))
def test_assignments_refusals(label, tmp_path, monkeypatch):
    with pytest.raises(TypeError) as err:
        RUNNERS[label](monkeypatch, tmp_path / label, MIXED)
    message, files = REFUSALS[label]
    assert str(err.value) == message and err.value.__cause__ is None
    assert sorted(p.name for p in Path("out").rglob("*") if p.is_file()) == files


def test_assignments_profile_calls(tmp_path, monkeypatch):
    calls = []
    real = JsonProfile.text

    def spy(self, value):
        calls.append((self.name, value))
        return real(self, value)

    monkeypatch.setattr(JsonProfile, "text", spy)
    seen = {}
    for label, run in RUNNERS.items():
        calls.clear()
        run(monkeypatch, tmp_path / label, None)
        seen[label] = list(calls)
    assert {label: [name for name, _ in rows] for label, rows in seen.items()} == {
        "ablation": [NAME] * 2, "uniform": [NAME] * 2, "gate": [NAME]}
    assert seen["ablation"][0][1] == dict.fromkeys(sorted(QNAMES), "BF16")
    bpp = {"bpp_base": 2.5, "bpp_after_promotion": 2.5, "bpp_delta": 0.0, "predicted_dloss_saved": 0.0}
    assert seen["ablation"][1][1] == {
        "base_assignment": "base-caf\u00e9.json", "probe": "probe.pkl", "costs": "costs.pkl",
        "base_bpp": 2.5, "categories": ["self_attn", "caf\u00e9"], "top_groups_per_category": 0,
        "variants": [
            {"label": "baseline", "kind": "baseline", "category": "baseline", **bpp,
             "path": "out/assignments/baseline.json", "entry_count": 0, "non_bf16_count": 0,
             "params": 0, "formats": {}, "missing_cost_count": 0, "missing_cost_sample": []},
            {"label": "promote_all_self_attn_to_bf16", "kind": "category", "category": "self_attn",
             **bpp, "path": "out/assignments/promote_all_self_attn_to_bf16.json",
             "formats": EDGE, "non_bf16_count": 2}]}
    assert seen["uniform"][0][1] is EDGE
    report = seen["uniform"][1][1]
    fresh = (report["achieved_bits"], report["meta"]["target_bits"], report["meta"]["achieved_bits"])
    assert all(bits != bits for bits in fresh), fresh
    meta = {"schema": "prismaquant.layer_config_meta.v1", "target_profile": "vllm_packed_moe",
            "target_profile_requested": None, "target_profile_default": "vllm_packed_moe",
            "target_bits": NAN, "achieved_bits": NAN, "uniform_format": "NVFP4",
            "fallback_format": "BF16", "generator": "tools/make_uniform_assignment.py",
            "model_path": "model-caf\u00e9", "model_profile": "p\u00e9",
            "note": "single-format ISOLATION build (format_choice_4p5 \u00a75 Stage 2 endpoint)"
                    " \u2014 not a shippable menu"}
    assert {**report, "achieved_bits": NAN,
            "meta": {**report["meta"], "target_bits": NAN, "achieved_bits": NAN}} == {
        "meta": meta, "unit_count": float("inf"), "assigned": 2,
        "format_counts": {"NVFP4": 1, "BF16": 1}, "params_by_format": {"BF16": 4, "NVFP4": NAN},
        "achieved_bits": NAN, "demoted_units": ["u\u2603"], "excluded": {"caf\u00e9": "tied (detail)"}}
    assert seen["gate"][0][1] == {
        "schema": "prismaquant.joint_layer_gate_comparison.v1", "single_run": "single-caf\u00e9",
        "quantum": "quantum", "all_match": True,
        "layers": [{"layer": 44, "verdict": "match", "matched": 1, "units_shared": 1,
                    "units_single": 1, "units_quantum": 1, "pending": [], "differed": [],
                    "differed_paths": EDGE}]}
