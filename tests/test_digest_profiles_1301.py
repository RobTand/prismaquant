"""PQ #1301: every digest site moved onto ``prismaquant.digests`` keeps its bytes.

A digest in this tree is an identity: another run, host or tool must reproduce
it byte for byte. The migrating PR ran each replaced function's pre-move code
(``git show c89726e4834:<file>``) next to the migrated site on the inputs below
and required the same outcome: the same return value, or the same exception
type, ``str()`` and chained cause. Those outcomes are frozen in
``fixtures/digest_profiles_1301.json`` (PQ #1330), and each site is checked
against them.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path, PurePosixPath
import sys

import pytest

from prismaquant import cost_stage_checkpoint, digests
from prismaquant.digests import (
    DIRECT_ASCII_LAX,
    DIRECT_ASCII_LAX_DEFAULT_STR,
    DIRECT_ASCII_STRICT,
    DIRECT_UTF8_STRICT,
    JsonProfile,
)
from tests.golden_table import GoldenTable, outcome


GOLDEN = GoldenTable("digest_profiles_1301")


_CYCLE_LIST: list = []
_CYCLE_LIST.append(_CYCLE_LIST)
_CYCLE_DICT: dict = {}
_CYCLE_DICT["self"] = _CYCLE_DICT

#: Inputs that separate the encodings: integer and mixed keys, non-ASCII and
#: a lone surrogate, the non-finite floats, float spelling, big integers,
#: tuples, objects JSON cannot encode, and cycles.
GENERIC_INPUTS = (
    {"b": 1, "a": [1, 2.5, None, True, False]},
    {10: "x", 9: "y"},
    {True: 1, 2: "b"},
    {1.5: "f", "z": 0},
    {"a": 1, True: 2},
    {None: 1},
    {"é": 1, " ": 2, "emoji": "\U0001f600"},
    {"k": "café"},
    "\ud800",
    {"k": "\ud800"},
    float("nan"),
    {"x": float("inf")},
    {"x": float("-inf")},
    {"x": -0.0, "y": 1e-7, "z": 1.0, "w": 2 ** 70},
    -0.0,
    2 ** 70,
    (1, 2),
    {"t": (1, "a")},
    {(1, 2): 3},
    {"p": PurePosixPath("/x")},
    {"b": b"raw"},
    {"s": {1, 2}},
    _CYCLE_LIST,
    _CYCLE_DICT,
    {"nested": {"z": {"y": [1, {"x": 2, "a": [None]}]}}},
    [],
    {},
    "",
    None,
)

_ROUTE = {"symbol": "torch._scaled_mm", "activation": "FP8", "operands": ["a", "b"]}

#: Inputs shaped like each site's real argument, added to the generic ones.
SITE_INPUTS = {
    ("prismaquant.native_operator_panel", "operator_route_identity"): (
        _ROUTE,
        {**_ROUTE, "note": "café"},
        {**_ROUTE, "scale": float("nan")},
        {"symbol": "   "},
        {"symbol": 3},
        {"activation": "FP8"},
    ),
    ("prismaquant.tessera_sampled_stack_proposal", "selected_assignment_sha256"): (
        {"model.layers.0.self_attn.q_proj": "NVFP4", "model.layers.0.mlp.up_proj": "FP8"},
        {"café": "NVFP4"},
        {"q": 1},
        {1: "NVFP4"},
    ),
    ("prismaquant.production_recache", "assignment_digest"): (
        {"b": "NVFP4", "a": "FP8_DYNAMIC"},
        {2: 1, 10: 3},
        {1: "a", "b": "c"},
        {"café": "ü"},
    ),
    ("prismaquant.runtime_provenance", "_source_tree_identity"): (
        {"src/a.py": b"x", "pyproject.toml": b"", "src/café.py": b"\xff"},
        {"a": "text"},
    ),
    ("prismaquant.tessera_footprint", "_recipe_identity"): (
        {"a": 1, "pre_render_recipe_identity_sha256": "x", "p": PurePosixPath("/x")},
        {"n": float("nan"), "s": {3}},
    ),
    ("prismaquant.source_class_format_plan", "_plan_digest"): (
        {"identity_sha256": "x", "b": [1], 3: "int key", "café": 1},
        {"identity_sha256": "x"},
        {"s": "\ud800"},
    ),
    ("prismaquant.sample_parallel_probe", "_runtime_snapshot_closure_sha256"): (
        [{"path": "a", "sha256": "0" * 64}, {"path": "b", "sha256": "1" * 64}],
        ({"b": 1},),
        [{"café": "ü"}],
        [{"x": float("nan")}],
    ),
    ("prismaquant.artifact_registry", "canonical_layer_config_json"): (
        {"model.layers.0.self_attn.q_proj": {"format": "NVFP4"}},
    ),
    ("prismaquant.artifact_registry", "layer_config_sha256"): (
        {"model.layers.0.self_attn.q_proj": {"format": "NVFP4"}},
    ),
    ("prismaquant.model_walk", "_canonical"): (
        {"path": PurePosixPath("/models/x"), "n": float("nan")},
    ),
}

_WHERE = {"where": "digest fixture"}

#: Every replaced function: (module, name, keyword arguments).
SITES = (
    ("prismaquant.aura_cost", "_canonical_json", _WHERE),
    ("prismaquant.aura_cost", "_canonical_json_sha256", _WHERE),
    ("prismaquant.source_class_format_plan", "_plan_digest", {}),
    ("prismaquant.artifact_collection", "_canonical_bytes", _WHERE),
    ("prismaquant.cb_compile_contract", "_canonical_sha256", {}),
    ("prismaquant.glm_capture_compatibility", "_digest", {}),
    ("prismaquant.boundary_control", "digest", {}),
    ("prismaquant.sample_parallel_probe", "_canonical_sha256", _WHERE),
    ("prismaquant.sample_parallel_probe", "_runtime_snapshot_closure_sha256", {}),
    ("prismaquant.cb_learned_bundle", "_canonical_json", {}),
    ("prismaquant.cb_warm_state", "_canonical_json", {}),
    ("prismaquant.cb_learned_promotion", "_canonical_json", _WHERE),
    ("prismaquant.joint_aura", "identity_sha256", {}),
    ("prismaquant.measured_runtime_prices", "identity_sha256", {}),
    ("prismaquant.glm_mtp_capture", "_json_sha256", {}),
    ("prismaquant.streaming_model", "_initialization_digest", {}),
    ("prismaquant.tessera_allocator", "_canonical_sha256", {}),
    ("tools.dsv4_afast_burn", "_sha", {}),
    ("tools.dsv4_afast_campaign", "_sha256_text", {}),
    ("tools.dsv4_afast_allocation_grid", "_content_key", {}),
    ("prismaquant.cb_banked_books", "_canonical_json", {}),
    ("prismaquant.cb_banked_books", "_burn_content_key", {}),
    ("prismaquant.perturbed_x_cache", "_exact_activation_json", {}),
    ("prismaquant.native_operator_panel", "operator_route_identity", {}),
    ("prismaquant.artifact_registry", "canonical_layer_config_json", {}),
    ("prismaquant.artifact_registry", "layer_config_sha256", {}),
    ("prismaquant.cost_streaming", "canonical_fingerprint_key", {}),
    ("prismaquant.nvfp4_cb_footprint", "_identity_key", {}),
    ("tools.dsv4_ldlq_burn", "_content_key", {}),
    ("prismaquant.production_recache", "assignment_digest", {}),
    ("prismaquant.tessera_sampled_stack_proposal", "selected_assignment_sha256", {}),
    ("prismaquant.runtime_provenance", "_source_tree_identity", {}),
    ("prismaquant.tessera_footprint", "_recipe_identity", {}),
    ("prismaquant.model_walk", "_canonical", {}),
)


def _inputs(module_name: str, name: str) -> tuple:
    return GENERIC_INPUTS + SITE_INPUTS.get((module_name, name), ())


ROUND_TRIP_NAMES = (
    "canonical_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_json_sha256_normalized",
)


def _check_inputs(function, inputs, kwargs) -> set:
    """Check each input's outcome against its frozen row; return the kinds seen."""
    return {next(iter(GOLDEN.call(lambda value=value: function(value, **kwargs))))
            for value in inputs}


@pytest.mark.parametrize(("module_name", "name", "kwargs"), SITES,
                         ids=[f"{module}.{name}" for module, name, _ in SITES])
def test_site_keeps_its_bytes_and_refusals(module_name, name, kwargs):
    new = getattr(importlib.import_module(module_name), name)
    kinds = _check_inputs(new, _inputs(module_name, name), kwargs)
    # The inputs must reach both outcomes, or the comparison proves little.
    assert kinds == {"returned", "raised"}


def test_joint_aura_keeps_the_validated_identity_short_circuit():
    from prismaquant import joint_aura

    identity = object.__new__(joint_aura._ValidatedProbeIdentity)
    object.__setattr__(identity, "_fields", ())
    object.__setattr__(identity, "_sha256", "f" * 64)
    assert joint_aura.identity_sha256(identity) == "f" * 64


@pytest.mark.parametrize("name", ROUND_TRIP_NAMES)
def test_round_trip_keeps_its_bytes(name):
    """The old ``cost_stage_checkpoint`` import path hands out the owner's function."""
    assert getattr(cost_stage_checkpoint, name) is getattr(digests, name)
    inputs = GENERIC_INPUTS + ({"k": "v", "n": [1, 2.5, None]},)
    _check_inputs(getattr(digests, name), inputs, _WHERE)


#: Sites that became a binding to a profile method: (module, name, profile, method).
PROFILE_BINDINGS = (
    ("prismaquant.cb_compile_contract", "_canonical_sha256", DIRECT_UTF8_STRICT, "sha256"),
    ("prismaquant.glm_capture_compatibility", "_digest", DIRECT_UTF8_STRICT, "sha256"),
    ("prismaquant.boundary_control", "digest", DIRECT_UTF8_STRICT, "sha256"),
    ("prismaquant.cb_learned_bundle", "_canonical_json", DIRECT_UTF8_STRICT, "text"),
    ("prismaquant.cb_warm_state", "_canonical_json", DIRECT_UTF8_STRICT, "text"),
    ("prismaquant.measured_runtime_prices", "identity_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.glm_mtp_capture", "_json_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.streaming_model", "_initialization_digest", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.tessera_allocator", "_canonical_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.perturbed_x_cache", "_exact_activation_json", DIRECT_ASCII_STRICT, "text"),
    ("tools.dsv4_afast_burn", "_sha", DIRECT_ASCII_STRICT, "sha256"),
    ("tools.dsv4_afast_campaign", "_sha256_text", DIRECT_ASCII_STRICT, "sha256"),
    ("tools.dsv4_afast_allocation_grid", "_content_key", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.artifact_registry", "canonical_layer_config_json", DIRECT_ASCII_LAX, "text"),
    ("prismaquant.artifact_registry", "layer_config_sha256", DIRECT_ASCII_LAX, "sha256"),
    ("prismaquant.cost_streaming", "canonical_fingerprint_key", DIRECT_ASCII_LAX, "text"),
    ("prismaquant.nvfp4_cb_footprint", "_identity_key", DIRECT_ASCII_LAX, "text"),
    ("tools.dsv4_ldlq_burn", "_content_key", DIRECT_ASCII_LAX, "sha256"),
    ("prismaquant.model_walk", "_canonical", DIRECT_ASCII_LAX_DEFAULT_STR, "text"),
)


@pytest.mark.parametrize(("module_name", "name", "profile", "method"), PROFILE_BINDINGS,
                         ids=[f"{module}.{name}" for module, name, _, _ in PROFILE_BINDINGS])
def test_binding_is_the_profile_method(module_name, name, profile, method):
    bound = getattr(importlib.import_module(module_name), name)
    assert bound.__self__ is profile
    assert bound.__func__ is getattr(JsonProfile, method)


def test_aura_round_trip_binding_is_the_owner():
    from prismaquant import aura_cost

    assert aura_cost._canonical_json is digests.canonical_json


# ---------------------------------------------------------------------------
# The profiles themselves.
# ---------------------------------------------------------------------------

_SAMPLE = {"b": [1.0, -0.0, 1e-07, 2 ** 70, True, None], "a": "é "}
_SAMPLE_ASCII = '{"a":"\\u00e9\\u2028","b":[1.0,-0.0,1e-07,1180591620717411303424,true,null]}'
_SAMPLE_UTF8 = '{"a":"é ","b":[1.0,-0.0,1e-07,1180591620717411303424,true,null]}'


@pytest.mark.parametrize(("profile", "text", "sha256"), (
    (DIRECT_UTF8_STRICT, _SAMPLE_UTF8,
     "8dbe545b94a2a86b8de7bfb0dfa7bfbe26a293a727598447a4b290b3dec925a5"),
    (DIRECT_ASCII_STRICT, _SAMPLE_ASCII,
     "5926fa56b9a582b1d82bb5ba2ea1ad0aa7c90adb031623a7402e8144a44e5d8a"),
    (DIRECT_ASCII_LAX, _SAMPLE_ASCII,
     "5926fa56b9a582b1d82bb5ba2ea1ad0aa7c90adb031623a7402e8144a44e5d8a"),
    (DIRECT_ASCII_LAX_DEFAULT_STR, _SAMPLE_ASCII,
     "5926fa56b9a582b1d82bb5ba2ea1ad0aa7c90adb031623a7402e8144a44e5d8a"),
), ids=lambda item: item.name if isinstance(item, JsonProfile) else None)
def test_profile_bytes_are_pinned(profile, text, sha256):
    assert profile.text(_SAMPLE) == text
    assert profile.encoded(_SAMPLE) == text.encode("utf-8")
    assert profile.sha256(_SAMPLE) == sha256
    assert digests.canonical_json_sha256(_SAMPLE, where="sample") == (
        "8dbe545b94a2a86b8de7bfb0dfa7bfbe26a293a727598447a4b290b3dec925a5")


#: One input per pair of neighbouring profiles on which their bytes differ, so
#: no two profiles can be merged without changing a stored digest.
@pytest.mark.parametrize(("left", "right", "value"), (
    (lambda value: digests.canonical_json_bytes(value, where="w"),
     DIRECT_UTF8_STRICT.encoded, {10: "x", 9: "y"}),
    (DIRECT_UTF8_STRICT.encoded, DIRECT_ASCII_STRICT.encoded, {"é": 1}),
    (DIRECT_ASCII_STRICT.encoded, DIRECT_ASCII_LAX.encoded, {"x": float("nan")}),
    (DIRECT_ASCII_LAX.encoded, DIRECT_ASCII_LAX_DEFAULT_STR.encoded,
     {"p": PurePosixPath("/x")}),
), ids=("round-trip-vs-direct", "utf8-vs-ascii", "strict-vs-lax", "lax-vs-default-str"))
def test_profiles_are_not_interchangeable(left, right, value):
    assert outcome(lambda: left(value)) != outcome(lambda: right(value))


def test_round_trip_and_direct_order_integer_keys_differently():
    value = {10: "x", 9: "y"}
    assert digests.canonical_json_bytes(value, where="w") == b'{"10":"x","9":"y"}'
    assert DIRECT_UTF8_STRICT.encoded(value) == b'{"9":"y","10":"x"}'


def test_profile_names_are_unique():
    profiles = [value for value in vars(digests).values() if isinstance(value, JsonProfile)]
    assert len(profiles) == 4
    assert len({profile.name for profile in profiles}) == len(profiles)


def test_digests_imports_only_the_standard_library(monkeypatch):
    """A light tool can import ``digests`` without the package or torch."""
    path = Path(digests.__file__)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "digests must not import from the package"
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.partition(".")[0]
            assert top == "__future__" or top in sys.stdlib_module_names, name
    spec = importlib.util.spec_from_file_location("_digests_standalone", path)
    standalone = importlib.util.module_from_spec(spec)
    # ``dataclass`` resolves string annotations through ``sys.modules``.
    monkeypatch.setitem(sys.modules, "_digests_standalone", standalone)
    spec.loader.exec_module(standalone)
    assert standalone.DIRECT_ASCII_STRICT.sha256(_SAMPLE) == DIRECT_ASCII_STRICT.sha256(_SAMPLE)
    assert standalone.canonical_json_sha256(_SAMPLE, where="w") == (
        digests.canonical_json_sha256(_SAMPLE, where="w"))
