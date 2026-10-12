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
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

import pytest

from prismaquant import cost_stage_checkpoint, digests
from prismaquant.digests import (
    DIRECT_ASCII_LAX,
    DIRECT_ASCII_LAX_DEFAULT_STR,
    DIRECT_ASCII_SPACED_STRICT,
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
    ("prismaquant.artifact_collection", "_canonical_bytes", _WHERE),
    ("prismaquant.glm_capture_compatibility", "_digest", {}),
    ("prismaquant.boundary_control", "digest", {}),
    ("prismaquant.sample_parallel_probe", "_canonical_sha256", _WHERE),
    ("prismaquant.sample_parallel_probe", "_runtime_snapshot_closure_sha256", {}),
    ("prismaquant.joint_aura", "identity_sha256", {}),
    ("prismaquant.measured_runtime_prices", "identity_sha256", {}),
    ("prismaquant.glm_mtp_capture", "_json_sha256", {}),
    ("prismaquant.streaming_model", "_initialization_digest", {}),
    ("prismaquant.tessera_allocator", "_canonical_sha256", {}),
    ("prismaquant.perturbed_x_cache", "_exact_activation_json", {}),
    ("prismaquant.native_operator_panel", "operator_route_identity", {}),
    ("prismaquant.artifact_registry", "canonical_layer_config_json", {}),
    ("prismaquant.artifact_registry", "layer_config_sha256", {}),
    ("prismaquant.cost_streaming", "canonical_fingerprint_key", {}),
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
    ("prismaquant.glm_capture_compatibility", "_digest", DIRECT_UTF8_STRICT, "sha256"),
    ("prismaquant.boundary_control", "digest", DIRECT_UTF8_STRICT, "sha256"),
    ("prismaquant.measured_runtime_prices", "identity_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.glm_mtp_capture", "_json_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.streaming_model", "_initialization_digest", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.tessera_allocator", "_canonical_sha256", DIRECT_ASCII_STRICT, "sha256"),
    ("prismaquant.perturbed_x_cache", "_exact_activation_json", DIRECT_ASCII_STRICT, "text"),
    ("prismaquant.artifact_registry", "canonical_layer_config_json", DIRECT_ASCII_LAX, "text"),
    ("prismaquant.artifact_registry", "layer_config_sha256", DIRECT_ASCII_LAX, "sha256"),
    ("prismaquant.cost_streaming", "canonical_fingerprint_key", DIRECT_ASCII_LAX, "text"),
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


@pytest.mark.parametrize("value", GENERIC_INPUTS)
def test_original_spaced_strict_profile_keeps_direct_bytes_and_errors(value):
    """Original snapshots/writers keep ASCII spaces, not canonical normalization."""
    previous = lambda: json.dumps(value, sort_keys=True, allow_nan=False)
    assert outcome(lambda: DIRECT_ASCII_SPACED_STRICT.text(value)) == outcome(previous)
    assert outcome(lambda: DIRECT_ASCII_SPACED_STRICT.encoded(value)) == outcome(
        lambda: previous().encode("utf-8"))
    assert outcome(lambda: DIRECT_ASCII_SPACED_STRICT.sha256(value)) == outcome(
        lambda: hashlib.sha256(previous().encode("utf-8")).hexdigest())
    assert outcome(lambda: json.loads(DIRECT_ASCII_SPACED_STRICT.text(value))) == outcome(
        lambda: json.loads(previous()))


def test_original_spaced_strict_profile_pins_unicode_and_surrogate_bytes():
    value = {"z": "\ud800", "a": ["café", -0.0, 1.0, None, True]}
    expected = b'{"a": ["caf\\u00e9", -0.0, 1.0, null, true], "z": "\\ud800"}'
    assert DIRECT_ASCII_SPACED_STRICT.encoded(value) == expected


#: One input per pair of neighbouring profiles on which their bytes differ, so
#: no two profiles can be merged without changing a stored digest.
@pytest.mark.parametrize(("left", "right", "value"), (
    (lambda value: digests.canonical_json_bytes(value, where="w"),
     DIRECT_UTF8_STRICT.encoded, {10: "x", 9: "y"}),
    (DIRECT_UTF8_STRICT.encoded, DIRECT_ASCII_STRICT.encoded, {"é": 1}),
    (DIRECT_ASCII_STRICT.encoded, DIRECT_ASCII_LAX.encoded, {"x": float("nan")}),
    (DIRECT_ASCII_LAX.encoded, DIRECT_ASCII_LAX_DEFAULT_STR.encoded,
     {"p": PurePosixPath("/x")}),
    (DIRECT_ASCII_SPACED_STRICT.encoded, DIRECT_ASCII_STRICT.encoded, {"k": 1}),
    (DIRECT_ASCII_SPACED_STRICT.encoded, digests.DIRECT_ASCII_SPACED_LAX.encoded,
     {"x": float("nan")}),
), ids=("round-trip-vs-direct", "utf8-vs-ascii", "strict-vs-lax", "lax-vs-default-str",
        "spaced-vs-compact", "spaced-strict-vs-lax"))
def test_profiles_are_not_interchangeable(left, right, value):
    assert outcome(lambda: left(value)) != outcome(lambda: right(value))


def test_round_trip_and_direct_order_integer_keys_differently():
    value = {10: "x", 9: "y"}
    assert digests.canonical_json_bytes(value, where="w") == b'{"10":"x","9":"y"}'
    assert DIRECT_UTF8_STRICT.encoded(value) == b'{"9":"y","10":"x"}'


def test_profile_names_are_unique():
    profiles = [value for value in vars(digests).values() if isinstance(value, JsonProfile)]
    assert len(profiles) == 9
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


# PQ #1403: the pickle profile. A pickle writes a shared object once and refers
# back to it, so plain ``pickle.dumps`` bytes depend on which equal objects the
# builder happened to share; a resumed campaign row is exactly such an equal
# but distinct object.

def _distinct(text):
    """An equal ``str`` that is not the same object."""
    copy = "".join([text[:1], text[1:]])
    assert copy == text and copy is not text
    return copy


def test_canonical_pickle_bytes_do_not_depend_on_string_sharing():
    import pickle
    shared = "fp8_e4m3"
    one, two = {"x": shared, "y": shared}, {"x": shared, "y": _distinct(shared)}
    assert pickle.dumps(one) != pickle.dumps(two)
    assert digests.canonical_pickle_bytes(one) == digests.canonical_pickle_bytes(two)


def test_canonical_pickle_bytes_do_not_depend_on_container_sharing():
    import pickle
    inner = [1, ("x", 2.5)]
    one, two = {"a": inner, "b": inner}, {"a": [1, ("x", 2.5)], "b": [1, ("x", 2.5)]}
    assert pickle.dumps(one) != pickle.dumps(two)
    assert digests.canonical_pickle_bytes(one) == digests.canonical_pickle_bytes(two)


def test_canonical_pickle_bytes_keep_the_value_order_and_types():
    import pickle
    value = {"b": [1, (2, "z")], "a": {"k": b"\x00"}}
    assert pickle.loads(digests.canonical_pickle_bytes(value)) == value
    loaded = pickle.loads(digests.canonical_pickle_bytes(value))
    assert list(loaded) == ["b", "a"] and type(loaded["b"][1]) is tuple
    assert digests.canonical_pickle_bytes(value) != digests.canonical_pickle_bytes(
        {"a": value["a"], "b": value["b"]})


def test_canonical_pickle_bytes_are_pinned():
    value = {"units": {"a": ["fp8_e4m3", ("x", 1.5)], "b": ["fp8_e4m3", ("x", 1.5)]},
             "blob": b"\x00\x01"}
    assert hashlib.sha256(digests.canonical_pickle_bytes(value)).hexdigest() == (
        "9a756781f28fff2028046167fcd6fb9ded1068c0323b996903ff192abeb2dbe7")


def test_canonical_pickle_bytes_pass_protocol_four_explicitly(monkeypatch):
    import pickle
    original = pickle.dumps
    calls = []

    def record(value, *args, **kwargs):
        calls.append((args, kwargs))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(pickle, "dumps", record)
    value = {"é": [1.5, ("x", b"\x00")]}
    encoded = digests.canonical_pickle_bytes(value)
    # Checking only the wire prefix would miss an omitted protocol on Python
    # 3.12, whose default is already 4. Pin the call and the wire independently.
    assert calls == [((), {"protocol": 4})]
    assert encoded[:2] == b"\x80\x04"
    assert pickle.loads(encoded) == value


def test_canonical_pickle_bytes_refuse_a_container_that_contains_itself():
    loop = []
    loop.append(loop)
    with pytest.raises(ValueError, match="contains itself"):
        digests.canonical_pickle_bytes({"loop": loop})


@pytest.mark.parametrize("raw, expected", [
    (b"", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    (b"hello\n", "ce013625030ba8dba906f756967f9e9ca394464a"),
])
def test_native_git_blob_profile_uses_git_type_length_and_payload(raw, expected):
    from prismaquant.digests import git_blob_sha1hex

    assert git_blob_sha1hex(raw) == expected
    assert git_blob_sha1hex(bytearray(raw)) == expected
    assert git_blob_sha1hex(memoryview(raw)) == expected
# ---------------------------------------------------------------------------
# PQ #2646 routes 20 byte-digest sites to bytes_sha256hex (part of #2541).
# ---------------------------------------------------------------------------
# Each site keeps its input bytes, its result slice and its refusals. The
# rows below freeze the pre-route hashlib outcomes for small inputs shaped
# like each site's real argument. The owner reproduces them byte for byte.

_BYTE_DIGEST_IDS = (
    "hashlib:prismaquant/allocator.py::main@5513:21",
    "hashlib:tools/build_diverse_calibration.py::_json_digest@137:11",
    "hashlib:tools/build_diverse_calibration.py::_sha256_text@132:11",
    "hashlib:tools/build_glm_derivative_image.py::main@100:24",
    "hashlib:tools/build_glm_derivative_image.py::main@123:15",
    "hashlib:tools/build_glm_derivative_image.py::main@125:15",
    "hashlib:tools/build_glm_derivative_image.py::main@145:27",
    "hashlib:tools/build_glm_derivative_image.py::main@85:19",
    "hashlib:tools/build_t4_recovery_request.py::sha@5:20",
    "hashlib:tools/capture_glm_routed_layers.py::main.publish@128:55",
    "hashlib:tools/capture_glm_routing_replay.py::main@99:85",
    "hashlib:tools/check_t4_overlay_prepare.py::sha@36:11",
    "hashlib:tools/compose_tessera_cached_units.py::main@77:41",
    "hashlib:tools/glm_mtp_capture.py::projection_phase@227:62",
    "hashlib:tools/inspect_t4_binding.py::main@12:211",
    "hashlib:tools/materialize_wikitext_corpus.py::main@82:22",
    "hashlib:tools/pq282_equality/recheck_refusal_order.py::<module>@51:58",
    "hashlib:tools/qualify_t4_overlay.py::decoder_provenance@19:303",
    "hashlib:tools/rebind_t4_qualified_results.py::sha@40:11",
    "hashlib:tools/stagea_produced_live_cycle.py::_digest@55:11",
)

#: Small input bytes per site, shaped like the census recipe input. Only the
#: allocator row keeps a result slice (its [:12] label digest).
_BYTE_DIGEST_PAYLOADS = {
    "hashlib:prismaquant/allocator.py::main@5513:21":
        (b'{"model.layers.0.self_attn.q_proj":"NVFP4"}', True),
    "hashlib:tools/build_diverse_calibration.py::_json_digest@137:11":
        (b'{"a": null, "b": [1, "\xc3\xa9"]}', False),
    "hashlib:tools/build_diverse_calibration.py::_sha256_text@132:11":
        (b'caf\xc3\xa9 \xe2\x98\x83 \xe2\x80\xa8\xe8\xb7\xa8', False),
    "hashlib:tools/build_glm_derivative_image.py::main@100:24":
        (b'{"architecture":"glm","os":"linux"}', False),
    "hashlib:tools/build_glm_derivative_image.py::main@123:15":
        (b"config-bytes-123", False),
    "hashlib:tools/build_glm_derivative_image.py::main@125:15":
        (b"layer-bytes-125", False),
    "hashlib:tools/build_glm_derivative_image.py::main@145:27":
        (b"hello-hub-kernels", False),
    "hashlib:tools/build_glm_derivative_image.py::main@85:19":
        (b"tar-layer\x00bytes-001", False),
    "hashlib:tools/build_t4_recovery_request.py::sha@5:20":
        (b'{"roster":{"tasks":[]}}', False),
    "hashlib:tools/capture_glm_routed_layers.py::main.publish@128:55":
        (b"torch-save-bytes-128", False),
    "hashlib:tools/capture_glm_routing_replay.py::main@99:85":
        (b"replay-boundary-bytes", False),
    "hashlib:tools/check_t4_overlay_prepare.py::sha@36:11":
        (b"catalog-cell-bytes", False),
    "hashlib:tools/compose_tessera_cached_units.py::main@77:41":
        (b'{"manifest":"child"}\n', False),
    "hashlib:tools/glm_mtp_capture.py::projection_phase@227:62":
        (b"mtp-projection-bytes", False),
    "hashlib:tools/inspect_t4_binding.py::main@12:211":
        (b"model.language_model.layers.10.mlp.experts.0.down_proj", False),
    "hashlib:tools/materialize_wikitext_corpus.py::main@82:22":
        (b"first row\n\nsecond row", False),
    "hashlib:tools/pq282_equality/recheck_refusal_order.py::<module>@51:58":
        (b"\x80\x04N.", False),
    "hashlib:tools/qualify_t4_overlay.py::decoder_provenance@19:303":
        (b'{"url":"https://example.invalid/r","vcs":"git"}', False),
    "hashlib:tools/rebind_t4_qualified_results.py::sha@40:11":
        (b"qualified-result-bytes", False),
    "hashlib:tools/stagea_produced_live_cycle.py::_digest@55:11":
        (b"staged-input-bytes", False),
}


@pytest.mark.parametrize("site", _BYTE_DIGEST_IDS, ids=_BYTE_DIGEST_IDS)
def test_byte_digest_site_keeps_its_bytes(site):
    """The owner repeats the site's frozen pre-route digest."""
    payload, sliced = _BYTE_DIGEST_PAYLOADS[site]
    if sliced:
        GOLDEN.call(lambda: digests.bytes_sha256hex(payload)[:12])
    else:
        GOLDEN.call(lambda: digests.bytes_sha256hex(payload))


#: Owner contract probes: Unicode, mixed keys, nonfinite values, truncation,
#: framing, the final line feed, and the non-bytes refusals.
_BYTE_DIGEST_CONTRACT = (
    ("unicode-utf8", "café ☃ 日本語".encode("utf-8"), False),
    ("unicode-escaped", b'{"a": "\\u00e9\\u2028"}', False),
    ("mixed-keys", b'{"9": "y", "10": "x"}', False),
    ("nonfinite-lax", b'{"x": NaN, "y": Infinity, "z": -Infinity}', False),
    ("truncation", b"0123456789abcdef", False),
    ("truncation-slice", b"0123456789abcdef", True),
    ("framing-length", b"\x00\x00\x00\x05hello", False),
    ("framing-b64", b"aGVsbG8gd29ybGQ=", False),
    ("final-lf", b'{"a": 1}\n', False),
    ("no-lf", b'{"a": 1}', False),
    ("empty", b"", False),
    ("bytearray", bytearray(b"abc"), False),
    ("memoryview", memoryview(b"abc"), False),
    ("refusal-str", "text", False),
    ("refusal-none", None, False),
    ("refusal-int", 42, False),
)


@pytest.mark.parametrize(
    ("name", "value", "sliced"),
    _BYTE_DIGEST_CONTRACT,
    ids=[name for name, _, _ in _BYTE_DIGEST_CONTRACT],
)
def test_byte_digest_owner_contract(name, value, sliced):
    """The owner keeps the contract bytes and the non-bytes refusals."""
    if sliced:
        GOLDEN.call(lambda: digests.bytes_sha256hex(value)[:12])
    else:
        GOLDEN.call(lambda: digests.bytes_sha256hex(value))


#: Pinned call per site: census id, input source kept at the caller, and
#: whether the caller keeps a result slice.
_BYTE_DIGEST_CALLS = (
    ("hashlib:prismaquant/allocator.py::main@5513:21", "digest_src.encode()", True),
    ("hashlib:tools/build_diverse_calibration.py::_json_digest@137:11", "payload", False),
    ("hashlib:tools/build_diverse_calibration.py::_sha256_text@132:11", 'text.encode("utf-8")', False),
    ("hashlib:tools/build_glm_derivative_image.py::main@100:24", "encoded", False),
    ("hashlib:tools/build_glm_derivative_image.py::main@123:15", "outer.extractfile(config_name).read()", False),
    ("hashlib:tools/build_glm_derivative_image.py::main@125:15", "actual_layer", False),
    ("hashlib:tools/build_glm_derivative_image.py::main@145:27", "base64.b64decode(read['hub'])", False),
    ("hashlib:tools/build_glm_derivative_image.py::main@85:19", "added_layer", False),
    ("hashlib:tools/build_t4_recovery_request.py::sha@5:20", "raw", False),
    ("hashlib:tools/capture_glm_routed_layers.py::main.publish@128:55", "raw", False),
    ("hashlib:tools/capture_glm_routing_replay.py::main@99:85", "raw", False),
    ("hashlib:tools/check_t4_overlay_prepare.py::sha@36:11", "raw", False),
    ("hashlib:tools/compose_tessera_cached_units.py::main@77:41", "raw", False),
    ("hashlib:tools/glm_mtp_capture.py::projection_phase@227:62", "raw", False),
    ("hashlib:tools/inspect_t4_binding.py::main@12:211", "q.encode()", False),
    ("hashlib:tools/materialize_wikitext_corpus.py::main@82:22", "raw", False),
    ("hashlib:tools/pq282_equality/recheck_refusal_order.py::<module>@51:58", "data", False),
    ("hashlib:tools/qualify_t4_overlay.py::decoder_provenance@19:303", "direct_url.encode()", False),
    ("hashlib:tools/rebind_t4_qualified_results.py::sha@40:11", "raw", False),
    ("hashlib:tools/stagea_produced_live_cycle.py::_digest@55:11", "raw", False),
)

_OWNER_CALLEES = frozenset({
    "prismaquant.digests.bytes_sha256hex",
    "digests.bytes_sha256hex",
})


def _scoped_calls(path, scope):
    """Every call in the dotted scope, with its enclosing subscript."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def resolved(func):
        if isinstance(func, ast.Name):
            return aliases.get(func.id, func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return aliases.get(func.value.id, func.value.id) + "." + func.attr
        return None

    calls = []
    subscripts = {}
    stack = [(tree, [])]
    while stack:
        node, chain = stack.pop()
        for child in ast.iter_child_nodes(node):
            sub = chain
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sub = chain + [child.name]
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Call):
                subscripts[id(child.value)] = child
            if isinstance(child, ast.Call) and (".".join(sub) or "<module>") == scope:
                calls.append(child)
            stack.append((child, sub))
    binds_file_owner = "digests.py" in source and "bytes_sha256hex" in source
    return calls, subscripts, resolved, binds_file_owner


def test_byte_digest_sites_call_the_owner():
    """Every pinned site calls bytes_sha256hex on its kept input bytes."""
    root = Path(__file__).resolve().parents[1]
    missed = []
    for site, kept, sliced in _BYTE_DIGEST_CALLS:
        rel, _, scoped = site.partition(":")[2].partition("::")
        scope = scoped.split("@")[0]
        calls, subscripts, resolved, binds_file_owner = _scoped_calls(root / rel, scope)
        wanted = ast.dump(ast.parse(kept, mode="eval").body)
        raw = []
        owned = []
        for call in calls:
            func = call.func
            if (isinstance(func, ast.Attribute) and func.attr == "hexdigest"
                    and isinstance(func.value, ast.Call)
                    and resolved(func.value.func) == "hashlib.sha256"):
                raw.append(call)
                continue
            callee = resolved(func)
            if callee in _OWNER_CALLEES or (
                    callee == "bytes_sha256hex" and binds_file_owner):
                owned.append(call)
        match = [call for call in owned
                 if call.args and ast.dump(call.args[0]) == wanted]
        if raw or not match:
            missed.append(site)
            continue
        if sliced:
            if not any(id(call) in subscripts
                       and isinstance(subscripts[id(call)].slice, ast.Slice)
                       for call in match):
                missed.append(site)
    assert not missed, f"sites not routed to bytes_sha256hex: {missed}"
