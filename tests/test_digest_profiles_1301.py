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

from prismaquant import joint_aura_rows
from prismaquant import aura_checkpoint_journal
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


GOLDEN = GoldenTable("digest_profiles_1301", every_row_used=True)


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
    ("prismaquant.aura_checkpoint_journal", "_canonical_json", _WHERE),
    ("prismaquant.aura_checkpoint_journal", "_checkpoint_json_sha256", _WHERE),
    ("prismaquant.artifact_collection", "_canonical_bytes", _WHERE),
    ("prismaquant.glm_capture_compatibility", "_digest", {}),
    ("prismaquant.boundary_control", "digest", {}),
    ("prismaquant.sample_parallel_artifacts", "_sample_artifact_sha256", _WHERE),

    ("prismaquant.joint_aura_rows", "joint_identity_sha256", {}),
    ("prismaquant.measured_runtime_prices", "identity_sha256", {}),
    ("prismaquant.glm_mtp_capture", "_json_sha256", {}),
    ("prismaquant.streaming_model", "_initialization_digest", {}),
    ("prismaquant.tessera_allocator", "_canonical_sha256", {}),
    ("prismaquant.perturbed_x_cache", "_exact_activation_json", {}),
    ("prismaquant.native_operator_panel", "operator_route_identity", {}),
    ("prismaquant.artifact_registry", "canonical_layer_config_json", {}),
    ("prismaquant.artifact_registry", "layer_config_sha256", {}),
    ('prismaquant.streamed_model_identity', "canonical_fingerprint_key", {}),
    ("prismaquant.production_recache", "assignment_digest", {}),
    ("prismaquant.tessera_sampled_stack_proposal", "selected_assignment_sha256", {}),
    ("prismaquant.runtime_provenance", "_source_tree_identity", {}),
    ("prismaquant.tessera_footprint", "_recipe_identity", {}),
    ("prismaquant.model_walk", "_canonical", {}),
)


# PQ #2660: historical census IDs remain the fixture namespace even where the
# implementation has moved. These shaped cases are deliberately not in SITES:
# not every consumer has both a returned and a raised outcome.
_CENSUS_2660 = (
    "hashlib:prismaquant/kl_fisher.py::fisher_probe_scalar@166:42",
    "hashlib:prismaquant/sample_parallel_probe_contract.py::activation_priority_key@90:13",
    "sorted-json:prismaquant/dev_mode.py::_shown@130:18",
    "hashlib:tools/band_serial_handoff_live_pair.py::prepare@152:15",
    "hashlib:prismaquant/proposal_validation.py::_bootstrap_differences@502:21",
    "hashlib:prismaquant/proposal_validation.py::_bootstrap_differences@508:25",
    "hashlib:prismaquant/tessera_campaign.py::_keystream@4267:17",
)


def _census_fisher_outcomes(monkeypatch):
    import torch
    from prismaquant import kl_fisher

    # Equal logits and a Rademacher probe avoid architecture-dependent Gaussian
    # draws/libm rounding while still exercising global-row SHA-derived seeds.
    traces = []
    real_sha256 = hashlib.sha256

    def traced_sha256(payload=b"", *args, **kwargs):
        hashed = real_sha256(payload, *args, **kwargs)
        if payload.startswith(b"prismaquant.kl_fisher.global_row.v1:"):
            prefix = hashed.digest()[:8]
            traces.append((payload, prefix, int.from_bytes(prefix, "little")))
        return hashed

    rows = []
    with monkeypatch.context() as patch:
        patch.setattr(hashlib, "sha256", traced_sha256)
        for offset in (7, 8):
            logits = torch.zeros((1, 1, 4), requires_grad=True)
            scalar = kl_fisher.fisher_probe_scalar(
                logits, seed=19, distribution="rademacher",
                global_row_offset=offset, token_count_override=1)
            scalar.backward()
            rows.append(outcome(lambda: {
                "row_offset": offset, "scalar": scalar.item(),
                "gradient": logits.grad.tolist(), "seed_trace": traces[-1],
            }))
    assert len(traces) == 2 and all(len(trace[1]) == 8 for trace in traces)
    return rows


def _census_activation_outcomes(monkeypatch):
    from prismaquant import sample_parallel_artifacts as artifacts

    key = artifacts.activation_priority_key
    rows = [outcome(lambda group=group: key("ab" * 16, group, is_group=True))
            for group in ("audit-group", "组/café😀", "")]
    rows.append(outcome(lambda: key("ab" * 16, "model.layers.0.self_attn.q_proj")))
    malformed = [outcome(lambda value=value: key(value, "group", is_group=True))
                 for value in ("", "AB" * 16, "g" * 32, "a" * 31)]
    assert all(row["raised"] == "builtins.ValueError" for row in malformed)
    rows.extend(malformed)
    # Reach the actual byte-length guard without allocating a 4-GiB string.
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "_U32_MASK", 3)
        too_long = outcome(lambda: key("ab" * 16, "éé", is_group=True))
    assert too_long["raised"] == "builtins.ValueError"
    assert too_long["text"] == "activation priority group name is too long"
    rows.append(too_long)
    return rows


def _census_shown_outcomes():
    from prismaquant import dev_mode

    cycle = {"padding": "x" * 180}
    cycle["self"] = cycle
    mixed = {1: "x" * 180, "é": 2}
    # Verify these are genuinely the two fallback branches, not short values
    # that bypass the JSON attempt altogether.
    for value, kind in ((mixed, "builtins.TypeError"), (cycle, "builtins.ValueError")):
        assert len(repr(value)) > dev_mode._SHOWN_CHARS
        assert outcome(lambda: json.dumps(value, sort_keys=True, default=repr))["raised"] == kind
    cases = (
        ("boundary", "x" * dev_mode._SHOWN_CHARS),
        ("unicode-truncated", "café😀" * 50),
        ("nonfinite-truncated", {"padding": "n" * 180, "nan": float("nan"),
                                  "positive": float("inf"), "negative": -float("inf")}),
        ("typeerror-fallback", mixed),
        ("valueerror-fallback", cycle),
    )
    rows = []
    for label, value in cases:
        shown = dev_mode._shown(value)
        text = value if isinstance(value, str) else repr(value)
        if len(text) > dev_mode._SHOWN_CHARS:
            try:
                encoded = json.dumps(value, sort_keys=True, default=repr).encode()
            except (TypeError, ValueError):
                encoded = text.encode()
            suffix = hashlib.sha256(encoded).hexdigest()[:16]
            assert shown == (f"{text[:dev_mode._SHOWN_CHARS]}... "
                             f"({len(text)} chars, sha256 {suffix}...)")
            assert len(suffix) == 16
        else:
            assert shown == text
        rows.append(outcome(lambda: (label, shown)))
    return rows


def _census_prepare_outcomes(tmp_path, monkeypatch):
    from tools import band_serial_handoff_live_pair as live_pair
    import dispatch_joint_quanta as dispatch

    # Template construction is a PrismaBuild SDK boundary, unrelated to the
    # head-byte recipe. Keep this unit case offline and deployment-independent;
    # the real tiny records, receipt, wire sealing and prepare body still run.
    def local_template(*args, **kwargs):
        template = tmp_path / "template.json"
        template.write_text('{"fixture": "digest-freeze-2660"}\n', encoding="utf-8")
        return template

    monkeypatch.setattr(dispatch, "handoff_template_path", local_template)
    root = tmp_path / "pair"
    pair = live_pair.prepare(root, "nvme")
    raw = Path(pair["head"]["path"]).read_bytes()
    # A lossless, compact representation of all head bytes, not a second hash
    # of an unobserved file. Avoid root-dependent manifest identities.
    assert len(raw) == live_pair.HEAD_BYTES and len(raw) % 32 == 0
    assert raw == raw[:32] * (len(raw) // 32)
    rows = [outcome(lambda: (len(raw), raw[:32], pair["head"]["sha256"]))]
    refusal = outcome(lambda: live_pair.prepare(root, "nvme"), tmp=tmp_path)
    assert refusal["raised"] == "builtins.SystemExit"
    rows.append(refusal)
    return rows


def _census_bootstrap_outcomes(monkeypatch, *, reject):
    from types import SimpleNamespace

    # This lane was retired after the census. Exercise its actual archived
    # implementation with the live shared canonical helper, without restoring
    # a production import or depending on historical git objects.
    path = (Path(__file__).resolve().parents[1] / "archive" /
            "proposal_validation_2026-10-14" / "prismaquant" / "proposal_validation.py")
    name = "prismaquant._digest_freeze_bootstrap_2660"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    clusters = tuple((
        (module.SequenceLoss(str(index), 1, "complete", float(loss)),
         module.SequenceLoss(str(index), 1, "complete", 1.0)),
    ) for index, loss in enumerate((2, 4, 8)))
    config = module.PairedBootstrapConfig(0, 4, 0.95, 0.0)
    if not reject:
        return [outcome(lambda: module._bootstrap_differences(clusters, config))]

    real_sha256 = hashlib.sha256
    crafted = b"\xff" * 32
    injected = 0
    rejections = 0

    def keyed_sha256(payload=b"", *args, **kwargs):
        nonlocal injected, rejections
        if payload == b"0:0:0":
            injected += 1
            return SimpleNamespace(digest=lambda: crafted, hexdigest=lambda: crafted.hex())
        if payload == crafted:
            rejections += 1
        return real_sha256(payload, *args, **kwargs)

    # Patch the stdlib entry, not a consumer-local alias, so the same keyed
    # stand-in still reaches the operation after a future shared-owner route.
    with monkeypatch.context() as patch:
        patch.setattr(hashlib, "sha256", keyed_sha256)
        draws = module._bootstrap_differences(clusters, config)
    assert injected == 1 and rejections == 1
    return [outcome(lambda: {"draws": draws, "rejections": rejections})]


def _census_keystream_outcomes():
    from itertools import islice
    from prismaquant.tessera_campaign_table import _keystream

    rows = [outcome(lambda seed=seed, label=label: list(islice(_keystream(seed, label), 9)))
            for seed, label in ((0, "audit:stack"), (-17, "café😀"))]
    refusal = outcome(lambda: list(islice(_keystream(0, "\ud800"), 1)))
    assert refusal["raised"] == "builtins.UnicodeEncodeError"
    rows.append(refusal)
    return rows


@pytest.mark.parametrize("census_id", _CENSUS_2660, ids=_CENSUS_2660)
def test_census_site_freeze(census_id, tmp_path, monkeypatch):
    """Freeze current consumer outcomes under the seven stable census IDs.

    Compute all cases before checking rows, so even an absent first fixture
    row cannot hide a setup error or an unexecuted refusal/rejection branch.
    Recording belongs to #2660's implementation, with source unchanged.
    """
    if census_id == _CENSUS_2660[0]:
        rows = _census_fisher_outcomes(monkeypatch)
    elif census_id == _CENSUS_2660[1]:
        rows = _census_activation_outcomes(monkeypatch)
    elif census_id == _CENSUS_2660[2]:
        rows = _census_shown_outcomes()
    elif census_id == _CENSUS_2660[3]:
        rows = _census_prepare_outcomes(tmp_path, monkeypatch)
    elif census_id == _CENSUS_2660[4]:
        rows = _census_bootstrap_outcomes(monkeypatch, reject=False)
    elif census_id == _CENSUS_2660[5]:
        rows = _census_bootstrap_outcomes(monkeypatch, reject=True)
    else:
        assert census_id == _CENSUS_2660[6]
        rows = _census_keystream_outcomes()
    for row in rows:
        GOLDEN.check(row)


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
                         # Oracle keys retain the pre-migration measured site names.
                         ids=[f"{module.replace('aura_checkpoint_journal', 'aura_cost').replace('joint_aura_rows', 'joint_aura')}.{name.replace('_checkpoint_json_sha256', '_canonical_json_sha256').replace('joint_identity_sha256', 'identity_sha256')}"
                              for module, name, _ in SITES])
def test_site_keeps_its_bytes_and_refusals(module_name, name, kwargs):
    new = getattr(importlib.import_module(module_name), name)
    kinds = _check_inputs(new, _inputs(module_name, name), kwargs)
    # The inputs must reach both outcomes, or the comparison proves little.
    assert kinds == {"returned", "raised"}


def test_joint_aura_keeps_the_validated_identity_short_circuit():
    identity = object.__new__(joint_aura_rows._ValidatedProbeIdentity)
    object.__setattr__(identity, "_fields", ())
    object.__setattr__(identity, "_sha256", "f" * 64)
    assert joint_aura_rows.joint_identity_sha256(identity) == "f" * 64


@pytest.mark.parametrize("name", ROUND_TRIP_NAMES)
def test_round_trip_keeps_its_bytes(name):
    """The byte owner keeps its actual value and malformed-input behavior.

    ``cost_stage_checkpoint`` still re-exports these names for its many import
    sites; that path must hand out the owner's function, not a second copy.
    """
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
    ('prismaquant.streamed_model_identity', "canonical_fingerprint_key", DIRECT_ASCII_LAX, "text"),
    ("prismaquant.model_walk", "_canonical", DIRECT_ASCII_LAX_DEFAULT_STR, "text"),
)


@pytest.mark.parametrize(("module_name", "name", "profile", "method"), PROFILE_BINDINGS,
                         ids=[f"{module}.{name}" for module, name, _, _ in PROFILE_BINDINGS])
def test_binding_is_the_profile_method(module_name, name, profile, method):
    bound = getattr(importlib.import_module(module_name), name)
    assert bound.__self__ is profile
    assert bound.__func__ is getattr(JsonProfile, method)


def test_aura_round_trip_binding_is_the_owner():
    assert aura_checkpoint_journal._canonical_json is digests.canonical_json


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
    # A digest profile is a byte identity: adding one updates this count.
    assert len(profiles) == 11
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
