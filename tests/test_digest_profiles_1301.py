"""PQ #1301: every digest site moved onto ``prismaquant.digests`` keeps its bytes.

A digest in this tree is an identity: another run, host or tool must reproduce
it byte for byte. For each replaced function this file holds the code as it
stood before the move (``git show c89726e4834:<file>``), runs it next to the
migrated site on the same inputs, and requires the same outcome: the same
return value, or the same exception type, the same ``str()`` and the same
chained cause. The old code runs over a copy of its own module's globals, so it
raises the module's own exception classes and calls the module's own helpers.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
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


# ---------------------------------------------------------------------------
# The old code, verbatim.
# ---------------------------------------------------------------------------

#: ``cost_stage_checkpoint``'s round-trip encoding, which moved to ``digests``.
OLD_ROUND_TRIP = '''\
def canonical_json(value: object, *, where: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def canonical_json_bytes(value: object, *, where: str) -> bytes:
    """The exact bytes ``canonical_json_sha256`` digests.

    Consumers that must *publish* canonical bytes -- not only hash them --
    read them here, so there is one canonical JSON encoding in the tree
    rather than a second spelling of the same ``json.dumps`` keywords.
    """
    canonical = canonical_json(value, where=where)
    try:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc


def canonical_json_sha256(value: object, *, where: str) -> str:
    return hashlib.sha256(canonical_json_bytes(value, where=where)).hexdigest()


#: The exact types ``json.loads`` produces. Exact, not ``isinstance``: a subclass
#: may override ``__str__``/``__repr__``, which would encode to bytes the stdlib
#: encoder would never write for parsed JSON.
_ALLOWED_SCALARS = (str, int, float, bool, type(None))

#: One encoder with the one canonical set of options, so the normalized path and
#: the generic path cannot drift apart in a keyword.
_CANONICAL_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _require_normalized_json(value: object, *, where: str) -> None:
    """Refuse anything ``json.loads`` cannot produce, before encoding it.

    The generic path normalizes first: ``json.dumps`` stringifies an ``int``,
    ``float``, ``bool`` or ``None`` key, so a graph whose keys are not strings
    encodes to bytes that a reload would encode again differently. That
    normalization is why the generic path cannot simply be streamed, and it is
    why this path refuses such a graph instead of digesting different bytes. A
    ``tuple`` key is refused by ``json`` itself, in both paths.
    """
    active: set = set()
    stack = [(value, False)]
    while stack:
        item, leaving = stack.pop()
        if leaving:
            active.discard(id(item))
            continue
        if item is None or item is True or item is False:
            continue
        kind = type(item)
        if kind is str or kind is int:
            continue
        if kind is float:
            if not math.isfinite(item):
                raise ValueError(f"{where} is not canonical JSON data")
            continue
        if kind is list:
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} is not canonical JSON data: a cycle")
            active.add(marker)
            stack.append((item, True))
            stack.extend((child, False) for child in item)
            continue
        if kind is dict:
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} is not canonical JSON data: a cycle")
            active.add(marker)
            stack.append((item, True))
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(
                        f"{where} is not normalized JSON: a mapping key is not a "
                        f"string but {type(key).__name__}")
                stack.append((child, False))
            continue
        raise ValueError(
            f"{where} is not normalized JSON: {kind.__name__} has no canonical "
            "JSON encoding")


def canonical_json_sha256_normalized(value: object, *, where: str) -> str:
    """``canonical_json_sha256``, for input that is already normalized JSON.

    ``json.loads`` output is normalized by construction: object keys are
    ``str``, containers are ``dict`` and ``list``, scalars are the JSON scalars.
    For that shape the ``dumps``->``loads`` in ``canonical_json`` is the
    identity on the encoding, so this validates the shape and then streams
    ``json.JSONEncoder.iterencode`` -- the stdlib encoder, with the same options
    -- into the digest, instead of building the encoded string, the second
    graph, and the second encoded string.

    The digest is the value ``canonical_json_sha256`` returns for the same
    value. ``tests/test_canonical_json_normalized.py`` holds that equality over
    Unicode, escapes, floats, negative zero, null, booleans and key order, and
    holds the refusals for input that is not normalized. A caller whose value is
    not already normalized JSON -- a non-string key, a tuple, a cycle -- must
    use ``canonical_json_sha256``; a tuple key is refused by ``json`` in both.
    """
    _require_normalized_json(value, where=where)
    digest = hashlib.sha256()
    for chunk in _CANONICAL_ENCODER.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()
'''

#: Each replaced function, by module.
OLD_SOURCES = {
    'prismaquant.aura_cost': '''\
def _canonical_json(value: object, *, where: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def _canonical_json_sha256(value: object, *, where: str) -> str:
    canonical = _canonical_json(value, where=where)
    return hashlib.sha256(json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
''',
    'prismaquant.source_class_format_plan': '''\
def _plan_digest(body: Mapping[str, object]) -> str:
    digest_body = {
        str(key): value
        for key, value in body.items()
        if str(key) != "identity_sha256"
    }
    canonical = canonical_json(digest_body, where="source-class format plan")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
''',
    'prismaquant.artifact_collection': '''\
def _canonical_bytes(value: object, *, where: str) -> bytes:
    try:
        canonical = canonical_json(value, where=where)
    except ValueError as exc:
        raise ArtifactCollectionError(f"{where}: not finite canonical JSON data") from exc
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
''',
    'prismaquant.cb_compile_contract': '''\
def _canonical_sha256(value: object) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
''',
    'prismaquant.glm_capture_compatibility': '''\
def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
''',
    'prismaquant.boundary_control': '''\
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
''',
    'prismaquant.sample_parallel_probe': '''\
def _canonical_sha256(value: object, *, where: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SampleParallelProbeError(
            f"{where} is not canonical JSON data"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _runtime_snapshot_closure_sha256(
    entries: Sequence[Mapping[str, object]],
) -> str:
    encoded = json.dumps(
        list(entries), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
''',
    'prismaquant.cb_learned_bundle': '''\
def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
''',
    'prismaquant.cb_warm_state': '''\
def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
''',
    'prismaquant.cb_learned_promotion': '''\
def _canonical_json(value: object, *, where: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CBLPromotionReceiptError(
            f"{where} is not strict canonical JSON data"
        ) from exc
''',
    'prismaquant.joint_aura': '''\
def identity_sha256(value) -> str:
    if type(value) is _ValidatedProbeIdentity:
        return value._sha256
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
''',
    'prismaquant.measured_runtime_prices': '''\
def identity_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
''',
    'prismaquant.glm_mtp_capture': '''\
def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
''',
    'prismaquant.streaming_model': '''\
def _initialization_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()
''',
    'prismaquant.tessera_allocator': '''\
def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
''',
    'tools.dsv4_afast_burn': '''\
def _sha(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()
''',
    'tools.dsv4_afast_campaign': '''\
def _sha256_text(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
''',
    'tools.dsv4_afast_allocation_grid': '''\
def _content_key(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
''',
    'prismaquant.cb_banked_books': '''\
def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BankedCBLBookError(
            f"burn identity is not canonical JSON data: {exc}"
        ) from exc


def _burn_content_key(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
''',
    'prismaquant.perturbed_x_cache': '''\
def _exact_activation_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
''',
    'prismaquant.native_operator_panel': '''\
def operator_route_identity(route):
    """The route class a table binding names, not the GEMM symbol alone.

    `TESSERA_FP8` and `TESSERA_NVFP4` both execute `torch._scaled_mm`, on
    differently packed operands under different activation contracts. A
    binding that carried only the symbol made the two indistinguishable in the
    one field a downstream consumer compares, so two route classes read as one.

    The identity is therefore the declared route itself -- every coordinate the
    producer declared and `consume_native_receipt` admitted the observed route
    against -- spelled by the same canonical `json.dumps` `identity_sha256`
    hashes with, so no second canonical form of a route exists in this tree and
    key order is not part of the answer. A route with no named symbol is
    refused rather than given an identity: a route nobody can execute is not a
    class.
    """
    if not isinstance(route, Mapping) or not isinstance(route.get("symbol"), str) or not route["symbol"].strip():
        raise ValueError("a declared route must name the symbol it executes")
    return json.dumps(route, sort_keys=True, separators=(",", ":"), allow_nan=False)
''',
    'prismaquant.artifact_registry': '''\
def canonical_layer_config_json(layer_config: Mapping) -> str:
    return json.dumps(layer_config, sort_keys=True, separators=(",", ":"))


def layer_config_sha256(layer_config: Mapping) -> str:
    return hashlib.sha256(
        canonical_layer_config_json(layer_config).encode("utf-8")
    ).hexdigest()
''',
    'prismaquant.cost_streaming': '''\
def canonical_fingerprint_key(fingerprint: dict[str, object]) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
''',
    'prismaquant.nvfp4_cb_footprint': '''\
def _identity_key(identity: Mapping) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))
''',
    'tools.dsv4_ldlq_burn': '''\
def _content_key(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
''',
    'prismaquant.production_recache': '''\
def assignment_digest(assignment: Mapping[str, str]) -> str:
    """Stable digest for the concrete assignment used during re-cache."""
    payload = json.dumps(
        {str(k): str(v) for k, v in sorted(assignment.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
''',
    'prismaquant.tessera_sampled_stack_proposal': '''\
def selected_assignment_sha256(assignment):
    """Bind the expanded member assignment, not its optimizer super-item IDs."""
    _require(isinstance(assignment, dict) and assignment and
             all(isinstance(k, str) and isinstance(v, str) for k, v in assignment.items()),
             'selected research assignment must be complete qname/format pairs')
    return hashlib.sha256(json.dumps(assignment, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
''',
    'prismaquant.runtime_provenance': '''\
def _source_tree_identity(tree):
    """The source-tree installer's own identity, recomputed from the bytes.

    ``experiments/full_engine_plugin_install.py`` seals a source-tree install
    as a SHA-256 over the compact JSON map ``{archive member: sha256}`` of the
    build metadata plus every file under ``src/``, and records the map's size
    as ``plugin_source_members``. That installer emits no archive digest, so
    an archive-only binding refuses every source-tree install outright. This
    is Tessera's function recomputed here, the way :func:`_source_digest`
    already recomputes its source-byte seal: the producer's declared identity
    is checked against bytes this side holds, never accepted as stated.
    """
    members = {name: hashlib.sha256(raw).hexdigest() for name, raw in tree.items()}
    body = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest(), len(members)
''',
    'prismaquant.tessera_footprint': '''\
def _recipe_identity(breakdown: Mapping[str, object]) -> str:
    """SHA-256 over canonical JSON of everything but the digest itself.

    A content address, not an authorization signature.  Recomputing it detects
    a report that has been edited or has drifted from the layout that produced
    it -- which is exactly what a downstream price must not be built on.
    """
    body = {k: v for k, v in breakdown.items()
            if k != "pre_render_recipe_identity_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(payload).hexdigest()
''',
    'prismaquant.model_walk': '''\
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
''',
}


# ---------------------------------------------------------------------------
# Running old and new side by side.
# ---------------------------------------------------------------------------

def _compile(source: str, label: str) -> object:
    # ``from __future__ import annotations`` keeps the old annotations
    # (``Mapping[str, Any]`` and the like) from being evaluated at definition
    # time; it changes nothing the functions do.
    return compile("from __future__ import annotations\n" + source, label, "exec")


def _old_round_trip() -> dict:
    namespace = {"__name__": "old_cost_stage_checkpoint",
                 "hashlib": hashlib, "json": json, "math": math}
    exec(_compile(OLD_ROUND_TRIP, "<old cost_stage_checkpoint>"), namespace)
    return namespace


ROUND_TRIP_NAMES = (
    "canonical_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_json_sha256_normalized",
)


def _old_namespace(module_name: str) -> dict:
    """The site's module globals, with the old code defined over them.

    The migration removed ``hashlib``/``json`` imports that only the old code
    used, so both are put back. ``canonical_json`` is the old round-trip copy,
    the one the old code imported from ``cost_stage_checkpoint``.
    """
    module = importlib.import_module(module_name)
    namespace = dict(vars(module))
    namespace.update(hashlib=hashlib, json=json)
    old_round_trip = _old_round_trip()
    for name in ROUND_TRIP_NAMES:
        namespace[name] = old_round_trip[name]
    exec(_compile(OLD_SOURCES[module_name], f"<old {module_name}>"), namespace)
    return namespace


def _outcome(function, value, kwargs):
    """The return value, or the exception type, text and chained cause.

    Values are compared by ``repr`` as well as type, so ``-0.0`` and ``0.0``
    or ``1`` and ``True`` cannot pass as equal.
    """
    try:
        result = function(value, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the refusal is the outcome
        cause = exc.__cause__
        return ("raises", type(exc), str(exc),
                None if cause is None else (type(cause), str(cause)))
    return ("returns", type(result), repr(result))


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


def _mismatches(old, new, inputs, kwargs) -> list:
    found = []
    for index, value in enumerate(inputs):
        before = _outcome(old, value, kwargs)
        after = _outcome(new, value, kwargs)
        if before != after:
            found.append((index, before, after))
    return found


def test_every_old_source_is_a_site():
    assert sorted({module for module, _, _ in SITES}) == sorted(OLD_SOURCES)


@pytest.mark.parametrize(("module_name", "name", "kwargs"), SITES,
                         ids=[f"{module}.{name}" for module, name, _ in SITES])
def test_site_keeps_its_bytes_and_refusals(module_name, name, kwargs):
    old = _old_namespace(module_name)[name]
    new = getattr(importlib.import_module(module_name), name)
    assert old is not new
    inputs = _inputs(module_name, name)
    assert _mismatches(old, new, inputs, kwargs) == []
    # The inputs must reach both outcomes, or the comparison proves little.
    kinds = {_outcome(new, value, kwargs)[0] for value in inputs}
    assert kinds == {"returns", "raises"}


def test_joint_aura_keeps_the_validated_identity_short_circuit():
    from prismaquant import joint_aura

    identity = object.__new__(joint_aura._ValidatedProbeIdentity)
    object.__setattr__(identity, "_fields", ())
    object.__setattr__(identity, "_sha256", "f" * 64)
    old = _old_namespace("prismaquant.joint_aura")["identity_sha256"]
    assert old(identity) == joint_aura.identity_sha256(identity) == "f" * 64


@pytest.mark.parametrize("name", ROUND_TRIP_NAMES)
def test_round_trip_moved_verbatim(name):
    """``digests`` holds the old ``cost_stage_checkpoint`` text, and the old
    import path still hands out the same function."""
    assert OLD_ROUND_TRIP in Path(digests.__file__).read_text(encoding="utf-8")
    assert getattr(cost_stage_checkpoint, name) is getattr(digests, name)
    old = _old_round_trip()[name]
    inputs = GENERIC_INPUTS + ({"k": "v", "n": [1, 2.5, None]},)
    assert _mismatches(old, getattr(digests, name), inputs, _WHERE) == []


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
    assert _outcome(left, value, {}) != _outcome(right, value, {})


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
