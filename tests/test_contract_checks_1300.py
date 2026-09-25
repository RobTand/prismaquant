"""The shared contract checks refuse exactly as each module's own copy did (PQ #1300).

PQ #1300 moved the strict-JSON loaders and the ``_require`` / ``_fail`` /
``_exact_mapping`` / ``_string`` / ``_integer`` / ``_sha256`` / path checks of
many modules onto ``prismaquant.schemas`` (``strict_json_loads`` and
``Contract``). Each ``_old_*`` function below is the pre-#1300 code, copied
verbatim except that module globals are reached through the module. Every
case runs the old copy and the migrated site on the same input and requires
the same outcome: the same returned value, or the same exception type with the
same text, and the same chained cause. The accepting paths are also covered by
each module's own tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any

import pytest

from prismaquant import (
    artifact_collection,
    artifact_collection_legacy,
    cb_banked_books,
    cb_learned_bundle,
    cb_learned_promotion,
    cluster_campaign_contract,
    joint_catalog_extension,
    joint_cost_read_schedule,
    joint_head_walk_quanta,
    joint_served_activation,
    joint_stageb_resources,
    measured_runtime_prices,
    native_runtime_cohort,
    nvfp4_cb_footprint,
    prismasnap_checkpoint,
    prismasnap_moe_checkpoint,
    prismasnap_validation,
    quality_prefill_contract,
    quality_prefill_coverage,
    quality_prefill_knee,
    quality_prefill_pb_adapter,
    quality_prefill_screen,
    runtime_provenance,
    sample_parallel_probe,
    schemas,
    tessera_anchored_surface,
    tessera_campaign,
    tessera_joint_aura,
    tessera_sampled_stack_proposal,
)


def _outcome(call):
    """What a call did, in a form two calls can be compared by."""
    try:
        value = call()
    except BaseException as exc:  # noqa: BLE001 - the exception is the result
        cause = exc.__cause__
        return (
            "raised",
            type(exc),
            str(exc),
            None if cause is None else (type(cause), str(cause)),
        )
    return ("returned", repr(value))


def _same(old, new):
    expected, actual = _outcome(old), _outcome(new)
    assert actual == expected
    return expected


# Documents every strict loader is fed. Each refusal case appears at least once:
# a repeated key (top level and nested), each non-JSON constant, malformed and
# empty text, a non-object root, and an accepted object.
DOCUMENTS = (
    '{"a": 1, "b": [1, 2.5, "x"]}',
    '{"a": 1, "a": 2}',
    '{"outer": {"k": 1, "k": 1}}',
    '[{"k": 1, "k": 2}]',
    '{"a": NaN}',
    '{"a": Infinity}',
    '{"a": -Infinity}',
    '{"a": 1',
    "",
    "[1, 2]",
    '"text"',
)
BYTE_DOCUMENTS = tuple(text.encode("utf-8") for text in DOCUMENTS) + (b"\xff\xfe{}",)


# --- strict JSON loaders --------------------------------------------------------


def _old_artifact_collection_fail(where: str, message: str) -> None:
    raise artifact_collection.ArtifactCollectionError(f"{where}: {message}")


def _old_artifact_collection_reject_duplicate_members(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _old_artifact_collection_fail("JSON", f"duplicate member {key!r}")
        result[key] = value
    return result


def _old_load_record(path) -> dict[str, object]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_old_artifact_collection_reject_duplicate_members,
            parse_constant=lambda item: (_ for _ in ()).throw(
                artifact_collection.ArtifactCollectionError(f"JSON: non-finite value {item}")
            ),
        )
    except artifact_collection.ArtifactCollectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise artifact_collection.ArtifactCollectionError(
            f"unreadable artifact collection record: {source}"
        ) from exc
    return artifact_collection.verify_record(artifact_collection._mapping(value, where=str(source)))


def _old_load_json_source(path: Path, *, logical_schema: str):
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_old_artifact_collection_reject_duplicate_members,
            parse_constant=lambda item: (_ for _ in ()).throw(
                artifact_collection.ArtifactCollectionError(f"JSON: non-finite value {item}")
            ),
        )
    except artifact_collection.ArtifactCollectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise artifact_collection.ArtifactCollectionError(f"unreadable JSON file: {path}") from exc
    if not isinstance(value, dict):
        _old_artifact_collection_fail(str(path), "expected a top-level object")
    hexdigest = hashlib.sha256(encoded).hexdigest()
    return (
        value,
        artifact_collection_legacy.make_reference(
            subject_schema=logical_schema,
            subject_id=hexdigest,
            content_sha256=hexdigest,
            size_bytes=len(encoded),
        ),
        hexdigest,
    )


def _old_cb_banked_books_strict_json_loads(raw: object, *, where: str) -> object:
    if not isinstance(raw, str):
        raise cb_banked_books.BankedCBLBookError(f"{where}: metadata value must be JSON text")

    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise cb_banked_books.BankedCBLBookError(
                    f"{where}: duplicate JSON member {key!r}"
                )
            out[key] = value
        return out

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise cb_banked_books.BankedCBLBookError(f"{where}: malformed JSON: {exc}") from exc


def _old_cb_learned_bundle_strict_json_loads(raw: str, *, where: str) -> object:
    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{where}: duplicate JSON member {key!r}")
            out[key] = value
        return out

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: malformed bundle metadata: {exc}") from exc


def _old_cb_learned_promotion_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise cb_learned_promotion.CBLPromotionReceiptError(f"{where} must be an object")
    return value


def _old_read_promotion_receipt_payload(path) -> Mapping[str, object]:
    receipt_path = Path(path)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise cb_learned_promotion.CBLPromotionReceiptError(
                    f"promotion receipt has duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            receipt_path.read_text(),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise cb_learned_promotion.CBLPromotionReceiptError(
            f"cannot read promotion receipt {receipt_path}: {exc}"
        ) from exc
    return _old_cb_learned_promotion_mapping(payload, where="promotion receipt")


def _old_parse_campaign_manifest(text: str) -> dict[str, object]:
    module = cluster_campaign_contract

    def _fail(message: str) -> None:
        raise module.ClusterCampaignContractError(message)

    if type(text) is not str:
        _fail("campaign manifest JSON must be text")

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"campaign manifest contains duplicate JSON member {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        _fail(f"campaign manifest contains non-JSON constant {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except module.ClusterCampaignContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise module.ClusterCampaignContractError(
            "campaign manifest is not valid strict JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        _fail("campaign manifest JSON root must be an object")
    return module.validate_campaign_manifest(decoded)


def _old_decode_unique(raw: bytes) -> dict:
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"joint COST read schedule: {message}")

    def pairs_hook(pairs):
        row = {}
        for key, value in pairs:
            _require(key not in row, f"duplicate JSON key {key!r}")
            row[key] = value
        return row

    def no_constant(value):
        raise ValueError(f"joint COST read schedule: non-finite JSON {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook,
                          parse_constant=no_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("joint COST read schedule: invalid UTF-8 JSON") from exc


def _old_measured_runtime_prices_json(path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise measured_runtime_prices.RuntimePriceError(
                    f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, ValueError) as exc:
        raise measured_runtime_prices.RuntimePriceError(f"cannot load {path}: {exc}") from exc


def _old_nvfp4_reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _old_nvfp4_unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _old_nvfp4_strict_json_loads(raw: str, *, where: str):
    try:
        return json.loads(
            raw,
            object_pairs_hook=_old_nvfp4_unique_json_object,
            parse_constant=_old_nvfp4_reject_json_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{where}: invalid strict JSON: {exc}") from exc


def _old_prismasnap_checkpoint_load_json(path: Path, *, where: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-JSON constant {value}")

    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("not one regular file")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except Exception as exc:
        raise RuntimeError(f"{where} {path} is unreadable/corrupt") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} {path} must contain a JSON object")
    return value


def _old_prismasnap_validation_json_from_bytes(data: bytes, *, where: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate member {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=exact_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {value}")
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"{where} is corrupt") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{where} must be a JSON object")
    return value


def _old_decode_strict_json(text: str, *, where: str) -> object:
    error = quality_prefill_contract.QualityPrefillContractError

    def _fail(message: str) -> None:
        raise error(message)

    if type(text) is not str:
        _fail(f"{where} must be text")

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{where} contains duplicate JSON member {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        _fail(f"{where} contains non-JSON constant {value}")
        raise AssertionError  # pragma: no cover - _fail always raises

    try:
        return json.loads(
            text,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except error:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise error(f"{where} is not valid strict JSON") from exc


def _old_load_strict_json(raw: bytes | str, *, where: str) -> object:
    error = quality_prefill_pb_adapter.QualityPrefillAdapterError

    def _fail(message: str) -> None:
        raise error(message)

    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        seen: dict[str, object] = {}
        for key, value in items:
            if key in seen:
                _fail(f"{where} repeats the JSON key {key!r}")
            seen[key] = value
        return seen

    def constant(text: str) -> object:
        _fail(f"{where} carries the non-finite number {text}")
        raise AssertionError("unreachable")  # pragma: no cover

    try:
        return json.loads(
            raw.decode("utf-8") if isinstance(raw, bytes) else raw,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except error:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"{where} is not strict UTF-8 JSON: {exc}")
        raise AssertionError("unreachable")


def _old_runtime_provenance_strict_json(raw, path, where):
    error = runtime_provenance.RuntimePriceError

    def _unique_json_object(pairs, where):
        result = {}
        for key, value in pairs:
            if key in result:
                raise error(f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def _reject_nonfinite(value):
        raise ValueError("nonfinite JSON number " + value)

    try:
        value = json.loads(raw, object_pairs_hook=lambda pairs: _unique_json_object(pairs, where),
                           parse_constant=_reject_nonfinite)
    except (ValueError, UnicodeError) as exc:
        raise error(f"{where}: invalid JSON artifact {path}: {exc}") from exc
    return runtime_provenance._mapping(value, where)


def _old_sample_parallel_probe_strict_json_loads(payload: str, *, where: str) -> object:
    error = sample_parallel_probe.SampleParallelProbeError

    def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise error(f"{where} contains duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=_object_from_pairs)
    except error:
        raise
    except (TypeError, ValueError) as exc:
        raise error(f"{where} is invalid JSON") from exc


def _old_parse_family_restriction(value):
    if value is None:
        return None
    from prismaquant.tessera_formats import get_tessera_family

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"family restriction repeats field {key!r}")
            result[key] = item
        return result

    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=unique_object)
    if not isinstance(value, Mapping) or set(value) != {"schema", "dense", "routed_moe"}:
        raise ValueError("family restriction requires exactly schema, dense and routed_moe")
    if value["schema"] != tessera_campaign.FAMILY_RESTRICTION_SCHEMA:
        raise ValueError("family restriction has an unsupported schema")
    result = {"schema": tessera_campaign.FAMILY_RESTRICTION_SCHEMA}
    for structure in ("dense", "routed_moe"):
        names = value[structure]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise ValueError(f"family restriction {structure} requires a nonempty family list")
        if len(set(names)) != len(names):
            raise ValueError(f"family restriction {structure} repeats a family")
        for name in names:
            try:
                family = get_tessera_family(name)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"family restriction has unknown family {name!r}") from exc
            if family.name != name:
                raise ValueError(f"family restriction requires canonical family name {name!r}")
        result[structure] = sorted(names)
    return result


TEXT_LOADERS = {
    "cb_banked_books": (
        lambda text: _old_cb_banked_books_strict_json_loads(text, where="book"),
        lambda text: cb_banked_books._strict_json_loads(text, where="book"),
    ),
    "cb_learned_bundle": (
        lambda text: _old_cb_learned_bundle_strict_json_loads(text, where="bundle"),
        lambda text: cb_learned_bundle._strict_json_loads(text, where="bundle"),
    ),
    "cluster_campaign_contract": (
        _old_parse_campaign_manifest,
        cluster_campaign_contract.parse_campaign_manifest,
    ),
    "nvfp4_cb_footprint": (
        lambda text: _old_nvfp4_strict_json_loads(text, where="footprint"),
        lambda text: nvfp4_cb_footprint._strict_json_loads(text, where="footprint"),
    ),
    "quality_prefill_contract": (
        lambda text: _old_decode_strict_json(text, where="manifest"),
        lambda text: quality_prefill_contract.decode_strict_json(text, where="manifest"),
    ),
    "quality_prefill_pb_adapter": (
        lambda text: _old_load_strict_json(text, where="adapter"),
        lambda text: quality_prefill_pb_adapter.load_strict_json(text, where="adapter"),
    ),
    "runtime_provenance": (
        lambda text: _old_runtime_provenance_strict_json(text, "/p/m.json", "manifest"),
        lambda text: runtime_provenance._strict_json(text, "/p/m.json", "manifest"),
    ),
    "sample_parallel_probe": (
        lambda text: _old_sample_parallel_probe_strict_json_loads(text, where="probe"),
        lambda text: sample_parallel_probe._strict_json_loads(text, where="probe"),
    ),
    "tessera_campaign": (
        _old_parse_family_restriction,
        tessera_campaign.parse_family_restriction,
    ),
}

BYTE_LOADERS = {
    "joint_cost_read_schedule": (
        _old_decode_unique,
        joint_cost_read_schedule._decode_unique,
    ),
    "prismasnap_validation": (
        lambda data: _old_prismasnap_validation_json_from_bytes(data, where="evidence"),
        lambda data: prismasnap_validation._json_from_bytes(data, where="evidence"),
    ),
    "quality_prefill_pb_adapter": (
        lambda data: _old_load_strict_json(data, where="adapter"),
        lambda data: quality_prefill_pb_adapter.load_strict_json(data, where="adapter"),
    ),
    "runtime_provenance": (
        lambda data: _old_runtime_provenance_strict_json(data, "/p/m.json", "manifest"),
        lambda data: runtime_provenance._strict_json(data, "/p/m.json", "manifest"),
    ),
}

FILE_LOADERS = {
    "artifact_collection": (_old_load_record, artifact_collection.load_record),
    "artifact_collection_legacy": (
        lambda path: _old_load_json_source(path, logical_schema="s.v1"),
        lambda path: artifact_collection_legacy._load_json_source(path, logical_schema="s.v1"),
    ),
    "cb_learned_promotion": (
        _old_read_promotion_receipt_payload,
        cb_learned_promotion.read_promotion_receipt_payload,
    ),
    "measured_runtime_prices": (
        _old_measured_runtime_prices_json,
        measured_runtime_prices._json,
    ),
    "prismasnap_checkpoint": (
        lambda path: _old_prismasnap_checkpoint_load_json(path, where="checkpoint"),
        lambda path: prismasnap_checkpoint._load_json(path, where="checkpoint"),
    ),
}


@pytest.mark.parametrize("site", sorted(TEXT_LOADERS))
@pytest.mark.parametrize("document", DOCUMENTS + (None, 7))
def test_text_loader_refuses_as_before(site, document):
    old, new = TEXT_LOADERS[site]
    _same(lambda: old(document), lambda: new(document))


@pytest.mark.parametrize("site", sorted(BYTE_LOADERS))
@pytest.mark.parametrize("document", BYTE_DOCUMENTS)
def test_byte_loader_refuses_as_before(site, document):
    old, new = BYTE_LOADERS[site]
    _same(lambda: old(document), lambda: new(document))


@pytest.mark.parametrize("site", sorted(FILE_LOADERS))
@pytest.mark.parametrize("document", BYTE_DOCUMENTS + (None,))
def test_file_loader_refuses_as_before(site, document, tmp_path):
    path = tmp_path / "document.json"
    if document is not None:  # None: the file does not exist
        path.write_bytes(document)
    old, new = FILE_LOADERS[site]
    _same(lambda: old(path), lambda: new(path))


def test_every_refusal_case_is_exercised():
    """The loaders refuse each case the table claims, so no case is vacuous."""
    outcomes = {
        document: _outcome(lambda document=document: _old_decode_strict_json(document, where="m"))
        for document in DOCUMENTS
    }
    assert outcomes['{"a": 1, "b": [1, 2.5, "x"]}'][0] == "returned"
    assert "duplicate JSON member 'a'" in outcomes['{"a": 1, "a": 2}'][2]
    assert "duplicate JSON member 'k'" in outcomes['{"outer": {"k": 1, "k": 1}}'][2]
    for constant in ("NaN", "Infinity", "-Infinity"):
        assert outcomes['{"a": %s}' % constant][2] == f"m contains non-JSON constant {constant}"
    assert outcomes['{"a": 1'][2] == "m is not valid strict JSON"
    assert _outcome(lambda: joint_cost_read_schedule._decode_unique(b"\xff"))[2] == (
        "joint COST read schedule: invalid UTF-8 JSON")


# --- contract checks -----------------------------------------------------------


def _old_contract_family(error, path_component_re, sha256_re):
    """The cluster-campaign, quality-prefill and PB-adapter checks.

    Their copies are the same code apart from the exception class, docstrings
    and one local alias (``text = value`` in quality_prefill_contract._string).
    """

    def _fail(message: str) -> None:
        raise error(message)

    def _exact_mapping(value, *, keys, where):
        if not isinstance(value, Mapping):
            _fail(f"{where} must be an object")
        if any(type(key) is not str for key in value):
            _fail(f"{where} keys must be strings")
        actual = set(value)
        if actual != keys:
            missing = sorted(keys - actual)
            extra = sorted(actual - keys)
            _fail(f"{where} fields differ: missing={missing}, extra={extra}")
        return value

    def _string(value, *, where, pattern=None):
        if type(value) is not str or not value:
            _fail(f"{where} must be a non-empty string")
        if value != value.strip() or any(ord(char) < 32 for char in value):
            _fail(f"{where} contains whitespace padding or control characters")
        if pattern is not None and pattern.fullmatch(value) is None:
            _fail(f"{where} has an invalid value")
        return value

    def _integer(value, *, where, minimum, maximum=2**63 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            _fail(f"{where} must be an integer in [{minimum}, {maximum}]")
        return value

    def _sha256(value, *, where):
        return _string(value, where=where, pattern=sha256_re)

    def _absolute_posix_path(value, *, where):
        raw = _string(value, where=where)
        if not raw.startswith("/") or raw == "/":
            _fail(f"{where} must be a non-root absolute POSIX path")
        components = raw.split("/")[1:]
        if (
            not components
            or any(
                not component
                or component in {".", ".."}
                or path_component_re.fullmatch(component) is None
                for component in components
            )
            or str(PurePosixPath(raw)) != raw
        ):
            _fail(f"{where} must be normalized and traversal-free")
        return raw

    return {
        "_fail": _fail,
        "_exact_mapping": _exact_mapping,
        "_string": _string,
        "_integer": _integer,
        "_sha256": _sha256,
        "_absolute_posix_path": _absolute_posix_path,
    }


_OLD_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
_OLD_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_KEYS = frozenset({"a", "b"})

_ALL_SIX = {name: name for name in (
    "_fail", "_exact_mapping", "_string", "_integer", "_sha256", "_absolute_posix_path")}

# module -> (module, its exception, {old helper: the name the module binds it to}).
# The PB adapter moved only these two; its own _integer and _sha256 differ.
CONTRACT_MODULES = {
    "cluster_campaign_contract": (
        cluster_campaign_contract,
        cluster_campaign_contract.ClusterCampaignContractError,
        {**_ALL_SIX, "_absolute_posix_path": "_safe_absolute_path"},
    ),
    "quality_prefill_contract": (
        quality_prefill_contract,
        quality_prefill_contract.QualityPrefillContractError,
        _ALL_SIX,
    ),
    "quality_prefill_pb_adapter": (
        quality_prefill_pb_adapter,
        quality_prefill_pb_adapter.QualityPrefillAdapterError,
        {"_fail": "_fail", "_exact_mapping": "_exact_mapping"},
    ),
}

CONTRACT_CASES = (
    ("_fail", ("the message",), {}),
    ("_exact_mapping", ({"a": 1, "b": 2},), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", (MappingProxyType({"a": 1, "b": 2}),), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ([("a", 1)],), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", (None,), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ({1: 2, "a": 1},), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ({"a": 1},), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ({"a": 1, "b": 2, "c": 3, "d": 4},), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ({"c": 1, "b": 2},), {"keys": _KEYS, "where": "row"}),
    ("_exact_mapping", ({},), {"keys": frozenset(), "where": "row"}),
    ("_string", ("text",), {"where": "name"}),
    ("_string", ("",), {"where": "name"}),
    ("_string", (None,), {"where": "name"}),
    ("_string", (7,), {"where": "name"}),
    ("_string", (" text",), {"where": "name"}),
    ("_string", ("text\n",), {"where": "name"}),
    ("_string", ("te\x01xt",), {"where": "name"}),
    ("_string", ("host-1",), {"where": "name", "pattern": _ID_RE}),
    ("_string", ("Host",), {"where": "name", "pattern": _ID_RE}),
    ("_integer", (5,), {"where": "count", "minimum": 0}),
    ("_integer", (0,), {"where": "count", "minimum": 0}),
    ("_integer", (-1,), {"where": "count", "minimum": 0}),
    ("_integer", (True,), {"where": "count", "minimum": 0}),
    ("_integer", (1.0,), {"where": "count", "minimum": 0}),
    ("_integer", ("1",), {"where": "count", "minimum": 0}),
    ("_integer", (2**63,), {"where": "count", "minimum": 0}),
    ("_integer", (11,), {"where": "count", "minimum": 1, "maximum": 10}),
    ("_integer", (10,), {"where": "count", "minimum": 1, "maximum": 10}),
    ("_sha256", ("a" * 64,), {"where": "digest"}),
    ("_sha256", ("A" * 64,), {"where": "digest"}),
    ("_sha256", ("a" * 63,), {"where": "digest"}),
    ("_sha256", ("a" * 65,), {"where": "digest"}),
    ("_sha256", (b"a" * 64,), {"where": "digest"}),
    ("_sha256", (" " + "a" * 64,), {"where": "digest"}),
    ("_absolute_posix_path", ("/models/run.1/x_y-z",), {"where": "root"}),
    ("_absolute_posix_path", ("/",), {"where": "root"}),
    ("_absolute_posix_path", ("relative/path",), {"where": "root"}),
    ("_absolute_posix_path", ("/a/../b",), {"where": "root"}),
    ("_absolute_posix_path", ("/a/./b",), {"where": "root"}),
    ("_absolute_posix_path", ("/a//b",), {"where": "root"}),
    ("_absolute_posix_path", ("/a/b/",), {"where": "root"}),
    ("_absolute_posix_path", ("//a",), {"where": "root"}),
    ("_absolute_posix_path", ("/a b",), {"where": "root"}),
    ("_absolute_posix_path", ("/a$",), {"where": "root"}),
    ("_absolute_posix_path", ("",), {"where": "root"}),
    ("_absolute_posix_path", (None,), {"where": "root"}),
    ("_absolute_posix_path", ("/a\t",), {"where": "root"}),
)


def _contract_cases():
    for module_name, (_module, _error, helpers) in CONTRACT_MODULES.items():
        for name, args, kwargs in CONTRACT_CASES:
            if name in helpers:
                site = helpers[name]
                yield pytest.param(module_name, name, site, args, kwargs,
                                   id=f"{module_name}.{site}{args!r:.40}")


@pytest.mark.parametrize("module_name,name,site,args,kwargs", list(_contract_cases()))
def test_contract_check_refuses_as_before(module_name, name, site, args, kwargs):
    module, error, _aliases = CONTRACT_MODULES[module_name]
    old = _old_contract_family(error, _OLD_PATH_COMPONENT_RE, _OLD_SHA256_RE)[name]
    new = getattr(module, site)
    _same(lambda: old(*args, **kwargs), lambda: new(*args, **kwargs))


def test_contract_bindings_are_the_shared_checks():
    """Each module binds the shared check under the name its callers use."""
    for module, _error, helpers in CONTRACT_MODULES.values():
        for name, site in helpers.items():
            bound = getattr(module, site)
            assert bound.__func__ is getattr(schemas.Contract, name.removeprefix("_")), site
            assert bound.__self__.error is _error and bound.__self__.prefix == ""


def _old_require(error, prefix, *, concatenate):
    # joint_cost_read_schedule used f"{prefix}{message}"; the other prefixed
    # modules used prefix + message; the unprefixed ones raised error(message).
    def _require(condition, message):
        if not condition:
            raise error(prefix + message if concatenate else f"{prefix}{message}")
    return _require


REQUIRE_SITES = {
    "joint_catalog_extension": (
        joint_catalog_extension, _old_require(ValueError, "joint catalog extension: ", concatenate=True)),
    "joint_cost_read_schedule": (
        joint_cost_read_schedule, _old_require(ValueError, "joint COST read schedule: ", concatenate=False)),
    "joint_head_walk_quanta": (
        joint_head_walk_quanta, _old_require(ValueError, "", concatenate=True)),
    "joint_served_activation": (
        joint_served_activation, _old_require(ValueError, "served activation policy: ", concatenate=True)),
    "joint_stageb_resources": (
        joint_stageb_resources, _old_require(ValueError, "Stage B resources: ", concatenate=True)),
    "tessera_anchored_surface": (
        tessera_anchored_surface, _old_require(tessera_anchored_surface.ReplayError, "", concatenate=True)),
    "tessera_joint_aura": (
        tessera_joint_aura, _old_require(ValueError, "", concatenate=True)),
    "tessera_sampled_stack_proposal": (
        tessera_sampled_stack_proposal, _old_require(ValueError, "", concatenate=True)),
}


@pytest.mark.parametrize("site", sorted(REQUIRE_SITES))
@pytest.mark.parametrize("condition", (True, 1, "x", [0], False, 0, "", [], None))
@pytest.mark.parametrize("message", ("a refusal", "brace {kept} and 100%"))
def test_require_refuses_as_before(site, condition, message):
    module, old = REQUIRE_SITES[site]
    _same(lambda: old(condition, message), lambda: module._require(condition, message))


FAIL_SITES = {
    "native_runtime_cohort": (
        native_runtime_cohort, native_runtime_cohort.RuntimePriceError, "native runtime cohort: "),
    "quality_prefill_coverage": (
        quality_prefill_coverage, quality_prefill_coverage.CoverageError, ""),
    "quality_prefill_knee": (quality_prefill_knee, quality_prefill_knee.KneeSelectionError, ""),
    "quality_prefill_screen": (quality_prefill_screen, quality_prefill_screen.ScreenError, ""),
}


@pytest.mark.parametrize("site", sorted(FAIL_SITES))
def test_fail_refuses_as_before(site):
    module, error, prefix = FAIL_SITES[site]

    def old(message):
        raise error(prefix + message)

    outcome = _same(lambda: old("a refusal"), lambda: module._fail("a refusal"))
    assert outcome[0] == "raised" and outcome[1] is error


@pytest.mark.parametrize("value", ({}, {"a": 1}, MappingProxyType({}), [], None, "text"))
def test_promotion_receipt_mapping_refuses_as_before(value):
    _same(
        lambda: _old_cb_learned_promotion_mapping(value, where="receipt"),
        lambda: cb_learned_promotion._mapping(value, where="receipt"),
    )


def _old_legacy_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _old_artifact_collection_fail(where, "expected a JSON object")
    return value


def _old_legacy_exact_keys(value: Mapping[str, object], expected: set[str], *, where: str) -> None:
    if set(value) != expected:
        _old_artifact_collection_fail(
            where,
            "field set differs "
            f"(missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)})",
        )


@pytest.mark.parametrize("value", ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "c": 3}, {}, [], None))
@pytest.mark.parametrize("expected", ({"a", "b"}, frozenset({"a", "b"})))
def test_legacy_collection_checks_refuse_as_before(value, expected):
    _same(
        lambda: _old_legacy_mapping(value, where="legacy"),
        lambda: artifact_collection_legacy._mapping(value, where="legacy"),
    )
    if isinstance(value, Mapping):
        _same(
            lambda: _old_legacy_exact_keys(value, expected, where="legacy"),
            lambda: artifact_collection_legacy._exact_keys(value, expected, where="legacy"),
        )
    _same(
        lambda: _old_artifact_collection_fail("legacy", "text"),
        lambda: artifact_collection_legacy._fail("legacy", "text"),
    )


def _old_moe_exact_keys(value: Mapping[str, object], expected: frozenset[str], where: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise RuntimeError(
            f"{where} fields differ: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


@pytest.mark.parametrize("value", ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "c": 3}, {}))
def test_moe_checkpoint_exact_keys_refuse_as_before(value):
    assert prismasnap_moe_checkpoint._require_exact_keys is prismasnap_checkpoint._require_exact_keys
    _same(
        lambda: _old_moe_exact_keys(value, _KEYS, "PrismaSnap MoE plan"),
        lambda: prismasnap_moe_checkpoint._require_exact_keys(value, _KEYS, where="PrismaSnap MoE plan"),
    )
