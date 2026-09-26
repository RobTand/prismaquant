"""The shared contract checks refuse exactly as each module's own copy did (PQ #1300).

PQ #1300 moved the strict-JSON loaders and the ``_require`` / ``_fail`` /
``_exact_mapping`` / ``_string`` / ``_integer`` / ``_sha256`` / path checks of
many modules onto ``prismaquant.schemas`` (``strict_json_loads`` and
``Contract``). The migrating PR ran each module's pre-#1300 code next to the
migrated site on every input below and required the same outcome. Those
outcomes are frozen in ``fixtures/contract_checks_1300.json`` (PQ #1330), so
each case now checks the migrated site against its frozen row: the same
returned value, or the same exception type, text and chained cause.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from types import MappingProxyType

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
from tests.golden_table import GoldenTable, outcome


GOLDEN = GoldenTable("contract_checks_1300")


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


TEXT_LOADERS = {
    "cb_banked_books": lambda text: cb_banked_books._strict_json_loads(text, where="book"),
    "cb_learned_bundle": lambda text: cb_learned_bundle._strict_json_loads(text, where="bundle"),
    "cluster_campaign_contract": cluster_campaign_contract.parse_campaign_manifest,
    "nvfp4_cb_footprint": lambda text: nvfp4_cb_footprint._strict_json_loads(text, where="footprint"),
    "quality_prefill_contract": lambda text: quality_prefill_contract.decode_strict_json(text, where="manifest"),
    "quality_prefill_pb_adapter": lambda text: quality_prefill_pb_adapter.load_strict_json(text, where="adapter"),
    "runtime_provenance": lambda text: runtime_provenance._strict_json(text, "/p/m.json", "manifest"),
    "sample_parallel_probe": lambda text: sample_parallel_probe._strict_json_loads(text, where="probe"),
    "tessera_campaign": tessera_campaign.parse_family_restriction,
}

BYTE_LOADERS = {
    "joint_cost_read_schedule": joint_cost_read_schedule._decode_unique,
    "prismasnap_validation": lambda data: prismasnap_validation._json_from_bytes(data, where="evidence"),
    "quality_prefill_pb_adapter": lambda data: quality_prefill_pb_adapter.load_strict_json(data, where="adapter"),
    "runtime_provenance": lambda data: runtime_provenance._strict_json(data, "/p/m.json", "manifest"),
}

FILE_LOADERS = {
    "artifact_collection": artifact_collection.load_record,
    "artifact_collection_legacy": lambda path: artifact_collection_legacy._load_json_source(path, logical_schema="s.v1"),
    "cb_learned_promotion": cb_learned_promotion.read_promotion_receipt_payload,
    "measured_runtime_prices": measured_runtime_prices._json,
    "prismasnap_checkpoint": lambda path: prismasnap_checkpoint._load_json(path, where="checkpoint"),
}


@pytest.mark.parametrize("site", sorted(TEXT_LOADERS))
@pytest.mark.parametrize("document", DOCUMENTS + (None, 7))
def test_text_loader_refuses_as_before(site, document):
    GOLDEN.call(lambda: TEXT_LOADERS[site](document))


@pytest.mark.parametrize("site", sorted(BYTE_LOADERS))
@pytest.mark.parametrize("document", BYTE_DOCUMENTS)
def test_byte_loader_refuses_as_before(site, document):
    GOLDEN.call(lambda: BYTE_LOADERS[site](document))


@pytest.mark.parametrize("site", sorted(FILE_LOADERS))
@pytest.mark.parametrize("document", BYTE_DOCUMENTS + (None,))
def test_file_loader_refuses_as_before(site, document, tmp_path):
    path = tmp_path / "document.json"
    if document is not None:  # None: the file does not exist
        path.write_bytes(document)
    GOLDEN.call(lambda: FILE_LOADERS[site](path), tmp=tmp_path)


def test_every_refusal_case_is_exercised():
    """The loaders refuse each case the table claims, so no case is vacuous."""
    outcomes = {
        document: outcome(lambda document=document: quality_prefill_contract.decode_strict_json(
            document, where="m"))
        for document in DOCUMENTS
    }
    assert "returned" in outcomes['{"a": 1, "b": [1, 2.5, "x"]}']
    assert "duplicate JSON member 'a'" in outcomes['{"a": 1, "a": 2}']["text"]
    assert "duplicate JSON member 'k'" in outcomes['{"outer": {"k": 1, "k": 1}}']["text"]
    for constant in ("NaN", "Infinity", "-Infinity"):
        assert outcomes['{"a": %s}' % constant]["text"] == f"m contains non-JSON constant {constant}"
    assert outcomes['{"a": 1']["text"] == "m is not valid strict JSON"
    assert outcome(lambda: joint_cost_read_schedule._decode_unique(b"\xff"))["text"] == (
        "joint COST read schedule: invalid UTF-8 JSON")


# --- contract checks -----------------------------------------------------------


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
    module = CONTRACT_MODULES[module_name][0]
    GOLDEN.call(lambda: getattr(module, site)(*args, **kwargs))


def test_contract_bindings_are_the_shared_checks():
    """Each module binds the shared check under the name its callers use."""
    for module, _error, helpers in CONTRACT_MODULES.values():
        for name, site in helpers.items():
            bound = getattr(module, site)
            assert bound.__func__ is getattr(schemas.Contract, name.removeprefix("_")), site
            assert bound.__self__.error is _error and bound.__self__.prefix == ""


REQUIRE_SITES = {
    "joint_catalog_extension": joint_catalog_extension,
    "joint_cost_read_schedule": joint_cost_read_schedule,
    "joint_head_walk_quanta": joint_head_walk_quanta,
    "joint_served_activation": joint_served_activation,
    "joint_stageb_resources": joint_stageb_resources,
    "tessera_anchored_surface": tessera_anchored_surface,
    "tessera_joint_aura": tessera_joint_aura,
    "tessera_sampled_stack_proposal": tessera_sampled_stack_proposal,
}


@pytest.mark.parametrize("site", sorted(REQUIRE_SITES))
@pytest.mark.parametrize("condition", (True, 1, "x", [0], False, 0, "", [], None))
@pytest.mark.parametrize("message", ("a refusal", "brace {kept} and 100%"))
def test_require_refuses_as_before(site, condition, message):
    GOLDEN.call(lambda: REQUIRE_SITES[site]._require(condition, message))


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
    row = GOLDEN.call(lambda: module._fail("a refusal"))
    assert row["raised"] == f"{error.__module__}.{error.__qualname__}"
    assert row["text"] == prefix + "a refusal"


@pytest.mark.parametrize("value", ({}, {"a": 1}, MappingProxyType({}), [], None, "text"))
def test_promotion_receipt_mapping_refuses_as_before(value):
    GOLDEN.call(lambda: cb_learned_promotion._mapping(value, where="receipt"))


@pytest.mark.parametrize("value", ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "c": 3}, {}, [], None))
@pytest.mark.parametrize("expected", ({"a", "b"}, frozenset({"a", "b"})))
def test_legacy_collection_checks_refuse_as_before(value, expected):
    GOLDEN.call(lambda: artifact_collection_legacy._mapping(value, where="legacy"))
    if isinstance(value, Mapping):
        GOLDEN.call(lambda: artifact_collection_legacy._exact_keys(value, expected, where="legacy"))
    GOLDEN.call(lambda: artifact_collection_legacy._fail("legacy", "text"))


@pytest.mark.parametrize("value", ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "c": 3}, {}))
def test_moe_checkpoint_exact_keys_refuse_as_before(value):
    assert prismasnap_moe_checkpoint._require_exact_keys is prismasnap_checkpoint._require_exact_keys
    GOLDEN.call(lambda: prismasnap_moe_checkpoint._require_exact_keys(
        value, _KEYS, where="PrismaSnap MoE plan"))
