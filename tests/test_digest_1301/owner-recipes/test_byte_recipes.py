"""Native research-base byte recipes; frozen oracles never call digest owners.

Only small source snippets execute, with no historical checkout or consumer imports.
Policy resolution and canonical layer JSON stay caller-owned, as do scientific
seed interpretation and NVFP4 framing. This is not a consumer-migration claim.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import textwrap
from types import SimpleNamespace

import pytest

from prismaquant import digests as owner


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_FROZEN = json.loads(
    (_FIXTURES / "digest_1301/owner-recipes/byte_recipes.json").read_text()
)
_RECIPES = _FROZEN["recipes"]
_NATIVE_SHA256 = hashlib.sha256


def _namespace(recipe, **dependencies):
    namespace = {"hashlib": hashlib, "json": json, "struct": struct, "re": re,
                 **dependencies}
    dependency = recipe.get("dependency")
    sources = ([dependency["source"]] if dependency else []) + [recipe["source"]]
    for source in sources:
        exec(compile("from __future__ import annotations\n" + source,
                     recipe["file"], "exec"), namespace)
    return namespace


def _syntax(source):
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


def test_frozen_original_provenance():
    registry = json.loads((_FIXTURES / "digest_parent_completion_1301.json").read_text())
    assert _FROZEN["source_commit"] == registry["source_commit"]
    calls = {row["id"]: row for row in registry["calls"]}
    assert len(_RECIPES) == 5
    for recipe in _RECIPES.values():
        row = calls[recipe["id"]]
        for key in ("file", "scope", "callee", "source_blob", "syntax_sha256"):
            assert recipe[key] == row[key]
        syntax = _syntax(recipe["expression"])
        assert hashlib.sha256(syntax.encode()).hexdigest() == row["syntax_sha256"]
        tree = ast.parse(textwrap.dedent(recipe["source"]))
        assert any(ast.dump(node, include_attributes=False) == syntax
                   for node in ast.walk(tree) if isinstance(node, ast.Call))
    dependency = _RECIPES["priority"]["dependency"]
    assert dependency["file"] == "prismaquant/perturbed_x_cache.py"
    assert dependency["source_blob"] == "99ed093fdf903ecf0ca8e4023a8c71b252829bea"


@pytest.mark.parametrize("vector", _FROZEN["byte_vectors"], ids=["empty", "ascii", "all-bytes", "binary"])
@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview], ids=["bytes", "bytearray", "memoryview"])
def test_bytes_native_vectors(vector, wrap):
    data = wrap(bytes.fromhex(vector["hex"]))
    sha = owner.bytes_sha256digest(data)
    b8 = owner.bytes_blake2b8digest(data)
    b16 = owner.bytes_blake2b16hex(data)
    assert type(sha) is bytes and len(sha) == 32
    assert type(b8) is bytes and len(b8) == 8
    assert type(b16) is str and re.fullmatch("[0-9a-f]{32}", b16)
    assert sha.hex() == vector["sha256"]
    assert b8.hex() == vector["blake8"]
    assert b16 == vector["blake16"]
    # BLAKE2's digest size is an algorithm parameter, not post-hash truncation.
    full = hashlib.blake2b(data).digest()
    assert b8 != full[:8]
    assert bytes.fromhex(b16) != full[:16]
    assert b8 != bytes.fromhex(b16)[:8]


@pytest.mark.parametrize("case", _FROZEN["seed_cases"])
def test_scientific_seed_raw_sha_only(case):
    recipe = _RECIPES["scientific_seed"]
    native = eval(recipe["digest_expression"], {"hashlib": hashlib, "domain": case["domain"]})
    actual = owner.bytes_sha256digest(case["domain"].encode("ascii"))
    assert actual == native == bytes.fromhex(case["expected"])
    # Endian conversion and truncation are the scientific caller's decisions.
    assert int.from_bytes(actual[:8], "little") == int.from_bytes(native[:8], "little")


@pytest.mark.parametrize("case", _FROZEN["priority_cases"])
def test_activation_priority_full_original(case):
    recipe = _RECIPES["priority"]
    original = _namespace(recipe)["activation_priority_key"]
    adapted = dict(recipe, source=recipe["source"].replace(
        recipe["expression"] + ".digest()", "owner.bytes_blake2b8digest(material)"))
    route = _namespace(adapted, owner=owner)["activation_priority_key"]
    args = (case["calibration_hash"], case["name"])
    expected = tuple(case["expected"])
    assert original(*args, is_group=case["is_group"]) == expected
    assert route(*args, is_group=case["is_group"]) == expected


@pytest.mark.parametrize("calibration_hash", ["", "F" * 32, "a" * 31, "g" * 32])
def test_activation_priority_validation_stays_caller_owned(calibration_hash):
    recipe = _RECIPES["priority"]
    original = _namespace(recipe)["activation_priority_key"]
    adapted = dict(recipe, source=recipe["source"].replace(
        recipe["expression"] + ".digest()", "owner.bytes_blake2b8digest(material)"))
    route = _namespace(adapted, owner=owner)["activation_priority_key"]
    with pytest.raises(ValueError) as native:
        original(calibration_hash, "layer.q_proj")
    with pytest.raises(type(native.value), match=re.escape(str(native.value))):
        route(calibration_hash, "layer.q_proj")


@pytest.mark.parametrize("case", _FROZEN["calibration_cases"])
def test_calibration_exact_original_expression(case):
    calls = []
    config = object()

    def canonical_layer_config_json(value):
        assert value is config
        calls.append(value)
        return case["text"]

    namespace = {"hashlib": hashlib, "layer_config": config,
                 "canonical_layer_config_json": canonical_layer_config_json}
    native = eval(_RECIPES["calibration"]["digest_expression"], namespace)
    actual = owner.bytes_blake2b16hex(canonical_layer_config_json(config).encode("utf-8"))
    assert calls == [config, config]
    assert native == actual == case["expected"]


_PREFIX = b"prismaquant.joint_prepare.replay_read_order.v1\n"


def _rows(items):
    for kind, name, fmt in items:
        yield json.dumps((kind, name, fmt), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize("case", _FROZEN["framed_cases"])
def test_joint_replay_native_binary_existing_length_frames(case):
    original = _namespace(_RECIPES["framed"])["_framed"]
    native = original(case["items"])
    actual = bytes.fromhex(owner.length_framed_bytes_sha256(_rows(case["items"]), prefix=_PREFIX))
    assert type(native) is bytes and len(native) == 32
    assert actual == native == bytes.fromhex(case["expected"])


def test_joint_replay_frame_order_and_boundaries():
    original = _namespace(_RECIPES["framed"])["_framed"]
    items = _FROZEN["framed_cases"][1]["items"]
    assert original(items) != original(list(reversed(items)))
    frames = [b"a", b"bc"]
    alternate = [b"ab", b"c"]
    assert b"".join(frames) == b"".join(alternate)
    for sequence in (frames, alternate):
        native = _NATIVE_SHA256(_PREFIX)
        for frame in sequence:
            native.update(len(frame).to_bytes(8, "big"))
            native.update(frame)
        assert owner.length_framed_bytes_sha256(iter(sequence), prefix=_PREFIX) == native.hexdigest()
    assert owner.length_framed_bytes_sha256(frames, prefix=_PREFIX) != owner.length_framed_bytes_sha256(alternate, prefix=_PREFIX)


class _TraceSHA:
    """Record native calls without patching the process-global hashlib module."""
    def __init__(self, events):
        self.events = events

    def sha256(self, data=b""):
        self.events.append(("new", bytes(data)))
        native = _NATIVE_SHA256(data)
        events = self.events

        class Digest:
            def update(self, chunk):
                # Record before native validation so failed feeds are observable.
                events.append(("update", chunk))
                native.update(chunk)

            def hexdigest(self):
                events.append(("hexdigest",))
                return native.hexdigest()

        return Digest()


def _explicit_policy(policy):
    assert policy == "canonical-policy"  # No alias, environment or default emulation.
    return policy


def _target_chunks(scales, *, policy):
    policy = _explicit_policy(policy)
    for field in ("prismaquant.nvfp4_w4a4_activation.v1", policy):
        encoded = field.encode("utf-8")
        yield struct.pack("<I", len(encoded))
        yield encoded
    for target in sorted(scales):
        encoded = str(target).encode("utf-8")
        yield struct.pack("<I", len(encoded))
        yield encoded
        yield struct.pack("<f", float(scales[target]))


def _outcome(function):
    try:
        return ("ok", function())
    except Exception as error:
        return ("error", type(error), str(error))


@pytest.mark.parametrize("case", _FROZEN["target_cases"])
def test_nvfp4_original_chunks_exact_transcript(case, monkeypatch):
    original_events, route_events = [], []
    original = _namespace(_RECIPES["targets"], hashlib=_TraceSHA(original_events),
                          resolve_input_global_scale_policy=_explicit_policy)["target_values_sha256"]
    monkeypatch.setattr(owner, "hashlib", _TraceSHA(route_events))
    assert original(case["scales"], policy=case["policy"]) == case["expected"]
    assert owner.chunks_sha256hex(_target_chunks(case["scales"], policy=case["policy"])) == case["expected"]
    assert route_events == original_events


@pytest.mark.parametrize("bad", ["not-a-float", 1e100, object()])
def test_nvfp4_conversion_errors_preserve_partial_feed(bad, monkeypatch):
    original_events, route_events = [], []
    original = _namespace(_RECIPES["targets"], hashlib=_TraceSHA(original_events),
                          resolve_input_global_scale_policy=_explicit_policy)["target_values_sha256"]
    monkeypatch.setattr(owner, "hashlib", _TraceSHA(route_events))
    scales = {"a": 1.0, "z": bad}
    native = _outcome(lambda: original(scales, policy="canonical-policy"))
    route = _outcome(lambda: owner.chunks_sha256hex(_target_chunks(scales, policy="canonical-policy")))
    assert native[0] == "error"
    assert route == native
    assert route_events == original_events
    assert not any(event[0] == "hexdigest" for event in route_events)


def _native_chunks(chunks, sha):
    digest = sha.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("failure", [None, "generator", "invalid-chunk"])
def test_chunks_one_pass_update_before_next_and_errors(failure, monkeypatch):
    def run(route):
        events = []
        sha = _TraceSHA(events)

        class OnePass:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                assert self.iterations == 1
                for chunk in (b"", bytearray(b"ab"), memoryview(b"\x00\xff")):
                    events.append(("yield", bytes(chunk)))
                    yield chunk
                    assert events[-1] == ("update", chunk)  # No look-ahead/materialization.
                if failure == "generator":
                    events.append(("raise",))
                    raise RuntimeError("source failed after partial input")
                if failure == "invalid-chunk":
                    events.append(("yield-invalid",))
                    yield "not bytes"
                    pytest.fail("invalid update must stop before requesting next chunk")

        chunks = OnePass()
        if route:
            monkeypatch.setattr(owner, "hashlib", sha)
            outcome = _outcome(lambda: owner.chunks_sha256hex(chunks))
        else:
            outcome = _outcome(lambda: _native_chunks(chunks, sha))
        assert chunks.iterations == 1
        return outcome, events

    native = run(False)
    actual = run(True)
    assert actual == native
    assert actual[0][0] == ("ok" if failure is None else "error")


@pytest.mark.parametrize("chunks", [[], [b""], [b"a", b"bc"], [b"ab", b"c"]])
def test_chunks_empty_and_unframed_boundaries(chunks):
    assert owner.chunks_sha256hex(iter(chunks)) == _NATIVE_SHA256(b"".join(chunks)).hexdigest()


@pytest.mark.parametrize("function", [owner.bytes_sha256digest, owner.bytes_blake2b8digest,
                                      owner.bytes_blake2b16hex])
@pytest.mark.parametrize("data", ["text", None, 123])
def test_byte_functions_reject_non_buffers(function, data):
    with pytest.raises(TypeError):
        function(data)
