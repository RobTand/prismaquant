"""Direct JSON recipe compatibility, using frozen *consumer* call expressions.

No expected JSON is obtained from the digest owner. The fixture preserves the
original call spelling and registry identity. References replace only its input
expression; native dump references bind the original stream name. Where the
registry has only one API, the companion call changes only dump/dumps and its
stream argument, retaining every original JSON keyword (including omissions).
"""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import digests


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/digest_1301/owner-recipes/json_recipes.json"
DOCUMENT = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECIPES = DOCUMENT["recipes"]
PROFILE_NAMES = (
    "DIRECT_ASCII_INDENT1_LAX",
    "DIRECT_ASCII_INDENT1_STRICT",
    "DIRECT_ASCII_INDENT2_STRICT",
    "DIRECT_ASCII_INDENT2_LAX_DEFAULT_STR",
    "DIRECT_ASCII_INDENT2_STRICT_DEFAULT_STR",
    "DIRECT_ASCII_SPACED_STRICT_DEFAULT_STR",
    "DIRECT_ASCII_SPACED_LAX_DEFAULT_REPR",
    "DIRECT_ASCII_SPACED_STRICT_DEFAULT_REPR",
    "DIRECT_ASCII_LAX_DEFAULT_REPR",
    "DIRECT_UTF8_SPACED_LAX",
    "INSERTION_ASCII_SPACED_LAX",
    "INSERTION_ASCII_SPACED_STRICT",
    "INSERTION_ASCII_SPACED_LAX_DEFAULT_STR",
    "INSERTION_UTF8_SPACED_LAX",
    "INSERTION_ASCII_LAX",
    "INSERTION_ASCII_STRICT",
    "INSERTION_ASCII_LAX_DEFAULT_STR",
    "INSERTION_UTF8_LAX",
    "INSERTION_ASCII_INDENT1_LAX",
    "INSERTION_ASCII_INDENT1_LAX_DEFAULT_STR",
    "INSERTION_ASCII_INDENT2_LAX",
    "INSERTION_ASCII_INDENT2_STRICT",
    "INSERTION_ASCII_INDENT2_LAX_DEFAULT_STR",
    "INSERTION_UTF8_INDENT2_STRICT",
)


class Fallback:
    def __str__(self):
        return "fallback-str: café 雪"

    def __repr__(self):
        return "fallback-repr: café 雪"


class WriteFailure(OSError):
    pass


class Trace:
    """Caller-owned stream: record write boundaries, not just joined text."""

    def __init__(self, fail_after=None):
        self.writes = []
        self.fail_after = fail_after
        self.flushes = 0
        self.closes = 0

    def write(self, chunk):
        assert isinstance(chunk, str)
        if self.fail_after is not None and len(self.writes) == self.fail_after:
            raise WriteFailure("caller stream refused write")
        self.writes.append(chunk)
        return len(chunk)

    def flush(self):
        self.flushes += 1
        raise AssertionError("encoder must not flush caller stream")

    def close(self):
        self.closes += 1
        raise AssertionError("encoder must not close caller stream")


def reference(recipe, value, operation="dumps", stream=None):
    row = recipe.get("dump_source", recipe["source"]) if operation == "dump" else recipe["source"]
    call = ast.parse(row["expression"], mode="eval").body
    original_keywords = ast.dump(ast.Module(body=copy.deepcopy(call.keywords), type_ignores=[]))
    native_operation = call.func.attr
    # Only the input expression is replaced in a native call. Never unparse or
    # reconstruct its options using a profile or a table of assumed defaults.
    call.args[0] = ast.Name(id="fixture_value", ctx=ast.Load())
    namespace = {"json": json, "str": str, "repr": repr, "fixture_value": value}
    if native_operation == "dump":
        assert isinstance(call.args[1], ast.Name)
        namespace[call.args[1].id] = stream
    if operation != native_operation:
        call.func.attr = operation
        if operation == "dump":
            call.args.append(ast.Name(id="fixture_stream", ctx=ast.Load()))
            namespace["fixture_stream"] = stream
        else:
            del call.args[1]
    assert ast.dump(ast.Module(body=call.keywords, type_ignores=[])) == original_keywords
    expression = ast.fix_missing_locations(ast.Expression(call))
    return eval(compile(expression, row["id"], "eval"), {"__builtins__": {}}, namespace)


def outcome(action):
    try:
        return ("return", action())
    except Exception as exc:
        # Compare native exception classes and payloads, not broad 'raises'.
        return ("raise", type(exc), exc.args)


def value_for(case):
    if case == "cycle-list":
        value = ["prefix"]
        value.append(value)
        return value
    if case == "cycle-dict":
        value = {"a": "prefix"}
        value["z"] = value
        return value
    values = {
        "unicode": {"z": "café 雪 😀", "a": ["\\\"\n\t", "e\u0301", "é"]},
        "order": {"z": 1, "a": 2, "m": {"y": 3, "b": 4}},
        "reverse-order": {"a": 2, "z": 1},
        "nan": ["prefix", float("nan")],
        "infinity": ["prefix", float("inf")],
        "negative-infinity": ["prefix", -float("inf")],
        "mixed-keys": {"z": 0, 2: "two", "a": 1},
        "integer-keys": {2: "two", 1: "one"},
        "bool-null-keys": {None: "null", False: "false", True: "true"},
        "key-collisions": {1: "number", "1": "string", None: "none", "null": "text"},
        "float-key-collisions": {-0.0: "float-zero", "-0.0": "text-zero"},
        "nonfinite-keys": {float("inf"): 1, float("nan"): 2},
        "invalid-key": {"a": 0, (1, 2): "bad"},
        "bytes": ["prefix", b"unsupported"],
        "set": ["prefix", {3, 1}],
        "fallback": {"a": "prefix", "z": Fallback()},
        "zero": [0, -0, 0.0, -0.0, 1, 1.0],
        "lone-high-surrogate": {"a": "prefix", "z": "\ud800"},
        "lone-low-surrogate": ["prefix", "\udfff"],
        "surrogate-key": {"\ud800": "value"},
        "first-error": [float("nan"), b"unsupported"],
        "empty": {"list": [], "dict": {}, "tuple": ()},
        "scalar": "café 雪 😀",
    }
    return values[case]


CASES = (
    "unicode", "order", "reverse-order", "nan", "infinity", "negative-infinity",
    "mixed-keys", "integer-keys", "bool-null-keys", "key-collisions",
    "float-key-collisions", "nonfinite-keys", "invalid-key", "bytes", "set",
    "fallback", "zero", "lone-high-surrogate", "lone-low-surrogate",
    "surrogate-key", "first-error", "empty", "scalar", "cycle-list", "cycle-dict",
)


def test_fixture_provenance_and_coverage():
    registry = json.loads((ROOT / DOCUMENT["registry"]).read_text(encoding="utf-8"))
    assert DOCUMENT["source_commit"] == registry["source_commit"]
    assert DOCUMENT["source_commit"] == "6d8288adc94b3c4b624c143ed372fb6513f32784"
    assert [row["profile"] for row in RECIPES] == list(PROFILE_NAMES)
    original = {row["id"]: row for row in registry["calls"]}
    dump_count = 0
    for recipe in RECIPES:
        for key in ("source", "dump_source"):
            if key not in recipe:
                continue
            row = recipe[key]
            assert {k: row[k] for k in original[row["id"]]} == original[row["id"]]
            call = ast.parse(row["expression"], mode="eval").body
            syntax = ast.dump(call, include_attributes=False).encode("utf-8")
            assert hashlib.sha256(syntax).hexdigest() == row["syntax_sha256"]
            assert ast.unparse(call.func) == row["callee"]
            assert len(row["source_blob"]) == 40
            if key == "dump_source":
                assert row["callee"] == "json.dump"
                dump_count += 1
    assert dump_count == 5


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda row: row["profile"])
@pytest.mark.parametrize("case", CASES)
def test_original_text_bytes_and_hash(recipe, case):
    profile = getattr(digests, recipe["profile"])
    value = value_for(case)
    assert outcome(lambda: profile.text(value)) == outcome(lambda: reference(recipe, value))
    assert outcome(lambda: profile.encoded(value)) == outcome(
        lambda: reference(recipe, value).encode("utf-8")
    )
    assert outcome(lambda: profile.sha256(value)) == outcome(
        lambda: hashlib.sha256(reference(recipe, value).encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda row: row["profile"])
@pytest.mark.parametrize("case", CASES)
def test_original_dump_trace_and_native_errors(recipe, case):
    profile = getattr(digests, recipe["profile"])
    value = value_for(case)
    expected, actual = Trace(), Trace()
    result = outcome(lambda: reference(recipe, value, "dump", expected))
    assert outcome(lambda: profile.dump(value, actual)) == result
    assert actual.writes == expected.writes
    assert (actual.flushes, actual.closes) == (0, 0)
    assert (expected.flushes, expected.closes) == (0, 0)
    if result[0] == "return":
        assert result[1] is None
        assert "".join(actual.writes) == reference(recipe, value)
        assert not "".join(actual.writes).endswith("\n")


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda row: row["profile"])
@pytest.mark.parametrize("fail_after", [0, 1, 4])
def test_dump_propagates_write_failure_without_cleanup(recipe, fail_after):
    value = value_for("unicode")
    expected, actual = Trace(fail_after), Trace(fail_after)
    result = outcome(lambda: reference(recipe, value, "dump", expected))
    assert result == ("raise", WriteFailure, ("caller stream refused write",))
    assert outcome(lambda: getattr(digests, recipe["profile"]).dump(value, actual)) == result
    assert actual.writes == expected.writes
    assert len(actual.writes) == fail_after
    assert (actual.flushes, actual.closes) == (0, 0)


def test_dump_and_dumps_keep_distinct_nonfinite_error_and_partial_writes():
    # Use an unindented original call to exercise dumps' fast encoder too.
    recipe = next(row for row in RECIPES if row["profile"] == "INSERTION_ASCII_STRICT")
    value = ["written first", "second prefix", float("nan"), b"later error"]
    expected, actual = Trace(), Trace()
    dumps_error = outcome(lambda: reference(recipe, value))
    dump_error = outcome(lambda: reference(recipe, value, "dump", expected))
    assert dumps_error[1] is dump_error[1] is ValueError
    # Compare each native message independently: some Python releases omit
    # the offending repr in dumps, whereas 3.14 includes it in both APIs.
    assert expected.writes and "written first" in "".join(expected.writes)
    profile = getattr(digests, recipe["profile"])
    assert outcome(lambda: profile.text(value)) == dumps_error
    assert outcome(lambda: profile.dump(value, actual)) == dump_error
    assert actual.writes == expected.writes

    # A write failure wins over the later invalid float in dump. dumps has no
    # stream, so it instead reaches the ValueError. A dumps-then-write shim
    # would produce the wrong exception and lose the already-written prefix.
    refusing_native, refusing_owner = Trace(fail_after=1), Trace(fail_after=1)
    early_error = outcome(lambda: reference(recipe, value, "dump", refusing_native))
    assert early_error == ("raise", WriteFailure, ("caller stream refused write",))
    assert early_error != dumps_error
    assert outcome(lambda: profile.dump(value, refusing_owner)) == early_error
    assert refusing_owner.writes == refusing_native.writes
    assert len(refusing_owner.writes) == 1
    assert (refusing_owner.flushes, refusing_owner.closes) == (0, 0)


@pytest.mark.parametrize("sorting", ["legacy-six", "seventh-false", "keyword-false"])
def test_json_profile_positional_compatibility(sorting):
    args = ("positional-compatibility", False, True, str, (", ", ": "), 2)
    if sorting == "legacy-six":
        profile = digests.JsonProfile(*args)
        sort_keys = True
    elif sorting == "seventh-false":
        profile = digests.JsonProfile(*args, False)
        sort_keys = False
    else:
        profile = digests.JsonProfile(*args, sort_keys=False)
        sort_keys = False
    value = {"z": Fallback(), "a": ["雪", -0.0]}
    # Independent stdlib oracle for constructor layout, not fixture generation.
    options = dict(ensure_ascii=False, allow_nan=True, default=str,
                   separators=(", ", ": "), indent=2, sort_keys=sort_keys)
    expected = json.dumps(value, **options)
    assert profile.sort_keys is sort_keys
    assert profile.text(value) == expected
    assert profile.encoded(value) == expected.encode("utf-8")
    assert profile.sha256(value) == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    native, actual = Trace(), Trace()
    assert json.dump(value, native, **options) is None
    assert profile.dump(value, actual) is None
    assert actual.writes == native.writes
    assert (actual.flushes, actual.closes) == (0, 0)


@pytest.mark.parametrize("behavior", ["structured", "raise", "cycle", "nan"])
@pytest.mark.parametrize("operation", ["text", "dump"])
def test_caller_supplied_custom_fallback(behavior, operation):
    # Original task-suite expression, not a substitute str/repr policy. The
    # actual lm_eval callable stays consumer-owned; these adapters exercise
    # the existing callable-default API without requiring that dependency.
    source = DOCUMENT["custom_fallback_source"]
    registry = json.loads((ROOT / DOCUMENT["registry"]).read_text())
    frozen = next(row for row in registry["calls"] if row["id"] == source["id"])
    assert source["source_blob"] == frozen["source_blob"]
    tree = ast.parse(source["expression"], mode="eval")
    assert hashlib.sha256(ast.dump(tree.body, include_attributes=False).encode()).hexdigest() == frozen["syntax_sha256"]
    value = {"z": ["prefix", Fallback()], "a": "雪"}

    def run(owner):
        events = []

        def callback(obj):
            events.append(type(obj).__name__)
            if behavior == "raise":
                raise LookupError("caller fallback refused")
            if behavior == "cycle":
                return obj
            if behavior == "nan":
                return float("nan")
            return {"z": "雪", "a": [3, 2]}

        stream = Trace()
        profile = digests.JsonProfile(
            "caller-supplied-fallback", True, False, default=callback,
            separators=(", ", ": "), sort_keys=False)
        if owner:
            result = outcome(lambda: profile.text(value) if operation == "text"
                             else profile.dump(value, stream))
        else:
            # Adapt only the original input and optional dump operation.
            call = ast.parse(source["expression"], mode="eval")
            call.body.args = [ast.Name(id="value", ctx=ast.Load())]
            if operation == "dump":
                call.body.func.attr = "dump"
                call.body.args.append(ast.Name(id="stream", ctx=ast.Load()))
            ast.fix_missing_locations(call)
            result = outcome(lambda: eval(compile(call, "original-task-suite", "eval"),
                {"json": json, "value": value, "stream": stream,
                 "handle_non_serializable": callback}))
        return result, events, stream.writes, stream.flushes, stream.closes

    assert run(True) == run(False)
