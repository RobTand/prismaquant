"""Consumer adapter: bind a serving-backed task config to the Tessera public witness.

Tessera owns the producer contract (issue 1056 closed by PR 1089, completed on
master). It publishes one versioned public witness with schema
``tessera.endpoint_runtime_witness.v1`` and the standalone CPU verifier CLI
``tools/verify_endpoint_witness.py`` (``--offline`` returns a machine-readable
verdict without a live endpoint). PrismaQuant consumes that public contract
with the standard library only. This module never imports the Tessera serving
runtime, starts no rank, and adds no identity seal. Every comparison below
refuses in both dev and certified modes with plain equality.

Pinned producer contract (Tessera master ``b2875875a4``; witness semantics and
CLI last changed in ``f34305799d``):

- witness schema ``tessera.endpoint_runtime_witness.v1``
- byte coverage kind ``successful-loader-inputs-and-post-load-resident-state``
- qualification scope ``runtime_byte_binding``
- offline proof scope ``recorded_runtime_byte_binding``

A served task config names ``backend.name == "served"`` and carries the
producer witness path, the expectation document path, and the explicit
expected endpoint, served alias, launch attempt, and rank list. The adapter
rechecks the witness self-join (attempt, ranks, coverage, scope, fingerprint)
and the expected-fact equality, then records the verdict. Alias-only or
size-only evidence refuses: file digests, byte counts, tokenizer bytes, and
the fingerprint are all required. A passed check proves the recorded launch,
never current endpoint state. HF configs pass through unchanged elsewhere;
this adapter changes no HF behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: Backend name of a serving-backed task config.
SERVED_BACKEND = "served"

#: HF backend name. The adapter passes HF configs through unchanged.
HF_BACKEND = "hf"

#: The one producer witness schema this adapter reads.
TESSERA_WITNESS_SCHEMA = "tessera.endpoint_runtime_witness.v1"

#: Byte coverage kind the producer stamps on complete loader-input coverage.
TESSERA_COVERAGE_KIND = "successful-loader-inputs-and-post-load-resident-state"

#: Qualification scope a witness may claim. Anything else refuses.
TESSERA_QUALIFICATION_SCOPE = "runtime_byte_binding"

#: Proof scope a passed offline check reports.
PROOF_SCOPE_RECORDED = "recorded_runtime_byte_binding"

#: Tokenizer source files the producer may report.
TOKENIZER_NAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json")

#: Tokenizer special-ID labels the witness joins.
SPECIAL_ID_LABELS = ("bos", "eos", "pad", "unk", "sep", "cls", "mask")

#: Payload bytes per element for tensor byte-bound checks.
DTYPE_BYTES = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4,
               "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4,
               "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}

#: Machine-readable verdicts this adapter returns.
VERDICT_PASS = "pass"
VERDICT_REFUSE = "refuse"


class _Refusal(Exception):
    """One refused witness join or expected fact."""


def _refuse(reason: str) -> _Refusal:
    return _Refusal(str(reason))


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise _refuse(reason)


def _fields(value: object, names: tuple[str, ...], where: str) -> None:
    _require(isinstance(value, dict), f"{where} is not an object")
    _require(set(value) == set(names),
             f"{where} fields differ from {sorted(names)}")


def _text(value: object, where: str) -> None:
    _require(isinstance(value, str) and bool(value),
             f"{where} is not non-empty text")


def _sha(value: object, where: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value),
             f"{where} is not sha256")


def _number(value: object, where: str) -> None:
    _require(type(value) in (float, int) and math.isfinite(value)
             and value > 0, f"{where} is not a positive finite time")


def _relative(name: object) -> None:
    _text(name, "file name")
    assert isinstance(name, str)
    _require(not name.startswith("/") and "\\" not in name
             and all(p not in ("", ".", "..") for p in name.split("/")),
             f"file name {name!r} is not relative")


def canonical(value: Any) -> str:
    """Canonical JSON text: sorted keys, compact separators, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def fingerprint(receipt: Mapping[str, Any]) -> str:
    """The witness fingerprint: SHA-256 over the receipt without itself."""
    return hashlib.sha256(canonical(
        {k: v for k, v in receipt.items() if k != "fingerprint"}).encode(
        "utf-8")).hexdigest()


def _owner(value: object, where: str) -> None:
    _fields(value, ("host", "boot_id", "pid", "start_ticks", "started_unix"),
            where)
    assert isinstance(value, dict)
    _text(value["host"], f"{where} host")
    _text(value["boot_id"], f"{where} boot_id")
    for key in ("pid", "start_ticks"):
        _require(type(value[key]) is int and value[key] > 0,
                 f"{where} {key} is not positive")
    _number(value["started_unix"], f"{where} started_unix")


def _attempt_id(owner_value: Mapping[str, Any]) -> str:
    return (f"{owner_value['host']}/{owner_value['boot_id']}/"
            f"{owner_value['pid']}/{owner_value['start_ticks']}")


def _tokenizer_vocab(backend: object) -> dict[str, int]:
    _require(isinstance(backend, dict), "tokenizer backend is not an object")
    assert isinstance(backend, dict)
    model = backend.get("model")
    _require(isinstance(model, dict), "tokenizer backend has no model")
    assert isinstance(model, dict)
    vocab = model.get("vocab")
    if isinstance(vocab, list):
        vocab = {item[0]: i for i, item in enumerate(vocab)}
    _require(isinstance(vocab, dict) and bool(vocab),
             "tokenizer backend has no vocabulary")
    assert isinstance(vocab, dict)
    vocab = dict(vocab)
    for item in backend.get("added_tokens", []):
        _require(isinstance(item, dict),
                 "tokenizer added token is not an object")
        vocab[item["content"]] = item["id"]
    _require(all(isinstance(k, str) and type(v) is int and v >= 0
                 for k, v in vocab.items()),
             "tokenizer vocabulary has an invalid token ID")
    _require(len(set(vocab.values())) == len(vocab),
             "tokenizer vocabulary repeats an ID")
    return vocab


def _file_fact(value: object, where: str) -> None:
    _fields(value, ("sha256", "bytes", "dtype", "shape"), where)
    assert isinstance(value, dict)
    _sha(value["sha256"], where)
    _require(type(value["bytes"]) is int and value["bytes"] > 0,
             f"{where} bytes is not positive")
    _text(value["dtype"], f"{where} dtype")
    _require(isinstance(value["shape"], list)
             and all(type(n) is int and n > 0 for n in value["shape"])
             and value["bytes"] % math.prod(value["shape"]) == 0,
             f"{where} shape has invalid byte bounds")


def _check_join(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the producer self-join. Return the agreed artifact files."""
    _fields(receipt, ("schema", "listener", "launch", "lifetime",
                      "artifacts", "tokenizer", "byte_coverage",
                      "qualification_scope", "fingerprint"), "receipt")
    _require(receipt["schema"] == TESSERA_WITNESS_SCHEMA,
             "receipt schema is not supported")
    listener = receipt["listener"]
    _fields(listener, ("endpoint", "served_alias", "owner"), "listener")
    _owner(listener["owner"], "process owner")
    _text(listener["served_alias"], "listener alias")
    parsed = urlsplit(listener["endpoint"])
    _require(parsed.scheme in ("http", "https") and parsed.hostname
             and parsed.port and parsed.path in ("", "/")
             and not parsed.username and not parsed.query
             and not parsed.fragment,
             "listener endpoint is not a direct HTTP address")
    launch = receipt["launch"]
    _fields(launch, ("attempt_id", "ranks"), "launch")
    _require(launch["attempt_id"] == _attempt_id(listener["owner"]),
             "launch attempt differs from the observed listener process")
    lifetime = receipt["lifetime"]
    _fields(lifetime, ("request_id", "started_unix", "finished_unix"),
            "lifetime")
    _text(lifetime["request_id"], "lifetime request_id")
    _number(lifetime["started_unix"], "lifetime started_unix")
    _number(lifetime["finished_unix"], "lifetime finished_unix")
    _require(listener["owner"]["started_unix"] <= lifetime["started_unix"]
             <= lifetime["finished_unix"],
             "listener observation is outside its process lifetime")
    ranks = launch["ranks"]
    _require(isinstance(ranks, list) and ranks
             and all(type(r) is int for r in ranks)
             and ranks == list(range(len(ranks))),
             "launch ranks do not cover the complete world")
    artifacts = receipt["artifacts"]
    _require(isinstance(artifacts, list) and len(artifacts) == len(ranks),
             "artifact ranks are incomplete")
    _require([a["rank"] for a in artifacts] == ranks,
             "artifact ranks differ from launch ranks")
    common_files: dict[str, Any] | None = None
    consumed: dict[tuple[str, str], list[tuple[int, int]]] = {}
    process_owners: set[str] = set()
    for rank in artifacts:
        _fields(rank, ("rank", "world_size", "owner", "request_id",
                       "observed_unix", "models"), "rank")
        _require(type(rank["rank"]) is int and type(rank["world_size"]) is int
                 and rank["world_size"] == len(ranks),
                 "rank world size is inconsistent")
        _owner(rank["owner"], "process owner")
        process_key = _attempt_id(rank["owner"])
        _require(process_key not in process_owners,
                 "two ranks name the same worker process")
        process_owners.add(process_key)
        _require(rank["request_id"] == lifetime["request_id"],
                 "rank observation mixes runtime requests")
        _number(rank["observed_unix"], "rank observed_unix")
        _require(lifetime["started_unix"] <= rank["observed_unix"]
                 <= lifetime["finished_unix"],
                 "rank observation is outside the runtime observation lifetime")
        _require(isinstance(rank["models"], list) and rank["models"],
                 "rank has no loaded model")
        rank_files: dict[str, Any] = {}
        for model in rank["models"]:
            _fields(model, ("model_path", "load_started_unix",
                            "load_finished_unix", "files", "inputs",
                            "resident"), "loaded model")
            _text(model["model_path"], "loaded model path")
            _number(model["load_started_unix"], "model load start")
            _number(model["load_finished_unix"], "model load finish")
            _require(rank["owner"]["started_unix"]
                     <= model["load_started_unix"]
                     <= model["load_finished_unix"]
                     <= lifetime["started_unix"],
                     "loaded byte observation is outside the worker process "
                     "lifetime")
            _require(isinstance(model["files"], dict) and model["files"],
                     "loaded model has no source files")
            for name, source in model["files"].items():
                _relative(name)
                _fields(source, ("sha256", "bytes", "data_start", "tensors"),
                        "loaded source")
                _sha(source["sha256"], "loaded source digest")
                _require(type(source["bytes"]) is int
                         and type(source["data_start"]) is int
                         and 8 < source["data_start"] < source["bytes"],
                         "source file has invalid byte bounds")
                _require(isinstance(source["tensors"], dict)
                         and source["tensors"],
                         "source file has no tensor roster")
                ranges = []
                for tensor_name, descriptor in source["tensors"].items():
                    _text(tensor_name, "source tensor name")
                    _fields(descriptor, ("dtype", "shape", "data_offsets"),
                            "source tensor")
                    _text(descriptor["dtype"], "source dtype")
                    _require(isinstance(descriptor["shape"], list) and all(
                        type(n) is int and n >= 0
                        for n in descriptor["shape"]),
                        "source tensor shape is invalid")
                    bounds = descriptor["data_offsets"]
                    _require(isinstance(bounds, list) and len(bounds) == 2
                             and all(type(n) is int for n in bounds)
                             and 0 <= bounds[0] <= bounds[1]
                             <= source["bytes"] - source["data_start"],
                             "source tensor offsets are invalid")
                    _require(descriptor["dtype"] in DTYPE_BYTES
                             and math.prod(descriptor["shape"])
                             * DTYPE_BYTES[descriptor["dtype"]]
                             == bounds[1] - bounds[0],
                             "source dtype or shape differs from byte bounds")
                    ranges.append(bounds)
                cursor = 0
                for start, end in sorted(ranges):
                    _require(start == cursor, "source tensor roster has "
                             "incomplete byte coverage")
                    cursor = end
                _require(cursor + source["data_start"] == source["bytes"],
                         "source tensor roster omits file bytes")
                _require(name not in rank_files
                         or rank_files[name] == source,
                         "rank sources disagree on file bytes")
                rank_files[name] = source
            _require(isinstance(model["resident"], dict)
                     and model["resident"],
                     "model has no resident byte observations")
            for name, fact in model["resident"].items():
                _text(name, "resident tensor name")
                _file_fact(fact, "resident tensor")
            _require(isinstance(model["inputs"], list) and model["inputs"],
                     "model has no successful loader inputs")
            for item in model["inputs"]:
                _fields(item, ("file", "tensor", "start", "end", "sha256",
                               "source_sha256", "target", "loaded_unix"),
                        "loader input")
                _require(item["file"] in model["files"],
                         "loader input names an unrelated source file")
                source = model["files"][item["file"]]
                _require(item["tensor"] in source["tensors"],
                         "loader input names an unrelated source tensor")
                start, end = source["tensors"][item["tensor"]]["data_offsets"]
                _require(type(item["start"]) is int
                         and type(item["end"]) is int
                         and start <= item["start"] < item["end"] <= end,
                         "loader input has invalid byte bounds")
                _text(item["target"], "loader target")
                _sha(item["sha256"], "loader input digest")
                _require(item["sha256"] == item["source_sha256"],
                         "loaded input bytes differ from source bytes")
                _number(item["loaded_unix"], "loaded input time")
                _require(model["load_started_unix"] <= item["loaded_unix"]
                         <= model["load_finished_unix"],
                         "loader input is outside its load lifetime")
                consumed.setdefault((item["file"], item["tensor"]),
                                    []).append((item["start"], item["end"]))
        if common_files is None:
            common_files = rank_files
        _require(rank_files == common_files,
                 "serving ranks disagree on loaded artifact files")
    assert common_files is not None
    payload_bytes = 0
    for name, source in common_files.items():
        for tensor_name, descriptor in source["tensors"].items():
            start, end = descriptor["data_offsets"]
            cursor = start
            for left, right in sorted(consumed.get((name, tensor_name), [])):
                _require(left <= cursor, "incomplete loaded byte coverage "
                         f"for {name}:{tensor_name}")
                cursor = max(cursor, right)
            _require(cursor == end, "incomplete loaded byte coverage "
                     f"for {name}:{tensor_name}")
            payload_bytes += end - start
    _require(receipt["byte_coverage"] == {"kind": TESSERA_COVERAGE_KIND,
                                          "files": sorted(common_files),
                                          "tensor_payload_bytes":
                                          payload_bytes},
             "byte_coverage differs from observed loader inputs")
    tokenizer = receipt["tokenizer"]
    _fields(tokenizer, ("path", "request_id", "observed_unix", "files",
                        "backend", "vocab", "special_ids"), "tokenizer")
    _text(tokenizer["path"], "server tokenizer path")
    _require(tokenizer["request_id"] == lifetime["request_id"],
             "tokenizer observation mixes runtime requests")
    _number(tokenizer["observed_unix"], "tokenizer observed_unix")
    _require(lifetime["started_unix"] <= tokenizer["observed_unix"]
             <= lifetime["finished_unix"],
             "tokenizer observation is outside the runtime observation "
             "lifetime")
    _require(isinstance(tokenizer["files"], dict)
             and "tokenizer.json" in tokenizer["files"],
             "tokenizer byte evidence is incomplete")
    for name, fact in tokenizer["files"].items():
        _require(name in TOKENIZER_NAMES,
                 "tokenizer source file is not supported")
        _fields(fact, ("sha256", "bytes", "content"), "tokenizer source")
        _sha(fact["sha256"], "tokenizer source digest")
        _require(type(fact["bytes"]) is int and fact["bytes"] > 0,
                 "tokenizer source size is invalid")
    _require(tokenizer["backend"]
             == tokenizer["files"]["tokenizer.json"]["content"],
             "loaded tokenizer backend differs from tokenizer bytes")
    _require(tokenizer["vocab"] == _tokenizer_vocab(tokenizer["backend"]),
             "loaded tokenizer mapping differs from tokenizer bytes")
    config = tokenizer["files"].get("tokenizer_config.json", {}).get(
        "content", {})
    special_map = tokenizer["files"].get("special_tokens_map.json", {}).get(
        "content", {})
    _fields(tokenizer["special_ids"], SPECIAL_ID_LABELS,
            "tokenizer special IDs")
    for label, value in tokenizer["special_ids"].items():
        if value is None:
            continue
        declared = special_map.get(label + "_token",
                                   config.get(label + "_token"))
        if isinstance(declared, dict):
            declared = declared.get("content")
        _require(type(value) is int and isinstance(declared, str)
                 and tokenizer["vocab"].get(declared) == value,
                 "loaded tokenizer special ID differs from tokenizer bytes")
    _require(receipt["qualification_scope"] == TESSERA_QUALIFICATION_SCOPE,
             "receipt overstates its qualification scope")
    _require(receipt["fingerprint"] == fingerprint(receipt),
             "receipt fingerprint differs from its bytes")
    return common_files


def _check_expected_files(expected: object, observed: Mapping[str, Any],
                           where: str) -> None:
    _require(isinstance(expected, dict) and expected,
             f"{where} has no file bytes")
    assert isinstance(expected, dict)
    for name, fact in expected.items():
        _relative(name)
        _fields(fact, ("sha256", "bytes"), f"{where} file {name}")
        assert isinstance(fact, dict)
        _sha(fact["sha256"], f"{where} file {name}")
        _require(type(fact["bytes"]) is int and fact["bytes"] > 0,
                 f"{where} file {name} bytes is not positive")
    actual = {name: {key: fact[key] for key in ("sha256", "bytes")}
              for name, fact in observed.items()}
    _require(expected == actual,
             f"{where} file bytes differ from runtime observations")


def _check_expectations(receipt: Mapping[str, Any],
                        expected: Mapping[str, Any]) -> None:
    """Join explicit consumer facts to the witness. Refuse any mismatch."""
    sources = _check_join(receipt)
    _fields(expected, ("endpoint", "served_alias", "artifacts", "tokenizer",
                       "attempt_id", "ranks"), "expected facts")
    for key, actual, where in (
            ("endpoint", receipt["listener"]["endpoint"], "endpoint"),
            ("served_alias", receipt["listener"]["served_alias"], "alias"),
            ("attempt_id", receipt["launch"]["attempt_id"], "attempt")):
        _text(expected[key], f"expected {where}")
        _require(expected[key] == actual,
                 f"expected {where} differs from runtime observations")
    ranks = expected["ranks"]
    _require(isinstance(ranks, list) and ranks
             and all(type(rank) is int for rank in ranks)
             and ranks == list(range(len(ranks))),
             "expected ranks do not cover the complete world")
    _require(ranks == receipt["launch"]["ranks"],
             "receipt ranks differ from expected ranks")
    _check_expected_files(expected["artifacts"], sources, "expected artifact")
    tokenizer = expected["tokenizer"]
    _fields(tokenizer, ("files", "backend", "vocab", "special_ids"),
            "expected tokenizer")
    _check_expected_files(tokenizer["files"], receipt["tokenizer"]["files"],
                          "expected tokenizer")
    for key in ("backend", "vocab", "special_ids"):
        _require(canonical(tokenizer[key])
                 == canonical(receipt["tokenizer"][key]),
                 f"expected tokenizer {key} differs from runtime observations")


def _reject_constant(value: str) -> Any:
    """Refuse NaN and Infinity tokens: the witness must be strict JSON."""
    raise ValueError(f"non-strict JSON constant {value}")


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def read_witness(path: str | Path, *, where: str = "witness") -> dict[str, Any]:
    """Parse ``path`` as strict JSON with duplicate-key refusal."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=_duplicate_key,
                           parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"{where} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} is not a JSON object")
    return dict(value)


def read_expected(path: str | Path) -> dict[str, Any]:
    """Parse the consumer expectation document."""
    return read_witness(path, where="expected facts")


def _parse_ranks(text: str) -> list[int]:
    try:
        ranks = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError(
            "expected ranks must be comma-separated integers") from exc
    if not ranks:
        raise ValueError("expected ranks must be comma-separated integers")
    return ranks


def verify(witness: Mapping[str, Any],
           expected: Mapping[str, Any]) -> dict[str, Any]:
    """Join ``witness`` to ``expected``. Never touches a rank or endpoint.

    Plain equality refuses every mismatch in both dev and certified modes.
    No identity seal, no stamp, no continuation with stored data.
    """
    try:
        if not isinstance(witness, Mapping) or not isinstance(expected,
                                                               Mapping):
            raise _refuse("witness and expected facts must be objects")
        _check_expectations(witness, expected)
    except _Refusal as exc:
        return {"schema": TESSERA_WITNESS_SCHEMA, "verdict": VERDICT_REFUSE,
                "reason": str(exc)}
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError,
            IndexError) as exc:
        return {"schema": TESSERA_WITNESS_SCHEMA, "verdict": VERDICT_REFUSE,
                "reason": f"REFUSED: {exc}"}
    return {"schema": TESSERA_WITNESS_SCHEMA, "verdict": VERDICT_PASS,
            "witness_fingerprint": fingerprint(witness),
            "endpoint": expected["endpoint"],
            "served_alias": expected["served_alias"],
            "attempt_id": expected["attempt_id"],
            "ranks": list(expected["ranks"]),
            "proof_scope": PROOF_SCOPE_RECORDED,
            "note": "Recorded launch only; not current endpoint state."}


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def served_binding_problems(config: Mapping[str, Any]) -> list[str]:
    """Name each missing served-binding field. Empty means complete."""
    backend = config.get("backend") if isinstance(config, Mapping) else None
    if not isinstance(backend, Mapping) or backend.get("name") != SERVED_BACKEND:
        return ["served binding requires backend.name == 'served'"]
    problems: list[str] = []
    binding = backend.get("serving_runtime")
    if not isinstance(binding, Mapping):
        return ["served binding requires backend.serving_runtime"]
    if not _is_nonempty_str(binding.get("witness")):
        problems.append("served binding requires serving_runtime.witness")
    if not _is_nonempty_str(binding.get("endpoint")):
        problems.append("served binding requires serving_runtime.endpoint")
    if not _is_nonempty_str(binding.get("served_alias")):
        problems.append("served binding requires serving_runtime.served_alias")
    if not _is_nonempty_str(binding.get("attempt_id")):
        problems.append("served binding requires serving_runtime.attempt_id")
    ranks = binding.get("ranks")
    if (not isinstance(ranks, list) or not ranks
            or any(type(r) is not int or r < 0 for r in ranks)):
        problems.append(
            "served binding requires serving_runtime.ranks as a rank list")
    if not _is_nonempty_str(binding.get("expected")):
        problems.append("served binding requires serving_runtime.expected")
    return problems


def bind_served_task(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the producer witness and return the verdict record.

    Reads the witness and expectation files the config names, joins them
    with the explicit expected endpoint, alias, attempt, and ranks, and
    refuses on any verifier refusal. Starts no rank, imports no serving
    runtime, and seals no identity: binding labels must equal the
    expectation file labels with plain equality in every mode.
    """
    problems = served_binding_problems(config)
    if problems:
        raise ValueError("; ".join(problems))
    binding = config["backend"]["serving_runtime"]
    witness = read_witness(binding["witness"], where="served witness")
    expected_doc = read_expected(binding["expected"])
    for key in ("endpoint", "served_alias", "attempt_id"):
        if expected_doc.get(key) != binding[key]:
            raise ValueError(
                f"served binding {key} differs from its expectation file")
    if sorted(expected_doc.get("ranks") or []) != sorted(binding["ranks"]):
        raise ValueError(
            "served binding ranks differ from their expectation file")
    expected = {"endpoint": binding["endpoint"],
                "served_alias": binding["served_alias"],
                "attempt_id": binding["attempt_id"],
                "ranks": sorted(binding["ranks"]),
                **{k: v for k, v in expected_doc.items()
                   if k not in ("endpoint", "served_alias", "attempt_id",
                                "ranks")}}
    verdict = verify(witness, expected)
    if verdict.get("verdict") != VERDICT_PASS:
        raise ValueError(
            f"served runtime witness refused: {verdict.get('reason')}")
    return {"witness_fingerprint": verdict["witness_fingerprint"],
            "endpoint": binding["endpoint"],
            "served_alias": binding["served_alias"],
            "attempt_id": binding["attempt_id"],
            "ranks": sorted(binding["ranks"]),
            "proof_scope": verdict["proof_scope"],
            "verdict": VERDICT_PASS}


def is_served_config(config: Mapping[str, Any]) -> bool:
    """Whether ``config`` names the served backend."""
    backend = config.get("backend") if isinstance(config, Mapping) else None
    return isinstance(backend, Mapping) and backend.get("name") == SERVED_BACKEND


def main(argv: list[str] | None = None) -> int:
    """Verify one witness file and print the machine-readable verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        witness = read_witness(args.witness)
        expected_doc = read_expected(args.expected)
        expected = {"endpoint": args.endpoint, "served_alias": args.alias,
                    "attempt_id": args.attempt,
                    "ranks": _parse_ranks(args.ranks),
                    **{k: v for k, v in expected_doc.items()
                       if k not in ("endpoint", "served_alias", "attempt_id",
                                    "ranks")}}
        verdict = verify(witness, expected)
    except ValueError as exc:
        verdict = {"schema": TESSERA_WITNESS_SCHEMA, "verdict": VERDICT_REFUSE,
                   "reason": str(exc)}
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if verdict["verdict"] == VERDICT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
