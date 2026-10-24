"""Validate the endpoint receipt with the standard library only.

Coverage means successful weight-loader input bytes, before the loader's
rank-local cut or format conversion. Resident observations cover the resulting
parameters and Tessera resident tensors. They do not prove numerical equivalence.
The listener obtains rank observations through its live engine RPC. Live proof
requires that listener. Offline proof covers only the recorded launch and bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = "tessera.endpoint_runtime_witness.v1"
COVERAGE = "successful-loader-inputs-and-post-load-resident-state"
QUALIFICATION_SCOPE = "runtime_byte_binding"
TOKENIZER_NAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json")
# ByteLevel flags that no token ID depends on. A pre-tokenizer's `trim_offsets` only
# shapes reported offsets, and the ByteLevel decoder ignores every flag it carries.
# transformers rewrites them after loading `tokenizer.json`; the `tokenizers` library
# alone reproduces the file exactly (measured on the pinned image, tessera#1134).
# A pre-tokenizer's `add_prefix_space` and `use_regex` stay exact, because they change IDs.
BYTE_LEVEL_UNREAD_FLAGS = {"pre_tokenizer": ("trim_offsets",),
                           "decoder": ("add_prefix_space", "trim_offsets", "use_regex")}
DTYPE_BYTES = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
               "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fields(value, names, where):
    require(isinstance(value, dict), f"{where} is not an object")
    require(set(value) == set(names), f"{where} fields differ from {sorted(names)}")


def text(value, where):
    require(isinstance(value, str) and value, f"{where} is not non-empty text")


def sha(value, where):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{where} is not sha256")


def number(value, where):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0,
            f"{where} is not a positive finite time")


def relative(name):
    text(name, "file name")
    require(not name.startswith("/") and "\\" not in name
            and all(p not in ("", ".", "..") for p in name.split("/")),
            f"file name {name!r} is not relative")


def owner(value):
    fields(value, ("host", "boot_id", "pid", "start_ticks", "started_unix"), "process owner")
    text(value["host"], "owner host")
    text(value["boot_id"], "owner boot_id")
    for key in ("pid", "start_ticks"):
        require(type(value[key]) is int and value[key] > 0, f"owner {key} is not positive")
    number(value["started_unix"], "owner started_unix")


def attempt_id(value):
    owner(value)
    return f"{value['host']}/{value['boot_id']}/{value['pid']}/{value['start_ticks']}"


def fingerprint(receipt):
    return digest({k: v for k, v in receipt.items() if k != "fingerprint"})


def tokenizer_vocab(backend):
    model = backend["model"]
    vocab = model["vocab"]
    if isinstance(vocab, list):  # Unigram vocabulary: [token, score], in ID order.
        vocab = {item[0]: i for i, item in enumerate(vocab)}
    require(isinstance(vocab, dict) and vocab, "tokenizer backend has no vocabulary")
    vocab = dict(vocab)
    for item in backend.get("added_tokens", []):
        vocab[item["content"]] = item["id"]
    require(all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in vocab.items()),
            "tokenizer vocabulary has an invalid token ID")
    require(len(set(vocab.values())) == len(vocab), "tokenizer vocabulary repeats an ID")
    return vocab


def token_identity(backend):
    """The backend without the ByteLevel flags in BYTE_LEVEL_UNREAD_FLAGS."""
    require(isinstance(backend, dict), "tokenizer backend is not an object")
    def strip(node, flags):
        if not isinstance(node, dict):
            return node
        node = dict(node)
        if node.get("type") == "ByteLevel":
            for flag in flags:
                node.pop(flag, None)
        for key in ("pretokenizers", "decoders"):
            if isinstance(node.get(key), list):
                node[key] = [strip(child, flags) for child in node[key]]
        return node
    result = dict(backend)
    for slot, flags in BYTE_LEVEL_UNREAD_FLAGS.items():
        if slot in result:
            result[slot] = strip(result[slot], flags)
    return result


def file_fact(value, where):
    fields(value, ("sha256", "bytes", "dtype", "shape"), where)
    sha(value["sha256"], where)
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"{where} bytes is not positive")
    text(value["dtype"], f"{where} dtype")
    require(isinstance(value["shape"], list) and all(type(n) is int and n > 0 for n in value["shape"])
            and value["bytes"] % math.prod(value["shape"]) == 0, f"{where} shape has invalid byte bounds")


def check_join(receipt):
    fields(receipt, ("schema", "listener", "launch", "lifetime", "artifacts", "tokenizer",
                     "byte_coverage", "qualification_scope", "fingerprint"), "receipt")
    require(receipt["schema"] == SCHEMA, "receipt schema is not supported")
    listener, launch, lifetime = receipt["listener"], receipt["launch"], receipt["lifetime"]
    fields(listener, ("endpoint", "served_alias", "owner"), "listener")
    owner(listener["owner"])
    text(listener["served_alias"], "listener alias")
    parsed = urlsplit(listener["endpoint"])
    require(parsed.scheme in ("http", "https") and parsed.hostname and parsed.port
            and parsed.path in ("", "/") and not parsed.username and not parsed.query
            and not parsed.fragment, "listener endpoint is not a direct HTTP address")
    fields(launch, ("attempt_id", "ranks"), "launch")
    require(launch["attempt_id"] == attempt_id(listener["owner"]),
            "launch attempt differs from the observed listener process")
    fields(lifetime, ("request_id", "started_unix", "finished_unix"), "lifetime")
    text(lifetime["request_id"], "lifetime request_id")
    number(lifetime["started_unix"], "lifetime started_unix")
    number(lifetime["finished_unix"], "lifetime finished_unix")
    require(listener["owner"]["started_unix"] <= lifetime["started_unix"] <= lifetime["finished_unix"],
            "listener observation is outside its process lifetime")
    ranks = launch["ranks"]
    require(isinstance(ranks, list) and ranks and all(type(r) is int for r in ranks)
            and ranks == list(range(len(ranks))), "launch ranks do not cover the complete world")
    artifacts = receipt["artifacts"]
    require(isinstance(artifacts, list) and len(artifacts) == len(ranks), "artifact ranks are incomplete")
    require([a["rank"] for a in artifacts] == ranks, "artifact ranks differ from launch ranks")
    common_files = None
    consumed = {}
    process_owners = set()
    for rank in artifacts:
        fields(rank, ("rank", "world_size", "owner", "request_id", "observed_unix", "models"), "rank")
        require(type(rank["rank"]) is int and type(rank["world_size"]) is int
                and rank["world_size"] == len(ranks), "rank world size is inconsistent")
        owner(rank["owner"])
        process_key = attempt_id(rank["owner"])
        require(process_key not in process_owners, "two ranks name the same worker process")
        process_owners.add(process_key)
        require(rank["request_id"] == lifetime["request_id"], "rank observation mixes runtime requests")
        number(rank["observed_unix"], "rank observed_unix")
        require(lifetime["started_unix"] <= rank["observed_unix"] <= lifetime["finished_unix"],
                "rank observation is outside the runtime observation lifetime")
        require(isinstance(rank["models"], list) and rank["models"], "rank has no loaded model")
        rank_files = {}
        for model in rank["models"]:
            fields(model, ("model_path", "load_started_unix", "load_finished_unix", "files", "inputs", "resident"),
                   "loaded model")
            text(model["model_path"], "loaded model path")
            number(model["load_started_unix"], "model load start")
            number(model["load_finished_unix"], "model load finish")
            require(rank["owner"]["started_unix"] <= model["load_started_unix"]
                    <= model["load_finished_unix"] <= lifetime["started_unix"],
                    "loaded byte observation is outside the worker process lifetime")
            require(isinstance(model["files"], dict) and model["files"], "loaded model has no source files")
            for name, source in model["files"].items():
                relative(name)
                fields(source, ("sha256", "bytes", "data_start", "tensors"), "loaded source")
                sha(source["sha256"], "loaded source digest")
                require(type(source["bytes"]) is int and type(source["data_start"]) is int
                        and 8 < source["data_start"] < source["bytes"], "source file has invalid byte bounds")
                require(isinstance(source["tensors"], dict) and source["tensors"], "source file has no tensor roster")
                ranges = []
                for tensor_name, descriptor in source["tensors"].items():
                    text(tensor_name, "source tensor name")
                    fields(descriptor, ("dtype", "shape", "data_offsets"), "source tensor")
                    text(descriptor["dtype"], "source dtype")
                    require(isinstance(descriptor["shape"], list)
                            and all(type(n) is int and n >= 0 for n in descriptor["shape"]), "source tensor shape is invalid")
                    bounds = descriptor["data_offsets"]
                    require(isinstance(bounds, list) and len(bounds) == 2
                            and all(type(n) is int for n in bounds) and 0 <= bounds[0] <= bounds[1]
                            <= source["bytes"] - source["data_start"], "source tensor offsets are invalid")
                    require(descriptor["dtype"] in DTYPE_BYTES
                            and math.prod(descriptor["shape"]) * DTYPE_BYTES[descriptor["dtype"]]
                            == bounds[1] - bounds[0], "source dtype or shape differs from byte bounds")
                    ranges.append(bounds)
                cursor = 0
                for start, end in sorted(ranges):
                    require(start == cursor, "source tensor roster has incomplete byte coverage")
                    cursor = end
                require(cursor + source["data_start"] == source["bytes"], "source tensor roster omits file bytes")
                require(name not in rank_files or rank_files[name] == source, "rank sources disagree on file bytes")
                rank_files[name] = source
            require(isinstance(model["resident"], dict) and model["resident"], "model has no resident byte observations")
            for name, fact in model["resident"].items():
                text(name, "resident tensor name")
                file_fact(fact, "resident tensor")
            require(isinstance(model["inputs"], list) and model["inputs"], "model has no successful loader inputs")
            for item in model["inputs"]:
                fields(item, ("file", "tensor", "start", "end", "sha256", "source_sha256", "target", "loaded_unix"),
                       "loader input")
                require(item["file"] in model["files"], "loader input names an unrelated source file")
                source = model["files"][item["file"]]
                require(item["tensor"] in source["tensors"], "loader input names an unrelated source tensor")
                start, end = source["tensors"][item["tensor"]]["data_offsets"]
                require(type(item["start"]) is int and type(item["end"]) is int
                        and start <= item["start"] < item["end"] <= end, "loader input has invalid byte bounds")
                text(item["target"], "loader target")
                sha(item["sha256"], "loader input digest")
                require(item["sha256"] == item["source_sha256"], "loaded input bytes differ from source bytes")
                number(item["loaded_unix"], "loaded input time")
                require(model["load_started_unix"] <= item["loaded_unix"] <= model["load_finished_unix"],
                        "loader input is outside its load lifetime")
                consumed.setdefault((item["file"], item["tensor"]), []).append((item["start"], item["end"]))
        if common_files is None:
            common_files = rank_files
        require(rank_files == common_files, "serving ranks disagree on loaded artifact files")
    payload_bytes = 0
    for name, source in common_files.items():
        for tensor_name, descriptor in source["tensors"].items():
            start, end = descriptor["data_offsets"]
            cursor = start
            for left, right in sorted(consumed.get((name, tensor_name), [])):
                require(left <= cursor, f"incomplete loaded byte coverage for {name}:{tensor_name}")
                cursor = max(cursor, right)
            require(cursor == end, f"incomplete loaded byte coverage for {name}:{tensor_name}")
            payload_bytes += end - start
    expected_coverage = {"kind": COVERAGE, "files": sorted(common_files), "tensor_payload_bytes": payload_bytes}
    require(receipt["byte_coverage"] == expected_coverage, "byte_coverage differs from observed loader inputs")
    tokenizer = receipt["tokenizer"]
    fields(tokenizer, ("path", "request_id", "observed_unix", "files", "backend", "vocab", "special_ids"), "tokenizer")
    text(tokenizer["path"], "server tokenizer path")
    require(tokenizer["request_id"] == lifetime["request_id"], "tokenizer observation mixes runtime requests")
    number(tokenizer["observed_unix"], "tokenizer observed_unix")
    require(lifetime["started_unix"] <= tokenizer["observed_unix"] <= lifetime["finished_unix"],
            "tokenizer observation is outside the runtime observation lifetime")
    require(isinstance(tokenizer["files"], dict) and "tokenizer.json" in tokenizer["files"], "tokenizer byte evidence is incomplete")
    for name, fact in tokenizer["files"].items():
        require(name in TOKENIZER_NAMES, "tokenizer source file is not supported")
        fields(fact, ("sha256", "bytes", "content"), "tokenizer source")
        sha(fact["sha256"], "tokenizer source digest")
        require(type(fact["bytes"]) is int and fact["bytes"] > 0, "tokenizer source size is invalid")
    require(token_identity(tokenizer["backend"]) == token_identity(tokenizer["files"]["tokenizer.json"]["content"]),
            "loaded tokenizer backend differs from tokenizer bytes")
    require(tokenizer["vocab"] == tokenizer_vocab(tokenizer["backend"]),
            "loaded tokenizer mapping differs from tokenizer bytes")
    config = tokenizer["files"].get("tokenizer_config.json", {}).get("content", {})
    special_map = tokenizer["files"].get("special_tokens_map.json", {}).get("content", {})
    fields(tokenizer["special_ids"], ("bos", "eos", "pad", "unk", "sep", "cls", "mask"), "tokenizer special IDs")
    for label, value in tokenizer["special_ids"].items():
        if value is None:
            continue
        declared = special_map.get(label + "_token", config.get(label + "_token"))
        if isinstance(declared, dict):
            declared = declared.get("content")
        require(type(value) is int and isinstance(declared, str)
                and tokenizer["vocab"].get(declared) == value, "loaded tokenizer special ID differs from tokenizer bytes")
    require(receipt["qualification_scope"] == QUALIFICATION_SCOPE,
            "receipt overstates its qualification scope")
    require(receipt["fingerprint"] == fingerprint(receipt), "receipt fingerprint differs from its bytes")
    return common_files


def check_expected_files(expected, observed, where):
    require(isinstance(expected, dict) and expected, f"{where} has no file bytes")
    for name, fact in expected.items():
        relative(name)
        fields(fact, ("sha256", "bytes"), f"{where} file {name}")
        sha(fact["sha256"], f"{where} file {name}")
        require(type(fact["bytes"]) is int and fact["bytes"] > 0,
                f"{where} file {name} bytes is not positive")
    actual = {name: {key: fact[key] for key in ("sha256", "bytes")}
              for name, fact in observed.items()}
    require(expected == actual, f"{where} file bytes differ from runtime observations")


def check_expectations(receipt, expected):
    """Check consumer facts without using them as runtime observations."""
    sources = check_join(receipt)
    fields(expected, ("endpoint", "served_alias", "artifacts", "tokenizer", "attempt_id", "ranks"),
           "expected facts")
    for key, actual, where in (
        ("endpoint", receipt["listener"]["endpoint"], "endpoint"),
        ("served_alias", receipt["listener"]["served_alias"], "alias"),
        ("attempt_id", receipt["launch"]["attempt_id"], "attempt"),
    ):
        text(expected[key], f"expected {where}")
        require(expected[key] == actual, f"expected {where} differs from runtime observations")
    ranks = expected["ranks"]
    require(isinstance(ranks, list) and ranks and all(type(rank) is int for rank in ranks)
            and ranks == list(range(len(ranks))), "expected ranks do not cover the complete world")
    require(ranks == receipt["launch"]["ranks"], "receipt ranks differ from expected ranks")
    check_expected_files(expected["artifacts"], sources, "expected artifact")
    tokenizer = expected["tokenizer"]
    fields(tokenizer, ("files", "backend", "vocab", "special_ids"), "expected tokenizer")
    check_expected_files(tokenizer["files"], receipt["tokenizer"]["files"], "expected tokenizer")
    for key in ("backend", "vocab", "special_ids"):
        # Canonical JSON keeps booleans distinct from integer token IDs.
        require(canonical(tokenizer[key]) == canonical(receipt["tokenizer"][key]),
                f"expected tokenizer {key} differs from runtime observations")


def prove_files(receipt, served_dir, tokenizer_dir=None):
    sources = check_join(receipt)
    root = Path(served_dir)
    for name, fact in sources.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "source file escapes the artifact directory")
        raw_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                raw_hash.update(block)
            require(handle.tell() == fact["bytes"] and raw_hash.hexdigest() == fact["sha256"], "artifact file bytes differ from loaded source")
            handle.seek(0)
            length = int.from_bytes(handle.read(8), "little")
            require(length + 8 == fact["data_start"], "artifact header size differs from the loaded source")
            header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            require(header == fact["tensors"], "artifact tensor roster differs from the loaded source")
            observed_ranges = {}
            for rank in receipt["artifacts"]:
                for model in rank["models"]:
                    for item in model["inputs"]:
                        if item["file"] != name:
                            continue
                        bounds = (item["start"], item["end"])
                        if bounds not in observed_ranges:
                            handle.seek(fact["data_start"] + item["start"])
                            remaining = item["end"] - item["start"]
                            tensor_hash = hashlib.sha256()
                            while remaining:
                                block = handle.read(min(remaining, 8 * 1024 * 1024))
                                require(block, "artifact tensor bytes ended before the loaded input")
                                tensor_hash.update(block)
                                remaining -= len(block)
                            observed_ranges[bounds] = tensor_hash.hexdigest()
                        require(observed_ranges[bounds] == item["sha256"], "artifact tensor bytes differ from loaded input")
    token_root = Path(tokenizer_dir) if tokenizer_dir is not None else root
    for name, fact in receipt["tokenizer"]["files"].items():
        path = token_root / name
        require(not path.is_symlink(), "tokenizer file is a symlink")
        raw = path.read_bytes()
        require(len(raw) == fact["bytes"] and hashlib.sha256(raw).hexdigest() == fact["sha256"], "tokenizer file bytes differ from server observations")
        content = json.loads(raw) if name.endswith(".json") else raw.decode()
        require(content == fact["content"], "tokenizer source content differs from its bytes")


def verify_recorded_witness(receipt, *, expected, served_dir=None, tokenizer_dir=None):
    """Return a refusal, or None after recorded proof. Prove no current endpoint."""
    try:
        check_expectations(receipt, expected)
        require(served_dir is not None, "no artifact directory proves the loaded bytes")
        prove_files(receipt, served_dir, tokenizer_dir)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError, IndexError) as exc:
        return f"REFUSED: {exc}"
    return None


def binding(receipt):
    """The current byte binding, without request-specific observation times."""
    return {"listener": receipt["listener"], "launch": receipt["launch"],
            "artifacts": [{"rank": a["rank"], "world_size": a["world_size"], "owner": a["owner"],
                           "models": a["models"]} for a in receipt["artifacts"]],
            "tokenizer": {k: v for k, v in receipt["tokenizer"].items()
                          if k not in ("request_id", "observed_unix")}}


def verify_witness(receipt, *, ranks=None, served_dir=None, tokenizer_dir=None, live=None):
    """Return a refusal, or None after byte proof and a current listener proof."""
    try:
        check_join(receipt)
        if ranks is not None:
            require(ranks == receipt["launch"]["ranks"], "receipt ranks differ from expected ranks")
        require(served_dir is not None, "no artifact directory proves the loaded bytes")
        prove_files(receipt, served_dir, tokenizer_dir)
        require(live is not None, "no live runtime binding proves listener ownership")
        check_join(live)
        require(binding(receipt) == binding(live), "live runtime binding differs; the listener or loaded state changed")
    except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError, IndexError) as exc:
        return f"REFUSED: {exc}"
    return None
