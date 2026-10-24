"""Synthetic CPU observations, NOT real launch evidence.

Fixture constructors ported from PR #2470 (80f95d0271). Public validation is
owned by the unchanged producer oracle under fixtures/tessera_endpoint_public.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

PAYLOAD = b"\x01\x02\x03\x04"
TENSORS = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
BACKEND = {"version": "1.0", "model": {"type": "WordLevel",
           "vocab": {"<s>": 0, "a": 1}, "unk_token": "<s>"},
           "added_tokens": [], "truncation": None, "padding": None,
           "normalizer": None, "pre_tokenizer": None, "post_processor": None,
           "decoder": None}


def _owner(pid):
    return {"host": "fixture", "boot_id": "fixture-boot", "pid": pid,
            "start_ticks": 10, "started_unix": 100.0}


def _attempt(owner):
    return (f"{owner['host']}/{owner['boot_id']}/"
            f"{owner['pid']}/{owner['start_ticks']}")


def _source():
    header = json.dumps(TENSORS).encode()
    raw = len(header).to_bytes(8, "little") + header + PAYLOAD
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "data_start": 8 + len(header), "tensors": copy.deepcopy(TENSORS)}


def make_witness(*, world=2):
    loaded = hashlib.sha256(PAYLOAD).hexdigest()
    source = _source()
    model = {"model_path": "artifact",
             "load_started_unix": 200.0, "load_finished_unix": 300.0,
             "files": {"model.safetensors": source},
             "inputs": [{"file": "model.safetensors", "tensor": "weight",
                         "start": 0, "end": 4, "sha256": loaded,
                         "source_sha256": loaded, "target": "weight",
                         "loaded_unix": 250.0}],
             "resident": {"parameter:weight": {"sha256": loaded, "bytes": 4,
                                               "dtype": "torch.uint8",
                                               "shape": [4]}}}
    token_contents = {"tokenizer.json": copy.deepcopy(BACKEND),
                      "tokenizer_config.json": {"bos_token": "<s>"}}
    token_files = {}
    for name, content in token_contents.items():
        raw = json.dumps(content).encode()
        token_files[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                             "bytes": len(raw), "content": content}
    owner = _owner(10)
    witness = {"schema": "tessera.endpoint_runtime_witness.v1",
               "listener": {"endpoint": "http://127.0.0.1:8142",
                            "served_alias": "fixture", "owner": owner},
               "launch": {"attempt_id": _attempt(owner),
                          "ranks": list(range(world))},
               "lifetime": {"request_id": "fixture-request",
                            "started_unix": 400.0, "finished_unix": 600.0},
               "artifacts": [{"rank": rank, "world_size": world,
                              "owner": _owner(20 + rank),
                              "request_id": "fixture-request",
                              "observed_unix": 500.0,
                              "models": [copy.deepcopy(model)]}
                             for rank in range(world)],
               "tokenizer": {"path": "artifact",
                             "request_id": "fixture-request",
                             "observed_unix": 500.0, "files": token_files,
                             "backend": copy.deepcopy(BACKEND),
                             "vocab": {"<s>": 0, "a": 1},
                             "special_ids": {"bos": 0, "eos": None,
                                             "pad": None, "unk": None,
                                             "sep": None, "cls": None,
                                             "mask": None}},
               "byte_coverage": {"kind": "successful-loader-inputs-and-"
                                  "post-load-resident-state",
                                 "files": ["model.safetensors"],
                                 "tensor_payload_bytes": 4},
               "qualification_scope": "runtime_byte_binding"}
    witness["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in witness.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return witness


def make_expected(witness):
    sources = witness["artifacts"][0]["models"][0]["files"]
    return {"endpoint": witness["listener"]["endpoint"],
            "served_alias": witness["listener"]["served_alias"],
            "artifacts": {name: {"sha256": fact["sha256"],
                                 "bytes": fact["bytes"]}
                          for name, fact in sources.items()},
            "tokenizer": {"files": {name: {"sha256": fact["sha256"],
                                           "bytes": fact["bytes"]}
                                    for name, fact in
                                    witness["tokenizer"]["files"].items()},
                          "backend": copy.deepcopy(
                              witness["tokenizer"]["backend"]),
                          "vocab": copy.deepcopy(
                              witness["tokenizer"]["vocab"]),
                          "special_ids": copy.deepcopy(
                              witness["tokenizer"]["special_ids"])},
            "attempt_id": witness["launch"]["attempt_id"],
            "ranks": list(witness["launch"]["ranks"])}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def served_config(witness_path, expected_path, witness):
    return {"backend": {"name": "served",
                        "serving_runtime": {
                            "witness": str(witness_path),
                            "expected": str(expected_path),
                            "endpoint":
                            witness["listener"]["endpoint"],
                            "served_alias":
                            witness["listener"]["served_alias"],
                            "attempt_id":
                            witness["launch"]["attempt_id"],
                            "ranks": list(
                                witness["launch"]["ranks"])}}}


def resign(witness):
    witness["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in witness.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return witness



PUBLIC_CLI = Path(__file__).parent / "fixtures/tessera_endpoint_public/tools/verify_endpoint_witness.py"


def case(tmp_path):
    """Write tiny bytes and freeze independent expectations before any mutation."""
    served = tmp_path / "artifact"
    served.mkdir()
    token_dir = tmp_path / "tokenizer"
    token_dir.mkdir()
    header = json.dumps(TENSORS).encode()
    raw = len(header).to_bytes(8, "little") + header + PAYLOAD
    (served / "model.safetensors").write_bytes(raw)
    for name, content in {"tokenizer.json": BACKEND,
                          "tokenizer_config.json": {"bos_token": "<s>"}}.items():
        (token_dir / name).write_bytes(json.dumps(content).encode())
    witness = make_witness()
    # Expected bytes come from the independently selected files, not observed
    # rank records. Expected loaded facts are the fixed synthetic model facts.
    def fact(path):
        data = path.read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    expected = {"endpoint": "http://127.0.0.1:8142", "served_alias": "fixture",
                "attempt_id": "fixture/fixture-boot/10/10", "ranks": [0, 1],
                "artifacts": {"model.safetensors": fact(served / "model.safetensors")},
                "tokenizer": {"files": {p.name: fact(p) for p in token_dir.iterdir()},
                              "backend": copy.deepcopy(BACKEND),
                              "vocab": {"<s>": 0, "a": 1},
                              "special_ids": {"bos": 0, "eos": None, "pad": None,
                                              "unk": None, "sep": None, "cls": None,
                                              "mask": None}}}
    witness_path = write_json(tmp_path / "witness.json", witness)
    expected_path = write_json(tmp_path / "expected.json", expected)
    config = served_config(witness_path, expected_path, witness)
    config["backend"]["serving_runtime"].update(
        verifier=str(PUBLIC_CLI), served_dir=str(served), tokenizer_dir=str(token_dir))
    config.update(tasks=["arc_easy"], sampling={"limit": 1, "num_fewshot": 0,
                  "random_seed": 0, "numpy_seed": 0, "torch_seed": 0, "fewshot_seed": 0})
    return config, witness, expected


def public_call(config, expected):
    """Arguments for the existing PR's public-process consumer API."""
    runtime = config["backend"]["serving_runtime"]
    return dict(cli=runtime["verifier"], witness=runtime["witness"],
                served_dir=runtime["served_dir"], tokenizer_dir=runtime["tokenizer_dir"],
                expected=expected, python=sys.executable, timeout=10)
