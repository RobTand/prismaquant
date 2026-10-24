"""Current source-bound gold operations, not historical producer qualification.

References are literal wire bytes or stdlib hashes, never replacement digest
owners. Selected function extraction retains the consumer bodies unchanged and
avoids constructing vLLM engines; engine/input integration has existing witnesses.
"""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import random
import sys
from types import SimpleNamespace, ModuleType
from typing import Any, Mapping

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def selected(filename, *names, **bindings):
    """Compile only named, current top-level definitions (including constants)."""
    path = ROOT / "tools" / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in tree.body:
        declared = {node.name} if isinstance(node, ast.FunctionDef) else {
            target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)
        }
        if declared.intersection(names):
            nodes.append(node)
            found.update(declared.intersection(names))
    assert found == set(names)
    scope = dict(__file__=str(path), Path=Path, hashlib=hashlib, json=json,
                 os=os, re=re, torch=torch, Any=Any, Mapping=Mapping)
    scope.update(bindings)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def test_serve_fingerprint_wire_recipe_and_exclusions(tmp_path):
    s = selected("serve_fingerprint.py", "fingerprint", "fingerprint_payload",
                 "_FINGERPRINT_EXCLUDED", "_canonical_sha256", "_file_identity")
    manifest = {"z": Path("relative/é"), "a": [1, True, None]}
    expected = hashlib.sha256(b'{"a":[1,true,null],"z":"relative/\\u00e9"}').hexdigest()
    assert s["fingerprint"](manifest) == expected
    assert s["fingerprint"](dict(reversed(list(manifest.items())))) == expected
    excluded = {"created", "launch_argv", "processes", "model", "container", "hostname",
                "serve_fingerprint", "schema", "served_model_name", "written_by",
                "attestation_phase", "measurement_parent_pid", "engine_descendant_pids",
                "native_extension_status"}
    assert s["_FINGERPRINT_EXCLUDED"] == excluded
    for key in excluded:
        assert s["fingerprint"]({**manifest, key: "changed"}) == expected
    assert s["fingerprint"]({**manifest, "resident_extensions": []}) != expected
    # The stack's default=str is intentionally NOT the strict canonical leaf.
    with pytest.raises(TypeError):
        s["_canonical_sha256"](manifest)
    assert s["_canonical_sha256"]({"z": "é", "a": 1}) == hashlib.sha256(
        b'{"a":1,"z":"\\u00e9"}').hexdigest()
    path = tmp_path / "source"
    for raw in (b"", b"\x00\xff\r\n" + b"x" * (1024 * 1024 + 3)):
        path.write_bytes(raw)
        assert s["_file_identity"](path) == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_current_producer_roster_binds_source_and_supplemental_json(tmp_path):
    s = selected("serve_fingerprint.py", "gold_producer_identity", "_file_identity",
                 "_canonical_sha256", "_GOLD_PRODUCER_COMMON_FILES", "_GOLD_PRODUCER_TOOL_FILES",
                 "GOLD_PRODUCER_IDENTITY_SCHEMA")
    # Fake only the provenance observation: this is NOT a clean-tree attestation.
    s["git_provenance"] = lambda: {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    common = s["_GOLD_PRODUCER_COMMON_FILES"]
    tools = s["_GOLD_PRODUCER_TOOL_FILES"]
    assert set(tools) == {"build_streamed_full_kl_teacher", "measure_vllm_full_kl",
                          "measure_vllm_wikitext_ppl", "measure_served_gold"}
    supplemental = "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json"
    assert supplemental in common
    for tool, members in tools.items():
        record = s["gold_producer_identity"](tool)
        names = sorted(set(common + tuple(members)))
        expected = {}
        for name in names:
            raw = (ROOT / name).read_bytes()
            expected[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        assert list(record["source_files"]) == names
        assert record["source_files"] == expected
        assert record["source_files_sha256"] == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Actual refusal paths, using a disposable closure rather than editing sources.
    s["__file__"] = str(tmp_path / "tools" / "serve_fingerprint.py")
    for name in set(common).union(*(set(v) for v in tools.values())):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    before = s["gold_producer_identity"]("measure_served_gold")
    pin = tmp_path / supplemental
    pin.write_bytes(pin.read_bytes() + b"\n")
    after = s["gold_producer_identity"]("measure_served_gold")
    assert before["source_files_sha256"] != after["source_files_sha256"]
    pin.unlink()
    with pytest.raises(ValueError, match="missing or a symlink"):
        s["gold_producer_identity"]("measure_served_gold")
    pin.symlink_to(ROOT / supplemental)
    with pytest.raises(ValueError, match="missing or a symlink"):
        s["gold_producer_identity"]("measure_served_gold")
    with pytest.raises(ValueError, match="unknown gold"):
        s["gold_producer_identity"]("unknown")
    s["git_provenance"] = lambda: {"commit": "a" * 40, "dirty": True}
    with pytest.raises(ValueError, match="proven clean"):
        s["gold_producer_identity"]("measure_served_gold")


def test_dsv4_tensor_and_payload_tensor_descriptors():
    d = selected("dsv4_wikitext_inputs.py", "_tensor_sha256")
    class PayloadError(ValueError):
        pass
    p = selected("full_kl_teacher_payload.py", "tensor_descriptor", TeacherPayloadError=PayloadError)
    value = torch.tensor([[1, 2], [-3, 4]], dtype=torch.int32).t()
    assert not value.is_contiguous()
    # Logical row-major bytes, including DSv4's mandatory int64 conversion.
    assert d["_tensor_sha256"](value) == hashlib.sha256(struct.pack("=4q", 1, -3, 2, 4)).hexdigest()
    assert p["tensor_descriptor"](value) == {
        "dtype": "int32", "shape": [2, 2],
        "sha256": hashlib.sha256(struct.pack("=4i", 1, -3, 2, 4)).hexdigest()}
    with pytest.raises(PayloadError, match="requires a torch.Tensor"):
        p["tensor_descriptor"]([[1]])


def test_payload_atomic_json_publication_and_refusals(tmp_path, monkeypatch):
    class PayloadError(ValueError):
        pass
    p = selected("full_kl_teacher_payload.py", "atomic_json_write", TeacherPayloadError=PayloadError)
    target = tmp_path / "out" / "meta.json"
    expected = b'{\n  "a": "\xc3\xa9",\n  "z": 2\n}\n'
    replace = os.replace
    calls = []
    def observe(src, dst):
        assert Path(src).parent == target.parent
        assert Path(src).read_bytes() == expected
        assert not target.exists() or target.read_bytes() == b"prior"
        calls.append((src, dst))
        replace(src, dst)
    monkeypatch.setattr(os, "replace", observe)
    p["atomic_json_write"]({"z": 2, "a": "é"}, target)
    assert calls and target.read_bytes() == expected
    target.write_bytes(b"prior")
    def refuse(src, dst):
        assert Path(src).read_bytes() == expected
        raise OSError("publication refused")
    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="publication refused"):
        p["atomic_json_write"]({"z": 2, "a": "é"}, target)
    assert target.read_bytes() == b"prior"
    assert list(target.parent.iterdir()) == [target]
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            p["atomic_json_write"]({"bad": value}, target)
        assert target.read_bytes() == b"prior"
        assert list(target.parent.iterdir()) == [target]
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_bytes(b"someone else's evidence")
    with pytest.raises(PayloadError, match="already exists"):
        p["atomic_json_write"]({}, target)
    assert temporary.read_bytes() == b"someone else's evidence"


def test_gold_hash_leaves_and_strict_print_recipes(tmp_path):
    served = selected("measure_served_gold.py", "_sha256_file", "_canonical_sha256")
    ppl = selected("measure_vllm_wikitext_ppl.py", "_canonical_sha256", "_strict_json")
    kl = selected("measure_vllm_full_kl.py", "_strict_json_text")
    value = {"z": "é", "a": [1, True, None]}
    assert served["_canonical_sha256"](value) == hashlib.sha256(
        b'{"a":[1,true,null],"z":"\\u00e9"}').hexdigest()
    assert ppl["_canonical_sha256"](value) == hashlib.sha256(
        b'{"a":[1,true,null],"z":"\xc3\xa9"}').hexdigest()
    raw = b"\x00\xff\r\n" + b"x" * (8 * 1024 * 1024 + 1)
    path = tmp_path / "teacher.json"
    path.write_bytes(raw)
    assert served["_sha256_file"](path) == hashlib.sha256(raw).hexdigest()
    expected = '{\n  "z": "\\u00e9",\n  "a": [\n    1,\n    true,\n    null\n  ]\n}'
    assert ppl["_strict_json"](value) == expected
    assert kl["_strict_json_text"](value) == expected
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            served["_canonical_sha256"]({"bad": bad})
        with pytest.raises(ValueError):
            ppl["_canonical_sha256"]({"bad": bad})
        with pytest.raises(ValueError):
            ppl["_strict_json"]({"bad": bad})
        with pytest.raises(RuntimeError, match="non-finite full-KL"):
            kl["_strict_json_text"]({"bad": bad})


def test_prepare_corpus_descriptor_hashes_exact_joined_utf8(monkeypatch):
    import sys
    from types import SimpleNamespace
    from tools import prepare_dsv4_wikitext_inputs as prepare

    class Dataset(list):
        _fingerprint = "immutable-fixture"

    dataset = Dataset([{"text": "é"}, {"text": "  "}, {"text": None},
                       {"text": " kept ", "other": 1}, {"text": "終"}])
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    observed, text, descriptor = prepare._load_corpus(
        split="train", cache_dir="configured-cache", dataset_repo="Salesforce/wikitext")
    assert observed is dataset
    raw = b"\xc3\xa9\n\n kept \n\n\xe7\xb5\x82"
    assert text.encode() == raw
    assert descriptor == {
        "name": prepare.WIKITEXT_DATASET, "config": prepare.WIKITEXT_CONFIG,
        "split": "train", "revision": prepare.WIKITEXT_REVISION,
        "fingerprint": "immutable-fixture", "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    assert calls == [(("Salesforce/wikitext", prepare.WIKITEXT_CONFIG), {
        "split": "train", "cache_dir": "configured-cache", "revision": prepare.WIKITEXT_REVISION})]
    dataset._fingerprint = ""
    with pytest.raises(RuntimeError, match="immutable fingerprint"):
        prepare._load_corpus(split="test", cache_dir="configured-cache")


def test_prepare_main_publishes_file_digest_descriptor(tmp_path, monkeypatch, capsys):
    payload = {"schema": "fixture/1", "semantic_sha256": "a" * 64,
               "full_kl": {"token_ids_tensor_sha256": "b" * 64},
               "ppl": {"token_ids_sha256": "c" * 64}, "note": "é"}
    writer = selected("full_kl_teacher_payload.py", "atomic_json_write",
                      TeacherPayloadError=ValueError)["atomic_json_write"]
    s = selected("prepare_dsv4_wikitext_inputs.py", "main", argparse=argparse,
                 atomic_json_write=writer, _build_payload=lambda **kwargs: payload)
    output = tmp_path / "é.json"
    monkeypatch.setattr(sys, "argv", ["prepare", "--model", str(tmp_path),
        "--dataset-cache-dir", "configured-cache", "--output", str(output)])
    assert s["main"]() == 0
    expected = {"output": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "schema": "fixture/1", "semantic_sha256": "a" * 64,
        "full_kl_token_ids_tensor_sha256": "b" * 64, "ppl_token_ids_sha256": "c" * 64}
    assert capsys.readouterr().out == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        s["main"]()
    assert output.read_bytes() == before


def test_host_identity_ascii_machine_hash():
    values = {"/proc/sys/kernel/random/boot_id": " BOOT-A\n",
              "/etc/machine-id": " AbC123\n"}
    class File:
        def __init__(self, path):
            self.path = path
        def read_text(self):
            return values[self.path]
    s = selected("serve_fingerprint.py", "host_identity", Path=File,
                 socket=SimpleNamespace(gethostname=lambda: "fixture-host"),
                 _readlink=lambda path: "pid:[fixture]")
    assert s["host_identity"]() == {
        "hostname": "fixture-host", "boot_id": "boot-a",
        "machine_id_sha256": hashlib.sha256(b"abc123").hexdigest(),
        "pid_namespace": "pid:[fixture]"}
    values["/etc/machine-id"] = ""
    assert s["host_identity"]()["machine_id_sha256"] is None
    values["/etc/machine-id"] = "é"
    with pytest.raises(UnicodeEncodeError):
        s["host_identity"]()


def test_artifact_binding_strict_inventory_hash(tmp_path, monkeypatch):
    shipcard = ModuleType("prismaquant.shipcard")
    shipcard.CARD_FIGURE_FILENAMES = ()
    shipcard.compute_model_sha = lambda root: "fixture-model-sha"
    monkeypatch.setitem(sys.modules, "prismaquant.shipcard", shipcard)
    s = selected("serve_fingerprint.py", "artifact_binding")
    (tmp_path / "weights").write_bytes(b"\x00\xff")
    inventory = {"schema": "prismaquant.cb_export_artifact_inventory.v1",
                 "scope": "all_regular_files_recursive", "note": "é",
                 "file_bytes": {"weights": 2, "quant_config.json": 0},
                 "export_directory_bytes": 0}
    quant = tmp_path / "quant_config.json"
    def publish():
        # The inventory includes its own containing file; converge its byte count.
        for _ in range(10):
            raw = json.dumps({"provenance": {"artifact_inventory": inventory}}).encode()
            quant.write_bytes(raw)
            if inventory["file_bytes"]["quant_config.json"] == len(raw):
                return
            inventory["file_bytes"]["quant_config.json"] = len(raw)
            inventory["export_directory_bytes"] = len(raw) + 2
        pytest.fail("inventory size did not converge")
    publish()
    binding = s["artifact_binding"](tmp_path, launch_model=tmp_path)
    expected = hashlib.sha256(json.dumps(inventory, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert binding["artifact_inventory_sha256"] == expected
    # Valid byte accounting must not allow a non-finite JSON inventory hash.
    inventory["note"] = float("nan")
    publish()
    with pytest.raises(ValueError, match="Out of range float"):
        s["artifact_binding"](tmp_path)
    inventory["note"] = "é"
    publish()
    (tmp_path / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ from finalized inventory"):
        s["artifact_binding"](tmp_path)


def test_serve_publication_default_str(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PQ_RUNTIME_PRISMAQUANT_ROOT", raising=False)
    manifest = {"serve_fingerprint": "a" * 64, "resident_extensions": [],
                "residency_readable": True, "path": Path("é/relative")}
    s = selected("serve_fingerprint.py", "_cmd_write", "_cmd_show",
                 collect_manifest=lambda **kw: manifest,
                 load_manifest=lambda path: manifest)
    out = tmp_path / "nested" / "manifest.json"
    args = SimpleNamespace(pid=None, out=out, image=None, artifact_dir=None, base_url=None,
                           attestation_phase="fixture", vllm_runtime_pin=None,
                           manifest=out)
    expected = (b'{\n  "serve_fingerprint": "' + b'a' * 64 +
                b'",\n  "resident_extensions": [],\n  "residency_readable": true,\n'
                b'  "path": "\\u00e9/relative"\n}\n')
    assert s["_cmd_write"](args) == 0
    assert out.read_bytes() == expected
    capsys.readouterr()
    assert s["_cmd_show"](args) == 0
    assert capsys.readouterr().out.encode() == expected


@pytest.mark.parametrize("consumer", ["kl", "ppl"])
def test_measurement_loaders_hash_raw_file_and_joined_dataset(consumer, tmp_path, monkeypatch):
    class Dataset(list):
        _fingerprint = "fixture-revision"
    ds = Dataset([{"text": "é\r\nkept "}, {"text": "  "}, {"text": "終"}])
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return ds
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    seen = []
    def tokenizer(text, **kwargs):
        seen.append((text, kwargs))
        return SimpleNamespace(input_ids=torch.tensor([list(range(32))]))
    if consumer == "kl":
        s = selected("measure_vllm_full_kl.py", "_load_wikitext_calibration", random=random)
        def invoke(path=None):
            windows, starts, total, corpus = s["_load_wikitext_calibration"](
                tokenizer, cache_dir="configured-cache", n_samples=2, seqlen=3,
                text_file=path)
            assert windows == [list(range(start, start + 3)) for start in starts]
            assert total == 32
            return corpus
    else:
        s = selected("measure_vllm_wikitext_ppl.py", "_load_ids",
                     WIKITEXT_DATASET="wikitext", WIKITEXT_CONFIG="wikitext-2-raw-v1",
                     WIKITEXT_REVISION="fixture-revision")
        def invoke(path=None):
            ids, corpus = s["_load_ids"](tokenizer, cache_dir="configured-cache",
                split="test", n_tokens=4, text_file=path)
            assert ids == [0, 1, 2, 3]
            return corpus
    raw = b'\xef\xbb\xbfraw\r\n\xc3\xa9\n\n'
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    file_record = invoke(str(path))
    assert file_record["corpus_sha256"] == hashlib.sha256(raw).hexdigest()
    assert seen[-1] == (raw.decode("utf-8"), {"return_tensors": "pt", "add_special_tokens": False})
    assert not calls
    joined = b'\xc3\xa9\r\nkept \n\n\xe7\xb5\x82'
    dataset_record = invoke()
    assert dataset_record["corpus_sha256"] == hashlib.sha256(joined).hexdigest()
    assert seen[-1][0].encode() == joined
    assert calls[0][1]["split"] == ("train" if consumer == "kl" else "test")
    if consumer == "kl":
        assert file_record["bytes"] == len(raw)
        assert dataset_record["bytes"] == len(joined)
        assert dataset_record["datasets_fingerprint"] == ds._fingerprint
    else:
        assert file_record["fingerprint"] == "file:sha256:" + hashlib.sha256(raw).hexdigest()
        assert dataset_record["fingerprint"] == ds._fingerprint

