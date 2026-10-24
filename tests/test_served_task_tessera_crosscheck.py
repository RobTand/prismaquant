"""Offline cross-check against an unchanged committed PUBLIC producer fixture.

No raw-GitHub fetching, skip-on-network-failure, serving runtime, or installed
package upgrade is required. See fixtures/tessera_endpoint_public/README.md.
"""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from _served_task_fixture import PUBLIC_CLI, case, public_call, resign, write_json
from prismaquant import task_suite

pytestmark = pytest.mark.task_suite


def _oracle(config, expected, tmp_path):
    runtime = config["backend"]["serving_runtime"]
    artifacts = write_json(tmp_path / "oracle_artifacts.json", expected["artifacts"])
    tokenizer = write_json(tmp_path / "oracle_tokenizer.json", expected["tokenizer"])
    proc = subprocess.run(
        [sys.executable, str(PUBLIC_CLI), runtime["witness"], "--offline",
         "--served-dir", runtime["served_dir"], "--tokenizer-dir", runtime["tokenizer_dir"],
         "--expect-endpoint", expected["endpoint"], "--expect-alias", expected["served_alias"],
         "--expect-attempt", expected["attempt_id"], "--expect-ranks", "0,1",
         "--expect-artifacts", str(artifacts), "--expect-tokenizer", str(tokenizer)],
        capture_output=True, text=True, timeout=10, check=False)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    verdict = json.loads(proc.stdout)
    assert verdict["proof_scope"] == "recorded_runtime_byte_binding"
    assert verdict["current_endpoint_verified"] is False


@pytest.mark.parametrize("kind", ["plain", "bytelevel", "nested-bytelevel"])
def test_archived_producer_fixture_is_valid_offline(tmp_path, kind):
    # Guard against bad test data before blaming the consumer. The same command
    # also selects the consumer regressions below, which fail on unbound main.
    if kind == "plain":
        config, _, expected = case(tmp_path)
    else:
        config, _, expected = normalized_case(tmp_path, nested=kind.startswith("nested"))
    _oracle(config, expected, tmp_path)


def test_public_oracle_fixture_provenance_is_unchanged():
    root = PUBLIC_CLI.parents[1]
    for path, digest in {
        "src/tessera/endpoint_witness.py": "d5dc6b1ee9aba42a2a803cb9ac1a0dc4ee0dae8ab2e5b7c5cd3a060917bdc14a",
        "tools/verify_endpoint_witness.py": "e08369a7021bf3fed3548e6a7544ed422305f55679d199acb5b21fca4c6c8c2d",
    }.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def consumer():
    name = "prismaquant.served_task_public_verifier"
    assert importlib.util.find_spec(name) is not None, (
        "PR #2470 public verifier consumer has not been reconciled onto main")
    return importlib.import_module(name)


def normalized_case(tmp_path, *, nested=False):
    config, witness, expected = case(tmp_path)
    file_backend = copy.deepcopy(expected["tokenizer"]["backend"])
    pre = {"type": "ByteLevel", "add_prefix_space": False,
           "trim_offsets": True, "use_regex": True}
    decoder = {"type": "ByteLevel", "add_prefix_space": True,
               "trim_offsets": True, "use_regex": True}
    if nested:
        file_backend["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [pre]}
        file_backend["decoder"] = {"type": "Sequence", "decoders": [decoder]}
    else:
        file_backend.update(pre_tokenizer=pre, decoder=decoder)
    loaded_backend = copy.deepcopy(file_backend)
    loaded_pre = loaded_backend["pre_tokenizer"]
    loaded_decoder = loaded_backend["decoder"]
    if nested:
        loaded_pre = loaded_pre["pretokenizers"][0]
        loaded_decoder = loaded_decoder["decoders"][0]
    loaded_pre["trim_offsets"] = False
    for key in ("add_prefix_space", "trim_offsets", "use_regex"):
        loaded_decoder[key] = False
    raw = json.dumps(file_backend).encode()
    runtime = config["backend"]["serving_runtime"]
    (Path(runtime["tokenizer_dir"]) / "tokenizer.json").write_bytes(raw)
    fact = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    expected["tokenizer"]["files"]["tokenizer.json"] = fact
    expected["tokenizer"]["backend"] = copy.deepcopy(loaded_backend)
    witness["tokenizer"]["files"]["tokenizer.json"] = dict(fact, content=file_backend)
    witness["tokenizer"]["backend"] = loaded_backend
    write_json(Path(runtime["witness"]), resign(witness))
    write_json(Path(runtime["expected"]), expected)
    return config, witness, expected


def test_complete_fixture_checked_by_public_process(tmp_path):
    config, _, expected = case(tmp_path)
    verdict = consumer().run_public_verifier(**public_call(config, expected))
    assert verdict == {"schema": "tessera.endpoint_witness_verdict.v1",
                       "mode": "offline", "verdict": "valid", "reason": None,
                       "proof_scope": "recorded_runtime_byte_binding",
                       "current_endpoint_verified": False}


@pytest.mark.parametrize("nested", [False, True])
def test_supported_bytelevel_normalization_not_overruled_locally(tmp_path, nested):
    config, _, expected = normalized_case(tmp_path, nested=nested)
    verdict = consumer().run_public_verifier(**public_call(config, expected))
    assert verdict["verdict"] == "valid"
    # This is the key drift regression: old local _check_join used exact
    # backend==tokenizer.json equality and refused producer-supported loading.
    record = task_suite.preflight_tasks(config)
    assert record["binding"]["public_verifier"] == verdict


@pytest.mark.parametrize("damage", ["prefix-space", "regex", "decoder-type", "vocab"])
def test_bytelevel_token_id_affecting_changes_are_not_ignored(tmp_path, damage):
    config, witness, expected = normalized_case(tmp_path)
    loaded = witness["tokenizer"]["backend"]
    if damage == "prefix-space":
        loaded["pre_tokenizer"]["add_prefix_space"] = True
    elif damage == "regex":
        loaded["pre_tokenizer"]["use_regex"] = False
    elif damage == "decoder-type":
        loaded["decoder"]["type"] = "OtherDecoder"
    else:
        loaded["model"]["vocab"] = {"<s>": 1, "a": 0}
        witness["tokenizer"]["vocab"] = copy.deepcopy(loaded["model"]["vocab"])
        expected["tokenizer"]["vocab"] = copy.deepcopy(loaded["model"]["vocab"])
    expected["tokenizer"]["backend"] = copy.deepcopy(loaded)
    runtime = config["backend"]["serving_runtime"]
    write_json(Path(runtime["witness"]), resign(witness))
    write_json(Path(runtime["expected"]), expected)
    with pytest.raises(ValueError, match="refus"):
        consumer().run_public_verifier(**public_call(config, expected))
    with pytest.raises(ValueError, match="refus|tokenizer"):
        task_suite.preflight_tasks(config)


def test_offline_public_process_cannot_import_runtime_or_use_network(tmp_path):
    config, _, expected = case(tmp_path)
    # Run the unchanged CLI behind a test-only guard, not a producer substitute.
    driver = tmp_path / "guarded_public_cli.py"
    driver.write_text('''import builtins, runpy, socket, sys, urllib.request
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'torch', 'transformers', 'lm_eval'} or name.startswith('tessera.serving') or name == 'tessera.endpoint_observer':
        raise AssertionError('offline verifier imported runtime: ' + name)
    return original_import(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('offline verifier tried network access')
builtins.__import__ = guarded_import
socket.socket.connect = forbidden
socket.create_connection = forbidden
urllib.request.urlopen = forbidden
runpy.run_path(''' + repr(str(PUBLIC_CLI)) + ''', run_name='__main__')
''')
    config["backend"]["serving_runtime"]["verifier"] = str(driver)
    verdict = consumer().run_public_verifier(**public_call(config, expected))
    assert verdict["current_endpoint_verified"] is False
    assert task_suite.preflight_tasks(config)["binding"]["public_verifier"] == verdict
