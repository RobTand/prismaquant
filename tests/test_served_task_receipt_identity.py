"""Owned receipt regressions using the actual test-only public producer CLI.

Synthetic CPU evidence only; no live endpoint, launch or measurement claim.
"""
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from _served_task_fixture import case, resign, write_json
from prismaquant.served_task_backend import bind_served_task

pytestmark = pytest.mark.task_suite


@pytest.mark.parametrize("original_valid", [False, True], ids=["invalid-to-valid", "valid-to-valid"])
def test_binding_verifies_owned_receipt_despite_source_replacement(tmp_path, monkeypatch, original_valid):
    config, original, _ = case(tmp_path)
    source = Path(config["backend"]["serving_runtime"]["witness"])
    replacement = deepcopy(original)
    replacement["lifetime"]["request_id"] = "replacement-request"
    for rank in replacement["artifacts"]:
        rank["request_id"] = "replacement-request"
    replacement["tokenizer"]["request_id"] = "replacement-request"
    resign(replacement)
    if not original_valid:
        original["fingerprint"] = "0" * 64
        write_json(source, original)
    original_bytes = source.read_bytes()
    real_run = subprocess.run
    invocations = []

    def replace_source_then_run(argv, **kwargs):
        write_json(source, replacement)
        invocations.append((Path(argv[2]), Path(argv[2]).read_bytes()))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", replace_source_then_run)
    if original_valid:
        record = bind_served_task(config)
        assert record["witness_fingerprint"] == original["fingerprint"]
        assert record["witness_fingerprint"] != replacement["fingerprint"]
        assert record["public_verifier"]["verdict"] == "valid"
    else:
        with pytest.raises(ValueError, match="public verifier"):
            bind_served_task(config)
    assert len(invocations) == 1
    verified_path, verified_bytes = invocations[0]
    assert verified_path != source
    assert verified_bytes == original_bytes
    assert not verified_path.exists()  # owned temporary receipt is cleaned up
