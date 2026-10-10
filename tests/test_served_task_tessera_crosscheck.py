"""Cross-check the consumer against the real Tessera verifier.

Fetch ``src/tessera/endpoint_witness.py`` at the pinned Tessera commit and
run its ``check_expectations`` on the same cases as the consumer. The two
must agree on the pass and on every refusal class. This guards the pinned
transcription against drift. It never imports the serving runtime, starts
no rank, and runs no model inference.
"""
import copy
import importlib.util
import urllib.request

import pytest

from test_served_task_backend import make_expected, make_witness
from prismaquant.served_task_backend import verify
from prismaquant.served_task_public_verifier import TESSERA_VERIFIER_COMMIT

RAW_URL = ("https://raw.githubusercontent.com/RobTand/tessera/"
           + TESSERA_VERIFIER_COMMIT + "/src/tessera/endpoint_witness.py")


def _tessera():
    try:
        with urllib.request.urlopen(RAW_URL, timeout=60) as response:
            source = response.read().decode("utf-8")
    except Exception as exc:
        pytest.skip(f"Tessera pin is unreachable: {exc}")
    name = "tessera_pinned_endpoint_witness"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, RAW_URL, "exec"), module.__dict__)
    assert module.SCHEMA == "tessera.endpoint_runtime_witness.v1"
    return module


def _tessera_verdict(tessera, witness, expected):
    try:
        assert tessera.check_expectations(witness, expected) is None
    except ValueError:
        return "refuse"
    return "pass"


def _cases():
    witness = make_witness()
    expected = make_expected(witness)
    cases = [("pass", witness, expected)]

    missing = copy.deepcopy(witness)
    del missing["launch"]
    cases.append(("missing fact", missing, copy.deepcopy(expected)))

    alias_only = copy.deepcopy(expected)
    alias_only["artifacts"] = {}
    alias_only["tokenizer"]["files"] = {}
    cases.append(("alias only", copy.deepcopy(witness), alias_only))

    size_only = copy.deepcopy(expected)
    size_only["artifacts"] = {name: {"bytes": fact["bytes"]}
                              for name, fact in size_only["artifacts"].items()}
    cases.append(("size only", copy.deepcopy(witness), size_only))

    short_ranks = copy.deepcopy(expected)
    short_ranks["ranks"] = [0]
    cases.append(("incomplete ranks", copy.deepcopy(witness), short_ranks))

    bad_join = copy.deepcopy(expected)
    bad_join["endpoint"] = "http://127.0.0.1:9999"
    cases.append(("join mismatch", copy.deepcopy(witness), bad_join))

    bad_digest = copy.deepcopy(expected)
    name = next(iter(bad_digest["artifacts"]))
    bad_digest["artifacts"][name]["sha256"] = "0" * 64
    cases.append(("artifact digest", copy.deepcopy(witness), bad_digest))

    bad_tokenizer = copy.deepcopy(expected)
    bad_tokenizer["tokenizer"]["vocab"]["a"] = 7
    cases.append(("tokenizer content", copy.deepcopy(witness), bad_tokenizer))
    return cases


def test_pinned_tessera_accepts_complete_witness():
    tessera = _tessera()
    witness = make_witness()
    expected = make_expected(witness)
    assert _tessera_verdict(tessera, witness, expected) == "pass"
    assert verify(witness, expected)["verdict"] == "pass"


def test_consumer_agrees_with_pinned_tessera():
    tessera = _tessera()
    for label, witness, expected in _cases():
        mine = verify(witness, expected)["verdict"]
        theirs = _tessera_verdict(tessera, witness, expected)
        assert mine == theirs, f"case {label}: consumer {mine}, Tessera {theirs}"
        if label != "pass":
            assert mine == "refuse", f"case {label} must refuse"
