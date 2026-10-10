"""Exercise the public verifier call contract and its refusal gates."""
import copy
import json
import sys

import pytest

from test_served_task_backend import make_expected, make_witness
from prismaquant.served_task_public_verifier import (
    TESSERA_VERIFIER_COMMIT,
    run_public_verifier,
    split_expected,
    verifier_cli,
)

STUB = """
import json
import sys
argv = sys.argv[1:]
def opt(name):
    return argv[argv.index(name) + 1]
alias = opt("--expect-alias")
if alias == "refuse-me":
    verdict = {"schema": "tessera.endpoint_witness_verdict.v1",
               "mode": "offline", "verdict": "refused",
               "reason": "REFUSED: stub", "proof_scope": None,
               "current_endpoint_verified": False}
    print(json.dumps(verdict))
    raise SystemExit(4)
scope = ("other-scope" if alias == "wrong-scope"
         else "recorded_runtime_byte_binding")
verdict = {"schema": "tessera.endpoint_witness_verdict.v1",
           "mode": "offline", "verdict": "valid", "reason": None,
           "proof_scope": scope, "current_endpoint_verified": False}
print(json.dumps(verdict))
"""


def _setup(tmp_path):
    witness = make_witness()
    witness_path = tmp_path / "witness.json"
    witness_path.write_text(json.dumps(witness))
    served_dir = tmp_path / "served"
    served_dir.mkdir()
    cli = tmp_path / "stub_cli.py"
    cli.write_text(STUB)
    return witness_path, served_dir, cli


def test_valid_offline_verdict_passes(tmp_path):
    witness_path, served_dir, cli = _setup(tmp_path)
    verdict = run_public_verifier(
        cli=cli, witness=witness_path,
        expected=make_expected(make_witness()), served_dir=served_dir,
        python=sys.executable, timeout=60)
    assert verdict["verdict"] == "valid"
    assert verdict["proof_scope"] == "recorded_runtime_byte_binding"


def test_cli_refusal_refuses(tmp_path):
    witness_path, served_dir, cli = _setup(tmp_path)
    expected = make_expected(make_witness())
    expected["served_alias"] = "refuse-me"
    with pytest.raises(ValueError, match="refused"):
        run_public_verifier(cli=cli, witness=witness_path,
                            expected=expected, served_dir=served_dir,
                            python=sys.executable, timeout=60)


def test_overstated_scope_refuses(tmp_path):
    witness_path, served_dir, cli = _setup(tmp_path)
    expected = make_expected(make_witness())
    expected["served_alias"] = "wrong-scope"
    with pytest.raises(ValueError, match="proof scope"):
        run_public_verifier(cli=cli, witness=witness_path,
                            expected=expected, served_dir=served_dir,
                            python=sys.executable, timeout=60)


def test_missing_cli_refuses(tmp_path):
    witness = make_witness()
    with pytest.raises(ValueError, match="absent"):
        run_public_verifier(cli=tmp_path / "absent.py",
                            witness=tmp_path / "absent.json",
                            expected=make_expected(witness),
                            served_dir=tmp_path)


def test_missing_served_dir_refuses(tmp_path):
    _, _, cli = _setup(tmp_path)
    witness = make_witness()
    witness_path = tmp_path / "witness.json"
    witness_path.write_text(json.dumps(witness))
    with pytest.raises(ValueError, match="served artifact dir"):
        run_public_verifier(cli=cli, witness=witness_path,
                            expected=make_expected(witness),
                            served_dir=tmp_path / "absent")


def test_split_expected_names_missing_inputs():
    with pytest.raises(ValueError, match="CLI inputs"):
        split_expected({})
    with pytest.raises(ValueError, match="CLI inputs"):
        split_expected({"artifacts": {}, "tokenizer": {}})


def test_verifier_cli_names_absent_checkout(tmp_path):
    with pytest.raises(ValueError, match="absent"):
        verifier_cli(tmp_path)


def test_pin_is_explicit():
    assert len(TESSERA_VERIFIER_COMMIT) == 40
