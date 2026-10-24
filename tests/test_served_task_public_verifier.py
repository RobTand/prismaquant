"""Acceptance checks for the public verifier process, not the local pre-check.

Adapted from 80f95d0271 without its backend-fixture or application imports at
collection time. Tiny CLI stubs exercise the subprocess contract; they make no
claim about Tessera witness validity or a currently running endpoint.
"""
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.task_suite


def _public_verifier():
    """Missing functionality must fail a test, not break pytest collection."""
    name = "prismaquant.served_task_public_verifier"
    assert importlib.util.find_spec(name) is not None, (
        "The served backend needs a public verifier process authority: "
        "prismaquant.served_task_public_verifier must provide "
        "run_public_verifier rather than trusting only the local pre-check."
    )
    return importlib.import_module(name)


def _expected():
    # Independent CLI inputs: no backend witness builder or installed runtime.
    return {
        "endpoint": "http://fixture.invalid:8142",
        "served_alias": "fixture-alias",
        "attempt_id": "fixture-attempt",
        "ranks": [0, 1],
        "artifacts": {"model.safetensors": {"sha256": "a" * 64, "bytes": 4}},
        "tokenizer": {
            "files": {"tokenizer.json": {"sha256": "b" * 64, "bytes": 2}},
            "vocab": {"token": 0},
        },
    }


def _valid_verdict():
    return {
        "schema": "tessera.endpoint_witness_verdict.v1",
        "mode": "offline",
        "verdict": "valid",
        "reason": None,
        "proof_scope": "recorded_runtime_byte_binding",
        "current_endpoint_verified": False,
    }


def _setup(tmp_path, stdout, *, status=0, prelude=""):
    witness = tmp_path / "witness.json"
    witness.write_text("{}\n", encoding="utf-8")
    served_dir = tmp_path / "served"
    served_dir.mkdir()
    cli = tmp_path / "stub_cli.py"
    cli.write_text(
        "import json\nimport sys\nfrom pathlib import Path\n"
        + prelude
        + f"\nsys.stdout.write({stdout!r})\nraise SystemExit({status!r})\n",
        encoding="utf-8",
    )
    return {
        "cli": cli,
        "witness": witness,
        "expected": _expected(),
        "served_dir": served_dir,
        "python": sys.executable,
        "timeout": 10,
    }


@pytest.mark.parametrize("pretty", [False, True], ids=["compact", "multiline"])
def test_valid_offline_verdict_passes_and_forwards_cli_inputs(tmp_path, pretty):
    public = _public_verifier()
    verdict = _valid_verdict()
    report = tmp_path / "invocation.json"
    # Read temporary expected-input files inside the child before they vanish.
    prelude = f"""
argv = sys.argv[1:]
def opt(name):
    return argv[argv.index(name) + 1]
Path({str(report)!r}).write_text(json.dumps({{
    "argv": argv,
    "artifacts": json.loads(Path(opt("--expect-artifacts")).read_text()),
    "tokenizer": json.loads(Path(opt("--expect-tokenizer")).read_text()),
}}))
"""
    kwargs = _setup(
        tmp_path, json.dumps(verdict, indent=2 if pretty else None) + "\n",
        prelude=prelude,
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    kwargs["tokenizer_dir"] = tokenizer_dir

    assert public.run_public_verifier(**kwargs) == verdict
    invocation = json.loads(report.read_text(encoding="utf-8"))
    argv = invocation["argv"]
    assert argv[0] == str(kwargs["witness"])
    assert argv.count("--offline") == 1
    for flag, value in {
        "--served-dir": str(kwargs["served_dir"]),
        "--tokenizer-dir": str(tokenizer_dir),
        "--expect-endpoint": kwargs["expected"]["endpoint"],
        "--expect-alias": kwargs["expected"]["served_alias"],
        "--expect-attempt": kwargs["expected"]["attempt_id"],
        "--expect-ranks": "0,1",
    }.items():
        assert argv.count(flag) == 1
        assert argv[argv.index(flag) + 1] == value
    assert invocation["artifacts"] == kwargs["expected"]["artifacts"]
    assert invocation["tokenizer"] == kwargs["expected"]["tokenizer"]


def test_missing_cli_refuses(tmp_path):
    public = _public_verifier()
    kwargs = _setup(tmp_path, json.dumps(_valid_verdict()))
    kwargs["cli"] = tmp_path / "absent.py"
    with pytest.raises(ValueError, match="absent"):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize("status", [1, 4])
def test_nonzero_exit_refuses_even_with_valid_stdout(tmp_path, status):
    public = _public_verifier()
    kwargs = _setup(tmp_path, json.dumps(_valid_verdict()), status=status)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)


def test_timeout_refuses_without_waiting(tmp_path, monkeypatch):
    public = _public_verifier()
    kwargs = _setup(tmp_path, json.dumps(_valid_verdict()))
    kwargs["timeout"] = 7
    calls = []

    def timeout_run(argv, **options):
        calls.append((argv, options))
        # Even a partial valid verdict cannot overrule a process timeout.
        raise subprocess.TimeoutExpired(
            argv, options["timeout"], output=json.dumps(_valid_verdict())
        )

    monkeypatch.setattr(subprocess, "run", timeout_run)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 7


def test_oserror_refuses(tmp_path, monkeypatch):
    public = _public_verifier()
    kwargs = _setup(tmp_path, json.dumps(_valid_verdict()))
    calls = []

    def failed_run(argv, **options):
        calls.append(argv)
        raise OSError("fixture process could not start")

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [0, 4])
def test_cli_refusal_refuses(tmp_path, status):
    public = _public_verifier()
    verdict = _valid_verdict()
    verdict.update(verdict="refused", reason="REFUSED: stub", proof_scope=None)
    kwargs = _setup(tmp_path, json.dumps(verdict), status=status)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "tessera.endpoint_witness_verdict.v999"),
        ("mode", "online"),
        ("proof_scope", "served_task_quality"),
        ("current_endpoint_verified", True),
    ],
    ids=["unknown-schema", "wrong-mode", "wrong-scope", "live-endpoint-claim"],
)
def test_verdict_contract_mismatch_refuses(tmp_path, field, value):
    public = _public_verifier()
    verdict = _valid_verdict()
    verdict[field] = value
    kwargs = _setup(tmp_path, json.dumps(verdict))
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize(
    "stdout",
    ["", " \n\t", "{broken", "null", "[]", '"valid"', "1", "true"],
    ids=["empty", "whitespace", "malformed", "null", "array", "string",
         "number", "boolean"],
)
def test_missing_malformed_or_nonobject_output_refuses(tmp_path, stdout):
    public = _public_verifier()
    kwargs = _setup(tmp_path, stdout)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize(
    "variant",
    ["diagnostics-before", "diagnostics-after", "two-objects",
     "duplicate-key", "NaN", "Infinity", "-Infinity"],
)
def test_stdout_must_be_one_strict_machine_json_object(tmp_path, variant):
    public = _public_verifier()
    valid = json.dumps(_valid_verdict())
    if variant == "diagnostics-before":
        stdout = "diagnostic: verifier started\n" + valid + "\n"
    elif variant == "diagnostics-after":
        stdout = valid + "\ndiagnostic: verifier finished\n"
    elif variant == "two-objects":
        stdout = valid + "\n" + valid + "\n"
    elif variant == "duplicate-key":
        # A permissive parser keeps the last value and would accept this.
        stdout = '{"verdict": "refused", ' + valid[1:]
    else:
        # Non-finite numbers are forbidden even in otherwise unused fields.
        stdout = valid[:-1] + ', "diagnostic_value": ' + variant + "}"
    kwargs = _setup(tmp_path, stdout)
    with pytest.raises(ValueError):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize("missing", ["witness", "served_dir", "tokenizer_dir"])
def test_missing_verifier_inputs_refuse(tmp_path, missing):
    public = _public_verifier()
    kwargs = _setup(tmp_path, json.dumps(_valid_verdict()))
    kwargs[missing] = tmp_path / "absent"
    with pytest.raises(ValueError, match="absent"):
        public.run_public_verifier(**kwargs)


@pytest.mark.parametrize("expected", [{}, {"artifacts": {}, "tokenizer": {}}])
def test_split_expected_names_missing_inputs(expected):
    public = _public_verifier()
    with pytest.raises(ValueError, match="CLI inputs"):
        public.split_expected(expected)


def test_verifier_cli_names_absent_checkout(tmp_path):
    public = _public_verifier()
    with pytest.raises(ValueError, match="absent"):
        public.verifier_cli(tmp_path)
