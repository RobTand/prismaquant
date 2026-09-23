"""A chain-mode quantum refuses a staged manifest its record does not seal.

PQ #1008 (finding 9 of #996). The band-serial branch already refuses a
``--data-manifest-sha256`` that is not the readset its handoff derives
(``require_band_serial_readset``). Chain mode had no such check: any digest,
or none, reached ``bind_residency_manifest``. The quantum now checks the
digest against the one its sealed record names, with the rule the
dispatcher uses to choose what it stages (``_row_manifest_sha256`` and the
``executable is not None`` branch of ``quantum_argv``).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.joint_cost_quantum import (
    EXIT_IDENTITY_REFUSED,
    IDENTITY_REFUSED_MARKER,
    QuantumIdentityRefused,
    main,
)

from test_joint_cost_quantum_runtime import (  # noqa: F401 (fixture)
    _hex,
    _offline_tier_policy,
    _valid_record,
    identity_files,
)
from dispatch_joint_quanta import _row_manifest_sha256  # noqa: E402


def chain_readset_sha256(record):
    # Imported at call time, so the CLI tests below run (and fail on
    # behaviour, not on import) against a tree that predates the check.
    from prismaquant.joint_cost_quantum import chain_readset_sha256 as sealed
    return sealed(record)


def require_chain_readset(record, **kwargs):
    from prismaquant.joint_cost_quantum import require_chain_readset as check
    return check(record, **kwargs)


def _executable(record, *, digest=_hex("c")):
    record = json.loads(json.dumps(record))
    record.pop("identity_sha256")
    record["executable_readset"] = {
        "manifest_path": "executable.json.gz", "manifest_sha256": digest,
        "phases": ["head", "checkpoint-load"]}
    record["identity_sha256"] = canonical_json_sha256(record, where="record")
    return record


def _argv(identity_files, record, *extra):
    record_path = identity_files["output_root"].parent / "record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return [
        "--quantum", str(record_path),
        "--quantum-sha256", hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "--plan", str(identity_files["plan"][0]),
        "--plan-sha256", identity_files["plan"][1],
        "--prepared", str(identity_files["prepared"][0]),
        "--prepared-sha256", identity_files["prepared"][1],
        "--adjoint-slice", str(identity_files["adjoint"][0]),
        "--adjoint-slice-sha256", identity_files["adjoint"][1],
        "--output-root", str(identity_files["output_root"]),
        *extra,
    ]


class _GatePassed(Exception):
    """Raised by the first step after the identity gates."""


@pytest.fixture
def past_the_gate(monkeypatch):
    """Stop ``main`` at its first step after the identity gates."""
    import prismaquant.tessera_joint_aura as aura_mod

    def stop(*_args, **_kwargs):
        raise _GatePassed()

    monkeypatch.setattr(aura_mod, "_load_plan", stop)


def test_sealed_digest_follows_the_dispatcher_rule(identity_files):
    legacy = _valid_record(identity_files)
    executable = _executable(legacy)
    assert chain_readset_sha256(legacy) == legacy["read_set"]["manifest_sha256"]
    assert chain_readset_sha256(executable) == _hex("c")
    for record in (legacy, executable):
        assert chain_readset_sha256(record) == _row_manifest_sha256(record)


@pytest.mark.parametrize("block", [
    "not-a-block", {"manifest_path": "x"}, {"manifest_sha256": "abc"},
])
def test_malformed_executable_block_refuses(identity_files, block):
    record = _valid_record(identity_files)
    record["executable_readset"] = block
    # The dispatcher refuses such a row before publishing it (quantum_argv
    # takes the executable branch for any non-None block); the quantum
    # refuses too, rather than fall back to the slice manifest.
    with pytest.raises(QuantumIdentityRefused, match="executable readset"):
        chain_readset_sha256(record)


@pytest.mark.parametrize("digest", [None, "", "B" * 64, _hex("0")])
def test_helper_refuses_a_missing_or_other_digest(identity_files, digest):
    record = _valid_record(identity_files)
    with pytest.raises(QuantumIdentityRefused, match="data manifest"):
        require_chain_readset(record, data_manifest_sha256=digest)


def test_helper_accepts_the_sealed_digest(identity_files):
    record = _valid_record(identity_files)
    require_chain_readset(record, data_manifest_sha256=_hex("b"))
    require_chain_readset(_executable(record), data_manifest_sha256=_hex("c"))


@pytest.mark.parametrize("extra", [
    (),
    ("--data-manifest-sha256", _hex("0")),
    # A legacy digest on an executable row: the row stages the other one.
    ("--data-manifest-sha256", _hex("b")),
])
def test_chain_cli_refuses_an_unsealed_manifest(
        identity_files, monkeypatch, capsys, past_the_gate, extra):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    record = _executable(_valid_record(identity_files))
    code = main(_argv(identity_files, record, *extra))
    assert code == EXIT_IDENTITY_REFUSED
    out = capsys.readouterr().out
    assert IDENTITY_REFUSED_MARKER in out
    assert "data manifest" in out
    assert not identity_files["output_root"].exists()


@pytest.mark.parametrize("executable", [False, True])
def test_chain_cli_passes_the_sealed_manifest(
        identity_files, monkeypatch, past_the_gate, executable):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    record = _valid_record(identity_files)
    if executable:
        record = _executable(record)
    sealed = (record["executable_readset"]["manifest_sha256"] if executable
              else record["read_set"]["manifest_sha256"])
    with pytest.raises(_GatePassed):
        main(_argv(identity_files, record, "--data-manifest-sha256", sealed))
