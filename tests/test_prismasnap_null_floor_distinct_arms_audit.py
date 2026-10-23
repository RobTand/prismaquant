from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def validation_fixture() -> ModuleType:
    fixture_path = Path.cwd() / "tests" / "test_prismasnap_validation.py"
    spec = importlib.util.spec_from_file_location(
        "_prismasnap_null_floor_fixture_audit", fixture_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attestation_inputs(
    tmp_path: Path, validation_fixture: ModuleType
) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint, student, teacher, source_identity = validation_fixture._fixture(
        tmp_path, kl=6.2e-4
    )
    receipt_path = validation_fixture._null_floor_receipt(
        tmp_path, student, checkpoint
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for index, arm in enumerate(receipt["arms"]):
        for key in ("student_result_sha256", "splice_receipt_sha256"):
            arm[key] = hashlib.sha256(f"{key}:{index}".encode()).hexdigest()
    validation_fixture._write_json(receipt_path, receipt)
    return checkpoint, student, teacher, source_identity, receipt_path


@pytest.mark.parametrize(
    "reused_field", ["arm", "student_result_sha256", "splice_receipt_sha256"]
)
def test_attestation_refuses_reused_null_arm_identity_or_evidence(
    tmp_path: Path, validation_fixture: ModuleType, reused_field: str
) -> None:
    checkpoint, student, teacher, source_identity, receipt_path = _attestation_inputs(
        tmp_path, validation_fixture
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arms"][1][reused_field] = receipt["arms"][0][reused_field]
    validation_fixture._write_json(receipt_path, receipt)
    provenance_path = checkpoint / validation_fixture.PROVENANCE_JSON
    materialized = provenance_path.read_bytes()

    with pytest.raises(RuntimeError, match="independent|duplicate|distinct"):
        validation_fixture.attest_fold_fidelity(
            checkpoint,
            student,
            teacher,
            source_identity,
            null_floor_receipt_path=receipt_path,
        )

    assert provenance_path.read_bytes() == materialized


def test_attestation_accepts_distinct_arm_evidence_on_same_serving_stack(
    tmp_path: Path, validation_fixture: ModuleType
) -> None:
    checkpoint, student, teacher, source_identity, receipt_path = _attestation_inputs(
        tmp_path, validation_fixture
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["arms"][0]["serve_fingerprint"] == receipt["arms"][1][
        "serve_fingerprint"
    ]

    result = validation_fixture.attest_fold_fidelity(
        checkpoint,
        student,
        teacher,
        source_identity,
        null_floor_receipt_path=receipt_path,
    )

    assert result["state"] == "VERIFIED"
    fold = result["fold_fidelity"]
    assert fold["threshold"] == max(5e-4, 2.0 * 6.5e-4)
    assert fold["threshold_derivation"]["arms"] == receipt["arms"]
    validation_fixture.validate_prismasnap_provenance_payload(
        result, require_verified=True, where="distinct null-arm audit"
    )
    validation_fixture.require_verified_prismasnap_if_present(checkpoint)
    assert validation_fixture.attest_fold_fidelity(
        checkpoint,
        student,
        teacher,
        source_identity,
        null_floor_receipt_path=receipt_path,
    ) == result


@pytest.mark.parametrize(
    "reused_field", ["arm", "student_result_sha256", "splice_receipt_sha256"]
)
def test_persisted_derivation_refuses_reused_identity_or_evidence(
    tmp_path: Path, validation_fixture: ModuleType, reused_field: str
) -> None:
    checkpoint, student, teacher, source_identity, receipt_path = _attestation_inputs(
        tmp_path, validation_fixture
    )
    validation_fixture.attest_fold_fidelity(
        checkpoint,
        student,
        teacher,
        source_identity,
        null_floor_receipt_path=receipt_path,
    )
    provenance_path = checkpoint / validation_fixture.PROVENANCE_JSON
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    arms = payload["fold_fidelity"]["threshold_derivation"]["arms"]
    arms[1][reused_field] = arms[0][reused_field]
    payload["provenance_sha256"] = validation_fixture.validation._provenance_digest(
        payload
    )

    with pytest.raises(RuntimeError, match="independent|duplicate|distinct"):
        validation_fixture.validate_prismasnap_provenance_payload(
            payload, require_verified=True, where="persisted null-arm audit"
        )

    validation_fixture._write_json(provenance_path, payload)
    with pytest.raises(RuntimeError, match="independent|duplicate|distinct"):
        validation_fixture.require_verified_prismasnap_if_present(checkpoint)
