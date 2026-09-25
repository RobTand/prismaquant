"""A band-serial dry run publishes nothing; a template names ``--output-root`` (PQ #1200).

``dispatch_joint_quanta --dry-run --band-serial`` resolved each fresh row's
band role before its ``--dry-run`` branch, and resolving it published the
producer's handoff template and the consumer's band-serial readset. The
template's ``output_prefix`` also came from the record's sealed output space
while the file went under ``--output-root``, so a dispatch under another
root filed a template under one root that named the other.

The fixture is ``test_band_serial_dispatch``'s two-band campaign, whose
records name ``<tmp>/run`` while every dispatch here passes
``--output-root <tmp>/out``. PrismaBuild's template validator is replaced by
the identity, so these tests run where no published PrismaBuild is visible;
``test_band_serial_dispatch`` and ``test_band_serial_handoff_produced``
validate the same template through PrismaBuild on the fleet.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_band_serial_dispatch as band  # noqa: E402
from prismaquant.joint_quantum_handoff import (  # noqa: E402
    HANDOFF_LOAD_PHASE, handoff_root)
from prismaquant.readset_coverage import load_manifest  # noqa: E402

BAND = ("--band-serial", "--handoff-tier", band.TIER)


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Run outside campaign scope with no staged-tier policy (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _identity_template_validator(monkeypatch):
    """PrismaBuild's ``produced_output`` template check, as the identity."""
    import prismaquant.stage_a_produced_output as produced

    monkeypatch.setattr(produced, "_produced_output_module", lambda: SimpleNamespace(
        TEMPLATE_SCHEMA_V1="fixture.produced-output-template.v1",
        validate_template=dict))


def _layout(tmp_path, monkeypatch):
    monkeypatch.setattr(band, "_pin_published_helper_root",
                        _identity_template_validator)
    return band._dispatch_layout(tmp_path, monkeypatch)


def _run(dispatch, gateway, records, receipt_path, out, *extra, coverage=None):
    return dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path), *BAND, *extra],
        _gateway=gateway,
        # The fixture model has no checkpoint to plan sources from; the
        # coverage gate has its own tests (test_readset_coverage_1095).
        _coverage=coverage or (lambda rows: []))


def _files(root: Path) -> dict[str, str]:
    """Every file under ``root`` and its digest."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _dry_rows(capsys) -> dict[str, dict]:
    printed = json.loads(capsys.readouterr().out)
    return {row["quantum_id"]: row for row in printed["rows"]
            if row["kind"] == "quantum"}


def test_a_dry_run_publishes_no_handoff_template(tmp_path, monkeypatch, capsys):
    dispatch, bound, records, receipt_path, out = _layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    before = _files(tmp_path)
    capsys.readouterr()

    assert _run(dispatch, gateway, records, receipt_path, out, "--dry-run") == 0
    assert gateway.submitted == []
    # Nothing written anywhere: no template, no readset, no state.
    assert _files(tmp_path) == before
    rows = _dry_rows(capsys)
    assert sorted(rows) == ["layer-001", "layer-003"]
    band_serial = out / "layer-quanta" / dispatch.BAND_SERIAL_DIRECTORY
    for quantum_id, row in rows.items():
        # The template the real run would publish: its path, digest and body.
        path = Path(row["handoff_template"])
        assert path.parent == band_serial
        assert not path.exists()
        template = row["handoff_template_document"]
        payload = (json.dumps(template, sort_keys=True, indent=2) + "\n").encode()
        assert row["handoff_template_sha256"] == hashlib.sha256(payload).hexdigest()
        assert path.name == (f"{quantum_id}.handoff-template."
                             f"{row['handoff_template_sha256'][:16]}.json")
        # The prefix and the file derive from one root, --output-root, not
        # the <tmp>/run the record seals.
        assert template["output_prefix"] == str(
            handoff_root(out / "layer-quanta" / quantum_id).resolve())
        record_space = Path(bound[int(quantum_id[-3:])]["output_space"]["root"])
        assert not record_space.is_relative_to(out)

    # The real run publishes exactly what the dry run printed.
    capsys.readouterr()
    assert _run(dispatch, gateway, records, receipt_path, out) == 0
    submitted = band._by_id(gateway)
    assert sorted(submitted) == sorted(rows)
    for quantum_id, row in rows.items():
        path = Path(row["handoff_template"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            row["handoff_template_sha256"]
        assert json.loads(path.read_text()) == row["handoff_template_document"]
        assert submitted[quantum_id]["argv"] == row["argv"]


def test_a_dry_run_derives_a_consumers_readset_without_publishing_it(
        tmp_path, monkeypatch, capsys):
    dispatch, bound, records, receipt_path, out = _layout(tmp_path, monkeypatch)
    gateway = dispatch.FakeGateway()
    assert _run(dispatch, gateway, records, receipt_path, out) == 0
    # Layer 3 executes and publishes its handoff: layer 2 is now publishable.
    _complete_producer(bound, gateway)
    gateway.submitted = []
    before = _files(tmp_path)
    capsys.readouterr()

    covered = []
    assert _run(dispatch, gateway, records, receipt_path, out, "--dry-run",
                coverage=lambda rows: covered.extend(rows) or []) == 0
    assert gateway.submitted == []
    assert _files(tmp_path) == before
    rows = _dry_rows(capsys)
    readset = rows["layer-002"]["band_serial_readset"]
    path = Path(readset["path"])
    assert path.parent == out / "layer-quanta" / dispatch.BAND_SERIAL_DIRECTORY
    assert not path.exists()
    assert rows["layer-002"]["manifest_sha256"] == readset["sha256"]
    # The source-coverage gate checks the derived bytes, which it is handed
    # because no file holds them.
    row = next(row for row in covered if row["record"]["quantum_id"] == "layer-002")
    assert row["manifest_path"] == str(path)
    manifest = load_manifest(row["manifest_path"], row["manifest_sha256"],
                             wire=row["manifest_wire"])
    assert [phase["name"] for phase in manifest["read_plan"]["phases"]][:2] == [
        "head", HANDOFF_LOAD_PHASE]

    # The real run publishes the readset the dry run printed, and submits
    # the argv it printed.
    capsys.readouterr()
    assert _run(dispatch, gateway, records, receipt_path, out) == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == readset["sha256"]
    assert band._by_id(gateway)["layer-002"]["argv"] == rows["layer-002"]["argv"]


def _complete_producer(bound, gateway):
    published = band._emit(bound[3])
    band._complete(bound[3], published)
    gateway.mark_terminal(band._by_id(gateway)["layer-003"]["action_key"])


def test_a_manifest_handed_as_bytes_is_checked_against_its_digest(tmp_path):
    import gzip

    wire = gzip.compress(json.dumps({"entries": []}).encode())
    digest = hashlib.sha256(wire).hexdigest()
    absent = str(tmp_path / "never-published.json.gz")
    assert load_manifest(absent, digest, wire=wire) == {"entries": []}
    with pytest.raises(ValueError, match="does not hash"):
        load_manifest(absent, "0" * 64, wire=wire)
    with pytest.raises(OSError):
        load_manifest(absent, digest)
