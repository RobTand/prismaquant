"""A file-oriented stage reuses the journal's existing identity envelope."""
import json

import pytest

from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, prepare_journal, write_unit,
)


def test_explicit_manifest_path_preserves_same_identity_resume(tmp_path):
    root = tmp_path / "cost.anchors.json.parts"
    manifest = tmp_path / "cost.anchors.json"
    identity = {"source": "one", "units": ["unit"]}
    journal, digest, state = prepare_journal(
        root, stage="tessera campaign", resume=True, identity=identity,
        qnames=["unit"], manifest_path=manifest,
    )
    assert journal == root and state == {}
    assert json.loads(manifest.read_text())["schema"] == MANIFEST_SCHEMA
    assert not (root / "manifest.json").exists()
    write_unit(root, stage="tessera campaign", qname="unit",
               identity_sha256=digest, state={"anchors": [1]})
    assert prepare_journal(
        root, stage="tessera campaign", resume=True, identity=identity,
        qnames=["unit"], manifest_path=manifest,
    )[2] == {"unit": {"anchors": [1]}}
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        prepare_journal(
            root, stage="tessera campaign", resume=True,
            identity={"source": "changed", "units": ["unit"]},
            qnames=["unit"], manifest_path=manifest,
        )


def test_explicit_manifest_path_refuses_legacy_without_overwrite(tmp_path):
    manifest = tmp_path / "cost.anchors.json"
    manifest.write_text(json.dumps({"schema": "old campaign", "anchors": []}))
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        prepare_journal(
            tmp_path / "parts", stage="tessera campaign", resume=True,
            identity={"source": "current"}, qnames=["unit"], manifest_path=manifest,
        )
    assert json.loads(manifest.read_text())["schema"] == "old campaign"


def _stored_producer_journal(tmp_path):
    identity = {"implementation_sha256": "old implementation",
                "encoder_source_sha256": "old encoder", "model": "same model"}
    root, digest, _ = prepare_journal(
        tmp_path / "journal", stage="fixture", resume=True,
        identity=identity, qnames=["unit", "next"])
    write_unit(root, stage="fixture", qname="unit", identity_sha256=digest,
               state={"measured": [1, 2, 3]})
    return root, digest, identity


def test_declared_producer_resume_reuses_original_shards_and_identity(tmp_path, monkeypatch, capsys):
    from prismaquant.cost_stage_checkpoint import unit_path
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    root, digest, identity = _stored_producer_journal(tmp_path)
    manifest = (root / "manifest.json").read_bytes()
    shard = unit_path(root, "unit").read_bytes()
    current = {**identity, "implementation_sha256": "new implementation",
               "encoder_source_sha256": "new encoder"}
    for _ in range(2):
        _, resumed_digest, completed = prepare_journal(
            root, stage="fixture", resume=True, identity=current,
            qnames=["unit", "next"],
            seal_fields={"implementation_sha256", "encoder_source_sha256"})
        assert resumed_digest == digest
        assert completed["unit"] == {"measured": [1, 2, 3]}
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and lines[0].startswith("[DEV-MODE]")
        for text in ("implementation_sha256", "encoder_source_sha256",
                     "old implementation", "new implementation", "old encoder", "new encoder"):
            assert text in lines[0]
        assert (root / "manifest.json").read_bytes() == manifest
        assert unit_path(root, "unit").read_bytes() == shard
    write_unit(root, stage="fixture", qname="next", identity_sha256=resumed_digest,
               state={"measured": [4]})
    assert prepare_journal(root, stage="fixture", resume=True, identity=current,
        qnames=["unit", "next"], seal_fields={"implementation_sha256", "encoder_source_sha256"})[2]["next"] == {"measured": [4]}


@pytest.mark.parametrize("change,field", [
    ({"model": "other model", "implementation_sha256": "new implementation"}, "implementation_sha256"),
    ({"new_field": 1}, "new_field"),
])
def test_declared_producer_mixed_or_new_comparability_refuses(tmp_path, monkeypatch, capsys, change, field):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    root, _digest, identity = _stored_producer_journal(tmp_path)
    manifest = (root / "manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match=f"checkpoint identity mismatch at {field}:.*refusing reuse or recompute"):
        prepare_journal(root, stage="fixture", resume=True,
            identity={**identity, **change},
            qnames=["unit", "next"], seal_fields={"implementation_sha256"})
    assert capsys.readouterr().out == ""
    assert (root / "manifest.json").read_bytes() == manifest


def test_default_producer_mismatch_keeps_original_refusal_and_bytes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    root, digest, identity = _stored_producer_journal(tmp_path)
    manifest = (root / "manifest.json").read_bytes()
    assert prepare_journal(root, stage="fixture", resume=True, identity=identity,
                           qnames=["unit", "next"])[1] == digest
    with pytest.raises(RuntimeError) as error:
        prepare_journal(root, stage="fixture", resume=True,
            identity={**identity, "implementation_sha256": "new implementation"},
            qnames=["unit", "next"])
    assert str(error.value) == (
        "fixture checkpoint identity mismatch at implementation_sha256: "
        "stored='old implementation' current='new implementation'; refusing reuse or recompute")
    assert capsys.readouterr().out == ""
    assert (root / "manifest.json").read_bytes() == manifest


def test_certified_declared_producer_mismatch_still_refuses(tmp_path, monkeypatch, capsys):
    root, _digest, identity = _stored_producer_journal(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch at implementation_sha256"):
        prepare_journal(root, stage="fixture", resume=True,
            identity={**identity, "implementation_sha256": "new implementation"},
            qnames=["unit", "next"], seal_fields={"implementation_sha256"})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("damage", ["manifest_digest", "unit_bytes"])
def test_declared_producer_resume_keeps_byte_integrity_refusals(tmp_path, monkeypatch, damage):
    from prismaquant.cost_stage_checkpoint import unit_path
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    root, _digest, identity = _stored_producer_journal(tmp_path)
    if damage == "manifest_digest":
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["identity_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
    else:
        unit_path(root, "unit").write_bytes(b"corrupt unit")
    with pytest.raises(RuntimeError, match="mismatch|corrupt"):
        prepare_journal(root, stage="fixture", resume=True,
            identity={**identity, "implementation_sha256": "new implementation"},
            qnames=["unit", "next"], seal_fields={"implementation_sha256"})
