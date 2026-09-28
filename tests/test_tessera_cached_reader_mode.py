"""The producer's dev switch is the only opt-in to permissive cached intake."""
import json
from types import SimpleNamespace

import pytest

from prismaquant import tessera_export_lane as export
from prismaquant.tessera_reuse_authority import PRODUCER_AUTHORITY
from tessera import cached_unit


@pytest.mark.parametrize("value,expected", [(None, "permissive"), ("", "permissive"),
                                            ("1", "permissive"), ("0", "strict")])
def test_reader_mode_is_explicit_and_warnings_are_not_dropped(monkeypatch, capsys, value, expected):
    if value is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", value)
    stamp = {"unit": "model.layers.3.mlp.experts", "reason": "missing_encoder_source_proof"}
    warnings = [stamp] if expected == "permissive" else []
    bundle = SimpleNamespace(encoder_source_proof_mode=expected, warnings=warnings)
    arguments = ({"schema": "fixture"}, "directory", {"unit"}, {"source": "fixture"})

    def reader(*args, **kwargs):
        assert args == arguments
        assert kwargs == {"encoder_source_proof_mode": expected, "authority": PRODUCER_AUTHORITY}
        return bundle

    monkeypatch.setattr(cached_unit, "CachedUnitBundle", reader)
    assert export.read_cached_unit_bundle(*arguments) is bundle
    assert export.cached_unit_encoder_source_proof_mode() == expected
    output = capsys.readouterr()
    assert not output.out
    if warnings:
        assert json.dumps(stamp, sort_keys=True) in output.err
    else:
        assert not output.err


@pytest.mark.parametrize("value,expected", [("0", "strict"), ("1", "permissive")])
def test_composed_preflight_retains_reader_warnings(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", value)
    path = tmp_path / "composed.json"
    path.write_text(json.dumps({"schema": cached_unit.COMPOSED_CACHE_SCHEMA}))
    body, mtp = {"body": {"wire": "a"}}, {"mtp": {"wire": "b"}}
    source = {"checkpoint": "fixture"}
    warning = {"unit": "body", "reason": "missing_encoder_source_proof"}
    stamps = [warning] if expected == "permissive" else []

    def reader(*args, **kwargs):
        assert args[2:] == ({"body", "mtp"}, source)
        assert kwargs == {"encoder_source_proof_mode": expected, "authority": PRODUCER_AUTHORITY}
        return SimpleNamespace(units={**body, **mtp}, child_manifests=[],
                               encoder_source_proof_mode=expected, warnings=stamps)

    monkeypatch.setattr(cached_unit, "CachedUnitBundle", reader)
    accepted = export.require_composed_cached_units(path,
        scope={"by_unit": {**body, **mtp}, "expert_projection": {"source": source, "units": body},
               "mtp_expert_projection": {"source": source}},
        metadata={"mtp_selection": {"mtp_expert_wires": mtp}})
    assert accepted["encoder_source_proof_mode"] == expected
    assert accepted["warnings"] == stamps
