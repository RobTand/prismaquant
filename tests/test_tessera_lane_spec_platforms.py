"""The lane spec's executed-contract claim, asked per platform.

``served_activation_quantization.executes`` states what the serving runtime
executes, and principle 14 says such a field is derived from the runtime's own
machine-readable table or refused.  It has been derived from ``formats[]``
since the field existed -- but ``formats[]`` is platform-blind.  It answers
"what does this family's route RUN?", which is a property of the family and is
true wherever the family is served, and it is therefore silent about the first
question a producer targeting an AMD device has to answer: is it served THERE?

Contract v23 publishes that separately, in
``lane_eligibility.platforms[*].executes``, and v23 is the first grammar that
CAN: before it a platforms entry was a bare key, so the only way to say
anything about a device was to have served on it -- and a cell is a receipt.
There was no way to publish "this family has no native route on this device",
and silence had to stand in for it.

So this module holds the same refusal the flat list already gets, asked per
platform: derive from the contract, compare for equality, refuse on drift.  It
also pins the coverage rule that keeps the derivation honest while the AMD
serving profiles are still landing -- every platform the CONTRACT declares is
covered, not only the ones some profile happens to target, so the gate is
already exercising ``gfx1151`` before anything targets it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from importlib.resources import as_file

from prismaquant import tessera_export_lane as tel
from prismaquant import tessera_render as tr
from prismaquant.lane_spec import load_lane_spec


def _packaged() -> dict:
    with as_file(tr.tessera_serving_contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _declared():
    spec = load_lane_spec("tessera")
    assert spec.served_activation_quantization is not None
    return spec.served_activation_quantization


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def test_executes_by_platform_is_derived_from_the_packaged_contract():
    """Rebuilt by a second implementation, not read back from the same code."""
    contract = _packaged()
    expected = {
        key: dict(entry["executes"])
        for key, entry in contract["lane_eligibility"]["platforms"].items()
    }
    assert dict(_declared().executes_by_platform) == expected
    # and through the gate, which is what a preflight actually calls
    assert tel.derive_executes_by_platform() == expected
    tel.require_platform_executes_derived_from_contract()


def test_every_declared_platform_is_covered_and_names_every_family():
    contract = _packaged()
    families = {str(row["family"]) for row in contract["formats"]}
    by_platform = _declared().executes_by_platform
    assert set(by_platform) == set(contract["lane_eligibility"]["platforms"])
    for platform, entry in by_platform.items():
        assert set(entry) == families, platform


def test_a_non_null_value_is_the_families_own_activation_contract():
    """The rule that stops a platform naming a contract the dispatch runs.

    Checked here as well as inside the parser, because this is the property
    the lane spec's own claim rests on: a platform entry may say a family is
    unbacked, and it may say it is served -- but it does not get to say it is
    served BY SOMETHING ELSE.
    """
    contract = _packaged()
    per_family = {
        str(row["family"]): row.get("activation_contract")
        for row in contract["formats"]
    }
    for platform, entry in _declared().executes_by_platform.items():
        for family, executed in entry.items():
            if executed is None:
                continue
            assert executed == per_family[family], (platform, family)


def test_the_family_level_derivation_is_unchanged():
    """v22-era behaviour for sm_121 is what it was; nothing widened."""
    contract = _packaged()
    derived = {
        str(row["name_pattern"]).replace("{k}", "*")
        for row in contract["formats"]
    }
    assert set(_declared().executes) == derived
    sm121 = _declared().executes_by_platform["sm_121"]
    assert not [f for f, c in sm121.items() if c is None], sm121


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("drift", [
    pytest.param({"gfx1151": {"TESSERA_E4M3_K1": "fp8_per_token_dynamic"}},
                 id="claims_an_unbacked_family"),
    pytest.param({"sm_121": {"TESSERA_BF16_K1": None}},
                 id="drops_a_backed_family"),
    pytest.param({"gfx9999": {}}, id="invents_a_platform"),
])
def test_the_gate_refuses_a_lane_spec_that_drifts(monkeypatch, drift):
    """Driven through the loader, not by editing the shipped JSON.

    A fixture edited to be wrong proves the file can be wrong; what has to be
    shown is that the GATE bites, so the drift is injected into the object the
    gate reads and the shipped spec is left alone.
    """
    import dataclasses

    from prismaquant import lane_spec as ls

    real = load_lane_spec("tessera")
    moved = {k: dict(v) for k, v in
             real.served_activation_quantization.executes_by_platform.items()}
    for platform, entry in drift.items():
        moved.setdefault(platform, {}).update(entry)
    declared = dataclasses.replace(
        real.served_activation_quantization, executes_by_platform=moved)
    monkeypatch.setattr(
        ls, "load_lane_spec",
        lambda lane_id: dataclasses.replace(
            real, served_activation_quantization=declared))
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_platform_executes_derived_from_contract()
    assert "PRINCIPLE 14" in str(excinfo.value)


def test_the_undrifted_spec_passes_through_the_same_call(monkeypatch):
    """The control: the injection seam itself does not cause the refusal."""
    import dataclasses

    from prismaquant import lane_spec as ls

    real = load_lane_spec("tessera")
    monkeypatch.setattr(ls, "load_lane_spec", lambda lane_id: real)
    tel.require_platform_executes_derived_from_contract()


# ---------------------------------------------------------------------------
# Profiles point at platforms the contract declares
# ---------------------------------------------------------------------------
def test_every_profile_target_platform_is_one_the_contract_declares():
    """A target the runtime never heard of prices against nothing.

    Asserted over the specs on disk rather than over a list here, so a profile
    added for a new device fails this until the pinned contract declares the
    device -- which is the order the two have to land in.
    """
    from prismaquant.serving_profiles import serving_profile_names

    declared = set(_packaged()["lane_eligibility"]["platforms"])
    root = Path(__file__).resolve().parents[1] / "prismaquant" / "serving_profile_specs"
    targets = {}
    for name in serving_profile_names():
        if not (root / f"{name}.json").exists():
            continue
        payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        target = payload.get("target_platform")
        if target:
            targets[name] = str(target)
    assert targets, "no serving profile declares a target_platform"
    tessera = {name: t for name, t in targets.items() if "tessera" in name}
    assert tessera, sorted(targets)
    for name, target in tessera.items():
        assert target in declared, (
            f"serving profile {name!r} targets {target!r}, which the pinned "
            f"Tessera contract does not declare (it publishes "
            f"{sorted(declared)})")
