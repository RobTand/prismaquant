"""The joint loader reads its checkpoint through the stdlib, interned reader.

The reader's own cases are in ``tests/test_interned_json.py``. These ones hold
the *loader* to it: the merged checkpoint goes through
``interned_json.load_json_file``, the seal the loader recomputes still binds
the parsed identity (so a rewritten menu or scale refuses), and equal menus in
the loaded identity really are the same ``str`` objects in different lists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import tessera_joint_aura
from prismaquant.cost_stage_checkpoint import canonical_json_sha256, write_unit
from test_tessera_joint_aura import bind, fixture


def test_the_loader_parses_the_checkpoint_with_the_interned_reader(tmp_path, monkeypatch):
    config, names, _fmt, _payload, _states = fixture(tmp_path)
    seen = []
    real = tessera_joint_aura.load_json_file

    def spy(source, **kwargs):
        seen.append(Path(source))
        return real(source, **kwargs)

    monkeypatch.setattr(tessera_joint_aura, "load_json_file", spy)
    data = tessera_joint_aura.load_measured_anchor_input(config)
    assert [path.name for path in seen] == ["cost.anchors.json"]
    assert set(data.formats_by_qname) == set(names)


def _rewritten_checkpoint(tmp_path, config, edit):
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text())
    identity = manifest["identity"]
    name = sorted(identity["units"])[0]
    unit = identity["units"][name]
    if edit == "menu":
        unit["menu"] = [*unit["menu"], "TESSERA_E2M1_K2_R896"]
    elif edit == "scale":
        unit["input_global_scale"] = 0.5 if unit["input_global_scale"] is None else 1.0
    else:
        unit["hessian"] = {"algorithm": "fixture", "sha256": "9" * 64}
    path.write_text(json.dumps(manifest))
    # Rebind the file's own SHA, so the refusal under test is the SEAL and not
    # the artifact check that would otherwise fire first.
    config["merged_checkpoint"] = bind(path)
    return name, manifest, identity


@pytest.mark.parametrize("edit", ["menu", "scale", "hessian"])
def test_a_rewritten_identity_value_refuses_at_the_recomputed_seal(tmp_path, edit):
    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    stale = json.loads(Path(config["merged_checkpoint"]["path"]).read_text())["identity_sha256"]
    _name, manifest, identity = _rewritten_checkpoint(tmp_path, config, edit)
    assert manifest["identity_sha256"] == stale, "the recorded seal was not the one under test"
    assert canonical_json_sha256(identity, where="fixture") != stale, "the edit changed nothing"
    with pytest.raises(ValueError, match="checkpoint seal"):
        tessera_joint_aura.load_measured_anchor_input(config)


def test_equal_menus_in_the_loaded_identity_share_strings_and_not_lists(tmp_path):
    config, names, fmt, _payload, _states = fixture(tmp_path)
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text())
    menu = [fmt, "TESSERA_E4M3_K1_R896", "TESSERA_E4M3_K1_R2048"]
    for name in names:
        manifest["identity"]["units"][name]["menu"] = list(menu)
    seal = canonical_json_sha256(manifest["identity"], where="fixture")
    manifest["identity_sha256"] = seal
    path.write_text(json.dumps(manifest))
    config["merged_checkpoint"] = bind(path)
    # The per-unit envelopes are sealed under the identity, so a rewritten
    # checkpoint moves them too; this case is about the menus, not about that
    # refusal, which the cases above cover.
    parts = path.with_name(path.name + ".parts")
    for name in names:
        write_unit(parts, stage=tessera_joint_aura.STAGE, qname=name,
                   identity_sha256=seal, state=_states[name])
    data = tessera_joint_aura.load_measured_anchor_input(config)
    first, second = (data.manifest["identity"]["units"][name] for name in names)
    assert first["menu"] == second["menu"] == menu
    assert first["menu"] is not second["menu"]
    assert first["menu"][0] is second["menu"][0]
    first["menu"][0] = "mutated"
    assert second["menu"][0] == fmt


def test_a_menu_holding_a_number_is_never_substituted_across_units(tmp_path):
    # The loader path, not just the reader: two unit menus that are equal as
    # *values* but differ in member type must both survive intact and still
    # seal, which is what a value-keyed memo would break.
    config, names, fmt, _payload, _states = fixture(tmp_path)
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text())
    manifest["identity"]["units"][names[0]]["menu"] = [fmt, 1]
    manifest["identity"]["units"][names[1]]["menu"] = [fmt, True]
    seal = canonical_json_sha256(manifest["identity"], where="fixture")
    manifest["identity_sha256"] = seal
    path.write_text(json.dumps(manifest))
    config["merged_checkpoint"] = bind(path)
    parts = path.with_name(path.name + ".parts")
    for name in names:
        write_unit(parts, stage=tessera_joint_aura.STAGE, qname=name,
                   identity_sha256=seal, state=_states[name])
    data = tessera_joint_aura.load_measured_anchor_input(config)
    units = data.manifest["identity"]["units"]
    assert type(units[names[0]]["menu"][1]) is int
    assert type(units[names[1]]["menu"][1]) is bool
    assert units[names[0]]["menu"][1] == 1
    assert units[names[1]]["menu"][1] is True
