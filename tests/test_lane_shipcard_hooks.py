"""A lane's ship-record slots reach core through its plugin (#1553).

Decoupling step 6, part 3. ``shipcard`` and ``shipcard_cli`` import no lane
module:

- A lane that declares an evidence slot registers the replay ``verify`` runs
  for it (``shipcard_slot_verifiers``) and the commands that fill it
  (``shipcard_cli_commands``) on its plugin.
- ``shipcard.LANE_SLOT_VERIFIERS`` holds only the replays core itself owns,
  and ``shipcard.lane_slot_verifiers()`` is the merged view every check reads.
- The rate axis, the uniform control's block schema, the checkpoint's
  ``quant_method`` and the route-histogram obligation are lane-spec data.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from prismaquant import lane_spec, shipcard, shipcard_cli

ROOT = Path(__file__).resolve().parents[1]


def _with_fourth_lane_verifiers(monkeypatch, verifiers):
    """Add a fourth lane whose plugin registers ``verifiers``."""
    real = lane_spec.lane_hooks

    def hooks(name):
        found = real(name)
        if name == "shipcard_slot_verifiers":
            found = found + (("fourth", lambda: dict(verifiers)),)
        return found

    monkeypatch.setattr(lane_spec, "lane_hooks", hooks)


def test_importing_the_ship_record_imports_no_lane_module():
    code = textwrap.dedent("""
        import sys
        import prismaquant.shipcard
        import prismaquant.shipcard_cli
        loaded = sorted(m for m in sys.modules
                        if m == "tessera" or m.startswith(("tessera.", "prismaquant.tessera_")))
        assert not loaded, loaded
    """)
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_the_tessera_plugin_registers_its_slot_replays():
    from prismaquant import tessera_shipcard as ts

    registered = lane_spec.lane_plugin("tessera").shipcard_slot_verifiers()
    assert registered == {
        "route.census": ts.verify_route_census_record,
        "route.trace": ts.verify_route_trace_record,
    }
    merged = shipcard.lane_slot_verifiers()
    assert {"route.census", "route.trace", "route.sweep"} <= set(merged)
    # Core's own registry holds only the replays core owns.
    assert set(shipcard.LANE_SLOT_VERIFIERS) == {"route.sweep"}


def test_a_slot_with_two_owners_is_refused(monkeypatch):
    _with_fourth_lane_verifiers(
        monkeypatch, {"route.census": lambda *args, **kwargs: []})
    with pytest.raises(KeyError, match="route.census"):
        shipcard.lane_slot_verifiers()


def test_a_fourth_lanes_plugin_replay_owns_its_slot(monkeypatch):
    """A plugin's replay runs once, with the card, and decides the slot alone."""
    calls = []

    def verify_entropy(slot, record, *, card=None, model_dir=None):
        calls.append((slot, card is not None))
        if record.get("route_entropy") != {"match": True}:
            return [f"{slot}: route_entropy does not match"]
        return []

    _with_fourth_lane_verifiers(monkeypatch, {"route.entropy": verify_entropy})
    real = lane_spec.lane_spec_for_container("tessera")
    bogus = dataclasses.replace(real, gates=real.gates + (
        lane_spec.LaneGate(id="route.entropy", runner="true",
                           shipcard_slot="route.entropy"),))
    monkeypatch.setattr(lane_spec, "lane_spec_for_container", lambda _lane: bogus)

    assert "route.entropy" in shipcard.lane_gate_slots("tessera")
    sha = "abc"
    good = shipcard.make_record(
        slot="route.entropy", tool="true", passed=True, model_sha=sha,
        extra={"route_entropy": {"match": True}})
    card = {"model_sha": sha, "slots": {"route.entropy": good}}
    assert not [p for p in shipcard.verify(card, required=["route.entropy"])
                if "route.entropy" in p]
    assert calls == [("route.entropy", True)]

    bad = dict(good, route_entropy={"match": False})
    card["slots"]["route.entropy"] = bad
    problems = [p for p in shipcard.verify(card, required=["route.entropy"])
                if "route.entropy" in p]
    assert problems == ["route.entropy: route_entropy does not match"]


def test_the_lane_registers_its_fill_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        shipcard_cli.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    order = [text.index(name) for name in (
        "override-control", "fill-route-census", "fill-route-trace",
        "fill-route-sweep")]
    assert order == sorted(order), "the lane's commands keep their --help place"
    with pytest.raises(SystemExit) as exc:
        shipcard_cli.main(["fill-route-trace", "--help"])
    assert exc.value.code == 0


def test_a_world_without_the_lane_has_no_lane_fill_commands(monkeypatch):
    monkeypatch.setattr(lane_spec, "lane_hooks", lambda name: ())
    with pytest.raises(SystemExit) as exc:
        shipcard_cli.main(["fill-route-census", "card.json", "--census", "c.json"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        shipcard_cli.main(["fill-route-sweep", "--help"])
    assert exc.value.code == 0


def test_the_rate_axis_facts_are_lane_data():
    family = lane_spec.format_family_by_id("tessera")
    assert family.rate_axis
    assert family.uniform_control_schema == "tessera.uniform_control.v1"
    assert lane_spec.load_lane_spec("tessera").quant_method == "tessera"
    assert shipcard.uniform_control_schemas() == ("tessera.uniform_control.v1",)
    assert shipcard._is_rate_axis_artifact({"build": {"quant_method": "tessera"}})
    assert shipcard._is_rate_axis_artifact({"build": {"export_container": " Tessera "}})
    assert not shipcard._is_rate_axis_artifact(
        {"build": {"quant_method": "compressed-tensors"}})


def test_no_rate_axis_lane_means_no_rate_axis_artifact(monkeypatch):
    monkeypatch.setattr(lane_spec, "rate_axis_lanes", lambda: ())
    assert not shipcard._is_rate_axis_artifact({"build": {"quant_method": "tessera"}})
    assert shipcard.uniform_control_schemas() == ()


def test_a_rate_axis_family_must_name_its_control_schema():
    with pytest.raises(ValueError, match="uniform_control_schema"):
        lane_spec.LaneFormatFamily.from_dict(
            {"id": "x", "name_prefix": "X_", "rate_axis": True}, lane="x")


def test_the_route_histogram_obligation_is_the_lanes_declaration(monkeypatch):
    assert lane_spec.load_lane_spec("tessera").route_histogram_required
    assert not lane_spec.load_lane_spec("compressed_tensors").route_histogram_required
    assert shipcard._verify_route_histogram({"lane": "tessera", "build": {}})
    assert shipcard._verify_route_histogram(
        {"lane": "compressed-tensors", "build": {}}) == []

    real = lane_spec.load_lane_spec("tessera")
    silent = dataclasses.replace(real, route_histogram_required=False)
    monkeypatch.setattr(lane_spec, "lane_spec_for_container", lambda _lane: silent)
    assert shipcard._verify_route_histogram({"lane": "tessera", "build": {}}) == []
