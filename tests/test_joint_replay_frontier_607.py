"""Check sealed replay frontiers and runtime replay refusals."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import test_glm_joint_data_manifest_at_submit as campaign_fixture  # noqa: E402


def _load_by_path(name: str, relative: str):
    """Load a torch-free module without importing ``prismaquant``."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = _load_by_path("joint_replay_frontier", "prismaquant/joint_replay_frontier.py")

_canonical_sha256 = campaign_fixture.canonical_sha256
_workspace_with_sealed_checkpoint = campaign_fixture.workspace_with_sealed_checkpoint
_plant_journal = campaign_fixture.plant_journal

U1, U2 = campaign_fixture.UNITS[0][0], campaign_fixture.UNITS[0][1]
V1, V2 = campaign_fixture.UNITS[1][0], campaign_fixture.UNITS[1][1]
PRODUCED_BY = campaign_fixture.PRODUCED_BY
FRESH_ARGV = ["python3", "-u", "-m", "prismaquant.tessera_joint_aura", "prepare"]
RESUME_ARGV = [*FRESH_ARGV, "--resume"]


@pytest.fixture()
def scratch(request, tmp_path):
    root = tmp_path / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_replay_scratch_stays_outside_the_source_checkout(scratch):
    assert not scratch.resolve().is_relative_to(ROOT)


# --- the sealed contract ---------------------------------------------------


def test_read_order_routes_original_frames_through_the_byte_owner(monkeypatch):
    owner = getattr(replay, "_length_framed_bytes_sha256", None)
    assert owner is not None
    record = json.loads((ROOT / "tests/fixtures/joint_replay_read_order_2541.json").read_bytes())
    items = replay.replay_read_items(record["cells"], record["cells"])
    prefix = bytes.fromhex(record["prefix_hex"])
    digest = Mock(wraps=hashlib.sha256(prefix))
    sha256 = Mock(return_value=digest)
    monkeypatch.setitem(owner.__globals__, "hashlib", SimpleNamespace(sha256=sha256))
    raw = replay._framed(items)
    sha256.assert_called_once_with(prefix)
    assert [call.args[0].hex() for call in digest.update.call_args_list] == [
        part for frame in zip(record["length_hex"], record["row_hex"], strict=True) for part in frame
    ]
    assert type(raw).__name__ == record["result_type"]
    assert raw == bytes.fromhex(record["digest_hex"])
    monkeypatch.undo()
    assert replay.read_order_sha256(items) == record["digest_hex"]
    refused = replay.replay_read_items([record["refused_name"]], {record["refused_name"]: []})
    with pytest.raises(UnicodeEncodeError) as exc:
        replay._framed(refused)
    assert [exc.value.encoding, exc.value.reason] == record["refusal"]


# --- the runtime-side refusals ---------------------------------------------


def _sealed(**overrides):
    cells = {U1: list(campaign_fixture.MEASURED), U2: list(campaign_fixture.MEASURED)}
    starts = {name: replay.replay_phase_name(0) for name in sorted(cells)}
    sealed = replay.seal_frontier(sorted(cells), cells,
                                 phase_start_units=starts,
                                 journal_identity_sha256="e" * 64)
    sealed.update(overrides)
    return sealed, cells


def test_a_replay_phase_table_is_a_contiguous_partition_of_the_roster():
    cells = {name: list(campaign_fixture.MEASURED) for name in (U1, V1)}
    good = {U1: replay.replay_phase_name(0), V1: replay.replay_phase_name(1)}
    replay.seal_frontier(sorted(cells), cells, phase_start_units=good,
                         journal_identity_sha256="e" * 64)
    for broken in ({U1: replay.replay_phase_name(1), V1: replay.replay_phase_name(0)},
                   {U1: replay.replay_phase_name(0), V1: replay.replay_phase_name(2)},
                   {U1: "layer-0-part-0", V1: replay.replay_phase_name(1)},
                   {U1: replay.replay_phase_name(0)}):
        with pytest.raises(ValueError):
            replay.seal_frontier(sorted(cells), cells, phase_start_units=broken,
                                 journal_identity_sha256="e" * 64)


def test_a_sealed_manifest_carries_its_part_contiguity_into_the_runtime():
    sealed, cells = _sealed()
    sealed[replay.PHASE_START_UNITS_KEY] = {U1: replay.replay_phase_name(0),
                                            U2: replay.replay_phase_name(2)}
    with pytest.raises(ValueError, match="skips a part"):
        replay.sealed_from_annotations(sealed)
    # A part cannot start after a later one has already begun: the runtime
    # announces a part when it reaches its first unit, so such a table would
    # release a prefix the action has not read.
    three = {name: list(campaign_fixture.MEASURED) for name in (U1, V1, U2)}
    sealed = replay.seal_frontier(
        sorted(three), three,
        phase_start_units={name: replay.replay_phase_name(0) for name in three},
        journal_identity_sha256="e" * 64)
    # In roster order (U1, U2, V1) the parts must never step backwards.
    sealed[replay.PHASE_START_UNITS_KEY] = {U1: replay.replay_phase_name(0),
                                            U2: replay.replay_phase_name(1),
                                            V1: replay.replay_phase_name(0)}
    with pytest.raises(ValueError, match="read order"):
        replay.sealed_from_annotations(sealed)


def test_the_sealed_frontier_round_trips_through_a_manifest():
    sealed, _ = _sealed()
    assert replay.sealed_from_annotations(sealed) == sealed


def test_a_partly_sealed_frontier_is_refused():
    sealed, _ = _sealed()
    for key in (replay.ROSTER_SHA256_KEY, replay.READ_ORDER_SHA256_KEY,
                replay.JOURNAL_IDENTITY_KEY, replay.PHASE_START_UNITS_KEY):
        partial = {k: v for k, v in sealed.items() if k != key}
        with pytest.raises(ValueError, match="sealed replay frontier"):
            replay.sealed_from_annotations(partial)


def test_a_roster_that_does_not_hash_to_its_digest_is_refused():
    sealed, _ = _sealed()
    sealed[replay.ROSTER_SHA256_KEY] = "0" * 64
    with pytest.raises(ValueError, match="does not hash"):
        replay.sealed_from_annotations(sealed)


def test_replay_frontier_refuses_a_repeated_unit_with_a_matching_digest():
    sealed, _ = _sealed()
    roster = sealed[replay.ROSTER_KEY]
    roster.append(roster[-1])
    sealed[replay.ROSTER_SHA256_KEY] = replay.roster_sha256(roster)
    before = deepcopy(sealed)
    with pytest.raises(ValueError) as exc:
        replay.sealed_from_annotations(sealed)
    assert str(exc.value) == "a sealed replay roster is a sorted, unique list of qnames"
    assert sealed == before


def test_a_replayed_unit_must_start_in_a_replay_phase():
    sealed, _ = _sealed()
    sealed[replay.PHASE_START_UNITS_KEY] = {name: "layer-0-part-0"
                                           for name in sealed[replay.ROSTER_KEY]}
    with pytest.raises(ValueError, match="replay phase"):
        replay.sealed_from_annotations(sealed)


def test_the_matching_roster_is_accepted_and_every_move_is_refused():
    sealed, cells = _sealed()
    replay.require_replay_matches(sealed, completed=sorted(cells),
                                  cells_by_unit=cells,
                                  journal_identity_sha256="e" * 64)

    with pytest.raises(RuntimeError, match="journal identity changed"):
        replay.require_replay_matches(sealed, completed=sorted(cells),
                                      cells_by_unit=cells,
                                      journal_identity_sha256="f" * 64)

    with pytest.raises(RuntimeError, match="no longer holds the sealed replay roster"):
        replay.require_replay_matches(sealed, completed=[U1], cells_by_unit=cells,
                                      journal_identity_sha256="e" * 64)

    with pytest.raises(RuntimeError, match="no longer holds the sealed replay roster"):
        replay.require_replay_matches(sealed, completed=[*sorted(cells), V1],
                                      cells_by_unit={**cells, V1: list(campaign_fixture.MEASURED)},
                                      journal_identity_sha256="e" * 64)


def test_a_reordered_read_set_is_refused():
    sealed, cells = _sealed()
    reordered = {name: list(reversed(cells[name])) for name in cells}
    with pytest.raises(RuntimeError, match="read order"):
        replay.require_replay_matches(sealed, completed=sorted(cells),
                                      cells_by_unit=reordered,
                                      journal_identity_sha256="e" * 64)


def test_the_replay_read_order_is_a_complete_unit_at_a_time():
    cells = {U1: list(campaign_fixture.MEASURED)}
    items = replay.replay_read_items(sorted(cells), cells)
    assert items == [
        (replay.CAPTURE, U1, None),
        (replay.WIRE, U1, campaign_fixture.MEASURED[0]),
        (replay.RENDER, U1, campaign_fixture.MEASURED[0]),
        (replay.WIRE, U1, campaign_fixture.MEASURED[1]),
        (replay.RENDER, U1, campaign_fixture.MEASURED[1]),
    ]
    assert replay.replay_phase_name(7) == "replay-0007"
