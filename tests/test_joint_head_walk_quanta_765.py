"""PB-publishable verification quanta for the joint head walk (#765, #688).

#765 distributes the head walk as PrismaBuild verification quanta: the roster
is data, not a container's job. The contract under test:

* **Producer** -- ``head_walk_quanta`` cuts the sorted roster into
  deterministic ``[lo, hi)`` descriptors. Data only: no threads, no
  placement, no dispatcher.
* **Admission check** -- ``check_quantum_for_roster`` binds one descriptor to
  the live census before a slice may bank; a foreign descriptor refuses.
* **Coverage proof** -- ``verify_quanta_coverage`` admits a set of
  descriptors only when they tile ``[0, N)`` exactly once under one roster
  digest. A gap, an overlap, or a mixed roster is a refusal, never a hole.
* **Collector** -- ``join_head_walk_journals`` custody-checks per-quantum
  journals and re-envelopes the verified unit states under the full-roster
  identity, so the joined journal is indistinguishable from a
  single-container walk's and the existing resume path consumes it unchanged.
* **Banking gate** -- a scoped walk banks only with its quantum descriptor;
  without one the refusal stands (the #688 half: scope without a joinable
  identity is still not the campaign's input).

The fixture is the campaign intake fixture of ``test_tessera_joint_aura``:
two units, one measured rung each, both renders already on disk.
"""
from __future__ import annotations

import json

import pytest

from prismaquant.cost_stage_checkpoint import unit_path
from prismaquant.joint_head_walk_quanta import (
    QUANTUM_SCHEMA,
    check_quantum_for_roster,
    head_walk_quanta,
    join_head_walk_journals,
    roster_digest,
    verify_quanta_coverage,
)
from tests.test_tessera_joint_aura import fixture


NAMES = [f"unit-{index:02d}" for index in range(7)]


# -- producer: the roster is data --------------------------------------------


def test_quanta_tile_the_sorted_roster_deterministically():
    first = head_walk_quanta(NAMES, max_units_per_quantum=3)
    second = head_walk_quanta(list(reversed(NAMES)), max_units_per_quantum=3)
    assert first == second, "descriptors depend on the roster, not its order"
    assert [(quantum["lo"], quantum["hi"]) for quantum in first] == [(0, 3), (3, 6), (6, 7)]
    assert [quantum["quantum_id"] for quantum in first] == [
        "quantum-0000", "quantum-0001", "quantum-0002"]
    assert [quantum["units"] for quantum in first] == [
        sorted(NAMES)[0:3], sorted(NAMES)[3:6], sorted(NAMES)[6:7]]
    assert all(quantum["schema"] == QUANTUM_SCHEMA and quantum["quanta"] == 3
               for quantum in first)
    assert len({quantum["roster_sha256"] for quantum in first}) == 1


def test_quanta_cover_a_single_quantum_roster():
    (only,) = head_walk_quanta(NAMES[:2], max_units_per_quantum=10)
    assert (only["lo"], only["hi"]) == (0, 2)
    assert only["units"] == sorted(NAMES[:2])
    assert only["roster_sha256"] == roster_digest(NAMES[:2])


def test_quanta_refuse_bad_inputs():
    with pytest.raises(ValueError):
        head_walk_quanta([], max_units_per_quantum=3)
    with pytest.raises(ValueError):
        head_walk_quanta([*NAMES, NAMES[0]], max_units_per_quantum=3)
    with pytest.raises(ValueError):
        head_walk_quanta(NAMES, max_units_per_quantum=0)
    with pytest.raises(ValueError):
        head_walk_quanta(["ok", ""], max_units_per_quantum=3)


# -- admission check: one descriptor against the live roster ------------------


def test_check_quantum_accepts_its_own_descriptor():
    (quantum,) = head_walk_quanta(NAMES[:2], max_units_per_quantum=10)
    assert check_quantum_for_roster(quantum, NAMES[:2]) == (0, 2)


def test_check_quantum_refuses_a_foreign_descriptor():
    (quantum,) = head_walk_quanta(NAMES[:2], max_units_per_quantum=10)
    with pytest.raises(ValueError, match="another roster"):
        check_quantum_for_roster(quantum, NAMES[:3])
    tampered = dict(quantum, roster_sha256="0" * 64)
    with pytest.raises(ValueError, match="another roster"):
        check_quantum_for_roster(tampered, NAMES[:2])
    moved = dict(quantum, lo=1, hi=2)
    with pytest.raises(ValueError, match="differ from the roster"):
        check_quantum_for_roster(moved, NAMES[:2])
    escaped = dict(quantum, lo=0, hi=99)
    with pytest.raises(ValueError, match="outside"):
        check_quantum_for_roster(escaped, NAMES[:2])
    with pytest.raises(ValueError, match="schema"):
        check_quantum_for_roster({"schema": "other"}, NAMES[:2])
    with pytest.raises(ValueError, match="mapping"):
        check_quantum_for_roster("quantum-0000", NAMES[:2])


# -- coverage proof: exactly once ---------------------------------------------


def test_coverage_accepts_a_complete_tiling_in_any_order():
    quanta = head_walk_quanta(NAMES, max_units_per_quantum=3)
    ordered = verify_quanta_coverage(list(reversed(quanta)), NAMES)
    assert [(quantum["lo"], quantum["hi"]) for quantum in ordered] == [(0, 3), (3, 6), (6, 7)]


def test_coverage_refuses_gap_overlap_and_mixed_rosters():
    quanta = head_walk_quanta(NAMES, max_units_per_quantum=3)
    with pytest.raises(ValueError, match="breaks at roster position 3"):
        verify_quanta_coverage([quanta[0], quanta[2]], NAMES)
    doubled = [quanta[0], quanta[0], quanta[1], quanta[2]]
    with pytest.raises(ValueError, match="quantum_id"):
        verify_quanta_coverage(doubled, NAMES)
    foreign = head_walk_quanta([*NAMES, "unit-99"], max_units_per_quantum=3)
    with pytest.raises(ValueError, match="another roster"):
        verify_quanta_coverage([quanta[0], foreign[0]], [*NAMES, "unit-99"])
    with pytest.raises(ValueError, match="non-empty"):
        verify_quanta_coverage([], NAMES)


# -- banking gate: scope banks only with its descriptor ------------------------


def _quantum_intake(bridge, config, names, quantum, journal):
    low, high = quantum["lo"], quantum["hi"]
    assert sorted(names)[low:high] == quantum["units"]
    return bridge.load_measured_anchor_input(
        config, verify_payloads=False, unit_scope=(low, high),
        head_checkpoint=journal, head_walk_quantum=quantum)


def test_scoped_read_without_a_descriptor_still_cannot_bank(tmp_path):
    """The #688 pin: scope without a joinable identity is not an input."""
    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    with pytest.raises(ValueError, match="a scoped read cannot bank a head-walk journal"):
        bridge.load_measured_anchor_input(config, verify_payloads=False,
                                          unit_scope=(0, 1),
                                          head_checkpoint=tmp_path / "scoped-walk")


def test_quantum_mismatch_refuses_before_anything_banks(tmp_path):
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    quanta = head_walk_quanta(names, max_units_per_quantum=1)
    # The descriptor names slice [0, 1) but the read asks for the whole roster.
    with pytest.raises(ValueError, match="does not name this scoped read"):
        bridge.load_measured_anchor_input(
            config, verify_payloads=False, head_checkpoint=tmp_path / "walk-a",
            head_walk_quantum=quanta[0])
    # A quantum that banks nothing is a refused quantum.
    with pytest.raises(ValueError, match="banks its slice journal or refuses"):
        bridge.load_measured_anchor_input(
            config, verify_payloads=False, unit_scope=(0, 1),
            head_walk_quantum=quanta[0])
    # A descriptor from another roster never reaches the walk.
    foreign = head_walk_quanta([*names, "model.layers.9.self_attn.q_proj"],
                               max_units_per_quantum=1)[0]
    with pytest.raises(ValueError, match="another roster"):
        bridge.load_measured_anchor_input(
            config, verify_payloads=False, unit_scope=(0, 1),
            head_checkpoint=tmp_path / "walk-b", head_walk_quantum=foreign)
    assert not (tmp_path / "walk-a").exists()
    assert not (tmp_path / "walk-b").exists()


# -- collector: shards join into the walk's own journal ------------------------


def test_quantum_slices_bank_join_and_resume_as_one_walk(tmp_path):
    """Two quantum shards join into the journal a fresh walk would have banked."""
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    quanta = head_walk_quanta(names, max_units_per_quantum=1)
    assert len(quanta) == 2
    first, second = sorted(names)
    assert quanta[0]["units"] == [first] and quanta[1]["units"] == [second]

    fresh = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                              head_checkpoint=tmp_path / "fresh-walk")
    shards = []
    for index, quantum in enumerate(quanta):
        journal = tmp_path / f"shard-{index}"
        data = _quantum_intake(bridge, config, names, quantum, journal)
        assert data.progress_committed == 1
        assert unit_path(journal, quantum["units"][0]).is_file()
        shards.append({"quantum": quantum, "journal_dir": str(journal)})

    joined = join_head_walk_journals(shards, dest=tmp_path / "joined-walk", names=names)
    assert joined["units"] == sorted(names) and joined["quanta"] == 2
    manifest = json.loads((tmp_path / "joined-walk" / "manifest.json").read_text())
    assert "quantum" not in manifest["identity"], "the joined journal is the walk's own"
    assert manifest["identity"]["roster_sha256"] == roster_digest(names)

    resumed = bridge.load_measured_anchor_input(config, verify_payloads=False,
                                                head_checkpoint=tmp_path / "joined-walk",
                                                head_resume=True)
    assert resumed.head_walk_resumed_units == 2
    assert (resumed.cells, resumed.formats_by_qname, resumed.progress_committed) == \
           (fresh.cells, fresh.formats_by_qname, fresh.progress_committed)


def test_join_refuses_gap_tampering_and_an_existing_dest(tmp_path):
    from prismaquant import tessera_joint_aura as bridge

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    quanta = head_walk_quanta(names, max_units_per_quantum=1)
    shards = []
    for index, quantum in enumerate(quanta):
        journal = tmp_path / f"shard-{index}"
        _quantum_intake(bridge, config, names, quantum, journal)
        shards.append({"quantum": quantum, "journal_dir": str(journal)})

    # A lost quantum is a refusal, not a silent hole.
    with pytest.raises(ValueError, match="cover"):
        join_head_walk_journals(shards[:1], dest=tmp_path / "gap-walk", names=names)
    # A receipt retargeted at another descriptor does not match its journal.
    retargeted = dict(shards[0]["quantum"], quantum_id="quantum-9999")
    with pytest.raises(ValueError, match="another descriptor"):
        join_head_walk_journals(
            [{"quantum": retargeted, "journal_dir": shards[0]["journal_dir"]}, shards[1]],
            dest=tmp_path / "retarget-walk", names=names)
    # The collector never merges into a directory it did not create.
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        join_head_walk_journals(shards, dest=tmp_path / "taken", names=names)
