"""Stage A chain split: a resumed chain run as sample-range quanta (PQ #738).

The claims:

* A run interrupted below a sealed checkpoint, prepped once, rolled by two
  quanta over disjoint sample ranges and joined, publishes the checkpoint
  and the band the uninterrupted single owner does, byte for byte, in the
  default chain regime, in a batched, fused one, and in R13's shape: several
  fused batches per read window. Two rounds chain through a joined checkpoint.
* A quantum names every entry, row and shared state by global batch; owns
  only its range (its status file, its borrowed plane); and stops at its
  ``through`` boundary, whose entries outlive it because its partial names
  them.
* A prep runs the round's resume record once and removes only the rolling
  entries of the ranges it launches, so a retry never touches a range that
  finished.
* A join refuses a plane with a missing sample and a partial whose bytes
  changed.

The fixture is ``test_stage_a_chain_resume``'s five-layer dense model at
stride 2 over five calibration rows, read in windows of two: checkpoints 5,
4 and 2. The interrupted run seals 5 and 4; the split round rolls 4 -> 2 as
the quanta ``0:2`` and ``2:5``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_space,
    checkpoint_directory,
    unpack_shared_states,
)
from prismaquant.joint_cost_stage_a import AdjointIdentityRefused

from test_stage_a_chain_resume import (  # noqa: F401  (autouse fixture)
    N_PROBES,
    _at,
    _band,
    _generation,
    _interrupted,
    _offline_tier_policy,
    _resume,
    _run,
)

N_BATCHES = 5
RANGES = [[0, 2], [2, 5]]
R13_REGIME = {"chain_batch_size": 2, "chain_probe_fusion": True}
# R13's shape: several fused batches in one read window (R13 reads 64 samples
# per window in fused batches of 4). Windows of four here, so the ranges are
# 0:4 and the last sample.
R13_SHAPE = {"window": 4, "chain_batch_size": 2, "chain_probe_fusion": True}
R13_SHAPE_RANGES = [[0, 4], [4, 5]]
REGIMES = pytest.mark.parametrize("regime, ranges", [
    ({}, RANGES), (R13_REGIME, RANGES), (R13_SHAPE, R13_SHAPE_RANGES)],
    ids=["default", "r13-batched-fused", "r13-shape-window-4"])


# Imported where used, so that on a tree without the split each test fails
# on its own first use of it instead of the whole module at collection.
def join_split_checkpoint(*args, **kwargs):
    from prismaquant.stage_a_chain_split import join_split_checkpoint as join
    return join(*args, **kwargs)


def partial_directory(*args):
    from prismaquant.stage_a_chain_split import partial_directory as directory
    return directory(*args)


def quantum_label(*args):
    from prismaquant.stage_a_chain_split import quantum_label as label
    return label(*args)


def _refused():
    from prismaquant.stage_a_chain_split import ChainSplitRefused
    return ChainSplitRefused


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint_files(root, boundary) -> dict:
    """Every byte a sealed checkpoint is: its own files and the entries it names."""
    directory = checkpoint_directory(adjoint_space(root), boundary)
    record = json.loads((directory / "checkpoint.json").read_text())
    files = {str(path.relative_to(root)): _sha(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    for row in record["activation_entries"]:
        files[str(Path(row["path"]).relative_to(root))] = _sha(row["path"])
    return files


def _members(root, boundary) -> dict:
    directory = checkpoint_directory(adjoint_space(root), boundary)
    payload = (directory / "entries" / "shared-states.pack").read_bytes()
    return {name: bytes(member) for name, member in unpack_shared_states(payload)}


def _windowed(root, regime) -> dict:
    """One fixture invocation's keywords under ``regime``.

    A ``window`` key reads the calibration in windows of that many rows: the
    fixture's own execution, with its boundary policy's window and resident
    cap scaled the way ``test_stage_a_chain_resume`` scales them for two.
    """
    regime = dict(regime)
    window = regime.pop("window", None)
    if window is None:
        return regime
    from test_joint_cost_quantum_runtime import _boundary_policy, _execution

    execution = _execution(root)
    policy = _boundary_policy(root / "boundaries", window=window)
    policy["max_resident_bytes"] = window * (1 + N_PROBES) * 256 + 256
    execution["boundary_storage"] = policy
    return {**regime, "execution": execution}


def _prep(root, monkeypatch, *, through=2, ranges=RANGES, resume_from=4, **kw):
    return _run(root, monkeypatch, chain_resume=_resume(root, resume_from=resume_from),
                chain_split={"role": "prep", "through": through, "ranges": ranges}, **kw)


def _quantum(root, monkeypatch, samples, *, through=2, resume_from=4, digest_layer=None,
             **kw):
    return _run(root, monkeypatch, chain_resume=_resume(root, resume_from=resume_from),
                chain_split={"role": "quantum", "through": through, "samples": samples,
                             "digest_layer": digest_layer}, **kw)


def _split_round(root, monkeypatch, *, regime, ranges=RANGES, writes=None):
    """Interrupt after checkpoint 4, prep, run both quanta, join at 2."""
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2), **_windowed(root, regime))
    prep = _prep(root, monkeypatch, ranges=ranges, **_windowed(root, regime))
    receipts = [_quantum(root, monkeypatch, samples, writes=writes,
                         **_windowed(root, regime))
                for samples in ranges]
    record = join_split_checkpoint(adjoint_space(root), 2, n_probes=N_PROBES,
                                   n_batches=N_BATCHES)
    return prep, receipts, record


# -- acceptance 2: a split round is the single owner's bytes ----------------------

@REGIMES
def test_two_quanta_joined_are_the_single_owners_checkpoint_and_band(
        tmp_path, monkeypatch, regime, ranges):
    root = tmp_path / "run"
    identities = []
    baseline = _run(root, monkeypatch, identities=identities, **_windowed(root, regime))
    assert [c["boundary"] for c in baseline["checkpoints"]] == [5, 4, 2]
    sources = {"bind_identity": identities[-1]}
    want_files = _checkpoint_files(root, 2)
    want_members = _members(root, 2)
    want_band = _band(root, 2, sources)
    want_record = json.loads((checkpoint_directory(adjoint_space(root), 2)
                              / "checkpoint.json").read_text())
    root.rename(tmp_path / "baseline")

    prep, receipts, record = _split_round(root, monkeypatch, regime=regime, ranges=ranges)

    assert prep["resume"]["split"]["ranges"] == ranges
    assert [receipt["split"]["samples"] for receipt in receipts] == ranges
    assert record == want_record
    got_members = _members(root, 2)
    assert sorted(got_members) == sorted(want_members)
    for name, member in want_members.items():
        assert got_members[name] == member, f"shared state {name} differs"
    assert _checkpoint_files(root, 2) == want_files
    assert _band(root, 2, sources) == want_band


def test_a_split_restores_and_joins_a_non_empty_shared_adjoint(tmp_path, monkeypatch):
    """The four-layer shared-K/V model at stride 1, split from checkpoint 2 to 1.

    Checkpoint 2 carries the shared adjoint layers 2 and 3 accumulated for
    layer 1, so each quantum must restore its own range's states. A pickled
    tensor's storage key is an address, so two single-owner runs already
    spell equal shared states differently (``_shared_contents``): the
    states are compared by content, and every cotangent entry by bytes.
    """
    import pickle

    from test_stage_a_chain_resume import _content

    root = tmp_path / "run"
    shared = {"model": "shared", "stride": 1}
    _run(root, monkeypatch, **shared)
    want_record = json.loads((checkpoint_directory(adjoint_space(root), 1)
                              / "checkpoint.json").read_text())
    want_entries = {row["name"]: _sha(row["path"]) for row in want_record["activation_entries"]}
    want_states = {name: _content(pickle.loads(member))
                   for name, member in _members(root, 1).items()}
    assert any(name.startswith("shared-adjoint") and "accumulators" in repr(content)
               for name, content in want_states.items())
    root.rename(tmp_path / "baseline")

    _interrupted(root, monkeypatch, interrupt=_at(1, 1, 0), **shared)
    _prep(root, monkeypatch, through=1, resume_from=2, **shared)
    for samples in RANGES:
        _quantum(root, monkeypatch, samples, through=1, resume_from=2, **shared)
    record = join_split_checkpoint(adjoint_space(root), 1, n_probes=N_PROBES,
                                   n_batches=N_BATCHES)
    assert record["activation_entries"] == want_record["activation_entries"]
    assert {row["name"]: _sha(row["path"]) for row in record["activation_entries"]} \
        == want_entries
    assert {name: _content(pickle.loads(member))
            for name, member in _members(root, 1).items()} == want_states


@REGIMES
def test_two_split_rounds_chain_through_a_joined_checkpoint(tmp_path, monkeypatch, regime,
                                                            ranges):
    """Rounds are bands: round 2's prep and quanta resume from round 1's join.

    At stride 1 the dense model seals 5, 4, 3, 2 and 1. Round 1 rolls 4 -> 3
    and joins 3; round 2 resumes from the joined 3, rolls to 2 and joins 2.
    The resume must take the joined checkpoint as sealed and hold its rows
    (they are the rolling entries of round 1's quanta), and the run's status
    must still say interrupted, so both joins are the single owner's bytes.
    """
    root = tmp_path / "run"

    def stride():
        return {"stride": 1, **_windowed(root, regime)}

    identities = []
    baseline = _run(root, monkeypatch, identities=identities, **stride())
    assert [c["boundary"] for c in baseline["checkpoints"]][:4] == [5, 4, 3, 2]
    sources = {"bind_identity": identities[-1]}
    want = {mark: (json.loads((checkpoint_directory(adjoint_space(root), mark)
                               / "checkpoint.json").read_text()),
                   _checkpoint_files(root, mark), _members(root, mark))
            for mark in (3, 2)}
    want_band = _band(root, 2, sources, stride=1)
    root.rename(tmp_path / "baseline")

    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2), **stride())
    space = adjoint_space(root)
    for index, (start, through) in enumerate([(4, 3), (3, 2)], start=1):
        prep = _prep(root, monkeypatch, through=through, resume_from=start, ranges=ranges,
                     **stride())
        assert prep["resume"]["index"] == index
        for samples in ranges:
            _quantum(root, monkeypatch, samples, through=through, resume_from=start,
                     **stride())
        join_split_checkpoint(space, through, n_probes=N_PROBES, n_batches=N_BATCHES)
    for mark, (record, files, members) in want.items():
        got = json.loads((checkpoint_directory(space, mark) / "checkpoint.json").read_text())
        assert got == record, f"checkpoint {mark}"
        assert _members(root, mark) == members, f"checkpoint {mark} shared states"
        assert _checkpoint_files(root, mark) == files, f"checkpoint {mark} files"
    assert _band(root, 2, sources, stride=1) == want_band


# -- decision 4: global indices end to end ---------------------------------------

def test_a_quantum_names_its_entries_rows_and_states_by_global_batch(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    writes = []
    _quantum(root, monkeypatch, RANGES[1], writes=writes)
    assert {batch for (_boundary, _probe, batch), _digest in writes} == {2, 3, 4}
    directory = partial_directory(adjoint_space(root), 2, 2, 5)
    record = json.loads((directory / "checkpoint.json").read_text())
    assert sorted(row["name"] for row in record["activation_entries"]) == sorted(
        f"cotangent-{probe}-{batch}-at-2" for probe in range(N_PROBES)
        for batch in (2, 3, 4))
    for row in record["activation_entries"]:
        coordinates = row["metadata"]["identity"]["coordinates"]
        assert row["name"] == (f"cotangent-{coordinates['probe']}-"
                               f"{coordinates['batch']}-at-2")
    pack = (directory / "entries" / "shared-states.pack").read_bytes()
    names = [name for name, _member in unpack_shared_states(pack)]
    assert sorted(names) == sorted(
        [f"shared-adjoint-{probe}-{batch}" for probe in range(N_PROBES)
         for batch in (2, 3, 4)] + [f"shared-pass-{batch}" for batch in (2, 3, 4)])


# -- decision 5: several owners of one generation --------------------------------

def test_a_quantums_clean_exit_leaves_the_generation_to_its_other_owners(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    generation = _generation(root)
    _prep(root, monkeypatch)
    before = json.loads(generation.read_text())
    assert before["status"] == "failed"
    first = _quantum(root, monkeypatch, RANGES[0])
    after = json.loads(generation.read_text())
    assert after == before, "a quantum never rewrites the run's generation status"
    label = quantum_label(4, 2, *RANGES[0])
    owner = json.loads((generation.parent / "owners" / f"{label}.json").read_text())
    assert owner["status"] == "complete" and owner["owner"]["label"] == label
    assert owner["owner"]["split"] == {"from": 4, "through": 2, "samples": RANGES[0]}
    assert first["retention"]["generation_manifest"].endswith(f"owners/{label}.json")
    # The second owner rebinds the same generation after the first finished.
    second = _quantum(root, monkeypatch, RANGES[1])
    assert second["status"] == "complete"


# -- decision 6: a quantum borrows its own range of the plane --------------------

def test_a_quantum_borrows_only_its_ranges_boundaries_and_cotangents(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    borrowed = []
    authorize = StreamedBoundaryArtifacts.authorize_resume_inputs

    def spy(self, boundaries, checkpoint, *, boundary):
        for reference in [*boundaries, *checkpoint]:
            identity = json.loads(reference.metadata_json)["identity"]
            borrowed.append((identity["kind"], identity["coordinates"]["batch"]))
        return authorize(self, boundaries, checkpoint, boundary=boundary)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "authorize_resume_inputs", spy)
    _quantum(root, monkeypatch, RANGES[1])
    assert {batch for _kind, batch in borrowed} == {2, 3, 4}
    assert sum(kind == "cotangent" for kind, _batch in borrowed) == N_PROBES * 3


# -- decision 7: a prep, once per round, scoped to its ranges --------------------

def test_the_prep_seals_the_rounds_resume_and_the_quanta_seal_none(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    space = adjoint_space(root)
    entries = _generation(root).parent / "entries"
    rolling = sorted(path.name for path in entries.glob("cotangent-*-at-3.pt"))
    assert rolling, "the interrupted attempt left rolling entries at 3"
    prep = _prep(root, monkeypatch)
    assert prep["resume"]["index"] == 1
    assert prep["resume"]["removed_rolling_entries"] == len(rolling)
    assert prep["resume"]["split"] == {
        "schema": "prismaquant.stage_a.chain_split.v1", "from": 4, "through": 2,
        "boundaries": [2], "ranges": RANGES}
    assert not list(entries.glob("cotangent-*-at-3.pt"))
    for samples in RANGES:
        _quantum(root, monkeypatch, samples)
    assert sorted(path.name for path in (space / "resumes").iterdir()) == [
        "resume-001.json"]


def test_a_retry_prep_leaves_the_finished_range_alone(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    space = adjoint_space(root)
    entries = _generation(root).parent / "entries"
    _prep(root, monkeypatch)
    _quantum(root, monkeypatch, RANGES[0])
    finished = {path.name: _sha(path) for path in entries.glob("cotangent-*-at-2.pt")}
    assert len(finished) == N_PROBES * 2
    # The second range dies after rolling part of layer 3.
    _interrupted(root, monkeypatch, interrupt=_at(3, 2, 3), chain_resume=_resume(
        root, resume_from=4), chain_split={"role": "quantum", "through": 2,
                                           "samples": RANGES[1], "digest_layer": None})
    dead = sorted(path.name for path in entries.glob("cotangent-*-at-3.pt"))
    assert dead and all(int(name.split("-")[2]) >= 2 for name in dead)
    retry = _prep(root, monkeypatch, ranges=[RANGES[1]])
    assert retry["resume"]["index"] == 2
    assert retry["resume"]["removed_rolling_entries"] == len(dead)
    label = quantum_label(4, 2, *RANGES[1])
    assert any(Path(path).name.startswith(f"{label}.resume-002") for path in
               retry["set_aside"])
    assert {path.name: _sha(path) for path in entries.glob("cotangent-*-at-2.pt")} \
        == finished
    assert (partial_directory(space, 2, *RANGES[0]) / "checkpoint.json").is_file()
    _quantum(root, monkeypatch, RANGES[1])
    join_split_checkpoint(space, 2, n_probes=N_PROBES, n_batches=N_BATCHES)


# -- decision 8: a quantum stops at its through boundary -------------------------

def test_a_quantum_rolls_nothing_below_through_and_keeps_its_last_plane(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    writes = []
    receipt = _quantum(root, monkeypatch, RANGES[0], writes=writes, digest_layer=3)
    assert {boundary for (boundary, _probe, _batch), _digest in writes} == {3, 2}
    assert [layer["layer"] for layer in receipt["telemetry"]["chain_layers"]] == [3, 2]
    entries = _generation(root).parent / "entries"
    assert sorted(path.name for path in entries.glob("cotangent-*-at-2.pt")) == sorted(
        f"cotangent-{probe}-{batch}-at-2.pt" for probe in range(N_PROBES)
        for batch in (0, 1))
    assert not list(entries.glob("cotangent-*-at-3.pt")), "at-3 entries retire"
    assert receipt["digests"]["layer"] == 3
    rolled = {f"{probe}-{batch}": digest
              for (boundary, probe, batch), digest in writes if boundary == 3}
    assert receipt["digests"]["payload_sha256"] == rolled


# -- decision 9: the join is whole and verified ----------------------------------

def test_the_join_refuses_a_missing_range(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    _quantum(root, monkeypatch, RANGES[0])
    with pytest.raises(_refused(), match="cover samples 0:2, not 0:5"):
        join_split_checkpoint(adjoint_space(root), 2, n_probes=N_PROBES,
                              n_batches=N_BATCHES)
    assert not checkpoint_directory(adjoint_space(root), 2).exists()


def test_the_join_refuses_a_partial_whose_pack_changed(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    for samples in RANGES:
        _quantum(root, monkeypatch, samples)
    pack = partial_directory(adjoint_space(root), 2, *RANGES[1]) / "entries" \
        / "shared-states.pack"
    data = bytearray(pack.read_bytes())
    data[0] ^= 0xFF
    pack.write_bytes(bytes(data))
    with pytest.raises(_refused(), match="is not the pack its manifest names"):
        join_split_checkpoint(adjoint_space(root), 2, n_probes=N_PROBES,
                              n_batches=N_BATCHES)


def test_a_quantum_refuses_a_range_that_splits_a_read_window(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    with pytest.raises(AdjointIdentityRefused, match="not whole read windows of 2"):
        _quantum(root, monkeypatch, [1, 3])


def test_a_split_names_a_stride_checkpoint_below_the_resume_point(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    with pytest.raises(AdjointIdentityRefused, match="not a stride checkpoint"):
        _prep(root, monkeypatch, through=3)


# -- the command lines -----------------------------------------------------------

def test_the_join_command_refuses_with_exit_2_and_joins_a_whole_plane(
        tmp_path, monkeypatch, capsys):
    from prismaquant import stage_a_chain_split

    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    _prep(root, monkeypatch)
    _quantum(root, monkeypatch, RANGES[0])
    argv = ["--output-root", str(root), "--boundary", "2"]
    assert stage_a_chain_split.main(argv) == 2
    assert "join refused" in capsys.readouterr().err
    assert not checkpoint_directory(adjoint_space(root), 2).exists()
    _quantum(root, monkeypatch, RANGES[1])
    receipt_path = tmp_path / "join.json"
    assert stage_a_chain_split.main([*argv, "--receipt", str(receipt_path)]) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "prismaquant.stage_a.chain_split_join.v1"
    record = json.loads((checkpoint_directory(adjoint_space(root), 2)
                         / "checkpoint.json").read_text())
    assert receipt["cotangent_sha256"] == record["cotangent_sha256"]


_STAGE_A_ARGV = ["--plan", "/p", "--plan-sha256", "b" * 64, "--prepared", "/q",
                 "--prepared-sha256", "c" * 64, "--output-root", "/o"]
_RESUME_ARGV = ["--resume-chain-state-sha256", "f" * 64, "--resume-from-checkpoint", "45"]


@pytest.mark.parametrize("argv, message", [
    (["--chain-split-quantum", "40:0:64"], "needs --resume-chain-state-sha256"),
    ([*_RESUME_ARGV, "--chain-split-prep", "40", "--chain-split-quantum", "40:0:64"],
     "not both"),
    ([*_RESUME_ARGV, "--chain-split-prep", "40"], "takes --chain-split-ranges"),
    ([*_RESUME_ARGV, "--chain-split-quantum", "40:0"], "THROUGH:START:STOP"),
    ([*_RESUME_ARGV, "--chain-split-quantum", "40:0:64", "--chain-split-ranges", "0:64"],
     "names its own samples"),
    ([*_RESUME_ARGV, "--chain-split-ranges", "0:64"], "belong to a split prep or quantum"),
    ([*_RESUME_ARGV, "--chain-split-prep", "40", "--chain-split-ranges", "0:64,64"],
     "a sample range is START:STOP"),
], ids=["no-resume", "both-roles", "prep-without-ranges", "short-quantum",
        "quantum-with-ranges", "ranges-alone", "malformed-range"])
def test_the_split_flags_refuse_a_malformed_row(capsys, argv, message):
    from prismaquant import joint_cost_stage_a as stage_a

    with pytest.raises(SystemExit) as exit_info:
        stage_a.main([*_STAGE_A_ARGV, *argv])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_the_split_flags_build_the_cores_split():
    import argparse
    from types import SimpleNamespace

    from prismaquant import joint_cost_stage_a as stage_a

    parser = argparse.ArgumentParser()
    args = SimpleNamespace(resume_chain_state_sha256="f" * 64, resume_from_checkpoint=45,
                           chain_split_prep=40, chain_split_ranges="0:64,64:128",
                           chain_split_quantum=None, chain_split_digest_layer=None)
    assert stage_a._chain_split_argument(args, parser) == {
        "role": "prep", "through": 40, "ranges": [[0, 64], [64, 128]]}
    args.chain_split_prep, args.chain_split_ranges = None, None
    args.chain_split_quantum, args.chain_split_digest_layer = "40:64:128", 43
    assert stage_a._chain_split_argument(args, parser) == {
        "role": "quantum", "through": 40, "samples": [64, 128], "digest_layer": 43}
    args.chain_split_quantum, args.chain_split_digest_layer = None, None
    assert stage_a._chain_split_argument(args, parser) is None
