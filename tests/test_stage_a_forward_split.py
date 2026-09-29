"""Stage A forward split: a fresh run's forward capture as partition-range quanta (PQ #738).

The claims:

* A prep, two quanta over disjoint window-aligned partition ranges and a
  join publish what the single owner writes after its tail, byte for byte:
  every forward boundary entry, the tail checkpoint (its manifest, its
  shared-state pack and the cotangent entries it names) and the chain state.
  This holds for the dense model, for a model whose layers share K/V state
  (a non-empty shared adjoint), and in R13's shape (read windows of four).
* The chain above the first stride checkpoint is separable: after the join
  the run is a chain resume at the tail checkpoint, and chain split rounds
  5 -> 4 -> 2 and a resume to 0 write the single owner's checkpoints,
  bands, entries and receipt.
* A quantum owns only its range: its entries and rows are named by global
  batch, its status is its own file, and the generation stays ``running``.
* The join refuses a range whose owner did not finish, a missing range, and
  an entry record whose file changed; the prep and the quanta refuse ranges
  that do not tile the plane or split a read window.

The fixture is ``test_stage_a_chain_resume``'s five-layer dense model at
stride 2 over five calibration rows, read in windows of two: checkpoints 5
(the tail), 4 and 2. The forward quanta are ``0:2`` and ``2:5``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.joint_adjoint_checkpoints import adjoint_space, checkpoint_directory
from prismaquant.joint_cost_stage_a import AdjointIdentityRefused
from prismaquant.stage_a_chain_resume import chain_state_path

from test_stage_a_chain_resume import (  # noqa: F401  (autouse fixture)
    N_PROBES,
    VARIABLE,
    _band,
    _generation,
    _offline_tier_policy,
    _resume,
    _run,
    _without,
)
from test_stage_a_chain_split import (
    R13_SHAPE,
    R13_SHAPE_RANGES,
    RANGES,
    _checkpoint_files,
    _members,
    _prep,
    _quantum,
    _windowed,
)

N_BATCHES = 5
SHAPES = pytest.mark.parametrize("model, regime, ranges, tail", [
    ("dense", {}, RANGES, 5),
    ("shared", {}, RANGES, 4),
    ("dense", R13_SHAPE, R13_SHAPE_RANGES, 5)],
    ids=["dense", "shared-kv", "r13-shape-window-4"])


# Imported where used, so that on a tree without the forward split each test
# fails on its own first use of it instead of the whole module at collection.
def _forward():
    from prismaquant import stage_a_forward_split
    return stage_a_forward_split


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entries(root) -> dict:
    """Every file in the run's generation entry directory, by name."""
    directory = _generation(root).parent / "entries"
    return {path.name: _sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def _forward_entries(root) -> dict:
    return {name: digest for name, digest in _entries(root).items()
            if name.startswith("boundary-")}


def _differences(got, want, where="") -> list:
    """The paths at which two JSON documents differ, first few only."""
    if isinstance(got, dict) and isinstance(want, dict):
        found = [f"{where}.{key}: missing on one side" for key in sorted(set(got) ^ set(want))]
        for key in sorted(set(got) & set(want)):
            found += _differences(got[key], want[key], f"{where}.{key}")
        return found[:8]
    if isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        found = []
        for index, (left, right) in enumerate(zip(got, want)):
            found += _differences(left, right, f"{where}[{index}]")
        return found[:8]
    return [] if got == want else [f"{where}: {str(got)[:160]} != {str(want)[:160]}"]


def _single_owner(root, monkeypatch, **kw) -> dict:
    """The single owner's full run: everything the forward split must reproduce."""
    identities = []
    receipt = _run(root, monkeypatch, identities=identities, **kw)
    space = adjoint_space(root)
    tail = receipt["checkpoints"][0]["boundary"]
    return {
        "receipt": receipt,
        "sources": {"bind_identity": identities[-1]},
        "chain_state": chain_state_path(space).read_bytes(),
        "forward_entries": _forward_entries(root),
        "entries": _entries(root),
        "checkpoints": {record["boundary"]: (
            json.loads((checkpoint_directory(space, record["boundary"])
                        / "checkpoint.json").read_text()),
            _checkpoint_files(root, record["boundary"]),
            _members(root, record["boundary"]))
            for record in receipt["checkpoints"]},
        "tail": tail,
    }


def _forward_prep(root, monkeypatch, ranges=RANGES, **kw):
    return _run(root, monkeypatch, forward_split={"role": "prep", "ranges": ranges}, **kw)


def _forward_quantum(root, monkeypatch, samples, **kw):
    return _run(root, monkeypatch, forward_split={"role": "quantum", "samples": samples},
                **kw)


def _forward_round(root, monkeypatch, *, ranges=RANGES, **kw):
    prep = _forward_prep(root, monkeypatch, ranges=ranges, **kw)
    receipts = [_forward_quantum(root, monkeypatch, samples, **kw) for samples in ranges]
    joined = _forward().join_forward_split(adjoint_space(root))
    return prep, receipts, joined


# -- acceptance 1: the joined forward capture is the single owner's bytes ----------

@SHAPES
def test_forward_quanta_joined_are_the_single_owners_capture(
        tmp_path, monkeypatch, model, regime, ranges, tail):
    """Byte for byte, except where Torch's pickle spells a shared state.

    A pickled tensor's storage key is its storage's address
    (``test_stage_a_chain_resume._shared_contents``), so two single-owner
    runs of the shared-K/V model already write different shared-state pack
    bytes for equal states. There the pack members are compared by
    content, and the digests that cover the pack bytes (the pack's own row,
    the record's ``cotangent_sha256`` and the chain state's copy of it) are
    set aside; every other byte, every cotangent and every forward entry is
    compared as written.
    """
    import pickle

    from test_stage_a_chain_resume import _content

    root = tmp_path / "run"
    kw = {"model": model, **_windowed(root, regime)}
    want = _single_owner(root, monkeypatch, **kw)
    assert want["tail"] == tail
    root.rename(tmp_path / "baseline")

    prep, receipts, joined = _forward_round(root, monkeypatch, ranges=ranges, **kw)

    space = adjoint_space(root)
    assert prep["ranges"] == ranges
    assert [receipt["forward_split"]["samples"] for receipt in receipts] == ranges
    assert joined["ranges"] == ranges
    got_state = json.loads(chain_state_path(space).read_bytes())
    want_state = json.loads(want["chain_state"])
    record, files, members = want["checkpoints"][tail]
    got = json.loads((checkpoint_directory(space, tail) / "checkpoint.json").read_text())
    got_files, got_members = _checkpoint_files(root, tail), _members(root, tail)
    pack = f"layer-quanta/adjoint/checkpoints/boundary-{tail:03d}/entries/shared-states.pack"
    manifest = f"layer-quanta/adjoint/checkpoints/boundary-{tail:03d}/checkpoint.json"
    pickled = model == "shared"
    if pickled:
        for document in (got_state, want_state):
            document["tail_checkpoint"].pop("cotangent_sha256")
            document.pop("chain_state_sha256")
        for document in (got, record):
            document.pop("cotangent_sha256")
            for row in document["shared_state_entries"]:
                row.pop("sha256"), row.pop("file_bytes")
        for document in (got_files, files):
            document.pop(pack), document.pop(manifest)
        got_members = {name: _content(pickle.loads(member))
                       for name, member in got_members.items()}
        members = {name: _content(pickle.loads(member)) for name, member in members.items()}
        assert any(name.startswith("shared-pass") for name in members)
    assert {
        "chain_state": _differences(got_state, want_state),
        "tail_record": _differences(got, record),
        "tail_members": sorted(name for name in set(got_members) | set(members)
                               if got_members.get(name) != members.get(name))[:8],
    } == {"chain_state": [], "tail_record": [], "tail_members": []}
    if not pickled:
        # The chain state is the single owner's, byte for byte.
        assert chain_state_path(space).read_bytes() == want["chain_state"]
        assert joined["chain_state"]["sha256"] == hashlib.sha256(
            want["chain_state"]).hexdigest()
    # Every forward boundary entry file is the single owner's.
    assert _forward_entries(root) == want["forward_entries"]
    # The tail checkpoint: every file it is and every cotangent entry it names.
    assert got_files == files
    # Nothing below the tail was rolled: the chain split rolls it.
    assert not any((checkpoint_directory(space, mark)).exists()
                   for mark in want["checkpoints"] if mark != tail)


# -- acceptance 2: the segment above the first stride checkpoint is separable -------

def test_the_chain_split_rolls_a_joined_forward_capture_to_the_single_owners_run(
        tmp_path, monkeypatch):
    """Forward round, chain rounds 5 -> 4 and 4 -> 2, a resume to 0: the single owner."""
    root = tmp_path / "run"
    want = _single_owner(root, monkeypatch)
    assert sorted(want["checkpoints"]) == [2, 4, 5]
    want_bands = {mark: _band(root, mark, want["sources"]) for mark in (5, 4, 2)}
    root.rename(tmp_path / "baseline")

    _forward_round(root, monkeypatch)
    space = adjoint_space(root)
    from prismaquant.stage_a_chain_split import join_split_checkpoint
    for start, through in [(5, 4), (4, 2)]:
        _prep(root, monkeypatch, through=through, resume_from=start)
        for samples in RANGES:
            _quantum(root, monkeypatch, samples, through=through, resume_from=start)
        join_split_checkpoint(space, through, n_probes=N_PROBES, n_batches=N_BATCHES)
    finished = _run(root, monkeypatch, chain_resume=_resume(root, resume_from=2))

    for mark, (record, files, members) in want["checkpoints"].items():
        got = json.loads((checkpoint_directory(space, mark) / "checkpoint.json").read_text())
        assert got == record, f"checkpoint {mark}"
        assert _members(root, mark) == members, f"checkpoint {mark} shared states"
        assert _checkpoint_files(root, mark) == files, f"checkpoint {mark} files"
        assert _band(root, mark, want["sources"]) == want_bands[mark], f"band {mark}"
    assert _entries(root) == want["entries"]
    assert chain_state_path(space).read_bytes() == want["chain_state"]
    ignored = (*VARIABLE, "telemetry")
    assert _without(finished, *ignored) == _without(want["receipt"], *ignored)


# -- a quantum owns its range ----------------------------------------------------------

def test_a_forward_quantum_names_its_entries_by_global_batch_and_leaves_the_generation(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    generation = _generation(root)
    assert json.loads(generation.read_text())["status"] == "running"
    prep_owner = generation.parent / "owners" / "forward-prep.json"
    assert json.loads(prep_owner.read_text())["status"] == "complete"
    receipt = _forward_quantum(root, monkeypatch, RANGES[1])
    assert json.loads(generation.read_text())["status"] == "running"
    status = json.loads((generation.parent / "owners"
                         / "forward-samples-000002-000005.json").read_text())
    assert status["status"] == "complete"
    assert status["owner"]["forward_split"] == {"samples": [2, 5]}
    assert {name for name in _entries(root) if name.startswith("boundary-")} == {
        f"boundary-{batch}-{boundary}-at-{boundary}.pt"
        for batch in (2, 3, 4) for boundary in range(5)}
    [partial] = receipt["partials"]
    record = json.loads((Path(partial["directory"]) / "checkpoint.json").read_text())
    assert record["boundary"] == 5
    assert sorted(row["name"] for row in record["activation_entries"]) == sorted(
        f"cotangent-{probe}-{batch}-at-5" for probe in range(N_PROBES)
        for batch in (2, 3, 4))
    # A quantum writes no chain state: only the join does.
    assert not chain_state_path(adjoint_space(root)).exists()


# -- refusals ----------------------------------------------------------------------

def _refused():
    return _forward().ForwardSplitRefused


def test_the_join_refuses_a_range_whose_quantum_did_not_run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    _forward_quantum(root, monkeypatch, RANGES[0])
    with pytest.raises(_refused(), match="no owner status"):
        _forward().join_forward_split(adjoint_space(root))
    assert not chain_state_path(adjoint_space(root)).exists()


def test_the_join_refuses_a_range_whose_quantum_failed(tmp_path, monkeypatch):
    from test_stage_a_chain_resume import _Interrupted

    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    _forward_quantum(root, monkeypatch, RANGES[0])
    with pytest.raises(_Interrupted):
        _forward_quantum(root, monkeypatch, RANGES[1], interrupt=lambda kw: (
            kw.get("probe_index"), kw["boundary_index"], kw["batch_index"]) == (1, 5, 3))
    with pytest.raises(_refused(), match="'failed'"):
        _forward().join_forward_split(adjoint_space(root))


def test_the_join_refuses_an_entry_whose_file_changed(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    for samples in RANGES:
        _forward_quantum(root, monkeypatch, samples)
    entry = _generation(root).parent / "entries" / "boundary-3-2-at-2.pt"
    entry.write_bytes(entry.read_bytes() + b"x")
    with pytest.raises(_refused(), match="not the size"):
        _forward().join_forward_split(adjoint_space(root))
    assert not chain_state_path(adjoint_space(root)).exists()


def test_the_join_runs_once(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _forward_round(root, monkeypatch)
    with pytest.raises(_refused(), match="joined once"):
        _forward().join_forward_split(adjoint_space(root))


@pytest.mark.parametrize("ranges, match", [
    ([[0, 2], [4, 5]], "do not tile"),
    ([[0, 2], [2, 3], [3, 5]], "whole read windows"),
])
def test_the_prep_refuses_ranges_that_do_not_tile_whole_windows(
        tmp_path, monkeypatch, ranges, match):
    root = tmp_path / "run"
    with pytest.raises(AdjointIdentityRefused, match=match):
        _forward_prep(root, monkeypatch, ranges=ranges)
    assert not _forward().prep_record_path(adjoint_space(root)).exists()


def test_a_quantum_refuses_a_range_the_prep_did_not_launch(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    with pytest.raises(AdjointIdentityRefused, match="not a range the prep launched"):
        _forward_quantum(root, monkeypatch, [0, 4])


def test_a_quantum_refuses_without_a_prep_and_a_prep_runs_once(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with pytest.raises(AdjointIdentityRefused, match="no forward split prep"):
        _forward_quantum(root, monkeypatch, RANGES[0])
    root = tmp_path / "run2"
    _forward_prep(root, monkeypatch)
    with pytest.raises(AdjointIdentityRefused, match="prepped once|exact boundary"):
        _forward_prep(root, monkeypatch, generation=1002)


def test_a_forward_split_is_never_a_resume_seed_or_chain_split(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with pytest.raises(AdjointIdentityRefused, match="not a chain resume"):
        _run(root, monkeypatch, chain_resume={"chain_state_sha256": "0" * 64,
                                              "declaration": None, "resume_from": None},
             forward_split={"role": "quantum", "samples": [0, 2]})


# -- the planner is one function -----------------------------------------------------

@pytest.mark.parametrize("n_batches, group_size, quanta", [
    (5, 2, 2), (5, 2, 3), (512, 64, 3), (512, 64, 8), (513, 64, 9), (7, 1, 7)])
def test_even_ranges_are_the_partition_planners_window_ranges(n_batches, group_size, quanta):
    from prismaquant.stage_a_chain_split import check_ranges, even_ranges, require_whole_plane

    ranges = even_ranges(n_batches, group_size, quanta)
    assert len(ranges) == quanta
    check_ranges(ranges, n_batches=n_batches, group_size=group_size, where="planned")
    require_whole_plane(ranges, n_batches=n_batches)
    windows = [-(-(stop - start) // group_size) for start, stop in ranges]
    assert max(windows) - min(windows) <= 1


# -- the command lines -----------------------------------------------------------------

@pytest.mark.parametrize("argv, message", [
    (["--forward-split-prep", "0:2,2:5", "--forward-split-quantum", "0:2"], "not both"),
    (["--forward-split-quantum", "0:2,2:5"], "one START:STOP"),
    (["--forward-split-quantum", "0:2", "--resume-chain-state-sha256", "0" * 64],
     "fresh run"),
])
def test_the_forward_split_flags_refuse_a_malformed_row(capsys, argv, message):
    from prismaquant.joint_cost_stage_a import main

    base = ["--plan", "p", "--plan-sha256", "0" * 64, "--prepared", "q",
            "--prepared-sha256", "0" * 64, "--output-root", "o"]
    with pytest.raises(SystemExit) as raised:
        main(base + argv)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_the_join_command_refuses_with_exit_2_and_joins_a_whole_plane(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    _forward_prep(root, monkeypatch)
    _forward_quantum(root, monkeypatch, RANGES[0])
    assert _forward().main(["--output-root", str(root)]) == 2
    assert "join refused" in capsys.readouterr().err
    _forward_quantum(root, monkeypatch, RANGES[1])
    receipt_path = tmp_path / "join.json"
    assert _forward().main(["--output-root", str(root), "--receipt", str(receipt_path)]) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["chain_state"]["sha256"] == _sha(chain_state_path(adjoint_space(root)))
    assert receipt["tail_checkpoint"]["cotangents"] == N_PROBES * N_BATCHES
