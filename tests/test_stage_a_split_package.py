"""The data manifests of a Stage A chain split round (PQ #738).

Every row of a split round stages what it reads. The builder derives the
prep's and each quantum's manifest from the source run's own; these tests
run the rows the manifests are for and hold the manifests to what the rows
actually read:

* Every file that existed before a quantum ran and that it reads (a forward
  boundary row, a sealed cotangent, the resume checkpoint's shared-state
  pack) is declared in that quantum's manifest, and every row the builder
  adds is one the quantum reads. A quantum stages its own samples only.
* The prep stages the head and the round's small records and reads no entry.
* A later round stages the joined checkpoint, whose rows are the earlier
  round's rolling entries.

The fixture is ``test_stage_a_chain_split``'s: five layers, five
calibration rows in windows of two, interrupted below checkpoint 4. The
source manifest is synthesized with one entry per phase, and two head walk
reads the builder drops (PQ #1051).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import adjoint_space, checkpoint_directory
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.tessera_joint_aura import HEAD_WALK_INPUT_KEYS

from test_stage_a_chain_resume import (  # noqa: F401  (autouse fixture)
    N_PROBES,
    _at,
    _interrupted,
    _offline_tier_policy,
)
from test_stage_a_chain_split import N_BATCHES, RANGES, _prep, _quantum
from tools.build_stagea_split_package import (
    PACKAGE_NAME,
    PREP_MANIFEST_NAME,
    SplitPackageRefused,
    build,
    quantum_manifest_name,
)

LAYERS = 5
WALK_READS = ("walk/census.json", "walk/merged_checkpoint.json.parts/units/0.pkl")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plan(root):
    inputs = {key: {"path": f"{root}/walk/{key}.json", "sha256": "0" * 64}
              for key in HEAD_WALK_INPUT_KEYS}
    path = root / "plan.json"
    path.write_text(json.dumps({"inputs": inputs}, sort_keys=True))
    return {"path": str(path), "sha256": _sha(path)}


def _original(root, plan_sha256):
    """The source run's manifest: one source file per phase, the walk's reads in head."""
    names = ["head", *(f"forward-{layer:03d}" for layer in range(LAYERS)),
             *(f"chain-{layer:03d}" for layer in range(LAYERS - 1, -1, -1))]
    entries = [{"path": f"{root}/source/{name}.safetensors", "offset": 0,
                "bytes": 10 + index, "sha256": None} for index, name in enumerate(names)]
    walk = [{"path": f"{root}/{name}", "offset": 0, "bytes": 1000, "sha256": None}
            for name in WALK_READS]
    phases, cumulative = [], 0
    for index, name in enumerate(names):
        indices = [index] + ([len(names), len(names) + 1] if name == "head" else [])
        size = entries[index]["bytes"] + (2000 if name == "head" else 0)
        cumulative += size
        phases.append({"name": name, "entry_indices": indices, "bytes": size,
                       "cumulative_bytes": cumulative})
    entries += walk
    return {"schema": "prismaquant.prismabuild.data_manifest.v2",
            "mount_prefix": "/mnt/shared", "produced_by": {"tool": "fixture"},
            "annotations": {"plan_sha256": plan_sha256, "prepared_sha256": "c" * 64},
            "entries": entries, "entry_count": len(entries),
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "read_plan": {"phases": phases, "read_bytes": cumulative}}


def _lowest(root, boundary):
    return _sha(checkpoint_directory(adjoint_space(root), boundary) / "checkpoint.json")


def _build(tmp_path, root, *, through=2, ranges=RANGES, from_boundary=4, output="package",
           **overrides):
    plan = _plan(tmp_path)
    path = tmp_path / "source-manifest.json.gz"
    wire = gzip.compress(json.dumps(_original(tmp_path, plan["sha256"])).encode(), mtime=0)
    path.write_bytes(wire)
    kwargs = dict(original_manifest=path, original_manifest_sha256=_sha(path), plan=plan,
                  output_root=root, chain_state_sha256=_sha(chain_state_path(
                      adjoint_space(root))),
                  checkpoint_sha256=_lowest(root, from_boundary), through=through,
                  ranges=ranges, output=tmp_path / output)
    kwargs.update(overrides)
    return build(**kwargs)


def _phases(path):
    manifest = json.loads(gzip.decompress(Path(path).read_bytes()))
    return manifest, {phase["name"]: [manifest["entries"][i]["path"]
                                      for i in phase["entry_indices"]]
                      for phase in manifest["read_plan"]["phases"]}


class _Reads:
    """Every entry file and shared-state pack a row reads."""

    def __init__(self, monkeypatch):
        import prismaquant.joint_adjoint_checkpoints as checkpoints

        self.paths = []
        prefetch = StreamedBoundaryArtifacts.prefetch
        load = checkpoints.load_checkpoint_shared_states

        def spied_prefetch(storage, references):
            references = tuple(references)
            self.paths.extend(str(reference.path) for reference in references)
            return prefetch(storage, references)

        def spied_load(space, record, **kw):
            self.paths.extend(row["path"] for row in record["shared_state_entries"])
            return load(space, record, **kw)

        monkeypatch.setattr(StreamedBoundaryArtifacts, "prefetch", spied_prefetch)
        monkeypatch.setattr(checkpoints, "load_checkpoint_shared_states", spied_load)


def _files(root) -> set[str]:
    return {str(path) for path in Path(root).rglob("*") if path.is_file()}


def _run_rows_against(package_root, root, monkeypatch, *, ranges, **kw):
    """Run the round's prep and quanta; check each one's reads against its manifest."""
    _, prep = _phases(package_root / PREP_MANIFEST_NAME)
    reads = _Reads(monkeypatch)
    _prep(root, monkeypatch, ranges=ranges, **kw)
    assert reads.paths == [], "the prep reads no entry"
    for samples in ranges:
        manifest, phases = _phases(package_root / quantum_manifest_name(*samples))
        declared = {path for paths in phases.values() for path in paths}
        added = {entry["path"] for entry in manifest["entries"]
                 if entry["path"].startswith(str(root))}
        existing = _files(root)
        reads.paths.clear()
        _quantum(root, monkeypatch, samples, **kw)
        borrowed = {path for path in reads.paths if path in existing}
        assert borrowed <= declared, f"{samples} reads undeclared {sorted(borrowed - declared)}"
        small = {path for path in added if path.endswith(".json")}
        assert added - small == borrowed, f"{samples} declares rows it never reads"
    return prep


def test_each_row_of_a_round_stages_what_it_reads(tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    package = _build(tmp_path, root)
    package_root = tmp_path / "package"
    assert json.loads((package_root / PACKAGE_NAME).read_text()) == package
    for described in [package["prep"], *package["quanta"]]:
        path = Path(described["data_manifest"]["path"])
        assert described["data_manifest"]["sha256"] == _sha(path)

    space = adjoint_space(root)
    state = str(chain_state_path(space))
    records = [str(checkpoint_directory(space, mark) / "checkpoint.json") for mark in (5, 4)]
    manifest, prep = _phases(package_root / PREP_MANIFEST_NAME)
    assert prep == {"head": [f"{tmp_path}/source/head.safetensors", state, *records]}
    assert manifest["annotations"]["chain_split"]["role"] == "prep"
    assert package["head_walk_reads_dropped"] == {"entries": 2, "bytes": 2000}

    four = json.loads(Path(records[1]).read_text())
    for (start, stop), described in zip(RANGES, package["quanta"]):
        manifest, phases = _phases(package_root / quantum_manifest_name(start, stop))
        assert list(phases) == ["head", "chain-003", "chain-002"]
        assert phases["head"] == [f"{tmp_path}/source/head.safetensors", state, *records,
                                  four["shared_state_entries"][0]["path"]]
        plane = [row["path"] for row in four["activation_entries"]
                 if start <= int(row["name"].split("-")[2]) < stop]
        assert len(plane) == N_PROBES * (stop - start)
        rows = {layer: [f"boundary-{batch}-{layer}-at-{layer}" for batch in range(start, stop)]
                for layer in (3, 2)}
        assert phases["chain-003"][0] == f"{tmp_path}/source/chain-003.safetensors"
        assert phases["chain-003"][1:1 + len(plane)] == plane
        assert [Path(path).name.removesuffix(".pt") for path in
                phases["chain-003"][1 + len(plane):]] == rows[3]
        assert phases["chain-002"][0] == f"{tmp_path}/source/chain-002.safetensors"
        assert [Path(path).name.removesuffix(".pt")
                for path in phases["chain-002"][1:]] == rows[2]
        assert described["source_bytes"] == sum(
            entry["bytes"] for entry in manifest["entries"]
            if entry["path"].endswith(("chain-003.safetensors", "chain-002.safetensors")))
        assert manifest["annotations"]["chain_split"]["samples"] == [start, stop]
        # Every added entry carries its exact size and digest.
        for entry in manifest["entries"]:
            if entry["path"].startswith(str(root)):
                data = Path(entry["path"]).read_bytes()
                assert (entry["bytes"], entry["sha256"]) == (
                    len(data), hashlib.sha256(data).hexdigest())
        cumulative = 0
        for phase in manifest["read_plan"]["phases"]:
            cumulative += phase["bytes"]
            assert phase["cumulative_bytes"] == cumulative
        assert manifest["read_plan"]["read_bytes"] == cumulative == described["read_bytes"]
    # Each quantum reads the round's source once: the round reads it once per quantum.
    assert package["round_source_bytes"] == len(RANGES) * package["quanta"][0]["source_bytes"]

    _run_rows_against(package_root, root, monkeypatch, ranges=RANGES)


def test_a_later_round_stages_the_joined_checkpoint(tmp_path, monkeypatch):
    """At stride 1 round 1 rolls 4 -> 3 and joins 3; round 2 stages the joined 3."""
    from prismaquant.stage_a_chain_split import join_split_checkpoint

    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2), stride=1)
    first = _build(tmp_path, root, through=3, output="round-1")
    _run_rows_against(tmp_path / "round-1", root, monkeypatch, ranges=RANGES,
                      through=3, resume_from=4, stride=1)
    space = adjoint_space(root)
    join_split_checkpoint(space, 3, n_probes=N_PROBES, n_batches=N_BATCHES)

    second = _build(tmp_path, root, through=2, from_boundary=3, output="round-2")
    assert (first["from"], second["from"]) == (4, 3)
    joined = json.loads((checkpoint_directory(space, 3) / "checkpoint.json").read_text())
    _, phases = _phases(tmp_path / "round-2" / quantum_manifest_name(*RANGES[0]))
    assert phases["head"][-4:] == [
        *(str(checkpoint_directory(space, mark) / "checkpoint.json") for mark in (5, 4, 3)),
        joined["shared_state_entries"][0]["path"]]
    assert all(Path(path).parent.name == "entries" and path.endswith("-at-3.pt")
               for path in phases["chain-002"][1:1 + N_PROBES * 2])
    _run_rows_against(tmp_path / "round-2", root, monkeypatch, ranges=RANGES,
                      through=2, resume_from=3, stride=1)


def test_the_round_manifests_pass_the_prismabuild_validator(tmp_path, monkeypatch):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    package = _build(tmp_path, root)
    for described, names in [(package["prep"], ["head"]),
                             *((quantum, ["head", "chain-003", "chain-002"])
                               for quantum in package["quanta"])]:
        manifest, _ = _phases(described["data_manifest"]["path"])
        for entry in manifest["entries"]:
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len(str(tmp_path)):]
        ranges = tiers.manifest_phase_ranges(core.validate_data_manifest(manifest))
        assert [item["name"] for item in ranges] == names


REFUSALS = {
    "a checkpoint that is not the lowest sealed one": (
        lambda tmp, root: {"checkpoint_sha256": _lowest(root, 5)},
        "does not have the pinned digest"),
    "another chain state": (
        lambda tmp, root: {"chain_state_sha256": "0" * 64}, "does not have the pinned digest"),
    "a through that is not a stride checkpoint": (
        lambda tmp, root: {"through": 3}, "not a stride checkpoint"),
    "a range that splits a read window": (
        lambda tmp, root: {"ranges": [[0, 1], [1, 5]]}, "whole read windows"),
    "overlapping ranges": (
        lambda tmp, root: {"ranges": [[0, 4], [2, 5]]}, "overlap"),
    "a moved source manifest": (
        lambda tmp, root: {"original_manifest_sha256": "0" * 64},
        "source manifest does not have"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_builder_refuses(tmp_path, monkeypatch, case):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    overrides, match = REFUSALS[case]
    with pytest.raises(SplitPackageRefused, match=match):
        _build(tmp_path, root, **overrides(tmp_path, root))
    assert not (tmp_path / "package").exists()


def test_the_builder_refuses_a_row_that_is_not_on_disk_and_an_existing_package(
        tmp_path, monkeypatch):
    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    state = json.loads(chain_state_path(adjoint_space(root)).read_text())
    victim = Path(state["boundary_entries"]["2"][4]["path"])
    data = victim.read_bytes()
    victim.unlink()
    with pytest.raises(SplitPackageRefused, match="1 missing"):
        _build(tmp_path, root)
    assert not (tmp_path / "package").exists()
    victim.write_bytes(data)
    _build(tmp_path, root)
    with pytest.raises(FileExistsError):
        _build(tmp_path, root)
