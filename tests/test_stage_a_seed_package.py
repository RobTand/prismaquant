"""The seed data-manifest builder (PQ #1043, part of #997).

A seed's PrismaBuild action stages every byte it reads. The builder derives
its manifest from the source run's own: the head and the chain phases the
seed enters, plus the checkpoint files it borrows at the phases that read
them. The source run is the seed tests' fixture (stride 2, checkpoints 5, 4
and 2); its submitted manifest is synthesized with one entry per phase, and
its head also holds two of the head walk's reads, which a seed never opens
(PQ #1051).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.stage_a_chain_seed import load_seed_spec
from prismaquant.tessera_joint_aura import HEAD_WALK_INPUT_KEYS

from test_layer_major_boundary_capture import draw
from test_stage_a_chain_seed import N_PROBES, THREE, TWO, _manifest, _pin, _source
from tools.build_stagea_seed_package import (
    MANIFEST_NAME,
    PACKAGE_NAME,
    SPEC_NAME,
    SeedPackageRefused,
    build,
    peak_consecutive_bytes,
)

LAYERS = 5
PARENT = "d" * 64


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


def _plan(root):
    """The source run's plan: its inputs name the head walk's reads."""
    inputs = {key: {"path": f"{root}/walk/{key}.json", "sha256": "0" * 64}
              for key in HEAD_WALK_INPUT_KEYS}
    path = root / "plan.json"
    path.write_text(json.dumps({"inputs": inputs}, sort_keys=True))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


#: The head walk's reads in the source head: a bound input and a merged
#: checkpoint part.
WALK_READS = ("walk/census.json", "walk/merged_checkpoint.json.parts/units/0.pkl")


def _original(root, capsule, *, forward_recovery=True, plan_sha256=None):
    """The source run's submitted manifest: one entry per phase, sized by index,
    and the head walk's reads in the head."""
    names = ["head", *(f"forward-{layer:03d}" for layer in range(3, LAYERS)),
             *(f"chain-{layer:03d}" for layer in range(LAYERS - 1, -1, -1))]
    entries = [{"path": f"{root}/source/{name}", "offset": 0, "bytes": 10 + index,
                "sha256": None} for index, name in enumerate(names)]
    walk = [{"path": f"{root}/{name}", "offset": 0, "bytes": 1000, "sha256": None}
            for name in WALK_READS]
    phases, cumulative = [], 0
    for index, name in enumerate(names):
        indices = [index] + (list(range(len(names), len(names) + len(walk)))
                             if name == "head" else [])
        size = entries[index]["bytes"] + (sum(e["bytes"] for e in walk)
                                          if name == "head" else 0)
        cumulative += size
        phases.append({"name": name, "entry_indices": indices,
                       "bytes": size, "cumulative_bytes": cumulative})
    entries += walk
    annotations = {"parent_manifest_sha256": PARENT,
                   "plan_sha256": plan_sha256 or _plan(root)["sha256"],
                   "prepared_sha256": "c" * 64}
    if forward_recovery:
        annotations["forward_recovery"] = dict(capsule)
    return {"schema": "prismaquant.prismabuild.data_manifest.v2",
            "mount_prefix": "/mnt/shared", "produced_by": {"tool": "fixture"},
            "annotations": annotations, "entries": entries,
            "entry_count": len(entries),
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "read_plan": {"phases": phases, "read_bytes": cumulative}}


def _write(path, manifest):
    wire = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    path.write_bytes(wire)
    return hashlib.sha256(wire).hexdigest()


def _build(tmp_path, source, *, original=None, through=2, compare=2, output="package",
           **overrides):
    path = tmp_path / "source-manifest.json.gz"
    digest = _write(path, original or _original(tmp_path, source.capsule))
    kwargs = dict(
        original_manifest=path, original_manifest_sha256=digest, plan=_plan(tmp_path),
        checkpoint=_pin(_manifest(source.root, 4)), capsule=dict(source.capsule),
        through=through, implementation_from=TWO, implementation_to=THREE,
        compare=None if compare is None else _pin(_manifest(source.root, compare)),
        output=tmp_path / output)
    kwargs.update(overrides)
    return build(**kwargs)


def _read(package):
    raw = gzip.decompress(Path(package["data_manifest"]["path"]).read_bytes())
    manifest = json.loads(raw)
    return manifest, {phase["name"]: [manifest["entries"][i]["path"]
                                      for i in phase["entry_indices"]]
                      for phase in manifest["read_plan"]["phases"]}


def test_the_seed_stages_its_borrowed_checkpoint_where_it_reads_it(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    package = _build(tmp_path, source)
    manifest, phases = _read(package)
    root = tmp_path / "package"
    wire = (root / MANIFEST_NAME).read_bytes()
    assert package["data_manifest"]["sha256"] == hashlib.sha256(wire).hexdigest()
    assert json.loads((root / PACKAGE_NAME).read_text()) == package

    # The seed enters the head and the chain from 3 down to 2; nothing
    # forward, and none of the head walk's reads.
    assert list(phases) == ["head", "chain-003", "chain-002"]
    assert package["head_walk_reads_dropped"] == {"entries": 2, "bytes": 2000}
    assert not any("/walk/" in entry["path"] for entry in manifest["entries"])
    four = json.loads(_manifest(source.root, 4).read_text())
    two = json.loads(_manifest(source.root, 2).read_text())
    shared = [row["path"] for row in four["shared_state_entries"]]
    assert phases["head"] == [f"{tmp_path}/source/head", str(root / SPEC_NAME),
                              str(_manifest(source.root, 4)), *shared]
    assert phases["chain-003"] == [f"{tmp_path}/source/chain-003",
                                   *(row["path"] for row in four["activation_entries"])]
    assert phases["chain-002"] == [f"{tmp_path}/source/chain-002",
                                   str(_manifest(source.root, 2)),
                                   *(row["path"] for row in two["activation_entries"])]
    assert len(four["activation_entries"]) == N_PROBES * len(draw())

    # Every added entry carries its exact size and digest.
    for entry in manifest["entries"]:
        if entry["path"].startswith(str(source.root)):
            data = Path(entry["path"]).read_bytes()
            assert entry["bytes"] == len(data)
            assert entry["sha256"] == hashlib.sha256(data).hexdigest()

    # The totals are recomputed, and the package names the peak lead + next.
    cumulative = 0
    for phase in manifest["read_plan"]["phases"]:
        assert phase["bytes"] == sum(manifest["entries"][i]["bytes"]
                                     for i in phase["entry_indices"])
        cumulative += phase["bytes"]
        assert phase["cumulative_bytes"] == cumulative
    assert manifest["read_plan"]["read_bytes"] == cumulative == package["read_bytes"]
    assert manifest["entry_count"] == len(manifest["entries"])
    sizes = [phase["bytes"] for phase in manifest["read_plan"]["phases"]]
    assert package["peak_consecutive_phase_bytes"] == max(
        sizes[0] + sizes[1], sizes[1] + sizes[2])

    # The spec is the seed's own, and the manifest names it.
    spec = load_seed_spec(root / SPEC_NAME, package["seed_spec"]["sha256"])
    assert spec["through"] == 2 and spec["compare"] == _pin(_manifest(source.root, 2))
    assert spec["implementation_compatibility"] == {"from": TWO, "to": THREE}
    assert manifest["annotations"]["chain_seed"] == package["seed_spec"]
    assert manifest["annotations"]["parent_manifest_sha256"] == PARENT


def test_a_seed_without_a_compare_stages_no_reference(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    package = _build(tmp_path, source, through=3, compare=None)
    _, phases = _read(package)
    assert list(phases) == ["head", "chain-003"]
    assert all("boundary-002" not in path for paths in phases.values() for path in paths)


def test_the_seed_manifest_passes_the_prismabuild_validator(tmp_path, monkeypatch):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    source = _source(tmp_path, monkeypatch)
    package = _build(tmp_path, source)
    manifest, _ = _read(package)
    prefix = str(tmp_path)
    for entry in manifest["entries"]:
        entry["path"] = "/mnt/shared/fixture" + entry["path"][len(prefix):]
    ranges = tiers.manifest_phase_ranges(core.validate_data_manifest(manifest))
    assert [item["name"] for item in ranges] == ["head", "chain-003", "chain-002"]
    assert ranges[-1]["end_bytes"] == manifest["read_plan"]["read_bytes"]


def test_peak_consecutive_bytes():
    assert peak_consecutive_bytes([{"bytes": 5}]) == 5
    assert peak_consecutive_bytes([{"bytes": 1}, {"bytes": 9}, {"bytes": 3}]) == 12


REFUSALS = {
    "another capsule": (
        lambda tmp, s: {"capsule": {"path": s.capsule["path"], "sha256": "0" * 64}},
        "another forward-recovery capsule"),
    "a compare at another boundary": (
        lambda tmp, s: {"compare": 4, "through": 3},
        "compare checkpoint is boundary 4"),
    "through at the checkpoint": (
        lambda tmp, s: {"through": 4, "compare": None},
        "rolls to a boundary below it"),
    "a source run with no forward recovery": (
        lambda tmp, s: {"original": _original(tmp, s.capsule, forward_recovery=False)},
        "names no forward-recovery capsule"),
    "a plan other than the source manifest's": (
        lambda tmp, s: {"original": _original(tmp, s.capsule, plan_sha256="e" * 64)},
        "built for another plan"),
    "a plan that is not its digest": (
        lambda tmp, s: {"plan": {**_plan(tmp), "sha256": "e" * 64}},
        "does not have the pinned digest"),
    "a phase the source never staged": (
        lambda tmp, s: {"original": {**_original(tmp, s.capsule), "read_plan": {
            **_original(tmp, s.capsule)["read_plan"],
            "phases": [p for p in _original(tmp, s.capsule)["read_plan"]["phases"]
                       if p["name"] != "chain-003"]}}},
        r"no phase \['chain-003'\]"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_the_builder_refuses(tmp_path, monkeypatch, case):
    source = _source(tmp_path, monkeypatch)
    overrides, match = REFUSALS[case]
    with pytest.raises(SeedPackageRefused, match=match):
        _build(tmp_path, source, **overrides(tmp_path, source))
    assert not (tmp_path / "package").exists()


def test_the_builder_refuses_a_moved_source_manifest_and_an_existing_package(
        tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    with pytest.raises(SeedPackageRefused, match="source manifest does not have"):
        _build(tmp_path, source, original_manifest_sha256="0" * 64)
    _build(tmp_path, source)
    with pytest.raises(FileExistsError):
        _build(tmp_path, source)
