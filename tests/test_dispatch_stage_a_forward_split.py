"""A fresh run's forward split as PrismaBuild rows (PQ #738).

The package builder derives the forward split's manifests from the fresh
run's own submitted manifest; the dispatcher seals a prep row, one quantum
row per calibration partition range and one CPU join row, and submits each
only once the rows it follows have finished. These tests hold:

* The prep stages the head only; each quantum stages the head and every
  forward layer, and reads no entry another row wrote.
* The manifests pass PrismaBuild's own validator.
* Every Stage A row carries the gb10 tag alone, priority -10 and its own
  manifest and template; the join is a CPU row.
* The quanta wait for the prep's record and ending; the join waits for every
  quantum's ending and receipt. The rows the fake ``pbrun`` "runs" are the
  fixture's own core calls, so the gates read real records.
* A forward walk's read-ahead depth looks up the walk, not down.

The fixture is ``test_stage_a_chain_resume``'s five-layer dense model over
five calibration rows in windows of two; the forward quanta are ``0:2`` and
``2:5``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import adjoint_space
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.tessera_joint_aura import HEAD_WALK_INPUT_KEYS

from stage_a_spool_spec import stage_a_plan, with_spool
from test_stage_a_chain_resume import _offline_tier_policy, _run  # noqa: F401
from test_stage_a_chain_split import RANGES

import dispatch_stage_a_split as split_dispatch
from dispatch_stage_a_split import (
    SplitDispatchRefused,
    load_round,
    readahead_depth,
    seal_forward_round,
    submit,
)

TIER = "prismabuild-stage:fixture"
BUDGET = 1 << 30
LAYERS = 5
N_BATCHES = 5
FORWARD = [f"forward-{layer:03d}" for layer in range(LAYERS)]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True))
    return Path(path)


def _fresh_manifest(tmp_path, plan_sha256) -> Path:
    """The fresh run's submitted manifest: its head, its forward walk, its chain."""
    names = ["head", *FORWARD, *(f"chain-{layer:03d}" for layer in range(LAYERS - 1, -1, -1))]
    entries = [{"path": f"{tmp_path}/source/{name}.safetensors", "offset": 0,
                "bytes": 1000 * (index + 1), "sha256": None}
               for index, name in enumerate(names)]
    phases, total = [], 0
    for index, name in enumerate(names):
        total += entries[index]["bytes"]
        phases.append({"name": name, "entry_indices": [index],
                       "bytes": entries[index]["bytes"], "cumulative_bytes": total})
    path = tmp_path / "source-manifest.json.gz"
    path.write_bytes(gzip.compress(json.dumps({
        "schema": "prismaquant.prismabuild.data_manifest.v2", "mount_prefix": "/mnt/shared",
        "produced_by": {"tool": "fixture"}, "entries": entries,
        "entry_count": len(entries), "total_bytes": total,
        "annotations": {"parent_manifest_sha256": "d" * 64, "plan_sha256": plan_sha256,
                        "prepared_sha256": "b" * 64},
        "read_plan": {"phases": phases, "read_bytes": total}}).encode(), mtime=0))
    return path


def _package(tmp_path, root, **overrides):
    from tools.build_stagea_split_package import build_forward

    plan = _json(tmp_path / "plan.json", {
        **stage_a_plan(tmp_path, output_root=str(root)),
        "inputs": {key: {"path": f"{tmp_path}/walk/{key}.json", "sha256": "0" * 64}
                   for key in HEAD_WALK_INPUT_KEYS}})
    source = _fresh_manifest(tmp_path, _sha(plan))
    kwargs = dict(original_manifest=source, original_manifest_sha256=_sha(source),
                  plan={"path": str(plan), "sha256": _sha(plan)}, ranges=RANGES,
                  n_batches=N_BATCHES, group_size=2, output=tmp_path / "package")
    kwargs.update(overrides)
    return plan, build_forward(**kwargs)


def _phases(path):
    manifest = json.loads(gzip.decompress(Path(path).read_bytes()))
    return manifest, [phase["name"] for phase in manifest["read_plan"]["phases"]]


# -- the package ------------------------------------------------------------------

def test_the_prep_stages_the_head_and_each_quantum_the_forward_walk(tmp_path):
    _plan, package = _package(tmp_path, tmp_path / "run")
    assert package["ranges"] == RANGES
    _manifest, names = _phases(package["prep"]["data_manifest"]["path"])
    assert names == ["head"]
    for described, samples in zip(package["quanta"], RANGES):
        manifest, names = _phases(described["data_manifest"]["path"])
        assert names == ["head", *FORWARD]
        assert manifest["annotations"]["forward_split"] == {
            "role": "quantum", "ranges": RANGES, "n_batches": N_BATCHES, "group_size": 2}
        assert described["samples"] == samples
        assert described["source_bytes"] == sum(1000 * (index + 2)
                                                for index in range(LAYERS))
    # Every quantum reads every layer's weights: the round reads them once each.
    assert package["round_source_bytes"] == len(RANGES) * package["quanta"][0][
        "source_bytes"]


@pytest.mark.usefixtures("pinned_pb_source")
def test_the_forward_manifests_pass_the_prismabuild_validator(tmp_path):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    _plan, package = _package(tmp_path, tmp_path / "run")
    for described, names in [(package["prep"], ["head"]),
                             *((quantum, ["head", *FORWARD]) for quantum in package["quanta"])]:
        manifest, _ = _phases(described["data_manifest"]["path"])
        for entry in manifest["entries"]:
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len(str(tmp_path)):]
        ranges = tiers.manifest_phase_ranges(core.validate_data_manifest(manifest))
        assert [item["name"] for item in ranges] == names


@pytest.mark.parametrize("ranges, match", [
    ([[0, 2], [4, 5]], "do not tile"), ([[0, 3], [3, 5]], "whole read windows")])
def test_the_builder_refuses_ranges_that_do_not_tile_whole_windows(tmp_path, ranges, match):
    from tools.build_stagea_split_package import SplitPackageRefused

    with pytest.raises(SplitPackageRefused, match=match):
        _package(tmp_path, tmp_path / "run", ranges=ranges)
    assert not (tmp_path / "package").exists()


def test_a_forward_quantum_reads_no_entry_another_row_wrote(tmp_path, monkeypatch):
    """The manifests declare no entry because a forward quantum borrows none."""
    root = tmp_path / "run"
    read = []
    prefetch = StreamedBoundaryArtifacts.prefetch

    def spied(storage, references):
        references = tuple(references)
        read.extend(str(reference.path) for reference in references)
        return prefetch(storage, references)

    monkeypatch.setattr(StreamedBoundaryArtifacts, "prefetch", spied)
    _run(root, monkeypatch, forward_split={"role": "prep", "ranges": RANGES})
    assert read == []
    for samples in RANGES:
        existing = {str(path) for path in root.rglob("*") if path.is_file()}
        read.clear()
        _run(root, monkeypatch, forward_split={"role": "quantum", "samples": samples})
        assert read, "the quantum read its own boundaries back"
        assert not set(read) & existing, f"{samples} read another row's entries"


# -- the round -----------------------------------------------------------------------

@pytest.fixture
def sealed(tmp_path, monkeypatch):
    import dispatch_joint_quanta

    root = tmp_path / "run"
    plan, _package_record = _package(tmp_path, root)
    campaign = {"plan_path": str(plan), "plan_sha256": _sha(plan),
                "prepared_path": str(tmp_path / "prepared.json"),
                "prepared_sha256": "b" * 64, "read_manifest_sha256": "d" * 64}
    spec = _json(tmp_path / "spec.json", with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    prefetch = _json(tmp_path / "prefetch.json",
                     {"source_prefetch": {"prefetch_lookahead": 2}})
    space = adjoint_space(root)
    base = _json(tmp_path / "template.json", {
        "schema": "prismaquant.prismabuild.produced_output_template.v1", "version": 1,
        "template_id": "fixture-base", "output_prefix": str(space),
        "slots": {"boundary_entries": {"class": "payload"}},
        "durable_maxima": {"payload_max_bytes": BUDGET, "checkpoint_max_bytes": BUDGET,
                           "temp_max_bytes": BUDGET},
        "working_demands": {TIER: {"minimum_gib": 2, "window_gib": 48}},
        "permitted_tiers": [TIER]})
    monkeypatch.setattr(dispatch_joint_quanta, "_pbrun_seals_produced_output", lambda *a: True)
    monkeypatch.setattr(split_dispatch, "checkout_commit", lambda checkout: "c" * 40)
    monkeypatch.setattr(split_dispatch, "implementation_sha256", lambda checkout: "1" * 64)
    document = seal_forward_round(
        round_dir=tmp_path / "round", checkout=tmp_path / "checkout",
        forward_package=tmp_path / "package" / "forward-package.json", campaign=campaign,
        spec=spec, prefetch_override=prefetch, base_template=base, tier=TIER,
        template_prefix="fixture-forward", artifact_budget_bytes=BUDGET,
        chain_regime={"chain_batch_size": 1, "chain_probe_fusion": "off"},
        seconds_per_layer=100.0, n_probes=4, python="/venv/bin/python")
    return {"root": root, "round": tmp_path / "round", "document": document, "space": space}


def _payload(row):
    return row["argv"][row["argv"].index("prismaquant.joint_adjoint_capture") + 1:]


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _envelope(row):
    return row["argv"][:row["argv"].index("--")]


def test_a_sealed_forward_round_names_every_row(sealed):
    document = sealed["document"]
    rows = {row["name"]: row for row in document["rows"]}
    labels = ["forward-samples-000000-000002", "forward-samples-000002-000005"]
    assert list(rows) == ["forward-prep", *labels, "forward-join"]
    assert document["labels"] == labels and document["ranges"] == RANGES
    assert load_round(sealed["round"]) == document
    prep = _payload(rows["forward-prep"])
    assert _flag(prep, "--forward-split-prep") == "0:2,2:5"
    assert "--resume-chain-state-sha256" not in prep and "--forward-recovery" not in prep
    for label, (start, stop) in zip(labels, RANGES):
        payload = _payload(rows[label])
        assert _flag(payload, "--forward-split-quantum") == f"{start}:{stop}"
        assert _flag(payload, "--chain-batch-size") == "1"
        assert "--forward-split-prep" not in payload
    templates = set()
    for name in ["forward-prep", *labels]:
        envelope = _envelope(rows[name])
        assert [envelope[i + 1] for i, word in enumerate(envelope) if word == "--tag"] == [
            "gb10"]
        assert _flag(envelope, "--priority") == "-10"
        assert _flag(envelope, "--data-manifest") == rows[name]["data_manifest"]["path"]
        assert "--host-class" not in envelope and "--measurement" not in envelope
        templates.add(json.loads(Path(_flag(
            envelope, "--produced-output-template")).read_text())["template_id"])
        # A forward walk reads up: two layers ahead of the one it reads.
        depth = rows[name]["reader"]["depth"]
        if name != "forward-prep":
            ahead = {row["reading"]: row["ahead"] for row in depth["rows"]}
            assert ahead["forward-001"] == ["forward-002", "forward-003"]
    assert len(templates) == 3
    join = rows["forward-join"]["argv"]
    assert "--data-manifest" not in join and "gpu" not in _flag(join, "--demand")
    assert join[join.index("--", join.index("--detach")) + 1:] == [
        "/venv/bin/python", "-m", "prismaquant.stage_a_forward_split",
        "--output-root", str(sealed["root"]),
        "--receipt", str(sealed["round"] / "joins" / "forward-join.json")]


class _FakePbrun:
    def __init__(self, tmp_path):
        self.root = tmp_path / "queue"
        self.submitted = []

    def __call__(self, argv, **kw):
        import subprocess

        key = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
        self.submitted.append(key)
        line = json.dumps({"action_key": key, "status": "submitted",
                           "done": str(self.root / "done" / f"{key}.json"),
                           "failed": str(self.root / "failed" / f"{key}.json")})
        return subprocess.CompletedProcess(argv, 0, stdout=line + "\n", stderr="")

    def finish(self, key, returncode=0):
        _json(self.root / "done" / f"{key}.json",
              {"status": "executed", "detail": {"returncode": returncode}})


def test_each_forward_row_waits_for_the_rows_it_follows(sealed, tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import _write_split_receipt
    from prismaquant.stage_a_forward_split import join_forward_split

    round_dir, root, space = sealed["round"], sealed["root"], sealed["space"]
    labels = sealed["document"]["labels"]
    fake = _FakePbrun(tmp_path)

    def go(*names):
        return submit(round_dir, list(names), run=fake)

    with pytest.raises(SplitDispatchRefused, match="forward-prep was never submitted"):
        go(labels[0])
    go("forward-prep")
    with pytest.raises(SplitDispatchRefused, match="has not finished"):
        go(labels[0])
    fake.finish(fake.submitted[-1])
    with pytest.raises(SplitDispatchRefused, match="no forward split prep"):
        go(labels[0])
    _write_split_receipt(space, _run(root, monkeypatch, forward_split={
        "role": "prep", "ranges": RANGES}))
    go(*labels)
    with pytest.raises(SplitDispatchRefused, match="has not finished"):
        go("forward-join")
    for key, samples in zip(fake.submitted[-2:], RANGES):
        fake.finish(key)
        with pytest.raises(SplitDispatchRefused, match="left no receipt"):
            go("forward-join")
        _write_split_receipt(space, _run(root, monkeypatch, forward_split={
            "role": "quantum", "samples": samples}))
    go("forward-join")
    join_forward_split(space)
    assert chain_state_path(space).is_file()


def test_a_failed_forward_prep_holds_the_quanta(sealed, tmp_path):
    fake = _FakePbrun(tmp_path)
    submit(sealed["round"], ["forward-prep"], run=fake)
    fake.finish(fake.submitted[-1], returncode=3)
    with pytest.raises(SplitDispatchRefused, match="ended executed with exit 3"):
        submit(sealed["round"], [sealed["document"]["labels"][0]], run=fake)


def test_the_forward_readahead_depth_looks_up_the_walk():
    def entry(name, size):
        return {"path": f"/m/{name}", "offset": 0, "bytes": size}

    entries = [entry("head.safetensors", 5), entry("0.safetensors", 100),
               entry("1.safetensors", 100), entry("2.safetensors", 100)]
    phases = [{"name": "head", "entry_indices": [0]},
              {"name": "forward-000", "entry_indices": [1]},
              {"name": "forward-001", "entry_indices": [2]},
              {"name": "forward-002", "entry_indices": [3]}]
    depth = readahead_depth({"entries": entries, "read_plan": {"phases": phases}}, 2)
    reach = {row["reading"]: row["reach_bytes"] for row in depth["rows"]}
    assert reach == {"head": 200, "forward-000": 200, "forward-001": 100, "forward-002": 0}
