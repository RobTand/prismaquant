"""The Stage A split round dispatcher (PQ #738).

The dispatcher seals a round's rows from one checkout and submits each row
only once the rows it follows have finished, by what they left on disk and
by their PrismaBuild endings. These tests seal a round of the split
fixture's run (``test_stage_a_chain_split``: five layers, five samples in
windows of two, interrupted below checkpoint 4, split 4 -> 2 as ``0:2`` and
``2:5``) and drive the ordering with a fake ``pbrun``; the rows the fake
"runs" are the fixture's own core calls, so the receipts the gates read are
real ones.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prismaquant.joint_adjoint_checkpoints import adjoint_space
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.tessera_joint_aura import HEAD_WALK_INPUT_KEYS

from stage_a_spool_spec import stage_a_plan, with_spool
from test_stage_a_chain_resume import (  # noqa: F401  (autouse fixture)
    N_PROBES,
    ONE,
    TWO,
    _at,
    _interrupted,
    _offline_tier_policy,
    _resume,
    _run,
)
from test_stage_a_chain_split import RANGES, _quantum

import dispatch_stage_a_split as split_dispatch
from dispatch_stage_a_split import (
    SplitDispatchRefused,
    compare_digests,
    load_round,
    quantum_read_rate,
    readahead_depth,
    seal_round,
    submit,
    window_groups,
)

TIER = "prismabuild-stage:fixture"
BUDGET = 1 << 30


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True))
    return Path(path)


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """An interrupted run, its split package and its sealed round."""
    import dispatch_joint_quanta
    from tools.build_stagea_split_package import build

    root = tmp_path / "run"
    _interrupted(root, monkeypatch, interrupt=_at(3, 1, 2))
    plan = _json(tmp_path / "plan.json", {
        **stage_a_plan(tmp_path, output_root=str(root)),
        "inputs": {key: {"path": f"{tmp_path}/walk/{key}.json", "sha256": "0" * 64}
                   for key in HEAD_WALK_INPUT_KEYS}})
    campaign = {"plan_path": str(plan), "plan_sha256": _sha(plan),
                "prepared_path": str(tmp_path / "prepared.json"),
                "prepared_sha256": "b" * 64, "read_manifest_sha256": "d" * 64}
    names = ["head", *(f"chain-{layer:03d}" for layer in range(4, -1, -1))]
    entries = [{"path": f"{tmp_path}/source/{name}.safetensors", "offset": 0,
                "bytes": 1000 * (index + 1), "sha256": None}
               for index, name in enumerate(names)]
    phases, total = [], 0
    for index, name in enumerate(names):
        total += entries[index]["bytes"]
        phases.append({"name": name, "entry_indices": [index],
                       "bytes": entries[index]["bytes"], "cumulative_bytes": total})
    source = tmp_path / "source-manifest.json.gz"
    source.write_bytes(gzip.compress(json.dumps({
        "schema": "prismaquant.prismabuild.data_manifest.v2", "mount_prefix": "/mnt/shared",
        "produced_by": {"tool": "fixture"}, "entries": entries,
        "entry_count": len(entries), "total_bytes": total,
        "annotations": {"parent_manifest_sha256": "d" * 64,
                        "plan_sha256": campaign["plan_sha256"],
                        "prepared_sha256": "b" * 64},
        "read_plan": {"phases": phases, "read_bytes": total}}).encode(), mtime=0))
    space = adjoint_space(root)
    build(original_manifest=source, original_manifest_sha256=_sha(source),
          plan={"path": str(plan), "sha256": campaign["plan_sha256"]}, output_root=root,
          chain_state_sha256=_sha(chain_state_path(space)),
          checkpoint_sha256=_sha(space / "checkpoints" / "boundary-004" / "checkpoint.json"),
          through=2, ranges=RANGES, output=tmp_path / "package")
    spec = _json(tmp_path / "spec.json", with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}}))
    prefetch = _json(tmp_path / "prefetch.json",
                     {"source_prefetch": {"prefetch_lookahead": 2}})
    base = _json(tmp_path / "template.json", {
        "schema": "prismaquant.prismabuild.produced_output_template.v1", "version": 1,
        "template_id": "fixture-base", "output_prefix": str(space),
        "slots": {"boundary_entries": {"class": "payload"}},
        "durable_maxima": {"payload_max_bytes": BUDGET, "checkpoint_max_bytes": BUDGET,
                           "temp_max_bytes": BUDGET},
        "working_demands": {TIER: {"minimum_gib": 2, "window_gib": 48}},
        "permitted_tiers": [TIER]})
    band_request = _json(tmp_path / "request.json", {"params": {}})
    earlier = _json(tmp_path / "band-005.json", {"band": {"boundary": 5}})
    monkeypatch.setattr(dispatch_joint_quanta, "_pbrun_seals_produced_output", lambda *a: True)
    monkeypatch.setattr(split_dispatch, "checkout_commit", lambda checkout: "c" * 40)
    monkeypatch.setattr(split_dispatch, "implementation_sha256", lambda checkout: TWO)
    document = seal_round(
        round_dir=tmp_path / "round", checkout=tmp_path / "checkout",
        split_package=tmp_path / "package" / "split-package.json", campaign=campaign,
        spec=spec, prefetch_override=prefetch, base_template=base, tier=TIER,
        template_prefix="fixture-round-1", artifact_budget_bytes=BUDGET,
        chain_regime={"chain_batch_size": 1, "chain_probe_fusion": "off"},
        seconds_per_layer=100.0, digest_layer=3,
        band_request={"path": str(band_request), "sha256": _sha(band_request)},
        band_references=[{"path": str(earlier)}], python="/venv/bin/python")
    return {"root": root, "round": tmp_path / "round", "document": document,
            "campaign": campaign, "space": space}


def _payload(row):
    return row["argv"][row["argv"].index("prismaquant.joint_adjoint_capture") + 1:]


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _envelope(row):
    return row["argv"][:row["argv"].index("--")]


def test_a_sealed_round_names_every_row_and_its_order(sealed):
    document = sealed["document"]
    rows = {row["name"]: row for row in document["rows"]}
    labels = ["from-004-through-002-samples-000000-000002",
              "from-004-through-002-samples-000002-000005"]
    assert list(rows) == ["prep", *labels, "join-002", "band-002", "band-set"]
    assert document["labels"] == labels and document["ranges"] == RANGES
    assert load_round(sealed["round"]) == document

    # The prep: the resume, the split flags and the declaration of the
    # implementation switch (the run was sealed by ONE; the checkout is TWO).
    prep = _payload(rows["prep"])
    assert _flag(prep, "--chain-split-prep") == "2"
    assert _flag(prep, "--chain-split-ranges") == "0:2,2:5"
    assert _flag(prep, "--resume-from-checkpoint") == "4"
    assert _flag(prep, "--resume-chain-state-sha256") == _sha(
        chain_state_path(sealed["space"]))
    assert _flag(prep, "--resume-implementation-compatibility") == f"{ONE}:{TWO}"
    assert document["implementation_declaration"] == f"{ONE}:{TWO}"
    # Each quantum: its range and digest layer, and no declaration.
    for label, (start, stop) in zip(labels, RANGES):
        payload = _payload(rows[label])
        assert _flag(payload, "--chain-split-quantum") == f"2:{start}:{stop}"
        assert _flag(payload, "--chain-split-digest-layer") == "3"
        assert "--resume-implementation-compatibility" not in payload
        assert "--chain-split-prep" not in payload
    # Every Stage A row: the gb10 tag alone, priority -10, the full residency
    # path, its own manifest and its own template.
    templates = set()
    for name in ["prep", *labels]:
        envelope = _envelope(rows[name])
        assert [envelope[i + 1] for i, word in enumerate(envelope) if word == "--tag"] == [
            "gb10"]
        assert _flag(envelope, "--priority") == "-10"
        assert _flag(envelope, "--residency") == "stage"
        assert _flag(envelope, "--residency-ram") == "auto"
        assert _flag(envelope, "--data-manifest") == rows[name]["data_manifest"]["path"]
        template = json.loads(Path(_flag(envelope, "--produced-output-template")).read_text())
        templates.add(template["template_id"])
        assert template["output_prefix"] == str(sealed["space"])
        assert template["durable_maxima"]["payload_max_bytes"] == BUDGET
        assert "--host-class" not in envelope and "--measurement" not in envelope
    assert len(templates) == 3
    # CPU rows: no GPU, no manifest; the join writes its receipt into the round.
    join = rows["join-002"]["argv"]
    assert "--data-manifest" not in join and "gpu" not in _flag(join, "--demand")
    assert join[join.index("--", join.index("--detach")) + 1:] == [
        "/venv/bin/python", "-m", "prismaquant.stage_a_chain_split",
        "--output-root", str(sealed["root"]), "--boundary", "2",
        "--receipt", str(sealed["round"] / "joins" / "join-002.json")]
    assert rows["band-set"]["bands"] == [str(sealed["round"].parent / "band-005.json"),
                                         str(sealed["round"] / "bands" / "band-002.json")]


def test_a_quantum_seals_its_own_windows(sealed):
    import dispatch_joint_quanta

    rows = {row["name"]: row for row in sealed["document"]["rows"]}
    for label, samples in zip(sealed["document"]["labels"], RANGES):
        envs = dict(value.split("=", 1) for flag, value in zip(
            _envelope(rows[label]), _envelope(rows[label])[1:]) if flag == "--env")
        assert int(envs["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"]) == (
            dispatch_joint_quanta.stage_a_spool_window_bytes(sealed["campaign"], samples))
    # The prep writes nothing; it is sized as the widest quantum.
    envs = dict(value.split("=", 1) for flag, value in zip(
        _envelope(rows["prep"]), _envelope(rows["prep"])[1:]) if flag == "--env")
    assert int(envs["PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"]) == (
        dispatch_joint_quanta.stage_a_spool_window_bytes(sealed["campaign"], RANGES[1]))
    assert rows["prep"]["window"] == rows[sealed["document"]["labels"][1]]["window"]
    # Windows of two batches: 0:2 is one per boundary, 2:5 two; each boundary
    # holds a boundary group and one per probe, two boundaries at once.
    first, second = (rows[label]["window"] for label in sealed["document"]["labels"])
    assert (first["groups"], first["window_gib"]) == (2 * (1 + N_PROBES), 20)
    assert (second["groups"], second["window_gib"]) == (min(24, 4 * (1 + N_PROBES)), 40)


class _FakePbrun:
    """Records each submission and answers with a detach line."""

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


def test_each_row_waits_for_the_rows_it_follows(sealed, tmp_path, monkeypatch):
    from prismaquant.joint_cost_stage_a import _write_split_receipt

    round_dir, root, space = sealed["round"], sealed["root"], sealed["space"]
    labels = sealed["document"]["labels"]
    fake = _FakePbrun(tmp_path)

    def go(*names):
        return submit(round_dir, list(names), run=fake)

    with pytest.raises(SplitDispatchRefused, match="row prep was never submitted"):
        go(labels[0])
    with pytest.raises(SplitDispatchRefused, match="never submitted"):
        go("join-002")
    go("prep")
    with pytest.raises(SplitDispatchRefused, match="already submitted"):
        go("prep")
    with pytest.raises(SplitDispatchRefused, match="has not finished"):
        go(labels[0])
    fake.finish(fake.submitted[-1])
    # PrismaBuild says the prep executed, but it left no receipt yet.
    with pytest.raises(SplitDispatchRefused, match="not this round's"):
        go(labels[0])
    # The prep the row ran, under the implementation the round was sealed with.
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    receipt = _run(root, monkeypatch, implementation=TWO,
                   chain_resume=_resume(root, declaration={"from": ONE, "to": TWO},
                                        resume_from=4),
                   chain_split={"role": "prep", "through": 2, "ranges": RANGES})
    _write_split_receipt(space, receipt)
    go(*labels)
    with pytest.raises(SplitDispatchRefused, match="has not finished"):
        go("join-002")
    for key, samples in zip(fake.submitted[-2:], RANGES):
        fake.finish(key)
        with pytest.raises(SplitDispatchRefused, match="left no receipt"):
            go("join-002")
        _write_split_receipt(space, _quantum(root, monkeypatch, samples,
                                             implementation=TWO))
    go("join-002")
    with pytest.raises(SplitDispatchRefused, match="has not finished"):
        go("band-002")
    fake.finish(fake.submitted[-1])
    with pytest.raises(SplitDispatchRefused, match="not joined"):
        go("band-002")
    submissions = json.loads((round_dir / "submissions" / "prep.json").read_text())
    assert submissions["detach"]["action_key"] == fake.submitted[0]


def test_a_failed_prep_holds_the_quanta(sealed, tmp_path):
    fake = _FakePbrun(tmp_path)
    submit(sealed["round"], ["prep"], run=fake)
    fake.finish(fake.submitted[-1], returncode=3)
    with pytest.raises(SplitDispatchRefused, match="ended executed with exit 3"):
        submit(sealed["round"], [sealed["document"]["labels"][0]], run=fake)


def test_a_changed_input_refuses_every_row(sealed, tmp_path):
    manifest = Path(sealed["document"]["rows"][1]["data_manifest"]["path"])
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(SplitDispatchRefused, match="changed since the round was sealed"):
        submit(sealed["round"], ["prep"], run=_FakePbrun(tmp_path))


def test_the_digest_tripwire(sealed, tmp_path):
    from prismaquant.stage_a_chain_split import quantum_directory

    labels = sealed["document"]["labels"]
    directory = quantum_directory(sealed["space"])
    baseline = {f"cotangent-{p}-{b}-at-3.pt": f"{p}{b}" * 32
                for p in range(N_PROBES) for b in range(5)}
    path = _json(tmp_path / "baseline.json", {"payload_sha256": baseline})
    report = compare_digests(sealed["document"], path)
    assert report["mismatches"] == 0 and {q["status"] for q in report["quanta"].values()} == {
        "pending"}
    _json(directory / f"{labels[0]}.digests.json", {"layer": 3, "payload_sha256": {
        f"{p}-{b}": f"{p}{b}" * 32 for p in range(N_PROBES) for b in range(2)}})
    good = compare_digests(sealed["document"], path)
    assert good["quanta"][labels[0]]["status"] == "match" and good["compared"] == 8
    _json(directory / f"{labels[1]}.digests.json", {"layer": 3, "payload_sha256": {
        "0-2": "0" * 64, "1-2": "12" * 32}})
    bad = compare_digests(sealed["document"], path)
    assert bad["quanta"][labels[1]]["status"] == "mismatch" and bad["mismatches"] == 1
    assert bad["quanta"][labels[1]]["mismatches"][0]["entry"] == "cotangent-0-2-at-3.pt"


def test_the_readahead_depth_counts_the_chain_windows_source_ahead():
    """``chain-044`` reads ahead 043 and 042; the depth is the furthest source byte
    past the end of the phase being read. Rows are read in their own phase, so
    the last phase ahead counts to the end of its source, not of its rows."""
    def entry(name, size):
        return {"path": f"/m/{name}", "offset": 0, "bytes": size}

    entries = [entry("head.safetensors", 5), entry("44.safetensors", 100),
               entry("plane.pt", 50), entry("43.safetensors", 100), entry("rows43.pt", 7),
               entry("42.safetensors", 100), entry("rows42.pt", 7)]
    phases = [{"name": "head", "entry_indices": [0]},
              {"name": "chain-044", "entry_indices": [1, 2]},
              {"name": "chain-043", "entry_indices": [3, 4]},
              {"name": "chain-042", "entry_indices": [5, 6]}]
    depth = readahead_depth({"entries": entries, "read_plan": {"phases": phases}}, 2)
    reach = {row["reading"]: row["reach_bytes"] for row in depth["rows"]}
    assert reach == {"head": 50 + 100 + 100, "chain-044": 7 + 100 + 100,
                     "chain-043": 100, "chain-042": 0}
    assert depth["declared_gib"] == 1
    assert readahead_depth({"entries": entries[:1], "read_plan": {
        "phases": phases[:1]}}, 2)["declared_gib"] == 0


def test_the_read_rate_estimate_and_the_window():
    described = {"phases": [{"name": "head", "bytes": 1}, {"name": "chain-044", "bytes": 9e9},
                            {"name": "chain-043", "bytes": 4e9},
                            {"name": "chain-042", "bytes": 5e9}]}
    rate = quantum_read_rate(described, samples=[64, 128], n_batches=512,
                             seconds_per_layer=1000.0)
    assert rate["phase"] == "chain-042" and rate["read_mb_s"] == 40
    base = {"working_demands": {TIER: {"minimum_gib": 2, "window_gib": 48}}}
    assert window_groups(base, tier=TIER, n_probes=4, range_batches=64,
                         group_size=64)["window_gib"] == 20
    assert window_groups(base, tier=TIER, n_probes=4, range_batches=512,
                         group_size=64)["window_gib"] == 48
