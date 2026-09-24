"""Stage A seed mode: another implementation continues a sealed checkpoint.

RobTand/prismaquant#1016, part of #997. A seed run binds a fresh scratch
root, borrows one sealed checkpoint of a source run and the capsule rows its
chain reads, and rolls the chain down to a ``through`` boundary. The claims
these tests hold it to:

* A seed under another implementation, declared, continues the source run
  bitwise: every rolled plane equals the source run's, and so does the
  checkpoint it seals. Its receipt records the comparison against the source
  run's own checkpoint at ``through``, and it runs no forward pass.
* A seed never writes under the source run's root, and the source run's tree
  is unchanged by a seed or by any refused seed.
* The band tool refuses a seed's space and a seed's sealed request.
* A wrong borrowed digest refuses, as does every other mismatch between the
  seed and what it borrows, before the scratch root holds anything.

The source run is the fixture's five-layer dense model at stride 2
(checkpoints 5, 4 and 2), bound under implementation 2 to a capsule whose
frontier is 3, so a seed from checkpoint 4 through 2 reads capsule rows only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_band import BandRefused, build_band_receipt, stage_a_argv
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_receipt_path,
    adjoint_space,
    checkpoint_cotangent_plane,
    checkpoint_directory,
    checkpoint_entry_session,
    read_exact_entry_tensors,
)
from prismaquant.joint_cost_stage_a import AdjointIdentityRefused
from prismaquant.stage_a_chain_resume import chain_state_path
from prismaquant.stage_a_chain_seed import (
    SEED_COMPARISON_SCHEMA,
    SEED_RECEIPT_SCHEMA,
    SEED_SPEC_SCHEMA,
    seed_marker_path,
    seed_receipt_path,
    tensor_payload_sha256,
)

from test_layer_major_boundary_capture import draw
from test_stage_a_chain_resume import (
    CAMPAIGN,
    DIGESTS,
    N_PROBES,
    ONE,
    THREE,
    TWO,
    _interrupted,
    _merged,
    _names,
    _run,
    _tree,
)


@pytest.fixture(autouse=True)
def _offline_tier_policy():
    """Run with no staged-tier policy, as the runtime tests do (PQ #845)."""
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    yield
    deactivate_staged_tier_policy_for_tests()


SEED_GENERATION = 2002


def _capsule(base, monkeypatch, *, frontier):
    """R9's forward pass, contained after boundary ``frontier``, frozen into a capsule."""
    from prismaquant import joint_forward_resume as recovery_mod
    from test_joint_forward_resume import _bound, _chain_sdk, _owner_spool

    monkeypatch.setattr(recovery_mod, "_sdk", _chain_sdk)
    base.mkdir(parents=True, exist_ok=True)
    refs, identities = {}, []
    last = len(draw()) - 1
    _interrupted(base / "r9", monkeypatch, generation=9, forward_refs=refs,
                 identities=identities,
                 interrupt=lambda kw: kw.get("probe_index") is None and (
                     kw["boundary_index"], kw["batch_index"]) == (frontier, last))
    spool, instance, template = _owner_spool(base, "e" * 64, refs)
    ref = refs[0, 0]
    capsule = base / "r9-capsule.json"
    recovery_mod.build_forward_recovery(specification={
        "schema": recovery_mod.SCHEMA, "queue_root": str(base / "queue"),
        "instance": instance, "template": template,
        "session": json.loads(ref.metadata_json)["identity"]["session"],
        "original_bind_identity": identities[-1],
        "campaign_identity": {**CAMPAIGN, "campaign_scope": None},
        "implementation_compatibility": {"original": ONE, "recovery": TWO,
                                         "scope": "forward-identical-memory-only"},
        "n_batches": len(draw()), "entry_shape": list(ref.shape), "entry_dtype": ref.dtype},
        spool_directory=spool, output=capsule, frontier=frontier)
    return _bound(capsule)


def _pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _manifest(root, boundary):
    return checkpoint_directory(adjoint_space(root), boundary) / "checkpoint.json"


def _source(tmp_path, monkeypatch):
    """The source run A: capsule-bound under implementation 2, stride 2."""
    capsule = _capsule(tmp_path / "capsule", monkeypatch, frontier=3)
    root, writes = tmp_path / "A", []
    receipt = _run(root, monkeypatch, forward_recovery=capsule, implementation=TWO,
                   writes=writes)
    assert [c["boundary"] for c in receipt["checkpoints"]] == [5, 4, 2]
    return SimpleNamespace(root=root, capsule=capsule, planes=_merged(writes),
                           receipt=receipt)


def _spec(source, *, boundary=4, through=2, declaration=None, compare=2, **fields):
    spec = {
        "schema": SEED_SPEC_SCHEMA,
        "checkpoint": _pin(_manifest(source.root, boundary)),
        "capsule": dict(source.capsule),
        "through": through,
        "implementation_compatibility": (
            {"from": TWO, "to": THREE} if declaration is None else declaration),
        "compare": None if compare is None else _pin(_manifest(source.root, compare)),
    }
    spec.update(fields)
    return spec


def _seed(root, monkeypatch, spec, *, implementation=THREE, writes=None,
          forward_refs=None, **kw):
    return _run(root, monkeypatch, implementation=implementation, chain_seed=spec,
                writes=writes, forward_refs=forward_refs, generation=SEED_GENERATION,
                **kw)


def _plane(record):
    """``{(probe, batch): payload sha256}`` of a sealed checkpoint's cotangents."""
    plane = {}
    for key, row in checkpoint_cotangent_plane(record).items():
        tensors = read_exact_entry_tensors(
            [row], expected_session=checkpoint_entry_session(record))
        plane[key] = tensor_payload_sha256(tensors.pop(row["name"]))
    return plane


# -- a declared implementation switch continues bitwise ------------------------

@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_a_seed_under_another_implementation_continues_bitwise(
        tmp_path, monkeypatch, fusion):
    source = _source(tmp_path, monkeypatch)
    before_names, before = _names(source.root), _tree(source.root)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    scratch, writes, forward = tmp_path / "seed", [], {}
    receipt = _seed(scratch, monkeypatch, _spec(source), writes=writes,
                    forward_refs=forward, chain_batch_size=1, chain_probe_fusion=fusion)

    # Every rolled plane is the source run's, and nothing ran forward.
    assert forward == {}
    ours = _merged(writes)
    assert {boundary for boundary, _, _ in ours} == {3, 2}
    assert ours == {key: digest for key, digest in source.planes.items()
                    if key[0] in (3, 2)}
    comparison = receipt["plane_comparison"]
    assert comparison["schema"] == SEED_COMPARISON_SCHEMA
    assert comparison["boundary"] == 2
    assert comparison["bitwise_equal"] is True
    assert comparison["different"] == 0
    assert comparison["equal"] == N_PROBES * len(draw())
    assert all(entry["seed_sha256"] == ours[2, entry["probe"], entry["batch"]]
               for entry in comparison["entries"])

    # The checkpoint it seals holds the source run's tensors under its own
    # session.
    [sealed] = receipt["checkpoints"]
    assert sealed["boundary"] == 2
    reference = json.loads(_manifest(source.root, 2).read_text())
    assert sealed["session"] != reference["session"]
    assert _plane(sealed) == _plane(reference)

    # A measurement, not a campaign run.
    space = adjoint_space(scratch)
    assert receipt["schema"] == SEED_RECEIPT_SCHEMA and receipt["bandable"] is False
    assert receipt["seed"]["implementation_compatibility"] == {
        "scope": "stage-a-chain-seed", "from_implementation_sha256": TWO,
        "to_implementation_sha256": THREE, "seed_checkpoint": 4}
    assert receipt["seed"]["capsule"]["rows"] == [2, 3]
    assert receipt["run_identity"]["implementation_sha256"] == THREE
    # The bf16 reduction flag the chain ran under, read and never set (#1038).
    import torch
    assert receipt["matmul_reduction"] == {
        "allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    assert "matmul_reduction" not in receipt["run_identity"]
    assert seed_marker_path(space).is_file()
    assert not adjoint_receipt_path(space).exists()
    assert not chain_state_path(space).exists()

    # The source run's tree is exactly as it was.
    assert _names(source.root) == before_names
    assert _tree(source.root) == before


# -- a seed seals its through plane, stride boundary or not (#997) --------------

def _stride_plane(source, boundary):
    return {(probe, batch): digest for (at, probe, batch), digest in source.planes.items()
            if at == boundary}


@pytest.mark.parametrize("fusion", [False, True], ids=["unfused", "fused"])
def test_a_seed_seals_its_through_plane_off_the_stride(tmp_path, monkeypatch, fusion):
    """A one-step seed from checkpoint 4 through 3, where 3 is no stride
    boundary: the walk's end retires the rolling entries, so the plane at
    ``through`` survives only as the seed's own sealed checkpoint. The stride
    in its receipt stays the plan's."""
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    receipt = _seed(tmp_path / "seed", monkeypatch, _spec(source, through=3, compare=None),
                    chain_batch_size=1, chain_probe_fusion=fusion)
    [sealed] = receipt["checkpoints"]
    assert sealed["boundary"] == 3
    assert _plane(sealed) == _stride_plane(source, 3)
    assert receipt["stride"]["boundaries"] == source.receipt["stride"]["boundaries"]
    assert 3 not in receipt["stride"]["boundaries"]
    assert receipt["plane_comparison"] is None
    [layer] = receipt["telemetry"]["chain_layers"]
    assert layer["layer"] == 3 and layer["checkpoint"] is True
    assert layer["checkpoint_seal_s"] is not None


def _sealed_pin(root, boundary):
    return _pin(_manifest(root, boundary))


def test_the_plane_distance_of_two_seeds(tmp_path, monkeypatch):
    """``checkpoint_plane_distance`` against known answers: a plane against
    itself is zero and bitwise equal everywhere, a plane rolled as twice the
    reference is exactly one relative L2 away, and a batched seed's distance
    agrees with its payload digests."""
    from prismaquant.stage_a_chain_seed import (
        PLANE_DISTANCE_SCHEMA,
        checkpoint_plane_distance,
    )

    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    spec = _spec(source, through=3, compare=None)
    _seed(tmp_path / "one", monkeypatch, spec)
    reference = _sealed_pin(tmp_path / "one", 3)
    entries = N_PROBES * len(draw())

    same = checkpoint_plane_distance(reference, reference)
    assert same["schema"] == PLANE_DISTANCE_SCHEMA and same["boundary"] == 3
    assert same["equal"] == entries and same["different"] == 0
    assert same["relative_l2"] == {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0}
    assert same["max_abs"] == 0.0

    # Mutation: a seed whose roll writes twice every cotangent.
    roll = stage_a.render_free_layer_roll

    def doubled(*args, **kw):
        inner = kw["roll"]
        kw["roll"] = lambda tensor, batch, probe: inner(tensor * 2, batch, probe)
        return roll(*args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(stage_a, "render_free_layer_roll", doubled)
        _seed(tmp_path / "two", monkeypatch, spec)
    twice = checkpoint_plane_distance(reference, _sealed_pin(tmp_path / "two", 3))
    assert twice["different"] == entries
    assert {entry["relative_l2"] for entry in twice["entries"]} == {1.0}
    record = json.loads(Path(reference["path"]).read_text())
    largest = max(float(tensor.abs().max()) for tensor in (
        read_exact_entry_tensors(
            [row], expected_session=checkpoint_entry_session(record)).popitem()[1]
        for row in record["activation_entries"]))
    assert twice["max_abs"] == largest

    # A batched, fused seed: the distance names exactly the entries whose
    # payload digests differ.
    _seed(tmp_path / "batched", monkeypatch, spec, chain_batch_size=2,
          chain_probe_fusion=True)
    batched = _sealed_pin(tmp_path / "batched", 3)
    distance = checkpoint_plane_distance(reference, batched, read_ahead=1)
    ours = _plane(json.loads(Path(reference["path"]).read_text()))
    theirs = _plane(json.loads(Path(batched["path"]).read_text()))
    assert {(entry["probe"], entry["batch"]): entry["bitwise_equal"]
            for entry in distance["entries"]} == {
        key: ours[key] == theirs[key] for key in ours}
    assert all((entry["relative_l2"] == 0.0) == entry["bitwise_equal"]
               for entry in distance["entries"])
    assert distance["relative_l2"]["max"] < 1e-3


def test_the_plane_distance_refuses_what_it_cannot_compare(tmp_path, monkeypatch):
    from prismaquant.stage_a_chain_seed import ChainSeedRefused, checkpoint_plane_distance

    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    _seed(tmp_path / "seed", monkeypatch, _spec(source, through=3, compare=None))
    ours = _sealed_pin(tmp_path / "seed", 3)
    with pytest.raises(ChainSeedRefused, match="reference checkpoint is boundary 2 and "
                                                "the candidate 3"):
        checkpoint_plane_distance(_sealed_pin(source.root, 2), ours)
    # The source run's checkpoint 2 and a seed's are one boundary, but not
    # one science: the seed runs under another implementation.
    _seed(tmp_path / "seed-2", monkeypatch, _spec(source, compare=None))
    with pytest.raises(ChainSeedRefused, match="different bind identities"):
        checkpoint_plane_distance(_sealed_pin(source.root, 2),
                                  _sealed_pin(tmp_path / "seed-2", 2))
    with pytest.raises(ChainSeedRefused, match="does not have the pinned digest"):
        checkpoint_plane_distance(ours, _wrong(ours))
    with pytest.raises(ValueError, match="read_ahead"):
        checkpoint_plane_distance(ours, ours, read_ahead=0)


def test_the_plane_distance_tool_writes_its_record_once(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from compare_stage_a_checkpoints import main

    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    _seed(tmp_path / "seed", monkeypatch, _spec(source, through=3, compare=None))
    ours = _sealed_pin(tmp_path / "seed", 3)
    output = tmp_path / "distance.json"
    argv = ["--reference", ours["path"], "--reference-sha256", ours["sha256"],
            "--candidate", ours["path"], "--candidate-sha256", ours["sha256"],
            "--output", str(output), "--device", "cpu"]
    capsys.readouterr()
    assert main(argv) == 0
    record = json.loads(output.read_text())
    assert record["equal"] == N_PROBES * len(draw()) and record["device"] == "cpu"
    assert json.loads(capsys.readouterr().out.strip())["different"] == 0
    with pytest.raises(SystemExit):
        main(argv)
    assert json.loads(output.read_text()) == record


def test_the_seed_receipt_is_written_once_beside_its_marker(tmp_path, monkeypatch):
    from prismaquant.stage_a_chain_seed import ChainSeedRefused, write_seed_receipt

    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    scratch = tmp_path / "seed"
    receipt = _seed(scratch, monkeypatch, _spec(source))
    space = adjoint_space(scratch)
    written = write_seed_receipt(space, receipt)
    assert written["path"] == str(seed_receipt_path(space))
    assert json.loads(seed_receipt_path(space).read_text()) == receipt
    with pytest.raises(ChainSeedRefused, match="no seed marker"):
        write_seed_receipt(adjoint_space(source.root), receipt)
    with pytest.raises(ChainSeedRefused, match="written once"):
        write_seed_receipt(space, receipt)
    with pytest.raises(ChainSeedRefused, match="only a seed receipt"):
        write_seed_receipt(space, {**receipt, "schema": "prismaquant.adjoint_capture.v1"})


# -- the band tool refuses a seed ----------------------------------------------

def test_the_band_tool_refuses_a_seed_space_and_a_seed_request(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    scratch = tmp_path / "seed"
    # The receipt records the flag's value, whichever it is (#1038).
    import torch
    monkeypatch.setattr(torch.backends.cuda.matmul,
                        "allow_bf16_reduced_precision_reduction", False)
    receipt = _seed(scratch, monkeypatch, _spec(source))
    assert receipt["matmul_reduction"] == {
        "allow_bf16_reduced_precision_reduction": False}
    with pytest.raises(BandRefused, match="marks a Stage A seed run"):
        build_band_receipt(output_root=scratch, boundary=2, stride_value=2,
                           forward_recovery=source.capsule, **DIGESTS)
    # The source run's own band is still built.
    band = build_band_receipt(output_root=source.root, boundary=2, stride_value=2,
                              forward_recovery=source.capsule, **DIGESTS)
    assert band["band"]["boundary"] == 2
    command = ["python3", "-m", "prismaquant.joint_cost_stage_a", "--plan", "/p",
               "--plan-sha256", "b" * 64, "--prepared", "/q", "--prepared-sha256",
               "c" * 64, "--output-root", str(scratch)]
    assert stage_a_argv(command)["--output-root"] == str(scratch)
    with pytest.raises(BandRefused, match="is a Stage A seed run"):
        stage_a_argv(command + ["--chain-seed", "/s", "--chain-seed-sha256", "f" * 64])


# -- refusals -------------------------------------------------------------------

def _wrong(pin):
    return {**pin, "sha256": "0" * 64}


#: Each seed and the reason it must refuse for.
REFUSALS = {
    "checkpoint-digest": (
        lambda s: _spec(s, checkpoint=_wrong(_pin(_manifest(s.root, 4)))),
        "checkpoint .* does not have the pinned digest"),
    "capsule-digest": (
        lambda s: _spec(s, capsule=_wrong(s.capsule)), "the seed's capsule refuses"),
    "compare-digest": (
        lambda s: _spec(s, compare=None) | {"compare": _wrong(_pin(_manifest(s.root, 2)))},
        "compare checkpoint .* does not have the pinned digest"),
    "undeclared-switch": (
        lambda s: _spec(s, implementation_compatibility=None),
        "was not sealed by this science under implementation 3{64}"),
    "declared-from-the-wrong-implementation": (
        lambda s: _spec(s, declaration={"from": ONE, "to": THREE}),
        "was not sealed by this science under implementation 1{64}"),
    "declaration-without-a-switch": (
        lambda s: _spec(s, declaration={"from": THREE, "to": THREE}),
        "names no switch"),
    "through-at-the-checkpoint": (
        lambda s: _spec(s, through=4, compare=4), "rolls down to a boundary below it"),
    "rows-beyond-the-capsule": (
        lambda s: _spec(s, boundary=5), r"does not hold the forward boundaries \[4\]"),
    "compare-at-another-boundary": (
        lambda s: _spec(s, compare=4), "compare checkpoint is boundary 4"),
    "extra-field": (
        lambda s: _spec(s, stride=2), "with exactly the fields"),
}


def _refused(tmp_path, source, monkeypatch, scratch, spec, match, **kw):
    """Refuse the seed for ``match``; the source run and the scratch root are untouched."""
    before_names, before = _names(source.root), _tree(source.root)
    with pytest.raises(AdjointIdentityRefused, match=f"chain seed refused: .*{match}"):
        _seed(scratch, monkeypatch, spec, **kw)
    assert _names(source.root) == before_names
    assert _tree(source.root) == before


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_seed_that_does_not_match_what_it_borrows_refuses(tmp_path, monkeypatch, name):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    build, match = REFUSALS[name]
    scratch = tmp_path / "seed"
    _refused(tmp_path, source, monkeypatch, scratch, build(source), match)
    # Nothing but the empty adjoint space: no marker, no generation, no entry.
    assert [path for path in scratch.rglob("*") if path.is_file()] == []



def test_a_seed_declared_to_another_implementation_stamps_and_continues(
        tmp_path, monkeypatch, capsys):
    """The declaration's TO is a run seal (PQ #1147); its FROM still binds the session."""
    source = _source(tmp_path, monkeypatch)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    capsys.readouterr()
    _seed(tmp_path / "seed", monkeypatch, _spec(source, declaration={"from": TWO, "to": ONE}))
    assert "[DEV-MODE] seal seed implementation differs" in capsys.readouterr().out

def test_a_seed_never_writes_under_the_source_root(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    # Inside the source root first: a refusal after the core's first mkdir
    # would still leave a directory there.
    for scratch in (source.root / "scratch", adjoint_space(source.root) / "seed",
                    source.root, tmp_path):
        _refused(tmp_path, source, monkeypatch, scratch, _spec(source),
                 "is the source run's root .*, inside it, or around it")
    assert not (source.root / "scratch").exists()


def test_certified_mode_refuses_a_seed(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    scratch = tmp_path / "seed"
    _refused(tmp_path, source, monkeypatch, scratch, _spec(source),
             "requires PRISMAQUANT_DEV_MODE=1")
    assert not scratch.exists()


def test_a_seed_is_neither_a_resume_nor_a_forward_recovery(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    scratch = tmp_path / "seed"
    for extra in ({"forward_recovery": source.capsule},
                  {"chain_resume": {"chain_state_sha256": "f" * 64, "declaration": None,
                                    "resume_from": None}}):
        _refused(tmp_path, source, monkeypatch, scratch, _spec(source),
                 "neither a chain resume nor a forward recovery", **extra)
    assert not scratch.exists()


def test_a_seed_binds_a_fresh_root_and_another_runs_checkpoint(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    scratch = tmp_path / "seed"
    _seed(scratch, monkeypatch, _spec(source))
    # A second seed into the same scratch root.
    _refused(tmp_path, source, monkeypatch, scratch, _spec(source),
             "a seed binds a fresh scratch root, and this one already holds")
    # A seed whose compare checkpoint is another run's (the first seed's).
    spec = _spec(source) | {"compare": _pin(_manifest(scratch, 2))}
    _refused(tmp_path, source, monkeypatch, tmp_path / "again", spec,
             "not the source run's own checkpoint")
    # A seed of a seed: its checkpoint 2 is under the first seed's root, and
    # the first seed's session is not this science under implementation 2.
    spec = _spec(source, compare=None, through=0) | {
        "checkpoint": _pin(_manifest(scratch, 2))}
    with pytest.raises(AdjointIdentityRefused, match="was not sealed by this science"):
        _seed(tmp_path / "third", monkeypatch, spec)


# -- the command line ------------------------------------------------------------

def test_the_seed_flags(tmp_path, capsys):
    base = ["--plan", "/p", "--plan-sha256", "b" * 64, "--prepared", "/q",
            "--prepared-sha256", "c" * 64, "--output-root", "/o"]
    for extra, message in (
            (["--chain-seed", "/s"], "must be paired"),
            (["--chain-seed", "/s", "--chain-seed-sha256", "f" * 64,
              "--forward-recovery", "/r", "--forward-recovery-sha256", "e" * 64],
             "takes no --resume-chain-state-sha256 or --forward-recovery")):
        with pytest.raises(SystemExit) as exit_info:
            stage_a.main(base + extra)
        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
    spec = tmp_path / "seed.json"
    spec.write_text(json.dumps({"schema": SEED_SPEC_SCHEMA}))
    args = SimpleNamespace(chain_seed=spec, chain_seed_sha256="f" * 64)
    with pytest.raises(AdjointIdentityRefused, match="does not have the pinned digest"):
        stage_a._chain_seed_argument(args)
    args.chain_seed_sha256 = hashlib.sha256(spec.read_bytes()).hexdigest()
    with pytest.raises(AdjointIdentityRefused, match="with exactly the fields"):
        stage_a._chain_seed_argument(args)
    assert stage_a._chain_seed_argument(SimpleNamespace(chain_seed=None)) is None


def test_the_cli_takes_the_plan_of_the_source_run(tmp_path, monkeypatch):
    """The plan names the source run's root; --output-root is the scratch root."""
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr("prismaquant.gpu_guard.require_cuda_hot_path", lambda *a: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    from prismaquant.tessera_joint_aura import ACTIVATION_SCALE_ENV
    monkeypatch.setenv(ACTIVATION_SCALE_ENV, "0")
    spec = _spec(source)
    common = dict(plan_sha256="b" * 64, prepared={"path": "/q", "sha256": "c" * 64},
                  chain_seed=spec)
    execution = {"production_act_scales": "0"}
    with pytest.raises(AdjointIdentityRefused,
                       match="seed checkpoint is not under the plan's output_root"):
        stage_a.run_adjoint_capture(
            {"execution": execution, "output_root": str(tmp_path / "elsewhere")},
            output_root=tmp_path / "seed", **common)
    with pytest.raises(AdjointIdentityRefused, match="inside it, or around it"):
        stage_a.run_adjoint_capture(
            {"execution": execution, "output_root": str(source.root)},
            output_root=source.root / "seed", **common)
    assert not (source.root / "seed").exists()
    with pytest.raises(AdjointIdentityRefused, match="neither a chain resume"):
        stage_a.run_adjoint_capture(
            {"execution": execution, "output_root": str(source.root)},
            output_root=tmp_path / "seed", forward_recovery=source.capsule, **common)
    assert not (tmp_path / "seed").exists()
