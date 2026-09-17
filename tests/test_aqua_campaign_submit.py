"""The campaign's A-side is a submitted stage, not an operator command.

``--require-complete-coverage`` is the campaign's requirement (#655): every
requested ``(unit, format)`` cell the lane's activation contract OWES a price
either carries one -- a joint AURA row, or an A-side priced and merged by
``prismaquant.aqua_activation_cost`` -- or the run refuses. The flag existing
is not the requirement being met; something has to *pass* it, with the plan's
own roster behind it, and that is ``submit-aqua``.

What these tests pin:

* the inner command carries ``--require-complete-coverage`` unconditionally, so
  no invocation of this stage can be a campaign arm with the gate left off;
* the requested roster is the plan's: a cost payload whose units have drifted
  from the plan's bound census refuses, in either direction;
* ``--formats`` has to name exactly the menu the payload carries -- naming a
  subset would move the coverage denominator instead of filling it;
* a payload whose every requested cell is already joint-priced is not
  submitted at all: the requirement is satisfied by that artifact, and queuing
  a stage to add nothing is redundant GPU work;
* the declared read set is the model's own unit extents plus the card, the
  payload and the plan -- the same shared-mount bytes the stage opens;
* ``--cost-out`` may not already exist, because the stage writes it with a
  truncating write and a leftover path is indistinguishable from a result.

The fixture is torch-free on the producer side on purpose: the manifest is
built by the CPU path, exactly as it is at submission time. The stage itself is
never executed here -- these tests run no GPU work.
"""

import hashlib
import json
import pickle
import shlex
import shutil
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from experiments import glm_data_manifests  # noqa: E402

CENSUS_SCHEMA = "prismaquant.tessera_campaign_census.v1"
TENSOR_BYTES = 4096
UNITS = {0: ["model.language_model.layers.0.mlp.gate_proj",
             "model.language_model.layers.0.mlp.up_proj"]}
MEASURED = ("NVFP4", "TESSERA_E4M3_K1_R1000")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256(path)}


def _write_model(model: Path, names: list) -> None:
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    header = {}
    for index, name in enumerate(names):
        header[name + ".weight"] = {
            "dtype": "BF16", "shape": [TENSOR_BYTES // 2],
            "data_offsets": [index * TENSOR_BYTES, (index + 1) * TENSOR_BYTES]}
    blob = json.dumps(header).encode()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob
                      + b"\0" * (TENSOR_BYTES * len(names)))
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name + ".weight": shard.name for name in names}}))


def _cost_payload(names: list, *, formats=MEASURED) -> dict:
    return {"provenance": {},
            "costs": {name: {fmt: {"predicted_dloss": 0.1, "output_mse": 0.2}
                             for fmt in formats} for name in names}}


def _workspace(scratch: Path) -> dict:
    """A one-layer campaign, its joint plan, and the frozen campaign identity."""
    names = [name for layer in sorted(UNITS) for name in UNITS[layer]]
    model = scratch / "model"
    _write_model(model, names)

    campaign = scratch / "campaign"
    campaign.mkdir(parents=True)
    capture = campaign / "capture_manifest.json"
    capture.write_text(json.dumps({"entries": {}}))
    census = campaign / "census.json"
    census.write_text(json.dumps({
        "schema": CENSUS_SCHEMA,
        "model": str(model),
        # The draw, not just the window count: the campaign scope binds which
        # corpus revision and tokenizer ids the calibration windows came from.
        "text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
        "seed": 0, "layer_stride": 1,
        "nsamples": 512, "seqlen": 512,
        "unit_shapes": {name: [16, 16] for name in names},
        "anchor_groups": {"g:layer-0": list(names)},
    }))
    campaign_plan = campaign / "plan.json"
    campaign_plan.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_plan.v1",
        "model": str(model),
        "census": str(census),
        "calibration_cache": _bind(capture),
        "rows": [{"row_id": "row-0000", "dir": str(campaign / "rows/row-0000"),
                  "units": str(campaign / "units/row-0000.json"),
                  "members": list(names), "groups": ["g:layer-0"]}],
    }))
    checkpoint = scratch / "merged.anchors.json"
    checkpoint.write_text(json.dumps({"identity": {}, "identity_sha256": "0" * 64}))
    merged_cost = scratch / "weight-only-cost.pkl"
    merged_cost.write_bytes(pickle.dumps(_cost_payload(names)))
    calibration_tokens = scratch / "calibration_tokens.safetensors"
    calibration_tokens.write_bytes(b"t" * 64)

    plan = scratch / "joint.plan.json"
    plan.write_text(json.dumps({
        "schema": glm_data_manifests.JOINT_PLAN_SCHEMA,
        "model": str(model),
        "output_root": str(scratch / "out"),
        "execution": {"calib_seqlen": 512, "n_calib_samples": 512},
        "calibration_input": _bind(calibration_tokens),
        "canonical_capture": _bind(capture),
        "aggregate_memory_bytes": 114 * 1024 ** 3,
        "max_gpu_bytes": 80 * 1024 ** 3,
        "inputs": {
            "campaign_plan": _bind(campaign_plan),
            "census": _bind(census),
            "merged_checkpoint": _bind(checkpoint),
            "merged_cost": _bind(merged_cost),
            "required_source_units": len(names),
            "required_campaign_groups": 1,
        },
    }))
    cost_in = scratch / "weight-only.pkl"
    cost_in.write_bytes(pickle.dumps(_cost_payload(names)))
    card = scratch / "card.npz"
    card.write_bytes(b"card")
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"},
                                "cpu_memory_gb": 34, "box_memory_gb": 114}))

    import dispatch_tessera_campaign as dispatch

    scope = dispatch.joint_campaign_scope(json.loads(plan.read_text()))
    identity = scratch / "campaign-identity.json"
    identity.write_text(json.dumps(dispatch.campaign_identity(scope),
                                   sort_keys=True))
    return {"plan": plan, "identity": identity, "cost_in": cost_in, "card": card,
            "spec": spec, "checkpoint": checkpoint, "names": names,
            "campaign": campaign, "model": model, "capture": capture}


def submit_aqua(fixture, *extra, cost_out=None):
    import dispatch_tessera_campaign as dispatch

    out = cost_out or (Path(fixture["plan"]).parent / "aqua-cost.pkl")
    return dispatch.main([
        "submit-aqua",
        "--plan", str(fixture["plan"]),
        "--require-scope", "complete_campaign",
        "--campaign-identity", str(fixture["identity"]),
        "--cost-in", str(fixture["cost_in"]),
        "--cost-out", str(out),
        "--card", str(fixture["card"]),
        "--formats", ",".join(MEASURED),
        "--serving-lane", "tessera",
        "--spec", str(fixture["spec"]),
        "--demand", "gpu=1,mem_gb=114", "--cpus", "6",
        "--manifest-dir", str(Path(fixture["plan"]).parent / "manifests"),
        *extra,
    ])


@pytest.fixture()
def scratch(request):
    """A per-test directory that is never under ``/tmp``, like the joint tests."""
    root = ROOT / ".data-manifest-scratch" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def shared_mount(scratch, monkeypatch):
    """The fixture's files are not on /mnt/shared, so the prefix moves to it."""
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(scratch))
    return scratch


@pytest.fixture()
def campaign(scratch, shared_mount, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    monkeypatch.setattr(dispatch, "_manifest_producer",
                        lambda: glm_data_manifests)
    return _workspace(scratch)


def test_the_stage_is_submitted_with_strict_coverage_and_the_plans_roster(
    campaign, capsys,
):
    """The gate is on the command line, and the roster comes from the plan."""
    assert submit_aqua(campaign, "--dry-run") == 0
    printed = capsys.readouterr().out
    command = next(line for line in printed.splitlines()
                   if line.startswith("[dry-run] "))
    argv = shlex.split(command[len("[dry-run] "):])
    inner = argv[argv.index("--data-manifest") + 2:]

    # The gate is unconditional: there is no spelling of this submission that
    # reaches the stage without it.
    assert "--require-complete-coverage" in inner
    assert inner[inner.index("--cost-in") + 1] == str(campaign["cost_in"])
    assert inner[inner.index("--card") + 1] == str(campaign["card"])
    assert inner[inner.index("--model-path") + 1] == str(campaign["model"])
    assert inner[inner.index("--formats") + 1] == ",".join(MEASURED)
    assert inner[inner.index("--serving-lane") + 1] == "tessera"
    # The manifest declares a read set to PrismaBuild, and the reservation is
    # checked against the row's own physical bound rather than taken on trust.
    assert "--data-manifest" in argv
    assert argv[argv.index("--demand") + 1] == "gpu=1,mem_gb=114"
    summary = json.loads(printed[printed.index("{\n"):])
    scope = summary["campaign_scope"]
    assert scope["kind"] == "complete_campaign"
    assert scope["source_unit_count"] == len(campaign["names"])
    # The requested roster the coverage gate reads, bound into the submission's
    # own identity rather than left in the caller's shell history.
    assert scope["requested_units"] == len(campaign["names"])
    assert scope["requested_cells"] == len(campaign["names"]) * len(MEASURED)
    assert scope["formats"] == sorted(MEASURED)


def test_the_declared_read_set_is_the_units_extents_and_the_card(
    campaign, capsys,
):
    """What is queued is what the stage opens, not the whole tree."""
    assert submit_aqua(campaign, "--dry-run") == 0
    manifest = glm_data_manifests.build_aqua_manifest(
        str(campaign["plan"]), card=str(campaign["card"]),
        cost_in=str(campaign["cost_in"]), act_dir=None,
        produced_by={"entry_point": glm_data_manifests.AQUA_ENTRY_POINT})
    paths = {entry["path"] for entry in manifest["entries"]}
    shard = campaign["model"] / "model-00001-of-00001.safetensors"
    assert str(shard) in paths
    assert str(campaign["card"]) in paths
    assert str(campaign["cost_in"]) in paths
    assert str(campaign["plan"]) in paths
    assert manifest["annotations"]["units"] == len(campaign["names"])
    assert manifest["annotations"]["read_order_attested"] is False
    # Per-unit extents, not whole shards: the two 4 KiB tensors are coalesced
    # into their shard's own byte range.
    extents = [entry for entry in manifest["entries"]
               if entry["path"] == str(shard)]
    size = shard.stat().st_size
    assert extents
    assert all(entry["offset"] + entry["bytes"] <= size for entry in extents)
    assert [phase["name"] for phase in manifest["annotations"]["phases"]] == [
        "head", "layer-0"]


def test_a_cost_artifact_that_drifted_from_the_plan_roster_refuses(campaign):
    """A payload is not the campaign's roster just because it is well formed."""
    payload = _cost_payload(campaign["names"][:-1])
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    assert "not the plan's roster" in str(refused.value)

    payload = _cost_payload(campaign["names"] + ["model.language_model.layers.9.mlp.down_proj"])
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    assert "never priced" in str(refused.value)


def test_formats_must_name_the_menu_the_payload_carries(campaign):
    """A narrowed --formats would move the denominator, not fill the hole."""
    import dispatch_tessera_campaign as dispatch

    argv = [
        "submit-aqua", "--plan", str(campaign["plan"]),
        "--require-scope", "complete_campaign",
        "--campaign-identity", str(campaign["identity"]),
        "--cost-in", str(campaign["cost_in"]),
        "--cost-out", str(Path(campaign["plan"]).parent / "narrowed.pkl"),
        "--card", str(campaign["card"]),
        "--formats", MEASURED[0],
        "--serving-lane", "tessera",
        "--spec", str(campaign["spec"]),
        "--demand", "gpu=1,mem_gb=114", "--cpus", "6", "--dry-run",
    ]
    with pytest.raises(RuntimeError) as refused:
        dispatch.main(argv)
    assert "not the menu this artifact carries" in str(refused.value)


def test_an_all_joint_payload_is_not_submitted(campaign, capsys, monkeypatch):
    """A fulfilled artifact is the answer, not a GPU window that adds nothing."""
    from prismaquant import allocator_candidates

    monkeypatch.setattr(allocator_candidates, "cost_entry_is_joint_aura",
                        lambda entry: True)
    assert submit_aqua(campaign, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "[dry-run]" not in printed
    summary = json.loads(printed)
    assert summary["submitted"] is False
    assert "joint A-side" in summary["reason"]


def test_an_existing_cost_out_refuses(campaign):
    """A leftover path is not a previous result, and is never overwritten."""
    out = Path(campaign["plan"]).parent / "already-there.pkl"
    out.write_bytes(b"half a merge")
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run", cost_out=out)
    assert "already exists" in str(refused.value)
