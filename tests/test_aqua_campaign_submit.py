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
* the requested roster is the plan's *cells*, not the artifact's: the plan
  binds its own cost table (``inputs.merged_cost``) by sha256, and ``--cost-in``
  has to reproduce that table's ``(unit, format)`` cells unit for unit. A table
  that dropped one unit's cell while another unit still carries that format is
  the case a unit-roster check and a carried-format union check both pass, and
  it refuses here; a unit whose entry is empty refuses too; and a cost table
  edited after the plan was sealed refuses on the plan's own sha256;
* ``--formats`` has to name exactly the menu the payload carries -- naming a
  subset would move the coverage denominator instead of filling it;
* a payload whose every requested cell is already joint-priced is not
  submitted at all: the requirement is satisfied by that artifact, and queuing
  a stage to add nothing is redundant GPU work;
* a joint row is only its own cell. A row carries the activation term of the
  operator it measured, so a valid row donated from another unit or another
  rung refuses instead of reporting coverage for the cell it landed on -- and
  a joint-priced cell the plan's own table never priced refuses too, because
  a row bound to its own coordinate is still not bound to *this* campaign's
  draw, capture and candidate menu. Reusing such an artifact is explicit,
  ``--accept-joint-cells-outside-plan``, and recorded as unverified;
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


def _joint_row(name: str, fmt: str, *, produced_for=None) -> dict:
    """A joint-shaped row whose operator identity names a cell.

    ``cost_entry_is_joint_aura`` is stubbed in these tests -- the real
    predicate is exercised against a measured joint table in
    ``test_joint_aura_allocator_currency`` -- so what this row has to carry is
    the coordinate the coverage decision compares. ``produced_for`` is how a
    test donates a row: it is the cell the row claims, which is not the cell it
    will be found under.
    """
    qname, format_name = produced_for or (name, fmt)
    return {"predicted_dloss": 0.1,
            "joint_operator_identity": {"qname": qname, "format": format_name}}


@pytest.fixture()
def joint_row_is_joint(monkeypatch):
    """The stub: anything carrying a joint operator identity is a joint row."""
    from prismaquant import allocator_candidates

    monkeypatch.setattr(allocator_candidates, "cost_entry_is_joint_aura",
                        lambda entry: "joint_operator_identity" in entry)
    return allocator_candidates


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


def test_a_cell_dropped_from_one_unit_refuses_though_another_unit_keeps_it(
    campaign,
):
    """The unit roster and the carried-format union both survive this narrowing.

    Both units were priced for both formats. Deleting one unit's ``NVFP4`` cell
    leaves every unit present and every format still carried somewhere, so a
    check stated against the artifact's own keys reads as complete -- and an
    all-joint remainder would then short-circuit the gate. The plan's cost table
    is what says the cell was owed, and it is what refuses this.
    """
    names = campaign["names"]
    payload = _cost_payload(names)
    del payload["costs"][names[0]][MEASURED[0]]
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    message = str(refused.value)
    assert "not the plan's priced surface" in message
    assert f"{names[0]}@{MEASURED[0]}" in message
    assert "does not carry 1 planned (unit, format) cell(s)" in message


def test_a_unit_whose_entry_is_empty_refuses(campaign):
    """An entry that survived the roster but lost every cell is a hole, not a unit."""
    names = campaign["names"]
    payload = _cost_payload(names)
    payload["costs"][names[0]] = {}
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    message = str(refused.value)
    assert "not the plan's priced surface" in message
    assert f"{names[0]}@<entry empty>" in message


def test_a_cost_table_edited_after_the_plan_was_sealed_refuses(campaign):
    """The plan's sha256 is the binding, so the file it names has to be that file."""
    names = campaign["names"]
    payload = _cost_payload(names)
    del payload["costs"][names[0]][MEASURED[0]]
    merged = Path(campaign["plan"].parent / "weight-only-cost.pkl")
    merged.write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    message = str(refused.value)
    assert "inputs.merged_cost" in message
    assert "not the one on disk" in message


def test_the_plan_may_price_a_narrower_surface_than_the_census(campaign):
    """The roster the gate reads is the plan's cells, not a cartesian product.

    A campaign whose plan prices one format for one unit and two for another is
    not narrowed *by* the artifact: the plan's own cost table says so, and the
    artifact reproducing it is complete. This is the shape the real campaign has
    -- dense targets carry the whole menu, routed experts only their measured
    rungs -- and it is why the expected set is read per ``(unit, format)``
    rather than as units times formats.
    """
    import dispatch_tessera_campaign as dispatch

    names = campaign["names"]
    payload = {"provenance": {},
               "costs": {names[0]: {MEASURED[0]: {"predicted_dloss": 0.1}},
                         names[1]: {fmt: {"predicted_dloss": 0.1}
                                    for fmt in MEASURED}}}
    merged = Path(campaign["plan"].parent / "weight-only-cost.pkl")
    merged.write_bytes(pickle.dumps(payload))
    plan = json.loads(campaign["plan"].read_text())
    plan["inputs"]["merged_cost"] = _bind(merged)
    campaign["plan"].write_text(json.dumps(plan))
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    assert submit_aqua(campaign, "--dry-run") == 0
    record, cells = dispatch.aqua_requested_cells(
        json.loads(campaign["plan"].read_text()), payload, list(MEASURED))
    assert record["requested_cells"] == len(MEASURED) + 1
    assert cells == frozenset({(names[0], MEASURED[0]),
                               (names[1], MEASURED[0]),
                               (names[1], MEASURED[1])})


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
    message = str(refused.value)
    assert "is not the menu the plan prices" in message
    # The format it left out is exactly the one whose A-side would go unchecked.
    assert MEASURED[1] in message


def test_a_joint_row_donated_from_another_unit_refuses(campaign, joint_row_is_joint):
    """A valid row under the wrong key is not this cell's A-side.

    Every internal check passes: the row is a joint row and its operator
    identity names a real cell. It names a *different* cell, which is exactly
    what a copied, mis-merged or re-keyed row looks like, and the cell it
    landed on would otherwise be reported as covered while carrying no
    activation term of its own.
    """
    names = campaign["names"]
    payload = _cost_payload(names)
    payload["costs"][names[0]][MEASURED[0]] = _joint_row(
        names[0], MEASURED[0], produced_for=(names[1], MEASURED[0]))
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    message = str(refused.value)
    assert "produced for" in message
    assert names[1] in message


def test_a_joint_row_donated_from_another_rung_refuses(campaign, joint_row_is_joint):
    """The same, one rung over: the coordinate is the pair, not the unit."""
    names = campaign["names"]
    payload = _cost_payload(names)
    payload["costs"][names[0]][MEASURED[0]] = _joint_row(
        names[0], MEASURED[0], produced_for=(names[0], MEASURED[1]))
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    assert MEASURED[1] in str(refused.value)


def test_a_joint_cell_the_plan_never_priced_refuses_or_is_recorded_unverified(
        campaign, joint_row_is_joint):
    """Self-consistent is not the same as bound to *this* plan.

    The row names its own cell, so it cannot hide a hole -- but nothing in it
    was compared with the plan's draw, capture or candidate menu. An artifact
    from a pass over a wider roster would therefore present prices this
    campaign never priced beside the plan's own table. By default that
    refuses; the explicit flag reuses it and says so in the record.
    """
    import dispatch_tessera_campaign as dispatch

    names = campaign["names"]
    payload = _cost_payload(names)
    payload["costs"][names[0]]["FP8_UNPLANNED"] = _joint_row(
        names[0], "FP8_UNPLANNED")
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run")
    message = str(refused.value)
    assert "never priced" in message and "FP8_UNPLANNED" in message

    record, cells = dispatch.aqua_requested_cells(
        json.loads(campaign["plan"].read_text()), payload, list(MEASURED),
        accept_joint_cells_outside_plan=True)
    assert record["joint_cells_outside_plan"] == 1
    assert record["joint_cells_outside_plan_accepted_unverified"] is True
    # The unplanned cell is not part of the requested roster either way.
    assert not any(fmt == "FP8_UNPLANNED" for _, fmt in cells)
    # And the flag is what makes the submission reach a stage at all.
    assert submit_aqua(campaign, "--dry-run",
                       "--accept-joint-cells-outside-plan") == 0


def test_an_all_joint_payload_is_not_submitted(campaign, capsys, joint_row_is_joint):
    """A fulfilled artifact is the answer, not a GPU window that adds nothing."""
    names = campaign["names"]
    campaign["cost_in"].write_bytes(pickle.dumps({
        "provenance": {},
        "costs": {name: {fmt: _joint_row(name, fmt) for fmt in MEASURED}
                  for name in names}}))
    assert submit_aqua(campaign, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "[dry-run]" not in printed
    summary = json.loads(printed)
    assert summary["submitted"] is False
    assert "joint A-side" in summary["reason"]
    # The record, not a flattened subset: a caller reading only the counts
    # cannot tell an artifact the plan priced from one carrying more.
    assert summary["requested_roster"]["joint_cells_outside_plan"] == 0
    assert summary["requested_roster"]["joint_cells_outside_plan_accepted_unverified"] is False


def test_the_shortcut_cannot_report_a_satisfied_artifact_that_reused_extras(
        campaign, capsys, joint_row_is_joint):
    """Every requested cell joint AND an unplanned joint cell, explicitly kept.

    This is the case the shortcut must not launder: the run is genuinely
    fulfilled *and* the artifact holds a joint-priced cell the plan never
    priced, reused only because the caller asked for that. The skip summary has
    to say both, because it is the only record the operator sees -- no stage is
    queued, so there is no receipt to read afterwards.
    """
    names = campaign["names"]
    costs = {name: {fmt: _joint_row(name, fmt) for fmt in MEASURED}
             for name in names}
    costs[names[0]]["FP8_UNPLANNED"] = _joint_row(names[0], "FP8_UNPLANNED")
    campaign["cost_in"].write_bytes(pickle.dumps({"provenance": {},
                                                  "costs": costs}))
    assert submit_aqua(campaign, "--dry-run",
                       "--accept-joint-cells-outside-plan") == 0
    printed = capsys.readouterr().out
    assert "[dry-run]" not in printed
    summary = json.loads(printed)
    assert summary["submitted"] is False
    assert summary["joint_cells_already_priced"] == len(MEASURED) * len(names)
    roster = summary["requested_roster"]
    assert roster["joint_cells_outside_plan"] == 1
    assert roster["joint_cells_outside_plan_accepted_unverified"] is True


def test_the_submitted_summary_carries_the_unverified_reuse(
        campaign, capsys, joint_row_is_joint):
    """The stage path states it too: a submission is not a quieter record.

    Here one requested cell is still weight-only, so a stage IS queued -- and
    the reuse decision has to be visible on that path as well, in the same
    record (which the submission also folds into the sealed manifest's
    ``campaign_scope`` annotation, so it enters the action key).
    """
    names = campaign["names"]
    payload = _cost_payload(names)
    payload["costs"][names[0]][MEASURED[0]] = _joint_row(names[0], MEASURED[0])
    payload["costs"][names[0]]["FP8_UNPLANNED"] = _joint_row(
        names[0], "FP8_UNPLANNED")
    campaign["cost_in"].write_bytes(pickle.dumps(payload))
    assert submit_aqua(campaign, "--dry-run",
                       "--accept-joint-cells-outside-plan") == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    command = next(line for line in lines if line.startswith("[dry-run] "))
    assert "--require-complete-coverage" in command
    # The command line comes first and is not JSON; the summary is everything
    # else, which is also what a caller piping this output reads.
    summary = json.loads("\n".join(line for line in lines
                                   if not line.startswith("[dry-run] ")))
    roster = summary["requested_roster"]
    assert roster["joint_cells_outside_plan"] == 1
    assert roster["joint_cells_outside_plan_accepted_unverified"] is True
    assert summary["campaign_scope"]["joint_cells_outside_plan"] == 1
    assert (summary["campaign_scope"]
            ["joint_cells_outside_plan_accepted_unverified"] is True)


def test_an_existing_cost_out_refuses(campaign):
    """A leftover path is not a previous result, and is never overwritten."""
    out = Path(campaign["plan"]).parent / "already-there.pkl"
    out.write_bytes(b"half a merge")
    with pytest.raises(RuntimeError) as refused:
        submit_aqua(campaign, "--dry-run", cost_out=out)
    assert "already exists" in str(refused.value)
