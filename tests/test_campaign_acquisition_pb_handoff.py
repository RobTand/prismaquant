"""Original authenticated joint request -> actual PB rows -> scalar wires -> strict merge."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch
from experiments import glm_data_manifests as manifests
from prismaquant import tessera_campaign as campaign, tessera_hessian as th
from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.model_profiles import DefaultProfile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.tessera_full_domain_acquisition import (
    joint_acquisition_from_cost_data, load_joint_campaign_acquisition,

)
from prismaquant.tessera_acquisition_inputs import joint_campaign_acquisition_control_inputs
from prismaquant.tessera_legal_domain import live_pins, tessera_source_state
from test_campaign_acquisition_scheduler import arguments
from test_streamed_cost_checkpoints import (
    _DenseTinyLM, _FakeStreamingContext, _model_identity,
)
from test_tessera_campaign_fanout import _plan_args, _shard

FAMILY = "TESSERA_E4M3_K1"
DEFERRED = "TESSERA_BF16_K1"
FORMATS = (FAMILY + "_R256", FAMILY + "_R2048")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def real_joint_run():
    import prismaquant.aura_cost as aura
    pytest.importorskip("tessera")
    torch.manual_seed(85)
    model = _DenseTinyLM(width=32, layers=3).eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _FakeStreamingContext(model)
    runner = StreamedCausalLM(context, DefaultProfile())
    names = sorted(name for name, _ in model.named_modules() if name.endswith(".proj"))
    source_weights = {name: model.get_submodule(name).weight.detach().clone() for name in names}
    captures, handles = {}, []
    for name in names:
        def capture(_module, inputs, _output, name=name):
            if name not in captures:
                captures[name] = inputs[0].detach().clone().reshape(-1, inputs[0].shape[-1])
        handles.append(model.get_submodule(name).register_forward_hook(capture))
    cache = ProductionWeightCache(weights={(name, fmt): weight + delta
        for name, weight in source_weights.items() for fmt, delta in zip(FORMATS, (0.25, 0.03125))},
        levers={}, activation_max_abs={name: 1.0 for name in names})
    tokens = torch.tensor([[1, 2, 3, 4]])
    try:
        payload = aura.compute_aura_cost_streamed(runner, tokens, [*FORMATS, "BF16"],
            n_probes=3, min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-acquisition-pb-handoff"))
    finally:
        for handle in handles:
            handle.remove()
    payload = pickle.loads(pickle.dumps(payload))
    shapes = {name: list(weight.shape) for name, weight in source_weights.items()}
    active = joint_acquisition_from_cost_data(payload, shapes, [FAMILY], max_new_points=1)
    deferred = joint_acquisition_from_cost_data(payload, shapes, [DEFERRED],
        max_new_points=0, boundary_policy="defer")
    idle_name = next((r["unit_name"] for r in active["reports"] if not r["proposed_q256"]),
                     names[-1])
    idle = joint_acquisition_from_cost_data(payload, {idle_name: shapes[idle_name]}, [FAMILY],
        max_new_points=0, boundary_policy="defer")["reports"][0]
    active["reports"] = [idle if r["unit_name"] == idle_name else r for r in active["reports"]]
    active["reports"].extend(deferred["reports"])
    document = dict(schema="prismaquant.tessera_full_domain_campaign_acquisition.v1",
        source_tensor_inventory_sha256=sha(json.dumps([
            {"name": name + ".weight", "shape": shape} for name, shape in sorted(shapes.items())]).encode()),
        journal_bindings={}, active_encoder_source_sha256=None,
        domain_pins=live_pins().as_dict(), producer_source_state=tessera_source_state(),
        atomic_serving_group_expansion_required=True, allocator_payload=False,
        production_qualified=False, **active)
    document["total_requested_quality_measurements"] = sum(len(r["proposed_q256"]) for r in document["reports"])
    active_names = sorted(r["unit_name"] for r in document["reports"]
        if r["proposed_q256"])
    assert len(active_names) >= 2, "the two-atomic-row scenario needs two proposing units"
    assert all(not r["proposed_q256"] for r in document["reports"] if r["unit_name"] == idle_name)
    return SimpleNamespace(payload=payload, document=document, names=names, shapes=shapes,
        weights=source_weights, inputs=captures, tokens=tokens, idle_name=idle_name,
        active_names=active_names)


@pytest.fixture
def handoff(tmp_path, monkeypatch, real_joint_run):
    return build_handoff(tmp_path, monkeypatch, real_joint_run)


def build_handoff(tmp_path, monkeypatch, real_joint_run):
    """Build genuine acquisition controls and safetensors inputs at the caller root."""
    save_file = pytest.importorskip("safetensors.torch").save_file
    monkeypatch.setattr(manifests, "SHARED_MOUNT", str(tmp_path))
    raw_cost = pickle.dumps(real_joint_run.payload)
    cost = tmp_path / "original-joint.pkl"
    cost.write_bytes(raw_cost)
    document = copy.deepcopy(real_joint_run.document)
    document.update(cost_path=str(cost), cost_sha256=sha(raw_cost))
    request = tmp_path / "original-request.json"
    raw_request = json.dumps(document, allow_nan=False).encode()
    request.write_bytes(raw_request)
    binding = {"path": str(request), "sha256": sha(raw_request)}
    acquisition = load_joint_campaign_acquisition(binding)
    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    save_file({name + ".weight": weight.contiguous() for name, weight in real_joint_run.weights.items()}, str(shard))
    (model / "config.json").write_text(json.dumps({"model_type": "fixture", "num_hidden_layers": 3}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        name + ".weight": shard.name for name in real_joint_run.names}}))
    groups = campaign.resolve_anchor_groups(real_joint_run.names, profile=DefaultProfile(), expert_members={})
    census = dict(model=str(model), anchor_groups=groups, layer_stride=1, unit_shapes=real_joint_run.shapes)
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    (workspace / "census.json").write_text(json.dumps(census))
    spec = tmp_path / "spec.json"
    common_argv = ["--menu-mode", "research", "--max-rounds", "1",
        "--acquisition-request", binding["path"], "--acquisition-request-sha256", binding["sha256"]]
    spec.write_text(json.dumps(dict(model=str(model), campaign_argv=common_argv,
        cwd=str(tmp_path), python="python3", env={}, headroom_gb=0, box_memory_gb=104)))
    return SimpleNamespace(root=tmp_path, workspace=workspace, spec=spec, census=census,
        binding=binding, acquisition=acquisition, request=request, cost=cost,
        raw_request=raw_request, raw_cost=raw_cost, run=real_joint_run, common_argv=common_argv)


def plan(handoff):
    assert dispatch.cmd_plan(_plan_args(handoff.spec, handoff.workspace)) == 0
    return json.loads((handoff.workspace / "plan.json").read_text()), json.loads(
        (handoff.workspace / "manifest.json").read_text())


def test_actual_plan_projects_two_atomic_rows_and_records_deferred_cohort(handoff):
    planned, actions = plan(handoff)
    assert len(planned["rows"]) == len(actions) == 2
    assert len(planned["acquisition"]["requested_groups"]) == 3
    assert len(planned["acquisition"]["deferred_groups"]) == 1
    assert planned["acquisition"]["deferred_groups"][0]["members"] == [handoff.run.idle_name]
    assert "deferred" in planned["acquisition"]["deferred_groups"][0]["reason"]
    assert {name for row in planned["rows"] for name in row["members"]} == set(handoff.run.active_names)
    for row, action in zip(planned["rows"], actions):
        argv = dispatch._inner_campaign_argv(action)
        assert dispatch._campaign_acquisition_argv(argv) == handoff.binding
        selection = campaign.load_unit_selection(row["units"])
        members = campaign.selection_priced_units(selection)[0]
        projected, keys, actual = campaign._campaign_acquisition_row_scope(handoff.acquisition,
            handoff.census["anchor_groups"], selected=members)
        assert actual == members
        assert keys == row["groups"]
        assert projected["identity"] == handoff.acquisition["identity"]
        assert set(projected["source_weights"]) == members
    before = {str(path): path.read_bytes() for path in handoff.workspace.rglob("*.json")}
    plan(handoff)
    assert {str(path): path.read_bytes() for path in handoff.workspace.rglob("*.json")} == before
    assert handoff.request.read_bytes() == handoff.raw_request
    assert handoff.cost.read_bytes() == handoff.raw_cost
    for report in json.loads(handoff.raw_request)["reports"]:
        assert report["prices"] is None
        assert report["all_legal_rates_retained"] is True
        assert report["production_qualified"] is False


def test_readset_uses_whole_original_inputs_before_actual_capture_reads(handoff):
    planned, actions = plan(handoff)
    capture_root = handoff.workspace / "captures"
    capture_root.mkdir()
    capture_entries = {}
    for index, name in enumerate(handoff.run.names):
        path = capture_root / f"unit-{index}.pt"
        torch.save({"inputs": handoff.run.inputs[name], "hessian": th.hessian_from_rows(handoff.run.inputs[name])}, path)
        capture_entries[name] = {"path": path.name}
    capture_manifest = capture_root / "manifest.json"
    capture_manifest.write_text(json.dumps({"entries": capture_entries}))
    reader = manifests.Campaign(str(handoff.workspace), plan={**planned,
        "calibration_cache": {"path": str(capture_manifest)}})
    inputs = joint_campaign_acquisition_control_inputs(handoff.binding)
    for row, action in zip(planned["rows"], actions):
        argv = dispatch._inner_campaign_argv(action)
        readset = manifests.build_manifest(reader, row["row_id"], {}, argv)
        assert set(readset) == set(manifests.MANIFEST_KEYS)
        assert readset["entries"][:2] == [{**item, "offset": 0} for item in inputs]
        assert readset["annotations"]["phases"][0] == {
            "name": "acquisition_inputs", "bytes": sum(x["bytes"] for x in inputs),
            "cumulative_bytes": sum(x["bytes"] for x in inputs)}
        assert readset["entries"][2]["path"].startswith(str(capture_root))
        assert readset["annotations"]["counts"]["acquisition_inputs"] == 2
        assert readset["annotations"]["counts"]["captures"] == 1
        assert readset["total_bytes"] == sum(item["bytes"] for item in readset["entries"])
        assert readset["annotations"]["phases"][-1]["cumulative_bytes"] == readset["total_bytes"]
        assert all(item["sha256"] is None for item in readset["entries"][2:])
    assert handoff.cost.read_bytes() == handoff.raw_cost


@pytest.mark.parametrize("change", ["request_sha", "cost_drift", "unknown_scope", "overlapping_scope", "partial_experts", "sample"])
def test_plan_refuses_before_capture_preparation_or_publication(handoff, monkeypatch, change):
    spec = json.loads(handoff.spec.read_text())
    overrides = {}
    if change == "request_sha":
        spec["campaign_argv"][-1] = "0" * 64
    elif change == "cost_drift":
        handoff.cost.write_bytes(handoff.raw_cost + b"drift")
    elif change == "unknown_scope":
        census = copy.deepcopy(handoff.census)
        census["anchor_groups"].pop(next(iter(census["anchor_groups"])))
        (handoff.workspace / "census.json").write_text(json.dumps(census))
    elif change == "overlapping_scope":
        census = copy.deepcopy(handoff.census)
        census["anchor_groups"]["u:overlap"] = [handoff.run.names[0]]
        (handoff.workspace / "census.json").write_text(json.dumps(census))
    elif change == "partial_experts":
        overrides["experts_per_row"] = 1
    else:
        overrides["stack_sample"] = 1
    handoff.spec.write_text(json.dumps(spec))
    monkeypatch.setattr(dispatch, "_calibration_cache_binding",
        lambda *_a, **_k: pytest.fail("acquisition error reached capture preparation"))
    with pytest.raises((ValueError, RuntimeError), match="identity|scope|sampled|partial"):
        dispatch.cmd_plan(_plan_args(handoff.spec, handoff.workspace, **overrides))
    assert not (handoff.workspace / "manifest.json").exists()
    assert not (handoff.workspace / "plan.json").exists()
    assert not (handoff.workspace / "units").exists()


@pytest.mark.parametrize("target", ["request", "cost"])
def test_readset_refuses_missing_and_drifted_original_inputs(handoff, target):
    planned, actions = plan(handoff)
    reader = manifests.Campaign(str(handoff.workspace), plan=planned)
    row = planned["rows"][0]
    argv = dispatch._inner_campaign_argv(actions[0])
    path = getattr(handoff, target)
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="identity mismatch"):
        manifests.build_manifest(reader, row["row_id"], {}, argv)
    path.unlink()
    with pytest.raises((OSError, ValueError)):
        manifests.build_manifest(reader, row["row_id"], {}, argv)


def test_readset_refuses_controls_outside_shared_mount(handoff, monkeypatch):
    monkeypatch.setattr(manifests, "SHARED_MOUNT", str(handoff.root / "other-mount"))
    with pytest.raises(SystemExit, match="outside the shared mount"):
        dispatch.cmd_plan(_plan_args(handoff.spec, handoff.workspace))
    assert not (handoff.workspace / "manifest.json").exists()


def records(handoff, *, menu_families=None):
    result = {}
    from prismaquant.tessera_menu import expand_tessera_menu
    for index, name in enumerate(handoff.run.active_names):
        projected, keys, _ = campaign._campaign_acquisition_row_scope(handoff.acquisition,
            handoff.census["anchor_groups"], selected=[name])
        menus = {name: expand_tessera_menu(tuple(handoff.run.shapes[name]), mode="research",
                                         families=menu_families)}
        schedule = campaign._requested_acquisition_schedule({key: handoff.census["anchor_groups"][key]
            for key in keys}, projected["requests"])
        unit = {"weight": campaign._checkpoint_identity_api().tensor_identity(handoff.run.weights[name]),
                "acquisition_source_weight": projected["source_weights"][name],
                "menu": sorted(rung.format_name for rung in menus[name])}
        result[f"row-{index:04d}"] = dict(origin=campaign._campaign_acquisition_origin(projected, schedule, menus),
            schedule=schedule, sources=projected["source_weights"], units={name: unit},
            cells={name: [f"{family}_R{q}" for family, qs in schedule[name].items() for q in qs]})
    return result


@pytest.mark.parametrize("change", ["request", "cost", "probe", "source", "missing_q", "extra_q", "partial_scope", "overlap", "missing_row", "missing_cells", "extra_cells", "deferred_dropped", "deferred_unknown", "deferred_unrequested_dropped", "menu_missing"])
def test_semantic_merge_refuses_wrong_global_identity_source_and_row_work(handoff, change):
    rows = records(handoff)
    first, second = list(rows)
    name = next(iter(rows[first]["schedule"]))
    if change in ("request", "cost", "probe"):
        field = {"request": "request_control_sha256", "cost": "cost_sha256", "probe": "probe_identity_sha256"}[change]
        rows[first]["origin"][field] = "0" * 64
    elif change == "source":
        rows[first]["sources"][name]["content_sha256"] = "0" * 64
    elif change == "missing_q":
        rows[first]["schedule"][name][FAMILY] = []
    elif change == "extra_q":
        rows[first]["schedule"][name][FAMILY].append(-1)
    elif change == "partial_scope":
        rows[first]["units"]["unrequested.member"] = None
    elif change == "overlap":
        rows[second] = copy.deepcopy(rows[first])
    elif change == "missing_row":
        rows.pop(second)
    elif change == "missing_cells":
        rows[first]["cells"][name] = []
    elif change == "extra_cells":
        rows[first]["cells"][name].append(FAMILY + "_R999999")
    elif change == "deferred_unknown":
        rows[first]["origin"]["deferred_domain"][name].append("UNREVIEWED_FAMILY")
        rows[first]["origin"]["deferred_domain"][name].sort()
    elif change == "deferred_unrequested_dropped":
        extra = next(family for family in rows[first]["origin"]["deferred_domain"][name]
                     if family not in rows[first]["schedule"][name])
        rows[first]["origin"]["deferred_domain"][name].remove(extra)
    elif change == "menu_missing":
        rows[first]["units"][name].pop("menu")
    else:
        rows[first]["origin"]["deferred_domain"][name].remove(DEFERRED)
    with pytest.raises(dispatch.MergeRefused):
        dispatch._merge_acquisition_rows(rows, acquisition=handoff.acquisition,
            scope_groups=handoff.census["anchor_groups"])


def test_exact_deferred_domain_keeps_real_unrequested_menu_families(handoff):
    rows = records(handoff)
    merged = dispatch._merge_acquisition_rows(rows, acquisition=handoff.acquisition,
        scope_groups=handoff.census["anchor_groups"])
    for record in rows.values():
        for name, deferred in record["origin"]["deferred_domain"].items():
            assert DEFERRED in deferred
            assert any(family not in record["schedule"][name] for family in deferred)
            assert merged["origin"]["deferred_domain"][name] == deferred


def test_known_unrequested_family_outside_actual_restricted_menu_refuses(handoff):
    from prismaquant.tessera_formats import get_tessera_family

    full = records(handoff)
    rows = records(handoff, menu_families=[get_tessera_family(FAMILY)])
    first = next(iter(rows))
    name = next(iter(rows[first]["schedule"]))
    actual = dispatch._merge_acquisition_rows(rows, acquisition=handoff.acquisition,
        scope_groups=handoff.census["anchor_groups"])
    assert actual["origin"]["deferred_domain"][name] == [DEFERRED]
    excluded = next(family for family in full[first]["origin"]["deferred_domain"][name]
                    if family not in rows[first]["schedule"][name])
    rows[first]["origin"]["deferred_domain"][name].append(excluded)
    rows[first]["origin"]["deferred_domain"][name].sort()
    with pytest.raises(dispatch.MergeRefused, match="exact deferred families"):
        dispatch._merge_acquisition_rows(rows, acquisition=handoff.acquisition,
            scope_groups=handoff.census["anchor_groups"])


def test_runtime_refuses_unknown_partial_actual_groups_and_changed_source(handoff):
    names = handoff.run.names
    grouped = {"g:actual-complete-cohort": list(handoff.run.active_names),
               "u:" + handoff.run.idle_name: [handoff.run.idle_name]}
    with pytest.raises(ValueError, match="complete.*atomic"):
        campaign._campaign_acquisition_row_scope(handoff.acquisition, grouped,
            selected=[handoff.run.active_names[0]])
    with pytest.raises(ValueError, match="complete.*atomic"):
        campaign._campaign_acquisition_row_scope(handoff.acquisition, grouped, selected=["unknown"])
    with pytest.raises(ValueError, match="no requested measurement work|zero"):
        campaign._campaign_acquisition_row_scope(handoff.acquisition, handoff.census["anchor_groups"],
            selected=[handoff.run.idle_name])
    name = names[0]
    with pytest.raises(ValueError, match="source weight identity"):
        campaign._require_campaign_acquisition_source(name, handoff.run.weights[name] + 0.01,
            handoff.acquisition["source_weights"][name])


def test_manifest_refuses_request_fork_and_overlapping_rows(handoff):
    planned, actions = plan(handoff)
    fork = copy.deepcopy(actions)
    argv = fork[0]["argv"]
    argv[argv.index("--acquisition-request") + 1] = str(handoff.root / "fork.json")
    with pytest.raises(dispatch.DemandRefused, match="binding"):
        dispatch._check_acquisition_manifest(fork, planned, handoff.census, handoff.acquisition)
    with pytest.raises(dispatch.DemandRefused, match="overlap"):
        dispatch._check_acquisition_manifest([actions[0], actions[0]], planned,
            handoff.census, handoff.acquisition)
    for incomplete in ([], actions[:1], actions[1:]):
        with pytest.raises(dispatch.DemandRefused, match="active atomic coverage"):
            dispatch._check_acquisition_manifest(incomplete, planned,
                handoff.census, handoff.acquisition)


def test_no_acquisition_keeps_legacy_settings_and_manifest_shape():
    assert dispatch._campaign_acquisition_argv(["--max-rounds", "legacy-unparsed-value"]) is None
    assert dispatch._merge_acquisition_settings({"row": {"settings": {"legacy": True}, "units": {}}},
        {"row": {}}, acquisition=None, scope_groups=None) is None


def render_handoff(handoff, *, aliased=False):
    """Actual admission, row loader, renderer, wire and journal artifacts."""
    pytest.importorskip("tessera.export")
    from prismaquant.tessera_menu import expand_tessera_menu
    planned, actions = plan(handoff)
    if aliased:
        alias_cost = handoff.root / "same-cost-alias.pkl"
        alias_cost.write_bytes(handoff.raw_cost)
        document = json.loads(handoff.raw_request)
        document["cost_path"] = str(alias_cost)
        raw = json.dumps(document, allow_nan=False).encode()
        alias_request = handoff.root / "request-cost-alias.json"
        alias_request.write_bytes(raw)
        argv = actions[-1]["argv"]
        argv[argv.index("--acquisition-request") + 1] = str(alias_request)
        argv[argv.index("--acquisition-request-sha256") + 1] = sha(raw)
        dispatch._check_acquisition_manifest(actions, planned, handoff.census, handoff.acquisition)
    hessians = {name: th.hessian_from_rows(rows) for name, rows in handoff.run.inputs.items()}
    calibration = th.calibration_identity("real joint PB handoff", [handoff.run.tokens], fit_tokens=4)
    source = th.activation_source(hessians, calibration)
    counts = {name: int(rows.shape[0]) for name, rows in handoff.run.inputs.items()}
    handoff.census["counts"] = counts
    static_scales, static_policy = campaign._static_input_scales(
        {name: float(rows.abs().max()) for name, rows in handoff.run.inputs.items()}, profile=DefaultProfile())
    scope = dict(dense_targets=handoff.run.names, expert_targets=[], dense_all=handoff.run.names,
        pinned=[], declared_stacks={}, packed_in_scope={}, packed_outside_layer_stride={},
        anchor_groups=handoff.census["anchor_groups"], calibration_census=None)
    payloads, directories, schedules = {}, {}, {}
    row_identities, row_states = {}, {}
    for row, action in zip(planned["rows"], actions):
        members = campaign.selection_priced_units(campaign.load_unit_selection(row["units"]))[0]
        binding = dispatch._campaign_acquisition_argv(dispatch._inner_campaign_argv(action))
        loaded = load_joint_campaign_acquisition(binding, units=sorted(members))
        projected, keys, _ = campaign._campaign_acquisition_row_scope(loaded,
            handoff.census["anchor_groups"], selected=members)
        menus = {name: expand_tessera_menu(tuple(handoff.run.shapes[name]), mode="research") for name in members}
        weights = {name: handoff.run.weights[name] for name in members}
        rates = {name: {} for name in members}
        for name in members:
            for rung in menus[name]:
                rates[name].setdefault(rung.family, set()).add(rung.body_rate_q256)
        groups = {key: handoff.census["anchor_groups"][key] for key in keys}
        grids, _ = campaign.anchor_group_rate_grids(groups, rates, encode_structure=None, projected_units={})
        args = arguments(acquisition_request=binding["path"],
            acquisition_request_sha256=binding["sha256"])
        scheduled = campaign._campaign_round_one_schedule(groups, grids, args=args,
            audit_units=set(), snap=None, requests=projected["requests"], rates_by_unit=rates)
        schedules.update(scheduled)
        args.acquisition_schedule = scheduled
        args.acquisition_origin = campaign._campaign_acquisition_origin(projected, scheduled, menus)
        args.acquisition_source_weights = projected["source_weights"]
        row_dir = Path(row["dir"])
        wire_dir = row_dir / "cache" / "wire"
        wire_dir.mkdir(parents=True)
        primed = []
        checkpoint = row_dir / "cost.anchors.json"
        campaign._prime_first_anchor_batch(SimpleNamespace(prime=primed.append), args=args,
            checkpoint=checkpoint, targets=sorted(members), menus=menus, weights=weights,
            profile=DefaultProfile(), expert_members={}, encode_structure=None, projected_units={},
            audit_units=set(), route_cache={}, acquisition_requests=projected["requests"])
        assert set(primed[0]) <= members
        cache = SimpleNamespace(weights={}, cache_dir=None)
        memo = campaign._activation_kwargs_memo(source, weights, "cpu")
        measured, wire_records, keyword_names = {}, {}, set()
        for name in sorted(members):
            campaign._require_campaign_acquisition_source(name, weights[name], projected["source_weights"][name])
            for family, qs in scheduled[name].items():
                for q in qs:
                    anchor = campaign._measure_anchor(qname=name, weight=weights[name],
                        activations=handoff.run.inputs[name], format_name=f"{family}_R{q}", cache=cache,
                        wire_dir=wire_dir, hessian_required=True,
                        activation_kwargs_for=memo, static_input_scale=static_scales.get(name))
                    prepared = campaign._prepare_anchor(qname=name, format_name=anchor.format_name,
                        activation_kwargs_for=memo, hessian_required=True,
                        static_input_scale=static_scales.get(name))
                    keyword_names.update(prepared["activation_kwargs"] or {})
                    campaign._require_campaign_acquisition_anchor(anchor, scheduled)
                    identity = campaign._checkpoint_anchor_identity(anchor, weights=weights,
                        menus=menus, calibration_source=source, static_scales=static_scales, projected_units={})
                    wire_records.setdefault(name, {})[anchor.format_name] = campaign._checkpoint_wire_record(anchor, wire_dir, identity)
                    measured.setdefault(name, {}).setdefault(family, []).append(anchor)
        identity = campaign._campaign_checkpoint_identity(weights=weights,
            acts={name: handoff.run.inputs[name] for name in members},
            hessians={name: hessians[name] for name in members}, menus=menus, args=args,
            calibration_identity=calibration, serving_scope=None, static_scales=static_scales,
            static_scale_policy=static_policy)
        journal, identity_sha, _ = prepare_journal(checkpoint.with_name(checkpoint.name + ".parts"),
            stage="Tessera campaign", resume=False, identity=identity, qnames=sorted(members), manifest_path=checkpoint)
        row_identities[row["row_id"]] = identity
        row_states[row["row_id"]] = {}
        for name in sorted(members):
            state = {"anchors": [vars(anchor) for anchors in measured[name].values() for anchor in anchors],
                     "wire_records": wire_records[name]}
            write_unit(journal, stage="Tessera campaign", qname=name, identity_sha256=identity_sha,
                       state=state)
            row_states[row["row_id"]][name] = state
        name = next(iter(members))
        capture_path, scale_path, capture_digest = campaign.write_export_inputs(row_dir / "cache",
            hessians={name: hessians[name] for name in members}, hessian_rows=counts,
            hessian_identity=calibration, static_scales=static_scales, static_scale_policy=static_policy)
        prov = _shard(keys[0], name)["provenance"]
        prov.update(model=str(handoff.root / "model"), layer_stride=1, nsamples=1, seqlen=4, max_act_rows=4,
            max_rounds=1, rounds_run=1, max_artifact_bpp=0, campaign_scope=scope,
            wall_seconds=sum(a.seconds for fs in measured.values() for aa in fs.values() for a in aa),
            surfaces={name: {family: {"anchors": len(anchors)} for family, anchors in fs.items()}
                      for name, fs in measured.items()}, acquisition=args.acquisition_origin,
            acquisition_schedule=scheduled, acquisition_source_weights=projected["source_weights"],
            anchor_groups=groups, unit_selection={**campaign.load_unit_selection(row["units"]), "selected": True},
            activation_static_scales={"policy": static_policy, "path": str(scale_path), "units": static_scales},
            hessian={**calibration, "calibration_identity": calibration, "supplied": True,
                "capture_path": str(capture_path), "capture_sha256": capture_digest,
                "kwargs": sorted(keyword_names)})
        requested_menu = {name: [r for r in menus[name] if r.body_rate_q256 in scheduled[name].get(r.family, [])]
                          for name in members}
        payloads[row["row_id"]] = campaign.campaign_cost_payload(measured, requested_menu,
            loo={}, provenance={"provenance": prov})
        # The row runner publishes these after campaign_cost_payload; the merge
        # reads them from every row payload.
        payloads[row["row_id"]]["menu_sizes"] = {n: len(m) for n, m in menus.items()}
        payloads[row["row_id"]]["anchor_counts"] = {
            n: {f: len(a) for f, a in by_f.items()} for n, by_f in measured.items()}
        directories[row["row_id"]] = str(row_dir)
    coverage = dispatch.declared_coverage(planned, acquisition=handoff.acquisition, census=handoff.census)
    _, _, merged_capture_digest = dispatch.merge_export_inputs(directories, payloads,
        out_cache=handoff.root / "merged-cache", identity=calibration, policy=static_policy,
        static_scales=static_scales, census=handoff.census)
    return SimpleNamespace(planned=planned, payloads=payloads, directories=directories,
        schedules=schedules, row_identities=row_identities, row_states=row_states,
        coverage=coverage, capture_digest=merged_capture_digest,
        unit_identities={name: unit for identity in row_identities.values()
                         for name, unit in identity["units"].items()})


def test_real_cpu_requested_renderer_journals_and_merges_two_rows(handoff):
    actual = render_handoff(handoff)
    directories, payloads, schedules = actual.directories, actual.payloads, actual.schedules
    row_identities, row_states = actual.row_identities, actual.row_states
    coverage, merged_capture_digest = actual.coverage, actual.capture_digest
    checkpoint = dispatch.merge_checkpoint(directories, handoff.root / "merged.anchors.json",
        acquisition=handoff.acquisition, scope_groups=handoff.census["anchor_groups"])
    unit_identities = checkpoint["identity"]["units"]
    merged = dispatch.merge_payloads(payloads, census=handoff.census, capture_sha256=merged_capture_digest,
        plan_coverage=coverage, acquisition=handoff.acquisition,
        acquisition_unit_identities=unit_identities)
    assert merged["provenance"]["acquisition_schedule"] == schedules
    assert checkpoint["identity"]["settings"]["acquisition_schedule"] == schedules
    assert set(merged["costs"]) == set(handoff.run.active_names)
    assert set(checkpoint["identity"]["units"]) == set(handoff.run.active_names)
    for key, value in handoff.acquisition["identity"].items():
        assert merged["provenance"]["acquisition"][key] == value
        assert checkpoint["identity"]["settings"]["acquisition_origin"][key] == value
    assert merged["provenance"]["coverage"]["unpriced_groups"] == ["u:" + handoff.run.idle_name]
    assert handoff.request.read_bytes() == handoff.raw_request
    assert handoff.cost.read_bytes() == handoff.raw_cost
    for name in handoff.run.names:
        for fmt in FORMATS:
            original = handoff.run.payload["costs"][name][fmt]
            restored = pickle.loads(handoff.cost.read_bytes())["costs"][name][fmt]
            assert restored == original
            assert original["signed_components_per_probe"]
    with pytest.raises(dispatch.MergeRefused, match="bound checkpoint unit identities"):
        dispatch.merge_payloads(payloads, census=handoff.census, capture_sha256=merged_capture_digest,
            plan_coverage=coverage, acquisition=handoff.acquisition)
    first = next(iter(payloads))
    name = next(iter(payloads[first]["provenance"]["acquisition_schedule"]))
    for change in ("unknown", "active", "unrequested_dropped"):
        def alter_domain(origin):
            domain = origin["deferred_domain"][name]
            if change == "unrequested_dropped":
                extra = next(family for family in domain
                             if family not in payloads[first]["provenance"]["acquisition_schedule"][name])
                domain.remove(extra)
            else:
                domain.append("UNREVIEWED_FAMILY" if change == "unknown" else FAMILY)
                domain.sort()

        changed_payloads = copy.deepcopy(payloads)
        alter_domain(changed_payloads[first]["provenance"]["acquisition"])
        with pytest.raises(dispatch.MergeRefused, match="exact deferred families"):
            dispatch.merge_payloads(changed_payloads, census=handoff.census,
                capture_sha256=merged_capture_digest, plan_coverage=coverage,
                acquisition=handoff.acquisition, acquisition_unit_identities=unit_identities)
        changed_identities = copy.deepcopy(row_identities)
        alter_domain(changed_identities[first]["settings"]["acquisition_origin"])
        with pytest.raises(dispatch.MergeRefused, match="exact deferred families"):
            dispatch._merge_acquisition_settings(changed_identities, row_states,
                acquisition=handoff.acquisition, scope_groups=handoff.census["anchor_groups"])


def test_aliased_controls_reach_real_rows_and_both_merges(handoff, monkeypatch, capsys):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    actual = render_handoff(handoff, aliased=True)
    assert len({identity["settings"]["acquisition_origin"]["request_sha256"]
                for identity in actual.row_identities.values()}) == 2
    failures = []
    try:
        checkpoint = dispatch.merge_checkpoint(actual.directories, handoff.root / "alias-merged.anchors.json",
            acquisition=handoff.acquisition, scope_groups=handoff.census["anchor_groups"])
    except dispatch.MergeRefused as exc:
        failures.append(f"checkpoint merge: {exc}")
    try:
        merged = dispatch.merge_payloads(actual.payloads, census=handoff.census,
            capture_sha256=actual.capture_digest, plan_coverage=actual.coverage,
            acquisition=handoff.acquisition, acquisition_unit_identities=actual.unit_identities)
    except dispatch.MergeRefused as exc:
        failures.append(f"scalar payload merge: {exc}")
    assert not failures, "real admitted alias artifacts refused: " + "; ".join(failures)
    assert checkpoint["identity"]["settings"]["acquisition_schedule"] == actual.schedules
    assert merged["provenance"]["acquisition_schedule"] == actual.schedules
    assert set(merged["costs"]) == set(handoff.run.active_names)
    assert "[DEV-MODE]" in capsys.readouterr().out
    assert handoff.request.read_bytes() == handoff.raw_request
    assert handoff.cost.read_bytes() == handoff.raw_cost
    first = next(iter(actual.payloads))
    name = next(iter(actual.row_identities[first]["units"]))
    for change in ("authenticated_content", "probe", "source", "scope"):
        payloads = copy.deepcopy(actual.payloads)
        identities = copy.deepcopy(actual.row_identities)
        origin = identities[first]["settings"]["acquisition_origin"]
        if change == "authenticated_content":
            document = json.loads(handoff.raw_request)
            document["reports"][0]["max_new_points"] += 1
            raw = json.dumps(document, allow_nan=False).encode()
            path = handoff.root / "changed-authenticated-controls.json"
            path.write_bytes(raw)
            changed = load_joint_campaign_acquisition({"path": str(path), "sha256": sha(raw)})
            origin.update(changed["identity"])
        elif change == "probe":
            origin["probe_identity_sha256"] = "0" * 64
        elif change == "source":
            identities[first]["units"][name]["acquisition_source_weight"]["content_sha256"] = "0" * 64
            payloads[first]["provenance"]["acquisition_source_weights"][name]["content_sha256"] = "0" * 64
        else:
            origin["deferred_domain"].pop(name)
        payloads[first]["provenance"]["acquisition"] = copy.deepcopy(origin)
        directories, units = {}, {}
        for row_id, identity in identities.items():
            directory = handoff.root / ("negative-" + change) / row_id
            path = directory / "cost.anchors.json"
            journal, identity_sha, _ = prepare_journal(path.with_name(path.name + ".parts"),
                stage="Tessera campaign", resume=False, identity=identity,
                qnames=sorted(identity["units"]), manifest_path=path)
            for unit, state in actual.row_states[row_id].items():
                write_unit(journal, stage="Tessera campaign", qname=unit,
                    identity_sha256=identity_sha, state=state)
            directories[row_id] = str(directory)
            units.update(identity["units"])
        with pytest.raises(dispatch.MergeRefused):
            dispatch.merge_checkpoint(directories, handoff.root / (change + ".anchors.json"),
                acquisition=handoff.acquisition, scope_groups=handoff.census["anchor_groups"])
        with pytest.raises(dispatch.MergeRefused):
            dispatch.merge_payloads(payloads, census=handoff.census,
                capture_sha256=actual.capture_digest, plan_coverage=actual.coverage,
                acquisition=handoff.acquisition, acquisition_unit_identities=units)
