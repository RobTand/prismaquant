"""CPU regression for the old-checkout backfill incident (PQ #1761).

Only source-header accounting is synthetic. The phase arithmetic, campaign's
actual head-selection call and dispatcher's demand/refusal are production code.
No tensors, model files or GPU are allocated by this 864-unit geometry fixture.
"""
from __future__ import annotations

import json

import pytest

from prismaquant import autoscale
from prismaquant.tessera_row_stream import (
    MEMORY_PLANS, ROW_HEAD_LOAD_ALL, ROW_HEAD_STREAM,
    checkpoint_present,
)
from tools import dispatch_tessera_campaign as dispatch

GIB = 1024 ** 3
CAP_BYTES = 69 * GIB
BASELINE_BYTES = 954859520
LOAD_POLICY = {"schema": "prismaquant.verified_activation_load.v1",
               "max_buffer_bytes": 629147557, "max_scratch_bytes": 2097152}
REFERENCE_POLICY = {"schema": "tessera.hessian_reference_load.v1",
                    "max_file_bytes": 629147557, "max_hessian_bytes": 603979776,
                    "max_metadata_bytes": 134217728}


def _campaign_runtime_head(checkpoint, argv, monkeypatch):
    """Run the real CLI through head selection, stopping at source custody."""
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_calibration_cache as capture

    class BeforeSourceRead(BaseException):
        pass

    attempted = []
    def stop_at_source(*args, **kwargs):
        attempted.append((args, kwargs))
        raise BeforeSourceRead('head decision observed before source authentication')

    output = checkpoint.parent / 'actual-head.pkl'
    cache_dir = checkpoint.parent / 'actual-head-cache'
    missing_model = checkpoint.parent / 'never-opened-model'
    missing_selection = checkpoint.parent / 'never-opened-units.json'
    missing_census = checkpoint.parent / 'never-opened-census.json'
    missing_capture = checkpoint.parent / 'never-opened-capture.json'
    with monkeypatch.context() as scoped:
        scoped.setattr(capture, 'authenticate_selected_capture_source', stop_at_source)
        with pytest.raises(BeforeSourceRead):
            campaign.main([*argv, '--model', str(missing_model), '--out', str(output),
                '--cache-dir', str(cache_dir), '--checkpoint', str(checkpoint),
                '--units', str(missing_selection), '--calibration-census', str(missing_census),
                '--calibration-cache', str(missing_capture), '--calibration-cache-sha256', 'a'*64,
                '--attention-implementation', 'eager'])
    assert len(attempted) == 1
    assert attempted[0][1]['model'] == str(missing_model)
    assert all(not path.exists() for path in (
        missing_model, missing_selection, missing_census, missing_capture))
    assert not output.exists()
    assert not (cache_dir / 'row-head-execution.json').exists()
    record = json.loads(output.with_suffix('.waits.json').read_text())
    assert record['passed'] is False
    assert record['failure'].startswith('BeforeSourceRead:')
    return record['row_head']


@pytest.fixture
def geometry(monkeypatch):
    shapes = {}
    for expert in range(288):
        for leaf, shape in (("gate", (2048, 4096)), ("up", (2048, 4096)),
                            ("down", (4096, 2048))):
            shapes[f"layers.0.experts.{expert}.{leaf}"] = list(shape)
    counts = dict.fromkeys(shapes, 512)
    weights = {name: rows * cols * 2 for name, (rows, cols) in shapes.items()}
    source = dict(live_layer_prefix="layers.",
        terms=dict(nonbody_source_bytes=0, declared_headroom_bytes=24 * GIB),
        body_layer_bytes={"0": sum(weights.values())},
        body_loader_transient_bytes={"0": 0},
        body_source_file_bytes={"0": sum(weights.values())},
        body_source_shards={"0": ["synthetic.safetensors"]},
        unit_source_weight_bytes=weights, source_tensor_keys=sorted(shapes),
        full_hessian_bytes=sum(cols ** 2 * 4 for _, cols in shapes.values()),
        full_prefix_bytes=sum(512 * cols * 4 for _, cols in shapes.values()),
        source_header_sha256="a" * 64)
    monkeypatch.setattr(autoscale, "streamed_calibration_resources", lambda *a, **k: source)
    argv = ["--streaming", "--streaming-cache-slots", "2",
            "--streaming-prefetch-workers", "1", "--streaming-cache-headroom-gb", "24",
            "--anchor-batch-size", "16", "--max-rounds", "1",
            "--max-act-rows", "512", "--source-snapshot-policy", "selected-tensors-v1",
            "--capture-load-policy", json.dumps(LOAD_POLICY),
            "--export-hessian-reference-policy", json.dumps(REFERENCE_POLICY)]
    spec = dict(model="/synthetic", campaign_argv=argv, cpus=6, headroom_gb=24,
                process_baseline_bytes=BASELINE_BYTES)
    census = dict(unit_shapes=shapes, counts=counts,
                  anchor_groups={"s:stack": sorted(shapes)})
    return spec, census


@pytest.mark.parametrize("host_free_gib", [96, 120])
@pytest.mark.parametrize("checkpoint_kind", ["absent", "manifest", "unit-shards"])
def test_runtime_and_planner_keep_the_stream_plan_for_checkpoint_resume(
        tmp_path, monkeypatch, geometry, host_free_gib, checkpoint_kind):
    monkeypatch.setattr(autoscale, "_available_ram_bytes", lambda: host_free_gib * GIB)
    spec, census = geometry
    checkpoint = tmp_path / "cost.anchors.json"
    if checkpoint_kind == "manifest":
        checkpoint.write_text("{}")
    elif checkpoint_kind == "unit-shards":
        units = tmp_path / "cost.anchors.json.parts" / "units"
        units.mkdir(parents=True)
        (units / "unit.pkl").write_bytes(b"presence only")
    assert checkpoint_present(checkpoint) == (checkpoint_kind != "absent")
    members = sorted(census["unit_shapes"])
    plans = dispatch._streamed_resource_plan(spec, census, members, selected_source=True)
    stream_bytes = plans[MEMORY_PLANS[ROW_HEAD_STREAM]]
    load_all_bytes = plans[MEMORY_PLANS[ROW_HEAD_LOAD_ALL]]
    assert isinstance(stream_bytes, int) and isinstance(load_all_bytes, int)
    assert stream_bytes + BASELINE_BYTES < CAP_BYTES
    assert load_all_bytes + BASELINE_BYTES > CAP_BYTES
    head = _campaign_runtime_head(checkpoint, spec['campaign_argv'], monkeypatch)
    runtime_bytes = plans[MEMORY_PLANS[head]]
    assert runtime_bytes + BASELINE_BYTES <= CAP_BYTES, (
        "a checkpoint changed the runtime head after stream-sized admission: " + str(head))
    assert head == ROW_HEAD_STREAM
    assert dispatch._row_head_dependency(spec["campaign_argv"]) is None
    demand = dispatch._row_memory_demand(spec, members, census, selected_source=True)
    assert demand["plan_bytes"] == runtime_bytes
    assert demand["mem_gb"] <= 69


@pytest.mark.parametrize("extra,words", [
    (["--row-head", "load-all"], "--row-head"),
    (["--seed-checkpoint", "/external-seed"], "--seed-checkpoint"),
    (["--max-rounds", "2"], "--max-rounds"),
])
def test_genuine_load_all_dependencies_are_refused_before_submission(
        tmp_path, geometry, extra, words):
    spec, census = geometry
    argv = list(spec["campaign_argv"])
    if extra[0] in argv:
        argv[argv.index(extra[0]) + 1] = extra[1]
    else:
        argv.extend(extra)
    assert words in dispatch._row_head_dependency(argv)
    selection = tmp_path / "units.json"
    selection.write_text(json.dumps({"schema": "prismaquant.tessera_campaign_units.v1",
                                    "groups": [{"key": "s:stack", "members": sorted(census["counts"])}]}))
    row = dict(argv=["python", "-m", "prismaquant.tessera_campaign", *argv,
                     "--units", str(selection)], demand=dict(mem_gb=69))
    with pytest.raises(dispatch.DemandRefused, match="above.*GPU box declares"):
        dispatch.verify_row_demand(spec, census, row, box_memory_gb=104)
