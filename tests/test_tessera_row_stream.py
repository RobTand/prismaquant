"""The streaming row head (RobTand/prismaquant#640).

A selected-source row reads, verifies and receipts each capture entry on
reader threads while earlier batches encode, and writes the six
identity-bound outputs only at finalize. The CLI tests run the public campaign
on CPU over a real verified capture, a real shard reader and real Tessera
encodes; the window test drives the row stream directly.
"""
from __future__ import annotations

import gc
import importlib.metadata
import json
import shutil
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_tessera_hessian_reference_handoff import POLICY as REFERENCE_POLICY, handoff  # noqa: F401

UNITS = ["model.layers.0.a", "model.layers.0.b", "model.layers.1.c"]
COLUMNS = 256
ROWS = 4
FORMAT = "TESSERA_E4M3_K1_R1024"
LOAD_POLICY = dict(schema="prismaquant.verified_activation_load.v1",
                   max_buffer_bytes=1024 ** 2, max_scratch_bytes=1024 ** 2)
OUTPUTS = ("cost.pkl", "campaign.anchors.json", "campaign.anchors.json.parts", "cache")


def stream_fixture(monkeypatch, tmp_path):
    """Three dense units in two layers, a complete verified capture, one priced rung each."""
    from safetensors.torch import save_file
    from prismaquant import (autoscale, cost_streaming, layer_streaming, model_profiles,
                             tessera_render)
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant import tessera_campaign as campaign
    from prismaquant.model_profiles import DefaultProfile

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    generator = torch.Generator().manual_seed(640)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    linears = {}
    for name in UNITS:
        _, _, layer, leaf = name.split(".")
        linear = torch.nn.Linear(COLUMNS, 32, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(32, COLUMNS, generator=generator))
        setattr(model.model.layers[int(layer)], leaf, linear)
        linears[name] = linear
    model.lm_head = torch.nn.Linear(COLUMNS, 32, bias=False, dtype=torch.bfloat16)
    model.config = SimpleNamespace(_attn_implementation="eager")

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    shards = {
        "head.safetensors": {"lm_head.weight": model.lm_head.weight.detach().clone()},
        "layer0.safetensors": {name + ".weight": linears[name].weight.detach().clone()
                               for name in UNITS[:2]},
        "layer1.safetensors": {UNITS[2] + ".weight": linears[UNITS[2]].weight.detach().clone()},
    }
    weight_map = {}
    for filename, values in shards.items():
        save_file(values, str(source / filename))
        weight_map.update(dict.fromkeys(values, filename))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    tokens, text = [torch.ones(1, 256, dtype=torch.long)], "one draw"
    menu = [SimpleNamespace(format_name=FORMAT, family="TESSERA_E4M3_K1",
                            body_rate_q256=1024, bpp=4.0)]
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: DefaultProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {
        "accepted": True, "reason": "CPU test fixture", "kwargs": [], "recipe": {}})
    monkeypatch.setattr(campaign, "_calibration_tokens", lambda *_args: (tokens, text))
    monkeypatch.setattr(campaign, "_collect_activations",
                        lambda *_a, **_k: pytest.fail("repeated calibration forward"))
    monkeypatch.setattr(campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: {name: menu for name in targets})

    version = importlib.metadata.version("transformers")
    contract = dict(schema="prismaquant.streaming_initialization.v1",
        scope="streamed_text_source_forward", status="completed",
        transformers_version=version, model_class="SyntheticSource",
        dtype="torch.bfloat16", layers_prefix="model.layers.", num_layers=2,
        persistent_tensors=4, derived_buffers=0,
        state_sha256="a" * 64, source_map_sha256="b" * 64)
    calibration = campaign.th.calibration_identity(text, tokens, fit_tokens=ROWS,
        source="wikitext-2-raw-v1/train", split_role="calibration", model=str(source),
        seed=0, nsamples=32, seqlen=512, fit_tokens_min=ROWS)
    groups = campaign.resolve_anchor_groups(UNITS, profile=DefaultProfile(), expert_members={})
    census = campaign.calibration_census(
        dict.fromkeys(UNITS, ROWS), dict(zip(UNITS, (3.0, 5.0, 7.0))),
        args=SimpleNamespace(model=str(source), nsamples=32, seqlen=512, seed=0, layer_stride=1),
        groups=groups, dense_targets=UNITS, expert_targets=[],
        shapes={name: [32, COLUMNS] for name in UNITS}, identity=calibration,
        model_load_contract=contract, attention_implementation="eager",
        capture_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=version))
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    acts, hessians = {}, {}
    for name in UNITS:
        rows = torch.randn(ROWS, COLUMNS, generator=generator)
        acts[name], hessians[name] = rows, rows.T @ rows + torch.eye(COLUMNS)
    canonical = cc.capture_identity(census_path, calibration=calibration, max_act_rows=ROWS,
        model_load_contract=contract, attention_implementation="eager")
    complete = cc.publish_capture(tmp_path / "capture", census_path=census_path,
        identity=canonical, acts=acts, hessians=hessians, counts=census["counts"],
        maxima=census["max_abs"])

    snapshots, plans = [], []

    def build(*_args, **kwargs):
        # The public orchestration and the real shard reader; only model
        # construction is the synthetic fixture.
        authentication = kwargs.get("source_authentication")
        owned = {} if authentication is None else {"source_authentication": authentication}
        weight_files, keys = layer_streaming._build_weight_map(str(source), **owned)
        layer_streaming._materialize(model, ["lm_head."], weight_files, keys,
                                     torch.device("cpu"), torch.bfloat16, **owned)

        def snapshot(names, **options):
            snapshots.append((list(names), dict(options)))
            weights = {}
            for prefix in sorted({name.rsplit(".", 1)[0] + "." for name in names}):
                values = layer_streaming._read_layer_to_device(
                    prefix, weight_files, keys, torch.bfloat16, torch.device("cpu"), **owned)
                weights.update({name: values[name + ".weight"].clone()
                                for name in names if name.startswith(prefix)})
            return weights, dict(schema="prismaquant.selected_source_weights.v1",
                                 source_forward_count=0)

        return SimpleNamespace(model=model, snapshot_selected_weights=snapshot,
                               shutdown=lambda: None)

    def plan(*_args, **kwargs):
        plans.append(dict(kwargs))
        return dict(memory_bytes=1024 ** 3, stream_memory_bytes=1024 ** 3,
                    selected_source_weight_bytes=2 * 32 * COLUMNS * len(UNITS),
                    encoder_memo_capacity=1,
                    phases={"resident_anchors": {"factorization_scratch_bytes": 8 * COLUMNS ** 2}})

    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
    monkeypatch.setattr(autoscale, "selected_anchor_resources", plan)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(schema=campaign.UNITS_SCHEMA, groups=[
        dict(key=key, members=list(members)) for key, members in sorted(groups.items())])))
    argv = ["--model", str(source), "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"),
            "--checkpoint", str(tmp_path / "campaign.anchors.json"),
            "--hessian", "require", "--menu-mode", "research", "--max-rounds", "1",
            "--attention-implementation", "eager", "--streaming",
            "--streaming-cache-headroom-gb", "0", "--units", str(selection),
            "--calibration-census", str(census_path), "--calibration-cache", complete["path"],
            "--calibration-cache-sha256", complete["sha256"], "--nsamples", "32",
            "--seqlen", "512", "--layer-stride", "1", "--max-act-rows", str(ROWS),
            "--capture-load-policy", json.dumps(LOAD_POLICY),
            "--export-hessian-reference-policy", json.dumps(REFERENCE_POLICY)]
    state = dict(root=tmp_path, capture=Path(complete["path"]).parent,
                 manifest=Path(complete["path"]), census=census, canonical=canonical,
                 calibration=calibration, snapshots=snapshots, plans=plans)
    return campaign, argv, state


def produced(root):
    """Every file the row wrote, except the row head's own execution record."""
    from prismaquant.tessera_row_stream import EXECUTION_FILENAME
    files = {}
    for base in OUTPUTS:
        path = root / base
        if path.is_file():
            files[base] = path.read_bytes()
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.name != EXECUTION_FILENAME:
                    files[str(child.relative_to(root))] = child.read_bytes()
    return files


def test_the_stream_head_writes_the_load_all_heads_bytes(monkeypatch, tmp_path, capsys):
    campaign, argv, state = stream_fixture(monkeypatch, tmp_path)
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_row_stream import EXECUTION_FILENAME
    # Encode durations and the wall clock are the only run-varying values
    # these files carry; pin them so the comparison is of bytes.
    monkeypatch.setattr(campaign, "time", SimpleNamespace(
        time=lambda: 1789500000.0, monotonic=time.monotonic, perf_counter=time.perf_counter))
    prefetches = []
    original_prefetch = cc.prefetch_capture

    def prefetch(*args, **kwargs):
        prefetches.append(sorted(kwargs["names"]))
        return original_prefetch(*args, **kwargs)

    monkeypatch.setattr(cc, "prefetch_capture", prefetch)
    assert campaign.main(argv) == 0
    assert "[campaign] row head: stream (" in capsys.readouterr().out
    assert prefetches == []
    stream = produced(tmp_path)
    record = json.loads((tmp_path / "cache" / EXECUTION_FILENAME).read_text())
    assert record["row_head"] == "stream" and record["units"] == len(UNITS)
    assert record["entries_read"] == len(UNITS) and record["hash_only_entries"] == 0
    assert record["peak_resident_units"] <= record["window_units"] == 2
    aside = tmp_path / "stream-arm"
    aside.mkdir()
    for base in OUTPUTS:
        shutil.move(str(tmp_path / base), str(aside / base))

    assert campaign.main([*argv, "--row-head", "load-all"]) == 0
    assert ("[campaign] row head: load-all (--row-head load-all was requested)"
            in capsys.readouterr().out)
    assert prefetches == [sorted(UNITS)]
    load_all = produced(tmp_path)
    assert sorted(stream) == sorted(load_all)
    assert [name for name in sorted(stream) if stream[name] != load_all[name]] == []
    # The comparison covered every identity-bound write, not a subset.
    assert {"cost.pkl", "campaign.anchors.json", "cache/hessian_capture.references.json",
            "cache/input_scales.safetensors"} <= set(stream)
    assert sum(name.startswith("cache/capture-load-execution-") for name in stream) == 1
    assert sum(name.startswith("campaign.anchors.json.parts/units/") for name in stream) == 3
    assert sum(name.startswith("cache/wire/") for name in stream) == 3
    # Only the stream head asked for host weights: its readers hash them.
    assert [options.get("host") for _names, options in state["snapshots"]] == [True, None]


def test_the_first_encode_starts_before_the_late_entries_are_read(monkeypatch, tmp_path):
    import os
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_row_stream import EXECUTION_FILENAME
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    campaign, argv, state = stream_fixture(monkeypatch, tmp_path)
    encoding = threading.Event()
    waited, encodes = {}, []
    original_entry = cc._verified_capture_entry

    def delayed(path, name, **kwargs):
        if name != UNITS[0]:
            # A head that loads every entry before encoding waits here until
            # the timeout: nothing encodes before this returns.
            waited[name] = encoding.wait(timeout=30)
        return original_entry(path, name, **kwargs)

    original_measure = campaign._measure_anchor

    def measure(**kwargs):
        encodes.append(kwargs["qname"])
        encoding.set()
        return original_measure(**kwargs)

    monkeypatch.setattr(cc, "_verified_capture_entry", delayed)
    monkeypatch.setattr(campaign, "_measure_anchor", measure)
    assert campaign.main(argv) == 0
    assert waited == dict.fromkeys(UNITS[1:], True)
    assert encodes == UNITS
    # Reader threads default to the admitted CPUs, and the plan is charged
    # for that same count.
    record = json.loads((tmp_path / "cache" / EXECUTION_FILENAME).read_text())
    assert record["reader_threads"] == 3 and record["window_units"] == 2
    assert state["plans"][-1]["campaign_identity_threads"] == 3


def test_the_stream_head_reports_durable_progress_before_its_journal_exists(
        monkeypatch, tmp_path):
    """PQ #1362: the stream head opens its journal only at finalize, and a row
    that reported nothing until then was killed in ``startup`` while its
    anchors were landing. Each flush before the journal exists now reports
    the anchors whose wire receipts were read back, and never more than the
    wires on disk."""
    campaign, argv, state = stream_fixture(monkeypatch, tmp_path)
    from prismaquant import cost_stage_checkpoint, prismabuild_progress
    wire = state["root"] / "cache" / "wire"
    events = []
    original_write_unit = cost_stage_checkpoint.write_unit

    def write_unit(*args, **kwargs):
        events.append(("journal",))
        return original_write_unit(*args, **kwargs)

    def report(phase, units_completed, **_kwargs):
        landed = len(list(wire.glob("*.tessera"))) if wire.is_dir() else 0
        events.append(("report", phase, units_completed, landed))
        return True

    monkeypatch.setattr(cost_stage_checkpoint, "write_unit", write_unit)
    monkeypatch.setattr(prismabuild_progress, "report", report)
    assert campaign.main(argv) == 0
    first_journal = events.index(("journal",))
    early = [event for event in events[:first_journal] if event[0] == "report"]
    # Reports before the journal exists: all "pricing", each for work on
    # disk. One arrives mid-encode (the scalar cadence flushes at the first
    # anchor), and the round's drain reports every anchor (one per unit
    # here), all before finalize.
    assert {event[1] for event in early} == {"pricing"}
    assert all(0 < count <= landed for _, _, count, landed in early)
    assert early[0][2] < len(UNITS) and early[-1][2] == len(UNITS)
    counts = [event[2] for event in events if event[0] == "report"]
    assert counts == sorted(counts) and counts[-1] == len(UNITS)


def test_a_corrupt_late_entry_refuses_before_any_identity_bound_write(monkeypatch, tmp_path):
    campaign, argv, state = stream_fixture(monkeypatch, tmp_path)
    from prismaquant.perturbed_x_cache import activation_cache_filename
    artifact = state["capture"] / "inputs" / activation_cache_filename(UNITS[-1])
    data = bytearray(artifact.read_bytes())
    data[len(data) // 2] ^= 0xFF
    artifact.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="checksum"):
        campaign.main(argv)
    root, cache = state["root"], state["root"] / "cache"
    parts = root / "campaign.anchors.json.parts" / "units"
    # None of the six identity-bound writes exists.
    assert not (root / "campaign.anchors.json").exists()
    assert not parts.exists() or not any(parts.iterdir())
    assert not (cache / "hessian_capture.references.json").exists()
    assert not (cache / "input_scales.safetensors").exists()
    assert not list(cache.glob("capture-load-execution-*.json"))
    assert not (root / "cost.pkl").exists()
    # The batches before it encoded and published their wires, which cite no
    # run identity.
    assert sorted(path.name for path in (cache / "wire").glob("*.tessera")) == sorted(
        campaign._wire_path(cache / "wire", name, FORMAT).name for name in UNITS[:-1])


def test_a_present_checkpoint_runs_load_all_and_names_it(monkeypatch, tmp_path, capsys):
    campaign, argv, state = stream_fixture(monkeypatch, tmp_path)
    units = state["root"] / "campaign.anchors.json.parts" / "units"
    units.mkdir(parents=True)
    (units / "stale.pkl").write_bytes(b"left by an earlier attempt")
    with pytest.raises(RuntimeError, match="without a manifest"):
        campaign.main(argv)
    assert "[campaign] row head: load-all (a checkpoint exists" in capsys.readouterr().out


def test_the_window_holds_two_batches_and_releases_x_and_h(monkeypatch, tmp_path):
    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from prismaquant.tessera_row_stream import RowStream
    manifest, census = state["manifest"], state["census"]
    weights = {name: torch.ones(32, COLUMNS, dtype=torch.bfloat16) for name in UNITS}
    stream = RowStream(capture_path=manifest, expected_sha256=cc.sha256(manifest),
        expected_identity=state["canonical"], census=census, names=UNITS, policy=LOAD_POLICY,
        weights=weights, hessian_identity=state["calibration"],
        bind=lambda name, **_tensors: (None, dict(weight=name)), threads=2, batch_size=1,
        device="cpu", memo_capacity=1)
    family, rung = parse_tessera_format_name(FORMAT)
    plane = tessera_wire_recipe(family, rung).scale_plane
    stream.plan([[name] for name in UNITS])
    stream.admit(0)
    first = stream.entry(UNITS[0])
    released = [weakref.ref(first.inputs), weakref.ref(first.hessian)]
    stream.encoder_kwargs(UNITS[0], plane)
    assert list(stream._memo) == [(UNITS[0], plane)]
    del first
    stream.admit(1)
    stream.admit(2)
    gc.collect()
    assert [ref() for ref in released] == [None, None]
    assert UNITS[0] not in stream._live and not stream._memo
    assert stream.stats["peak_resident_units"] <= 2
    stream.finish()
    # Folded in name order, the per-entry receipts are the load-all prefetch's
    # execution record.
    execution = {}
    cc.prefetch_capture(manifest, expected_identity=state["canonical"], census=census,
                        names=UNITS, device="cpu", expected_sha256=cc.sha256(manifest),
                        verified_load_policy=LOAD_POLICY, load_execution=execution)
    assert stream.load_execution() == execution
    assert stream.observed_counts() == census["counts"]
    assert stream.observed_max_abs() == census["max_abs"]


def test_each_load_all_dependency_is_named(tmp_path):
    from prismaquant.tessera_row_stream import checkpoint_present, stream_head_dependency
    eligible = dict(row_head="stream", selected_source=True, capture_load_policy=LOAD_POLICY,
                    export_hessian_reference_policy=REFERENCE_POLICY, max_rounds=1,
                    seed_checkpoint=None, checkpoint_exists=False)
    assert stream_head_dependency(**eligible) is None
    for change, words in (
            (dict(row_head="load-all"), "--row-head load-all"),
            (dict(selected_source=False), "hash-bound selected capture"),
            (dict(capture_load_policy=None), "--capture-load-policy"),
            (dict(export_hessian_reference_policy=None), "hessian_capture.pt"),
            (dict(max_rounds=0), "--max-rounds"),
            (dict(seed_checkpoint="/seed"), "--seed-checkpoint"),
            (dict(checkpoint_exists=True), "a checkpoint exists")):
        assert words in stream_head_dependency(**{**eligible, **change})
    with pytest.raises(ValueError, match="unknown row head"):
        stream_head_dependency(**{**eligible, "row_head": "sideways"})
    checkpoint = tmp_path / "cost.anchors.json"
    units = tmp_path / "cost.anchors.json.parts" / "units"
    assert not checkpoint_present(checkpoint)
    units.mkdir(parents=True)
    assert not checkpoint_present(checkpoint)
    (units / "u.pkl").write_bytes(b"x")
    assert checkpoint_present(checkpoint)


def test_reader_threads_resolve_from_the_admitted_cpus(monkeypatch):
    import os
    from prismaquant.tessera_row_stream import resolve_identity_threads
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {4, 5, 6, 7})
    assert resolve_identity_threads(None) == 4
    assert resolve_identity_threads(9) == 9
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError):
            resolve_identity_threads(bad)


def test_a_reference_stand_in_without_a_receipt_refuses(handoff):  # noqa: F811
    from prismaquant import tessera_calibration_cache as cc
    f = handoff
    with pytest.raises(RuntimeError, match="neither a sealed receipt"):
        cc.canonical_hessian_reference_descriptor(
            hessians={"a": torch.empty(2, 2, device="meta")}, counts=f["census"]["counts"],
            provenance={**f["calibration"], "hessian_role": "fit"},
            canonical_capture=f["record"], census_path=f["census_path"],
            load_policy=REFERENCE_POLICY)


def test_the_stream_plan_charges_a_window_not_the_population(monkeypatch):
    from prismaquant import autoscale
    shapes = {f"layers.0.{leaf}": [3, 4] for leaf in "abc"}
    source = dict(live_layer_prefix="layers.",
        terms=dict(nonbody_source_bytes=100, declared_headroom_bytes=200),
        body_layer_bytes={"0": 1000}, body_loader_transient_bytes={"0": 100},
        body_source_file_bytes={"0": 900}, unit_source_weight_bytes=dict.fromkeys(shapes, 24),
        full_hessian_bytes=3 * 64, full_prefix_bytes=3 * 32, source_header_sha256="a" * 64)
    monkeypatch.setattr(autoscale, "streamed_calibration_resources", lambda *a, **k: source)
    kwargs = dict(unit_shapes=shapes, counts=dict.fromkeys(shapes, 9), max_act_rows=2,
                  cache_slots=2, prefetch_workers=1, headroom_gb=0, campaign_identity_threads=2)
    legacy = autoscale.selected_anchor_resources("/source", **kwargs)
    assert "stream_phases" not in legacy and "stream_memory_bytes" not in legacy
    plan = autoscale.selected_anchor_resources("/source", **kwargs, capture_load_policy=LOAD_POLICY)
    assert {name: plan["phases"][name] for name in legacy["phases"]} == legacy["phases"]
    window = plan["stream_phases"]["stream_window"]
    entry = 4 * (4 ** 2 + 2 * 4)
    assert window["window_capture_entry_bytes"] == 2 * entry
    assert window["reader_working_bytes"] == 2 * (
        2 * LOAD_POLICY["max_buffer_bytes"] + LOAD_POLICY["max_scratch_bytes"]
        + 2 * 4 ** 2 * 4 + 2 * 3 * 4 * 4)
    # No stream phase holds the population's H or X.
    assert not any(key in phase for phase in plan["stream_phases"].values()
                   for key in ("selected_hessian_bytes", "selected_prefix_bytes"))
    assert plan["stream_memory_bytes"] == max(sum(phase.values()) for phase in (
        plan["phases"]["source_preparation"], *plan["stream_phases"].values()))


def test_the_dispatcher_admits_a_stream_row_against_its_window(monkeypatch):
    from prismaquant import autoscale
    from tools import dispatch_tessera_campaign as dispatch
    seen = []

    def selected(_model, **kwargs):
        seen.append(kwargs)
        return {"memory_bytes": 10 * 1024 ** 3, "stream_memory_bytes": 1024 ** 3}

    monkeypatch.setattr(autoscale, "selected_anchor_resources", selected)
    monkeypatch.setattr(dispatch, "_process_baseline", lambda _spec: (0, "explicit"))
    monkeypatch.setattr(dispatch, "_guard_margin_bytes", lambda: 0)
    census = dict(unit_shapes={"a": [3, 4]}, counts={"a": 9})
    argv = ["--streaming", "--capture-load-policy", json.dumps(LOAD_POLICY),
            "--export-hessian-reference-policy", json.dumps(REFERENCE_POLICY),
            "--max-rounds", "1"]

    def demand(extra):
        return dispatch._row_memory_demand(dict(model="/source", cpus=6,
            campaign_argv=[*argv, *extra]), ["a"], census, selected_source=True)

    assert demand([])["plan_bytes"] == 1024 ** 3
    assert seen[-1]["campaign_identity_threads"] == 6
    assert demand(["--row-head", "load-all"])["plan_bytes"] == 10 * 1024 ** 3
    assert demand(["--seed-checkpoint", "/seed"])["plan_bytes"] == 10 * 1024 ** 3
    assert demand(["--campaign-identity-threads", "2"])["plan_bytes"] == 1024 ** 3
    assert seen[-1]["campaign_identity_threads"] == 2
