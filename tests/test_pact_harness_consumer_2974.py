"""Bounded real-consumer contract for #2974; no model download or GPU claim.

The imported campaign entry functions remain the entry points. Optional
profile/config keyword arguments carry the existing owner contract through
setup, CLI admission and frontier admission, not through a test-only resolver.
With no override, setup detects the declared profile from source_root/config.json.
The source directory assertion deliberately diagnoses the missing deliverable
before import (rather than causing a pytest collection/import error).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.model_profiles import profile_from_config

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((Path(__file__).parent / "fixtures/pact_harness_consumer_2974.json").read_text())


def harness(name):
    path = ROOT / "experiments" / "pact_replay" / (name + ".py")
    assert path.is_file(), (
        "#2974 deliverable missing: import the W1-approved real harness at "
        f"experiments/pact_replay/{name}.py (not another adapter-only resolver)"
    )
    return importlib.import_module("experiments.pact_replay." + name)


def put(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    return path


def rows_for(case):
    rows = []
    for expert in ([None] if case["kind"] == "dense" else [0, 1]):
        for role in ("gate_proj", "up_proj", "down_proj"):
            parent = "model.layers.0.mlp"
            if expert is not None:
                parent += f".experts.{expert}"
            rows.append({
                "qname": parent + "." + role, "kind": case["kind"],
                "role": role, "expert": expert, "layer": 0,
                "shape": [8, 4] if role == "down_proj" else [4, 8],
                "pick": {"a8s": "SOURCE"}, "formats": {},
            })
    return rows


def fixture_inputs(tmp_path, key):
    case = copy.deepcopy(CASES[key])
    source = tmp_path / "source"
    put(source / "config.json", case["config"])
    rows = rows_for(case)
    put(source / "model.safetensors.index.json", {"weight_map": {
        row["qname"] + ".weight": "fixture.safetensors" for row in rows
    }})
    manifest = put(tmp_path / "manifest.json", {"rows": rows})
    reference_rows = copy.deepcopy(rows)
    for row in reference_rows:
        if row["kind"] == "routed":
            row["formats"]["EXL3"] = {"fixture_metadata_only": True}
    reference = put(tmp_path / "reference.json", {"rows": reference_rows})
    a4 = tmp_path / "a4"
    put(a4 / "config.json", {})
    put(a4 / "model.safetensors.index.json", {"weight_map": {}})
    cohort = case["cohort"]
    # GLM's frozen draw has 512 rows; Qwen's declared miniature draw has 2.
    # No source/render weight bytes are needed for metadata/input preparation.
    draw_rows = 512 if key == "glm" else cohort["sample_range"][1]
    width = cohort["raw_tokens_per_sequence"]
    draw_tensor = torch.arange(draw_rows * width, dtype=torch.int64).reshape(draw_rows, width) % 17
    draw = tmp_path / "draw.safetensors"
    save_file({"input_ids": draw_tensor}, draw)
    args = SimpleNamespace(
        pq_root=ROOT, tessera_src=ROOT, g3_source=ROOT, source_root=source,
        a8_root=tmp_path / "a8", t8r_root=tmp_path / "t8", a4_root=a4,
        exl3_root=tmp_path / "exl3", manifest=manifest, reference_manifest=reference,
        added_inventory=None, option_manifest=None, t4_inventory=None,
        units_file=None, capture_root=tmp_path / "capture", draw=draw,
        draw_sha256=hashlib.sha256(draw.read_bytes()).hexdigest(),
        input_contract=cohort["input_contract"],
        sample_range="%d:%d" % tuple(cohort["sample_range"]),
        prefix_ids=cohort["prefix_ids"], device="cpu", dry_run_cpu=True,
        layer_range="0:1", chunk_rows=2, cpu_rows=2, weight_source=["A8S"],
        boundaries=None, prepare_readset=None, require_staged_inputs=False,
        boundary_dir=tmp_path / "boundaries",
    )
    return case, args, draw_tensor, rows


@pytest.mark.parametrize("key", ["glm", "qwen"])
def test_actual_setup_resolves_cohort_roster_roles_and_dimensions(tmp_path, key):
    setup = harness("stage1a_energy").setup
    case, args, draw, expected_rows = fixture_inputs(tmp_path, key)
    renders, cohort, batches, rows = setup(args)
    for field, expected in case["cohort"].items():
        assert cohort[field] == expected
    assert len(batches) == cohort["sample_range"][1] - cohort["sample_range"][0]
    original = draw[slice(*cohort["sample_range"])]
    assert cohort["token_sha256"] == hashlib.sha256(original.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    for i, batch in enumerate(batches):
        assert batch.dtype == torch.int64
        assert batch.tolist() == [cohort["prefix_ids"] + original[i].tolist()]
    assert rows == sorted(expected_rows, key=lambda r: (r["layer"], r["qname"]))
    assert set(renders.rows) == {r["qname"] for r in rows}
    # The actual setup must validate manifest roles and shapes against the
    # profile, not merely echo self-consistent but scientifically wrong rows.
    bad = json.loads(args.manifest.read_text())
    bad["rows"][0]["role"] = "down_proj"
    put(args.manifest, bad)
    with pytest.raises(ValueError, match="(?i)(role|projection|roster)"):
        setup(args)


@pytest.mark.parametrize("corruption", ["shape", "layer", "duplicate", "reference"])
def test_setup_refuses_mismatched_scientific_inventory(tmp_path, corruption):
    setup = harness("stage1a_energy").setup
    _, args, _, _ = fixture_inputs(tmp_path, "qwen")
    doc = json.loads(args.manifest.read_text())
    if corruption == "shape":
        doc["rows"][0]["shape"] = [4, 9]
    elif corruption == "layer":
        doc["rows"][0]["layer"] = 12
    elif corruption == "duplicate":
        doc["rows"].append(copy.deepcopy(doc["rows"][0]))
    else:
        reference = json.loads(args.reference_manifest.read_text())
        reference["rows"].pop()
        put(args.reference_manifest, reference)
    put(args.manifest, doc)
    with pytest.raises(ValueError, match="(?i)(shape|dimension|geometry|layer|roster|duplicate|population)"):
        setup(args)


@pytest.mark.parametrize("corruption", ["digest", "prefix", "sample_range", "token_width"])
def test_setup_refuses_mismatched_scientific_inputs(tmp_path, corruption):
    setup = harness("stage1a_energy").setup
    _, args, _, _ = fixture_inputs(tmp_path, "qwen")
    if corruption == "digest":
        args.draw_sha256 = "0" * 64
    elif corruption == "prefix":
        args.prefix_ids = [1, 2]
    elif corruption == "sample_range":
        args.sample_range = "0:1"
    else:
        save_file({"input_ids": torch.ones((2, 5), dtype=torch.int64)}, args.draw)
        args.draw_sha256 = hashlib.sha256(args.draw.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="(?i)(digest|draw|prefix|cohort|sample|token|shape)"):
        setup(args)


@pytest.mark.parametrize("scope", ["unknown", "undeclared", "incomplete_qwen", "invalid_dimension"])
def test_actual_setup_refuses_missing_or_invalid_scope(tmp_path, scope):
    setup = harness("stage1a_energy").setup
    case, args, _, _ = fixture_inputs(tmp_path, "qwen")
    config = case["config"]
    if scope == "unknown":
        config = {"model_type": "not_a_pact_model", "architectures": []}
    elif scope == "undeclared":
        config = {"model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}
    elif scope == "incomplete_qwen":
        del config["global_original_tokens"]
    else:
        config["hidden_size"] = 0
    put(args.source_root / "config.json", config)
    with pytest.raises(ValueError, match="(?i)(profile|scope|cohort|positive|dimension|hidden)"):
        setup(args)


def cli_documents(case):
    lo, hi = map(int, case["band"].split(":"))
    roster = {"schema": "pact.stream_roster.v1", "band": case["band"], "streams": [{
        "stream_id": "null", "class": "routed", "kind": "W_T8",
        "weight_source": "A8S", "amplitude": 1, "null_replay": True,
    }]}
    receipt = {"layer_start": lo, "layer_stop": hi,
               "start_reference_owner": {"session": {"generation": "fixture-start"}},
               "session": {"generation": "fixture-stop"}}
    return receipt, json.dumps(receipt, sort_keys=True).encode(), roster


def cli_args(api, args, case, deadline=3500):
    paths = ["pq-root", "tessera-src", "g3-source", "source-root", "a8-root", "t8r-root",
             "a4-root", "exl3-root", "manifest", "reference-manifest", "capture-root", "draw",
             "output", "run-manifest-out", "stream-manifest", "energy-rows", "teacher-frontier",
             "teacher-content", "output-dir", "checkpoint-dir"]
    argv = []
    for name in paths:
        argv += ["--" + name, str(getattr(args, name.replace("-", "_"), args.source_root / name))]
    argv += ["--draw-sha256", args.draw_sha256, "--teacher-content-sha256", "0" * 64,
             "--band", case["band"], "--layer-range", case["band"],
             "--input-contract", case["cohort"]["input_contract"], "--sample-range", args.sample_range,
             "--device", "cpu", "--window-layers", "21", "--window-start", str(case["window"][0]),
             "--window-stop", str(case["window"][1]), "--weight-source", "A8S",
             "--deadline-seconds=" + str(deadline)]
    return api.cli_parser().parse_args(argv)


@pytest.mark.parametrize("key", ["glm", "qwen"])
def test_normal_cli_uses_profile_extent_and_approved_3500_deadline(tmp_path, key):
    api = harness("multi_stream_replay")
    case, args, _, _ = fixture_inputs(tmp_path, key)
    parsed = cli_args(api, args, case)
    receipt, raw, roster = cli_documents(case)
    run = api.resolve_cli_run(parsed, receipt, raw, roster)
    assert run["replay_stop"] == case["config"]["num_hidden_layers"]
    assert run["bands"] == case["bands"]
    for field, expected in case["cohort"].items():
        assert run["cohort"][field] == expected
    assert run["hidden_layout"] == {"hidden_streams": case["hidden_streams"], "hidden_size": 8}
    assert run["plan"] == case["plan"]
    assert (run["window_start"], run["window_stop"]) == tuple(case["window"])
    assert run["generations"]["receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    assert run["manifest_streams"][0]["stream_id"] == "null"
    assert json.dumps(run, sort_keys=True, allow_nan=False) == json.dumps(
        api.resolve_cli_run(parsed, receipt, raw, roster), sort_keys=True, allow_nan=False)
    roster["band"] = "0:1"
    with pytest.raises(ValueError, match="(?i)(band|roster)"):
        api.resolve_cli_run(parsed, receipt, raw, roster)


def test_explicit_profile_config_precedence_reaches_setup_and_cli(tmp_path):
    setup = harness("stage1a_energy").setup
    api = harness("multi_stream_replay")
    case, args, _, _ = fixture_inputs(tmp_path, "qwen")
    declared = dict(case["config"], num_hidden_layers=48, vocab_size=999, prefix_ids=[8, 9])
    profile = profile_from_config(declared)
    explicit = {"text_config": {"num_hidden_layers": 12, "vocab_size": 32}, "prefix_ids": [30, 31]}
    # Neither the profile's captured config nor a re-read of source_root may
    # silently beat this explicit call configuration.
    put(args.source_root / "config.json",
        dict(case["config"], num_hidden_layers=36, vocab_size=64, prefix_ids=[28, 29]))
    _, cohort, _, _ = setup(args, profile=profile, config=explicit)
    assert cohort["vocab_size"] == 32
    assert cohort["prefix_ids"] == [30, 31]
    receipt, raw, roster = cli_documents(case)
    run = api.resolve_cli_run(cli_args(api, args, case), receipt, raw, roster,
                              profile=profile, config=explicit)
    assert run["replay_stop"] == 12
    assert run["plan"] == case["plan"]


@pytest.mark.parametrize("key", ["glm", "qwen"])
@pytest.mark.parametrize("bad_layout", [False, True])
def test_frontier_consumer_enforces_profile_hidden_layout(tmp_path, key, bad_layout):
    setup = harness("stage1a_energy").setup
    frontier = harness("frontier_replay")
    case, args, _, _ = fixture_inputs(tmp_path, key)
    _, cohort, batches, _ = setup(args)
    sample = cohort["sample_range"][0]
    ids = batches[0]
    state_path = tmp_path / "state.pt"
    torch.save({sample: {"input_ids": ids}}, state_path)
    # GLM's expanded residual has four streams; Qwen's residual is rank 3.
    shape = [1, ids.shape[1]]
    if case["hidden_streams"] != 1:
        shape += [case["hidden_streams"]]
    shape += [case["config"]["hidden_size"] + int(bad_layout)]
    from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record
    hidden = torch.zeros(shape, dtype=torch.bfloat16)
    ref = write_exact_activation_cache_entry(
        tmp_path / "cache", "fixture-hidden", hidden,
        identity={"coordinates": {"batch": sample, "boundary": 0}},
        max_tensor_bytes=hidden.numel() * hidden.element_size(), max_file_bytes=1024 * 1024,
    )
    receipt = {"schema": "pact.prefixed_clean_frontier.v1", "dry_run": True,
               "cohort": cohort, "sample_ids": [sample], "layer_start": 0, "layer_stop": 1,
               "start_batch_state_path": str(state_path), "start_references": {str(sample): exact_entry_record(ref)}}
    args.boundaries = put(tmp_path / "frontier.json", receipt)
    profile = profile_from_config(case["config"])
    if bad_layout:
        with pytest.raises(ValueError, match="(?i)(hidden|layout|width|shape)"):
            frontier.load_frontier(args, cohort, [ids], cpu_proof=True, profile=profile, config=case["config"])
    else:
        _, _, refs, samples = frontier.load_frontier(
            args, cohort, [ids], cpu_proof=True, profile=profile, config=case["config"])
        assert samples == [sample]
        assert list(refs[sample].shape) == shape


@pytest.mark.parametrize("key", ["glm", "qwen"])
def test_real_sequence_energy_uses_profile_tp_and_original_token_normalization(monkeypatch, key):
    import io
    consumer_api = harness("energy_consumer")
    case = copy.deepcopy(CASES[key])
    profile = profile_from_config(case["config"])
    calls = []
    owner_tp = profile.pact_tp_splits_for_role

    def observe(role):
        calls.append(role)
        return owner_tp(role)

    monkeypatch.setattr(profile, "pact_tp_splits_for_role", observe)
    row = next(r for r in rows_for(case) if r["role"] == "down_proj")
    source = torch.zeros((8, 4), dtype=torch.bfloat16)
    decoded = torch.ones_like(source)
    cohort = dict(case["cohort"], token_sha256="a" * 64)
    option = {"name": "fixture", "family": "BF16", "q256": None,
              "contract": "bf16_unquantized", "weight_reference_definition": "decoded_bf16"}
    stream = io.StringIO()
    consumer = consumer_api.SequenceEnergy(
        [row], {row["qname"]: {"source": source, "options": [(option, decoded)]}},
        cohort=cohort, device=torch.device("cpu"), chunk_rows=2,
        guard=lambda label: None, stream=stream, profile=profile,
    )
    sample = cohort["sample_range"][0]
    consumer.begin(sample)
    consumer.consume((row["qname"],), torch.ones((2, 4), dtype=torch.bfloat16))
    consumer.end()
    aggregates = consumer.finish([sample])
    record = json.loads(stream.getvalue())
    assert "down_proj" in calls, "The real consumer must ask the profile, not repeat a down_proj TP constant"
    assert record["tp_splits"] == owner_tp("down_proj")
    assert record["original_tokens"] == cohort["raw_tokens_per_sequence"]
    assert record["prefix_rows_enter_local_energies"] is False
    assert record["dry_run_cpu"] is True
    assert record["E_W_sum"] == 256.0
    assert record["E_A_sum"] == 0.0
    assert aggregates[0]["E_W_global"] == 256.0 / cohort["global_original_tokens"]


@pytest.mark.parametrize("key", ["glm", "qwen"])
def test_real_live_rows_excludes_prefix_before_consumption(key):
    api = harness("energy_consumer")
    case = CASES[key]
    profile = profile_from_config(case["config"])
    row = {"qname": "model.layers.0.mlp.gate_proj", "role": "gate_proj",
           "kind": "dense", "expert": None, "layer": 0, "shape": [4, 8]}
    model = torch.nn.Module()
    parent = model
    for name in row["qname"].split(".")[:-1]:
        child = torch.nn.Module()
        parent.add_module(name, child)
        parent = child
    linear = torch.nn.Linear(8, 4, bias=False)
    parent.add_module("gate_proj", linear)
    captured = []
    values = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
    with api.LiveRows(model, profile, [row], lambda names, x: captured.append((names, x.clone())),
                      prefix_rows=len(case["cohort"]["prefix_ids"])):
        linear(values)
    assert len(captured) == 1
    assert captured[0][0] == (row["qname"],)
    assert torch.equal(captured[0][1], values.reshape(6, 8)[2:])
    linear(values)
    assert len(captured) == 1, "Live input taps must not leak outside the lease"


@pytest.mark.parametrize("mutation", ["digest", "tokens", "cohort", "geometry", "population", "finite"])
def test_real_teacher_content_binding_keeps_scientific_refusals(tmp_path, mutation):
    api = harness("band_continuation")
    cohort = dict(CASES["qwen"]["cohort"], token_sha256="a" * 64)
    samples = [0, 1]
    arrays = [{"sample_id": s, "path": str(tmp_path / f"teacher-{s}.pt"),
               "bytes": 128, "shape": [3, 32], "dtype": "torch.float32"} for s in samples]
    receipt = {"cohort": cohort, "sample_ids": samples, "teacher_arrays": arrays}
    raw = json.dumps(receipt, sort_keys=True).encode()
    content = {"schema": "pact.prefixed_teacher_validation.v1", "cohort": cohort,
               "sample_ids": samples, "original_token_sha256": cohort["token_sha256"],
               "all_finite": True, "arrays_verified": 2, "batch_state_sha256": "b" * 64,
               "frontier_sha256": hashlib.sha256(raw).hexdigest(),
               "teacher_arrays": [dict(row, file_bytes=row["bytes"], sha256="c" * 64) for row in arrays]}

    def bind(doc, digest=None):
        encoded = json.dumps(doc, sort_keys=True).encode()
        return api.bind_teacher_content(raw, encoded,
            content_sha256=digest or hashlib.sha256(encoded).hexdigest(), cohort=cohort)

    assert bind(content)["arrays"] == {0: "c" * 64, 1: "c" * 64}
    changed = copy.deepcopy(content)
    if mutation == "tokens":
        changed["original_token_sha256"] = "d" * 64
    elif mutation == "cohort":
        changed["cohort"]["prefix_ids"] = [1, 2]
    elif mutation == "geometry":
        changed["teacher_arrays"][0]["shape"] = [3, 33]
    elif mutation == "population":
        changed["sample_ids"] = [0]
    elif mutation == "finite":
        changed["all_finite"] = False
    with pytest.raises(ValueError, match="(?i)(teacher|digest|cohort|token|array|sample|finite|geometry)"):
        bind(changed, digest="0" * 64 if mutation == "digest" else None)


@pytest.mark.parametrize("deadline", [0, -1, 3501])
def test_normal_cli_preserves_bounded_deadline_refusals(tmp_path, deadline):
    api = harness("multi_stream_replay")
    case, args, _, _ = fixture_inputs(tmp_path, "glm")
    receipt, raw, roster = cli_documents(case)
    with pytest.raises(ValueError, match="(?i)(bounded|deadline)"):
        api.resolve_cli_run(cli_args(api, args, case, deadline), receipt, raw, roster)
