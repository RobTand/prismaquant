"""Record the CUDA profile of a selected-source campaign row."""
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_calibration_cache as capture
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_hessian
from prismaquant import tessera_render
from prismaquant.autoscale import BOUNDED_CAPTURE_ENV
from prismaquant.model_profiles import DefaultProfile

FORMAT = "TESSERA_E4M3_K1_R1024"
NAMES = [f"model.layers.{i}.proj" for i in range(3)]
COLUMNS, OUTPUTS, ROWS, SEED = 256, 8, 4, 2955
TRACE = "2955-encode.trace.json"
SUMMARY = "2955-encode-summary.json"


def _profile(root, out, monkeypatch):
    from safetensors.torch import save_file
    from prismaquant import cost_streaming, layer_streaming, model_profiles
    from prismaquant import tessera_campaign_selection, tessera_row_stream

    assert torch.cuda.is_available(), "Use the Flux GPU lane."
    generator = torch.Generator().manual_seed(SEED)
    weights = {name: torch.randn(OUTPUTS, COLUMNS, generator=generator).bfloat16()
               for name in NAMES}
    acts = {name: torch.randn(ROWS, COLUMNS, generator=generator) for name in NAMES}
    hessians = {name: x.T @ x + torch.eye(COLUMNS) for name, x in acts.items()}
    source = root / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(dict(
        model_type="llama", hidden_size=COLUMNS, intermediate_size=COLUMNS,
        num_hidden_layers=len(NAMES), num_attention_heads=1)))
    save_file({name + ".weight": w for name, w in weights.items()},
              str(source / "model.safetensors"))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in NAMES)
    for layer in model.model.layers:
        layer.proj = torch.nn.Linear(COLUMNS, OUTPUTS, bias=False, device="meta")
    model.config = SimpleNamespace(_attn_implementation="eager")

    def build(*_args, source_authentication, **_kwargs):
        files, keys = layer_streaming._build_weight_map(
            str(source), source_authentication=source_authentication)

        def snapshot(names, **_options):
            return {name: layer_streaming._read_layer_to_device(
                name.rsplit(".", 1)[0] + ".", files, keys, torch.bfloat16,
                torch.device("cpu"), source_authentication=source_authentication
            )[name + ".weight"] for name in names}, dict(
                schema="prismaquant.selected_source_weights.v1", source_forward_count=0)

        return SimpleNamespace(model=model, snapshot_selected_weights=snapshot,
                               shutdown=lambda: None)

    profile = DefaultProfile()
    menu = [SimpleNamespace(format_name=FORMAT, family="TESSERA_E4M3_K1",
                            body_rate_q256=1024, bpp=4.0)]
    tokens, text = [torch.arange(COLUMNS).reshape(1, -1)], "synthetic profile draw"
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: profile)
    monkeypatch.setattr(campaign, "_calibration_tokens", lambda *_args: (tokens, text))
    monkeypatch.setattr(campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: dict.fromkeys(targets, menu))

    version = importlib.metadata.version("transformers")
    contract = dict(schema="prismaquant.streaming_initialization.v1",
        scope="streamed_text_source_forward", status="completed",
        transformers_version=version, model_class="SyntheticSource",
        dtype="torch.bfloat16", layers_prefix="model.layers.", num_layers=len(NAMES),
        persistent_tensors=len(NAMES), derived_buffers=0,
        state_sha256="a" * 64, source_map_sha256="b" * 64)
    calibration = campaign.th.calibration_identity(text, tokens, fit_tokens=ROWS,
        source="wikitext-2-raw-v1/train", split_role="calibration", model=str(source),
        seed=SEED, nsamples=1, seqlen=COLUMNS, fit_tokens_min=ROWS)
    groups = campaign.resolve_anchor_groups(NAMES, profile=profile, expert_members={})
    census = campaign.calibration_census(
        dict.fromkeys(NAMES, ROWS), {name: float(x.abs().max()) for name, x in acts.items()},
        args=SimpleNamespace(model=str(source), nsamples=1, seqlen=COLUMNS, seed=SEED,
                             layer_stride=1),
        groups=groups, dense_targets=NAMES, expert_targets=[],
        shapes=dict.fromkeys(NAMES, [OUTPUTS, COLUMNS]), identity=calibration,
        model_load_contract=contract, attention_implementation="eager",
        capture_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=version))
    census_path = root / "census.json"
    census_path.write_text(json.dumps(census))
    identity = capture.capture_identity(census_path, calibration=calibration,
        max_act_rows=ROWS, model_load_contract=contract, attention_implementation="eager")
    complete = capture.publish_capture(root / "capture", census_path=census_path,
        identity=identity, acts=acts, hessians=hessians, counts=census["counts"],
        maxima=census["max_abs"])
    selection = root / "units.json"
    selection.write_text(json.dumps(dict(schema=tessera_campaign_selection.UNITS_SCHEMA,
        groups=[dict(key=key, members=list(members)) for key, members in sorted(groups.items())])))
    args = ["--model", str(source), "--out", str(root / "cost.pkl"),
        "--cache-dir", str(root / "cache"), "--row-head", "stream",
        "--hessian", "require", "--menu-mode", "research", "--max-rounds", "1",
        "--attention-implementation", "eager", "--streaming",
        "--streaming-cache-headroom-gb", "0", "--units", str(selection),
        "--calibration-census", str(census_path), "--calibration-cache", complete["path"],
        "--calibration-cache-sha256", complete["sha256"], "--nsamples", "1",
        "--seqlen", str(COLUMNS), "--seed", str(SEED), "--layer-stride", "1",
        "--max-act-rows", str(ROWS), "--campaign-identity-threads", "1",
        "--capture-load-policy", json.dumps(dict(schema="prismaquant.verified_activation_load.v1",
            max_buffer_bytes=1024**2, max_scratch_bytes=1024**2)),
        "--export-hessian-reference-policy", json.dumps(dict(schema="tessera.hessian_reference_load.v1",
            max_metadata_bytes=1024**2, max_file_bytes=1024**2, max_hessian_bytes=1024**2))]

    def mark(owner, name, label):
        fn = getattr(owner, name)

        @functools.wraps(fn)
        def marked(*args, **kwargs):
            with torch.profiler.record_function(label):
                return fn(*args, **kwargs)

        monkeypatch.setattr(owner, name, marked)

    labels = ("campaign.admit", "campaign.factor", "campaign.encode")
    mark(tessera_row_stream.RowStream, "admit", labels[0])
    mark(tessera_hessian, "encoder_kwargs", labels[1])
    mark(tessera_render, "encode_tessera_unit", labels[2])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        assert campaign.main(args) == 0
    trace = out / TRACE
    prof.export_chrome_trace(str(trace))
    trace_bytes = trace.read_bytes()
    events = json.loads(trace_bytes)["traceEvents"]
    summary = dict(
        host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
        python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
        tessera=importlib.metadata.version("tessera-quant"),
        inputs_sha256=hashlib.sha256(b"".join(
            t.view(torch.uint8).numpy().tobytes() for inputs in (weights, acts, hessians)
            for t in inputs.values())).hexdigest(),
        trace=TRACE, trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        kernels=sum(event.get("cat") == "kernel" for event in events),
        copies=sum(event.get("cat") == "gpu_memcpy" for event in events),
        cpu_ranges_us={label: [round(event["dur"], 3) for event in events
            if event.get("name") == label and event.get("ph") == "X"] for label in labels})
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")


def test_encode_profile_matches_baseline_workload(tmp_path):
    record = Path(__file__).resolve().parents[1] / "docs/measurements/2955-encode-baseline.json"
    baseline = json.loads(record.read_text())["profile"]
    out = Path(os.environ.get("TL3B_OUT", tmp_path)).resolve()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(out)],
                   env={**os.environ, **BOUNDED_CAPTURE_ENV, "PYTHONPATH": "."}, check=True)
    observed = json.loads((out / SUMMARY).read_text())
    assert observed["inputs_sha256"] == baseline["inputs_sha256"]
    for profile in (baseline, observed):
        assert profile["kernels"] > 0 and profile["copies"] > 0
        for label in ("campaign.admit", "campaign.factor", "campaign.encode"):
            assert len(profile["cpu_ranges_us"][label]) >= len(NAMES)


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as monkeypatch:
        _profile(Path(sys.argv[1]), Path(sys.argv[2]), monkeypatch)
