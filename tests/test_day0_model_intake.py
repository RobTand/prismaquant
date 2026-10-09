"""The day-zero tool reads source metadata and never promotes its draft."""
import json
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from test_footprint import _write_safetensors
from prismaquant.model_profiles.structure import ModelStructureSpec

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1" * 40


def checkpoint(root, *, wrapper=False, indexed=False, unknown=False):
    root.mkdir()
    prefix = "model.language_model" if wrapper else "model"
    config = {
        "model_type": "future_model" if unknown else ("qwen3_5_moe" if wrapper else "qwen3"),
        "architectures": ["FutureForCausalLM" if unknown else (
            "Qwen3_5MoeForConditionalGeneration" if wrapper else "Qwen3ForCausalLM")],
        "num_hidden_layers": 2, "hidden_size": 4, "vocab_size": 8,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "intermediate_size": 6,
    }
    if wrapper:
        config = {"model_type": config["model_type"], "architectures": config["architectures"],
                  "text_config": config}
    (root / "config.json").write_text(json.dumps(config))
    tensors = {f"{prefix}.embed_tokens.weight": ("BF16", [8, 4]),
               "lm_head.weight": ("BF16", [8, 4])}
    for layer in range(2):
        tensors[f"{prefix}.layers.{layer}.self_attn.q_proj.weight"] = ("BF16", [4, 4])
        if wrapper:
            tensors[f"{prefix}.layers.{layer}.mlp.experts.0.gate_proj.weight"] = ("BF16", [6, 4])
    _write_safetensors(root / "model.safetensors", tensors)
    if indexed:
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    return config, tensors


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "tools/day0_model_intake.py"), *map(str, args)],
                          cwd=ROOT, text=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("wrapper,indexed,profile", [(False, False, "qwen3"), (True, True, "qwen3_5")])
def test_real_cli_two_layouts(tmp_path, wrapper, indexed, profile):
    root = tmp_path / "source"
    _, tensors = checkpoint(root, wrapper=wrapper, indexed=indexed)
    out = tmp_path / "out"
    run = cli("--checkpoint", root, "--output", out)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((out / "intake.json").read_text())
    draft = json.loads((out / "structure.draft.json").read_text())
    parsed = ModelStructureSpec.from_dict(draft)
    assert parsed.match.claims(report["model_type"], report["architectures"])
    assert draft["supported_lanes"] == []
    assert report["profile"] == profile
    assert report["profile_registered"] is True
    assert report["native_serving_qualified"] is False
    assert report["source"]["tensors"] == len(tensors)
    assert report["dimensions"] == {"layers": 2, "hidden_size": 4, "vocab_size": 8}
    assert draft["shard_regexes"]["body_layer_prefix"] == (
        "model.language_model.layers" if wrapper else "model.layers")
    assert report["bf16_tp2"]["status"] == "not_requested"
    assert not report["unsupported_module_kinds"]
    assert "vllm" not in report["runtime_checks_executed"]


def test_unknown_profile_and_unknown_tensor_are_not_supported(tmp_path):
    root = tmp_path / "source"
    checkpoint(root, unknown=True)
    out = tmp_path / "out"
    run = cli("--checkpoint", root, "--output", out)
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((out / "intake.json").read_text())
    assert report["profile"] == "default"
    assert report["profile_registered"] is False
    assert report["native_serving_qualified"] is False
    assert any(row["kind"] == "unregistered_architecture" for row in report["unsupported_module_kinds"])
    assert not ModelStructureSpec.from_dict(json.loads((out / "structure.draft.json").read_text())).supported_lanes


def test_unclassified_module_kind_is_explicit(tmp_path):
    root = tmp_path / "source"
    _, tensors = checkpoint(root)
    tensors["model.layers.0.custom.weight"] = ("BF16", [2, 4, 4])
    _write_safetensors(root / "model.safetensors", tensors)
    run = cli("--checkpoint", root, "--output", tmp_path / "out")
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((tmp_path / "out/intake.json").read_text())
    assert any(row["tensor"] == "model.layers.0.custom.weight" and row["kind"] == "unclassified_parameter"
               for row in report["unsupported_module_kinds"])


@pytest.mark.parametrize("change,error", [
    ("layer_count", "num_hidden_layers"), ("hidden", "hidden_size"),
    ("vocab", "vocab_size"), ("architectures", "architectures"),
    ("duplicate_config", "duplicate"), ("index_roster", "roster"),
    ("unsafe_index", "weight_map"), ("missing_shard", "missing"),
])
def test_inconsistent_source_refuses_before_runtime(tmp_path, change, error):
    root = tmp_path / "source"
    config, tensors = checkpoint(root, indexed=True)
    if change in {"layer_count", "hidden", "vocab", "architectures"}:
        field, value = {"layer_count": ("num_hidden_layers", 3), "hidden": ("hidden_size", 5),
                        "vocab": ("vocab_size", 9), "architectures": ("architectures", "not-a-list")}[change]
        config[field] = value
        (root / "config.json").write_text(json.dumps(config))
    elif change == "duplicate_config":
        (root / "config.json").write_text('{"model_type":"qwen3","model_type":"bad"}')
    elif change == "index_roster":
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {**{name: "model.safetensors" for name in tensors}, "ghost": "model.safetensors"}}))
    elif change == "unsafe_index":
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../bad"}}))
    else:
        (root / "model.safetensors").unlink()
    out = tmp_path / "out"
    run = cli("--checkpoint", root, "--output", out, "--bf16-tp2",
              "--runtime-image", "image:explicit", "--runtime-gpus", "all")
    assert run.returncode != 0
    assert error in run.stderr + run.stdout
    assert not out.exists()
    assert "docker" not in run.stderr.lower()


def test_cpu_preflight_uses_real_tp2_entry_point(tmp_path):
    root = tmp_path / "source"
    checkpoint(root)
    out = tmp_path / "out"
    run = cli("--checkpoint", root, "--output", out, "--bf16-tp2-preflight",
              "--runtime-image", "image:explicit", "--runtime-gpus", "all")
    assert run.returncode == 0, run.stderr + run.stdout
    report = json.loads((out / "intake.json").read_text())
    check = report["bf16_tp2"]
    assert check["status"] == "cpu_preflight_only"
    assert check["command"][0] == "docker"
    command = check["command"]
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "/source/tools/vllm_prompt_smoke.py" in command
    assert report["native_serving_qualified"] is False


@pytest.fixture
def hub(tmp_path):
    source = tmp_path / "hub_source"
    config, tensors = checkpoint(source, indexed=True)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            requests.append((self.command, self.path))
            names = ["config.json", "model.safetensors.index.json", "model.safetensors"]
            if "/tree/" in self.path:
                payload = json.dumps([{"type": "file", "path": name, "size": (source / name).stat().st_size,
                                       "oid": "2" * 40} for name in names]).encode()
            elif self.path.startswith("/api/models/"):
                payload = json.dumps({"id": "test/small", "sha": REVISION,
                                      "siblings": [{"rfilename": name} for name in names]}).encode()
            else:
                file = source / self.path.rsplit("/", 1)[-1]
                if not file.is_file():
                    self.send_error(404)
                    return
                payload = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"' + "2" * 64 + '"')
            self.send_header("X-Repo-Commit", REVISION)
            self.end_headers()
            if body:
                self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield source, config, requests, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("bad", [False, True])
def test_real_metadata_download_stops_before_weights(tmp_path, hub, bad):
    source, config, requests, endpoint = hub
    if bad:
        config["num_hidden_layers"] = 3
        (source / "config.json").write_text(json.dumps(config))
    out = tmp_path / "out"
    args = ["--model-id", "test/small", "--revision", REVISION, "--download-dir", tmp_path / "download",
            "--hub-endpoint", endpoint, "--output", out]
    if bad:
        args += ["--bf16-tp2", "--runtime-image", "image:explicit", "--runtime-gpus", "all"]
    run = cli(*args)
    assert run.returncode == (1 if bad else 0), run.stderr + run.stdout
    assert any(method == "GET" and path.endswith("config.json") for method, path in requests)
    assert not any(path.endswith("model.safetensors") for _, path in requests)
    if bad:
        assert "num_hidden_layers" in run.stderr
        assert not out.exists()
    else:
        report = json.loads((out / "intake.json").read_text())
        assert report["source"]["scope"] == "index_only"
        assert report["source"]["headers_read"] == 0
        assert report["revision"] == REVISION


def test_remote_revision_and_runtime_inputs_refuse_before_download(tmp_path, hub):
    _, _, requests, endpoint = hub
    run = cli("--model-id", "test/small", "--revision", "main", "--hub-endpoint", endpoint,
              "--download-dir", tmp_path / "download", "--output", tmp_path / "out")
    assert run.returncode != 0
    assert "revision" in run.stderr
    assert not requests
    run = cli("--model-id", "test/small", "--revision", REVISION, "--hub-endpoint", endpoint,
              "--download-dir", tmp_path / "download", "--output", tmp_path / "out", "--bf16-tp2")
    assert run.returncode != 0
    assert "runtime-image" in run.stderr
    assert not requests
