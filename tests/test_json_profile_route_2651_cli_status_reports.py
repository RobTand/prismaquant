import json
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig

from prismaquant import (
    digests, export_structure, glm_source_derivative, model_walk, prefill_frontier,
    release_receipts, shape_runtime_prices, ship_gates, streaming_model,
)


def test_listed_calls_use_insertion_ascii_spaced_lax(tmp_path, monkeypatch, capsys):
    real_text = digests.JsonProfile.text
    calls = []
    marker = False

    def observe(profile, value):
        text = real_text(profile, value)
        if profile is digests.INSERTION_ASCII_SPACED_LAX:
            calls.append(text.encode("ascii"))
            if marker:
                return (" " + text.replace("ROUTER", "other").replace("EXPERT", "other")
                        .replace('"lower_bound": -5', '"lower_bound": -6')
                        .replace('"torch.float32"', '"torch.float64"')
                        .replace("[1, 2]", "[1, 3]"))
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe)
    monkeypatch.setattr(export_structure, "verify_export_structure", _refuse)
    monkeypatch.setattr(release_receipts, "ingest_receipts", _refuse)
    price = {"table_id": "caf\u00e9", "rows": 2}
    monkeypatch.setattr(shape_runtime_prices, "load_shape_table",
                        lambda _: SimpleNamespace(identity=lambda: price))
    path = tmp_path / "report.json"
    path.write_bytes(b"{}")
    problems = ["caf\u00e9"]
    monkeypatch.setattr(ship_gates, "run_ship_gates", lambda *a, **kw: {
        "status": "refused", "stages": [], "problems": problems,
    })
    summary = {"status": "refused", "evidence": [
        {"path": str(path), "sha256": digests.bytes_sha256hex(b"{}")},
    ], "problems": problems}

    derivative = glm_source_derivative
    image = derivative.CORRECTED_IMAGE_CONTENT_SHA256
    build = {"corrected_image_content_sha256": image}
    build_path = tmp_path / "build.json"
    build_path.write_text(json.dumps(build))
    binding = {"path": str(build_path), "sha256": digests.file_sha256hex(build_path)}
    policy = {"schema": derivative.SCHEMA, "version": derivative.VERSION, "image_build": binding}
    observed = {"image_content_sha256": image, "gates": {"unit": {"lower_bound": -5}}}
    monkeypatch.setattr(derivative, "validate_image_build", lambda value: value)
    monkeypatch.setattr(derivative, "_observe", lambda *_: observed)
    monkeypatch.setenv("PRISMAQUANT_CONTAINER_CONTENT_SHA256", image)
    profile = SimpleNamespace(source_derivative_contract=derivative.declaration)

    config = AutoConfig.for_model(
        "qwen3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
        attention_dropout=0.0, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config.save_pretrained(tmp_path)
    rules = ({"disposition": "pin", "predicate": "ROUTER"},
             {"disposition": "decide", "predicate": "EXPERT"})
    walk_model = model_walk.walk_model

    def walk(*args, **kwargs):
        result = walk_model(*args, **kwargs)
        return replace(result, provenance=replace(result.provenance, claim_rules=rules))

    monkeypatch.setattr(model_walk, "walk_model", walk)
    model = nn.Module()
    model.embed_tokens = nn.Embedding(2, 2)
    model.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(2))
    weights = {name: name for name, _ in model.named_parameters()}
    audit = streaming_model._StreamingInitializationAudit(SimpleNamespace(
        model=model, layers_prefix="layers.", num_layers=2, dtype=torch.float32,
        weight_ckpt=weights, weight_shard=dict.fromkeys(weights, "tiny.safetensors"),
        install_resolvers=[{f"layers.{i}.{key}": None for key in ("weight", "bias")}
                           for i in range(2)],
    ))
    audit.observe_layer(0, audit.context.install_resolvers[0])

    expected = {}
    actual = {}

    def check_calls(site, *values):
        expected[site] = [json.dumps(value).encode("ascii") for value in values]
        actual[site] = calls[:]
        calls.clear()

    def check_print(site, status, payload):
        assert status == {"structure_error": 1, "error": 2, "refused": 1}.get(payload.get("status"), 0)
        assert capsys.readouterr().out == (" " if marker else "") + json.dumps(payload) + "\n"
        check_calls(site, payload)

    for marker in (False, True):
        check_print("export_structure", export_structure.main([str(tmp_path)]),
                    {"status": "structure_error", "error": "caf\u00e9"})
        check_print("release_receipts", release_receipts.main([str(tmp_path)]),
                    {"status": "error", "error": "caf\u00e9"})
        check_print("shape_runtime_prices", shape_runtime_prices.main(["check", str(path)]), price)
        path.write_bytes(b"{}")
        check_print("ship_gates", ship_gates.main(["--config", "local", "--output", str(path)]), summary)

        identity = derivative.bind_source_derivative(nn.Linear(1, 1), profile, policy)
        assert identity == {
            **observed, "image_build_sha256": binding["sha256"],
            "gates": {"unit": {"lower_bound": -6 if marker else -5}},
        }
        check_calls("bind_source_derivative", {**observed, "image_build_sha256": binding["sha256"]})
        assert prefill_frontier._as_json({"tuple": (1, 2)}) == {"tuple": [1, 3 if marker else 2]}
        check_calls("_as_json", {"tuple": (1, 2)})

        for method, layers in ((audit.complete_prefix, 0), (audit.complete_selected, [0])):
            state = method(layers)["state"]
            assert state == {name: {**record, "dtype": "torch.float64" if marker else "torch.float32"}
                             for name, record in audit.records.items()}
            assert state is not audit.records
            check_calls(method.__name__, audit.records)

        assert model_walk.main(["--model", str(tmp_path), "--output", str(path)]) == 0
        scope = json.loads(path.read_text())["gate"]["scope"]
        assert scope["router_pin_rules"] == (0 if marker else 1)
        assert scope["packed_expert_decide_rules"] == (0 if marker else 1)
        check_calls("model_walk", *rules)
        capsys.readouterr()
        assert actual == expected, "Each site must use the profile and keep the original bytes."


def _refuse(*args, **kwargs):
    raise ValueError("caf\u00e9")
