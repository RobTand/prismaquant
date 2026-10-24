"""PR #2470 wiring contract: offline served binding, not served task quality.

The shared fixture owns tiny local bytes and the archived producer public CLI.
No installed serving stack, downloads, live endpoint or runtime git is needed.
HF metadata adapters below exercise routing, not real model execution.
"""
import builtins
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from _served_task_fixture import case
from test_task_suite import config as hf_config
from prismaquant import task_suite

pytestmark = pytest.mark.task_suite


@pytest.fixture
def served_only(monkeypatch):
    """Forbid even cached HF/runtime imports during each served API call.

    Ordinary stdlib and PrismaQuant metadata imports remain available. This
    guard deliberately applies to served calls only: HF still owns its normal
    model_runtime initialization.
    """
    def no_hf_reader(*args, **kwargs):
        raise AssertionError("served task called require_hf_reader")

    monkeypatch.setattr(task_suite, "require_hf_reader", no_hf_reader)
    original_import = builtins.__import__
    forbidden = (
        "transformers", "lm_eval", "vllm", "torch", "tessera.serving",
        "model_runtime", "prismaquant.model_runtime",
    )

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = name
        if level and globals and globals.get("__package__"):
            absolute = importlib.util.resolve_name(
                "." * level + name, globals["__package__"]
            )
        requested = [absolute]
        requested.extend(
            absolute + "." + item for item in (fromlist or ())
            if isinstance(item, str) and item != "*"
        )
        for requested_name in requested:
            assert not any(
                requested_name == root or requested_name.startswith(root + ".")
                for root in forbidden
            ), f"served task imported forbidden runtime: {requested_name}"
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _invoke(api, config, tmp_path):
    if api == "validate":
        return task_suite.validate_config(config)
    if api == "preflight":
        return task_suite.preflight_tasks(config)
    if api == "measure":
        return task_suite.measure_tasks(config, tmp_path / "result.json")
    if api == "replay":
        # The binding-only gate must refuse before reading any HF raw result.
        return task_suite.verify_task_result({}, config)
    raise AssertionError(f"unknown test API: {api}")


def test_validation_recognizes_served_without_hf_fields(tmp_path, served_only):
    config, _, _ = case(tmp_path)
    assert "pretrained" not in config["backend"]
    validated = task_suite.validate_config(config)
    assert validated == config
    assert validated["backend"]["name"] == "served"


def test_served_preflight_records_public_binding_and_offline_scope(tmp_path, served_only):
    config, witness, expected = case(tmp_path)
    record = task_suite.preflight_tasks(config)

    assert record["identity"]["backend"] == "served"
    binding = record["binding"]
    assert binding["verdict"] == "pass"
    for field, value in {
        "witness_fingerprint": witness["fingerprint"],
        "endpoint": expected["endpoint"],
        "served_alias": expected["served_alias"],
        "attempt_id": expected["attempt_id"],
        "ranks": expected["ranks"],
    }.items():
        assert binding[field] == value
        assert record["identity"][field] == value
    assert binding["proof_scope"] == "recorded_runtime_byte_binding"

    # This must be the real archived public CLI verdict, not not_configured
    # or a successful local pre-check masquerading as producer authority.
    public = binding["public_verifier"]
    assert isinstance(public, dict)
    assert public["schema"] == "tessera.endpoint_witness_verdict.v1"
    assert public["verdict"] == "valid"
    assert public["mode"] == "offline"
    assert public["proof_scope"] == "recorded_runtime_byte_binding"
    assert public["current_endpoint_verified"] is False
    assert record["population"] == {
        "tasks": ["arc_easy"], "sampling": config["sampling"],
        "device": "served", "skips": [],
    }
    # Keep the human-readable qualification boundary as well as the flag.
    notes = " ".join([
        *record["limitations"], binding.get("note", ""), record.get("note", ""),
    ]).lower()
    assert "recorded" in notes
    assert "current endpoint" in notes
    assert re.search(r"\b(?:not|never|no)\b[^.\n]*\bcurrent endpoint\b", notes)


@pytest.mark.parametrize(
    "missing", [("verifier",), ("served_dir",), ("verifier", "served_dir")],
    ids=["missing-verifier", "missing-served-dir", "no-public-verifier-config"],
)
def test_served_preflight_requires_public_verifier_config(tmp_path, served_only, missing):
    config, _, _ = case(tmp_path)
    for field in missing:
        del config["backend"]["serving_runtime"][field]
    # In particular, dropping both must not return public_verifier=not_configured.
    with pytest.raises(ValueError, match=re.escape("serving_runtime." + missing[0])):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("api", ["validate", "preflight", "measure", "replay"])
@pytest.mark.parametrize(
    "field", ["witness", "expected", "endpoint", "served_alias", "attempt_id", "ranks"],
)
def test_served_apis_name_each_missing_binding_field(tmp_path, served_only, api, field):
    config, _, _ = case(tmp_path)
    del config["backend"]["serving_runtime"][field]
    # A generic "requires explicit hf model" ValueError is NOT a passing refusal.
    with pytest.raises(ValueError, match=re.escape("serving_runtime." + field)):
        _invoke(api, config, tmp_path)


@pytest.mark.parametrize("api", ["validate", "preflight", "measure", "replay"])
def test_served_apis_require_serving_runtime(tmp_path, served_only, api):
    config, _, _ = case(tmp_path)
    del config["backend"]["serving_runtime"]
    with pytest.raises(ValueError, match=r"backend\.serving_runtime"):
        _invoke(api, config, tmp_path)


@pytest.mark.parametrize("api", ["preflight", "measure", "replay"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("endpoint", "http://fixture.invalid:9999"),
     ("served_alias", "other-fixture"), ("attempt_id", "other-attempt"),
     ("ranks", [0])],
)
def test_served_binding_mismatch_refuses_before_hf_or_deferred_work(
    tmp_path, served_only, api, field, value,
):
    config, _, _ = case(tmp_path)
    config["backend"]["serving_runtime"][field] = value
    with pytest.raises(ValueError, match=re.escape(field)):
        _invoke(api, config, tmp_path)


def test_served_preflight_uses_public_cli_to_check_actual_bytes(tmp_path, served_only):
    config, _, _ = case(tmp_path)
    # The local witness and independent expectation document remain consistent.
    # Only the public verifier sees this same-size change to selected disk bytes.
    artifact = Path(config["backend"]["serving_runtime"]["served_dir"]) / "model.safetensors"
    raw = artifact.read_bytes()
    artifact.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match=r"(?i)(public|verifier|artifact|bytes|sha256)"):
        task_suite.preflight_tasks(config)


@pytest.mark.parametrize("api", ["measure", "replay"])
def test_valid_served_measurement_and_replay_defer_to_2430(tmp_path, served_only, api):
    config, _, _ = case(tmp_path)
    # Positive validation separately proves this is not an HF-config rejection.
    assert task_suite.validate_config(config)["backend"]["name"] == "served"
    with pytest.raises(ValueError, match="2430"):
        _invoke(api, config, tmp_path)
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.lm_eval.json").exists()


def test_original_hf_validation_and_metadata_preflight_route_without_downloads(monkeypatch):
    from prismaquant import model_runtime

    config = hf_config()
    assert task_suite.validate_config(config) == config
    calls = []
    model = SimpleNamespace(model_type="fixture-hf", _commit_hash="fixture-commit",
                            quantization_config=None)
    model.get_text_config = lambda **kwargs: SimpleNamespace(quantization_config=None)

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    def load_tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return ["token-a", "token-b"]

    manager = object()

    def get_task_dict(tasks, *, task_manager):
        assert tasks == config["tasks"]
        assert task_manager is manager
        calls.append(("tasks", list(tasks)))
        return {"arc_easy": object()}

    # Replace metadata providers only; exercise the real require_hf_reader.
    # Keep its explicit model_runtime initialization (with no runtime side effects).
    monkeypatch.setattr(model_runtime, "initialize_model_runtime",
                        lambda: calls.append(("initialize",)))
    transformers = ModuleType("transformers")
    transformers.__path__ = []
    transformers.AutoConfig = SimpleNamespace(from_pretrained=load_model)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    quantizers = ModuleType("transformers.quantizers")
    quantizers.__path__ = []
    auto = ModuleType("transformers.quantizers.auto")
    auto.AutoHfQuantizer = SimpleNamespace(
        supports_quant_method=lambda value: pytest.fail("unexpected quantized HF metadata")
    )
    lm_eval = ModuleType("lm_eval")
    lm_eval.__path__ = []
    tasks = ModuleType("lm_eval.tasks")
    tasks.TaskManager = lambda: manager
    tasks.get_task_dict = get_task_dict
    for module in (transformers, quantizers, auto, lm_eval, tasks):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(task_suite, "version", lambda name: "fixture-" + name)

    record = task_suite.preflight_tasks(config)
    options = {"revision": None, "trust_remote_code": False}
    assert calls == [
        ("initialize",), ("model", config["backend"]["pretrained"], options),
        ("tokenizer", config["backend"]["tokenizer"], options),
        ("tasks", ["arc_easy"]),
    ]
    assert record["identity"] == {
        "backend": {name: "fixture-" + name for name in
                    ("lm-eval", "torch", "transformers", "datasets")},
        "model_type": "fixture-hf", "model_commit": "fixture-commit",
        "tokenizer": config["backend"]["tokenizer"], "tokenizer_vocab_size": 2,
    }
    assert record["population"] == {
        "tasks": ["arc_easy"], "sampling": config["sampling"],
        "device": "cpu", "skips": [],
    }
    assert "binding" not in record
    assert any("does not run model inference" in note.lower() for note in record["limitations"])
