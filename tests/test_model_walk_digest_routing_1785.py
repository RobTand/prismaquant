"""Model-walk metadata preserves literal bytes while routing SHA-256 to its owner."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import model_walk as walk
from prismaquant.digests import bytes_sha256hex


def _legacy_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=True, default=str).encode("utf-8")


def _spy(monkeypatch):
    seen = []

    def record(raw):
        seen.append(raw)
        return bytes_sha256hex(raw)

    monkeypatch.setattr(walk, "bytes_sha256hex", record, raising=False)
    return seen


@pytest.mark.parametrize("rules", [(), (
    walk.ClaimRule("pin", "λ📦\r\n", leaf="weight", floating=True),
    walk.ClaimRule("decide", "linear", module_class="Linear", max_ndim=2),
)])
def test_rules_route_exact_legacy_bytes(monkeypatch, rules):
    seen = _spy(monkeypatch)
    raw = _legacy_bytes(list(walk.claim_rules_to_json(rules)))
    assert walk._claim_rules_digest(rules) == hashlib.sha256(raw).hexdigest()
    assert seen == [raw]


class _StableText:
    def __str__(self):
        return "µ𐐀"


@pytest.mark.parametrize("model_type", ["módelo", ""])
def test_config_routes_lax_ascii_json_and_keeps_identity_prefix(monkeypatch, model_type):
    seen = _spy(monkeypatch)
    data = {"z": [1, True, None], "a": "λ📦\r\n",
            "nan": float("nan"), "custom": _StableText()}
    model = nn.Linear(2, 3)
    monkeypatch.setattr(model, "config", SimpleNamespace(
        to_dict=lambda: data, model_type=model_type), raising=False)
    raw = _legacy_bytes(data)
    expected = f"{model_type or 'Linear'}:sha256:{hashlib.sha256(raw).hexdigest()}"
    assert walk._model_identity(model) == expected
    assert seen == [raw]


@pytest.mark.parametrize("case", ["mapping", "tuple", "single", "empty", "long"])
def test_provided_inputs_route_utf8_and_keep_digest_and_description_lengths(monkeypatch, case):
    seen = _spy(monkeypatch)
    matrix = torch.empty((2, 3), dtype=torch.float32)
    zero = torch.empty((0,), dtype=torch.float32)
    if case == "mapping":
        inputs = {"β": [matrix, [zero]], "a": torch.empty(4, dtype=torch.int64)}
        blob = "a:[4]@torch.int64;β:[2, 3]@torch.float32;β:[0]@torch.float32"
    elif case == "tuple":
        inputs = (matrix, [zero])
        blob = "[0]:[2, 3]@torch.float32;[1]:[0]@torch.float32"
    elif case == "single":
        inputs, blob = matrix, "in:[2, 3]@torch.float32"
    elif case == "empty":
        inputs, blob = {"ignored": "not a tensor"}, ""
    else:
        key = "λ" * 200
        inputs, blob = {key: matrix}, key + ":[2, 3]@torch.float32"
    raw = blob.encode("utf-8")
    expected = f"provided:{hashlib.sha256(raw).hexdigest()[:12]}:{blob[:160]}"
    assert walk._example_inputs_spec(inputs, 987, False) == expected
    assert seen == [raw]


@pytest.mark.parametrize("seq_len", [0, 16, -1])
def test_default_contract_stays_unhashed(monkeypatch, seq_len):
    seen = _spy(monkeypatch)
    assert walk._example_inputs_spec(None, seq_len, True) == (
        f"default:input_ids(1,{seq_len})+use_cache_if_accepted")
    assert seen == []


def test_surrogate_input_name_still_refuses_before_hashing(monkeypatch):
    seen = _spy(monkeypatch)
    with pytest.raises(UnicodeEncodeError):
        walk._example_inputs_spec({"\ud800": torch.empty(1)}, 16, False)
    assert seen == []


@pytest.mark.parametrize("broken_config", [False, True])
def test_model_metadata_fallback_stays_unhashed(monkeypatch, broken_config):
    seen = _spy(monkeypatch)
    model = nn.Linear(2, 3)
    if broken_config:
        def fail():
            raise ValueError("fixture config refused")
        monkeypatch.setattr(model, "config", SimpleNamespace(to_dict=fail), raising=False)
    assert walk._model_identity(model) == "Linear:params:9"
    assert seen == []


def test_parameter_introspection_failure_keeps_negative_one(monkeypatch):
    seen = _spy(monkeypatch)

    class Broken(nn.Module):
        def parameters(self, recurse=True):
            raise RuntimeError("fixture parameters refused")

    assert walk._model_identity(Broken()) == "Broken:params:-1"
    assert seen == []
