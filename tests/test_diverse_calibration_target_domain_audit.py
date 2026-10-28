"""Nonpositive token windows refuse before entering the real packing loop."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_diverse_calibration_builder import _FakeTokenizer, _load_builder_module


class RepeatedEmptyWindows(RuntimeError):
    pass


class BoundedTokenizer(_FakeTokenizer):
    def __init__(self):
        self.empty_windows = 0

    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            self.empty_windows += 1
            if self.empty_windows == 4:
                raise RepeatedEmptyWindows(
                    "actual pack loop attempted four empty windows without advancing")
        return super().decode(ids, skip_special_tokens=skip_special_tokens)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    builder = _load_builder_module()
    tokenizer = BoundedTokenizer()
    external = ModuleType("transformers")
    external.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", external)
    text = "word " * 12

    def tiny_dataset(_source):
        return iter([{"text": text, "code": text, "source": text,
                      "source_lang": "eng_Latn", "language": "python"}] * 8)

    monkeypatch.setattr(builder, "_load_dataset_stream", tiny_dataset)
    local_tokenizer = tmp_path / "tokenizer"
    local_tokenizer.mkdir()
    output = tmp_path / "diverse.jsonl"
    argv = ["--tokenizer", str(local_tokenizer), "--rows", "5",
            "--output", str(output)]
    return builder, tokenizer, output, argv


@pytest.mark.parametrize("target", [0, -1])
def test_cli_refuses_nonpositive_target_before_empty_window_loop(cli, capsys, target):
    builder, tokenizer, output, argv = cli
    with pytest.raises((ValueError, SystemExit)) as refused:
        builder.main([*argv, "--target-tokens", str(target)])
    if isinstance(refused.value, SystemExit):
        assert refused.value.code not in (0, None)
    reason = (str(refused.value) + capsys.readouterr().err).lower()
    assert "target" in reason and "positive" in reason
    assert tokenizer.empty_windows == 0
    assert not output.exists()


def test_positive_target_publishes_the_actual_jsonl(cli):
    builder, tokenizer, output, argv = cli
    assert builder.main([*argv, "--target-tokens", "2"]) == 0
    values = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = values[0]["__manifest__"]
    assert manifest["schema"] == "prismaquant.calibration.diverse_v1"
    assert manifest["row_count"] == 5
    assert manifest["target_tokens"] == 2
    assert len(values[1:]) == 5
    assert all(len(row["text"].split()) == 2 for row in values[1:])
    assert tokenizer.empty_windows == 0


@pytest.mark.parametrize("target", [0, -1, True, 1.5, "2"])
@pytest.mark.parametrize("needed", [0, 1])
def test_pack_refuses_invalid_target_before_consuming_inputs(target, needed):
    builder = _load_builder_module()

    def unused_texts():
        pytest.fail("invalid target consumed source text")
        yield

    with pytest.raises(ValueError, match="target_tokens must be a positive integer"):
        builder.pack_token_windows(
            unused_texts(), object(), domain="prose", needed=needed,
            target_tokens=target, seed=0)


@pytest.mark.parametrize("target", [0, -1])
def test_invalid_cli_target_does_not_load_external_inputs(cli, monkeypatch, target):
    builder, _, output, argv = cli

    def unused(*args, **kwargs):
        pytest.fail("invalid CLI target loaded an external input")

    monkeypatch.setattr(sys.modules["transformers"].AutoTokenizer, "from_pretrained", unused)
    monkeypatch.setattr(builder, "_load_dataset_stream", unused)
    with pytest.raises(SystemExit) as refused:
        builder.main([*argv, "--target-tokens", str(target)])
    assert refused.value.code == 2
    assert not output.exists()


def test_default_target_publishes_its_existing_window_size(cli, monkeypatch):
    builder, tokenizer, output, argv = cli
    text = "word " * builder.DEFAULT_TARGET_TOKENS
    monkeypatch.setattr(builder, "_load_dataset_stream", lambda source: iter([
        {"text": text, "code": text, "source": text,
         "source_lang": "eng_Latn", "language": "python"}] * 8))
    assert builder.main(argv) == 0
    values = [json.loads(line) for line in output.read_text().splitlines()]
    assert values[0]["__manifest__"]["target_tokens"] == builder.DEFAULT_TARGET_TOKENS == 4096
    assert len(values[1:]) == 5
    assert all(len(row["text"].split()) == 4096 for row in values[1:])
    assert tokenizer.empty_windows == 0
