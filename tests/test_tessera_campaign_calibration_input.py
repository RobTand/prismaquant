"""Immutable campaign draw intake (PQ #1827), using real tiny safetensors."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from test_tessera_campaign_resume import UNIT, _main_fixture

from prismaquant import digests
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_hessian as th
from prismaquant.incremental_shards import read_pickle


def _saved_draw(tmp_path, *, ids=None, changes=None, corpus=b"Real corpus\r\nwith UTF-8: \xc3\xa9\n"):
    ids = torch.tensor([[3, 1, 4, 2], [8, 6, 7, 5]]) if ids is None else ids
    provenance = {"source": "local-real-corpus/train", "seed": 275,
        "nsamples": 2, "seqlen": 4, "fit_tokens": 8,
        "text_sha256": digests.bytes_sha256hex(corpus),
        "fit_ids_sha256": th.token_ids_sha256([ids]),
        "sampler": "windowed-fixture", "model": "synthetic-current-model"}
    provenance.update(changes or {})
    path = tmp_path / "draw.safetensors"
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    text_path = tmp_path / "corpus.txt"
    text_path.write_bytes(corpus)
    sha = digests.bytes_sha256hex(path.read_bytes())
    options = ["--calibration-input", str(path), "--calibration-input-sha256", sha,
               "--calibration-corpus", str(text_path), "--nsamples", "2", "--seqlen", "4"]
    return ids, provenance, path, text_path, options


def _forbid_sampler(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("immutable intake invoked the dataset/tokenizer sampler")
    monkeypatch.setattr(campaign, "_calibration_tokens", forbidden)


def test_exact_draw_real_text_checkpoint_resume_and_receipt(monkeypatch, tmp_path):
    tool, checkpoint, argv, model, inputs = _main_fixture(monkeypatch, tmp_path, priced=True)
    model.config = type("Config", (), {"vocab_size": 32})()
    ids, provenance, _path, _text, options = _saved_draw(tmp_path)
    _forbid_sampler(monkeypatch)
    seen = []
    def collect(_model, _targets, tokens, *_args, **kwargs):
        seen.extend(tokens)
        return ({UNIT: inputs["rows"]}, {}, {UNIT: 8}, {UNIT: inputs["max_abs"]})
    monkeypatch.setattr(tool, "_collect_activations", collect)
    assert tool.main([*argv, *options, "--seed", "19"]) == 0
    assert len(seen) == 2
    assert all(list(batch.shape) == [1, 4] for batch in seen)
    assert torch.equal(torch.cat(seen), ids)
    assert seen[0].untyped_storage().data_ptr() == seen[1].untyped_storage().data_ptr()
    checkpoint_identity = json.loads(checkpoint.read_text())["identity"]
    assert not {"calibration_input", "calibration_input_sha256", "calibration_corpus",
                "calibration_input_receipt"} & checkpoint_identity["settings"].keys()
    identity = checkpoint_identity["calibration"]
    assert identity["text_sha256"] == provenance["text_sha256"]
    assert identity["fit_ids_sha256"] == provenance["fit_ids_sha256"]
    assert identity["seed"] == 275  # --seed is not the saved draw's seed.
    assert identity["source"] == provenance["source"]
    receipt = identity["calibration_input"]
    assert receipt["artifact_sha256"] == options[3]
    assert receipt["provenance"] == provenance
    payload = read_pickle(tmp_path / "cost.pkl")
    assert payload["provenance"]["hessian"]["calibration_identity"] == identity
    def no_encode(**kwargs):
        pytest.fail("same saved draw re-encoded on resume")
    monkeypatch.setattr(tool, "_measure_anchor", no_encode)
    assert tool.main([*argv, *options, "--seed", "19"]) == 0
    _path.rename(tmp_path / "renamed-draw.safetensors")
    _text.rename(tmp_path / "renamed-corpus.txt")
    options[1] = str(tmp_path / "renamed-draw.safetensors")
    options[5] = str(tmp_path / "renamed-corpus.txt")
    assert tool.main([*argv, *options, "--seed", "19"]) == 0
    with pytest.raises(RuntimeError, match="settings.seed"):
        tool.main([*argv, *options, "--seed", "20"])


@pytest.mark.parametrize("supplied", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
def test_partial_options_refuse_before_model(monkeypatch, tmp_path, capsys, supplied):
    tool, _checkpoint, argv, _model, _inputs = _main_fixture(monkeypatch, tmp_path)
    def no_model(*args, **kwargs):
        pytest.fail("partial options reached model work")
    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_model)
    pairs = [("--calibration-input", "draw"), ("--calibration-input-sha256", "a" * 64),
             ("--calibration-corpus", "text")]
    with pytest.raises(SystemExit) as error:
        tool.main([*argv, *(word for i in supplied for word in pairs[i])])
    assert error.value.code == 2
    assert "go together" in capsys.readouterr().err


@pytest.mark.parametrize("damage,match", [
    ("digest-grammar", "SHA256"), ("digest", "SHA256 mismatch"),
    ("missing-input", "draw.safetensors"), ("dtype", "dtype/shape"),
    ("shape", "dtype/shape"), ("negative", "identity domain"),
    ("overflow", "identity domain"), ("draw", "draw provenance"),
    ("provenance", "draw provenance"), ("text-hash", "corpus.*SHA256"),
    ("text-hash-grammar", "corpus.*SHA256"), ("missing-corpus", "corpus.txt"),
    ("corpus", "corpus.*SHA256"), ("utf8", "utf-8"),
    ("seed", "provenance.*seed"), ("source", "provenance.*source"),
    ("model", "provenance.*model"), ("empty-corpus", "real text"),
    ("missing-provenance", "calibration_provenance"),
    ("missing-text-hash", "corpus.*SHA256"),
])
def test_damaged_input_refuses_before_model(monkeypatch, tmp_path, damage, match):
    tool, _checkpoint, argv, _model, _inputs = _main_fixture(monkeypatch, tmp_path)
    ids = torch.tensor([[3, 1, 4, 2], [8, 6, 7, 5]])
    changes = {}
    if damage == "dtype":
        ids = ids.to(torch.int32)
    if damage == "shape":
        ids = ids.reshape(1, 8)
    if damage == "negative":
        ids[0, 0] = -1
    if damage == "overflow":
        ids[0, 0] = 2**31
    changes.update({
        "draw": {"fit_ids_sha256": "0" * 64}, "provenance": {"nsamples": 3},
        "text-hash": {"text_sha256": "0" * 64},
        "text-hash-grammar": {"text_sha256": "not-a-sha"},
        "seed": {"seed": True}, "source": {"source": ""},
        "model": {"model": "another-model"},
    }.get(damage, {}))
    _, _, path, text_path, options = _saved_draw(tmp_path, ids=ids, changes=changes,
        corpus={"utf8": b"\xff", "empty-corpus": b""}.get(damage, b"actual corpus"))
    if damage == "missing-provenance":
        save_file({"calibration_ids": ids}, str(path))
        options[3] = digests.bytes_sha256hex(path.read_bytes())
    if damage == "missing-text-hash":
        _ids, provenance, _, _, _ = _saved_draw(tmp_path, ids=ids, changes=changes)
        del provenance["text_sha256"]
        save_file({"calibration_ids": ids}, str(path),
                  metadata={"calibration_provenance": json.dumps(provenance)})
        options[3] = digests.bytes_sha256hex(path.read_bytes())
    if damage == "digest-grammar":
        options[3] = "not-a-sha"
    if damage == "digest":
        options[3] = "0" * 64
    if damage == "missing-input":
        path.unlink()
    if damage == "missing-corpus":
        text_path.unlink()
    if damage == "corpus":
        text_path.write_bytes(b"replaced")
    def no_model(*args, **kwargs):
        pytest.fail("damaged input reached model work")
    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_model)
    _forbid_sampler(monkeypatch)
    with pytest.raises((ValueError, OSError), match=match):
        tool.main([*argv, *options])


def test_model_vocabulary_refuses_before_capture(monkeypatch, tmp_path):
    tool, _, argv, model, _ = _main_fixture(monkeypatch, tmp_path)
    model.config = type("Config", (), {"vocab_size": 8})()
    *_, options = _saved_draw(tmp_path)
    _forbid_sampler(monkeypatch)
    def no_capture(*args, **kwargs):
        pytest.fail("out-of-vocabulary draw reached capture")
    monkeypatch.setattr(tool, "_collect_activations", no_capture)
    with pytest.raises(ValueError, match="vocab"):
        tool.main([*argv, *options])


class _WrapperConfig:
    """A multimodal wrapper config shaped like GLM-5.3's Glm5NextConfig (#1913): no top-level
    vocab_size; the decoder's vocabulary lives on its text sub-config."""

    _attn_implementation = "eager"

    def __init__(self, vocab_size):
        self.text_config = SimpleNamespace(vocab_size=vocab_size)

    def __getattr__(self, name):
        raise AttributeError(f"'_WrapperConfig' object has no attribute {name!r}")

    def get_text_config(self, decoder=None, encoder=None):
        return self.text_config


def test_wrapper_config_vocabulary_comes_from_its_text_config(monkeypatch, tmp_path):
    tool, _, argv, model, inputs = _main_fixture(monkeypatch, tmp_path, priced=True)
    model.config = _WrapperConfig(32)
    ids, *_, options = _saved_draw(tmp_path)
    _forbid_sampler(monkeypatch)
    seen = []
    def collect(_model, _targets, tokens, *_args, **kwargs):
        seen.extend(tokens)
        return ({UNIT: inputs["rows"]}, {}, {UNIT: 8}, {UNIT: inputs["max_abs"]})
    monkeypatch.setattr(tool, "_collect_activations", collect)
    assert tool.main([*argv, *options]) == 0
    assert torch.equal(torch.cat(seen), ids)


def test_wrapper_config_text_vocabulary_still_refuses_out_of_range_ids(monkeypatch, tmp_path):
    tool, _, argv, model, _ = _main_fixture(monkeypatch, tmp_path)
    model.config = _WrapperConfig(8)
    *_, options = _saved_draw(tmp_path)
    _forbid_sampler(monkeypatch)
    def no_capture(*args, **kwargs):
        pytest.fail("out-of-vocabulary draw reached capture")
    monkeypatch.setattr(tool, "_collect_activations", no_capture)
    with pytest.raises(ValueError, match="vocab"):
        tool.main([*argv, *options])


def test_default_sampler_arguments_unchanged(monkeypatch, tmp_path):
    tool, _, argv, _, inputs = _main_fixture(monkeypatch, tmp_path)
    calls = []
    def sampler(*args):
        calls.append(args)
        return inputs["tokens"], inputs["text"]
    monkeypatch.setattr(tool, "_calibration_tokens", sampler)
    assert tool.main(argv) == tool.EXIT_EMPTY_MENU
    assert calls == [("synthetic-current-model", 8, 512, 0)]


def test_absent_intake_keeps_historical_checkpoint_settings_identity():
    common = {"weights": {}, "acts": {}, "hessians": {}, "menus": {},
              "calibration_identity": th.calibration_identity("real corpus", [], fit_tokens=0),
              "serving_scope": None, "static_scales": {}, "static_scale_policy": "fixture"}
    legacy = campaign._campaign_checkpoint_identity(args=SimpleNamespace(seed=0), **common)
    current = campaign._campaign_checkpoint_identity(args=SimpleNamespace(seed=0,
        calibration_input=None, calibration_input_sha256=None,
        calibration_corpus=None, calibration_input_receipt=None), **common)
    assert current == legacy


def test_corpus_staged_reader_never_rereads_pool(monkeypatch, tmp_path):
    from prismaquant import calibration_data, staged_tier_policy, staged_whole_file
    raw = b"real\r\ncorpus \xc3\xa9"
    digest = digests.bytes_sha256hex(raw)
    calls = []
    monkeypatch.setattr(staged_tier_policy, "policy_is_active", lambda: True)
    def read(path, sha, *, label):
        calls.append((path, sha, label))
        return raw
    monkeypatch.setattr(staged_whole_file, "read_staged_whole_file", read)
    def no_pool(path):
        pytest.fail("staged corpus reread the pool")
    monkeypatch.setattr(Path, "read_bytes", no_pool)
    reader = getattr(calibration_data, "load_calibration_corpus", None)
    assert callable(reader), "shared calibration corpus intake is missing"
    assert reader(tmp_path / "pool", expected_sha256=digest) == raw.decode("utf-8")
    assert calls == [(tmp_path / "pool", digest, "calibration-corpus")]


def test_census_and_capture_reuse_keep_saved_draw_identity(monkeypatch, tmp_path):
    import prismaquant
    from prismaquant import tessera_calibration_cache as store

    tool, _checkpoint, argv, model, inputs = _main_fixture(monkeypatch, tmp_path)
    model.config = type("Config", (), {"vocab_size": 32, "_attn_implementation": "eager"})()
    ids, provenance, _path, _text, options = _saved_draw(tmp_path)
    _forbid_sampler(monkeypatch)
    monkeypatch.setattr(prismaquant, "pretrained_initialization_contract", lambda model: {"fixture": True})
    monkeypatch.setattr(tool, "_collect_activations", lambda *args, **kwargs: (
        {}, {}, {UNIT: ids.numel()}, {UNIT: inputs["max_abs"]}))
    census_path = tmp_path / "census.json"
    census_options = [*argv, *options, "--seed", "19", "--attention-implementation", "eager"]
    assert tool.main([*census_options, "--census-out", str(census_path)]) == 0
    census = json.loads(census_path.read_text())
    assert census["seed"] == 275
    assert census["text_sha256"] == provenance["text_sha256"]
    assert census["fit_ids_sha256"] == provenance["fit_ids_sha256"]
    assert census["calibration_input"]["provenance"] == provenance
    seen = []
    def identity(_path, *, calibration, **kwargs):
        seen.append(calibration)
        # Every real identity names its units; the reuse row prefetches exactly those.
        return {"fixture": "capture", "calibration": calibration, "units": {UNIT: [1, 1]}}
    monkeypatch.setattr(store, "capture_identity", identity)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    (capture_dir / "capture_manifest.json").write_text("{}")
    # The campaign reads the completed manifest's scope before it asks for the
    # expected identity. This wiring test stubs the store, so its manifest is
    # a stub full-scope one; the contract itself is covered with real manifests.
    monkeypatch.setattr(store, "require_capture_contract",
                        lambda path, expected_sha256=None: {"identity": {}, "entries": {}})
    def prefetch(_path, *, expected_identity, **kwargs):
        assert expected_identity["calibration"] == seen[0]
        return {}, {"fixture": "reused"}
    monkeypatch.setattr(store, "prefetch_capture", prefetch)
    def no_capture(*args, **kwargs):
        pytest.fail("complete capture reuse repeated calibration")
    monkeypatch.setattr(tool, "_collect_activations", no_capture)
    assert tool.main([*census_options, "--seed", "20", "--hessian", "require",
        "--calibration-census", str(census_path),
        "--capture-calibration-out", str(capture_dir)]) == 0
    assert seen[0]["text_sha256"] == census["text_sha256"]
    assert seen[0]["fit_ids_sha256"] == census["fit_ids_sha256"]
    assert seen[0]["calibration_input"] == census["calibration_input"]
    assert seen[0]["seed"] == 275
    bad = {**census, "calibration_input": {**census["calibration_input"], "artifact_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="input receipt differs"):
        tool.require_census_draw(bad, seen[0], where="fixture")
    bad = {**census, "text_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="different draws"):
        tool.require_census_draw(bad, seen[0], where="fixture")
