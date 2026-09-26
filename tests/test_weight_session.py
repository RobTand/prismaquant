import pytest
import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.weight_session import WeightSession


class _ModelWithBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.model.proj(x)


def test_weight_session_accepts_language_model_alias_for_staged_body():
    model = _ModelWithBody().eval()
    session = WeightSession(model)

    session.initialize({"model.language_model.proj": "BF16"}, units=[])

    assert session.current_assignment() == {
        "model.language_model.proj": "BF16"
    }
    assert session.diagnostics()["n_bf16_snapshots"] == 0
    assert session.format_weight("model.language_model.proj", "BF16") is not None
    assert session.diagnostics()["n_bf16_snapshots"] == 1


def test_weight_session_rejects_strict_production_cache_miss():
    model = _ModelWithBody().eval()
    cache = ProductionWeightCache(weights={}, levers={})
    session = WeightSession(model, production_weight_cache=cache)

    with pytest.raises(RuntimeError, match="production_weight_cache miss"):
        session.initialize({"model.proj": "NVFP4"}, units=[])


def test_weight_session_rejects_strict_mxfp8_cache_miss():
    model = _ModelWithBody().eval()
    cache = ProductionWeightCache(weights={}, levers={})
    session = WeightSession(model, production_weight_cache=cache)

    with pytest.raises(RuntimeError, match="production_weight_cache miss"):
        session.initialize({"model.proj": "MXFP8_E4M3"}, units=[])


def test_weight_session_allows_mxfp8_rtn_fallback_when_not_strict():
    model = _ModelWithBody().eval()
    cache = ProductionWeightCache(weights={}, levers={})
    session = WeightSession(
        model,
        production_weight_cache=cache,
        strict_production_cache=False,
    )

    session.initialize({"model.proj": "MXFP8_E4M3"}, units=[])

    diag = session.diagnostics()
    assert diag["n_rtn_fallbacks"] == 1
    assert diag["n_cache_misses"] == 1


def test_weight_session_refuses_a_retired_codebook_rung():
    # The retired codebook lane, archived 2026-09-25, #1304: the RTN
    # fallback re-raises the refusal instead of returning None.
    model = _ModelWithBody().eval()
    session = WeightSession(model, strict_production_cache=False)

    with pytest.raises(fr.RetiredFormatError, match="gridbook_lane"):
        session.initialize({"model.proj": "NVFP4_CB_K16"}, units=[])


def test_initialize_does_not_record_format_when_materialization_fails(monkeypatch):
    model = _ModelWithBody().eval()
    session = WeightSession(model, strict_production_cache=False)
    monkeypatch.setattr(session, "_format_weight", lambda _qname, _fmt: None)

    session.initialize({"model.proj": "NVFP4"}, units=[])

    assert session.current_assignment().get("model.proj", "BF16") == "BF16"
    assert session.stage_format("model.proj", "NVFP4") is None


def test_weight_session_reuses_existing_spilled_snapshot(tmp_path):
    model = _ModelWithBody().eval()
    model.model.proj.weight.data.fill_(1.0)
    first = WeightSession(model, snapshot_dir=str(tmp_path))
    saved = first.format_weight("model.proj", "BF16")

    model.model.proj.weight.data.zero_()
    second = WeightSession(model, snapshot_dir=str(tmp_path))
    reused = second.format_weight("model.proj", "BF16")

    assert saved is not None
    assert reused is not None
    torch.testing.assert_close(reused, torch.ones_like(reused))
    assert second.diagnostics()["n_bf16_snapshots"] == 1


def test_stale_spill_file_with_wrong_shape_raises_on_record(tmp_path):
    """Cross-run snapshot_dir reuse: a pre-existing __bf16src.pt whose
    shape disagrees with the live param must raise when the recorded
    path would otherwise trust it blindly."""
    torch.save(torch.zeros(3, 3), tmp_path / "model_proj__bf16src.pt")
    model = _ModelWithBody().eval()
    rendered = torch.full((64, 64), 0.125)
    cache = ProductionWeightCache(
        weights={("model.proj", "NVFP4"): rendered},
        levers={},
    )
    session = WeightSession(
        model, production_weight_cache=cache, snapshot_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="snapshot_dir"):
        session.initialize({"model.proj": "NVFP4"}, units=[])


def test_stale_spill_file_with_wrong_shape_raises_on_reload(tmp_path):
    """The _spilled-hit reload path must validate too — a recorded entry
    whose backing file is stale would otherwise restore wrong-model
    weights at revert time."""
    torch.save(torch.zeros(3, 3), tmp_path / "model_proj__bf16src.pt")
    model = _ModelWithBody().eval()
    session = WeightSession(model, snapshot_dir=str(tmp_path))
    # Pin the _spilled-hit branch directly (the record path is covered
    # above and would already have raised).
    session._spilled["model.proj"] = "model_proj__bf16src.pt"

    with pytest.raises(RuntimeError, match="stale"):
        session.format_weight("model.proj", "BF16")


def test_matching_spill_file_still_records_and_reloads(tmp_path):
    model = _ModelWithBody().eval()
    model.model.proj.weight.data.fill_(1.0)
    torch.save(
        model.model.proj.weight.detach().cpu().clone(),
        tmp_path / "model_proj__bf16src.pt",
    )
    rendered = torch.full((64, 64), 0.125)
    cache = ProductionWeightCache(
        weights={("model.proj", "NVFP4"): rendered},
        levers={},
    )
    session = WeightSession(
        model, production_weight_cache=cache, snapshot_dir=str(tmp_path))

    session.initialize({"model.proj": "NVFP4"}, units=[])
    torch.testing.assert_close(model.model.proj.weight, rendered)
    session.apply_assignment({"model.proj": "BF16"})
    torch.testing.assert_close(
        model.model.proj.weight, torch.ones(64, 64))


def test_apply_assignment_records_bf16_before_overwrite(tmp_path):
    model = _ModelWithBody().eval()
    original = model.model.proj.weight.detach().clone()
    rendered = torch.full_like(original, 0.125)
    cache = ProductionWeightCache(
        weights={("model.proj", "NVFP4"): rendered.clone()},
        levers={},
    )
    session = WeightSession(
        model,
        production_weight_cache=cache,
        snapshot_dir=str(tmp_path),
    )

    session.initialize({"model.proj": "BF16"}, units=[])
    session.apply_assignment({"model.proj": "NVFP4"})
    torch.testing.assert_close(model.model.proj.weight, rendered)
    assert session.diagnostics()["n_bf16_snapshots"] == 1

    session.apply_assignment({"model.proj": "BF16"})
    torch.testing.assert_close(model.model.proj.weight, original)
