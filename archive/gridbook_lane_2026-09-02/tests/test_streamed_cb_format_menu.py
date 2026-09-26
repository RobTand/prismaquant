"""CPU contracts for the streamed CB format-menu render path.

These tests deliberately use the real menu traversal, render-score builder,
CB render identity, and cache retention policy.  Only the expensive CB QDQ is
stubbed: the replacement renderer is deterministic and format-sensitive, so
the tests can distinguish a complete Cartesian menu from an assignment-sized
subset without requiring CUDA.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.streaming_production_cache import run_streaming_render


_CB_MENU = ("FP8_CB_K28", "FP8_CB_K43")
_CALIBRATION_HASH = "a" * 64


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # iter_quantizable_tensors intentionally ignores tensors below 1,000
        # parameters.  32 x 32 keeps this fixture small while crossing that
        # production eligibility threshold.
        self.gate_proj = nn.Linear(32, 32, bias=False)
        self.up_proj = nn.Linear(32, 32, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(), _ToyLayer()])
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.linspace(
                    -0.75 + 0.01 * index,
                    0.75 + 0.01 * index,
                    parameter.numel(),
                    dtype=torch.float32,
                ).reshape_as(parameter)
                parameter.copy_(values)
        # A bf16 source makes the canonical transient tensor exactly the same
        # dtype as a materialized cache shard (the production storage rule).
        self.to(dtype=torch.bfloat16)


class _ActivationIndex:
    """Minimal in-memory implementation of the ActivationIndex interface."""

    def __init__(self, qnames: Sequence[str]) -> None:
        base = torch.linspace(-1.0, 1.0, 6 * 32).reshape(6, 32)
        self._inputs = {
            str(qname): (base + 0.025 * index).contiguous()
            for index, qname in enumerate(qnames)
        }

    def __contains__(self, qname: object) -> bool:
        return str(qname) in self._inputs

    def load_with_row_indices(
        self, qname: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs[str(qname)]
        return inputs.clone(), torch.arange(inputs.shape[0])

    def load(self, qname: str) -> torch.Tensor:
        return self._inputs[str(qname)].clone()


def _qnames() -> tuple[str, ...]:
    return tuple(
        f"layers.{layer}.{projection}"
        for layer in range(2)
        for projection in ("gate_proj", "up_proj")
    )


def _fake_cb_renderer(calls: list[tuple[str, str]]):
    offsets = {"FP8_CB_K28": 0.125, "FP8_CB_K43": 0.25}

    def _render(
        weight: torch.Tensor,
        render_format: str,
        *,
        qname: str,
        **_kwargs,
    ) -> torch.Tensor:
        calls.append((str(qname), str(render_format)))
        return (
            weight.detach().to(torch.float32) + offsets[str(render_format)]
        ).to(dtype=weight.dtype).contiguous()

    return _render


def _run_menu(
    model: nn.Module,
    root: Path,
    *,
    formats: Sequence[str] = _CB_MENU,
    retain_rendered: bool,
):
    root.mkdir(parents=True, exist_ok=True)
    qnames = _qnames()
    return run_streaming_render(
        model,
        layers_prefix="layers.",
        num_layers=2,
        render_scope="format-menu",
        render_assignment=None,
        act_index=_ActivationIndex(qnames),
        formats=formats,
        levers={"gptq": False, "weighted_vq": True},
        cache_dir_path=root,
        profile=None,
        skip_tokens=(),
        device=torch.device("cpu"),
        col_weights={qname: torch.ones(32) for qname in qnames},
        cb_serialization_context=CBSerializationContext.production(
            scale_sweep=True,
            ldlq=False,
            minchain=False,
            encode_tier="balanced",
            codebook_source="lattice",
        ),
        retain_rendered=retain_rendered,
        calibration_hash=_CALIBRATION_HASH,
        resume=False,
        progress=False,
    )


def _score_rows(cache) -> tuple[tuple[object, ...], ...]:
    records = cache.metadata["render_scores"]["records"]
    return tuple(
        (
            key,
            record["qname"],
            record["format"],
            record["metric"],
            record["score"],
            record["score_sum"],
            record["normalizer"],
            record["activation_rows"],
        )
        for key, record in sorted(records.items())
    )


def test_streamed_cb_menu_is_complete_and_discards_every_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.streaming_production_cache as streaming

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        streaming, "render_production_weight", _fake_cb_renderer(calls),
    )
    cache_dir = tmp_path / "transient"

    cache = _run_menu(_ToyModel(), cache_dir, retain_rendered=False)

    expected = {
        (qname, fmt) for qname in _qnames() for fmt in _CB_MENU
    }
    assert set(calls) == expected
    assert len(calls) == len(expected)
    assert set(cache.metadata["render_scores"]["records"]) == {
        f"{qname}|{fmt}" for qname, fmt in expected
    }
    assert cache.metadata["render_scope"] == "format-menu"
    assert cache.metadata["calib_hash"] == _CALIBRATION_HASH

    # The complete menu is represented by scores + consumed-artifact
    # attestations, never by retained bf16 weights.  In particular, disk
    # bounding must not be implemented by dropping candidate rows.
    assert cache.weights == {}
    assert not list(cache_dir.glob("*.pt"))
    transient = cache.metadata["transient_render_artifacts"]
    assert transient["entries"] == len(expected)
    assert set(transient["records"]) == {
        f"{qname}|{fmt}" for qname, fmt in expected
    }
    assert len(list(cache_dir.glob("*.pt.identity.json"))) == len(expected)
    for key, artifact in transient["records"].items():
        assert artifact["retained_weight"] is False
        assert artifact["tensor"]["dtype"] == "torch.bfloat16"
        receipt = artifact["consumer_receipt"]
        assert receipt["qname"] == artifact["render_score"]["qname"]
        assert receipt["format"] == artifact["render_score"]["format"]
        assert receipt["tensor"] == artifact["tensor"]
        assert receipt["result"] == transient["consumer_results"][key]


def test_streamed_format_plan_prices_exact_source_class_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.streaming_production_cache as streaming

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        streaming, "render_production_weight", _fake_cb_renderer(calls),
    )
    qnames = _qnames()
    # The lower-source-rate class stops at K28; the higher-rate class reaches
    # K43. The global menu remains complete, but illegal work is never issued.
    format_plan = {
        qname: (("FP8_CB_K28",) if index < 2 else _CB_MENU)
        for index, qname in enumerate(qnames)
    }
    cache_dir = tmp_path / "split"
    cache_dir.mkdir()
    cache = run_streaming_render(
        _ToyModel(),
        layers_prefix="layers.",
        num_layers=2,
        render_scope="format-menu",
        render_assignment=None,
        act_index=_ActivationIndex(qnames),
        formats=_CB_MENU,
        levers={"gptq": False, "weighted_vq": True},
        cache_dir_path=cache_dir,
        profile=None,
        skip_tokens=(),
        device=torch.device("cpu"),
        col_weights={qname: torch.ones(32) for qname in qnames},
        cb_serialization_context=CBSerializationContext.production(
            scale_sweep=True,
            ldlq=False,
            minchain=False,
            encode_tier="balanced",
            codebook_source="lattice",
        ),
        retain_rendered=False,
        calibration_hash=_CALIBRATION_HASH,
        format_plan=format_plan,
        format_plan_identity="b" * 64,
        progress=False,
    )
    expected = {
        (qname, fmt)
        for qname, formats in format_plan.items()
        for fmt in formats
    }
    assert set(calls) == expected
    assert len(calls) == 6
    assert cache.metadata["requested_entries"] == 6
    assert cache.metadata["format_plan_identity_sha256"] == "b" * 64
    assert not list(cache_dir.glob("*.pt"))


def test_streamed_and_materialized_cb_menus_have_identical_cost_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.streaming_production_cache as streaming

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        streaming, "render_production_weight", _fake_cb_renderer(calls),
    )
    model = _ToyModel()

    streamed = _run_menu(
        model, tmp_path / "streamed", retain_rendered=False,
    )
    materialized = _run_menu(
        model, tmp_path / "materialized", retain_rendered=True,
    )

    # Tuple equality intentionally compares the Python float payloads exactly,
    # not with an approximate tolerance: retention is not allowed to alter the
    # scalar table handed to the allocator.
    assert _score_rows(streamed) == _score_rows(materialized)
    assert streamed.metadata["render_scores"] == materialized.metadata[
        "render_scores"
    ]
    assert streamed.weights == {}
    assert set(materialized.weights) == {
        (qname, fmt) for qname in _qnames() for fmt in _CB_MENU
    }
    assert len(list((tmp_path / "materialized").glob("*.pt"))) >= (
        len(_qnames()) * len(_CB_MENU)
    )

    # The downstream allocator table admits the score-only cache only through
    # its transient consumer attestations. Compare the synthesized cost rows,
    # not merely the cache's internal score dictionary.
    from prismaquant.production_render_cost import (
        synthesize_production_render_cost_payload,
    )
    from prismaquant.production_weight_cache import (
        production_cache_cb_render_provenance,
    )

    baseline = {
        "formats": list(_CB_MENU),
        "costs": {
            qname: {
                fmt: {"predicted_dloss": 999.0}
                for fmt in _CB_MENU
            }
            for qname in _qnames()
        },
        "provenance": production_cache_cb_render_provenance(
            materialized,
            require_for_formats=_CB_MENU,
        ),
    }
    streamed_cost = synthesize_production_render_cost_payload(
        streamed,
        baseline,
        formats=_CB_MENU,
        require_render_scores=True,
    )
    materialized_cost = synthesize_production_render_cost_payload(
        materialized,
        baseline,
        formats=_CB_MENU,
        require_render_scores=True,
    )
    assert streamed_cost["costs"] == materialized_cost["costs"]


def test_transient_format_menu_fails_closed_for_non_cb_format(
    tmp_path: Path,
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)CB"):
        _run_menu(
            _ToyModel(),
            tmp_path / "non-cb",
            formats=("FP8_E4M3",),
            retain_rendered=False,
        )


def test_selected_assignment_refuses_rerender_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prismaquant.streaming_production_cache as streaming

    model = _ToyModel()
    cache_dir = tmp_path / "shared"
    first_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        streaming, "render_production_weight", _fake_cb_renderer(first_calls),
    )
    _run_menu(model, cache_dir, retain_rendered=False)

    selected_qname = _qnames()[0]
    selected_format = _CB_MENU[0]
    second_calls: list[tuple[str, str]] = []
    faithful_renderer = _fake_cb_renderer(second_calls)

    def _changed_renderer(
        weight: torch.Tensor,
        render_format: str,
        *,
        qname: str,
        **kwargs,
    ) -> torch.Tensor:
        faithful = faithful_renderer(
            weight, render_format, qname=qname, **kwargs,
        )
        return (faithful.to(torch.float32) + 1.0).to(faithful.dtype)

    monkeypatch.setattr(
        streaming, "render_production_weight", _changed_renderer,
    )

    with pytest.raises(
        RuntimeError,
        match=r"selected CB assignment re-render differs.*refusing publication",
    ):
        run_streaming_render(
            model,
            layers_prefix="layers.",
            num_layers=2,
            render_scope="assignment",
            render_assignment={selected_qname: selected_format},
            act_index=_ActivationIndex(_qnames()),
            formats=_CB_MENU,
            levers={"gptq": False, "weighted_vq": True},
            cache_dir_path=cache_dir,
            profile=None,
            skip_tokens=(),
            device=torch.device("cpu"),
            col_weights={qname: torch.ones(32) for qname in _qnames()},
            cb_serialization_context=CBSerializationContext.production(
                scale_sweep=True,
                ldlq=False,
                minchain=False,
                encode_tier="balanced",
                codebook_source="lattice",
            ),
            calibration_hash=_CALIBRATION_HASH,
            resume=True,
            progress=False,
        )

    assert second_calls == [(selected_qname, selected_format)]
    # The transient sidecar remains, but the mismatching selected bytes never
    # earn publication as a materialized weight shard.
    assert not list(cache_dir.glob("*.pt"))
