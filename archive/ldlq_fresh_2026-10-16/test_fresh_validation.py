"""Focused CPU tests for the fresh-text validation orchestration helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fresh_validation as fresh_module
from fresh_validation import (
    SAMPLES,
    build_disjoint_corpus,
    parse_heldout_report,
    validation_verdict,
    write_capture_manifest,
)


def test_build_disjoint_corpus_selects_next_records_and_records_hashes(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [{"__manifest__": {"ignored": True}}] + [
        {"text": f"record-{index}"} for index in range(40)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    corpus = tmp_path / "fresh.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_disjoint_corpus(source, corpus, manifest_path)

    fresh = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]
    assert fresh == [f"record-{index}" for index in range(16, 32)]
    assert manifest["disjoint"] is True
    assert manifest["exact_text_hash_overlap"] == []
    assert [row["source_line"] for row in manifest["production_records"]] == list(range(2, 18))
    assert [row["source_line"] for row in manifest["fresh_records"]] == list(range(18, 34))


def test_checkpoint_tokenizer_audit_recomputes_distinct_window_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, **_kwargs):
            values = [1 + (ord(char) % 251) for char in text]
            return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    fake = FakeTokenizer()
    source = tmp_path / "source.jsonl"
    rows = [{"__manifest__": {"ignored": True}}] + [
        {"text": f"record-{index}-" + chr(65 + index % 26) * 700}
        for index in range(40)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    corpus = tmp_path / "fresh.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_disjoint_corpus(source, corpus, manifest_path)

    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration

    expected = calibration_data_hash(
        load_calibration(fake, str(source), 16, 512, calib_seed=42)
    )
    monkeypatch.setattr(fresh_module, "PRODUCTION_CALIB_HASH", expected)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: fake,
    )
    model = tmp_path / "model"
    model.mkdir()

    audited = fresh_module.audit_corpus_with_checkpoint_tokenizer(
        manifest, source, corpus, model
    )

    contract = audited["loader_contract"]
    assert contract["production_calibration_data_hash_recomputed"] == expected
    assert contract["fresh_calibration_data_hash"] != expected
    assert min(row["checkpoint_token_count"] for row in audited["fresh_records"]) > 512


def test_parse_heldout_report_averages_three_splits_per_cell(tmp_path: Path):
    lines = ["## CAL32 decision table", ""]
    for slug in (
        "layers.40.attn.wq_b",
        "layers.40.experts.81.up_proj",
        "layers.20.experts.63.up_proj",
    ):
        for rung in (12, 15, 18):
            for split, reduction in enumerate((60.0, 70.0, 80.0)):
                lines.append(
                    f"| `{slug}` | {rung} | {split} | 0.1 | 90.00% | 0.3 | "
                    f"{reduction:.2f}% | 20.0 | 77.0% | verdict |"
                )
    lines.extend(["", "## next"])
    report = tmp_path / "REPORT.md"
    report.write_text("\n".join(lines))

    parsed = parse_heldout_report(report)

    assert len(parsed) == 9
    assert all(value == pytest.approx(0.70) for value in parsed.values())


def test_capture_manifest_checks_hash_and_records_target_files(tmp_path: Path):
    capture = tmp_path / "capture"
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    for layer in (20, 40):
        layer_dir = capture / f"l{layer}"
        (layer_dir / "logs").mkdir(parents=True)
        with (layer_dir / "probe.pkl").open("wb") as stream:
            pickle.dump(
                {
                    "meta": {
                        "dataset": "fresh.jsonl",
                        "calib_hash": "fresh-hash",
                        "nsamples": 16,
                        "seqlen": 512,
                        "activation_rows_limit": 64,
                        "linear_include": rf"layers\.{layer}\.",
                    }
                },
                stream,
            )
        (layer_dir / "logs" / "probe.log").write_text(
            "phase-1 forward: 10.5s\n"
            f"phase-3 reverse sweep [42->{layer}]: 2.5s\n"
        )
    for spec in SAMPLES.values():
        torch.save(
            {
                "inputs": torch.zeros(64, 4),
                "name": spec["live_name"],
                "row_indices": torch.arange(64),
            },
            act_dir / spec["activation_file"],
        )

    output = tmp_path / "capture_manifest.json"
    manifest = write_capture_manifest(
        capture,
        act_dir,
        {"loader_contract": {"fresh_calibration_data_hash": "fresh-hash"}},
        output,
    )

    assert manifest["measured_probe_hot_seconds"] == pytest.approx(26.0)
    assert manifest["production_activation_caches_written"] is False
    assert set(manifest["targets"]) == set(SAMPLES)
    assert all(record["shape"] == [64, 4] for record in manifest["targets"].values())


@pytest.mark.parametrize(
    "fresh,expected",
    [
        (0.51, "VALIDATED-PENDING-SERVED"),
        (0.50, "PARTIAL"),
        (0.10, "PARTIAL"),
        (0.099, "DISTRIBUTION-FRAGILE"),
    ],
)
def test_validation_verdict_uses_strict_fixed_thresholds(fresh: float, expected: str):
    verdict, retention = validation_verdict(1.0, fresh)
    assert verdict == expected
    assert retention == pytest.approx(fresh)


def test_validation_verdict_rejects_nonpositive_insample_gain():
    with pytest.raises(ValueError):
        validation_verdict(0.0, 0.5)
