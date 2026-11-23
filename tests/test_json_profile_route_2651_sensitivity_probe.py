import json
from pathlib import Path

from prismaquant import digests, sensitivity_probe


def test_listed_calls_use_insertion_ascii_indent2_lax(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    text_config = {"z": "é", "a": float("inf")}
    full_config = {**text_config, "vision_config": {}}
    (source / "config.json").write_text(json.dumps(full_config), encoding="utf-8")
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))

    real_text = digests.JsonProfile.text
    calls = []

    def observe(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT2_LAX:
            calls.append(self)
            return prefix + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe)
    for prefix in ("", " "):
        for stage, config in ((sensitivity_probe.stage_text_only, text_config),
                              (sensitivity_probe.stage_multimodal, full_config)):
            calls.clear()
            staged = Path(stage(str(source)))
            expected = (prefix + json.dumps(config, indent=2)).encode("utf-8")
            assert (staged / "config.json").read_bytes() == expected, stage.__name__
            assert len(calls) == 1, f"{stage.__name__} must use the shared JSON profile."
            assert capsys.readouterr().out == "", stage.__name__
