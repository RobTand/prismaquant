"""The default pipeline spec prints through the bound indent 2 JSON profile."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prismaquant import pipeline
from prismaquant.digests import JsonProfile

PROFILE_NAME = "direct-ascii-indent2-lax"

RICH = {
    "unicode": "snowman \u2603 / caf\u00e9",
    "nested": [None, True, -1, 1.5, [], {}],
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative_infinity": -float("inf"),
}

RICH_STDOUT = (
    b"{\n"
    b'  "infinity": Infinity,\n'
    b'  "nan": NaN,\n'
    b'  "negative_infinity": -Infinity,\n'
    b'  "nested": [\n'
    b"    null,\n"
    b"    true,\n"
    b"    -1,\n"
    b"    1.5,\n"
    b"    [],\n"
    b"    {}\n"
    b"  ],\n"
    b'  "unicode": "snowman \\u2603 / caf\\u00e9"\n'
    b"}\n"
)

CASES = {
    "empty": ({}, b"{}\n", None),
    "rich": (RICH, RICH_STDOUT, None),
    "mixed": (
        {1: "number", "x": "text"},
        b"",
        "'<' not supported between instances of 'str' and 'int'",
    ),
    "unsupported": (
        {"bad": object()},
        b"",
        "Object of type object is not JSON serializable",
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_pipeline_publication_profile_calls(case, tmp_path, monkeypatch, capsysbinary):
    value, expected_stdout, refusal = CASES[case]
    monkeypatch.chdir(tmp_path)
    spec = SimpleNamespace(
        validate=lambda: SimpleNamespace(errors=[], warnings=[], ok=True),
        to_dict=lambda: value,
    )
    monkeypatch.setattr(
        pipeline, "production_pipeline_spec_from_config", lambda **_kwargs: spec
    )
    calls = []
    original = JsonProfile.text

    def recording_text(self, item):
        calls.append((self.name, item))
        return original(self, item)

    monkeypatch.setattr(JsonProfile, "text", recording_text)

    if refusal is None:
        assert pipeline.main([]) == 0
    else:
        with pytest.raises(TypeError) as raised:
            pipeline.main([])
        assert str(raised.value) == refusal
        assert raised.value.__cause__ is None

    mine = [item for name, item in calls if name == PROFILE_NAME]
    assert len(mine) == 1
    assert mine[0] is value
    captured = capsysbinary.readouterr()
    assert captured.out == expected_stdout
    assert captured.err == b""
    assert list(tmp_path.iterdir()) == []
