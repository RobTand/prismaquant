"""The chain split publishes its receipt through the bound JSON profiles.

The file uses the indent 2 profile. Stdout uses the spaced profile.
The expected bytes are literals. No second encoder makes them.
"""
from __future__ import annotations

import pytest

from prismaquant import stage_a_chain_resume, stage_a_chain_split
from prismaquant.digests import JsonProfile

FILE_PROFILE = "direct-ascii-indent2-lax"
STDOUT_PROFILE = "direct-ascii-spaced-lax"

RICH = {
    "unicode": "snowman \u2603 / caf\u00e9",
    "nested": [None, True, -1, 1.5, [], {}],
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative_infinity": -float("inf"),
}
RICH_FILE = (
    b'{\n'
    b'  "infinity": Infinity,\n'
    b'  "nan": NaN,\n'
    b'  "negative_infinity": -Infinity,\n'
    b'  "nested": [\n'
    b'    null,\n'
    b'    true,\n'
    b'    -1,\n'
    b'    1.5,\n'
    b'    [],\n'
    b'    {}\n'
    b'  ],\n'
    b'  "unicode": "snowman \\u2603 / caf\\u00e9"\n'
    b'}\n'
)
RICH_STDOUT = (
    b'{"infinity": Infinity, "nan": NaN, "negative_infinity": -Infinity, '
    b'"nested": [null, true, -1, 1.5, [], {}], '
    b'"unicode": "snowman \\u2603 / caf\\u00e9"}\n'
)
MIXED_MESSAGE = "'<' not supported between instances of 'str' and 'int'"
OBJECT_MESSAGE = "Object of type object is not JSON serializable"


@pytest.mark.parametrize(
    "value, file_bytes, stdout_bytes, message",
    [
        pytest.param({}, b"{}\n", b"{}\n", None, id="empty"),
        pytest.param(RICH, RICH_FILE, RICH_STDOUT, None, id="rich"),
        pytest.param({1: "number", "x": "text"}, None, None, MIXED_MESSAGE, id="mixed"),
        pytest.param({"bad": object()}, None, None, OBJECT_MESSAGE, id="unsupported"),
    ],
)
def test_split_publication_profile_calls(
        tmp_path, monkeypatch, capsysbinary, value, file_bytes, stdout_bytes, message):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        stage_a_chain_resume, "read_chain_state",
        lambda space: {"run_identity": {"n_probes": 1}, "n_batches": 1})
    monkeypatch.setattr(
        stage_a_chain_split, "join_split_checkpoint",
        lambda space, boundary, **_kwargs: {"boundary": boundary})
    monkeypatch.setattr(stage_a_chain_split, "join_receipt", lambda space, record: value)
    original = JsonProfile.text
    calls = []

    def recording(self, candidate):
        calls.append((self.name, candidate))
        return original(self, candidate)

    monkeypatch.setattr(JsonProfile, "text", recording)
    argv = ["--output-root", "work", "--boundary", "2", "--receipt", "receipt.json"]

    if message is None:
        assert stage_a_chain_split.main(argv) == 0
        expected_names = [FILE_PROFILE, STDOUT_PROFILE]
    else:
        with pytest.raises(TypeError) as raised:
            stage_a_chain_split.main(argv)
        assert type(raised.value) is TypeError
        assert str(raised.value) == message
        assert raised.value.__cause__ is None
        expected_names = [FILE_PROFILE]

    in_scope = [call for call in calls if call[0] in (FILE_PROFILE, STDOUT_PROFILE)]
    assert [name for name, _ in in_scope] == expected_names
    assert all(candidate is value for _, candidate in in_scope)

    captured = capsysbinary.readouterr()
    receipt = tmp_path / "receipt.json"
    if message is None:
        assert receipt.read_bytes() == file_bytes
        assert captured.out == stdout_bytes
    else:
        assert not receipt.exists()
        assert captured.out == b""
    assert captured.err == b""
