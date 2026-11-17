"""The stock campaign sidecar and inventory JSON use the bound indent 2 profile.

The expected bytes are literals. The fixtures replace only the upstream seams:
the finished campaign payload and the inventory checks.
"""
from __future__ import annotations

import json
import pickle

import pytest

from prismaquant import digests
from prismaquant import glm53_stock_reprice as reprice

PROFILE_NAME = "direct-ascii-indent2-lax"

_VALUES = {
    "empty": lambda: {},
    "rich": lambda: {
        "unicode": "snowman \u2603 / caf\u00e9",
        "nested": [None, True, -1, 1.5, [], {}],
        "nan": float("nan"),
        "infinity": float("inf"),
        "negative_infinity": -float("inf"),
    },
    "mixed": lambda: {1: "number", "x": "text"},
    "unsupported": lambda: {"bad": object()},
}

_REFUSALS = {
    "mixed": "'<' not supported between instances of 'str' and 'int'",
    "unsupported": "Object of type object is not JSON serializable",
}

_CAMPAIGN_MESSAGE = (
    b"[glm53-stock] campaign cost payload -> out/cost.pkl "
    b"(anchored 1, empirical 2, pinned 3, cells 6; "
    b"sidecar cost.provenance.json)\n"
)

_CAMPAIGN_HEAD = [
    b"{",
]
_CAMPAIGN_TAIL = [
    b'  "ladder_refusals": 0,',
    b'  "merge_report": {',
    b'    "anchored_units": 1,',
    b'    "empirical_units": 2,',
    b'    "pinned_units": 3,',
    b'    "total_cells": 6',
    b"  },",
    b'  "output": "out/cost.pkl",',
    b'  "schema": "fixture",',
    b'  "unpriced_probe_units": []',
    b"}",
]

_RICH_LINES = [
    b'"infinity": Infinity,',
    b'"nan": NaN,',
    b'"negative_infinity": -Infinity,',
    b'"nested": [',
    b"  null,",
    b"  true,",
    b"  -1,",
    b"  1.5,",
    b"  [],",
    b"  {}",
    b"],",
    br'"unicode": "snowman \u2603 / caf\u00e9"',
]


def _indented(lines, spaces):
    return [b" " * spaces + line for line in lines]


_CAMPAIGN_BYTES = {
    "empty": b"\n".join(
        _CAMPAIGN_HEAD + [b'  "campaign_identity": {},'] + _CAMPAIGN_TAIL
    ) + b"\n",
    "rich": b"\n".join(
        _CAMPAIGN_HEAD
        + [b'  "campaign_identity": {']
        + _indented(_RICH_LINES, 4)
        + [b"  },"]
        + _CAMPAIGN_TAIL
    ) + b"\n",
}

_INVENTORY_BYTES = {
    "empty": b"\n".join([
        b"{",
        b'  "checks": [',
        b"    {",
        b'      "detail": {},',
        b'      "name": "fixture",',
        b'      "path": "checkpoint",',
        b'      "status": "PASS"',
        b"    }",
        b"  ]",
        b"}",
    ]) + b"\n",
    "rich": b"\n".join(
        [b"{", b'  "checks": [', b"    {", b'      "detail": {']
        + _indented(_RICH_LINES, 8)
        + [
            b"      },",
            b'      "name": "fixture",',
            b'      "path": "checkpoint",',
            b'      "status": "PASS"',
            b"    }",
            b"  ]",
            b"}",
        ]
    ) + b"\n",
}


def _campaign_argv(monkeypatch, value):
    payload = {
        "schema": "fixture",
        "provenance": {
            "merge_report": {
                "anchored_units": 1,
                "empirical_units": 2,
                "pinned_units": 3,
                "total_cells": 6,
            },
            "ladder_refusals": [],
            "unpriced_probe_units": [],
            "campaign_identity": value,
        },
    }
    monkeypatch.setattr(
        reprice, "run_campaign_from_artifacts", lambda **_: payload,
    )
    with open("probe.json", "w") as handle:
        json.dump({"schema": reprice.PROBE_CENSUS_SCHEMA}, handle)
    with open("checkpoint.json", "w") as handle:
        json.dump({"schema": reprice.CHECKPOINT_CENSUS_SCHEMA}, handle)
    for name in ("harvest.pkl", "expert.pkl"):
        with open(name, "wb") as handle:
            pickle.dump({}, handle)
    return [
        "campaign",
        "--probe-census", "probe.json",
        "--checkpoint-census", "checkpoint.json",
        "--harvest", "harvest.pkl",
        "--expert-empirical", "expert.pkl",
        "--output", "out/cost.pkl",
    ]


def _inventory_argv(monkeypatch, value):
    check = reprice.Check(
        name="fixture", status="PASS", detail=value, path="checkpoint",
    )
    monkeypatch.setattr(reprice, "inventory_checks", lambda **_: [check])
    with open("probe.json", "w") as handle:
        json.dump({"schema": reprice.PROBE_CENSUS_SCHEMA}, handle)
    with open("checkpoint.json", "w") as handle:
        json.dump({"schema": reprice.CHECKPOINT_CENSUS_SCHEMA}, handle)
    return [
        "inventory",
        "--model", "model",
        "--probe-census", "probe.json",
        "--checkpoint-census", "checkpoint.json",
        "--expert-empirical", "expert.pkl",
        "--checkpoint-dir", "checkpoint-dir",
        "--json",
    ]


@pytest.mark.parametrize("action", ["campaign", "inventory"])
@pytest.mark.parametrize("case", ["empty", "rich", "mixed", "unsupported"])
def test_stock_publication_profile_calls(
    tmp_path, monkeypatch, capsysbinary, action, case,
):
    monkeypatch.chdir(tmp_path)
    value = _VALUES[case]()
    calls = []
    original = digests.JsonProfile.text

    def recording_text(self, candidate):
        calls.append((self.name, candidate))
        return original(self, candidate)

    monkeypatch.setattr(digests.JsonProfile, "text", recording_text)
    build_argv = _campaign_argv if action == "campaign" else _inventory_argv
    argv = build_argv(monkeypatch, value)

    if case in _REFUSALS:
        with pytest.raises(TypeError) as refused:
            reprice.main(argv)
        assert type(refused.value) is TypeError
        assert str(refused.value) == _REFUSALS[case]
        assert refused.value.__cause__ is None
    else:
        assert reprice.main(argv) == 0
    out, err = capsysbinary.readouterr()
    assert err == b""

    sidecar = tmp_path / "out" / "cost.provenance.json"
    if action == "campaign":
        if case in _REFUSALS:
            assert out == b""
            assert not sidecar.exists()
        else:
            assert out == _CAMPAIGN_MESSAGE
            assert sidecar.read_bytes() == _CAMPAIGN_BYTES[case]
    else:
        assert out == (b"" if case in _REFUSALS else _INVENTORY_BYTES[case])

    own = [call for call in calls if call[0] == PROFILE_NAME]
    assert len(own) == 1
    published = own[0][1]
    if action == "campaign":
        assert published["schema"] == "fixture"
        assert published["campaign_identity"] is value
    else:
        assert published["checks"][0]["name"] == "fixture"
