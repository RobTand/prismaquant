import math
from pathlib import Path
import pickle

import pytest

from prismaquant import glm53_stock_reprice as stock
from prismaquant.digests import JsonProfile

CAMPAIGN_EMPTY = (
    b'{\n  "campaign_identity": {},\n  "ladder_refusals": 0,\n'
    b'  "merge_report": {\n    "anchored_units": 1,\n    "empirical_units": 2,\n'
    b'    "pinned_units": 3,\n    "total_cells": 6\n  },\n'
    b'  "output": "out/cost.pkl",\n  "schema": "fixture",\n  "unpriced_probe_units": []\n}\n'
)
CAMPAIGN_RICH = (
    b'{\n  "campaign_identity": {\n    "infinity": Infinity,\n'
    b'    "nan": NaN,\n    "negative_infinity": -Infinity,\n    "nested": [\n'
    b'      null,\n      true,\n      -1,\n      1.5,\n      [],\n      {}\n    ],\n'
    b'    "unicode": "snowman \\u2603 / caf\\u00e9"\n  },\n  "ladder_refusals": 0,\n'
    b'  "merge_report": {\n    "anchored_units": 1,\n    "empirical_units": 2,\n'
    b'    "pinned_units": 3,\n    "total_cells": 6\n  },\n'
    b'  "output": "out/cost.pkl",\n  "schema": "fixture",\n  "unpriced_probe_units": []\n}\n'
)
INVENTORY_EMPTY = (
    b'{\n  "checks": [\n    {\n      "detail": {},\n      "name": "fixture",\n'
    b'      "path": "checkpoint",\n      "status": "PASS"\n    }\n  ]\n}\n'
)
INVENTORY_RICH = (
    b'{\n  "checks": [\n    {\n      "detail": {\n        "infinity": Infinity,\n'
    b'        "nan": NaN,\n        "negative_infinity": -Infinity,\n        "nested": [\n'
    b'          null,\n          true,\n          -1,\n          1.5,\n          [],\n          {}\n'
    b'        ],\n        "unicode": "snowman \\u2603 / caf\\u00e9"\n      },\n'
    b'      "name": "fixture",\n      "path": "checkpoint",\n      "status": "PASS"\n    }\n  ]\n}\n'
)
CAMPAIGN_STDOUT = (
    b'[glm53-stock] campaign cost payload -> out/cost.pkl '
    b'(anchored 1, empirical 2, pinned 3, cells 6; sidecar cost.provenance.json)\n'
)
RICH = {
    "unicode": "snowman \u2603 / caf\u00e9",
    "nested": [None, True, -1, 1.5, [], {}],
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative_infinity": -float("inf"),
}


@pytest.mark.parametrize("action", ["campaign", "inventory"])
@pytest.mark.parametrize(
    "value, campaign_bytes, inventory_bytes, message",
    [
        pytest.param({}, CAMPAIGN_EMPTY, INVENTORY_EMPTY, None, id="empty"),
        pytest.param(RICH, CAMPAIGN_RICH, INVENTORY_RICH, None, id="rich"),
        pytest.param(
            {1: "number", "x": "text"}, None, None,
            "'<' not supported between instances of 'str' and 'int'", id="mixed",
        ),
        pytest.param(
            {"bad": object()}, None, None,
            "Object of type object is not JSON serializable", id="object",
        ),
    ],
)
def test_stock_publication_profile_calls(
    action, value, campaign_bytes, inventory_bytes, message,
    tmp_path, monkeypatch, capsysbinary,
):
    monkeypatch.chdir(tmp_path)
    Path("probe.json").write_text(f'{{"schema": "{stock.PROBE_CENSUS_SCHEMA}"}}')
    Path("checkpoint.json").write_text(f'{{"schema": "{stock.CHECKPOINT_CENSUS_SCHEMA}"}}')
    argv = [action, "--probe-census", "probe.json", "--checkpoint-census", "checkpoint.json",
            "--expert-empirical", "expert.pkl"]
    output = Path("out/cost.pkl")
    sidecar = output.with_suffix(".provenance.json")
    if action == "campaign":
        for name in ("harvest.pkl", "expert.pkl"):
            Path(name).write_bytes(pickle.dumps({}))
        report = {"anchored_units": 1, "empirical_units": 2, "pinned_units": 3, "total_cells": 6}
        payload = {"schema": "fixture", "provenance": {
            "merge_report": report, "ladder_refusals": [], "unpriced_probe_units": [],
            "campaign_identity": value,
        }}
        monkeypatch.setattr(stock, "run_campaign_from_artifacts", lambda **kwargs: payload)
        argv += ["--harvest", "harvest.pkl", "--output", str(output)]
        expected = {
            "schema": "fixture", "output": str(output), "merge_report": report,
            "ladder_refusals": 0, "unpriced_probe_units": [], "campaign_identity": value,
        }
    else:
        check = stock.Check(name="fixture", status="PASS", detail=value, path="checkpoint")
        monkeypatch.setattr(stock, "inventory_checks", lambda **kwargs: [check])
        argv += ["--model", "model", "--checkpoint-dir", "checkpoint", "--repo", ".", "--json"]
        expected = {"checks": [dict(name="fixture", status="PASS", detail=value, path="checkpoint")]}

    calls = []
    original = JsonProfile.text
    publication_key = "campaign_identity" if action == "campaign" else "checks"

    def recording_text(self, item):
        if isinstance(item, dict) and publication_key in item:
            calls.append((self.name, item, output.is_file(), sidecar.exists()))
        return original(self, item)

    monkeypatch.setattr(JsonProfile, "text", recording_text)
    inputs = set(tmp_path.iterdir())
    status = terminal = None
    if message is None:
        status = stock.main(argv)
    else:
        with pytest.raises(TypeError) as raised:
            status = stock.main(argv)
        terminal = (type(raised.value), str(raised.value), raised.value.__cause__)

    assert status == (0 if message is None else None)
    assert terminal == (None if message is None else (TypeError, message, None))
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert captured.out == (
        b"" if message is not None else CAMPAIGN_STDOUT if action == "campaign" else inventory_bytes
    )
    files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()} - {
        path.relative_to(tmp_path) for path in inputs
    }
    side_bytes = sidecar.read_bytes() if sidecar.is_file() else None
    assert side_bytes == (campaign_bytes if action == "campaign" else None)
    assert files == (
        set() if action == "inventory" else {output} if message is not None else {output, sidecar}
    )

    assert len(calls) == 1
    name, actual, has_pickle, has_sidecar = calls[0]
    assert name == "direct-ascii-indent2-lax"
    assert has_pickle == (action == "campaign")
    assert not has_sidecar
    detail = actual[publication_key] if action == "campaign" else actual["checks"][0]["detail"]
    if "nan" in value:
        assert math.isnan(detail["nan"])
        detail["nan"] = value["nan"]
    if "bad" in value:
        assert type(detail["bad"]) is object
        detail["bad"] = value["bad"]
    assert actual == expected
