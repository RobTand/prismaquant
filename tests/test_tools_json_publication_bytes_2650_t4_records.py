"""The T4 record tools publish the same JSON bytes through their bound profiles.

The literals below come from the unedited tools. A profile must not change
one byte of a file, of a printed line or of a digest.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.digests import JsonProfile
from tools import check_t4_overlay_prepare as check
from tools import rebind_t4_qualified_results as rebind

NAN, INF = float("nan"), float("inf")
FMT = "F\u00e9"
QNAME = "w.\u00e9.\U0001F600.down"
ANCHOR = {"dloss": NAN, "hi": INF, "lo": -INF, "e": "\u00e9", "nest": [1.5, [], {"b": 1, "a": {}}],
          "B": 2, "10": 3, "2": 4}
EVIDENCE = {"\u00e9": "\U0001F600", "nan": NAN, "inf": INF, "ninf": -INF, "empty": {}, "B": True,
            "nest": [[1, [2, {}]], {"b": [], "a": {"c": None}}], "10": 1, "2": 2}

CELLS = [({}, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
         ({"\u00e9": "\U0001F600", "B": [1, {}], "a": []},
          "c1582555bcecce61eaa00b275319199d1c73a44780e266032713a7b26a829a35"),
         ({"n": NAN, "i": INF, "m": -INF}, "95d375456607c10f29e29703638ee8a4fe661559266abbf7e18b6c20fb60c76a"),
         ({"x": [[1, [2, {}]], {"b": 1, "a": 2}]},
          "f55cae3e823827bf47f6db449350fcfb49a9b92585a014bc2f998f0f340ba36f")]
REBIND_FILE = (
    b'{"catalog":{"path":"catalog.json","sha256":"a80472e1fc8c608e8e5a58e0808961116f66fec262a04ba31d448a7a97da8e5c"},'
    b'"previous_catalog":{"path":"previous.json","sha256":"1d43c2c099dc606865c8d73cb7165301560afda99d880a94be1c9e84505d4dec"},'
    b'"qualified_dir":"qualified","rows":[{"cell_sha256":"6cbd147658f52c8141d2b95307ee45643fcbd2a3d4489cc32e4ce69c06460d55",'
    b'"format":"F\\u00e9","previous_anchor":{"10":3,"2":4,"B":2,"dloss":NaN,"e":"\\u00e9","hi":Infinity,"lo":-Infinity,'
    b'"nest":[1.5,[],{"a":{},"b":1}]},"previous_cell_sha256":"77a0be1deb1e7b152afa1b4932dda5fd5908a855fa3d47f84d33a65dc88f764d",'
    b'"qname":"w.\\u00e9.\\ud83d\\ude00.down","result_sha256":"a2b2af0c822274bd8f216efffb368b62f9b65d8f1557d7834c65c92456601289"}],'
    b'"schema":"prismaquant.t4_qualified_rebinding.v1","scope":"each new cell equals its previous cell except anchor; '
    b'the qualifier never reads the anchor"}\n')
REBIND_STDOUT = (b'{"path": "sub/rebinding.json", "sha256": "da4147efb31e2a855071d10a38588b0eeb7121412cd0e171f825fa48364a55b5", '
                 b'"rows": 1, "anchor_changed": 1}\n')
CHECK_SHA = "7c9c5461f16a6fc8b1f0abaf5f8a2a333980c1acfc0942c9bf9eb7c024813535"
CHECK_FILE = (
    b'{\n  "added_formats": [\n    "F",\n    "G"\n  ],\n  "evidence": {\n    "10": 1,\n    "2": 2,\n'
    b'    "B": true,\n    "empty": {},\n    "inf": Infinity,\n    "nan": NaN,\n    "nest": [\n      [\n'
    b'        1,\n        [\n          2,\n          {}\n        ]\n      ],\n      {\n        "a": {\n'
    b'          "c": null\n        },\n        "b": []\n      }\n    ],\n    "ninf": -Infinity,\n'
    b'    "\\u00e9": "\\ud83d\\ude00"\n  },\n  "inputs": {\n    "path": "inputs.json",\n'
    b'    "sha256": "' + CHECK_SHA.encode() + b'"\n  },\n  "loader_added_cells": 2,\n'
    b'  "loader_added_cells_by_format": {\n    "F": 1,\n    "G": 1\n  },\n  "loader_cells": 3,\n'
    b'  "loader_units": 1,\n  "scope": "verify_catalog_pair + load_measured_anchor_input(verify_payloads=False) + '
    b'model-free prepared-completion checks; opens under the source model refused",\n'
    b'  "seconds": 0.0,\n  "status": "passed"\n}\n')
CHECK_STDOUT = (
    b'{"step": "verify_catalog_pair", "s": 0.0}\n'
    b'{"step": "load_measured_anchor_input", "cells": 3, "added": 2, "s": 0.0}\n'
    b'{"status": "passed", "inputs": {"path": "inputs.json", "sha256": "' + CHECK_SHA.encode() + b'"}, '
    b'"evidence": {"\\u00e9": "\\ud83d\\ude00", "nan": NaN, "inf": Infinity, "ninf": -Infinity, "empty": {}, '
    b'"B": true, "nest": [[1, [2, {}]], {"b": [], "a": {"c": null}}], "10": 1, "2": 2}, "loader_cells": 3, '
    b'"loader_units": 1, "loader_added_cells": 2, "added_formats": ["F", "G"], '
    b'"loader_added_cells_by_format": {"F": 1, "G": 1}, "scope": "verify_catalog_pair + '
    b'load_measured_anchor_input(verify_payloads=False) + model-free prepared-completion checks; '
    b'opens under the source model refused", "seconds": 0.0, "result": {"path": "sub/out.json", '
    b'"sha256": "d4070aa0fba42968d192eb5d96bbf5e32c897f5320934c08b1b6377e7f8df253"}}\n')
REBIND_REFUSALS = [("exists", AssertionError, "sub/rebinding.json", b"keep"),
                   ("sha", AssertionError, "('catalog.json', 'SHA256 mismatch')", None),
                   ("race", FileExistsError, "[Errno 17] File exists: 'sub/rebinding.json'", b"keep")]
CHECK_REFUSALS = [("exists", AssertionError, "sub/out.json", b"keep"),
                  ("sha", AssertionError, f"{{'path': 'inputs.json', 'sha256': '{CHECK_SHA}'}}", None),
                  ("race", FileExistsError, "[Errno 17] File exists: 'sub/out.json'", b"keep")]


def _write(name, value):
    raw = json.dumps(value).encode()
    Path(name).write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def _cell(anchor):
    return {"qname": QNAME, "format": FMT, "anchor": anchor, "render": "/r/\u00e9.pt",
            "record": {"blob_sha256": "b" * 64, "z": [[], {}], "\u00e9": "\u00fc"}}


def _rebind_setup(root, mp):
    mp.chdir(root)
    previous_cell = _cell(ANCHOR)
    verified = {"render_origin": "\u00e9", "w": [NAN, INF]}
    result = {"qname": QNAME, "format": FMT, "cell_sha256": rebind.cell_sha256(previous_cell),
              "verified_cell": verified, "verified_cell_sha256": rebind.cell_sha256(verified)}
    Path("qualified").mkdir()
    rebind.result_path("qualified", QNAME, FMT, key="pair").write_text(json.dumps(result) + "\n")
    header = {"schema": "s", "note": "\u00e9", "nest": [{}]}
    previous = _write("previous.json", {**header, "cells": [previous_cell]})
    catalog = _write("catalog.json", {**header, "cells": [_cell({"dloss": 1.0})]})
    mp.setattr(sys, "argv", [
        "rebind", "--catalog", "catalog.json", "--catalog-sha256", catalog["sha256"],
        "--previous-catalog", "previous.json", "--previous-catalog-sha256", previous["sha256"],
        "--qualified-dir", "qualified", "--out", "sub/rebinding.json"])


def _check_setup(root, mp):
    mp.chdir(root)
    catalog = _write("catalog.json", {"cells": [{"qname": "q", "format": "F"},
                                                {"qname": "\u00e9", "format": "G"}]})
    plan = _write("plan.json", {"model": "model", "inputs": {"candidate_overlay": catalog}})
    Path("cache.pkl").write_bytes(pickle.dumps({}))
    cache = {"path": "cache.pkl", "sha256": hashlib.sha256(Path("cache.pkl").read_bytes()).hexdigest()}
    prepared = _write("prepared.json", {"production_cache": cache})
    inputs = _write("inputs.json", {"extended_plan": plan, "extended_prepared": prepared})
    mp.setattr(sys, "argv", ["check", "--inputs", "inputs.json", "--inputs-sha256", inputs["sha256"],
                             "--out", "sub/out.json"])
    cells = {("q", "F"): 1, ("\u00e9", "G"): 1, ("x", "F"): 1}
    mp.setattr(check, "refuse_model_reads", lambda root: None)
    mp.setattr(check, "check_completion", lambda *args, **kwargs: None)
    mp.setattr(check, "time", SimpleNamespace(monotonic=lambda: 0.0))
    mp.setattr("prismaquant.joint_catalog_extension.verify_catalog_pair", lambda inputs: EVIDENCE)
    mp.setattr("prismaquant.joint_catalog_extension.catalog_view", lambda catalog: {"formats": ["F", "G"]})
    mp.setattr("prismaquant.tessera_joint_aura.load_measured_anchor_input",
               lambda *args, **kwargs: SimpleNamespace(cells=cells, formats_by_qname={"q": ["F"]}))
    mp.setattr("prismaquant.production_weight_cache.ProductionWeightCache", dict)


def _break(how, mp, bound, owner, hook, out):
    """Make one refusal: the output exists, a bound file changed, or the output appears late."""
    def plant():
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(b"keep")

    if how == "exists":
        plant()
    elif how == "sha":
        Path(bound).write_bytes(b"{}")
    else:
        real = getattr(owner, hook)

        def late(*args, **kwargs):
            result = real(*args, **kwargs)
            plant()
            return result

        mp.setattr(owner, hook, late)


def _spy(mp):
    calls = []
    real = JsonProfile.text

    def text(self, value):
        calls.append((self.name, value))
        return real(self, value)

    mp.setattr(JsonProfile, "text", text)
    return calls


def _canon(value):
    """Give a text that compares a whole value, NaN included."""
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("cell,digest", CELLS)
def test_t4_records_cell_sha256_literals(cell, digest):
    assert rebind.cell_sha256(cell) == digest


def test_t4_records_rebind_bytes(tmp_path, monkeypatch, capsysbinary):
    _rebind_setup(tmp_path, monkeypatch)
    rebind.main()
    assert Path("sub/rebinding.json").read_bytes() == REBIND_FILE
    assert capsysbinary.readouterr().out == REBIND_STDOUT


def test_t4_records_check_bytes(tmp_path, monkeypatch, capsysbinary):
    _check_setup(tmp_path, monkeypatch)
    fsyncs = []
    real = os.fsync

    def fsync(fd):
        fsyncs.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    check.main()
    assert Path("sub/out.json").read_bytes() == CHECK_FILE
    assert capsysbinary.readouterr().out == CHECK_STDOUT
    assert len(fsyncs) == 1


def _refusal(main, out, error, message, left):
    with pytest.raises(error) as caught:
        main()
    assert type(caught.value) is error
    assert str(caught.value) == message
    assert caught.value.__cause__ is None
    if left is None:
        assert not out.exists()
    else:
        assert out.read_bytes() == left


@pytest.mark.parametrize("how,error,message,left", REBIND_REFUSALS)
def test_t4_records_rebind_refusals(tmp_path, monkeypatch, how, error, message, left):
    _rebind_setup(tmp_path, monkeypatch)
    out = Path("sub/rebinding.json")
    _break(how, monkeypatch, "catalog.json", rebind, "rebind_cell", out)
    _refusal(rebind.main, out, error, message, left)


@pytest.mark.parametrize("how,error,message,left", CHECK_REFUSALS)
def test_t4_records_check_refusals(tmp_path, monkeypatch, how, error, message, left):
    _check_setup(tmp_path, monkeypatch)
    out = Path("sub/out.json")
    _break(how, monkeypatch, "inputs.json", check, "check_completion", out)
    _refusal(check.main, out, error, message, left)


def test_t4_records_profile_calls(tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    cell = {"n": NAN, "a": [{}]}
    rebind.cell_sha256(cell)
    assert calls == [("direct-ascii-lax", cell)]
    (tmp_path / "a").mkdir()
    _rebind_setup(tmp_path / "a", monkeypatch)
    calls.clear()
    rebind.main()
    documents = [call for call in calls if isinstance(call[1], dict) and call[1].get("schema") == rebind.SCHEMA]
    assert [name for name, _ in documents] == ["direct-ascii-lax"]
    assert _canon(documents[0][1]) == _canon(json.loads(REBIND_FILE))
    (tmp_path / "b").mkdir()
    _check_setup(tmp_path / "b", monkeypatch)
    calls.clear()
    check.main()
    results = [call for call in calls if isinstance(call[1], dict) and call[1].get("status") == "passed"]
    assert [name for name, _ in results] == ["direct-ascii-indent2-lax"]
    assert _canon(results[0][1]) == _canon(json.loads(CHECK_FILE))

