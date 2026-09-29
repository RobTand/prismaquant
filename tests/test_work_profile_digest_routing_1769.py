"""Exact-byte ownership only: these synthetic timings make no speed claim."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from prismaquant import digests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # pyright: ignore[reportMissingImports]


@pytest.mark.parametrize("census_bytes", (b"", b"\x00\xff\r\nUTF-8:\xc3\xa9"))
@pytest.mark.parametrize("mismatch", (False, True))
def test_work_profile_routes_exact_bytes(tmp_path, monkeypatch, census_bytes, mismatch):
    census = tmp_path / "census.json"
    census.write_bytes(census_bytes)
    profile = {
        "schema": "prismaquant.campaign_row_work.v1",
        "census_sha256": "0" * 64 if mismatch else hashlib.sha256(census_bytes).hexdigest(),
        "campaign_argv": ["--format", "BF16"],
        "startup_seconds": 20.0,
        "max_startup_fraction": 0.4,
        "gpu_slots": 1,
        "group_pricing_seconds": {"u:a": 100.0, "u:b": 100.0},
        "evidence": {"startup": "synthetic-é", "pricing": "synthetic-test"},
    }
    raw = (json.dumps(profile, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")
    path = tmp_path / "work.json"
    path.write_bytes(raw)
    calls = []
    owner = digests.bytes_sha256hex

    def recording(value):
        calls.append(value)
        return owner(value)

    monkeypatch.setattr(digests, "bytes_sha256hex", recording)
    kwargs = dict(census_path=census, groups={"u:a": ["a"], "u:b": ["b"]},
                  campaign_argv=profile["campaign_argv"], groups_per_row=1)
    if mismatch:
        with pytest.raises(RuntimeError, match="row work profile census_sha256 differs from this census"):
            dispatch.work_profile_bundles(path, **kwargs)
        assert calls == [census_bytes]
    else:
        bundles, binding = dispatch.work_profile_bundles(path, **kwargs)
        assert bundles == [(0, ["u:a"]), (1, ["u:b"])]
        assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
        assert binding["profile"] == profile
        assert binding["path"] == str(path)
        assert calls == [census_bytes, raw]
