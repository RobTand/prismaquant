"""The release archive check must cover runtime serving-dispatch tables."""
from __future__ import annotations

import io
from pathlib import Path
import runpy
import sys
import tarfile
import zipfile

import pytest


@pytest.mark.parametrize("missing_from", ["wheel", "sdist"])
def test_release_check_refuses_missing_dispatch_table(tmp_path, monkeypatch, missing_from):
    source = tmp_path / "source"
    dist = tmp_path / "dist"
    dist.mkdir()
    assets = {
        "prismaquant/run-pipeline.sh": b"#!/bin/bash\n",
        "prismaquant/serve_dispatch_tables/example.json": b"{}\n",
    }
    for name, data in assets.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with zipfile.ZipFile(dist / "example.whl", "w") as archive:
        for name, data in assets.items():
            if missing_from != "wheel" or not name.endswith(".json"):
                archive.writestr(name, data)
    with tarfile.open(dist / "example.tar.gz", "w:gz") as archive:
        for name, data in assets.items():
            if missing_from != "sdist" or not name.endswith(".json"):
                member = tarfile.TarInfo("example-1/" + name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    script = Path(__file__).resolve().parents[1] / ".github/scripts/check_dist.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(sys, "argv", [str(script), str(dist), str(source)])
    with pytest.raises(SystemExit) as caught:
        main()
    message = str(caught.value)
    assert message.startswith(missing_from + " ")
    assert "is missing 1 runtime data file(s)" in message
    assert "prismaquant/serve_dispatch_tables/example.json" in message
