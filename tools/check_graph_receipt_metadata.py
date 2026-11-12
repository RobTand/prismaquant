#!/usr/bin/env python3
"""Check synthetic graph records against explicit external metadata sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.digests import file_sha256hex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    actual = file_sha256hex(args.archive)
    if actual != args.sha256:
        parser.error(f"external metadata archive changed: {actual}")
    with tempfile.TemporaryDirectory(prefix="graph-metadata-") as directory:
        root = Path(directory)
        with tarfile.open(args.archive) as package:
            if package.pax_headers.get("comment") != args.commit:
                parser.error("archive does not record the specified Git commit")
            package.extractall(root, filter="data")
        sys.path[:0] = [str(root / "src"), str(Path.cwd()), str(Path.cwd() / "tests")]
        from tessera import graph_receipt

        if not Path(graph_receipt.__file__).is_relative_to(root):
            parser.error("metadata module came from another source")
        if not callable(getattr(graph_receipt, "population_refusal", None)):
            parser.error("external metadata lacks a compatible population verifier")
        if any(name.startswith("tessera.serving") for name in sys.modules):
            parser.error("metadata import loaded the serving stack")
        print(json.dumps({"scope": "synthetic metadata fixtures only",
                          "tessera_source_commit": args.commit,
                          "archive_sha256": actual,
                          "module": str(graph_receipt.__file__),
                          "production_pins_changed": False}), flush=True)
        import pytest

        pytest_args = args.pytest_args
        if pytest_args[:1] == ["--"]:
            pytest_args = pytest_args[1:]
        return pytest.main(pytest_args or ["-q", "tests/test_graph_shipcard_numerical.py"])


if __name__ == "__main__":
    raise SystemExit(main())
