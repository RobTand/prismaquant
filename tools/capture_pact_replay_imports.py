#!/usr/bin/env python3
"""Observe the public PACT CPU CLI without a synthetic import sequence.

This diagnostic leaves the replay source and its identity gates unchanged.
It records modules present after the CLI returns, not function execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def file_record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"origin": str(path), "sha256": digest}


def capture_cli(entry, arguments, roots, output, source_head):
    """Run main with normal script argv and record its observed module files."""
    entry = Path(entry).resolve()
    output = Path(output)
    roots = {name: Path(root).resolve() for name, root in roots.items()}
    entry_record = file_record(entry)
    before = set(sys.modules)
    previous_argv, previous_path = sys.argv, sys.path[:]
    sys.argv = [str(entry), *arguments]
    sys.path[0] = str(entry.parent)
    initial_path = sys.path[:]
    returncode = 1
    error = None
    try:
        runpy.run_path(str(entry), run_name="__main__")
        returncode = 0
    except SystemExit as exc:
        returncode = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        modules, all_modules, non_file = {}, {}, {}
        for name, module in sorted(sys.modules.copy().items()):
            if name == "__main__":
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
            if not origin or origin in {"built-in", "frozen"}:
                non_file[name] = origin
                continue
            path = Path(origin).resolve()
            if not path.is_file():
                non_file[name] = {"origin": origin, "reason": "The module origin has no file."}
                continue
            record = file_record(path)
            record["loaded_before_cli"] = name in before
            all_modules[name] = record
            for tag, root in roots.items():
                if path.is_relative_to(root):
                    modules[name] = {**record, "source": tag, "path": str(path.relative_to(root))}
                    break
        sys.argv, sys.path[:] = previous_argv, previous_path
        output.mkdir(parents=True, exist_ok=True)
        library_path = output / "all-modules.json"
        library_bytes = (json.dumps({"modules": all_modules, "non_file_modules": non_file},
                                   indent=2, sort_keys=True) + "\n").encode()
        library_path.write_bytes(library_bytes)
        capture = {
            "schema": "pact.public_cli_import_capture.v1",
            "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
            "source_head": source_head,
            "entry": entry_record,
            "argv": [sys.executable, str(entry), *arguments],
            "initial_sys_path": initial_path,
            "source_roots": {name: str(root) for name, root in roots.items()},
            "returncode": returncode,
            "error": error,
            "modules": modules,
            "all_modules": {"file": str(library_path),
                            "sha256": hashlib.sha256(library_bytes).hexdigest()},
            "scope": "Modules present after the actual CLI call. Unobserved source files remain inferred dependencies.",
        }
        raw = (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode()
        (output / "cli-imports.json").write_bytes(raw)
        print(json.dumps({"cli_import_capture": str(output / "cli-imports.json"),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "returncode": returncode, "observed_modules": len(modules)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    replay = Path(__file__).resolve().parents[1] / "prismaquant" / "pact_replay"
    manifest = json.loads((replay / "DEPENDENCY_MANIFEST.json").read_bytes())
    roots = {"pact-replay": replay,
             **{name: source["path"] for name, source in manifest["sources"].items()}}
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    capture_cli(replay / "multi_stream_replay.py", arguments, roots, args.output, args.source_head)


if __name__ == "__main__":
    main()
