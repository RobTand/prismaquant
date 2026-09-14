#!/usr/bin/env python3
"""Run one dense native-operator measurement and keep the evidence its run needs.

``runtime_provenance._observe_run`` reads three artifacts per native run, and
only two of them come out of Tessera's harness by itself:

``post_core``
    ``{native_returncode, manifest_sha256, stock_files_unchanged}``, written by
    the parent after the measuring child exits, from a fresh walk of the
    installed vLLM package compared with the attested core manifest.
``post_package``
    ``tessera.loaded_package_identity.v1``, written *inside* the measuring
    process after the harness finishes, because which ``tessera.*`` modules were
    loaded and from which bytes is a fact about that process and nothing else.

This is the dense counterpart of Tessera's
``_pb_native_moe_measure/run_native.py`` and ``run_native_child.py``, which do
exactly this around ``bench_native_moe_operator``. It is a consumer-side
harness: it runs the measuring module unmodified through ``runpy`` and adds no
argument of its own to it. ``--module`` names that module and defaults to
Tessera's own dense harness; a consumer-side sweep over the same prepared cells
(``experiments.pq_prefill_load_sweep``) needs the same two audits and must not
fork this file to get them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

INSTALL_EVIDENCE_NAME = "per-job-runtime.json"
DEFAULT_BENCH_MODULE = "experiments.bench_native_operator"


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files(root) -> dict:
    """The installer's own file roster rule, so the two rosters compare."""
    root = Path(root)
    return {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def run(args) -> int:
    import importlib.util

    install = Path(args.install_evidence) / INSTALL_EVIDENCE_NAME
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    core_manifest = Path(args.core_manifest)
    manifest_sha256 = digest(core_manifest)
    stock = json.loads(core_manifest.read_text())
    child = [sys.executable, "-u", str(Path(__file__).resolve()), "child",
             "--install-evidence", str(args.install_evidence),
             "--evidence-dir", str(evidence), "--module", args.module, "--", *args.command]
    result = subprocess.run(child)
    core = Path(importlib.util.find_spec("vllm").origin).parent
    if files(core) != stock["files"]:
        raise SystemExit("native execution changed stock vLLM files")
    path = evidence / "post-native-core.json"
    path.write_text(json.dumps({"stock_files_unchanged": len(stock["files"]),
                                "manifest_sha256": manifest_sha256,
                                "native_returncode": result.returncode},
                               indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(path), "sha256": digest(path),
                      "native_returncode": result.returncode}), flush=True)
    return result.returncode


def child(args) -> int:
    import runpy

    install = Path(args.install_evidence) / INSTALL_EVIDENCE_NAME
    installer_sha256 = digest(install)
    evidence = Path(args.evidence_dir)
    sys.argv = [args.module, *args.command]
    status = 0
    try:
        runpy.run_module(args.module, run_name="__main__")
    except SystemExit as exc:
        status = int(exc.code or 0)
    finally:
        import tessera
        import tessera.cached_unit as cached_unit

        package = Path(tessera.__file__).resolve().parent
        package_files = files(package)
        if digest(install) != installer_sha256:
            raise SystemExit("installer evidence changed after the native child launched")
        installed = json.loads(install.read_text())["plugin_files"]
        cached_unit.encoder_source_sha256.cache_clear()
        modules, errors = {}, []
        for name, module in sorted(sys.modules.copy().items()):
            if name != "tessera" and not name.startswith("tessera."):
                continue
            filename = getattr(module, "__file__", None)
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            row = {"file": filename, "origin": origin}
            try:
                if not filename or not origin:
                    raise ValueError("module has no verifiable file and spec origin")
                actual = Path(filename).resolve(strict=True)
                source = Path(origin).resolve(strict=True)
                if actual != source:
                    raise ValueError("module file and spec origin differ")
                relative = actual.relative_to(package).as_posix()
                sha = hashlib.sha256(actual.read_bytes()).hexdigest()
                if relative not in installed or installed[relative]["sha256"] != sha:
                    raise ValueError("module bytes differ from the canonical installed package")
                row.update(file=str(actual), origin=str(source), sha256=sha)
            except (OSError, ValueError) as exc:
                errors.append({"module": name, "error": str(exc)})
            modules[name] = row
        record = {
            "schema": "tessera.loaded_package_identity.v1",
            "observation_scope": "same native child process, after harness completion and outside measurement",
            "installer_evidence_sha256": installer_sha256,
            "package_path": str(package),
            "cached_unit_path": str(Path(cached_unit.__file__).resolve()),
            "encoder_source_sha256": cached_unit.encoder_source_sha256(),
            "sys_path": sys.path,
            "loaded_tessera_modules": modules,
            "module_identity_errors": errors,
            "package_files": package_files,
            "package_files_unchanged_from_installer": package_files == installed,
        }
        path = evidence / "post-native-package.json"
        with path.open("x") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"artifact": str(path), "sha256": digest(path)}), flush=True)
        if not record["package_files_unchanged_from_installer"]:
            raise SystemExit("native execution changed installed Tessera package files")
        if errors:
            raise SystemExit("native loaded package origins differ from the installed bytes")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, function in (("run", run), ("child", child)):
        node = sub.add_parser(name)
        node.add_argument("--install-evidence", required=True,
                          help="directory holding the installer's per-job-runtime.json")
        node.add_argument("--evidence-dir", required=True, help="this measurement's evidence directory")
        node.add_argument("--module", default=DEFAULT_BENCH_MODULE,
                          help="the measuring module to run unmodified through runpy")
        if name == "run":
            node.add_argument("--core-manifest", required=True,
                              help="the attested vLLM core manifest this install was checked against")
        node.add_argument("command", nargs=argparse.REMAINDER)
        node.set_defaults(func=function)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a native harness command is required after --")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
