"""Pytest plugin for the pq1317 qualification runs. It only observes.

It records, for each test process:

* the Tessera native libraries (``tessera_*.so``) the process has mapped after
  each test, read from ``/proc/self/maps``, each with the sha256 of the bytes
  that are mapped (read through ``/proc/self/map_files``); and
* the evaluated text of each passing assertion that holds a floating-point
  value or names a launch (a symbol, a decoder or a ``tessera::`` op), through
  pytest's ``pytest_assertion_pass`` hook.

It also gives each xdist worker its own ``TORCH_EXTENSIONS_DIR`` under
``PQ1317_EXT_DIR_ROOT``. On the shared NFS mount a library that one worker
rebuilds while another worker has it mapped is renamed to ``.nfsXXXX``. The
mapping then loses its ``tessera_`` name and the binary that ran is unclear.

Load it with ``-p native_probe -o enable_assertion_pass_hook=true`` and set
``PQ1317_PROBE_DIR``. Each process appends JSON lines to
``probe.<worker>.jsonl`` in that directory. The plugin never fails a test:
every error becomes a record.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

PROBE_DIR_ENV = "PQ1317_PROBE_DIR"
EXT_DIR_ROOT_ENV = "PQ1317_EXT_DIR_ROOT"
NATIVE_PREFIX = "tessera_"
MAX_TEXT = 600
MAX_ASSERTIONS_PER_TEST = 64
#: A float literal or an exponent in the evaluated text: the assertion holds a number.
HAS_FLOAT = re.compile(r"\d+\.\d+|\d[eE][-+]?\d+")
#: The assertion names a launch: the route's symbol and decoder, or a Tessera op.
NAMES_LAUNCH = re.compile(r"symbol|decoder|launch_pair|tessera::|DENSE_")
#: The build directory of a JIT extension: <module>_<platform token>_tessera_guarded_v1.
BUILD_DIR = re.compile(r"^(tessera_[a-z0-9_]+?)_sm_\d+_tessera_guarded_v1$")

_digests: dict[str, str] = {}
_assertions: dict[str, list[dict]] = {}
_dropped: dict[str, int] = {}
_state: dict = {"config": None}


def _worker(config) -> str:
    return getattr(config, "workerinput", {}).get("workerid", "main")


def _executes_tests(config) -> bool:
    """True in the process that runs the tests: an xdist worker, or a plain run."""
    if hasattr(config, "workerinput"):
        return True
    return not getattr(config.option, "numprocesses", None)


def _append(record: dict) -> None:
    config = _state["config"]
    directory = os.environ.get(PROBE_DIR_ENV)
    if config is None or not directory:
        return
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"probe.{_worker(config)}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"worker": _worker(config), "pid": os.getpid(), **record},
                                    sort_keys=True) + "\n")
    except OSError:
        pass


def parse_maps(lines) -> list[dict]:
    """Tessera shared objects in ``/proc/self/maps`` lines.

    A library is recognised by its file name (``tessera_*.so``) or, when NFS has
    renamed a busy file to ``.nfsXXXX``, by the build directory it sits in.
    """
    found = []
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        address, path = fields[0], fields[5].strip()
        deleted = path.endswith(" (deleted)")
        if deleted:
            path = path[: -len(" (deleted)")]
        name = os.path.basename(path)
        module = None
        if name.startswith(NATIVE_PREFIX) and ".so" in name:
            module = name.split(".so", 1)[0]
        elif name.startswith(".nfs"):
            built = BUILD_DIR.match(os.path.basename(os.path.dirname(path)))
            module = built.group(1) if built else None
        if module:
            found.append({"module": module, "address": address, "path": path,
                          "nfs_renamed": name.startswith(".nfs"), "deleted": deleted})
    return found


def _digest(entry: dict) -> str:
    """sha256 of the mapped bytes: the map file first, the path as a fall back."""
    key = f"{entry['path']}@{entry['address']}"
    if key not in _digests:
        for candidate in (f"/proc/self/map_files/{entry['address']}", entry["path"]):
            try:
                _digests[key] = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
                break
            except OSError as error:
                _digests[key] = f"unreadable: {error}"
    return _digests[key]


def mapped_native_libraries() -> dict[str, dict]:
    """``{module: {path, sha256, nfs_renamed, deleted}}`` for the mapped Tessera libraries."""
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
            entries = parse_maps(handle)
    except OSError as error:
        return {"<unreadable /proc/self/maps>": {"error": str(error)}}
    return {e["module"]: {"path": e["path"], "sha256": _digest(e), "nfs_renamed": e["nfs_renamed"],
                          "deleted": e["deleted"]} for e in entries}


def pytest_configure(config):
    _state["config"] = config
    root = os.environ.get(EXT_DIR_ROOT_ENV)
    if root and hasattr(config, "workerinput"):
        directory = Path(root) / _worker(config)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(directory)


def pytest_assertion_pass(item, lineno, orig, expl):
    try:
        text = expl if isinstance(expl, str) else str(expl)
        if not (HAS_FLOAT.search(text) or NAMES_LAUNCH.search(text) or NAMES_LAUNCH.search(str(orig))):
            return
        bucket = _assertions.setdefault(item.nodeid, [])
        if len(bucket) >= MAX_ASSERTIONS_PER_TEST:
            _dropped[item.nodeid] = _dropped.get(item.nodeid, 0) + 1
            return
        bucket.append({"line": lineno, "assertion": str(orig)[:MAX_TEXT],
                       "evaluated": text[:MAX_TEXT]})
    except Exception as error:  # noqa: BLE001 -- an observer must not fail a test
        _append({"kind": "error", "where": "pytest_assertion_pass", "error": repr(error)})


def pytest_runtest_logreport(report):
    config = _state["config"]
    if config is None or not _executes_tests(config):
        return
    if report.when != "call" and not (report.when == "setup" and report.outcome != "passed"):
        return
    try:
        _append({
            "kind": "test",
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration_s": round(report.duration, 4),
            "native_libraries_mapped": mapped_native_libraries(),
            "extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "assertions": _assertions.pop(report.nodeid, []),
            "assertions_dropped": _dropped.pop(report.nodeid, 0),
        })
    except Exception as error:  # noqa: BLE001
        _append({"kind": "error", "where": "pytest_runtest_logreport", "error": repr(error)})


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    try:
        _append({"kind": "session", "exitstatus": int(exitstatus),
                 "executes_tests": _executes_tests(config),
                 "native_libraries_mapped": mapped_native_libraries()})
    except Exception as error:  # noqa: BLE001
        _append({"kind": "error", "where": "pytest_sessionfinish", "error": repr(error)})
