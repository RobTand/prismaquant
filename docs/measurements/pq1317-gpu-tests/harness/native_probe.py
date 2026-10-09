"""Pytest plugin for the pq1317 qualification runs. It only observes.

It records, for each test process:

* the Tessera native libraries (``tessera_*.so``) the process had mapped after
  each test, read from ``/proc/self/maps``, each with its sha256; and
* the evaluated text of each passing assertion that holds a floating-point
  value or names a launch (a symbol, a decoder or a ``tessera::`` op), through
  pytest's ``pytest_assertion_pass`` hook.

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
NATIVE_PREFIX = "tessera_"
MAX_TEXT = 600
MAX_ASSERTIONS_PER_TEST = 64
#: A float literal or an exponent in the evaluated text: the assertion holds a number.
HAS_FLOAT = re.compile(r"\d+\.\d+|\d[eE][-+]?\d+")
#: The assertion names a launch: the route's symbol and decoder, or a Tessera op.
NAMES_LAUNCH = re.compile(r"symbol|decoder|launch_pair|tessera::|DENSE_")

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


def _digest(path: str) -> str:
    if path not in _digests:
        try:
            _digests[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as error:
            _digests[path] = f"unreadable: {error}"
    return _digests[path]


def mapped_native_libraries() -> dict[str, str]:
    """``{path: sha256}`` of the Tessera shared objects this process has mapped."""
    found: dict[str, str] = {}
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.split(None, 5)
                if len(fields) < 6:
                    continue
                path = fields[5].strip()
                if path.endswith(" (deleted)"):
                    path = path[: -len(" (deleted)")]
                name = os.path.basename(path)
                if name.startswith(NATIVE_PREFIX) and ".so" in name:
                    found[path] = _digest(path)
    except OSError as error:
        found["<unreadable /proc/self/maps>"] = str(error)
    return found


def pytest_configure(config):
    _state["config"] = config


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
        libraries = mapped_native_libraries()
        _append({
            "kind": "test",
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration_s": round(report.duration, 4),
            "native_libraries_mapped": libraries,
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
