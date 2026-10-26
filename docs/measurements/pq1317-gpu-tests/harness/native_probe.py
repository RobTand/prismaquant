"""Pytest plugin for the pq1317 qualification runs. It only observes.

It records, for each test process:

* the Tessera native libraries (``tessera_*.so``) the process has mapped after
  each test, read from ``/proc/self/maps``, each with a SHA-256 of its backing
  file. Readers must match the mapping's device/inode and record provenance.
  This identifies the backing file, not relocated or privately modified RAM; and
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
import stat
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

MAPPING_IDENTITY = ("address", "permissions", "offset", "device_major", "device_minor", "inode")
FILE_IDENTITY = ("device_major", "device_minor", "inode")
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
            major, minor = fields[3].split(":", 1)
            found.append({"module": module, "address": address, "path": path,
                          "permissions": fields[1], "offset": int(fields[2], 16),
                          "device_major": int(major, 16), "device_minor": int(minor, 16),
                          "inode": int(fields[4]),
                          "nfs_renamed": name.startswith(".nfs"), "deleted": deleted})
    return found


def _maps() -> list[dict]:
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
        return parse_maps(handle)


def _file_identity(info: os.stat_result) -> dict:
    return {"device_major": os.major(info.st_dev), "device_minor": os.minor(info.st_dev),
            "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def _digest(entry: dict) -> dict:
    """Read an opened backing file, never an unchecked pathname or cached digest.

    map_files can require privileges unavailable in a container. The fallback is
    safe only if fstat on the *opened descriptor* matches maps' device/inode.
    Check file stability and the mapping again after reading to reject races.
    Only acquisition failures permit another reader; all errors after a successful
    open are terminal, including failures before the first fstat is recorded.
    """
    record = {**entry, "sha256": None, "identity_verified": False, "reader": None,
              "attempts": []}
    for reader, path in (("map_files", f"/proc/self/map_files/{entry['address']}"),
                         ("pathname", entry["path"])):
        attempt = {"reader": reader, "path": path}
        record["attempts"].append(attempt)
        try:
            handle = Path(path).open("rb")
        except (OSError, ValueError) as error:
            attempt.update(error=str(error), error_stage="acquisition")
            continue
        try:
            with handle:
                before = os.fstat(handle.fileno())
                identity = _file_identity(before)
                attempt["opened_file"] = identity
                if not stat.S_ISREG(before.st_mode) or entry["inode"] == 0:
                    raise ValueError("mapping has no regular backing file")
                if any(identity[key] != entry[key] for key in FILE_IDENTITY):
                    raise ValueError("opened file device/inode differs from mapping")
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if _file_identity(os.fstat(handle.fileno())) != identity:
                    raise ValueError("backing file changed while hashing")
                if not any(all(current[key] == entry[key] for key in MAPPING_IDENTITY)
                           for current in _maps()):
                    raise ValueError("mapping changed while hashing")
            record.update(sha256=digest, identity_verified=True, reader=reader,
                          backing_file=identity)
            return record
        except (OSError, ValueError) as error:
            # Once open succeeds, even an fstat/read/maps/close error is terminal.
            # A later stable read cannot erase an observed integrity failure.
            attempt.update(error=str(error), error_stage="integrity")
            record["error"] = "mapped backing-file integrity check failed"
            return record
    record["error"] = "no reader established mapped backing-file identity"
    return record


def mapped_native_libraries() -> dict[str, dict]:
    """Preserve every segment; a failed or conflicting identity has no digest."""
    try:
        entries = _maps()
    except (OSError, ValueError) as error:
        return {"<unreadable /proc/self/maps>": {"identity_verified": False, "sha256": None,
                                               "error": str(error)}}
    libraries: dict[str, dict] = {}
    for entry in entries:
        info = libraries.setdefault(entry["module"], {
            "path": entry["path"], "nfs_renamed": entry["nfs_renamed"],
            "deleted": entry["deleted"], "mappings": []})
        info["mappings"].append(_digest(entry))
    for info in libraries.values():
        mappings = info["mappings"]
        identities = {tuple(m[key] for key in FILE_IDENTITY) for m in mappings}
        digests = {m["sha256"] for m in mappings}
        verified = (all(m["identity_verified"] for m in mappings)
                    and len(identities) == len(digests) == 1)
        info.update(identity_verified=verified, sha256=mappings[0]["sha256"] if verified else None)
        if not verified:
            info["error"] = "unverified or conflicting mapped backing files"
    return libraries


def _valid_attempts(mapping: dict, backing: dict) -> bool:
    """Require one final success, preceded only by a failed acquisition.

    Old records lack error_stage. Accept their ordinary permission/missing-file
    failures only when no opened descriptor identity was recorded; unknown old
    failures are ambiguous and must not be used to justify a later success.
    """
    attempts = mapping.get("attempts")
    if not isinstance(attempts, list) or not attempts or len(attempts) > 2:
        return False
    readers = ["map_files"] if len(attempts) == 1 else ["map_files", "pathname"]
    for index, (attempt, reader) in enumerate(zip(attempts, readers)):
        if not isinstance(attempt, dict) or attempt.get("reader") != reader:
            return False
        if index == len(attempts) - 1:
            if (reader != mapping.get("reader") or "error" in attempt
                    or "error_stage" in attempt or attempt.get("opened_file") != backing):
                return False
        else:
            error = attempt.get("error")
            stage = attempt.get("error_stage")
            if "opened_file" in attempt or not isinstance(error, str) or not error:
                return False
            if stage != "acquisition":
                if stage is not None or not re.match(r"\[Errno (?:1|2|13)\] ", error):
                    return False
    return True


def native_identity_errors(libraries: dict) -> list[str]:
    """Qualification gate, including refusal of legacy records without provenance."""
    errors = []
    for module, info in libraries.items():
        mappings = info.get("mappings", [])
        digest = info.get("sha256")
        valid = (info.get("identity_verified") is True and isinstance(digest, str)
                 and re.fullmatch(r"[0-9a-f]{64}", digest) and bool(mappings))
        identities = set()
        for mapping in mappings:
            backing = mapping.get("backing_file", {})
            identities.add(tuple(mapping.get(key) for key in FILE_IDENTITY))
            valid = (valid and mapping.get("identity_verified") is True
                     and mapping.get("sha256") == digest
                     and mapping.get("reader") in ("map_files", "pathname")
                     and mapping.get("inode", 0) > 0
                     and all(key in backing and backing[key] == mapping.get(key)
                             for key in FILE_IDENTITY)
                     and _valid_attempts(mapping, backing))
        if not valid or len(identities) != 1:
            errors.append(f"{module}: mapped backing-file identity is unverified")
    return errors


def pytest_configure(config):
    _state["config"] = config
    root = os.environ.get(EXT_DIR_ROOT_ENV)
    if root and hasattr(config, "workerinput"):
        directory = Path(root) / _worker(config)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(directory)


def pytest_assertion_pass(item, lineno, orig, expl):
    try:
        from bf16_oracle import observe_bf16_assertion

        numerical = observe_bf16_assertion(item, lineno, str(orig))
        text = expl if isinstance(expl, str) else str(expl)
        if numerical is None and not (HAS_FLOAT.search(text) or NAMES_LAUNCH.search(text)
                                      or NAMES_LAUNCH.search(str(orig))):
            return
        bucket = _assertions.setdefault(item.nodeid, [])
        if len(bucket) >= MAX_ASSERTIONS_PER_TEST:
            _dropped[item.nodeid] = _dropped.get(item.nodeid, 0) + 1
            return
        record = {"line": lineno, "assertion": str(orig)[:MAX_TEXT], "evaluated": text[:MAX_TEXT]}
        if numerical is not None:
            record["numerical"] = numerical
        bucket.append(record)
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
