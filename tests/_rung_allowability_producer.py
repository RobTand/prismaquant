"""Test-only external metadata producer; no serving package overlay or vendoring."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys

from functools import lru_cache
from pathlib import Path
import tempfile
from urllib.request import urlopen


def _verify_bytes(raw, digest, dependency):
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise ValueError(f"{dependency}: SHA-256 mismatch: expected {digest}, got {actual}")


@lru_cache(maxsize=1)
def _download_source(url, digest, dependency):
    try:
        with urlopen(url, timeout=60) as response:
            raw = response.read()
    except Exception as error:
        raise RuntimeError(
            f"cannot fetch declared dependency {dependency} from {url}: {error}; "
            "for offline use, set TESSERA_RUNG_ALLOWABILITY_MODULE and "
            "TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256 to authenticated source") from error
    _verify_bytes(raw, digest, dependency)
    directory = tempfile.TemporaryDirectory(prefix="pq-rung-metadata-")
    atexit.register(directory.cleanup)
    (Path(directory.name) / "rung_allowability.py").write_bytes(raw)
    return directory


def metadata_source():
    declared = json.loads((Path(__file__).parent / "fixtures" /
                           "rung_allowability_v3" / "producer.json").read_text())
    digest = declared["sha256"]
    dependency = (f"{declared['repository']}@{declared['commit']}:{declared['path']} "
                  f"(SHA-256 {digest})")
    module_var = "TESSERA_RUNG_ALLOWABILITY_MODULE"
    digest_var = "TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"
    if module_var in os.environ or digest_var in os.environ:
        supplied = os.environ.get(module_var)
        companion = os.environ.get(digest_var)
        selection = f"{dependency}; {module_var}={supplied!r}"
        if not supplied or not companion:
            raise ValueError(f"{selection}: supply both {module_var} and {digest_var}")
        if companion != digest:
            raise ValueError(f"{selection}: {digest_var}={companion!r} conflicts with declaration")
        path = Path(supplied)
    else:
        url = (f"https://raw.githubusercontent.com/{declared['repository']}/"
               f"{declared['commit']}/{declared['path']}")
        directory = _download_source(url, digest, dependency)
        path = Path(directory.name) / "rung_allowability.py"
        selection = dependency
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{selection}: cannot read source {path}: {error}") from error
    _verify_bytes(raw, digest, selection)
    return str(path), digest

_SCRIPT = r'''
import builtins, hashlib, importlib.abc, importlib.util, json, os, pathlib, sys
path = pathlib.Path(os.environ["TESSERA_RUNG_ALLOWABILITY_MODULE"])
source = path.read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
expected_sha = os.environ["TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"]
if source_sha != expected_sha:
    raise ValueError(f"producer {path}: SHA-256 mismatch: expected {expected_sha}, got {source_sha}")
attempted = set()
def forbidden_name(name):
    return (name.split(".")[0] in {"torch", "triton", "vllm"} or
            (name.split(".")[0] == "tessera" and name != "tessera.rung_allowability"))

def refuse_import(name):
    if forbidden_name(name):
        attempted.add(name)
        raise ImportError(f"forbidden producer import: {name}")

class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        refuse_import(fullname)

original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    resolved = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
    refuse_import(resolved)
    return original_import(name, globals, locals, fromlist, level)

def check_imports():
    forbidden = sorted(attempted | {name for name in sys.modules if forbidden_name(name)})
    if forbidden:
        raise ImportError(f"forbidden producer imports: {forbidden}")
    return forbidden

sys.meta_path.insert(0, ImportGuard())
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location("tessera.rung_allowability", path)
producer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = producer
exec(compile(source, str(path), "exec"), producer.__dict__)
check_imports()
request = json.load(sys.stdin)
try:
    result = getattr(producer, request["method"])(
        request["payload"], *request.get("args", []), **request["kwargs"])
    response = {"ok": True, "value": result}
except ValueError as error:
    response = {"ok": False, "error": str(error)}
forbidden = check_imports()
response.update(source_sha256=source_sha,
    interpreter_sha256=hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),
    forbidden_imports=forbidden)
print(json.dumps(response, allow_nan=False))
'''


class ExternalProducer:
    def __init__(self):
        self.evidence = None

    def _call(self, method, payload, *args, **kwargs):
        self.evidence = None
        path, digest = metadata_source()
        interpreter = os.environ.get("TESSERA_PRODUCER_PYTHON", sys.executable)
        try:
            completed = subprocess.run(
                [interpreter, "-I", "-c", _SCRIPT],
                input=json.dumps({"method": method, "payload": payload,
                                  "args": list(args), "kwargs": kwargs}),
                env={**os.environ, "TESSERA_RUNG_ALLOWABILITY_MODULE": path,
                     "TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256": digest},
                text=True, capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
            raise RuntimeError(
                f"canonical metadata producer {path} (SHA-256 {digest}) "
                f"failed with interpreter {interpreter}: {detail}") from error
        response = json.loads(completed.stdout)
        self.evidence = {key: response[key] for key in
                         ("source_sha256", "interpreter_sha256", "forbidden_imports")}
        if not response["ok"]:
            raise ValueError(response["error"])
        return response["value"]

    def validate_index(self, index):
        return self._call("validate_index", index)

    def validate_table(self, table):
        return self._call("validate_table", table)

    def admit_rung(self, table, **kwargs):
        return self._call("admit_rung", table, **kwargs)

    def __getattr__(self, method):
        """Forward newer owning calls (scope, speed, quality) unchanged."""
        def forward(payload, *args, **kwargs):
            return self._call(method, payload, *args, **kwargs)
        return forward
