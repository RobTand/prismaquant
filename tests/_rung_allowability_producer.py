"""Test-only external metadata producer; no serving package overlay or vendoring."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from functools import lru_cache
from pathlib import Path
import tempfile
from urllib.request import urlopen


@lru_cache(maxsize=1)
def metadata_source():
    declared = json.loads((Path(__file__).parent / "fixtures" /
                           "rung_allowability_v3" / "producer.json").read_text())
    supplied = os.environ.get("TESSERA_RUNG_ALLOWABILITY_MODULE")
    if supplied:
        return supplied, os.environ["TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"]
    url = (f"https://raw.githubusercontent.com/{declared['repository']}/"
           f"{declared['commit']}/{declared['path']}")
    # Only the pure metadata module enters the isolated child process.
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != declared["sha256"]:
        raise ValueError("canonical metadata bytes differ from the declared dependency")
    target = Path(tempfile.mkdtemp(prefix="pq-rung-metadata-")) / "rung_allowability.py"
    target.write_bytes(raw)
    return str(target), declared["sha256"]

_SCRIPT = r'''
import hashlib, importlib.util, json, os, pathlib, sys
path = pathlib.Path(os.environ["TESSERA_RUNG_ALLOWABILITY_MODULE"])
source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
assert source_sha == os.environ["TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"], "producer bytes changed"
spec = importlib.util.spec_from_file_location("tessera.rung_allowability", path)
producer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)
forbidden = sorted(n for n in sys.modules if
    (n.startswith("tessera.") and n != spec.name) or
    n.split(".")[0] in {"torch", "triton", "vllm"})
assert not forbidden, forbidden
request = json.load(sys.stdin)
try:
    result = getattr(producer, request["method"])(
        request["payload"], *request.get("args", []), **request["kwargs"])
    response = {"ok": True, "value": result}
except ValueError as error:
    response = {"ok": False, "error": str(error)}
response.update(source_sha256=source_sha,
    interpreter_sha256=hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),
    forbidden_imports=forbidden)
print(json.dumps(response, allow_nan=False))
'''


class ExternalProducer:
    def __init__(self):
        self.evidence = None

    def _call(self, method, payload, *args, **kwargs):
        path, digest = metadata_source()
        completed = subprocess.run(
            [os.environ.get("TESSERA_PRODUCER_PYTHON", sys.executable), "-c", _SCRIPT],
            input=json.dumps({"method": method, "payload": payload,
                              "args": list(args), "kwargs": kwargs}),
            env={**os.environ, "TESSERA_RUNG_ALLOWABILITY_MODULE": path,
                 "TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256": digest},
            text=True, capture_output=True, check=True)
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
