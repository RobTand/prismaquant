"""Explicit installed producer dependency, separate from the serving test pin."""
from pathlib import Path
import json
import os
import socket
import stat
import subprocess

import pytest

from prismaquant.tessera_expert_projection import PRODUCER_PYTHON_ENV, producer_plan_tool

DECLARATION = Path(__file__).with_name('projection_producer_environment.json')
# Standard installed-distribution files, not a private Tessera runtime schema.
SOURCE_PROBE = r'''
from pathlib import Path
import hashlib, importlib.metadata as md, json, sys
import tessera.producer_plan as producer
prefix = Path(sys.prefix).resolve()
assert Path(producer.__file__).resolve().is_relative_to(prefix)
dist = md.distribution('tessera-quant')
digest = hashlib.sha256()
for file in sorted(dist.files, key=str):
    name = str(file)
    if name.startswith('tessera/') and Path(name).suffix in {'.py', '.json', '.cu'}:
        path = Path(dist.locate_file(file)).resolve()
        assert path.is_relative_to(prefix), str(path)
        digest.update(name.encode() + b'\0' + path.read_bytes())
print(json.dumps({'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
                  'module_sha256': hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
                  'package_payload_sha256': digest.hexdigest()}))
'''


def require_projection_producer(monkeypatch):
    """Authenticate the declared external bytes before selecting their real CLI.

    The declaration travels in the checkout snapshot with its producer host,
    executable digest and complete package-code digest. An absent interpreter
    is a named skip only off that host. Other stat errors, non-regular paths
    and changed installed dependencies fail everywhere.
    """
    declaration = json.loads(DECLARATION.read_text())
    assert declaration['schema'] == 'prismaquant.test_projection_producer.v1'
    interpreter = os.environ.get(PRODUCER_PYTHON_ENV) or declaration['interpreter']
    assert interpreter == declaration['interpreter'], 'selected producer path differs from sealed dependency'
    required_host = declaration['required_host']
    try:
        interpreter_stat = os.stat(interpreter)
    except FileNotFoundError:
        reason = f'missing declared projection producer interpreter {interpreter}; required on {required_host}'
        if socket.gethostname().split('.')[0] == required_host:
            pytest.fail(reason)
        pytest.skip(reason)
    except OSError as error:
        pytest.fail(f'cannot stat declared projection producer interpreter {interpreter}: '
                    f'{type(error).__name__}: {error}')
    if not stat.S_ISREG(interpreter_stat.st_mode):
        pytest.fail(f'declared projection producer interpreter {interpreter} is not a regular file')
    # Probe the same inherited environment as the real CLI, not a sanitized
    # interpreter whose import origin could differ from the subsequent request.
    probe = subprocess.run([interpreter, '-c', SOURCE_PROBE],
                           check=True, capture_output=True, text=True)
    observed = json.loads(probe.stdout)
    for key in ('executable_sha256', 'module_sha256', 'package_payload_sha256'):
        assert observed[key] == declaration[key], f'declared producer {key} drift'
    assert declaration['module'] == 'tessera.producer_plan'
    assert declaration['output_schema'] == 'tessera.expert_projection.v1'
    monkeypatch.setenv(PRODUCER_PYTHON_ENV, interpreter)
    monkeypatch.delenv('TESSERA_REPO', raising=False)
    assert producer_plan_tool() == declaration['module']
    return interpreter
