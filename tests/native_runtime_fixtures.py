"""Small nonqualifying runtime observations for control-join unit fixtures.

These use actual import resolution and known on-disk bytes. They do not assert
that this artificial interpreter support is a production numerical runtime.
"""
import importlib
import sys
from pathlib import Path

import pytest

from prismaquant import native_results as nr


@pytest.fixture
def runtime_layout(tmp_path, monkeypatch):
    root = tmp_path / 'installed'
    contents = {
        'reviewpkg/__init__.py': b'from .numerics import answer\n',
        'reviewpkg/numerics.py': b'def answer(): return 42\n',
        'reviewpkg/nested/kernel.so': b'known extension bytes\x00',
        'reviewpkg/lib/libnumeric.so': b'known bundled library\xff',
        'reviewpkg/data/table.bin': b'\x01\x03\x05\x07',
        'reviewpkg/data/empty.bin': b'',
        'reviewpkg-1.dist-info/METADATA': b'Name: reviewpkg\nVersion: 1\n',
        'reviewpkg-1.dist-info/RECORD': (b'reviewpkg/__init__.py,,\nreviewpkg/numerics.py,,\n'
            b'reviewpkg-1.dist-info/METADATA,,\nreviewpkg-1.dist-info/RECORD,,\n'),
        'stdlib/numerical.py': b'answer = 42\n',
        'stdlib/lib-dynload/_numeric.so': b'known stdlib extension',
        'support/libpython.so': b'known libpython\x00',
        'support/libblas.so': b'known external numerical library\x00',
    }
    for name, raw in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    # Genuine importable package and real Distribution metadata. RECORD does
    # not list its nested extension/data; the tree census must discover them.
    monkeypatch.syspath_prepend(str(root))
    for name in ('reviewpkg', 'reviewpkg.numerics'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    package = importlib.import_module('reviewpkg')
    numerics = importlib.import_module('reviewpkg.numerics')
    monkeypatch.setitem(sys.modules, 'reviewpkg', package)
    monkeypatch.setitem(sys.modules, 'reviewpkg.numerics', numerics)
    assert package.answer() == 42
    real_imports = nr._runtime_imports
    real_libraries = nr._runtime_libraries
    # Scope only the current support inventory. File census, import binding,
    # stream digest and all fences remain the production implementation.
    imported = {'reviewpkg': nr._runtime_module_location(package),
                'reviewpkg.numerics': nr._runtime_module_location(numerics)}
    monkeypatch.setattr(nr, '_runtime_imports', lambda: {k: dict(v) for k, v in imported.items()})
    libraries = [root / 'support/libpython.so', root / 'support/libblas.so']
    monkeypatch.setattr(nr, '_runtime_libraries',
                        lambda: {str(p): nr._runtime_stat(p) for p in sorted(libraries)})
    monkeypatch.setattr(nr, '_runtime_standard_roots', lambda: [str(root / 'stdlib')])
    yield dict(root=root, contents=contents, package=package, numerics=numerics,
               imported=imported, real_imports=real_imports, real_libraries=real_libraries)
    sys.modules.pop('reviewpkg.numerics', None)
    sys.modules.pop('reviewpkg', None)


@pytest.fixture
def nonqualifying_runtime(runtime_layout, monkeypatch):
    runtime = nr.measure_runtime(dependencies=('reviewpkg',))
    # The scientific control tests exercise joins against an explicitly
    # nonqualifying CPU fixture, not the installed numerical closure.
    monkeypatch.setattr(nr, 'measure_runtime', lambda *, dependencies=(): runtime)
    return runtime
