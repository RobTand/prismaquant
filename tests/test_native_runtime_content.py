"""Real finite content and import-owner checks for native runtime v2."""
import copy
import hashlib
import importlib
import io
import os
from pathlib import Path
import sys

import pytest

from prismaquant import native_results as nr
from prismaquant.digests import FILE_BLOCK_BYTES, file_digest_sha256hex
from native_runtime_fixtures import runtime_layout  # noqa: F401


def observe():
    return nr.measure_runtime(dependencies=('reviewpkg',))


def test_known_complete_contents_and_imported_owner(runtime_layout):
    layout = runtime_layout
    runtime = observe()
    assert nr.validate_runtime(runtime) is runtime
    for name, raw in layout['contents'].items():
        assert runtime['files'][str(layout['root'] / name)] == {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert runtime['imports']['reviewpkg.numerics']['origin'] == str(layout['root'] / 'reviewpkg/numerics.py')
    assert runtime['dependencies']['reviewpkg']['imports']['reviewpkg']['origin'] == str(
        layout['root'] / 'reviewpkg/__init__.py')
    assert runtime['files'][str(layout['root'] / 'reviewpkg/data/empty.bin')]['bytes'] == 0
    assert observe() == runtime


@pytest.mark.parametrize('name', ['reviewpkg/numerics.py', 'reviewpkg/nested/kernel.so',
    'reviewpkg/lib/libnumeric.so', 'reviewpkg/data/table.bin', 'stdlib/numerical.py',
    'stdlib/lib-dynload/_numeric.so', 'support/libpython.so', 'support/libblas.so'])
def test_each_actual_implementation_axis_changes_identity(runtime_layout, name):
    before = observe()
    root = runtime_layout['root']
    anchors = {str(p): p.read_bytes() for p in [root / 'reviewpkg/__init__.py',
        root / 'reviewpkg-1.dist-info/METADATA', root / 'reviewpkg-1.dist-info/RECORD']}
    (root / name).write_bytes(b'different independently known arithmetic bytes\x00')
    after = observe()
    assert after != before
    assert after['files'][str(root / name)]['sha256'] == hashlib.sha256(
        b'different independently known arithmetic bytes\x00').hexdigest()
    assert {p: Path(p).read_bytes() for p in anchors} == anchors


@pytest.mark.parametrize('operation', ['add-python', 'add-data', 'remove', 'empty-directory'])
def test_closed_roster_changes_without_record_change(runtime_layout, operation):
    before = observe()
    root = runtime_layout['root']
    if operation == 'remove':
        (root / 'reviewpkg/data/table.bin').unlink()
    elif operation == 'empty-directory':
        (root / 'reviewpkg/new').mkdir()
        # Empty dirs carry no numerical bytes; a future file is measured.
        (root / 'reviewpkg/new/added.py').write_bytes(b'answer = 43\n')
    else:
        name = 'added.py' if operation == 'add-python' else 'added.bin'
        (root / 'reviewpkg' / name).write_bytes(b'new numerical content')
    assert observe() != before


@pytest.mark.parametrize('axis', ['resolution', 'already-imported', 'nested-imported', 'search-path'])
def test_foreign_import_roots_refuse(runtime_layout, tmp_path, monkeypatch, axis):
    before = observe()
    assert before['dependencies']['reviewpkg']['version'] == '1'
    foreign = tmp_path / 'foreign/reviewpkg'
    foreign.mkdir(parents=True)
    (foreign / '__init__.py').write_bytes(b'answer = 99\n')
    (foreign / 'numerics.py').write_bytes(b'answer = 99\n')
    if axis == 'resolution':
        monkeypatch.syspath_prepend(str(foreign.parent))
    else:
        imported = runtime_layout['imported']
        name = 'reviewpkg.numerics' if axis == 'nested-imported' else 'reviewpkg'
        imported[name] = dict(imported[name])
        if axis == 'search-path':
            imported[name]['locations'] = [str(foreign)]
        else:
            imported[name]['origin'] = str(foreign / ('numerics.py' if axis == 'nested-imported' else '__init__.py'))
    with pytest.raises(ValueError, match='foreign|owner differs'):
        observe()


@pytest.mark.parametrize('axis', ['new-import', 'new-library'])
def test_newly_loaded_implementation_cannot_remain_equal(runtime_layout, axis):
    before = observe()
    root = runtime_layout['root']
    added = root / 'support/new.py'
    added.write_bytes(b'new implementation\n')
    if axis == 'new-import':
        runtime_layout['imported']['new_implementation'] = {
            'origin': str(added), 'locations': [], 'loader': 'source', 'cached': None}
    else:
        # A newly loaded external owner is separately observed, not inferred
        # from a distribution version or from numerical outputs.
        old = nr._runtime_libraries
        nr._runtime_libraries = lambda: {**old(), str(added): nr._runtime_stat(added)}
    try:
        assert observe() != before
    finally:
        if axis == 'new-library':
            nr._runtime_libraries = old


@pytest.mark.parametrize('axis', ['missing', 'symlink', 'directory-symlink', 'fifo', 'escaped-record'])
def test_missing_escaped_and_nonregular_members_fail_closed(runtime_layout, monkeypatch, axis):
    observe()
    root = runtime_layout['root']
    target = root / 'reviewpkg/numerics.py'
    if axis == 'missing':
        target.unlink()
    elif axis == 'symlink':
        target.unlink(); target.symlink_to(root / 'stdlib/numerical.py')
    elif axis == 'directory-symlink':
        (root / 'reviewpkg/escape').symlink_to(root / 'support', target_is_directory=True)
    elif axis == 'fifo':
        os.mkfifo(root / 'reviewpkg/pipe')
    else:
        (root / 'reviewpkg-1.dist-info/RECORD').write_text('../../escaped.py,,\n')
    with pytest.raises((OSError, ValueError)):
        observe()


@pytest.mark.parametrize('limit', ['MAX_RUNTIME_DEPENDENCIES', 'MAX_RUNTIME_ROOTS',
    'MAX_RUNTIME_ENTRIES', 'MAX_RUNTIME_FILE_BYTES', 'MAX_RUNTIME_BYTES',
    'MAX_RUNTIME_IMPORTS', 'MAX_RUNTIME_DEPTH', 'MAX_RUNTIME_DOCUMENT_BYTES'])
def test_hard_finite_budgets_refuse(runtime_layout, monkeypatch, limit):
    runtime = observe()
    assert runtime['files'][str(runtime_layout['root'] / 'reviewpkg/data/table.bin')]['bytes'] == 4
    monkeypatch.setattr(nr, limit, 0)
    with pytest.raises(ValueError, match='bound|finite|budget|byte|count|depth|roster'):
        observe()


def test_aggregate_budget_is_not_only_per_file(runtime_layout, monkeypatch):
    runtime = observe()
    total = sum(row['bytes'] for row in runtime['files'].values())
    assert total > max(row['bytes'] for row in runtime['files'].values())
    monkeypatch.setattr(nr, 'MAX_RUNTIME_BYTES', total - 1)
    with pytest.raises(ValueError, match='aggregate'):
        observe()


@pytest.mark.parametrize('axis', ['replace', 'grow', 'same-size-write', 'add', 'remove', 'import'])
def test_file_and_roster_races_refuse(runtime_layout, monkeypatch, axis):
    observe()
    root = runtime_layout['root']
    target = root / 'reviewpkg/numerics.py'
    digest = nr.file_digest_sha256hex
    fired = False

    def racing_digest(handle, **kwargs):
        nonlocal fired
        result = digest(handle, **kwargs)
        if not fired and Path(handle.name).name == target.name:
            fired = True
            if axis == 'replace':
                replacement = root / 'replacement'
                replacement.write_bytes(target.read_bytes())
                assert replacement.stat().st_ino != target.stat().st_ino
                replacement.replace(target)
            elif axis == 'grow':
                with target.open('ab') as writer:
                    writer.write(b'extra')
            elif axis == 'same-size-write':
                target.write_bytes(b'!' * target.stat().st_size)
            elif axis == 'add':
                (root / 'reviewpkg/new.py').write_bytes(b'new file')
            elif axis == 'remove':
                (root / 'reviewpkg/data/table.bin').unlink()
            else:
                runtime_layout['imported']['late_module'] = {
                    'origin': str(target), 'locations': [], 'loader': 'source', 'cached': None}
        return result

    # fdopen exposes an integer name, so identify its actual open descriptor.
    def descriptor_digest(handle, **kwargs):
        class LocatedHandle:
            name = os.readlink(f'/proc/self/fd/{handle.fileno()}')
            def __getattr__(self, name):
                return getattr(handle, name)
        return racing_digest(LocatedHandle(), **kwargs)

    monkeypatch.setattr(nr, 'file_digest_sha256hex', descriptor_digest)
    with pytest.raises((ValueError, OSError), match='changed|raced|No such file'):
        observe()
    assert fired


def test_bounded_stream_hash_allocates_chunks_and_refuses_growth():
    raw = b'abc12345' * (FILE_BLOCK_BYTES // 8 + 2)
    class Bounded(io.BytesIO):
        def read(self, n=-1):
            assert 0 < n <= FILE_BLOCK_BYTES
            return super().read(n)
    assert file_digest_sha256hex(Bounded(raw), max_bytes=len(raw)) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='ceiling'):
        file_digest_sha256hex(Bounded(raw + b'x'), max_bytes=len(raw))
    assert file_digest_sha256hex(Bounded(b''), max_bytes=0) == hashlib.sha256(b'').hexdigest()


def test_old_runtime_evidence_is_not_restamped(runtime_layout):
    runtime = observe()
    old = copy.deepcopy(runtime)
    old['schema'] = 'prismaquant.flux_result_runtime.v1'
    with pytest.raises(ValueError, match='schema'):
        nr.validate_runtime(old)


def test_nested_dependency_code_changes_runtime_identity(runtime_layout):
    original = observe()
    assert nr.validate_runtime(original) == original
    # Preserve the parent's exact nested-byte mutation and assertion. The
    # fixture now establishes a real importable installed owner first.
    (runtime_layout['root'] / 'reviewpkg/numerics.py').write_bytes(b'def answer(): return 43\n')
    changed = observe()
    assert changed != original, 'runtime seal missed changed numerical implementation'


def test_declared_distribution_alias_uses_only_its_fully_measured_roots(runtime_layout, monkeypatch):
    root = runtime_layout['root']
    helper = root / 'reviewhelper.py'
    raw = b'answer = 84\n'
    helper.write_bytes(raw)
    with (root / 'reviewpkg-1.dist-info/RECORD').open('ab') as record:
        record.write(b'reviewhelper.py,,\n')
    monkeypatch.delitem(sys.modules, 'reviewhelper', raising=False)
    module = importlib.import_module('reviewhelper')
    monkeypatch.setitem(sys.modules, 'reviewhelper', module)
    assert module.answer == 84
    runtime_layout['imported']['reviewpkg.helper_alias'] = nr._runtime_module_location(module)
    runtime = observe()
    assert runtime['files'][str(helper)]['sha256'] == hashlib.sha256(raw).hexdigest()
    assert runtime['imports']['reviewpkg.helper_alias']['origin'] == str(helper)
    assert str(helper) in runtime['dependencies']['reviewpkg']['roots']
    runtime_layout['imported']['reviewpkg.helper_alias']['origin'] = str(root / 'support/libblas.so')
    with pytest.raises(ValueError, match='foreign'):
        observe()
    sys.modules.pop('reviewhelper', None)


def test_actual_descriptor_stream_peak_is_bounded_below_native_file_size(tmp_path):
    import tracemalloc
    path = tmp_path / 'large-native.so'
    pattern = b'native library bytes\x00' * 4096
    expected = hashlib.sha256()
    size = 0
    with path.open('wb') as stream:
        while size < 64 << 20:
            stream.write(pattern); expected.update(pattern); size += len(pattern)
    signature = nr._runtime_stat(path)
    tracemalloc.start()
    try:
        binding = nr._runtime_file(str(path), signature)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert binding == {'bytes':size,'sha256':expected.hexdigest()}
    assert size > 64 << 20
    assert peak < 24 << 20, peak


def test_growth_during_bounded_descriptor_read_refuses(runtime_layout, monkeypatch):
    observe()
    target = runtime_layout['root'] / 'reviewpkg/numerics.py'
    original = nr.file_digest_sha256hex
    def grow(handle, **kwargs):
        class GrowingHandle:
            fired = False
            def read(self, size=-1):
                chunk = handle.read(size)
                if not self.fired and chunk:
                    self.fired = True
                    with target.open('ab') as writer: writer.write(b'grew during read')
                return chunk
        if os.readlink(f'/proc/self/fd/{handle.fileno()}') == str(target):
            return original(GrowingHandle(), **kwargs)
        return original(handle, **kwargs)
    monkeypatch.setattr(nr, 'file_digest_sha256hex', grow)
    with pytest.raises(ValueError, match='ceiling'):
        observe()


def test_unsupported_loader_refuses_real_imported_owner(runtime_layout, monkeypatch):
    module = runtime_layout['numerics']
    assert nr._runtime_module_location(module)['origin'].endswith('/reviewpkg/numerics.py')
    monkeypatch.setattr(module.__spec__, 'loader', object())
    with pytest.raises(ValueError, match='unsupported imported loader'):
        nr._runtime_module_location(module)


def test_existing_loaded_bytecode_is_separately_bound(runtime_layout):
    original = observe()
    cache = runtime_layout['root'] / 'reviewpkg/__pycache__/numerics.cpython-312.pyc'
    cache.parent.mkdir()
    cache.write_bytes(b'known executable cache bytes')
    runtime_layout['imported']['reviewpkg.numerics']['cached'] = str(cache)
    before = observe()
    assert before != original
    assert before['files'][str(cache)]['sha256'] == hashlib.sha256(b'known executable cache bytes').hexdigest()
    cache.write_bytes(b'changed executable cache bytes')
    assert observe() != before


def test_claimed_interpreter_must_be_the_actual_executable(runtime_layout, monkeypatch):
    observe()
    monkeypatch.setattr(sys, 'executable', str(runtime_layout['root'] / 'support/libpython.so'))
    with pytest.raises(ValueError, match='actual executing owner'):
        observe()
