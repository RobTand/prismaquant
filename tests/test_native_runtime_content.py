"""Real finite content and import-owner checks for native runtime v5."""
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


def _forbid_eager_metadata(*args, **kwargs):
    raise AssertionError('unbounded distribution metadata helper entered')


def test_runtime_metadata_uses_bounded_actual_reader(runtime_layout, monkeypatch):
    positive = observe()
    assert positive['dependencies']['reviewpkg']['version'] == '1'
    expected = runtime_layout['contents']['reviewpkg-1.dist-info/METADATA']
    assert positive['files'][str(runtime_layout['root'] / 'reviewpkg-1.dist-info/METADATA')]['sha256'] == hashlib.sha256(expected).hexdigest()
    calls = []
    reader = nr.read_document
    def bounded(path, limit):
        calls.append((Path(path).name, limit))
        assert 0 < limit <= 12 << 20
        return reader(path, limit)
    monkeypatch.setattr(nr, 'read_document', bounded)
    monkeypatch.setattr(importlib.metadata.PathDistribution, 'read_text', _forbid_eager_metadata)
    assert observe() == positive
    assert {name for name, limit in calls} == {'METADATA', 'RECORD'}


@pytest.mark.parametrize('name', ['METADATA', 'RECORD'])
@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'oversize', 'replacement'])
def test_runtime_metadata_refuses_before_unbounded_helper(runtime_layout, monkeypatch, name, kind):
    assert observe()['dependencies']['reviewpkg']['version'] == '1'
    target = runtime_layout['root'] / ('reviewpkg-1.dist-info/' + name)
    monkeypatch.setattr(importlib.metadata.PathDistribution, 'read_text', _forbid_eager_metadata)
    if kind == 'fifo':
        target.unlink(); os.mkfifo(target)
    elif kind == 'symlink':
        alternate = target.with_name(name + '.held')
        target.rename(alternate); target.symlink_to(alternate)
    elif kind == 'oversize':
        monkeypatch.setattr(nr, 'MAX_RUNTIME_METADATA_BYTES', 16)
        assert target.stat().st_size > 16
    else:
        reader = nr.read_document
        fired = []
        def replaced(path, limit):
            value = reader(path, limit)
            if Path(path) == target and not fired:
                alternate = target.with_name(name + '.replacement')
                alternate.write_bytes(target.read_bytes())
                assert alternate.stat().st_ino != target.stat().st_ino
                alternate.replace(target)
                fired.append(True)
            return value
        monkeypatch.setattr(nr, 'read_document', replaced)
    with pytest.raises((OSError, ValueError)):
        nr._runtime_distribution('reviewpkg')


def test_runtime_metadata_record_rows_are_bounded(runtime_layout, monkeypatch):
    assert observe()['dependencies']['reviewpkg']['version'] == '1'
    target = runtime_layout['root'] / 'reviewpkg-1.dist-info/RECORD'
    target.write_bytes(b'reviewpkg/__init__.py,,\n' * 9)
    monkeypatch.setattr(nr, 'MAX_RUNTIME_ENTRIES', 8)
    monkeypatch.setattr(importlib.metadata.PathDistribution, 'read_text', _forbid_eager_metadata)
    with pytest.raises(ValueError, match='count|bound'):
        nr._runtime_distribution('reviewpkg')


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
                previous = target.stat()
                target.write_bytes(b'!' * target.stat().st_size)
                os.utime(target, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
                assert nr._runtime_stat(target) != nr.file_stat_signature(previous)
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


@pytest.mark.parametrize('schema', ['prismaquant.flux_result_runtime.v1',
    'prismaquant.flux_result_runtime.v2', 'prismaquant.flux_result_runtime.v3',
    'prismaquant.flux_result_runtime.v4'])
def test_old_runtime_evidence_is_not_restamped(runtime_layout, schema):
    runtime = observe()
    old = copy.deepcopy(runtime)
    old['schema'] = schema
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
    monkeypatch.setitem(sys.modules, 'reviewhelper', None)
    monkeypatch.delitem(sys.modules, 'reviewhelper')
    module = importlib.import_module('reviewhelper')
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
    cache.parent.mkdir(exist_ok=True)
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


def test_aggregate_directory_count_is_bounded_across_small_roots(runtime_layout, monkeypatch):
    runtime = observe()
    assert len(runtime['stdlib']) == 1
    # Each tree fits separately. Their combined known directory/file census
    # exceeds eleven before any native file needs to be streamed.
    for root in [runtime_layout['root'] / 'reviewpkg', runtime_layout['root'] / 'reviewpkg-1.dist-info', runtime_layout['root'] / 'stdlib']:
        files, directories = nr._runtime_tree(str(root))
        assert len(files) + len(directories) <= 11
    monkeypatch.setattr(nr, 'MAX_RUNTIME_ENTRIES', 11)
    with pytest.raises(ValueError, match='aggregate directory/file count'):
        observe()


def test_nonzero_per_file_ceiling_refuses_known_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'library.so'
    path.write_bytes(b'0123456789abcdef')
    expected = nr._runtime_stat(path)
    assert nr._runtime_file(str(path), expected)['sha256'] == hashlib.sha256(b'0123456789abcdef').hexdigest()
    monkeypatch.setattr(nr, 'MAX_RUNTIME_FILE_BYTES', 15)
    with pytest.raises(ValueError, match='byte bound'):
        nr._runtime_file(str(path), expected)


def test_directory_replacement_during_hash_refuses(runtime_layout, monkeypatch):
    observe()
    root = runtime_layout['root']
    original = nr.file_digest_sha256hex
    fired = False
    def replace(handle, **kwargs):
        nonlocal fired
        result = original(handle, **kwargs)
        if not fired and os.readlink(f'/proc/self/fd/{handle.fileno()}').endswith('/reviewpkg/numerics.py'):
            fired = True
            old = root / 'reviewpkg/data'
            replacement = root / 'replacement-data'
            replacement.mkdir()
            (replacement/'empty.bin').write_bytes(b'')
            (replacement/'table.bin').write_bytes(b'\x01\x03\x05\x07')
            assert old.stat().st_ino != replacement.stat().st_ino
            old.rename(root/'held-original-data')
            replacement.rename(old)
        return result
    monkeypatch.setattr(nr, 'file_digest_sha256hex', replace)
    with pytest.raises(ValueError, match='roster changed'):
        observe()
    assert fired


def test_imported_spec_and_file_must_have_same_actual_owner(runtime_layout, monkeypatch):
    module = runtime_layout['numerics']
    assert nr._runtime_module_location(module)['origin'].endswith('/reviewpkg/numerics.py')
    monkeypatch.setattr(module, '__file__', str(runtime_layout['root'] / 'support/libblas.so'))
    with pytest.raises(ValueError, match='file/spec owner differs'):
        nr._runtime_module_location(module)


@pytest.mark.parametrize('mutation', ['named-code', 'executable-stack', 'unnamed-code',
    'wrong-device', 'wrong-inode', 'deleted', 'added-library'])
def test_native_mapping_owner_parser_refuses_substitution(runtime_layout, monkeypatch, mutation):
    import builtins
    root = runtime_layout['root']
    library = root / 'support/libpython.so'
    known = library.stat()
    device = f'{os.major(known.st_dev):02x}:{os.minor(known.st_dev):02x}'
    prefix = f'1000-2000 r-xp 00000000 {device} {known.st_ino} '
    mapping = prefix + str(library) + '\n'
    raw = mapping.encode()
    original_open = builtins.open
    def mapped_open(path, *args, **kwargs):
        if path == '/proc/self/maps':
            return io.BytesIO(raw)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', mapped_open)
    monkeypatch.setattr(nr, '_runtime_mapped_stat', lambda address: None)
    # These synthetic maps use stat's device, independent of the host's
    # actual mount device. Full fdinfo/mountinfo joins have their own tests.
    monkeypatch.setattr(nr, '_runtime_mapped_device', lambda fd:
        (os.major(os.fstat(fd).st_dev), os.minor(os.fstat(fd).st_dev)))
    real = runtime_layout['real_libraries']
    assert real() == {str(library):nr.file_stat_signature(known)}
    if mutation == 'named-code':
        raw += b'3000-4000 r-xp 00000000 00:00 0 [anon:numerical-code]\n'
    elif mutation == 'executable-stack':
        raw += b'3000-4000 rwxp 00000000 00:00 0 [stack]\n'
    elif mutation == 'unnamed-code':
        raw += b'3000-4000 r-xp 00000000 00:00 0\n'
    elif mutation == 'wrong-device':
        raw = f'1000-2000 r-xp 00000000 00:00 {known.st_ino} {library}\n'.encode()
    elif mutation == 'wrong-inode':
        raw = f'1000-2000 r-xp 00000000 {device} {known.st_ino+1} {library}\n'.encode()
    elif mutation == 'deleted':
        raw = (mapping.rstrip() + ' (deleted)\n').encode()
    else:
        extra = root / 'support/libblas.so'
        info = extra.stat()
        raw += f'3000-4000 r-xp 00000000 {device} {info.st_ino} {extra}\n'.encode()
        assert set(real()) == {str(library), str(extra)}
        return
    with pytest.raises(ValueError, match='unsupported|replaced'):
        real()


def test_editable_path_injection_is_not_a_complete_distribution_owner(runtime_layout):
    observe()
    root = runtime_layout['root']
    (root/'reviewpkg-editable.pth').write_bytes(b'import unmeasured_editable_finder\n')
    with (root/'reviewpkg-1.dist-info/RECORD').open('ab') as record:
        record.write(b'reviewpkg-editable.pth,,\n')
    with pytest.raises(ValueError, match='unsupported editable/path-injection owner'):
        observe()


def test_specless_file_cannot_claim_an_imported_implementation(runtime_layout, monkeypatch):
    module = runtime_layout['numerics']
    assert nr._runtime_module_location(module)['origin'].endswith('/reviewpkg/numerics.py')
    monkeypatch.setattr(module,'__spec__',None)
    with pytest.raises(ValueError, match='unsupported specless'):
        nr._runtime_module_location(module)


@pytest.mark.parametrize('incoming_owner', ['absent', 'existing'])
def test_runtime_fixture_restores_incoming_import_owners(tmp_path, monkeypatch, incoming_owner):
    from types import ModuleType

    names = ('reviewpkg', 'reviewpkg.numerics')
    original = {name: ModuleType(name) for name in names}
    for name in names:
        monkeypatch.setitem(sys.modules, name, original[name])
        if incoming_owner == 'absent':
            monkeypatch.delitem(sys.modules, name)
    original_imports = nr._runtime_imports
    original_path = list(sys.path)
    inner = pytest.MonkeyPatch()
    fixture = runtime_layout.__wrapped__(tmp_path, inner)
    try:
        layout = next(fixture)
        assert layout['package'].answer() == 42
        for name in names:
            assert sys.modules[name] is not original[name]
    finally:
        fixture.close()
        inner.undo()
    for name in names:
        if incoming_owner == 'absent':
            assert name not in sys.modules
        else:
            assert sys.modules[name] is original[name]
    assert nr._runtime_imports is original_imports
    assert sys.path == original_path


def test_transient_roster_mutation_refuses_after_names_are_restored(runtime_layout, monkeypatch):
    before = observe()
    directory = runtime_layout['root'] / 'reviewpkg/data'
    original_names = sorted(p.name for p in directory.iterdir())
    original_signature = nr.file_stat_signature(directory.stat())
    digest = nr.file_digest_sha256hex
    fired = False

    def transient_mutation(handle, **kwargs):
        nonlocal fired
        result = digest(handle, **kwargs)
        if not fired:
            fired = True
            temporary = directory / 'transient.py'
            temporary.write_bytes(b'answer = 43\n')
            temporary.unlink()
            # Set an independently distinct timestamp; filesystem clock
            # resolution cannot decide whether this regression exercises it.
            current = directory.stat()
            os.utime(directory, ns=(current.st_atime_ns, original_signature[3] + 1_000_000_000))
            assert sorted(p.name for p in directory.iterdir()) == original_names
            assert nr.file_stat_signature(directory.stat()) != original_signature
        return result

    assert before['files'][str(directory / 'table.bin')]['sha256'] == hashlib.sha256(b'\x01\x03\x05\x07').hexdigest()
    monkeypatch.setattr(nr, 'file_digest_sha256hex', transient_mutation)
    with pytest.raises(ValueError, match='roster changed'):
        observe()
    assert fired
