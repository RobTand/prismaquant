"""A declared reader cannot replace or import the primary producer package."""
import hashlib
import sys
from pathlib import Path

import pytest


def package(tmp_path, *, escaping=False):
    root = tmp_path / 'tessera'; root.mkdir()
    (root / '__init__.py').write_text('MARKER = "reader"\n')
    (root / 'cached_unit.py').write_text('def verify_cached_unit(*args): return __name__, args\n')
    (root / 'unit_artifact.py').write_text(
        ('import tessera\n' if escaping else '') +
        'def read_unit_artifact(*args, **kwargs): return __name__, args, kwargs\n')
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes()); digest.update(b'\0')
    return {'path': str(root), 'source_sha256': digest.hexdigest()}


def test_declared_reader_uses_shared_sha256_grammar(tmp_path, monkeypatch):
    __tracebackhide__ = True
    from prismaquant.tessera_reader import load_declared_reader
    consumer = sys.modules['prismaquant.tessera_reader']
    owner = sys.modules['prismaquant.digests']
    declared = package(tmp_path)
    root = Path(declared['path'])
    (root / '__init__.py').write_text('MARKER = "grammar-routing"\n')
    profiles, _ = consumer._source_tree_profiles(root)
    declared['source_sha256'] = profiles[owner.SOURCE_TREE_V1]
    namespace = 'tessera_reader_' + profiles[owner.SOURCE_TREE_V2]
    assert namespace not in sys.modules
    real_predicate = owner.is_sha256hex
    calls = []

    def record(value):
        calls.append(value)
        return real_predicate(value)

    monkeypatch.setattr(owner, 'is_sha256hex', record, raising=False)
    monkeypatch.setattr(consumer, 'is_sha256hex', record, raising=False)
    try:
        reader = load_declared_reader(declared)
        assert reader.identity['namespace'] == namespace
        assert calls == [declared['source_sha256'],
                         profiles[owner.SOURCE_TREE_V1],
                         profiles[owner.SOURCE_TREE_V2],
                         declared['source_sha256']], 'wrapper did not record the owner calls'
    finally:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + '.'):
                del sys.modules[name]
        sys.meta_path[:] = [finder for finder in sys.meta_path
                            if getattr(finder, 'namespace', None) != namespace]


@pytest.fixture
def grammar_package(tmp_path):
    from prismaquant.tessera_reader import load_declared_reader
    consumer = sys.modules[load_declared_reader.__module__]
    owner = sys.modules['prismaquant.digests']
    declared = package(tmp_path)
    root = Path(declared['path'])
    (root / '__init__.py').write_text('MARKER = "grammar-controls"\n')
    profiles, _ = consumer._source_tree_profiles(root)
    declared['source_sha256'] = profiles[owner.SOURCE_TREE_V1]
    namespace = 'tessera_reader_' + profiles[owner.SOURCE_TREE_V2]
    assert namespace not in sys.modules
    try:
        yield declared
    finally:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + '.'):
                del sys.modules[name]
        sys.meta_path[:] = [finder for finder in sys.meta_path
                            if getattr(finder, 'namespace', None) != namespace]


@pytest.mark.parametrize('kind', ['plain', 'subclass'])
def test_reader_accepts_valid_sha256_grammar(grammar_package, kind):
    import tessera
    from prismaquant.tessera_reader import load_declared_reader

    class DigestText(str):
        def __len__(self):
            return 0

        def __iter__(self):
            raise AssertionError('do not iterate the declaration')

    declared = grammar_package
    digest = declared['source_sha256']
    if kind == 'subclass':
        declared['source_sha256'] = DigestText(digest)
    primary = {name: module for name, module in sys.modules.items()
               if name == 'tessera' or name.startswith('tessera.')}
    reader = load_declared_reader(declared)
    namespace = reader.identity['namespace']
    assert reader.identity['source_sha256'] == digest
    assert namespace.startswith('tessera_reader_')
    assert reader.read_unit_artifact(b'x')[0] == namespace + '.unit_artifact'
    assert reader.verify_cached_unit(b'x', {}, {})[0] == namespace + '.cached_unit'
    assert sys.modules['tessera'] is tessera
    assert all(sys.modules.get(name) is module for name, module in primary.items())
    assert load_declared_reader(declared).identity == reader.identity


@pytest.mark.parametrize('value', [
    pytest.param('A' * 64, id='uppercase'),
    pytest.param(None, id='malformed'),
    pytest.param('a' * 63, id='short'),
    pytest.param('a' * 65, id='long'),
    pytest.param('g' * 64, id='nonhex'),
    pytest.param('a' * 64 + '\n', id='newline'),
])
def test_reader_refuses_invalid_sha256_before_source_access(grammar_package, monkeypatch, value):
    from prismaquant.tessera_reader import load_declared_reader
    consumer = sys.modules['prismaquant.tessera_reader']
    declared = dict(grammar_package, source_sha256=value)

    def refuse_access(*args, **kwargs):
        raise AssertionError('invalid grammar reached source access or import')

    class ImportGuard:
        import_module = staticmethod(refuse_access)

    monkeypatch.setattr(consumer, 'Path', refuse_access)
    monkeypatch.setattr(consumer, '_source_tree_profiles', refuse_access)
    monkeypatch.setattr(consumer, 'importlib', ImportGuard)
    with pytest.raises(ValueError, match='^reader requires an exact source SHA256$'):
        load_declared_reader(declared)


@pytest.mark.parametrize('text_kind', ['valid', 'invalid'])
def test_reader_keeps_sha256_string_conversion(grammar_package, monkeypatch, text_kind):
    from prismaquant.tessera_reader import load_declared_reader
    consumer = sys.modules['prismaquant.tessera_reader']
    declared = grammar_package
    digest = declared['source_sha256']
    conversions, traversals = [], []
    real_profiles = consumer._source_tree_profiles

    class Declaration:
        def __str__(self):
            conversions.append(text_kind)
            return digest if text_kind == 'valid' else digest.upper()

    def record_profiles(root):
        traversals.append(root)
        return real_profiles(root)

    declared['source_sha256'] = Declaration()
    monkeypatch.setattr(consumer, '_source_tree_profiles', record_profiles)
    message = ('^reader package checksum changed: source profiles require labelled, exact known SHA256 values$'
               if text_kind == 'valid' else '^reader requires an exact source SHA256$')
    with pytest.raises(ValueError, match=message):
        load_declared_reader(declared)
    assert conversions == [text_kind]
    assert traversals == ([Path(declared['path']).resolve()] if text_kind == 'valid' else [])


def test_reader_namespace_preserves_primary_and_binds_actual_sources(tmp_path):
    import tessera
    from prismaquant.tessera_reader import load_declared_reader
    primary = dict((name, module) for name, module in sys.modules.items()
                   if name == 'tessera' or name.startswith('tessera.'))
    declared = package(tmp_path)
    reader = load_declared_reader(declared)
    assert reader.identity['source_sha256'] == declared['source_sha256']
    assert reader.read_unit_artifact(b'x')[0].startswith('tessera_reader_')
    assert reader.verify_cached_unit(b'x', {}, {})[0].startswith('tessera_reader_')
    assert sys.modules['tessera'] is tessera
    assert all(sys.modules.get(name) is module for name, module in primary.items())
    reader.identity['source_sha256'] = '0' * 64
    assert load_declared_reader(declared).identity['source_sha256'] == declared['source_sha256']


def test_reader_refuses_absolute_producer_import(tmp_path):
    from prismaquant.tessera_reader import load_declared_reader
    with pytest.raises(ImportError, match='absolute.*tessera'):
        load_declared_reader(package(tmp_path, escaping=True))


def test_reader_refuses_changed_source(tmp_path):
    from prismaquant.tessera_reader import load_declared_reader
    declared = package(tmp_path)
    (Path(declared['path']) / '__init__.py').write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='checksum'):
        load_declared_reader(declared)
