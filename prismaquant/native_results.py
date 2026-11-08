"""Finite app artifact publication and one explicitly selected native producer.

Flux owns execution. This module has no scheduler, discovery, retries or cache.
Scientific admission remains with each consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.machinery
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import sysconfig

from . import flux_jobs
from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex, file_digest_sha256hex
from .file_identity import file_stat_signature, durable_package_path
from .schemas import Contract, strict_json_loads
from .staged_manifest import read_document, publish_staged_json, open_regular

_contract = Contract(ValueError, 'native result: ')
SELECTION_SCHEMA = 'prismaquant.flux_result_selection.v1'
PUBLICATION_SCHEMA = 'prismaquant.flux_result_publication.v1'
PRODUCER_SCHEMA = 'prismaquant.flux_result_producer.v1'
RUNTIME_SCHEMA = 'prismaquant.flux_result_runtime.v3'
ANNOUNCEMENT_SCHEMA = 'prismaquant.flux_result_announcement.v1'
ANNOUNCEMENT_PREFIX = b'FLUX_RESULT_PUBLICATION '
SELECTION_KEYS = frozenset({'schema', 'job_id', 'record_dir', 'launch_sha256', 'publication',
    'expected_launch', 'producer_id', 'inputs', 'runtime', 'max_publication_bytes',
    'max_evidence_bytes', 'max_artifact_bytes'})


class ResultUnavailable(RuntimeError):
    """Query, evidence, identity or cleanup failure; never a payload verdict."""


class ProducerFailed(RuntimeError):
    def __init__(self, execution):
        self.execution = execution
        super().__init__(f'selected native producer {execution.status}: {execution.job_id}')


@dataclass(frozen=True)
class SelectedResult:
    publication: dict
    artifact_payloads: dict[str, bytes]
    input_payloads: dict[str, bytes]
    launch: dict
    execution: flux_jobs.ExecutionResult


def _result_fields(value, keys, label):
    return _contract.exact_mapping(value, keys=frozenset(keys), where=label)


def _document(raw):
    return strict_json_loads(raw,
        duplicate=lambda key: ValueError(f'native result: duplicate key {key}'),
        constant=lambda key: ValueError(f'native result: invalid constant {key}'))


def _result_file_binding(value):
    _result_fields(value, {'path', 'sha256', 'bytes'}, 'file binding')
    from .staged_manifest import path, digest, integer
    path(value['path']); digest(value['sha256']); integer(value['bytes'], positive=True)
    return value


def _owned(binding, limit, *, identity=None):
    _result_file_binding(binding)
    if binding['bytes'] > limit:
        raise ValueError('native result: file exceeds selected byte budget')
    raw, observed = read_document(binding['path'], binding['bytes'])
    if (len(raw) != binding['bytes'] or bytes_sha256hex(raw) != binding['sha256'] or
            file_stat_signature(os.stat(binding['path'], follow_symlinks=False)) != observed or
            (identity is not None and list(observed) != identity)):
        raise ValueError('native result: file digest/length/identity changed')
    return raw


def bind_result_artifact(filename, *, max_bytes):
    """Measure bounded owned bytes using the shared strict file reader."""
    raw, identity = read_document(filename, max_bytes)
    if not raw or file_stat_signature(os.stat(filename, follow_symlinks=False)) != identity:
        raise ValueError('native result: empty or replaced artifact')
    return {'path': str(filename), 'sha256': bytes_sha256hex(raw), 'bytes': len(raw),
            'file_identity': list(identity)}


MAX_RUNTIME_DEPENDENCIES = 64
MAX_RUNTIME_ROOTS = 256
MAX_RUNTIME_ENTRIES = 65536
MAX_RUNTIME_IMPORTS = 16384
MAX_RUNTIME_FILE_BYTES = 4 << 30
MAX_RUNTIME_BYTES = 32 << 30
MAX_RUNTIME_DOCUMENT_BYTES = 12 << 20
MAX_RUNTIME_DEPTH = 64


def _runtime_path(filename):
    """Locations supplied by the interpreter may be aliases; owned trees may not."""
    return str(Path(filename).resolve(strict=True))


def _runtime_standard_roots():
    roots = {_runtime_path(sysconfig.get_path('stdlib')),
             _runtime_path(sysconfig.get_path('platstdlib'))}
    dynload = sysconfig.get_config_var('DESTSHARED')
    if dynload and Path(dynload).exists():
        roots.add(_runtime_path(dynload))
    # A child is already fully enumerated with its parent.
    return sorted(p for p in roots if not any(Path(q) in Path(p).parents for q in roots))


def _runtime_stat(filename):
    info = os.stat(filename, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('native runtime requires regular members')
    return file_stat_signature(info)


def _runtime_file(filename, expected):
    fd = open_regular(filename)
    with os.fdopen(fd, 'rb') as handle:
        before = file_stat_signature(os.fstat(handle.fileno()))
        if before != expected or before[2] > MAX_RUNTIME_FILE_BYTES:
            raise ValueError('native runtime file changed or exceeds byte bound')
        digest = file_digest_sha256hex(handle, max_bytes=before[2])
        if (handle.tell() != before[2] or file_stat_signature(os.fstat(handle.fileno())) != before
                or _runtime_stat(filename) != before):
            raise ValueError('native runtime file changed during streaming read')
    return {'bytes': before[2], 'sha256': digest}


def _runtime_tree(root, *, stdlib=False):
    """Census durable files and directories without following any tree symlink."""
    files, directories = {}, {}
    pending = [Path(root)]
    while pending:
        current = pending.pop()
        relative = current.relative_to(root)
        if len(relative.parts) > MAX_RUNTIME_DEPTH:
            raise ValueError('native runtime directory depth exceeds bound')
        info = current.lstat()
        if stat.S_ISREG(info.st_mode) and current == Path(root):
            files['.'] = file_stat_signature(info)
            break
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('native runtime tree has non-directory or symlink owner')
        # Opening through the shared component walker also rejects ancestor
        # symlinks; files themselves are opened again by the streaming owner.
        directories[str(relative)] = (info.st_dev, info.st_ino)
        if len(files) + len(directories) > MAX_RUNTIME_ENTRIES:
            raise ValueError('native runtime tree entry count exceeds bound')
        with os.scandir(current) as entries:
            for entry in entries:
                child = current / entry.name
                member = child.relative_to(root)
                if not durable_package_path(member):
                    continue
                if stdlib and member.parts[0] in {'site-packages', 'dist-packages'}:
                    continue
                item = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(item.st_mode):
                    pending.append(child)
                elif stat.S_ISREG(item.st_mode):
                    files[str(member)] = file_stat_signature(item)
                else:
                    raise ValueError('native runtime tree contains symlink/nonregular member')
                if len(files) + len(directories) + len(pending) > MAX_RUNTIME_ENTRIES:
                    raise ValueError('native runtime tree entry count exceeds bound')
        if file_stat_signature(current.lstat()) != file_stat_signature(info):
            raise ValueError('native runtime directory raced during census')
    return dict(sorted(files.items())), dict(sorted(directories.items()))


def _runtime_module_location(module):
    spec = getattr(module, '__spec__', None)
    origin = getattr(spec, 'origin', None)
    filename = getattr(module, '__file__', None)
    if filename is not None:
        filename = _runtime_path(filename)
        if origin not in (None, 'built-in', 'frozen') and _runtime_path(origin) != filename:
            raise ValueError('native runtime imported file/spec owner differs')
        origin = filename
    elif origin not in (None, 'built-in', 'frozen'):
        origin = _runtime_path(origin)
    locations = sorted(_runtime_path(p) for p in getattr(module, '__path__', ()))
    loader = getattr(spec, 'loader', None)
    supported = (importlib.machinery.SourceFileLoader, importlib.machinery.ExtensionFileLoader)
    if loader is not None and not (isinstance(loader, supported) or
            loader in (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter)):
        raise ValueError('native runtime unsupported imported loader')
    cached = getattr(module, '__cached__', None)
    cached = _runtime_path(cached) if cached and os.path.lexists(cached) else None
    return {'origin': origin, 'locations': locations,
            'loader': f'{type(loader).__module__}.{type(loader).__qualname__}', 'cached': cached}


def _runtime_imports():
    modules = sys.modules.copy()
    if len(modules) > MAX_RUNTIME_IMPORTS:
        raise ValueError('native runtime imported module count exceeds bound')
    result = {}
    for name, module in sorted(modules.items()):
        if module is None:
            continue
        row = _runtime_module_location(module)
        if row['origin'] is None:
            # Extension-defined child modules often have no spec/file. Bind
            # their nearest located parent as well as their own loaded name.
            parent = name
            while '.' in parent and row['origin'] is None:
                parent = parent.rpartition('.')[0]
                if modules.get(parent) is not None:
                    row['origin'] = _runtime_module_location(modules[parent])['origin']
            if row['origin'] is None:
                raise ValueError(f'native runtime unsupported imported owner: {name}')
        if row['origin'] not in {'built-in', 'frozen'}:
            if not durable_package_path(Path(row['origin'])):
                raise ValueError('native runtime sourceless bytecode owner unsupported')
        result[name] = row
    return result


def _runtime_libraries():
    """Bind Linux's actual file-backed implementation mappings, not ldd text."""
    with open('/proc/self/maps', 'rb') as handle:
        raw = handle.read(4 << 20)
        if handle.read(1):
            raise ValueError('native runtime mappings exceed bound')
    libraries = {}
    for line in raw.decode('utf-8', errors='strict').splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            if 'x' in fields[1] and fields[4] == '0':
                raise ValueError('native runtime anonymous executable mapping unsupported')
            continue
        _, perms, _, device, inode, filename = fields
        if filename.startswith('['):
            continue  # Kernel-provided vdso/vsyscall, no mutable filesystem owner.
        if 'x' not in perms and '.so' not in filename:
            continue
        if not filename.startswith('/') or filename.endswith(' (deleted)') or '\\' in filename:
            raise ValueError('native runtime unsupported mapped implementation owner')
        path = _runtime_path(filename)
        info = os.stat(path)
        major, minor = (int(v, 16) for v in device.split(':'))
        if (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) != (major, minor, int(inode)):
            raise ValueError(f'native runtime mapped library replaced: {path}')
        libraries[path] = file_stat_signature(info)
    if not libraries or len(libraries) > MAX_RUNTIME_ENTRIES:
        raise ValueError('native runtime requires finite mapped libraries')
    return dict(sorted(libraries.items()))


def _runtime_distribution(name):
    dist = importlib.metadata.distribution(name)
    base = Path(_runtime_path(dist.locate_file('')))
    entries = []
    for entry in dist.files or ():
        if len(entries) >= MAX_RUNTIME_ENTRIES:
            raise ValueError('native runtime distribution count exceeds bound')
        entries.append(Path(entry))
    roots, modules = set(), {}
    for entry in entries:
        if entry.is_absolute() or '..' in entry.parts or not entry.parts:
            # Installed console scripts live outside site-packages, but must
            # still be measured and stay inside this interpreter's bin root.
            located = Path(os.path.abspath(dist.locate_file(entry)))
            if (entry.is_absolute() or located.parent != Path(sys.prefix) / 'bin'
                    or located.suffix in {'.py', '.so', '.dll', '.dylib'}):
                raise ValueError('native runtime distribution member escapes installed owner')
            roots.add(str(located))
            continue
        located = base / entry
        if Path(os.path.abspath(dist.locate_file(entry))) != located:
            raise ValueError('native runtime distribution substituted installed root')
        _runtime_stat(located)
        root = base / entry.parts[0]
        roots.add(str(root))
        if len(entry.parts) >= 2 and entry.parts[0].isidentifier() and (root / '__init__.py').is_file():
            modules[entry.parts[0]] = str(root / '__init__.py')
        elif len(entry.parts) == 1:
            for suffix in ('.py', *importlib.machinery.EXTENSION_SUFFIXES):
                if entry.name.endswith(suffix) and entry.name[:-len(suffix)].isidentifier():
                    modules[entry.name[:-len(suffix)]] = str(root)
                    break
    metadata = [str(base / e) for e in entries if str(e).endswith('.dist-info/METADATA')]
    record = [str(base / e) for e in entries if str(e).endswith('.dist-info/RECORD')]
    if len(metadata) != 1 or len(record) != 1 or not modules:
        raise ValueError(f'native runtime requires installed metadata and importable owner: {name}')
    owners = {}
    for module, entry in sorted(modules.items()):
        spec = importlib.machinery.PathFinder.find_spec(module)
        if (spec is None or spec.origin != entry or
                not isinstance(spec.loader, (importlib.machinery.SourceFileLoader,
                                             importlib.machinery.ExtensionFileLoader))):
            raise ValueError(f'native runtime foreign/unsupported import resolution: {module}')
        locations = sorted(str(p) for p in spec.submodule_search_locations or ())
        expected = [str(Path(entry).parent)] if Path(entry).name == '__init__.py' else []
        if locations != expected:
            raise ValueError('native runtime import search owner differs')
        owners[module] = {'origin': entry, 'locations': locations}
    return {'version': dist.version, 'metadata': metadata[0], 'record': record[0],
            'roots': sorted(roots), 'imports': owners}


def _runtime_owners(packages, imported):
    for package in packages.values():
        roots = [Path(p) for p in package['roots']]
        def installed(filename):
            location = Path(filename)
            return any(location == root or root in location.parents for root in roots)
        for name, owner in package['imports'].items():
            for module, observed in imported.items():
                if module == name or module.startswith(name + '.'):
                    # Packages can publish aliases across their own roots,
                    # e.g. pytest's py.path -> _pytest._py.path. Every such
                    # root is fully measured; a foreign distribution is not.
                    if (not installed(observed['origin']) or
                            any(not installed(p) for p in observed['locations'])):
                        raise ValueError('native runtime already imported foreign dependency owner')
                    if module == name and (observed['origin'] != owner['origin'] or
                                           observed['locations'] != owner['locations']):
                        raise ValueError('native runtime imported entry owner differs')


def measure_runtime(*, dependencies: tuple[str, ...] = ()) -> dict:
    """Measure the finite declared closure and current imported/mapped owners.

    No observation is reused between calls. New imports and dlopen mappings
    change identity, even when their owners were not declared for a full tree.
    This Linux host-interpreter contract does not attest arbitrary future code.
    """
    if (len(dependencies) > MAX_RUNTIME_DEPENDENCIES or len(set(dependencies)) != len(dependencies)
            or any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', n)
                   for n in dependencies)):
        raise ValueError('native runtime requires finite unique dependency names')
    # Resolve the modules used by the measurement itself before taking its
    # loaded roster. sysconfig may lazily load its platform data module.
    standard_roots = _runtime_standard_roots()
    executable = _runtime_path(sys.executable)
    if _runtime_path('/proc/self/exe') != executable:
        raise ValueError('native runtime interpreter differs from actual executing owner')
    version = platform.python_version()
    packages = {name: _runtime_distribution(name) for name in sorted(dependencies)}
    imported, libraries = _runtime_imports(), _runtime_libraries()
    _runtime_owners(packages, imported)
    roots = {root for p in packages.values() for root in p['roots']} | set(standard_roots)
    if len(roots) > MAX_RUNTIME_ROOTS:
        raise ValueError('native runtime root count exceeds bound')
    trees, signatures, directories = {}, {}, set()
    for root in sorted(roots):
        trees[root] = _runtime_tree(root, stdlib=root in standard_roots)
        directories.update(str(Path(root) / member) for member in trees[root][1])
        for relative, signature in trees[root][0].items():
            signatures[str(Path(root) / relative)] = signature
        if len(directories) + len(signatures) > MAX_RUNTIME_ENTRIES:
            raise ValueError('native runtime aggregate directory/file count exceeds bound')
    for filename in {executable, *libraries, *(r['origin'] for r in imported.values()
                      if r['origin'] not in {'built-in', 'frozen'}),
                      *(r['cached'] for r in imported.values() if r['cached'] is not None)}:
        signatures[filename] = _runtime_stat(filename)
    if (len(signatures) > MAX_RUNTIME_ENTRIES or
            sum(s[2] for s in signatures.values()) > MAX_RUNTIME_BYTES):
        raise ValueError('native runtime aggregate file count/byte budget exceeded')
    files = {p: _runtime_file(p, sig) for p, sig in sorted(signatures.items())}
    for root, before in trees.items():
        if _runtime_tree(root, stdlib=root in standard_roots) != before:
            raise ValueError('native runtime file/directory roster changed during measurement')
    if any(_runtime_stat(p) != sig for p, sig in signatures.items()):
        raise ValueError('native runtime member changed after streaming read')
    if (imported != _runtime_imports() or libraries != _runtime_libraries() or
            packages != {name: _runtime_distribution(name) for name in sorted(dependencies)}):
        raise ValueError('native runtime imported/mapped/distribution owner raced')
    for package in packages.values():
        package['roots'] = {root: list(trees[root][0]) for root in package['roots']}
    value = {'schema': RUNTIME_SCHEMA, 'python': {'path': executable, **files[executable],
        'version': version, 'implementation': sys.implementation.name,
        'cache_tag': sys.implementation.cache_tag, 'flags': list(sys.flags)}, 'dependencies': packages,
        'stdlib': {root: list(trees[root][0]) for root in standard_roots}, 'imports': imported,
        'libraries': list(libraries), 'files': files}
    return validate_runtime(value)


def validate_runtime(value):
    """Validate v3 evidence structurally; equality requires independent measurement."""
    _result_fields(value, {'schema', 'python', 'dependencies', 'stdlib', 'imports',
                           'libraries', 'files'}, 'runtime')
    _contract.require(value['schema'] == RUNTIME_SCHEMA, 'runtime schema differs')
    from .staged_manifest import path, digest, integer
    files = _contract.mapping(value['files'], where='runtime files')
    _contract.require(0 < len(files) <= MAX_RUNTIME_ENTRIES, 'runtime file count exceeds bound')
    total = 0
    for filename, row in files.items():
        path(filename)
        _result_fields(row, {'bytes', 'sha256'}, 'runtime file')
        integer(row['bytes']); digest(row['sha256'])
        _contract.require(row['bytes'] <= MAX_RUNTIME_FILE_BYTES, 'runtime file byte bound')
        total += row['bytes']
    _contract.require(total <= MAX_RUNTIME_BYTES, 'runtime aggregate byte bound')
    python = _result_fields(value['python'], {'path', 'sha256', 'bytes', 'version',
        'implementation', 'cache_tag', 'flags'}, 'interpreter')
    path(python['path'])
    _contract.require(files.get(python['path']) == {k: python[k] for k in ('sha256', 'bytes')}
                      and python['bytes'] > 0, 'interpreter content binding differs')
    for key in ('version', 'implementation', 'cache_tag'):
        _contract.string(python[key], where=f'interpreter {key}')
    _contract.require(isinstance(python['flags'], list) and 0 < len(python['flags']) <= 64
        and all(type(flag) in (int, bool) and -1 <= flag < (1 << 31) for flag in python['flags']),
        'invalid finite interpreter flags')
    used = {python['path']}
    roots = {}

    def trees(roster):
        _contract.mapping(roster, where='runtime roots')
        for root, members in roster.items():
            path(root)
            _contract.require(isinstance(members, list) and len(members) <= MAX_RUNTIME_ENTRIES
                              and members == sorted(set(members)), 'runtime sorted finite member roster required')
            for member in members:
                _contract.string(member, where='runtime relative member')
                relative = Path(member)
                _contract.require(not relative.is_absolute() and '..' not in relative.parts
                    and str(relative) == member and len(relative.parts) <= MAX_RUNTIME_DEPTH
                    and durable_package_path(relative), 'runtime member escapes root')
                filename = str(Path(root) / relative)
                _contract.require(filename in files, 'runtime tree member missing content')
                used.add(filename)
            _contract.require(root not in roots or roots[root] == members, 'conflicting runtime roots')
            roots[root] = members
    trees(value['stdlib'])
    _contract.require(0 < len(value['stdlib']) <= MAX_RUNTIME_ROOTS, 'finite measured stdlib required')
    deps = _contract.mapping(value['dependencies'], where='dependencies')
    _contract.require(len(deps) <= MAX_RUNTIME_DEPENDENCIES, 'dependency roster exceeds bound')
    for name, package in deps.items():
        _contract.string(name, where='dependency name')
        _result_fields(package, {'version', 'metadata', 'record', 'roots', 'imports'}, 'dependency')
        _contract.string(package['version'], where='dependency version')
        trees(package['roots'])
        for role in ('metadata', 'record'):
            _contract.require(package[role] in used, 'dependency metadata missing measured owner')
        owners = _contract.mapping(package['imports'], where='dependency import owners')
        _contract.require(0 < len(owners) <= MAX_RUNTIME_ROOTS, 'finite dependency import owners required')
        for name, owner in owners.items():
            _contract.string(name, where='dependency import name')
            _result_fields(owner, {'origin', 'locations'}, 'dependency import owner')
            _contract.require(owner['origin'] in used, 'dependency import missing content')
            _runtime_locations(owner['locations'])
    _contract.require(len(roots) <= MAX_RUNTIME_ROOTS, 'runtime root count exceeds bound')
    imported = _contract.mapping(value['imports'], where='imported owners')
    _contract.require(0 < len(imported) <= MAX_RUNTIME_IMPORTS, 'finite imported owner roster required')
    for name, row in imported.items():
        _contract.string(name, where='imported name')
        _result_fields(row, {'origin', 'locations', 'loader', 'cached'}, 'imported owner')
        _contract.string(row['loader'], where='imported loader')
        _runtime_locations(row['locations'])
        if row['origin'] not in {'built-in', 'frozen'}:
            _contract.require(row['origin'] in files, 'imported owner missing content')
            used.add(row['origin'])
        if row['cached'] is not None:
            _contract.require(row['cached'] in files, 'loaded bytecode cache missing content')
            used.add(row['cached'])
    _runtime_locations(value['libraries'])
    _contract.require(bool(value['libraries']), 'measured support libraries required')
    for filename in value['libraries']:
        _contract.require(filename in files, 'mapped library missing content')
        used.add(filename)
    _contract.require(used == set(files), 'runtime file roster has undeclared content')
    _runtime_owners(deps, imported)
    _contract.require(len(DIRECT_ASCII_STRICT.encoded(value)) <= MAX_RUNTIME_DOCUMENT_BYTES,
                      'runtime evidence document exceeds bound')
    return value


def _runtime_locations(value):
    from .staged_manifest import path
    _contract.require(isinstance(value, list) and len(value) <= MAX_RUNTIME_ENTRIES
                      and value == sorted(set(value)), 'sorted finite runtime locations required')
    for location in value:
        path(location)


def execution_context() -> dict:
    """Authenticate current source, native launch and actual CPU allocation."""
    out = Path(os.environ['OUT'])
    if not out.is_absolute() or out.resolve(strict=True) != out:
        raise ValueError('native result: current record must be its canonical absolute path')
    raw, _ = read_document(out / 'launch.json', 16 << 20)
    launch = _document(raw)
    source = flux_jobs._source_identity(Path.cwd().resolve())
    if (Path(__file__).resolve() != Path(source['checkout']) / 'prismaquant/native_results.py'
            or source != launch['source'] or str(out) != launch['record_dir']):
        raise ValueError('native result: executing source/record differs from launch')
    try:
        saved, authenticated = flux_jobs.verify_current_job(job_id=os.environ['FLUX_JOB_ID'],
                                                           record_dir=out, source=source)
    except flux_jobs.SubmissionUnknown as exc:
        raise ResultUnavailable('current native submission acknowledgement unavailable; context refused') from exc
    if saved.launch_sha256 != bytes_sha256hex(raw) or authenticated != launch:
        raise ValueError('native result: current launch changed during authentication')
    return {'schema': PRODUCER_SCHEMA, 'job_id': flux_jobs._job_id(os.environ['FLUX_JOB_ID']),
            'record_dir': str(out), 'launch_sha256': bytes_sha256hex(raw), 'source': source,
            'argv': launch['argv'], 'resources': launch['resources'], 'afterok': launch['afterok'],
            'staged_input': launch.get('staged_input')}


def _roster(value, *, limit, artifacts=False):
    _contract.require(isinstance(value, dict) and 0 < len(value) <= 64, 'finite nonempty role roster required')
    paths, total = set(), 0
    for role, row in value.items():
        _contract.string(role, where='role')
        if artifacts:
            _result_fields(row, {'path', 'sha256', 'bytes', 'file_identity'}, 'artifact')
            identity = row['file_identity']
            _contract.require(isinstance(identity, list) and len(identity) == 5 and
                all(type(item) is int and item >= 0 for item in identity), 'invalid artifact identity')
        else:
            _result_fields(row, {'path', 'sha256', 'bytes'}, 'input')
        _result_file_binding({key: row[key] for key in ('path', 'sha256', 'bytes')})
        _contract.require(row['path'] not in paths, 'role roster repeats a physical path')
        paths.add(row['path']); total += row['bytes']
    _contract.require(total <= limit, 'aggregate owned byte budget exceeded')


def publish_result(path: Path, *, producer_id: str, inputs: dict,
                   artifacts: dict[str, Path], dependencies: tuple[str, ...] = (),
                   max_artifact_bytes: int) -> dict:
    _contract.string(producer_id, where='producer ID')
    _ceiling(max_artifact_bytes, 1 << 30)
    _roster(inputs, limit=max_artifact_bytes)
    for binding in inputs.values():
        _owned(binding, max_artifact_bytes)
    if not isinstance(artifacts, dict) or not 0 < len(artifacts) <= 64:
        raise ValueError('native result: finite artifact roster required')
    measured = {role: bind_result_artifact(filename, max_bytes=max_artifact_bytes)
                for role, filename in artifacts.items()}
    _roster(measured, limit=max_artifact_bytes, artifacts=True)
    if sum(row['bytes'] for row in [*inputs.values(), *measured.values()]) > max_artifact_bytes:
        raise ValueError('native result: aggregate input/artifact budget exceeded')
    publication = {'schema': PUBLICATION_SCHEMA, 'producer_id': producer_id,
        'producer': execution_context(), 'runtime': measure_runtime(dependencies=dependencies),
        'inputs': inputs, 'artifacts': measured}
    raw = DIRECT_ASCII_STRICT.encoded(publication) + b'\n'
    if len(raw) > 16 << 20:
        raise ValueError('native result: publication exceeds bound')
    digest = publish_staged_json(path, publication)
    binding = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
    print((ANNOUNCEMENT_PREFIX + DIRECT_ASCII_STRICT.encoded(
        {'schema': ANNOUNCEMENT_SCHEMA, **binding})).decode('ascii'), flush=True)
    return binding


def _ceiling(value, maximum):
    _contract.require(type(value) is int and 0 < value <= maximum, 'invalid finite byte ceiling')


def read_selected_result(selection: dict) -> SelectedResult:
    """Read one publication only after authenticating actual terminal execution."""
    try:
        _result_fields(selection, SELECTION_KEYS, 'selection')
        _contract.require(selection['schema'] == SELECTION_SCHEMA, 'selection schema differs')
        for key, maximum in (('max_publication_bytes', 16 << 20), ('max_evidence_bytes', 64 << 20),
                             ('max_artifact_bytes', 1 << 30)):
            _ceiling(selection[key], maximum)
        from .staged_manifest import path, digest
        path(selection['record_dir']); digest(selection['launch_sha256'])
        _result_file_binding(selection['publication'])
        expected = _result_fields(selection['expected_launch'],
            {'argv', 'source', 'resources', 'afterok', 'environment', 'staged_input'}, 'expected launch')
        saved = flux_jobs.load_submission(Path(selection['record_dir']))
        if (str(saved.record_dir) != selection['record_dir'] or
                flux_jobs.job_id_number(saved.job_id) != flux_jobs.job_id_number(selection['job_id']) or
                saved.launch_sha256 != selection['launch_sha256']):
            raise ValueError('native result: selected producer coordinates differ')
        execution = flux_jobs.collect_job(saved)
        # A conclusive native failure keeps its classification even if cleanup failed.
        if execution.status in ('failed', 'canceled', 'dependency-failed'):
            raise ProducerFailed(execution)
        if execution.status != 'succeeded' or execution.query_error:
            raise ResultUnavailable(f'native execution unavailable: {execution.status}; {execution.query_error}')
        launch = flux_jobs.verify_native_launch(saved,
            max_evidence_bytes=selection['max_evidence_bytes'],
            **{**expected, 'resources': flux_jobs.FluxResources(**expected['resources'])})
        evidence = 0
        for name in ('eventlog.json', 'status.json', 'R.json', 'stdout', 'stderr'):
            stream, _ = read_document(saved.record_dir / name, selection['max_evidence_bytes'] - evidence)
            evidence += len(stream)
            if name == 'stdout':
                stdout = stream
        raw = _owned(selection['publication'], selection['max_publication_bytes'])
        publication = _result_fields(_document(raw),
            {'schema', 'producer_id', 'producer', 'runtime', 'inputs', 'artifacts'}, 'publication')
        announcement = {'schema': ANNOUNCEMENT_SCHEMA, **selection['publication']}
        lines = [line[len(ANNOUNCEMENT_PREFIX):] for line in stdout.splitlines()
                 if line.startswith(ANNOUNCEMENT_PREFIX)]
        if lines != [DIRECT_ASCII_STRICT.encoded(announcement)]:
            raise ValueError('native result: exact retained publisher announcement missing or ambiguous')
        producer = {'schema': PRODUCER_SCHEMA, 'job_id': saved.job_id,
            'record_dir': str(saved.record_dir), 'launch_sha256': saved.launch_sha256,
            **{key: launch[key] for key in ('source', 'argv', 'resources', 'afterok')},
            'staged_input': launch.get('staged_input')}
        observed_producer = dict(publication['producer'])
        observed_producer['job_id'] = flux_jobs.job_id_number(observed_producer['job_id'])
        producer['job_id'] = flux_jobs.job_id_number(producer['job_id'])
        if (publication['schema'] != PUBLICATION_SCHEMA or publication['producer_id'] != selection['producer_id']
                or observed_producer != producer or publication['inputs'] != selection['inputs']
                or validate_runtime(publication['runtime']) != validate_runtime(selection['runtime'])):
            raise ValueError('native result: producer/source/input/runtime differs from selection')
        _roster(publication['inputs'], limit=selection['max_artifact_bytes'])
        _roster(publication['artifacts'], limit=selection['max_artifact_bytes'], artifacts=True)
        if sum(row['bytes'] for roster in ('inputs', 'artifacts')
               for row in publication[roster].values()) > selection['max_artifact_bytes']:
            raise ValueError('native result: aggregate input/artifact budget exceeded')
        input_payloads = {role: _owned(binding, selection['max_artifact_bytes'])
                          for role, binding in publication['inputs'].items()}
        artifact_payloads = {role: _owned({k: row[k] for k in ('path', 'sha256', 'bytes')},
                            selection['max_artifact_bytes'], identity=row['file_identity'])
                             for role, row in publication['artifacts'].items()}
        return SelectedResult(publication, artifact_payloads, input_payloads, launch, execution)
    except (ProducerFailed, ResultUnavailable):
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        raise ResultUnavailable(str(exc)) from exc
