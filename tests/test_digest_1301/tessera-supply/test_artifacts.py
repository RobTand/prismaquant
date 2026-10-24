"""Behavioral comparisons against small, independently frozen original bodies."""
import ast
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/tessera-supply/artifacts'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def publish():
    new = load(ROOT / 'tools/publish_artifact.py', 'artifact_new')
    old = load(FIXTURES / 'publish_artifact_original.py', 'artifact_old')
    for module in (old, new):
        module.SNAPSHOT_BLOCK_BYTES = 8
        module.SNAPSHOT_INLINE_BYTES = 16
    return old, new


def outcome(fn):
    try:
        return ('ok', fn())
    except (Exception, SystemExit) as exc:
        return (type(exc).__name__, str(exc))


def test_descriptor_scan_replay_and_refusals(publish, tmp_path):
    old, new = publish
    for data in (b'', b'abc', b'0123456789abcdefG'):
        path = tmp_path / 'payload'
        path.write_bytes(data)
        with path.open('rb') as stream:
            a = old._scan_file(stream.fileno(), len(data))
            b = new._scan_file(stream.fileno(), len(data))
            assert dataclasses.asdict(a) == dataclasses.asdict(b)
            assert b.sha256 == hashlib.sha256(data).hexdigest()
            entry = SimpleNamespace(fd=stream.fileno(), content=None, size=len(data),
                                    relative_path='payload', sha256=b.sha256,
                                    block_sha256=b.block_sha256)
            snapshot = SimpleNamespace(entries=[entry])
            assert outcome(lambda: old._replay_frozen_digests(snapshot)) == outcome(lambda: new._replay_frozen_digests(snapshot))
            if data:
                assert old._VerifiedBlockReader(entry).read(100) == new._VerifiedBlockReader(entry).read(100) == data
                path.write_bytes(b'X' + data[1:])
                assert outcome(lambda: old._VerifiedBlockReader(entry).read(100)) == outcome(lambda: new._VerifiedBlockReader(entry).read(100))
                assert outcome(lambda: old._replay_frozen_digests(snapshot)) == outcome(lambda: new._replay_frozen_digests(snapshot))
            path.write_bytes(b'')
            assert outcome(lambda: old._scan_file(stream.fileno(), 4)) == outcome(lambda: new._scan_file(stream.fileno(), 4))


def assigned_expression(module, target, **values):
    tree = ast.parse(Path(module.__file__).read_text())
    expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))
    return eval(compile(ast.Expression(expr), module.__file__, 'eval'), vars(module), values)


def test_manifest_canonical_bytes_and_quote(publish, tmp_path):
    old, new = publish
    # Evaluate the actual production assignments, not a second test-only route.
    for value in ([{'z': 'é', 'a': [1, None]}], [{'a': 1, 'z': 'é'}], [], [float('nan')], [object()]):
        assert outcome(lambda: old.manifest_digest(value)) == outcome(lambda: assigned_expression(new, 'manifest_sha', manifest_rows=value))
        if all(isinstance(x, dict) for x in value):
            assert new.DIRECT_ASCII_STRICT.encoded(value) == json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    for data in (b'', b'{"x":1}\n', bytes(range(256))):
        assert old.canonical_digest(data) == assigned_expression(new, 'expected_shipcard_sha', canonical_bytes=data)
    path = tmp_path / 'é "quoted"'
    assert old.quote_path(path) == new.DIRECT_ASCII_SPACED_LAX.text(str(path.resolve()))


def test_provision_tree_identity_and_manifest(tmp_path):
    new = load(ROOT / 'tools/provision_tessera_pin.py', 'provision_new')
    old = load(FIXTURES / 'provision_tessera_pin_original.py', 'provision_old')
    for key in ('SKIP_DIRS', 'SKIP_SUFFIXES', 'SKIP_NAMES', 'MANIFEST', 'PACKAGE_IN_TREE', 'CONTRACT_IN_TREE', 'package_digest'):
        setattr(old, key, getattr(new, key))
    pkg = tmp_path / new.PACKAGE_IN_TREE
    pkg.mkdir(parents=True)
    contract = tmp_path / new.CONTRACT_IN_TREE
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_bytes(b'{"z":1,"a":"unicode"}\n')
    (pkg / 'a.py').write_bytes(b'first')
    (pkg / 'b.py').write_bytes(b'second')
    for skip in (None, lambda rel: rel.name == 'a.py'):
        assert old.tree_digest(pkg, skip) == new.tree_digest(pkg, skip)
    assert old.expected_identity(tmp_path) == new.expected_identity(tmp_path)
    before = new.tree_digest(pkg)
    (pkg / 'a.py').rename(pkg / 'c.py')
    assert new.tree_digest(pkg) == old.tree_digest(pkg) != before
    old._write_manifest(pkg, 'commit')
    expected = (pkg / new.MANIFEST).read_bytes()
    new._write_manifest(pkg, 'commit')
    assert (pkg / new.MANIFEST).read_bytes() == expected
    assert outcome(lambda: old.expected_identity(tmp_path / 'absent')) == outcome(lambda: new.expected_identity(tmp_path / 'absent'))


def test_provision_host_checkout_bootstrap_isolated(tmp_path):
    # The supported host context is a checkout, not a tool copied alone:
    # accept_tessera_pin_provisioner.py anchors TOOL at ROOT/tools, and the
    # dated provision_cpu.py invokes tools/provision_tessera_pin.py there.
    checkout = tmp_path / 'checkout'
    staged = (
        'tools/provision_tessera_pin.py',
        'prismaquant/digests.py',
        'prismaquant/tessera_runtime_contract.py',
    )
    for relative in staged:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assert not (checkout / 'prismaquant/__init__.py').exists()
    outside = tmp_path / 'outside'
    outside.mkdir()
    tool = checkout / staged[0]
    probe = '''
import json, runpy, sys
assert sys.flags.isolated and sys.flags.no_site
host = runpy.run_path(sys.argv[1])
pin = host['reviewed_pin']()
digest = host['bytes_sha256hex'](b'bootstrap')
assert host['bytes_sha256hex'].__globals__['__file__'] == sys.argv[2]
assert not any(name.split('.')[0] in {'prismaquant', 'tessera', 'torch', 'site'}
               for name in sys.modules)
print(json.dumps({'pin': pin, 'digest': digest}))
'''
    completed = subprocess.run(
        [sys.executable, '-I', '-S', '-c', probe, str(tool),
         str(checkout / staged[1])],
        cwd=outside, capture_output=True, text=True, check=True,
    )
    source = ast.parse((checkout / staged[2]).read_text())
    literals = {node.targets[0].id: ast.literal_eval(node.value)
                for node in source.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in (
                    'TESSERA_DEV_PIN_COMMIT', 'TESSERA_DEV_PIN_CONTRACT_SHA256')}
    assert json.loads(completed.stdout) == {
        'pin': [literals['TESSERA_DEV_PIN_COMMIT'],
                literals['TESSERA_DEV_PIN_CONTRACT_SHA256']],
        'digest': hashlib.sha256(b'bootstrap').hexdigest(),
    }
    # Also exercise the real script entry point, without provisioning a target.
    cli = subprocess.run(
        [sys.executable, '-I', '-S', str(tool), '--help'],
        cwd=outside, capture_output=True, text=True, check=True,
    )
    assert '--pin-source' in cli.stdout


def test_installed_probe_isolated_and_unchanged(monkeypatch):
    new = load(ROOT / 'tools/provision_tessera_pin.py', 'probe_new')
    old = load(FIXTURES / 'provision_tessera_pin_original.py', 'probe_old')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='{"reason":"missing"}', stderr='', returncode=0)
    monkeypatch.setattr(new.subprocess, 'run', run)
    assert old.installed_identity('target-python', ['tessera.extra']) == new.installed_identity('target-python', ['tessera.extra'])
    assert calls[0] == calls[1]
    assert calls[0][0][:3] == ['target-python', '-I', '-c']
    assert 'prismaquant' not in calls[0][0][3]


def test_archive_hash_before_load_and_inspect_order(tmp_path, monkeypatch):
    new = load(ROOT / 'tools/tessera_campaign_container.py', 'container_new')
    old = load(FIXTURES / 'tessera_campaign_container_original.py', 'container_old')
    archive = tmp_path / 'image.tar'
    archive.write_bytes(b'image archive\x00\xff')
    for digest in ('wrong', hashlib.sha256(archive.read_bytes()).hexdigest()):
        runs = []
        for module in (old, new):
            calls = []
            def run(argv, **kwargs):
                calls.append((argv, kwargs))
                if len(calls) == 1:
                    return SimpleNamespace(returncode=1, stderr='absent', stdout='')
                return SimpleNamespace(returncode=0, stderr='', stdout='[]')
            monkeypatch.setattr(module.subprocess, 'run', run)
            result = outcome(lambda: module.inspect_or_load({'image': 'declared', 'archive': {'path': str(archive), 'sha256': digest}}))
            runs.append((result, list(calls)))
        assert runs[0] == runs[1]
        assert len(runs[0][1]) == (1 if digest == 'wrong' else 3)
        if digest != 'wrong':
            assert runs[0][1][1][0] == ['docker', 'load', '--input', str(archive)]


def test_retained_algorithm_source_witnesses(publish):
    old, new = publish
    import inspect
    # The incremental replay is deliberately not replaced by a whole-file owner.
    assert ast.dump(ast.parse(inspect.getsource(old._replay_frozen_digests)), include_attributes=False) == ast.dump(ast.parse(inspect.getsource(new._replay_frozen_digests)), include_attributes=False)
    # Only the nested plain-bytes calls change; outer framing/descriptor loops
    # must still be the original algorithm, not merely match these samples.
    scan_source = inspect.getsource(new._scan_file).replace('bytes_sha256hex(data)', 'hashlib.sha256(data).hexdigest()')
    assert ast.dump(ast.parse(scan_source), include_attributes=False) == ast.dump(ast.parse(inspect.getsource(old._scan_file)), include_attributes=False)
    provision = load(ROOT / 'tools/provision_tessera_pin.py', 'witness_provision')
    original = load(FIXTURES / 'provision_tessera_pin_original.py', 'witness_original')
    tree_source = inspect.getsource(provision.tree_digest).replace('bytes_sha256hex(path.read_bytes())', 'hashlib.sha256(path.read_bytes()).hexdigest()')
    assert ast.dump(ast.parse(tree_source), include_attributes=False) == ast.dump(ast.parse(inspect.getsource(original.tree_digest)), include_attributes=False)
    assert ast.dump(ast.parse(inspect.getsource(provision.installed_identity)), include_attributes=False) == ast.dump(ast.parse(inspect.getsource(original.installed_identity)), include_attributes=False)
