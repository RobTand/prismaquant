"""Independent research-base consumer captures; no owner-generated expectations."""
import ast
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/stagea-custody'
NAMES = ('stage_a_chain_resume', 'stage_a_chain_split', 'stage_a_retirement',
         'joint_cost_stage_a', 'joint_prewarm_phases')


def original(name):
    path = FIXTURES / f'chain_original_{name}.py'
    spec = importlib.util.spec_from_file_location('prismaquant._chain_original_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if name == 'joint_prewarm_phases':
        # Isolate the selected digest/manifest consumer from the separately
        # owned replay parser; this fixture deliberately has no replay block.
        module._REPLAY_MODULE = SimpleNamespace(sealed_from_annotations=lambda annotations: None)
    return module


class Odd:
    def __repr__(self):
        return '<chain odd é>'


def capture_split(module, tmp):
    resume = importlib.import_module('prismaquant.stage_a_chain_resume')
    checkpoint = importlib.import_module('prismaquant.joint_adjoint_checkpoints')
    receipt = {'z': '层é', 'a': [float('nan'), 2]}
    writes = []
    output = io.StringIO()
    with patch.object(checkpoint, 'adjoint_space', return_value=tmp), \
         patch.object(resume, 'read_chain_state', return_value={'run_identity': {'n_probes': 1}, 'n_batches': 2}), \
         patch.object(module, 'join_split_checkpoint', return_value={}, create=True), \
         patch.object(module, 'join_receipt', return_value=receipt, create=True), \
         patch.object(module, 'atomic_write_bytes', side_effect=lambda p, b: writes.append(b.decode())), \
         redirect_stdout(output):
        status = module.main(['--output-root', str(tmp), '--boundary', '1', '--receipt', str(tmp / 'receipt')])
    return {'status': status, 'stdout': output.getvalue(), 'receipt': writes}


def capture_resume(module, tmp):
    payloads = []
    plan = SimpleNamespace(leftovers=[], partials=[], index=2, session='é', boundary=4,
                           implementation_sha256='a' * 64, compatibility={'z': 1, 'a': True})
    with patch.object(module, 'publish_new_bytes', side_effect=lambda p, b: payloads.append(b.decode()) or True):
        record = module.apply_chain_resume(tmp, plan, producer={'z': '层'})
    return {'record': record, 'payload': payloads[0]}


def capture_retirement(module):
    payloads = []
    queue = SimpleNamespace(root='queue')
    instance = {'owner_action_key': 'owner', 'template_id': 'template'}
    row = {'batch_id': 'batch'}
    plan = module.Retirement(space=Path('chain-space'), record={'record_sha256': 'a' * 64, 'z': 'é'},
                             entry_paths=[], batches=[(queue, instance, {}, row)])
    po = SimpleNamespace(reclaim_origin=lambda *a, **k: {'ok': False, 'refusal': '层 refused'})
    with patch.object(module, 'publish_new_bytes', side_effect=lambda p, b: payloads.append(b.decode()) or True), \
         patch.object(module, '_sdk', return_value={'produced_output': po}, create=True), \
         patch.object(module, 'durable_charge', return_value=17, create=True):
        try:
            module.apply_retirement(plan)
        except module.RetirementRefused as exc:
            refusal = str(exc)
        else:
            raise AssertionError('must refuse failed reclaim')
    return {'payload': payloads[0], 'refusal': refusal}


def capture_reports(module, tmp):
    reports = []
    for report in ({'z': Odd(), 'a': (1, 2), 7: 'integer key'}, {'z': Odd(), 'a': (1, 2)},
                   {'x': float('nan')}, None):
        storage = SimpleNamespace(produced_output_report=lambda: report)
        reports.append(module._produced_output_block(storage))
    class Broken:
        def produced_output_report(self):
            raise RuntimeError('report failed')
    reports.append(module._produced_output_block(Broken()))
    reports.append(module._produced_output_block(object()))
    target = tmp / 'failures.jsonl'
    target.unlink(missing_ok=True)
    storage = SimpleNamespace(produced_output_report=lambda: {'z': Odd(), 'a': (1, 2)})
    with patch.object(module.time, 'time', return_value=123):
        for _ in range(2):
            error = KeyboardInterrupt('capture stopped')
            try:
                with module._failed_produced_output_record(tmp, storage, path=target):
                    raise error
            except KeyboardInterrupt as caught:
                assert caught is error
    return {'reports': reports, 'failure_lines': target.read_text()}


def capture_prewarm(module, tmp):
    doc = {'annotations': {'entry_point': 'prismaquant.tessera_joint_aura:prepare',
           'plan_sha256': 'plan', 'phase_start_units': {'层': 'head'}, 'phases': [{'name': 'head'}]}}
    raw = json.dumps(doc, ensure_ascii=False).encode()
    results = []
    for name, blob in [('manifest.json', raw), ('manifest.json.gz', gzip.compress(raw, mtime=0))]:
        path = tmp / name
        path.write_bytes(blob)
        result = module.load_prepare_read_set(str(path), hashlib.sha256(blob).hexdigest(), 'plan')
        results.append([result[0], result[1], list(result[2])])
        try:
            module.load_prepare_read_set(str(path), '0' * 64, 'plan')
        except RuntimeError as exc:
            results.append(str(exc))
        else:
            raise AssertionError('must reject changed pin')
    return results


CAPTURES = {'split': ('stage_a_chain_split', capture_split),
            'resume': ('stage_a_chain_resume', capture_resume),
            'reports': ('joint_cost_stage_a', capture_reports),
            'prewarm': ('joint_prewarm_phases', capture_prewarm)}


@pytest.mark.parametrize('case', ['split', 'resume', 'reports', 'prewarm'])
def test_chain_consumer_outcomes(case, tmp_path):
    expected = json.loads((FIXTURES / 'chain_outcomes.json').read_text())[case]
    name, capture = CAPTURES[case]
    assert capture(original(name), tmp_path) == expected
    current = importlib.import_module('prismaquant.' + name)
    if case == 'resume':
        # Observe the actual consumer route as well as frozen original bytes.
        with patch.object(current, 'indent2_json_file_bytes',
                          wraps=current.indent2_json_file_bytes) as encode:
            assert capture(current, tmp_path) == expected
        encode.assert_called_once_with(expected['record'])
    else:
        assert capture(current, tmp_path) == expected


def test_chain_retirement_outcomes():
    expected = json.loads((FIXTURES / 'chain_outcomes.json').read_text())['retirement']
    assert capture_retirement(original('stage_a_retirement')) == expected
    assert capture_retirement(importlib.import_module('prismaquant.stage_a_retirement')) == expected


def test_chain_failure_record_containment(tmp_path, capsys):
    for module in (original('joint_cost_stage_a'), importlib.import_module('prismaquant.joint_cost_stage_a')):
        blocked = tmp_path / 'blocked'
        blocked.write_text('not a directory')
        error = RuntimeError('original failure')
        with pytest.raises(RuntimeError) as caught:
            with module._failed_produced_output_record(tmp_path, SimpleNamespace(
                    produced_output_report=lambda: {'a': 1}), path=blocked / 'record'):
                raise error
        assert caught.value is error
        assert 'record could not be written' in capsys.readouterr().out


def test_chain_strict_publication_and_conflicts(tmp_path):
    for module in (original('stage_a_chain_resume'), importlib.import_module('prismaquant.stage_a_chain_resume')):
        plan = SimpleNamespace(leftovers=[], partials=[], index=2, session='é', boundary=4,
                               implementation_sha256='a' * 64, compatibility={})
        with patch.object(module, 'publish_new_bytes', return_value=False):
            with pytest.raises(module.ChainResumeRefused, match='another relaunch'):
                module.apply_chain_resume(tmp_path, plan)
        with patch.object(module, 'publish_new_bytes') as publish:
            with pytest.raises(ValueError):
                module.apply_chain_resume(tmp_path, plan, producer={'bad': float('nan')})
            publish.assert_not_called()
    for module in (original('stage_a_retirement'), importlib.import_module('prismaquant.stage_a_retirement')):
        plan = module.Retirement(space=tmp_path, record={'bad': float('nan')}, entry_paths=[])
        with patch.object(module, 'publish_new_bytes') as publish:
            with pytest.raises(ValueError):
                module.apply_retirement(plan)
            publish.assert_not_called()
        plan.record = {'record_sha256': 'a' * 64}
        (tmp_path / module.RETIREMENT_NAME).write_text('different')
        with patch.object(module, 'publish_new_bytes', return_value=False):
            with pytest.raises(module.RetirementRefused, match='another retirement'):
                module.apply_retirement(plan)


def test_chain_retention_witnesses():
    """Exact retained executable nodes, not function-level search matches."""
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    ledger = json.loads((FIXTURES / 'ledger_chains.json').read_text())
    frozen = {row['id']: row for row in registry['calls']}
    for row in ledger['retained_calls']:
        old = frozen[row['original_id']]
        tree = ast.parse((ROOT / old['file']).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == old['scope'])
        found = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() == old['syntax_sha256']]
        assert len(found) == 1, row['original_id']
        node = found[0]
        assert row['current_node_id'] == f"{old['callee']}:{old['file']}::{old['scope']}@{node.lineno}:{node.col_offset}"


def test_chain_provenance_and_completed_nodes():
    ledger = json.loads((FIXTURES / 'ledger_chains.json').read_text())
    registry = json.loads((ROOT / 'tests/fixtures/digest_parent_completion_1301.json').read_text())
    frozen = {row['id']: row for row in registry['calls']}
    selected = {row['id'] for row in registry['calls'] if Path(row['file']).stem in NAMES}
    rows = ledger['completed_calls'] + ledger['retained_calls']
    assert {row['original_id'] for row in rows} == selected
    assert len(rows) == len(selected)
    selections = {entry['fixture']: entry for entry in ledger['source_selections']}
    for entry in selections.values():
        raw = (ROOT / entry['fixture']).read_bytes()
        assert len(raw) == entry['selected_bytes'] < entry['original_bytes']
        # Slice hashes were captured against the frozen original git objects.
        # Do not require repository history in source-only test deployments.
        fixture_lines = raw.splitlines(keepends=True)
        covered = set()
        for segment in entry['segments']:
            a, b = segment['original_lines']
            c, d = segment['fixture_lines']
            chunk = b''.join(fixture_lines[c - 1:d])
            assert b - a == d - c
            assert hashlib.sha256(chunk).hexdigest() == segment['sha256']
            covered.update(range(c - 1, d))
        # No unannotated executable source may be smuggled into the fixture.
        assert all(i in covered or not line.strip() or line.lstrip().startswith(b'#')
                   for i, line in enumerate(fixture_lines))
    for row in rows:
        old = frozen[row['original_id']]
        entry = selections[row['fixtures'][0]]
        assert entry['source_blob'] == old['source_blob'] == row['source_blob']
        assert entry['source_file'] == old['file']
        assert row['source_commit'] == registry['source_commit']
        assert row['tests']
    for row in ledger['completed_calls']:
        old = frozen[row['original_id']]
        raw = (ROOT / old['file']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['current_source_blob']
        tree = ast.parse(raw)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == old['scope'])
        nodes = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == row['owner_route'].removeprefix('prismaquant.digests.')]
        assert len(nodes) == 1
        node = nodes[0]
        assert row['current_expression'] == ast.get_source_segment(raw.decode(), node)
        assert row['current_node_id'] == f"{ast.unparse(node.func)}:{old['file']}::{old['scope']}@{node.lineno}:{node.col_offset}"


def test_chain_prewarm_standalone_import_isolated(tmp_path):
    script = '''import gzip, hashlib, importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('chain_standalone', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.bytes_sha256hex(b'chain') == hashlib.sha256(b'chain').hexdigest()
doc = {'annotations': {'entry_point': 'prismaquant.tessera_joint_aura:prepare',
       'plan_sha256': 'plan', 'phase_start_units': {'层': 'head'},
       'phases': [{'name': 'head'}]}}
raw = json.dumps(doc, ensure_ascii=False).encode()
for name, blob in [('manifest.json', raw), ('manifest.json.gz', gzip.compress(raw, mtime=0))]:
    path = pathlib.Path(name)
    path.write_bytes(blob)
    result = module.load_prepare_read_set(str(path), hashlib.sha256(blob).hexdigest(), 'plan')
    assert result[0] == doc['annotations']['phase_start_units']
    assert result[1] is None
    assert result[2] == ('head',)
# No replay stub: load_prepare_read_set invokes the real sibling replay parser.
assert module._REPLAY_MODULE is not None
assert 'prismaquant' not in sys.modules
assert 'torch' not in sys.modules
'''
    # No source checkout on sys.path; import must use only stdlib and siblings.
    result = subprocess.run([sys.executable, '-I', '-c', script,
                             str(ROOT / 'prismaquant/joint_prewarm_phases.py')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_chain_prewarm_standalone(tmp_path):
    """The path-imported manifest consumer must bind a checkout-local owner."""
    path = ROOT / 'prismaquant/joint_prewarm_phases.py'
    spec = importlib.util.spec_from_file_location('_chain_standalone_prewarm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.bytes_sha256hex.__module__.startswith('_prismaquant_standalone_digest_owner_')
    assert capture_prewarm(module, tmp_path) == json.loads((FIXTURES / 'chain_outcomes.json').read_text())['prewarm']
