"""Execute actual Stage-A consumers against selected independent base sources."""
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from prismaquant import joint_forward_resume as resume
from prismaquant import joint_replay_frontier as replay
from prismaquant import stage_a_chain_seed as seed

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures/digest_1301/stagea-custody'


@pytest.fixture
def original():
    return SimpleNamespace(**runpy.run_path(str(FIXTURES / 'original_consumers.py')))


@pytest.fixture
def outcomes():
    return json.loads((FIXTURES / 'outcomes.json').read_text())


def test_replay_original_outcomes(original, outcomes):
    roster, cells = outcomes['roster'], outcomes['cells']
    items = replay.replay_read_items(roster, cells)
    assert items == original.replay_read_items(roster, cells)
    for consumer in (original, replay):
        assert consumer.roster_sha256(iter(roster)) == outcomes['roster_sha256']
        assert consumer.read_order_sha256(items) == outcomes['read_order_sha256']
        assert consumer.roster_sha256([]) == outcomes['empty_roster_sha256']
        assert consumer.read_order_sha256([]) == outcomes['empty_read_order_sha256']
        assert isinstance(consumer._framed(items), bytes)
        assert consumer._framed(items).hex() == outcomes['read_order_sha256']
        assert consumer.read_order_sha256(list(reversed(items))) != outcomes['read_order_sha256']
        assert consumer.roster_sha256(reversed(roster)) != outcomes['roster_sha256']
    # Unsorted JSON is deliberately retained: even permissive off-annotation
    # callers must not silently acquire sorted-key serialization.
    odd = [('capture', {'z': 1, 'a': 2}, None)]
    assert replay._framed(odd) == original._framed(odd)
    assert replay.roster_sha256([{'z': 1, 'a': 2}]) == original.roster_sha256([{'z': 1, 'a': 2}])
    assert replay._framed([('a', 'bc', None)]) != replay._framed([('ab', 'c', None)])


def test_replay_sealed_consumer(original, outcomes):
    roster = sorted(outcomes['roster'])
    cells = outcomes['cells']
    sealed = {
        replay.ROSTER_KEY: roster,
        replay.ROSTER_SHA256_KEY: original.roster_sha256(roster),
        replay.READ_ORDER_SHA256_KEY: original.read_order_sha256(original.replay_read_items(roster, cells)),
        replay.JOURNAL_IDENTITY_KEY: 'a' * 64,
        replay.PHASE_START_UNITS_KEY: {name: replay.replay_phase_name(0) for name in roster},
    }
    assert replay.sealed_from_annotations(sealed) == sealed
    replay.require_replay_matches(sealed, completed=roster, cells_by_unit=cells,
                                  journal_identity_sha256='a' * 64)
    with pytest.raises(RuntimeError, match='read order'):
        replay.require_replay_matches(sealed, completed=roster,
                                      cells_by_unit={**cells, '层.é': ['f1', 'f2']},
                                      journal_identity_sha256='a' * 64)


def test_resume_pinned_and_unpinned(original, outcomes, tmp_path):
    path = tmp_path / 'proof.json'
    # Cross the 1 MiB streaming boundary without exceeding one proof's bound.
    raw = outcomes['proof_raw'].encode() + b' ' * (1 << 20)
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    for consumer in (original, resume):
        for pin in (None, expected):
            assert consumer._read(path, pin) == (outcomes['proof_document'], expected)


def test_resume_wrong_pin_before_loading(original, outcomes, tmp_path, monkeypatch):
    path = tmp_path / 'proof.json'
    path.write_text(outcomes['proof_raw'])
    def forbidden(self):
        raise AssertionError('wrong pin must refuse before materialization')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    for consumer in (original, resume):
        with pytest.raises(consumer.ForwardRecoveryRefused, match=outcomes['wrong_pin_refusal']):
            consumer._read(path, '0' * 64)


def test_resume_stat_fences(original, outcomes, tmp_path, monkeypatch):
    path = tmp_path / 'proof.json'
    raw = outcomes['proof_raw'].encode()
    expected = hashlib.sha256(raw).hexdigest()
    real_lstat = Path.lstat
    for consumer in (original, resume):
        path.write_bytes(raw)
        calls = 0
        def moving(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if self == path and calls == 2:
                path.write_bytes(raw + b' ')
            return real_lstat(self, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'lstat', moving)
            with pytest.raises(consumer.ForwardRecoveryRefused, match=outcomes['mutation_refusal']):
                consumer._read(path, expected)
    # Mutation on the second read is independently fenced too.
    real_read = Path.read_bytes
    for consumer in (original, resume):
        path.write_bytes(raw)
        def changing(self):
            result = real_read(self)
            self.write_bytes(result + b' ')
            return result
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_bytes', changing)
            with pytest.raises(consumer.ForwardRecoveryRefused, match=outcomes['mutation_refusal']):
                consumer._read(path, expected)


def test_resume_regular_and_declared_bounds(original, outcomes, tmp_path, monkeypatch):
    path = tmp_path / 'proof.json'
    path.write_text(outcomes['proof_raw'])
    link = tmp_path / 'link'
    link.symlink_to(path)
    for consumer in (original, resume):
        for invalid in (link, tmp_path):
            with pytest.raises(consumer.ForwardRecoveryRefused, match=outcomes['nonregular_refusal']):
                consumer._read(invalid)
        # runpy functions retain their globals dict, not SimpleNamespace attrs.
        with monkeypatch.context() as patch:
            patch.setitem(consumer._read.__globals__, 'PROOF_READ_MAX_BYTES', 4)
            with pytest.raises(consumer.ForwardRecoveryRefused, match='bounded regular'):
                consumer._read(path)
            capsule = json.dumps({'schema': consumer.SCHEMA, 'groups': []}).encode()
            path.write_bytes(capsule)
            with pytest.raises(consumer.ForwardRecoveryRefused, match='declared groups'):
                consumer._read(path, hashlib.sha256(capsule).hexdigest())
        path.write_text(outcomes['proof_raw'])


def test_seed_tensor_bounded_original(original, outcomes, monkeypatch):
    torch = pytest.importorskip('torch')
    tensors = [torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
               torch.tensor(1.25), torch.empty(0),
               torch.tensor([1.0, -0.0, float('inf')], dtype=torch.bfloat16),
               torch.arange(17, dtype=torch.uint8)]
    real_to = torch.Tensor.to
    for tensor in tensors:
        for chunk_bytes in (1, 9, 64):
            expected = original.tensor_payload_sha256(tensor, chunk_bytes=chunk_bytes)
            copies = []
            def tracked(self, *args, **kwargs):
                copies.append(self.numel() * self.element_size())
                return real_to(self, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor, 'to', tracked)
                assert seed.tensor_payload_sha256(tensor, chunk_bytes=chunk_bytes) == expected
            assert all(n <= max(chunk_bytes, tensor.element_size()) for n in copies)
    assert seed.tensor_payload_sha256(torch.empty(0)) == outcomes['tensor_empty_sha256']


def test_replay_seed_package_free_prewarm_end_to_end_1301(tmp_path):
    """Exercise prewarm -> replay with package imports forbidden in isolation."""
    import subprocess
    import sys

    sealed = replay.seal_frontier(
        ["a", "β"], {"a": ["int4"], "β": ["fp8"]},
        phase_start_units={"a": replay.replay_phase_name(0),
                           "β": replay.replay_phase_name(0)},
        journal_identity_sha256="a" * 64)
    annotations = {
        "entry_point": "prismaquant.tessera_joint_aura:prepare",
        "plan_sha256": "plan", "phase_start_units": {"next": "layer-0-part-0"},
        "phases": [{"name": "head"}, {"name": "layer-0-part-0"},
                   {"name": replay.replay_phase_name(0)}], **sealed}
    blob = json.dumps({"annotations": annotations}, ensure_ascii=False).encode()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(blob)
    script = r'''
import importlib.abc, importlib.util, json, sys
class NoPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'prismaquant', 'torch'}:
            raise AssertionError('package import forbidden: ' + fullname)
sys.meta_path.insert(0, NoPackage())
spec = importlib.util.spec_from_file_location('prewarm_without_package', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
starts, sealed, phases = module.load_prepare_read_set(sys.argv[2], sys.argv[3], 'plan')
assert module.__package__ == ''
assert module._REPLAY_MODULE.__package__ == ''
assert starts == {'next': 'layer-0-part-0'}
assert sealed == json.loads(sys.argv[4])
assert phases == ('head', 'layer-0-part-0', 'replay-0000')
assert 'prismaquant' not in sys.modules
print('package-free prewarm end-to-end passed')
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", script,
         str(Path(replay.__file__).with_name("joint_prewarm_phases.py")), str(manifest),
         hashlib.sha256(blob).hexdigest(), json.dumps(sealed)],
        cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "package-free prewarm end-to-end passed"
