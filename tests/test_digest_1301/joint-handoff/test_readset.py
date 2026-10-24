"""Actual readset verifier vs independent original consumer, including hash-before-use."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from tools import check_readset_coverage as live

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/digest_1301/joint-handoff/readset_original.py'


def original():
    spec = importlib.util.spec_from_file_location('readset_original', FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__doc__ = live.__doc__
    return module


def test_readset_hash_before_decode(tmp_path, monkeypatch):
    old = original()
    path = tmp_path / 'input.json'
    values = [b'{"z":"\\u96ea","a":[1,null]}\n', b'[]', b'{invalid', b'\xff', b'']
    feeds = []
    real_hash = live.bytes_sha256hex
    monkeypatch.setattr(live, 'bytes_sha256hex', lambda raw: (feeds.append(raw), real_hash(raw))[1])
    for raw in values:
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        for expected in [None, digest, '0' * 64]:
            outcomes = []
            for consumer in [old, live]:
                try:
                    outcomes.append(('ok', consumer._load_json(str(path), expected)))
                except Exception as exc:
                    outcomes.append((type(exc), str(exc)))
            assert outcomes[0] == outcomes[1]
        assert feeds[-2:] == [raw, raw]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        live._load_json(str(path), '0' * 64)


def test_readset_publication_and_refusal(tmp_path, monkeypatch, capsys):
    old = original()
    report = {'covered': False, 'z': '雪', 'object': Path('fixture')}
    outcomes = []
    target = tmp_path / 'report.json'
    for consumer in [old, live]:
        monkeypatch.setattr(consumer, '_stage_b', lambda args: report, raising=False)
        code = consumer.main(['--out', str(target), 'stage-b', '--records', 'unused'])
        output = capsys.readouterr()
        outcomes.append((code, output.out, output.err, target.read_bytes()))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == 3
    assert outcomes[0][3] == outcomes[0][1].encode()
    outcomes = []
    def refuse(args):
        raise ValueError('bad manifest')
    for consumer in [old, live]:
        monkeypatch.setattr(consumer, '_stage_b', refuse)
        code = consumer.main(['stage-b', '--records', 'unused'])
        output = capsys.readouterr()
        outcomes.append((code, output.out, output.err))
    assert outcomes[0] == outcomes[1] == (2, '', 'check_readset_coverage: refused: bad manifest\n')
