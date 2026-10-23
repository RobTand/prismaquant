"""A second process cannot consume a reseal writer's staged inode."""
from pathlib import Path
import subprocess
import sys

from prismaquant import tessera_campaign_reseal as reseal


def test_reseal_publication_keeps_staging_private_across_processes(tmp_path, monkeypatch):
    target = tmp_path / 'record.json'
    root = Path(__file__).resolve().parents[1]
    replace = reseal.os.replace
    interrupted = False

    def publish_other_writer(source, destination):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            script = (
                'import sys;sys.path.insert(0, ' + repr(str(root)) + ');'
                'from prismaquant.tessera_campaign_reseal import write_json;'
                'write_json(' + repr(str(target)) + ', {"writer": "other"})')
            result = subprocess.run([sys.executable, '-c', script],
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
        return replace(source, destination)

    monkeypatch.setattr(reseal.os, 'replace', publish_other_writer)
    reseal.write_json(target, {'writer': 'outer'})
    assert interrupted
    assert target.read_bytes() == b'{\n "writer": "outer"\n}\n'
    assert list(tmp_path.iterdir()) == [target]
