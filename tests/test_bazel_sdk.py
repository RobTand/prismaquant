"""Bazel's archive install proves the same reviewed SDK and actual package bytes."""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import pytest

from prismaquant import staged_lease

ARCHIVE_SHA256 = 'e2f64a500db635dcc619338e3ced8bfff14f28d23a588a4a01897a944f712970'


@pytest.fixture
def installed(tmp_path, monkeypatch):
    package = tmp_path / 'prismabuild'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'client.py').write_text('SDK_VERSION = 5\n')
    info = tmp_path / 'prismabuild-0.1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: prismabuild\nVersion: 0.1.0\n')
    (info / 'top_level.txt').write_text('prismabuild\n')
    source = {'schema': 'prismaquant.bazel-sdk-source.v1',
              'commit': staged_lease.PB_READER_LEASE_PIN_COMMIT,
              'url': 'https://codeload.github.com/RobTand/prismabuild/tar.gz/' + staged_lease.PB_READER_LEASE_PIN_COMMIT,
              'sha256': ARCHIVE_SHA256}
    (info / 'prismaquant_bazel_source.json').write_text(json.dumps(source))
    with (info / 'RECORD').open('w') as handle:
        writer = csv.writer(handle)
        for path in sorted(package.glob('*.py')):
            content = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
            writer.writerow([str(path.relative_to(tmp_path)), 'sha256='+digest, len(content)])
    dist = metadata.PathDistribution(info)
    monkeypatch.setattr(metadata, 'packages_distributions', lambda: {'prismabuild': ['prismabuild']})
    monkeypatch.setattr(metadata, 'distribution', lambda name: dist)
    return dist, package, info


def test_the_pinned_bazel_archive_and_its_record_are_accepted(installed):
    dist, _, _ = installed
    assert staged_lease.reviewed_installed_distribution() is dist


@pytest.mark.parametrize('field', ['commit', 'url', 'sha256'])
def test_bazel_cannot_claim_another_sdk_source(installed, field):
    _, _, info = installed
    path = info / 'prismaquant_bazel_source.json'
    source = json.loads(path.read_text())
    source[field] = 'wrong'
    path.write_text(json.dumps(source))
    with pytest.raises(RuntimeError):
        staged_lease.reviewed_installed_distribution()


@pytest.mark.parametrize('change', ['changed', 'missing', 'unlisted', 'no-record'])
def test_bazel_refuses_unproven_package_bytes(installed, change):
    _, package, info = installed
    if change == 'changed':
        (package / 'client.py').write_text('SDK_VERSION = 0\n')
    elif change == 'missing':
        (package / 'client.py').unlink()
    elif change == 'unlisted':
        (package / 'extra.py').write_text('EXTRA = True\n')
    else:
        (info / 'RECORD').unlink()
    with pytest.raises(RuntimeError):
        staged_lease.reviewed_installed_distribution()
