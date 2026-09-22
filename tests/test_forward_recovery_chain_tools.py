"""Chained forward recovery: capsule read bounds and the rendered launcher.

The R11 package was built by string-patching the reviewed R9 launcher and a
hard-coded original manifest digest. The builder now renders a template from
declared fields. Rendering R11's fields must reproduce the launcher R11 was
sealed with, statement for statement.

CPU-only; no campaign file is read.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / 'tools' / 'templates' / 'glm_full512_stagea_campaign_fields.json'

#: sha256(ast.dump(ast.parse(launch-r11.py))) under CPython 3.12. launch-r11.py
#: (sha256 e2e5b6f8c115...) is the launcher R11 was sealed with, at
#: /mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/r11-20260922.
R11_LAUNCHER_AST_SHA256 = '44929497b365e9d021b062ec8eef7f3391396297c5fd04e952f27e0782247975'
R11 = {
    'label': 'r11',
    'source_head': '34a07e0f2ebc3ebd11a8997fb50964289ce88004',
    'manifest_sha256': '7b1c2962495dbb384f075494d83ebf54b26538aa64ede6de42c50a2e598f9b1d',
    'capsule': {'path': '/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/'
                        'forward-recovery-20260922/r10-through-44-to-r11-capsule.json',
                'sha256': 'c68641aaae918d6037dceb0229e43686c45afca29e9a74eb4634b02a8316cec3'},
}


def _campaign():
    return json.loads(FIELDS.read_text())


def test_rendered_launcher_is_the_sealed_r11_launcher():
    from tools.build_stagea_forward_recovery_package import render_launcher
    text = render_launcher(_campaign(), **R11)
    compile(text, 'launch-r11.py', 'exec')
    digest = hashlib.sha256(ast.dump(ast.parse(text)).encode()).hexdigest()
    assert digest == R11_LAUNCHER_AST_SHA256


@pytest.mark.parametrize('mutate', [
    lambda fields: fields.pop('panel'),
    lambda fields: fields.update(unreviewed='x'),
])
def test_launcher_fields_are_exact(mutate):
    from tools.build_stagea_forward_recovery_package import render_launcher
    fields = _campaign()
    mutate(fields)
    with pytest.raises(ValueError, match='launcher fields'):
        render_launcher(fields, **R11)


def test_builder_carries_no_launcher_or_manifest_literal():
    source = (ROOT / 'tools' / 'build_stagea_forward_recovery_package.py').read_text()
    assert 'launch-r9.py' not in source
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not [s for s in strings if len(s) == 64 and set(s) <= set('0123456789abcdef')]


# ------------------------------------------------------------------ read bounds

def _capsule(groups, header_bytes=0):
    return {'schema': 'prismaquant.joint_forward_recovery.v1', 'padding': 'x' * header_bytes,
            'groups': [{'manifest': {'entries': []}, 'manifest_raw': 'y' * 400,
                        'receipt': {}, 'record': {}} for _ in range(groups)]}


def _write(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_an_unpinned_proof_keeps_the_record_bound(tmp_path, monkeypatch):
    from prismaquant import joint_forward_resume as recovery
    monkeypatch.setattr(recovery, 'PROOF_READ_MAX_BYTES', 1024)
    path = tmp_path / 'record.json'
    _write(path, {'blob': 'z' * 2048})
    with pytest.raises(recovery.ForwardRecoveryRefused, match='bounded'):
        recovery._read(path)


def test_a_pinned_capsule_is_bounded_by_its_declared_groups(tmp_path, monkeypatch):
    from prismaquant import joint_forward_resume as recovery
    monkeypatch.setattr(recovery, 'PROOF_READ_MAX_BYTES', 1024)
    path = tmp_path / 'capsule.json'
    # Twelve groups make a capsule far above one record's bound; it loads.
    digest = _write(path, _capsule(12))
    assert path.stat().st_size > 4 * 1024
    document, _ = recovery._read(path, digest)
    assert len(document['groups']) == 12
    # A header larger than its declared composition allows refuses.
    digest = _write(path, _capsule(0, header_bytes=3 * 1024))
    with pytest.raises(recovery.ForwardRecoveryRefused, match='declared groups'):
        recovery._read(path, digest)


def test_a_pinned_read_refuses_a_digest_mismatch_before_loading(tmp_path, monkeypatch):
    from prismaquant import joint_forward_resume as recovery
    path = tmp_path / 'capsule.json'
    _write(path, _capsule(1))
    loaded, real = [], Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes',
                        lambda self: loaded.append(self) or real(self))
    with pytest.raises(recovery.ForwardRecoveryRefused, match='SHA256'):
        recovery._read(path, '0' * 64)
    assert loaded == []
