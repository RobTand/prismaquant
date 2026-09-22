"""Owned temporary tensor transport, not scientific measurement evidence."""
import hashlib,json
from pathlib import Path
import pytest
import torch
from experiments.native_local_handoff import LocalHandoff,LocalSourceMapping,strict_read


def test_bounded_handoff_preserves_tensor_and_exact_wire_bytes(tmp_path):
    owner=LocalHandoff(tmp_path/'local',max_bytes=1<<20)
    tensor=torch.arange(64,dtype=torch.bfloat16).reshape(8,8)
    source=owner.write('source.pt',tensor=tensor)
    wire=owner.write('wire.bin',raw=b'original wire bytes')
    record=owner.seal('admitted-fixture')
    manifest=json.loads(Path(record['path']).read_text())
    assert torch.equal(LocalSourceMapping({'q':source})['q'],tensor)
    assert wire['sha256']==hashlib.sha256(b'original wire bytes').hexdigest()
    assert manifest['reserved_bytes']<=manifest['max_bytes']
    assert set(manifest['files'])=={'source.pt','wire.bin'}


def test_handoff_refuses_capacity_before_creating_payload(tmp_path):
    owner=LocalHandoff(tmp_path/'local',max_bytes=4)
    with pytest.raises(RuntimeError,match='ceiling'):owner.write('too-big',raw=b'12345')
    assert not (owner.root/'too-big').exists()
    owner.close()


def test_local_source_change_refuses(tmp_path):
    owner=LocalHandoff(tmp_path/'local',max_bytes=1<<20)
    record=owner.write('source.pt',tensor=torch.ones(1));owner.seal('fixture')
    path=Path(record['path']);path.chmod(0o600);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):LocalSourceMapping({'q':record})['q']


def test_strict_metadata_transport_cannot_fall_back_to_canonical_file(tmp_path,monkeypatch):
    from prismaquant import tessera_joint_aura as reader
    from prismaquant.staged_tier_policy import staged_tier_policy_test_context
    path=tmp_path/'metadata.json';path.write_bytes(b'{}')
    monkeypatch.setattr(reader,'residency_resolver',lambda:None)
    with staged_tier_policy_test_context('ram,ssd'):
        with pytest.raises(Exception,match='readset-not-staged'):
            strict_read({'path':str(path),'sha256':hashlib.sha256(b'{}').hexdigest()})


def test_policy_alternate_reader_must_return_exact_bound_bytes():
    from prismaquant.joint_served_activation import _policy_bytes
    data=b'{"original": true}';bound={'path':'unused','sha256':hashlib.sha256(data).hexdigest()}
    assert _policy_bytes(bound,'fixture',lambda b:data)==data
    with pytest.raises(ValueError,match='changed bound content'):
        _policy_bytes(bound,'fixture',lambda b:b'{"original": false}')
