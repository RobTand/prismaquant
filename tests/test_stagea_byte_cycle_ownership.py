"""Actual shipped caller's directory-creation boundary; admitted execution only."""
import os
from pathlib import Path
import pytest
from prismaquant.stage_a_produced_output import BoundaryProducedPublication, build_boundary_template, publish_document
from produced_native_fixtures import selected_publications
from tools import stagea_produced_byte_cycle as tool


def roster(root):
    result = {}
    for path in sorted(root.rglob('*')):
        result[str(path.relative_to(root))] = ('symlink', os.readlink(path)) if path.is_symlink() else (
            ('directory', None) if path.is_dir() else ('file', path.read_bytes()))
    return result


@pytest.mark.parametrize('layout', ['intact', 'occupied', 'prefix-symlink', 'generation-symlink'])
def test_actual_byte_caller_refuses_before_foreign_mkdir(tmp_path, monkeypatch, layout):
    foreign = tmp_path / 'foreign'; foreign.mkdir()
    (foreign / 'known').write_bytes(b'foreign bytes must stay exact')
    prefix = tmp_path / 'output'
    if layout == 'prefix-symlink':
        prefix.symlink_to(foreign, target_is_directory=True)
    else:
        prefix.mkdir()
    generation = prefix / 'known-byte-fixture'
    if layout == 'generation-symlink':
        generation.symlink_to(foreign, target_is_directory=True)
    elif layout == 'occupied':
        (generation / 'entries').mkdir(parents=True)
        (generation / 'entries' / 'occupied').write_bytes(b'occupied original')
    local = tmp_path / 'local'; local.mkdir()
    record = tmp_path / 'record'; record.mkdir()
    state = publish_document(tmp_path / 'state.json', {'original_bind_identity': {'fixture': 'private bytes'}})
    declaration = build_boundary_template(output_prefix=prefix, artifact_max_bytes=1 << 20,
        group_size=2, max_entry_tensor_bytes=128, concurrent_groups=8,
        spool={'root': str(local), 'max_bytes': 1 << 20}, inputs={'forward_state': state})
    binding = publish_document(tmp_path / 'declaration.json', declaration)
    producer = {'schema':'prismaquant.flux_result_producer.v1', 'job_id':'123',
        'record_dir':str(record), 'launch_sha256':'a'*64,
        'argv':['fixture','--produced-declaration',binding['path'],
            '--produced-declaration-sha256',binding['sha256']]}
    publication = BoundaryProducedPublication(declaration=declaration, binding=binding, producer=producer)
    selected_publications(monkeypatch)
    monkeypatch.setattr(BoundaryProducedPublication,'bind_from_admitted_owner',classmethod(lambda cls:publication))
    before_foreign, before_prefix = roster(foreign), roster(prefix)
    attempts=[]
    original=os.mkdir
    def observed(path,*args,**kwargs):
        parent = kwargs.get('dir_fd')
        actual = Path(os.readlink(f'/proc/self/fd/{parent}')) / path if parent is not None else Path(path)
        # mkdir follows parent components, but never its final component.
        actual = actual.parent.resolve(strict=False) / actual.name
        if actual == foreign or foreign in actual.parents:
            attempts.append(str(actual))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'mkdir',observed)
    if layout == 'intact':
        assert tool.main([]) == 0
        entries = generation/'entries'
        assert {p.name:p.read_bytes() for p in entries.iterdir()} == {
            'boundary-0-0-at-0.pt':b'\x00\x00\x00\xffknown-native-fixture',
            'boundary-1-0-at-0.pt':b'\x00\x00\x01\xffknown-native-fixture',
            'boundary-0-1-at-1.pt':b'\x00\x01\x00\xffknown-native-fixture',
            'boundary-1-1-at-1.pt':b'\x00\x01\x01\xffknown-native-fixture'}
        assert len(publication._groups)==2
    else:
        with pytest.raises((OSError, ValueError)):
            tool.main([])
        assert not attempts, 'no mkdir attempt may reach the foreign target'
        assert roster(foreign)==before_foreign
        assert roster(prefix)==before_prefix
        assert publication._groups=={}
        assert not (record/'produced'/'index.json').exists()
        assert not any(p.name.startswith('.copy-') for p in tmp_path.rglob('*'))
