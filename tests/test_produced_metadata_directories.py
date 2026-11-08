"""Publication metadata setup refuses links before touching foreign fixture paths."""
from pathlib import Path
import pytest
from prismaquant.stage_a_produced_output import BoundaryProducedPublication
from produced_native_fixtures import publication


def roster(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob('*')}


def test_metadata_setup_uses_owned_directories_and_replays(tmp_path):
    pub = publication(tmp_path)
    record = Path(pub.producer['record_dir'])
    assert (record / 'produced' / 'batches').is_dir()
    again = BoundaryProducedPublication(declaration=pub.declaration,
        binding=pub.binding, producer=pub.producer)
    assert again._directory == record / 'produced'
    assert roster(record) == {'produced': None, 'produced/batches': None}


@pytest.mark.parametrize('component', ['produced', 'batches'])
def test_metadata_setup_refuses_symlink_without_foreign_mutation(tmp_path, component):
    pub = publication(tmp_path / 'valid')
    foreign = tmp_path / 'foreign-fixture'; foreign.mkdir()
    (foreign / 'retained.bin').write_bytes(b'independently known foreign fixture')
    (foreign / 'subdirectory').mkdir()
    before = roster(foreign)
    target = Path(pub.producer['record_dir']) / 'produced'
    (target / 'batches').rmdir()
    if component == 'produced':
        target.rmdir()
    else:
        target = target / 'batches'
    target.symlink_to(foreign, target_is_directory=True)
    refused = False
    try:
        BoundaryProducedPublication(declaration=pub.declaration,
            binding=pub.binding, producer=pub.producer)
    except (OSError, ValueError, RuntimeError):
        refused = True
    assert roster(foreign) == before
    assert refused, 'metadata setup accepted a symlink component'
