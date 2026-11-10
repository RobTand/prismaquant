"""CPU execution controls layered above the immutable staged-input fixtures."""
import pytest

from staged_native_fixtures import bind, fixture, native_staged_state


@pytest.fixture
def empty_native_staged_inputs(tmp_path, monkeypatch, native_staged_state):
    """Bind a real manifest for CPU controls whose payload IO is stubbed."""
    _, _, binding, _ = fixture(tmp_path)
    bind(monkeypatch, binding)
    from prismaquant import tessera_joint_aura, residency_map
    binder = residency_map.bind_residency_manifest
    monkeypatch.setattr(tessera_joint_aura, 'bind_residency_manifest',
        lambda expected: binder(binding['read_contract_sha256'] if expected is None else expected))
    monkeypatch.setattr(residency_map, 'bind_residency_manifest',
        lambda expected: binder(binding['read_contract_sha256'] if expected is None else expected))
    return binding
