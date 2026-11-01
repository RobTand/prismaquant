"""The GLM release oracle must not depend on distinct automatic timestamps.

Keep this separate from the release oracle, which is the fixture being repaired.
The source adapter intentionally uses stat freshness after its one digest pass;
this regression does not require repeated payload hashing or stronger witnesses.
"""
from pathlib import Path
import runpy

from experiments import glm_original_graph_source as source_module


def test_modify_oracle_refuses_and_closes_with_colliding_write_timestamps(tmp_path, monkeypatch):
    """Model a filesystem tick collision, but honor deliberate mtime changes.

    Only the identity view of automatic writes is frozen. Device, inode, size,
    real bytes and descriptors are untouched. A fixture can establish observable
    drift with os.utime, as other source-freshness tests already do.
    """
    oracle_path = Path(__file__).with_name('test_glm_original_graph_profile.py')
    oracle = runpy.run_path(str(oracle_path))[
        'test_source_mutation_and_digest_refuse_and_close']
    target = tmp_path / 'source.safetensors'
    original_write = Path.write_bytes
    original_identity = source_module.stat_identity
    writes = []
    sources = []

    def write_with_colliding_automatic_timestamps(path, payload):
        result = original_write(path, payload)
        if path == target:
            writes.append(original_identity(path.stat()))
        return result

    def identity_with_colliding_automatic_timestamps(stat):
        identity = original_identity(stat)
        if writes and (identity['device'], identity['inode']) == (
                writes[0]['device'], writes[0]['inode']):
            # An implicit write may leave both freshness stamps unchanged.
            # Explicitly setting a different mtime remains observable.
            if identity['mtime_ns'] == writes[-1]['mtime_ns']:
                identity['mtime_ns'] = writes[0]['mtime_ns']
            identity['ctime_ns'] = writes[0]['ctime_ns']
        return identity

    original_authenticate = source_module.AuthenticatedSourceInputs.authenticate

    def observe_source(source, path):
        if source not in sources:
            sources.append(source)
        return original_authenticate(source, path)

    monkeypatch.setattr(Path, 'write_bytes', write_with_colliding_automatic_timestamps)
    monkeypatch.setattr(source_module, 'stat_identity', identity_with_colliding_automatic_timestamps)
    monkeypatch.setattr(source_module.AuthenticatedSourceInputs, 'authenticate', observe_source)

    # The existing oracle must still perform its real refusal and close checks;
    # on main this fails with DID NOT RAISE, not a setup or discovery failure.
    oracle(tmp_path, 'modify')

    assert len(writes) == 2, 'retain the initial payload and its in-place rewrite'
    assert writes[0]['inode'] == writes[1]['inode'], 'retain same-inode mutation coverage'
    assert writes[0]['bytes'] == writes[1]['bytes'], 'retain same-size mutation coverage'
    assert target.read_bytes() == b'changed'
    assert sources and all(not source.fds for source in sources)
