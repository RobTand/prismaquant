"""Exact write-only native application handoff publication and consumer joins.

The offline constructor grants no terminal native authority. The native proof
independently qualifies execution; these tests keep the actual scientific
emitter, tensor reader, state serializer and immutable application batches.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
import torch
from prismaquant.stage_a_produced_output import (BoundaryProducedPublication,
    build_boundary_template, publish_document, read_bound_document)
from test_quantum_executable_readset import (N_BATCHES, N_PROBES, _bind_slice,
    _tiny_receipt, _tiny_records)

PREFETCH = 2
ARTIFACT_MAX = 1 << 20
ENTRY_GROUPS = N_PROBES * -(-N_BATCHES // PREFETCH)

class _Owner:
    def __init__(self, state):
        self.state = state
    def state_dict(self):
        return dict(self.state)

def _adjoint_slice(record):
    return json.loads(Path(record["adjoint"]["slice_path"]).read_text())


def band_campaign(tmp_path, *, executable=False):
    """Layers 3 (producer) and 2 (consumer) of band 4, bound to Stage A."""
    from prismaquant.streamed_boundary_artifacts import BOUNDARY_STORAGE_SCHEMA

    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    bound = {record["layer"]: _bind_slice(
        record, receipt, tmp_path / "adjoint-slices")[0]
        for record in records if record["layer"] in (2, 3)}
    if executable:
        from prismaquant import joint_layer_quanta as jl
        from test_quantum_executable_readset import CALIB, STRIDED, RENDER_PREREQ
        kwargs = dict(strided_boundaries=STRIDED, n_probes=N_PROBES,
            calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ))
        manifest = jl.build_quantum_executable_manifest(bound[2], receipt, _parent, **kwargs)
        wire = jl.seal_manifest_bytes(manifest)
        path = Path(jl.bound_readset_directory(str(tmp_path))) / 'layer-002.executable.json.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wire)
        bound[2] = jl.bind_quantum_executable(bound[2], receipt, _parent,
            manifest=manifest, manifest_path=str(path),
            manifest_sha256=hashlib.sha256(wire).hexdigest(), output_root=str(tmp_path), **kwargs)
    storage = {"schema": BOUNDARY_STORAGE_SCHEMA,
               "directory": str(tmp_path / "exact"),
               "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
               "max_artifact_bytes": ARTIFACT_MAX, "prefetch_batches": PREFETCH}
    return bound[3], bound[2], storage


def emit_handoff(producer, storage, publication, *, emitters=None):
    from prismaquant.joint_quantum_handoff import HandoffEmitter

    plane = {(p, b): torch.full((2, 4), 10.0 * p + b)
             for p in range(N_PROBES) for b in range(N_BATCHES)}
    owners = [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
              for p in range(N_PROBES)]
    emitter = HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
                             boundary_storage=storage, capture_batch=1,
                             publication=publication)
    if emitters is not None:
        emitters.append(emitter)
    published = emitter.emit(grad_plane=plane, cotangent_owners=owners,
                             n_probes=N_PROBES, n_batches=N_BATCHES,
                             kda_capture_kernel=None)
    return published, plane


def check_consumer_binds(published, plane, consumer, publication):
    """The consumer's own checks accept the handoff, and it is the plane."""
    from prismaquant.joint_adjoint_checkpoints import read_exact_entry_tensors
    from prismaquant.joint_quantum_handoff import load_quantum_handoff

    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer,
                                   adjoint_slice=_adjoint_slice(consumer),
                                   kda_capture_kernel=None)
    for entry in handoff["activation_entries"]:
        assert publication.contains(Path(entry["path"]))
        coordinates = entry["metadata"]["identity"]["coordinates"]
        tensor = read_exact_entry_tensors(
            [entry], expected_session=handoff["session"])[entry["name"]]
        assert torch.equal(tensor, plane[(coordinates["probe"],
                                          coordinates["batch"])])
    assert publication.contains(Path(handoff["owner_states"]["path"]))
    assert publication.contains(Path(published["path"]))
    return handoff


def producer_owner(tmp_path, producer, storage, monkeypatch, *, spool=False):
    from prismaquant.joint_quantum_handoff import handoff_root
    from produced_native_fixtures import selected_publications
    prefix = handoff_root(producer['output_space']['root'])
    prefix.mkdir(parents=True, exist_ok=True)
    local = tmp_path / 'local'; local.mkdir()
    declaration = build_boundary_template(output_prefix=prefix,
        artifact_max_bytes=ARTIFACT_MAX, group_size=PREFETCH,
        max_entry_tensor_bytes=32, write_only=True,
        spool={'root': str(local), 'max_bytes': 1 << 20} if spool else None)
    binding = publish_document(tmp_path / 'declaration.json', declaration)
    record = tmp_path / 'native-record'; record.mkdir()
    active = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': '123',
        'record_dir': str(record), 'launch_sha256': 'a' * 64,
        'argv': ['fixture', '--produced-declaration', binding['path'],
                 '--produced-declaration-sha256', binding['sha256']]}
    pub = BoundaryProducedPublication(declaration=declaration, binding=binding, producer=active)
    register = selected_publications(monkeypatch)
    monkeypatch.setattr(pub, 'finalize', lambda *a, **k: register(pub))
    return pub


def check_origin_batches(publication, producer, published, handoff):
    from prismaquant.joint_quantum_handoff import (HANDOFF_ORIGIN_LIFETIME,
        HANDOFF_OWNER_STATES_NAME, HANDOFF_RECORD_BATCH_KIND, HANDOFF_RECORD_NAME)
    refs = published['origin_batches']
    assert len(refs) == ENTRY_GROUPS + 1
    groups = [read_bound_document(ref) for ref in refs]
    record_batch = publication.batch_id_for(kind=HANDOFF_RECORD_BATCH_KIND,
        boundary_index=int(producer['layer']), group_index=0)
    assert groups[-1]['batch_id'] == record_batch
    assert len({group['batch_id'] for group in groups}) == len(groups)
    for group in groups:
        assert group['producer'] == publication.producer
        assert group['session'] == handoff['session']
        assert group['lifetime'] == HANDOFF_ORIGIN_LIFETIME == 'consumed'
    directory = Path(published['path']).parent
    finals = [directory / HANDOFF_OWNER_STATES_NAME, directory / HANDOFF_RECORD_NAME]
    assert sorted(row['path'] for row in groups[-1]['entries']) == sorted(map(str, finals))
    assert not any(Path(str(f) + '.tmp').exists() for f in finals)
    rows = {row['path']: row for group in groups for row in group['entries']}
    expected = {entry['path']: (entry['file_bytes'], entry['sha256'])
                for entry in handoff['activation_entries']}
    expected[str(finals[0])] = (handoff['owner_states']['file_bytes'], handoff['owner_states']['sha256'])
    expected[str(finals[1])] = (Path(published['path']).stat().st_size, published['sha256'])
    assert {path: (row['bytes'], row['sha256']) for path, row in rows.items()} == expected
    for path, (size, sha) in expected.items():
        raw = Path(path).read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha
    return refs


@pytest.mark.parametrize('spool', [False, True])
def test_a_producer_commits_its_handoff_as_origin_batches(tmp_path, monkeypatch, spool):
    producer, consumer, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch, spool=spool)
    published, plane = emit_handoff(producer, storage, pub)
    handoff = check_consumer_binds(published, plane, consumer, pub)
    check_origin_batches(pub, producer, published, handoff)


def test_a_read_back_handoff_template_is_refused(tmp_path):
    from prismaquant.joint_quantum_handoff import HandoffEmitter, QuantumHandoffRefused
    producer, _, storage = band_campaign(tmp_path)
    class ReadBack:
        write_only = False
    with pytest.raises(QuantumHandoffRefused, match='not write-only'):
        HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
            boundary_storage=storage, capture_batch=1, publication=ReadBack())


def test_a_write_only_owner_names_its_lifetime_and_never_reads_back(tmp_path, monkeypatch):
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
    from prismaquant.joint_quantum_handoff import handoff_root
    producer, _, storage = band_campaign(tmp_path)
    pub = producer_owner(tmp_path, producer, storage, monkeypatch)
    settings = {**storage, 'directory': str(handoff_root(producer['output_space']['root']))}
    geometry = {'group_size': PREFETCH, 'n_batches': N_BATCHES, 'max_entry_tensor_bytes': 32}
    for lifetime in (None, 'forever'):
        with StreamedBoundaryArtifacts(settings) as owner:
            owner.bind({'fixture': 'lifetime'}, n_probes=N_PROBES, published=True)
            with pytest.raises(ValueError, match='lifetime and no read window'):
                owner.bind_produced_output(pub, origin_lifetime=lifetime, **geometry)
    from prismaquant.stage_a_produced_output import BoundaryProducedPublicationFailed
    with pytest.raises(BoundaryProducedPublicationFailed, match='cannot finalize empty or unbound publication'):
        with StreamedBoundaryArtifacts(settings) as owner:
            owner.bind({'fixture': 'read-back'}, n_probes=N_PROBES, published=True)
            owner.bind_produced_output(pub, origin_lifetime='consumed', **geometry)
            with pytest.raises(RuntimeError, match='never read back'):
                owner.write(torch.zeros(4), batch_index=0, boundary_index=3, probe_index=0)
            assert not list(owner.directory.rglob('*.pt*'))
            assert owner.produced_origin_batches() == []


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('maximum', [1024, 2048])
@pytest.mark.parametrize('write_only', [False, True])
def test_shared_stage_a_binding_preserves_handoff_policy(monkeypatch, native, maximum, write_only):
    from types import SimpleNamespace
    from prismaquant import joint_quantum_handoff as handoff
    from prismaquant import joint_cost_stage_a as stage_a
    from prismaquant.joint_adjoint_checkpoints import AdjointIdentityRefused

    if native:
        monkeypatch.setenv('FLUX_JOB_ID', '123')
    else:
        monkeypatch.delenv('FLUX_JOB_ID', raising=False)
    calls = []
    publication = SimpleNamespace(write_only=write_only,
        durable_maxima=lambda: {'payload_max_bytes': maximum})

    def bind():
        calls.append(True)
        return publication

    monkeypatch.setattr(BoundaryProducedPublication, 'bind_from_admitted_owner', bind)
    def normal():
        return stage_a.bind_stage_a_produced_output(artifact_max_bytes=1024)
    def transfer():
        return handoff.bind_handoff_publication(boundary_storage={'max_artifact_bytes': 1024})
    for invoke in (normal, transfer):
        if not native:
            assert invoke() is None
        elif maximum != 1024:
            with pytest.raises(AdjointIdentityRefused) as refused:
                invoke()
            assert str(refused.value) == 'launch-bound artifact maximum differs from effective Stage A budget'
        elif invoke is transfer and not write_only:
            with pytest.raises(handoff.QuantumHandoffRefused) as refused:
                invoke()
            assert str(refused.value) == 'handoff requires a write-only declaration'
        else:
            assert invoke() is publication
    assert len(calls) == (2 if native else 0)


def test_unadmitted_handoff_never_reads_the_declaration_budget(monkeypatch):
    from prismaquant.joint_quantum_handoff import bind_handoff_publication
    monkeypatch.delenv('FLUX_JOB_ID', raising=False)
    assert bind_handoff_publication(boundary_storage={}) is None
