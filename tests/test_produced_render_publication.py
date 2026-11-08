"""Render planning uses immutable application declarations and launch identity.

These direct-constructor fixtures grant no native selected-result authority.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.produced_render_publication import ProducedRenderPublication
from prismaquant.stage_a_produced_output import BoundaryProducedBindingError, build_boundary_template, publish_document


@pytest.fixture
def render_owner(tmp_path, request):
    options = getattr(request, 'param', {})
    prefix = tmp_path / 'renders'; prefix.mkdir()
    template = build_boundary_template(output_prefix=prefix, artifact_max_bytes=1 << 20,
        group_size=2, max_entry_tensor_bytes=1024, template_id='render-fixture',
        write_only=options.get('write_only', False))
    template['slot'] = options.get('slot', 'renders')
    template['durable_maxima'].update(options.get('durable_maxima', {}))
    binding = publish_document(tmp_path / 'declaration.json', template)
    record = tmp_path / 'record'; record.mkdir()
    producer = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': '123',
        'record_dir': str(record), 'launch_sha256': 'a' * 64,
        'argv': ['fixture', '--produced-declaration', binding['path'],
                 '--produced-declaration-sha256', binding['sha256']]}
    return SimpleNamespace(prefix=prefix, template=template, binding=binding, producer=producer)


def _bind(owner):
    return ProducedRenderPublication(declaration=owner.template, binding=owner.binding, producer=owner.producer)


def test_render_binding_uses_exact_declared_slot_and_launch(render_owner):
    publication = _bind(render_owner)
    assert publication.declaration['slot'] == 'renders'
    assert publication.binding == render_owner.binding
    assert publication.producer['job_id'] == '123'
    assert publication.producer['launch_sha256'] == 'a' * 64


def test_render_binding_refuses_foreign_launch_declaration(render_owner):
    _bind(render_owner)
    producer = {**render_owner.producer, 'argv': ['fixture', '--produced-declaration', '/foreign',
        '--produced-declaration-sha256', render_owner.binding['sha256']]}
    with pytest.raises(BoundaryProducedBindingError, match='launch intent'):
        ProducedRenderPublication(declaration=render_owner.template, binding=render_owner.binding, producer=producer)


def test_batch_identity_is_stable_across_rebinding(render_owner):
    first, second = _bind(render_owner), _bind(render_owner)
    batch = first.render_batch_id_for('model.layers.0.mlp', 'BF16')
    assert batch == second.render_batch_id_for('model.layers.0.mlp', 'bf16')
    assert re.fullmatch(r'render-[0-9a-f]{64}', batch)
    assert first.render_batch_id_for('layer', 'INT4_W4A16_g128') == second.render_batch_id_for('layer', 'int4_w4a16_g128')


@pytest.mark.parametrize('alias,canonical', [(' BF16 ', 'bf16'), ('FP8', 'FP8_E4M3'), ('FP8_DYNAMIC', 'fp8_e4m3')])
def test_batch_identity_uses_the_weight_cache_format_contract(render_owner, alias, canonical):
    pub = _bind(render_owner)
    assert pub.render_batch_id_for('layer', alias) == pub.render_batch_id_for('layer', canonical)


@pytest.mark.parametrize('left,right', [('layer/a', 'layer?a'), ('layer_a', 'layer.a'),
    ('x' * 180 + 'a', 'x' * 180 + 'b'), ('λ', 'μ')])
def test_batch_identity_keeps_complete_render_coordinate(render_owner, left, right):
    pub = _bind(render_owner)
    assert pub.render_batch_id_for(left, 'BF16') != pub.render_batch_id_for(right, 'BF16')
    assert pub.render_batch_id_for(left, 'BF16') != pub.render_batch_id_for(left, 'FP8')


@pytest.mark.parametrize('qname,fmt', [('', 'BF16'), (None, 'BF16'), ('layer', ''), ('layer', None), ('layer', '   ')])
def test_batch_identity_refuses_empty_or_untyped_coordinates(render_owner, qname, fmt):
    with pytest.raises(ValueError, match='nonempty strings'):
        _bind(render_owner).render_batch_id_for(qname, fmt)


@pytest.mark.parametrize('render_owner', [{'slot': 'boundary_entries'}, {'write_only': True}], indirect=True)
def test_invalid_render_declaration_refuses_before_prewrite(render_owner):
    with pytest.raises(BoundaryProducedBindingError, match='slot/readback'):
        _bind(render_owner)
    assert list(render_owner.prefix.iterdir()) == []


def test_prewrite_and_inline_descriptor_use_existing_lifecycle(render_owner):
    pub = _bind(render_owner)
    pub.bind_session({'generation': 'known', 'run_identity_sha256': 'b' * 64})
    batch = pub.render_batch_id_for('layer', 'BF16')
    path = render_owner.prefix / 'layer.pt'
    for _ in range(2):
        assert pub.require_prewrite(batch_id=batch, payload_ceiling_bytes=17,
            temp_ceiling_bytes=17, paths=[str(path), str(path) + '.tmp'])['ok']
    assert not path.exists()
    payload = b'render-fixture-v1'
    from prismaquant.streamed_boundary_artifacts import _write_new_file
    identity = _write_new_file(path, payload)
    ref = SimpleNamespace(path=str(path), file_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                          writer_file_identity=identity)
    descriptor = pub.descriptor_for(ref, producer_generation=batch)
    pub.publish(batch_id=batch, descriptors=[descriptor])
    assert path.read_bytes() == b'render-fixture-v1'
    ref.sha256 = None
    with pytest.raises(ValueError, match='SHA256'):
        pub.descriptor_for(ref, producer_generation=batch)
    assert not pub.abort_prewrite(batch_id=batch)['ok']
    pub.unlink_origin(batch, SimpleNamespace(path=str(path), file_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest()))
    assert pub.abort_prewrite(batch_id=batch)['ok']


def test_render_adapter_inherits_shared_publication_and_disposal():
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    assert 'bind_from_admitted_owner' not in ProducedRenderPublication.__dict__
    for method in ('require_prewrite', 'abort_prewrite',
                   'descriptor_for', 'publish', 'commit_origin', 'unlink_origin', 'finalize'):
        assert getattr(ProducedRenderPublication, method) is getattr(BoundaryProducedPublication, method)
