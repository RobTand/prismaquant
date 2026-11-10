"""Render planning uses immutable application declarations and launch identity.

These direct-constructor fixtures grant no native selected-result authority.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.own_process

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


def _plan(publication, qname="layer", fmt="BF16", *, archive_max_bytes=17):
    assert hasattr(publication, "plan_render_prewrite"), (
        "missing render destination/prewrite plan on the admitted publication")
    return publication.plan_render_prewrite(
        qname, fmt, archive_max_bytes=archive_max_bytes)


@pytest.mark.parametrize("left,right", [("layer.a", "layer_a"), ("layer/a", "layer__a")])
def test_plan_separates_legacy_filename_collision(render_owner, left, right):
    publication = _bind(render_owner)
    first, second = _plan(publication, left), _plan(publication, right)
    assert Path(first.final_path).name == Path(second.final_path).name
    assert first.final_path != second.final_path
    assert first.batch_id != second.batch_id
    assert Path(first.final_path).is_relative_to(render_owner.prefix)
    assert first.relative_final_path == str(Path(first.final_path).relative_to(render_owner.prefix))
    assert first.temporary_path == first.final_path + ".tmp"
    assert first.producer_generation == first.batch_id


def test_prewrite_plan_uses_existing_budget_and_abort(render_owner):
    from prismaquant.stage_a_produced_output import BoundaryProducedPrewriteRefused
    publication = _bind(render_owner)
    plan = _plan(publication)
    first = publication.require_render_prewrite(plan)
    before = dict(publication._prewrites[plan.batch_id])
    second = publication.require_render_prewrite(plan)
    assert first["ok"] and second["ok"]
    # Identical immutable records can succeed without a diagnostic duplicate
    # flag; observe the private fixture's reservation, not an invented API.
    assert publication._prewrites[plan.batch_id] == before
    assert before['ceilings'] == {'payload_max_bytes': 17, 'checkpoint_max_bytes': 0, 'temp_max_bytes': 17}
    assert set(before['paths']) == {plan.final_path, plan.temporary_path}
    changed = _plan(publication, archive_max_bytes=19)
    with pytest.raises(BoundaryProducedPrewriteRefused, match="changed or released prewrite"):
        publication.require_render_prewrite(changed)
    assert publication.abort_prewrite(batch_id=plan.batch_id)['ok']
    assert plan.batch_id in publication._released
    # Released immutable batch identities cannot acquire a different ceiling.
    with pytest.raises(BoundaryProducedPrewriteRefused):
        publication.require_render_prewrite(changed)
    assert publication.abort_prewrite(batch_id=plan.batch_id)["ok"] is True
    assert list(render_owner.prefix.iterdir()) == []


def test_planning_is_pure_and_frozen(render_owner, monkeypatch):
    import torch
    from prismaquant import production_weight_cache as pwc
    publication = _bind(render_owner)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not prepare or write payloads")

    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setattr(pwc, "_canonical_rendered_weight_tensor", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    plan = _plan(publication)
    assert plan.payload_ceiling_bytes == plan.temp_ceiling_bytes == 17
    assert plan.job_id == '123'
    assert plan.launch_sha256 == 'a' * 64
    assert plan.declaration_sha256 == render_owner.binding['sha256']
    assert plan.origin_root == str(render_owner.prefix)
    with pytest.raises(FrozenInstanceError):
        setattr(plan, "final_path", "elsewhere")
    assert list(render_owner.prefix.iterdir()) == []


@pytest.mark.parametrize("fmt", [" BF16 ", "bf16", "BF16"])
def test_rebinding_and_aliases_keep_the_same_plan(render_owner, fmt):
    assert _plan(_bind(render_owner), fmt=fmt) == _plan(_bind(render_owner))


@pytest.mark.parametrize("ceiling", [None, 0, -1, True, False, 1.0, "17", 1 << 40])
def test_invalid_archive_envelopes_refuse_before_admission(render_owner, ceiling, monkeypatch):
    publication = _bind(render_owner)
    monkeypatch.setattr(publication, "require_prewrite", lambda **kwargs: pytest.fail("unexpected admission"))
    with pytest.raises(ValueError, match="ceiling|archive"):
        _plan(publication, archive_max_bytes=ceiling)


@pytest.mark.parametrize("render_owner", [
    {"durable_maxima": {"payload_max_bytes": 16, "temp_max_bytes": 32}},
    {"durable_maxima": {"payload_max_bytes": 32, "temp_max_bytes": 16}},
], indirect=True)
def test_both_independent_class_maxima_bound_the_plan(render_owner):
    with pytest.raises(ValueError, match="sealed.*maxima"):
        _plan(_bind(render_owner), archive_max_bytes=17)


@pytest.mark.parametrize("qname,fmt", [
    ("layer\x00", "BF16"), ("layer\n", "BF16"),
    ("layer\\a", "BF16"), ("layer", "../BF16"),
    ("x" * 250, "BF16"), ("λ" * 125, "BF16"),
])
def test_unsafe_or_overlong_archive_leaves_refuse(render_owner, qname, fmt):
    with pytest.raises(ValueError, match="leaf|filename"):
        _plan(_bind(render_owner), qname, fmt)


@pytest.mark.parametrize("field,value", [
    ("qname", "other"), ("fmt", "FP8"),
    ("job_id", "foreign"), ("launch_sha256", "0" * 64),
    ("declaration_sha256", "0" * 64),
    ("batch_id", "foreign"), ("origin_root", "/outside"),
    ("launch_component", "foreign"), ("relative_final_path", "../outside"),
    ("final_path", "/outside"), ("temporary_path", "/outside.tmp"),
    ("payload_ceiling_bytes", 18), ("temp_ceiling_bytes", 18),
    ("producer_generation", "foreign"),
])
def test_forged_plan_refuses_without_sdk_admission(render_owner, field, value, monkeypatch):
    publication = _bind(render_owner)
    plan = _plan(publication)
    monkeypatch.setattr(publication, "require_prewrite", lambda **kwargs: pytest.fail("unexpected admission"))
    with pytest.raises(ValueError, match="plan|publication"):
        publication.require_render_prewrite(replace(plan, **{field: value}))


def test_nonplan_object_refuses(render_owner):
    with pytest.raises(ValueError, match="plan"):
        _bind(render_owner).require_render_prewrite(None)


@pytest.mark.parametrize("kind", ["final", "temporary", "symlink-parent"])
def test_known_occupied_or_symlink_destinations_refuse(render_owner, kind, monkeypatch):
    publication = _bind(render_owner)
    plan = _plan(publication)
    final = Path(plan.final_path)
    if kind == "symlink-parent":
        outside = render_owner.prefix.parent / "outside"
        outside.mkdir()
        (render_owner.prefix / "renders-v1").symlink_to(outside, target_is_directory=True)
    else:
        final.parent.mkdir(parents=True)
        Path(plan.final_path if kind == "final" else plan.temporary_path).write_bytes(b"existing")
    monkeypatch.setattr(publication, "require_prewrite", lambda **kwargs: pytest.fail("unexpected admission"))
    with pytest.raises(ValueError, match="occupied|symlink|contain"):
        publication.require_render_prewrite(plan)


def test_unstatable_destination_refuses_before_admission(render_owner, monkeypatch):
    publication = _bind(render_owner)
    plan = _plan(publication)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == plan.final_path:
            raise PermissionError("fixture cannot stat destination")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(publication, "contains", lambda path: True)
    monkeypatch.setattr(publication, "require_prewrite", lambda **kwargs: pytest.fail("unexpected admission"))
    with pytest.raises(ValueError, match="cannot stat render destination"):
        publication.require_render_prewrite(plan)


def test_present_payload_keeps_abort_accounting(render_owner):
    publication = _bind(render_owner)
    plan = _plan(publication)
    assert publication.require_render_prewrite(plan)["ok"] is True
    final = Path(plan.final_path)
    final.parent.mkdir(parents=True)
    final.write_bytes(b"fixture")
    refused = publication.abort_prewrite(batch_id=plan.batch_id)
    assert refused["ok"] is False
    assert refused["refusal"] == "owned paths remain"
    assert final.read_bytes() == b"fixture"
    final.unlink()
    assert publication.abort_prewrite(batch_id=plan.batch_id)['ok']
