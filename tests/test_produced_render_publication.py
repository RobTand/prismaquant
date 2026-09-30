"""Render binding reuses the qualified PB lifecycle, not a second adapter."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import re
from types import SimpleNamespace

import pytest

from test_stage_a_produced_boundary_chain import (
    TIER, _broker_control, _isolated_launch_context, _pb_source, _queue,
    _sealed_producer_request, _template,
)

pytestmark = pytest.mark.own_process


def _api():
    name = "prismaquant.produced_render_publication"
    assert importlib.util.find_spec(name) is not None, (
        "missing render publication adapter for the shared admitted-owner lifecycle")
    return importlib.import_module(name)


@pytest.fixture
def render_owner(tmp_path, request):
    _, pb_repo = _pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    from prismabuild import produced_output as po

    queue = _queue(tmp_path)
    prefix = tmp_path / "renders"
    prefix.mkdir()
    template = dict(_template(str(prefix)))
    template["template_id"] = "pq-rendered-weight-binding-fixture-v1"
    options = getattr(request, "param", {})
    template["slots"] = {options.get("slot", "renders"): {
        "class": options.get("artifact_class", "payload")}}
    if options.get("write_only"):
        template["write_only"] = True
        template["working_demands"][TIER] = {"minimum_gib": 0, "window_gib": 0}
    template = dict(po.validate_template(template))
    cas_root = tmp_path / "cas"
    owner = _sealed_producer_request(tmp_path, cas_root, pb_repo, template)
    queue.publish(
        action_key=owner, cas_root=str(cas_root),
        worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
        checkout_root=str(tmp_path / "mover-checkout"),
        resources={"cpu": 1, "mem_gb": 1, **po.owner_demand_terms(template)},
        produced_output_template=template)
    claim = queue.claim(owner="render-fixture")
    assert claim is not None and claim["action_key"] == owner
    control = _broker_control(queue, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"]}
    po.declare_template(queue.root, template)
    return SimpleNamespace(queue=queue, template=template, env=env,
                           cas_root=cas_root, prefix=prefix, po=po)


def _bind(owner):
    return _api().ProducedRenderPublication.bind_from_admitted_owner(
        queue_root=owner.queue.root, tier=TIER, env=owner.env)


def test_render_binding_uses_real_admitted_owner_and_declared_slot(render_owner):
    publication = _bind(render_owner)
    assert publication.slot == "renders"
    assert publication.template == render_owner.template
    assert publication.instance["owner_action_key"] == render_owner.env["PRISMABUILD_ACTION_KEY"]
    assert publication.cas_root == str(render_owner.cas_root)
    assert publication.admit_window()["ok"] is True


def test_render_binding_refuses_foreign_attempt(render_owner):
    from prismaquant.stage_a_produced_output import BoundaryProducedBindingError
    env = {**render_owner.env, "PRISMABUILD_ACTION_NONCE": "foreign-attempt"}
    with pytest.raises(BoundaryProducedBindingError, match="cannot bind"):
        _api().ProducedRenderPublication.bind_from_admitted_owner(
            queue_root=render_owner.queue.root, tier=TIER, env=env)


def test_batch_identity_is_stable_across_rebinding(render_owner):
    first = _bind(render_owner)
    second = _bind(render_owner)
    batch = first.render_batch_id_for("model.layers.0.mlp", "BF16")
    assert batch == second.render_batch_id_for("model.layers.0.mlp", "bf16")
    assert re.fullmatch(r"render-[0-9a-f]{64}", batch)
    assert first.render_batch_id_for("model.layers.0.mlp", "INT4_W4A16_g128") == (
        first.render_batch_id_for("model.layers.0.mlp", "int4_w4a16_g128"))


@pytest.mark.parametrize("alias,canonical", [
    (" BF16 ", "bf16"),
    ("FP8", "FP8_E4M3"),
    ("FP8_DYNAMIC", "fp8_e4m3"),
])
def test_batch_identity_uses_the_weight_cache_format_contract(render_owner, alias, canonical):
    publication = _bind(render_owner)
    assert publication.render_batch_id_for("layer", alias) == (
        publication.render_batch_id_for("layer", canonical))


@pytest.mark.parametrize("left,right", [
    ("layer/a", "layer?a"),
    ("layer_a", "layer.a"),
    ("x" * 180 + "a", "x" * 180 + "b"),
    ("λ", "μ"),
])
def test_batch_identity_keeps_complete_render_coordinate(render_owner, left, right):
    publication = _bind(render_owner)
    assert publication.render_batch_id_for(left, "BF16") != publication.render_batch_id_for(right, "BF16")
    assert publication.render_batch_id_for(left, "BF16") != publication.render_batch_id_for(left, "FP8")


@pytest.mark.parametrize("qname,fmt", [("", "BF16"), (None, "BF16"),
                                       ("layer", ""), ("layer", None),
                                       ("layer", "   ")])
def test_batch_identity_refuses_empty_or_untyped_coordinates(render_owner, qname, fmt):
    with pytest.raises(ValueError, match="nonempty strings"):
        _bind(render_owner).render_batch_id_for(qname, fmt)


@pytest.mark.parametrize("slot,artifact_class,write_only", [
    ("boundary_entries", "payload", False),
    ("renders", "checkpoint", False),
    ("renders", "payload", True),
])
def test_constructor_refuses_non_render_or_write_only_templates(
        render_owner, slot, artifact_class, write_only):
    from prismaquant.stage_a_produced_output import BoundaryProducedBindingError
    publication = _bind(render_owner)
    template = copy.deepcopy(render_owner.template)
    template["slots"] = {slot: {"class": artifact_class}}
    if write_only:
        template["write_only"] = True
        template["working_demands"][TIER] = {"minimum_gib": 0, "window_gib": 0}
    template = dict(render_owner.po.validate_template(template))
    with pytest.raises(BoundaryProducedBindingError, match="render"):
        _api().ProducedRenderPublication(
            queue=publication.queue, template=template,
            instance=publication.instance, tier=TIER,
            cas_root=publication.cas_root, env=render_owner.env)


@pytest.mark.parametrize("render_owner", [
    {"slot": "boundary_entries"},
    {"artifact_class": "checkpoint"},
    {"write_only": True},
], indirect=True)
def test_invalid_render_declaration_refuses_before_instance_admission(render_owner, monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryProducedBindingError
    calls = []
    original = render_owner.po.declare_instance

    def declare(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(render_owner.po, "declare_instance", declare)
    with pytest.raises(BoundaryProducedBindingError, match="render|slot"):
        _bind(render_owner)
    assert calls == []


def test_prewrite_and_authenticated_descriptor_use_existing_lifecycle(render_owner):
    publication = _bind(render_owner)
    batch = publication.render_batch_id_for("layer", "BF16")
    path = render_owner.prefix / "layer.pt"
    temporary = render_owner.prefix / "layer.pt.tmp"
    first = publication.require_prewrite(
        batch_id=batch, payload_ceiling_bytes=17, temp_ceiling_bytes=17,
        paths=[str(path), str(temporary)])
    second = publication.require_prewrite(
        batch_id=batch, payload_ceiling_bytes=17, temp_ceiling_bytes=17,
        paths=[str(path), str(temporary)])
    assert first["ok"] and second["ok"]
    assert not path.exists() and not temporary.exists()
    payload = b"render-fixture-v1"
    path.write_bytes(payload)  # The first byte follows the prewrite grant.
    reference = SimpleNamespace(path=str(path), file_bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest())
    descriptor = publication.descriptor_for(reference, producer_generation=batch)
    assert descriptor["slot"] == "renders"
    assert descriptor["sha256"] == reference.sha256
    assert descriptor["producer_generation"] == batch
    reference.sha256 = None
    with pytest.raises(render_owner.po.ProducedOutputError, match="sha256"):
        publication.descriptor_for(reference, producer_generation=batch)
    path.unlink()
    assert publication.abort_prewrite(batch_id=batch)["ok"] is True


def test_render_adapter_inherits_shared_publication_and_retirement():
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    cls = _api().ProducedRenderPublication
    assert "bind_from_admitted_owner" not in cls.__dict__
    for method in ("batch_id_for", "require_prewrite", "abort_prewrite",
                   "descriptor_for", "publish", "commit_origin",
                   "reader_context", "retire", "release"):
        assert getattr(cls, method) is getattr(BoundaryProducedPublication, method)
