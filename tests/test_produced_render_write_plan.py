"""Render destinations and byte reservations, without a tensor writer."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from test_produced_render_publication import _bind, render_owner, _isolated_launch_context

pytestmark = pytest.mark.own_process


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
    directory = render_owner.po.instance_dir(
        render_owner.queue.root, publication.instance) / "prewrites"
    before = sorted((p.name, p.read_bytes()) for p in directory.glob("*.prewrite.json"))
    assert len(before) == 1
    second = publication.require_render_prewrite(plan)
    assert first["ok"] and second["ok"]
    # Identical immutable records can succeed without a diagnostic duplicate
    # flag; observe the private fixture's reservation, not an invented API.
    assert {k: first[k] for k in ("batch_id", "class_bytes", "paths")} == {
        k: second[k] for k in ("batch_id", "class_bytes", "paths")}
    assert sorted((p.name, p.read_bytes()) for p in directory.glob("*.prewrite.json")) == before
    assert first["class_bytes"] == {"payload": 17, "checkpoint": 0, "temp": 17}
    assert set(first["paths"]) == {plan.final_path, plan.temporary_path}
    changed = _plan(publication, archive_max_bytes=19)
    with pytest.raises(BoundaryProducedPrewriteRefused, match="prewrite-conflict"):
        publication.require_render_prewrite(changed)
    assert publication.abort_prewrite(batch_id=plan.batch_id)["aborted"] is True
    assert list(directory.glob("*.prewrite.json")) == []
    assert publication.abort_prewrite(batch_id=plan.batch_id)["aborted"] is False
    assert publication.require_render_prewrite(changed)["ok"] is True
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
    assert plan.owner_nonce == render_owner.env["PRISMABUILD_ACTION_NONCE"]
    assert plan.owner_scope_id == render_owner.env["PRISMABUILD_ACTION_SCOPE"]
    assert plan.owner_action_key == publication.instance["owner_action_key"]
    assert plan.template_sha256 == publication.instance["template_sha256"]
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
    ("owner_action_key", "foreign"), ("owner_nonce", "foreign"),
    ("owner_scope_id", "foreign"), ("template_sha256", "0" * 64),
    ("batch_id", "foreign"), ("origin_root", "/outside"),
    ("attempt_component", "foreign"), ("relative_final_path", "../outside"),
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
    monkeypatch.setattr(publication, "require_prewrite", lambda **kwargs: pytest.fail("unexpected admission"))
    with pytest.raises(ValueError, match="cannot stat"):
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
    assert refused["refusal"] == "abort-files-present-retain"
    assert final.read_bytes() == b"fixture"
    final.unlink()
    assert publication.abort_prewrite(batch_id=plan.batch_id)["aborted"] is True
