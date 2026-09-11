"""The row head on threads: the same receipts, in the same order, less serial sha256.

The campaign identity hold, the capture seal, the H commitments and the
resumed-wire verify are the serial sha256 at the head of every campaign row.
Everything here checks one property: a head built on ``N`` threads stamps
byte for byte what the serial head stamped, and refuses where it refused.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
import torch

from test_tessera_hessian_reference_handoff import POLICY, handoff  # noqa: F401  (fixture)


def multi_unit_fixture():
    """Three priced units; two share one resident H object, as gate/up do."""
    from prismaquant import tessera_campaign as tc, tessera_hessian as th
    names = ["model.layers.0.a", "model.layers.0.b", "model.layers.1.c"]
    weights = {name: (torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
                      * (index + 1) / 512).to(torch.bfloat16)
               for index, name in enumerate(names)}
    shared = torch.eye(64) * 3
    hessians = {names[0]: shared, names[1]: shared, names[2]: torch.eye(64) * 5}
    source = th.activation_source(hessians, th.calibration_identity(
        "fixture", [torch.arange(16).reshape(1, 16)], fit_tokens=16))
    formats = [f"TESSERA_{family}_K1_R{rung}" for family in ("BF16", "E4M3")
               for rung in (896, 1024)]
    menus = {name: [SimpleNamespace(format_name=fmt) for fmt in formats] for name in names}
    kwargs = dict(weights=weights, menus=menus, calibration_source=source,
                  projected_units={}, static_scales={name: 0.125 for name in names})
    return tc, names, weights, hessians, source, kwargs


def run_level_identity(tc, names, weights, hessians, source, kwargs, bound):
    common = dict(weights=weights, acts={name: torch.ones(2, 2) for name in names},
                  hessians=hessians, menus=kwargs["menus"],
                  args=SimpleNamespace(family_restriction=None),
                  calibration_identity=source.provenance, serving_scope=None,
                  static_scales=kwargs["static_scales"], static_scale_policy="fixture",
                  expert_projection=None)
    return tc._campaign_checkpoint_identity(**common, bound_units=bound)


def anchor_templates(tc, names, weights, source, kwargs, bound):
    rosters = {name: tc._campaign_identity_anchor_roster(
        name, kwargs["menus"][name], calibration_source=source,
        static_scales=kwargs["static_scales"]) for name in names}
    return {(name, anchor.format_name): tc._checkpoint_anchor_identity(
        anchor, weights=weights, menus=kwargs["menus"], calibration_source=source,
        static_scales=kwargs["static_scales"], projected_units={}, bound_unit=bound[name])
        for name in names for anchor in rosters[name]}


def test_the_hold_on_threads_is_byte_identical_to_the_serial_hold(monkeypatch):
    from tessera import cached_unit
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    tc, names, weights, hessians, source, kwargs = multi_unit_fixture()
    # The owner is sealed before any builder starts, as the campaign does.
    source.capture_sha256()
    serial = tc._campaign_bound_identities(**kwargs, threads=1)
    expected_inputs = {name: unit.campaign_inputs() for name, unit in serial.items()}
    expected_templates = anchor_templates(tc, names, weights, source, kwargs, serial)
    expected_identity = run_level_identity(tc, names, weights, hessians, source, kwargs, serial)
    for unit in serial.values():
        unit.close()

    actual_hash = cached_unit.tensor_identity
    hashed, lock = [], threading.Lock()

    def observed(value):
        with lock:
            hashed.append((id(value), threading.current_thread().name))
        return actual_hash(value)

    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    threaded = tc._campaign_bound_identities(**kwargs, threads=3)
    # One weight receipt and one H receipt per unit, every one on a builder
    # thread; the sealed owner is not digested again by anyone.  (The
    # run-level identity below hashes the scoring rows too, so the hold's own
    # calls are read off before it runs.)
    held_hashes = list(hashed)
    assert sorted(value for value, _ in held_hashes) == sorted(
        [id(weights[name]) for name in names] + [id(hessians[name]) for name in names])
    assert {thread for _, thread in held_hashes} <= {f"campaign-identity_{i}" for i in range(3)}
    assert list(threaded) == sorted(names)
    assert {name: unit.campaign_inputs() for name, unit in threaded.items()} == expected_inputs
    assert anchor_templates(tc, names, weights, source, kwargs, threaded) == expected_templates
    actual_identity = run_level_identity(tc, names, weights, hessians, source, kwargs, threaded)
    assert canonical_json_sha256(actual_identity, where="threaded") == canonical_json_sha256(
        expected_identity, where="serial")
    for unit in threaded.values():
        unit.close()


def test_a_failed_threaded_hold_closes_every_holder_it_built(monkeypatch):
    tc, names, weights, hessians, source, kwargs = multi_unit_fixture()
    source.capture_sha256()
    weights[names[2]][0, 0] = float("nan")
    built = []
    real = tc.bind_checkpoint_unit_identity

    def recording(*args, **kw):
        unit = real(*args, **kw)
        built.append(unit)
        return unit

    monkeypatch.setattr(tc, "bind_checkpoint_unit_identity", recording)
    with pytest.raises(ValueError, match="nonfinite"):
        tc._campaign_bound_identities(**kwargs, threads=3)
    assert len(built) == 2 and all(unit._closed for unit in built)


def test_the_identity_plan_charges_one_more_roster_per_builder():
    tc, names, weights, hessians, source, kwargs = multi_unit_fixture()

    def plan(threads):
        return tc._campaign_identity_metadata_plan(
            weights=weights, menus=kwargs["menus"], calibration_source=source,
            projected_units={}, static_scales=kwargs["static_scales"], threads=threads)

    bounds_1, scratch_1 = plan(1)
    bounds_4, scratch_4 = plan(4)
    assert bounds_1 == bounds_4
    widest = max(len(menu) for menu in kwargs["menus"].values())
    roster = tc.IDENTITY_ROSTER_TRANSIENT_FORMAT_BYTES * widest + tc._frozenset_table_bytes(widest)
    assert scratch_4 - scratch_1 == 3 * roster
    with pytest.raises(ValueError, match="positive int"):
        plan(0)


def test_the_seal_taken_ahead_is_the_producer_seal_and_the_hold_does_not_retake_it(monkeypatch):
    from tessera import cached_unit
    tc, names, weights, hessians, source, kwargs = multi_unit_fixture()
    # The reference: the historical head, where the hold's first receipt
    # takes the seal.
    tc_, _, _, _, control, control_kwargs = multi_unit_fixture()
    control_kwargs = {**control_kwargs, "weights": weights,
                      "calibration_source": tc_.th.activation_source(hessians, control.provenance)}
    control_bound = tc._campaign_bound_identities(**control_kwargs, threads=1)
    expected = {name: unit.campaign_inputs() for name, unit in control_bound.items()}
    for unit in control_bound.values():
        unit.close()

    actual_hash = cached_unit.tensor_identity
    hashed, lock = [], threading.Lock()

    def observed(value):
        with lock:
            hashed.append((id(value), threading.current_thread().name))
        return actual_hash(value)

    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    ahead = tc._SealAhead(source)
    assert ahead.wait() >= 0.0 and ahead.seconds is not None
    # Exactly the producer's seal: every resident H digested once, on the
    # helper thread, and a sealed owner is not digested again.
    assert sorted(value for value, _ in hashed) == sorted(id(hessians[name]) for name in names)
    assert {thread for _, thread in hashed} == {"campaign-seal-ahead"}
    source.capture_sha256()
    assert len(hashed) == len(names)
    hashed.clear()
    bound = tc._campaign_bound_identities(**kwargs, threads=2)
    assert len(hashed) == 2 * len(names)
    assert {name: unit.campaign_inputs() for name, unit in bound.items()} == expected
    for unit in bound.values():
        unit.close()
    ahead.finish()


def test_the_seal_ahead_reraises_on_the_campaign_thread():
    from prismaquant import tessera_campaign as tc

    class Broken:
        def capture_sha256(self):
            raise RuntimeError("provenance moved")

    ahead = tc._SealAhead(Broken())
    with pytest.raises(RuntimeError, match="capture seal failed ahead") as info:
        ahead.wait()
    assert "provenance moved" in str(info.value.__cause__)
    ahead.finish()


def test_the_hold_hands_back_its_sealed_h_receipt_only_for_the_bound_tensor():
    from tessera.cached_unit import tensor_identity
    tc, names, weights, hessians, source, kwargs = multi_unit_fixture()
    source.capture_sha256()
    bound = tc._campaign_bound_identities(**kwargs, threads=1)
    identities = tc._bound_hessian_identities(bound, hessians)
    assert identities == {name: tensor_identity(hessians[name]) for name in names}
    assert identities[names[0]] == identities[names[1]]
    with pytest.raises(ValueError, match="no H receipt"):
        bound[names[0]].hessian_identity(hessians[names[2]])
    with pytest.raises(ValueError, match="no H receipt"):
        bound[names[0]].hessian_identity(hessians[names[0]].clone())
    # A unit whose H is not resident contributes nothing.
    assert names[2] not in tc._bound_hessian_identities(bound, {names[0]: hessians[names[0]]})
    for unit in bound.values():
        unit.close()
    with pytest.raises(ValueError, match="closed"):
        bound[names[0]].hessian_identity(hessians[names[0]])


def test_an_h_free_roster_hands_back_no_h_receipt(monkeypatch):
    from test_tessera_bound_identity import fixture
    from dataclasses import replace
    tc, name, weight, hessian, source, anchors, _, kwargs = fixture()
    monkeypatch.setattr(tc, "_checkpoint_anchor_identity", lambda *a, **k: {
        "source": {"sha256": "a" * 64}, "calibration": None, "recipe": {}})
    bound = tc.bind_checkpoint_unit_identity(
        [replace(anchor, hessian_applied=False) for anchor in anchors],
        source_weight=weight, calibration_source=source, projected_unit=None,
        static_scales=kwargs["static_scales"], retain_source_receipt=False)
    assert bound.hessian_identity(hessian) is None
    assert tc._bound_hessian_identities({name: bound}, {name: hessian}) == {}
    bound.close()


def test_the_reference_descriptor_takes_sealed_receipts_and_keeps_its_bytes(handoff, monkeypatch):
    import tessera.cached_unit
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant import tessera_campaign as campaign
    from tessera.cached_unit import tensor_identity
    f = handoff
    common = dict(counts=f["census"]["counts"],
                  provenance={**f["calibration"], "hessian_role": "fit"},
                  canonical_capture=f["record"], census_path=f["census_path"],
                  load_policy=POLICY)
    plain = cc.canonical_hessian_reference_descriptor(hessians=f["H"], **common)
    sealed = {"a": tensor_identity(f["H"]["a"])}
    digested = []
    real = tensor_identity
    monkeypatch.setattr(tessera.cached_unit, "tensor_identity",
                        lambda value: digested.append(id(value)) or real(value))
    given = cc.canonical_hessian_reference_descriptor(hessians=f["H"], identities=sealed, **common)
    assert given == plain
    assert json.dumps(given, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert digested == [id(f["H"]["b"])], "only the unit without a receipt is digested"
    with pytest.raises(RuntimeError, match="does not describe"):
        cc.canonical_hessian_reference_descriptor(
            hessians=f["H"], identities={"a": dict(sealed["a"], dtype="torch.bfloat16")}, **common)
    with pytest.raises(RuntimeError, match="does not describe"):
        cc.canonical_hessian_reference_descriptor(
            hessians=f["H"], identities={"a": dict(sealed["a"], shape=[2, 3])}, **common)
    with pytest.raises(RuntimeError, match="without resident H"):
        cc.canonical_hessian_reference_descriptor(
            hessians=f["H"], identities={"zz": sealed["a"]}, **common)
    monkeypatch.undo()

    # And through the campaign's writer: the same file, byte for byte.
    def write(name, **extra):
        root = f["tmp"] / name
        root.mkdir()
        return campaign.write_export_inputs(
            root, hessians=dict(f["H"]), hessian_rows=f["census"]["counts"],
            hessian_identity=f["calibration"], static_scales={}, static_scale_policy="fixture",
            hessian_reference=dict(canonical_capture=f["record"], census_path=f["census_path"],
                                   load_policy=POLICY), **extra)

    plain_path, _, plain_digest = write("plain")
    sealed_path, _, sealed_digest = write("sealed", hessian_identities={
        name: tensor_identity(value) for name, value in f["H"].items()})
    assert sealed_digest == plain_digest
    assert sealed_path.read_bytes() == plain_path.read_bytes()


def test_a_resume_with_the_same_identity_does_not_walk_it(tmp_path, monkeypatch):
    from prismaquant import production_weight_cache as pwc
    from prismaquant.cost_stage_checkpoint import prepare_journal
    identity = {"source": "one", "units": {
        f"u{index}": {"weight": {"sha256": "a" * 64}} for index in range(50)}}
    prepare_journal(tmp_path, stage="s", resume=False, identity=identity, qnames=["u0"])
    monkeypatch.setattr(pwc, "first_identity_difference",
                        lambda *a, **k: pytest.fail("walked an identical identity"))
    prepare_journal(tmp_path, stage="s", resume=True, identity=identity, qnames=["u0"])
    monkeypatch.undo()
    # A difference is still named by field, exactly as before.
    changed = json.loads(json.dumps(identity))
    changed["units"]["u7"]["weight"]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match=r"identity mismatch at units\.u7\.weight\.sha256"):
        prepare_journal(tmp_path, stage="s", resume=True, identity=changed, qnames=["u0"])
    # A manifest whose stored identity was edited under an unchanged digest
    # is still refused, by field: the digest alone does not admit it.
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["units"]["u3"]["weight"]["sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=r"identity mismatch at units\.u3\.weight\.sha256"):
        prepare_journal(tmp_path, stage="s", resume=True, identity=identity, qnames=["u0"])


def test_resumed_wire_receipts_verify_on_threads_in_adopt_order(monkeypatch):
    from prismaquant import tessera_campaign as tc
    pending = [(SimpleNamespace(qname=f"u{index}", format_name="F"), {"i": index},
                {"file": f"w{index}"}) for index in range(6)]

    def fake(anchor, wire_dir, identity, *, existing=None):
        if anchor.qname in {"u2", "u4"}:
            raise RuntimeError(f"refused {anchor.qname}")
        return dict(existing, verified=identity["i"], on=threading.current_thread().name)

    monkeypatch.setattr(tc, "_checkpoint_wire_record", fake)
    accepted = [row for row in pending if row[0].qname not in {"u2", "u4"}]
    records = tc._verify_wire_records_on_threads(accepted, "/wire", threads=3)
    assert [record["verified"] for record in records] == [0, 1, 3, 5]
    assert all(record["on"].startswith("campaign-wire-verify") for record in records)
    assert [record["verified"] for record in
            tc._verify_wire_records_on_threads(accepted, "/wire", threads=1)] == [0, 1, 3, 5]
    # The first refusal in adopt order is the one raised, on any thread count.
    for threads in (1, 3):
        with pytest.raises(RuntimeError, match="refused u2"):
            tc._verify_wire_records_on_threads(pending, "/wire", threads=threads)


def test_the_campaign_refuses_threads_without_a_hold_and_clamps_to_admitted_cpus(monkeypatch):
    import os
    from prismaquant import tessera_campaign as tc
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert tc._identity_threads_for_this_process(8) == 3
    assert tc._identity_threads_for_this_process(2) == 2
    with pytest.raises(ValueError, match="positive int"):
        tc._identity_threads_for_this_process(0)
    with pytest.raises(SystemExit):
        tc.main(["--model", "/nowhere", "--out", "/nowhere/out",
                 "--campaign-identity-threads", "4"])
