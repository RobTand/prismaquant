"""A Stage B row's spill ceiling is the sealed need, not a spec literal.

PB action ``6f4f058751e6`` (layer 44 of GLM-5.3-Flash, regime (d), sparklina,
2026-09-24) refused 60 s into its head phase:

    Stage B spill needs 199051640832 bytes for this layer (196494753792 of
    payload and slot padding for 554880 parts at 4096-byte direct-I/O
    alignment) but its ceiling is 196494753792

The ceiling was the spec's hand-set ``PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES``
(183 GiB), and pbrun charged the same literal to the box's ``spool_gb``. The
payload alone reached it; the direct-I/O slot padding that #1060 added to the
file's reservation was never in it. The need is known before dispatch: it is
``spill_geometry``'s payload plus ``max_parts`` slots of padding on the
spill's direct-I/O grid, the arithmetic ``StageBSpillScratch`` reserves.

The record builder seals that need beside the read plan, and the dispatcher
sets the row's ceiling from it. PrismaBuild's ``spool_gb`` charge is that
ceiling rounded up to whole GiB, so the demand and the ceiling are one
number, and both cover the need.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import prismaquant.joint_replay_spill as spill_mod  # noqa: E402
from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant.joint_replay_spill import SpillTarget  # noqa: E402
from prismaquant.perturbed_x_cache import SpillGridRefused, StageBSpillScratch  # noqa: E402
from test_stageb_prepared_render_inputs import (  # noqa: E402
    _bind_prepared,
    _dispatch_prepared,
    _fixture_spill_bound,
    _spill_spec,
)

GIB = 1 << 30
SPILL_ENV = spill_mod.SPILL_ENV
#: The campaign spec's regime (R13 Stage B): four stored batches per pass.
REGIME = "capture_batch=4,accumulation=operator_gemm,chunk_rows=65536"

# The refused layer-44 quantum (PB 6f4f058751e6): 867 targets, 4 probes and
# 128 capture groups (512 samples of one row, captured four at a time).
PAYLOAD = 196_494_753_792
PARTS = (4 + 1) * 867 * 128
BLOCK = 4096
ADDRESS_ALIGNMENT = 512
# What the file reserves: the payload, then each part's slot padding (fewer
# than 512 bytes of residue ahead of it, and the tail to the next grid
# boundary), rounded up to the grid.
NEED = PAYLOAD + PARTS * (ADDRESS_ALIGNMENT + BLOCK)
NEED += -NEED % BLOCK

HIDDEN, EXPERT_INTER, SHARED_INTER, EXPERTS, TOP_K = 4096, 2048, 2048, 288, 8
LAYER = "model.language_model.layers.44.mlp"


def _layer44_targets():
    """GLM-5.3-Flash layer 44's 867 targets as the seal sees them."""
    experts = f"{LAYER}.experts"
    targets = {f"{LAYER}.shared_experts.{name}": SpillTarget(out, inp)
               for name, out, inp in (("gate_proj", SHARED_INTER, HIDDEN),
                                      ("up_proj", SHARED_INTER, HIDDEN),
                                      ("down_proj", HIDDEN, SHARED_INTER))}
    for expert in range(EXPERTS):
        for projection, parent, out, inp in (
                ("gate_proj", "gate_up_proj", EXPERT_INTER, HIDDEN),
                ("up_proj", "gate_up_proj", EXPERT_INTER, HIDDEN),
                ("down_proj", "down_proj", HIDDEN, EXPERT_INTER)):
            targets[f"{experts}.{expert}.{projection}"] = SpillTarget(
                out, inp, (experts, parent, projection, expert))
    return targets


def _layer44_geometry():
    """The refused quantum's geometry through the seal's own functions.

    The sealed record's 15 windows put five experts' gate and up in
    different windows (283 experts in one, 5 in two); two windows with that
    split give the same layer bound.
    """
    targets = _layer44_targets()
    split = {f"{LAYER}.experts.{expert}.up_proj" for expert in range(5)}
    first = tuple(name for name in targets
                  if ".experts." in name and not name.endswith("down_proj")
                  and name not in split)
    second = tuple(name for name in targets if name not in first)
    return spill_mod.spill_geometry(
        targets, [first, second], pending=set(targets),
        batch_tokens=spill_mod.spill_capture_batch_tokens(
            512, 512, probe_microbatch=1, capture_batch=4),
        n_probes=4, element_size=2, experts_per_token=TOP_K)


def _sealed_bound():
    """The executable readset's spill block for the refused shape."""
    return spill_mod.seal_spill_bound(
        _layer44_geometry(), block=BLOCK, capture_batch=4,
        element_dtype="bfloat16")


def _reseal(record: dict) -> dict:
    body = {key: value for key, value in record.items() if key != "identity_sha256"}
    record["identity_sha256"] = jl.canonical_sha256(
        body, where=f"quantum record {record['quantum_id']}")
    return record


def _dispatch_row(tmp_path, monkeypatch, *, bound, spec):
    """Seal ``bound`` into a spill row, dispatch it under ``spec``; return argv."""
    (dispatch, sealed, _receipt, _manifest, records, _receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch, replay_mode="spill")
    dispatch.SPEC_PATH.write_text(json.dumps(spec))
    record = json.loads(json.dumps(sealed))
    if bound is None:
        record["executable_readset"].pop("spill_bound")
    else:
        record["executable_readset"]["spill_bound"] = bound
    path = records / "layer-002.json"
    path.write_text(json.dumps(_reseal(record)))
    return dispatch, dispatch.quantum_argv(record, record_path=path, output_root=out)


def _row_ceiling(tmp_path, monkeypatch):
    """Dispatch the spill row under the 183 GiB spec; return the sealed ceilings."""
    _dispatch, argv = _dispatch_row(tmp_path, monkeypatch, bound=_sealed_bound(),
                                    spec=_spill_spec(183 * GIB, regime=REGIME))
    outer = argv[:argv.index("--")]
    sealed = dict(outer[i + 1].split("=", 1) for i, word in enumerate(outer[:-1])
                  if word == "--env" and "=" in outer[i + 1])
    inner = json.loads(argv[argv.index("--spec") + 1])["env"]
    return int(sealed[SPILL_ENV[1]]), int(inner[SPILL_ENV[1]])


def test_the_refused_shape_needed_more_than_its_ceiling():
    """The arithmetic of the refusal message, reproduced."""
    assert PARTS == 554_880
    assert NEED == 199_051_640_832
    assert PAYLOAD == 183 * GIB < NEED


def test_the_seal_reproduces_the_refused_layer_and_its_need():
    geometry = _layer44_geometry()
    assert geometry.total_bytes == PAYLOAD and geometry.max_parts == PARTS
    # The one sizing rule, at the seal and in the scratch.
    assert spill_mod.spill_reservation_bytes(PAYLOAD, PARTS, block=BLOCK) == NEED
    assert StageBSpillScratch.reservation_bytes(
        PAYLOAD, parts=PARTS, part_padding=ADDRESS_ALIGNMENT, block=BLOCK) == NEED
    bound = _sealed_bound()
    assert bound["reservation_bytes"] == NEED
    assert spill_mod.check_spill_bound(json.loads(json.dumps(bound))) == NEED


def test_the_row_ceiling_and_pb_demand_cover_the_sealed_need(tmp_path, monkeypatch):
    outer, inner = _row_ceiling(tmp_path, monkeypatch)
    # The container reads the same ceiling pbrun charges.
    assert inner == outer
    # PB #911 charges the pair's ceiling rounded up to whole GiB.
    demand_gib = math.ceil(outer / GIB)
    assert outer >= NEED, (
        f"the row seals a {outer}-byte spill ceiling for a {NEED}-byte need")
    assert demand_gib * GIB >= NEED
    # The ceiling is the sealed need itself, not a padded literal.
    assert outer == NEED and demand_gib == 186


def test_the_scratch_reserves_the_one_rule_and_fits_a_sealed_ceiling(tmp_path):
    """Mutate the driver: the live file takes exactly the sealed reservation."""
    from test_stageb_one_pass_spill import _spill_root

    root = _spill_root(tmp_path)
    for parts in (0, 3, 7):
        probe = StageBSpillScratch(directory=root, max_bytes=1 << 30, nbytes=3000,
                                   parts=parts, part_padding=ADDRESS_ALIGNMENT,
                                   alignment=ADDRESS_ALIGNMENT)
        try:
            block = probe.block
            need = StageBSpillScratch.reservation_bytes(
                3000, parts=parts, part_padding=ADDRESS_ALIGNMENT, block=block)
            assert probe.capacity == need
        finally:
            probe.close()
        if block <= BLOCK:
            # A ceiling sealed on the 4 KiB grid covers a live grid no coarser.
            sealed = spill_mod.spill_reservation_bytes(3000, parts, block=BLOCK)
            assert sealed >= need
        exact = StageBSpillScratch(directory=root, max_bytes=need, nbytes=3000,
                                   parts=parts, part_padding=ADDRESS_ALIGNMENT,
                                   alignment=ADDRESS_ALIGNMENT, max_block=block)
        exact.close()
        with pytest.raises(RuntimeError, match=f"needs {need} bytes"):
            StageBSpillScratch(directory=root, max_bytes=need - 1, nbytes=3000,
                               parts=parts, part_padding=ADDRESS_ALIGNMENT,
                               alignment=ADDRESS_ALIGNMENT)
    # A live grid coarser than the sealed one refuses before the file exists.
    if block > ADDRESS_ALIGNMENT:
        with pytest.raises(SpillGridRefused, match=f"coarser than the {block // 2}-byte"):
            StageBSpillScratch(directory=root, max_bytes=1 << 30, nbytes=3000,
                               parts=3, part_padding=ADDRESS_ALIGNMENT,
                               alignment=ADDRESS_ALIGNMENT, max_block=block // 2)


def test_the_offline_seal_equals_the_live_geometry_on_packed_experts():
    """``sealed_spill_targets`` from shapes and the profile, against live modules."""
    import torch

    import test_stageb_one_pass_spill as fixture
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    torch.manual_seed(1)
    model = fixture._MoELM()
    profile = Lfm2MoeProfile()
    live = fixture._targets(model, profile)
    layer = sorted(name for name in live if ".layers.0." in name)
    shapes = {name: tuple(int(size) for size in live[name].weight.shape)
              for name in layer}
    sealed = spill_mod.sealed_spill_targets(shapes, profile)
    assert sealed == {name: spill_mod.spill_target(live[name]) for name in layer}
    assert sum(target.packed is not None for target in sealed.values()) == \
        3 * fixture.EXPERTS
    packed = [name for name in layer if sealed[name].packed is not None]
    windows = [tuple(name for name in layer if name not in packed[:4]),
               tuple(packed[:4])]
    for arrangement in ([tuple(layer)], windows):
        kwargs = dict(pending=set(layer), batch_tokens=[8, 8], n_probes=2,
                      element_size=2, experts_per_token=fixture.TOP_K)
        assert (spill_mod.spill_geometry(sealed, arrangement, **kwargs)
                == spill_mod.spill_geometry(live, arrangement, **kwargs))


def test_capture_batch_tokens_group_as_the_quantum_does():
    assert spill_mod.spill_capture_batch_tokens(
        512, 512, probe_microbatch=1, capture_batch=4) == [4 * 512] * 128
    # All rows in one batch when the microbatch is 0; a ragged last group.
    assert spill_mod.spill_capture_batch_tokens(
        6, 3, probe_microbatch=0, capture_batch=2) == [18]
    assert spill_mod.spill_capture_batch_tokens(
        5, 3, probe_microbatch=2, capture_batch=2) == [12, 3]


def test_a_sealed_bound_refuses_edits_and_foreign_launches():
    geometry = _layer44_geometry()
    bound = _sealed_bound()
    edited = json.loads(json.dumps(bound))
    edited["reservation_bytes"] = 199 * GIB
    with pytest.raises(spill_mod.SpillBoundRefused, match="needs 199051640832 bytes"):
        spill_mod.check_spill_bound(edited)
    extra = {**bound, "margin": 1}
    with pytest.raises(spill_mod.SpillBoundRefused, match="carries exactly"):
        spill_mod.check_spill_bound(extra)
    with pytest.raises(spill_mod.SpillBoundRefused, match="power of two"):
        spill_mod.check_spill_bound({**bound, "block": 256})
    live = dict(capture_batch=4, element_dtype="bfloat16", ceiling=NEED)
    assert spill_mod.require_sealed_spill_bound(bound, geometry, **live) == BLOCK
    other = spill_mod.spill_geometry(
        _layer44_targets(), [tuple(_layer44_targets())],
        pending=set(_layer44_targets()),
        batch_tokens=spill_mod.spill_capture_batch_tokens(
            512, 512, probe_microbatch=1, capture_batch=4),
        n_probes=4, element_size=2, experts_per_token=TOP_K)
    with pytest.raises(spill_mod.SpillBoundRefused, match="seals spill geometry"):
        spill_mod.require_sealed_spill_bound(bound, other, **live)
    with pytest.raises(spill_mod.SpillBoundRefused, match="seals spill capture_batch"):
        spill_mod.require_sealed_spill_bound(bound, geometry, **{**live, "capture_batch": 1})
    with pytest.raises(spill_mod.SpillBoundRefused, match="spill ceiling is"):
        spill_mod.require_sealed_spill_bound(bound, geometry, **{**live, "ceiling": 183 * GIB})



def test_dev_mode_stamps_the_spill_identity_and_keeps_the_capacity_bound(
        monkeypatch, capsys):
    """PQ #1147: the sealed spill identity stamps; the ceiling still bounds."""
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    geometry = _layer44_geometry()
    bound = _sealed_bound()
    live = dict(capture_batch=4, element_dtype="bfloat16", ceiling=NEED)
    capsys.readouterr()
    assert spill_mod.require_sealed_spill_bound(
        bound, geometry, **{**live, "capture_batch": 1}) == BLOCK
    assert "[DEV-MODE] seal spill capture_batch differs" in capsys.readouterr().out
    # A ceiling above the sealed reservation is room, not a refusal.
    assert spill_mod.require_sealed_spill_bound(
        bound, geometry, **{**live, "ceiling": NEED + GIB}) == BLOCK
    assert "[DEV-MODE] seal spill ceiling differs" in capsys.readouterr().out
    # The live geometry's scratch must still fit the admitted ceiling.
    with pytest.raises(spill_mod.SpillBoundRefused, match="over the admitted ceiling"):
        spill_mod.require_sealed_spill_bound(bound, geometry, **{**live, "ceiling": 183 * GIB})

def test_the_binder_seals_a_bound_only_on_a_spill_readset(tmp_path):
    bound = _fixture_spill_bound()
    sealed, *_ = _bind_prepared(tmp_path, str(tmp_path / "run"), replay_mode="spill",
                                spill_bound=bound)
    assert sealed["executable_readset"]["spill_bound"] == bound
    (tmp_path / "w").mkdir()
    with pytest.raises(ValueError, match="only a spill-mode readset"):
        _bind_prepared(tmp_path / "w", str(tmp_path / "w" / "run"), spill_bound=bound)
    (tmp_path / "e").mkdir()
    edited = {**bound, "reservation_bytes": bound["reservation_bytes"] + 4096}
    with pytest.raises(ValueError, match="reservation"):
        _bind_prepared(tmp_path / "e", str(tmp_path / "e" / "run"),
                       replay_mode="spill", spill_bound=edited)


def test_the_dispatcher_refuses_a_spill_row_without_its_bound(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    with pytest.raises(dispatch.DispatchRefused, match="seals no spill bound"):
        _dispatch_row(tmp_path, monkeypatch, bound=None,
                      spec=_spill_spec(183 * GIB, regime=REGIME))


def test_the_dispatcher_refuses_another_capture_batch(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    with pytest.raises(dispatch.DispatchRefused, match="capture batch 1"):
        _dispatch_row(tmp_path, monkeypatch, bound=_sealed_bound(),
                      spec=_spill_spec(183 * GIB))


def test_the_dispatcher_refuses_a_bound_without_a_declared_spill(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    spec = {"container": {"image": "sha256:" + "0" * 64},
            "env": {"PRISMAQUANT_STAGE_B_REPLAY_REGIME": REGIME}}
    with pytest.raises(dispatch.DispatchRefused):
        _dispatch_row(tmp_path, monkeypatch, bound=_sealed_bound(), spec=spec)
