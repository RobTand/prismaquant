"""``--publication-overlap-bytes`` through the real CLI, not through its parts.

The standalone publisher and ledger tests establish ordering on fakes. They
cannot establish that the campaign passes a publisher to anything, and the
first version of this branch did not: the flag built a writer, the ledger
staged nothing, and every measurement still ran synchronously. So these drive
``tessera_campaign.main`` end to end, on both the scalar and the batched anchor
path, and compare the artifacts against the same run with the flag unset.

Nothing here is a speed test. The claim is that the same bytes and the same
prices come out, that the receipt and journal order holds, and that a writer
failure stops the action instead of being absorbed by it.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

UNITS = ("model.layers.0.proj", "model.layers.1.proj")
FAMILY = "TESSERA_E4M3_K1"
FMT = f"{FAMILY}_R1024"


def _fixture(monkeypatch, tmp_path):
    """Two priceable units, one round, no GPU and no real calibration."""
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles import DefaultProfile

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(
        [torch.nn.Module(), torch.nn.Module()])
    generator = torch.Generator().manual_seed(186)
    for index in range(2):
        linear = torch.nn.Linear(256, 32, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(32, 256, generator=generator))
        model.model.layers[index].proj = linear
    rows = torch.randn(4, 256, generator=torch.Generator().manual_seed(183))

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: DefaultProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {
        "accepted": True, "reason": "CPU test fixture", "kwargs": [], "recipe": {},
    })
    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        lambda *_args: ([torch.ones(1, 256, dtype=torch.long)], "one draw"))
    monkeypatch.setattr(tessera_campaign, "_collect_activations", lambda *_args, **kwargs: (
        {name: rows for name in UNITS}, {}, {name: 0 for name in UNITS},
        {name: 3.0 for name in UNITS}))
    monkeypatch.setattr(tessera_campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: {
                            name: [SimpleNamespace(
                                format_name=FMT, family=FAMILY,
                                body_rate_q256=1024, bpp=4.0)] for name in targets})
    return tessera_campaign, tessera_render


def _argv(tmp_path):
    return ["--model", "synthetic-current-model",
            "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"),
            "--checkpoint", str(tmp_path / "campaign.anchors.json"),
            "--hessian", "off", "--menu-mode", "research", "--max-rounds", "1"]


def _artifacts(tmp_path):
    """Every file the campaign published, by name, with its bytes."""
    root = tmp_path / "cache"
    return {path.name: path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".pt", ".tessera"}}


def _payload(tmp_path):
    with (tmp_path / "cost.pkl").open("rb") as handle:
        return pickle.load(handle)


def _batched(monkeypatch, campaign, render, seen):
    """Force the batched adapter over two dense units.

    Which units may share a batch is `_anchor_batches`' decision and
    `tests/test_tessera_campaign_batch.py` owns it. This test is about what
    the batched path does with a publisher, so the grouping is supplied.
    """
    monkeypatch.setattr(render, "require_tessera_batch_encoder", lambda: None)
    monkeypatch.setattr(campaign, "_anchor_batches",
                        lambda pending, **_kwargs: [list(pending)])

    def encode(weights, format_name, *, recipe, activation_kwargs, hessian_required):
        seen.append(len(weights))
        return [campaign._encode_and_render(
            weight, format_name, recipe=recipe, activation_kwargs=kwargs,
            hessian_required=hessian_required)
            for weight, kwargs in zip(weights, activation_kwargs)]

    monkeypatch.setattr(render, "encode_tessera_units", encode)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_overlap_publishes_the_same_bytes_and_prices_as_the_default_path(
        monkeypatch, tmp_path, batch_size):
    from prismaquant.tessera_publication import SCHEMA

    seen: list[int] = []
    plain = tmp_path / "plain"
    campaign, render = _fixture(monkeypatch, plain)
    extra = []
    if batch_size > 1:
        _batched(monkeypatch, campaign, render, seen)
        extra = ["--anchor-batch-size", "2"]
    assert campaign.main([*_argv(plain), *extra]) == 0
    assert _payload(plain)["provenance"]["publication_overlap"] is None, (
        "the default path must not report a publisher it did not build")

    staged = tmp_path / "staged"
    campaign, render = _fixture(monkeypatch, staged)
    if batch_size > 1:
        _batched(monkeypatch, campaign, render, seen)
    assert campaign.main([
        *_argv(staged), *extra, "--publication-overlap-bytes", str(1 << 20)]) == 0

    if batch_size > 1:
        assert seen and max(seen) == 2, (
            f"the batched adapter never saw a real batch: {seen}")
    stats = _payload(staged)["provenance"]["publication_overlap"]
    assert stats["schema"] == SCHEMA
    assert stats["budget_bytes"] == 1 << 20
    assert stats["failed"] is False
    # Two file jobs, two receipts, and at least one journal write.
    assert stats["published"] >= 5, stats
    assert stats["peak_charged_bytes"] > 0, (
        "nothing was ever staged, so nothing was ever overlapped")

    assert _artifacts(plain) == _artifacts(staged), (
        "the staged path published different bytes")
    for unit in UNITS:
        for path in (plain, staged):
            assert _payload(path)["costs"][unit][FMT]["output_mse_measured"]
        # Everything but the clock. ``encode_seconds`` is what the encode took
        # on the box, and the first arm in a process pays the runtime's
        # one-time first-use cost; it is not a price and this is not a
        # measurement of one.
        rows = [{k: v for k, v in _payload(path)["costs"][unit][FMT].items()
                 if k != "encode_seconds"} for path in (plain, staged)]
        assert rows[0] == rows[1]


def test_the_checkpoint_carries_every_unit_with_its_wire_receipt(
        monkeypatch, tmp_path):
    campaign, _render = _fixture(monkeypatch, tmp_path)
    assert campaign.main([
        *_argv(tmp_path), "--publication-overlap-bytes", str(1 << 20)]) == 0
    from prismaquant.cost_stage_checkpoint import unit_path

    root = tmp_path / "campaign.anchors.json.parts"
    for unit in UNITS:
        envelope = pickle.loads(unit_path(root, unit).read_bytes())
        state = pickle.loads(envelope["payload"])
        assert state["wire_records"], f"{unit} was journalled with no receipt"
        assert {row["format_name"] for row in state["anchors"]} == set(
            state["wire_records"]), (
            f"{unit} has an anchor row whose bytes nothing witnessed")
        for record in state["wire_records"].values():
            wire = tmp_path / "cache" / "wire" / record["file"]
            assert wire.exists(), (
                f"{unit} carries a receipt for a file that does not exist")
            assert wire.stat().st_size == record["blob_bytes"]


def test_a_publication_failure_stops_the_action_instead_of_being_absorbed(
        monkeypatch, tmp_path):
    from prismaquant.cost_stage_checkpoint import unit_path
    from prismaquant.tessera_publication import PublicationError

    campaign, _render = _fixture(monkeypatch, tmp_path)
    real_save = torch.save
    saved: list[str] = []

    def failing(obj, path, *args, **kwargs):
        saved.append(Path(path).name)
        if len(saved) == 2:
            raise OSError("no space left on device")
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", failing)
    with pytest.raises(PublicationError):
        campaign.main([*_argv(tmp_path), "--publication-overlap-bytes", str(1 << 20)])
    # No cost table, and no journal row for the unit whose render never
    # landed. The failing save is the second one, and the second save is
    # unit one's, because the writer publishes in submission order and
    # submission order is the anchor order: so unit one has a row and unit
    # two must not.
    assert not (tmp_path / "cost.pkl").exists()
    root = tmp_path / "campaign.anchors.json.parts"
    assert not unit_path(root, UNITS[1]).exists(), (
        "a unit whose bytes failed to land was journalled anyway")
    assert unit_path(root, UNITS[0]).exists(), (
        "the unit that did land lost its row when the action unwound")


@pytest.mark.parametrize("apply_timing", ["eager", "deferred"])
def test_a_fatal_encode_error_still_journals_the_batch_that_succeeded(
        monkeypatch, tmp_path, apply_timing):
    """The regression the synchronous per-batch checkpoint did not have.

    With a writer behind, unit one's row is not journalled at the moment its
    encode returns; it is journalled when its bytes land.  If unit two then
    fails fatally, unwinding must still commit unit one.  Losing it would mean
    the resume re-encodes work the run really finished.

    Which of the two unwind paths runs depends on a RACE the test must not
    inherit: whether the writer had finished unit one by the time the next
    batch called ``apply_completed``.

    * ``deferred`` -- it had not, so the anchor is still staged and
      ``ledger.close`` is what journals it.
    * ``eager`` -- it had, so the row is already in the pending checkpoint
      and ``close`` finds nothing; only an unconditional flush writes it.

    On an unloaded box the second is the likely one, which is why the flush
    in the ``finally`` may not be made conditional on ``close`` having
    journalled something.  Both are forced here rather than sampled.
    """
    from prismaquant.cost_stage_checkpoint import unit_path
    from prismaquant.tessera_render import HessianContractError

    campaign, _render = _fixture(monkeypatch, tmp_path)
    ledger_cls = campaign._AnchorPublicationLedger
    original_apply = ledger_cls.apply_completed
    if apply_timing == "eager":
        def apply_completed(self):
            # Wait the writer out, so every finished receipt is applied and
            # sitting unflushed in the pending checkpoint when the encode
            # below raises.
            if self._publisher is not None:
                while self._publisher.outstanding:
                    time.sleep(0.005)
            return original_apply(self)
    else:
        def apply_completed(self):
            # Never applied, so everything is still staged at close.
            return 0
    monkeypatch.setattr(ledger_cls, "apply_completed", apply_completed)
    real_encode = campaign._encode_and_render
    calls = []

    def encode(weight, format_name, **kwargs):
        calls.append(format_name)
        if len(calls) == 2:
            raise HessianContractError("no Hessian for this Linear")
        return real_encode(weight, format_name, **kwargs)

    monkeypatch.setattr(campaign, "_encode_and_render", encode)
    with pytest.raises(HessianContractError, match="no Hessian"):
        campaign.main([*_argv(tmp_path), "--publication-overlap-bytes", str(1 << 20)])

    root = tmp_path / "campaign.anchors.json.parts"
    survived = [unit for unit in UNITS if unit_path(root, unit).exists()]
    assert survived, (
        "the batch that succeeded before the fatal one lost its journal row")
    for unit in survived:
        envelope = pickle.loads(unit_path(root, unit).read_bytes())
        state = pickle.loads(envelope["payload"])
        assert state["anchors"] and state["wire_records"]
        for record in state["wire_records"].values():
            wire = tmp_path / "cache" / "wire" / record["file"]
            assert wire.stat().st_size == record["blob_bytes"], (
                "a row was committed against bytes that are not there")


def test_the_post_work_of_one_batch_runs_while_the_next_batch_encodes(
        monkeypatch, tmp_path):
    """The overlap itself, held open by a barrier rather than sampled.

    Batch one's receipt is made on the writer by reading its published wire
    back.  Here that read-back is held until batch two's encode has BEGUN on
    the encode thread; the run can only complete if the two really overlap.
    On the synchronous path the same hold sits on the encode thread before
    batch two, so the barrier times out and the assertion fails rather than
    the shard hanging.
    """
    import threading

    campaign, _render = _fixture(monkeypatch, tmp_path)
    next_encode_started = threading.Event()
    calls: list[str] = []
    real_encode = campaign._encode_and_render

    def encode(weight, format_name, **kwargs):
        calls.append(format_name)
        if len(calls) == 2:
            next_encode_started.set()
        return real_encode(weight, format_name, **kwargs)

    monkeypatch.setattr(campaign, "_encode_and_render", encode)
    held: list[dict] = []
    real_record = campaign._checkpoint_wire_record

    def record(anchor, wire_dir, identity, **kwargs):
        if not held:
            held.append(dict(thread=threading.current_thread().name,
                             next_encode_started=next_encode_started.wait(20.0)))
        return real_record(anchor, wire_dir, identity, **kwargs)

    monkeypatch.setattr(campaign, "_checkpoint_wire_record", record)
    assert campaign.main([
        *_argv(tmp_path), "--publication-overlap-bytes", str(1 << 20)]) == 0
    assert held == [dict(thread="tessera-publication", next_encode_started=True)]
    from prismaquant.cost_stage_checkpoint import unit_path

    root = tmp_path / "campaign.anchors.json.parts"
    for unit in UNITS:
        state = pickle.loads(pickle.loads(unit_path(root, unit).read_bytes())["payload"])
        assert state["anchors"] and state["wire_records"]


def test_the_pipelined_selected_path_prices_the_same_bytes_and_identities(
        monkeypatch, tmp_path):
    """Overlap plus a bound identity, on the selected-source path, vs neither.

    This is the configuration the pipelined arm runs.  The bound identity is
    derived on the writer (the thread name is asserted), and everything a
    resume or an export reads -- the journal identity, each unit's envelope
    digest, its anchor rows, its wire receipt and the wire bytes -- is the
    synchronous path's, byte for byte.
    """
    import json
    import threading
    from test_selected_source_authentication import selected_source_fixture
    from prismaquant.cost_stage_checkpoint import unit_path

    from prismaquant import tessera_campaign

    UNIT = "model.layers.0.proj"
    runs: dict[str, dict] = {}
    derived_on: list[tuple[str, str]] = []
    current = ["none"]
    real_derive = tessera_campaign._BoundCheckpointUnitIdentity.derive

    def derive(self, **kwargs):
        derived_on.append((current[0], threading.current_thread().name))
        return real_derive(self, **kwargs)

    monkeypatch.setattr(tessera_campaign._BoundCheckpointUnitIdentity, "derive", derive)
    # One source, one census, one capture: the journal identity binds the
    # model path by value, so the two arms price the same selection out of
    # the same tree and differ only in where their outputs land.
    campaign, argv, _state = selected_source_fixture(monkeypatch, tmp_path, priced=True)
    for label, extra in (("sync", []), ("pipelined", [
            "--publication-overlap-bytes", str(1 << 20),
            "--campaign-identity-bytes", str(1 << 20)])):
        root = tmp_path / label
        root.mkdir()
        arm_argv = list(argv)
        for flag, leaf in (("--out", "cost.pkl"), ("--cache-dir", "cache"),
                           ("--checkpoint", "campaign.anchors.json")):
            arm_argv[arm_argv.index(flag) + 1] = str(root / leaf)
        current[0] = label
        assert campaign.main([*arm_argv, *extra]) == 0
        manifest = json.loads((root / "campaign.anchors.json").read_text())
        envelope = pickle.loads(unit_path(root / "campaign.anchors.json.parts", UNIT).read_bytes())
        state = pickle.loads(envelope["payload"])
        assert state["anchors"] and state["wire_records"]
        wires = {name: (root / "cache" / "wire" / name).read_bytes()
                 for name in (record["file"] for record in state["wire_records"].values())}
        costs = _payload(root)["costs"][UNIT]
        # The envelope's payload digest binds the anchor rows' wall-clock
        # ``seconds``, so two runs never share it; everything else in the
        # unit state is compared by value below.
        runs[label] = dict(
            identity_sha256=manifest["identity_sha256"],
            envelope_identity=envelope["identity_sha256"],
            anchors=[{k: v for k, v in row.items() if k not in ("seconds", "encoding_batch_size")}
                     for row in state["anchors"]],
            state={k: v for k, v in state.items() if k != "anchors"},
            wire_records=state["wire_records"], wires=wires,
            costs={fmt: {k: v for k, v in row.items() if k != "encode_seconds"}
                   for fmt, row in costs.items()},
            provenance=_payload(root)["provenance"])
    assert derived_on == [("pipelined", "tessera-publication")], derived_on
    assert runs["sync"]["provenance"]["publication_overlap"] is None
    assert runs["pipelined"]["provenance"]["publication_overlap"]["failed"] is False
    for key in ("identity_sha256", "envelope_identity", "anchors", "state",
                "wire_records", "wires", "costs"):
        assert runs["sync"][key] == runs["pipelined"][key], key
