"""A microbatched Stage A plan seals the two planes its capture writes (PQ #1121).

The capture splits its calibration rows into ``probe_microbatch``-row
batches, the last one partial, and writes one entry per batch into each
probe's cotangent plane (``joint_cost_stage_a.run_adjoint_capture_core``:
``row_offsets``, the tail's and the roll's ``storage.write(...,
batch_index=...)``). The Stage A spool window is two such planes, each entry
reserved at the full batch's tensor bound plus the 64 KiB envelope. The
dispatcher seals it (``stage_a_spool_window_bytes``) and the bind checks it
(``StreamedBoundaryArtifacts._require_local_window``). Both count batches,
not rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

#: The writer's per-entry envelope over the tensor bytes.
ENVELOPE = 65536
#: GLM's text config, as R12's plan reads it.
GLM = {"hidden_size": 4096, "hc_mult": 4, "dtype": "bfloat16"}
_DTYPE_NAMES = {torch.float32: "float32", torch.bfloat16: "bfloat16",
                torch.float16: "float16"}


def _campaign(root: Path, config: dict, **execution) -> dict:
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps(config))
    plan = root / "plan.json"
    plan.write_text(json.dumps({"model": str(model), "execution": execution}))
    return {"plan_path": str(plan)}


def test_a_microbatched_plan_seals_two_planes_of_its_batches_not_its_rows(tmp_path):
    """R12's shape at probe_microbatch 4: a plane is 128 four-row entries.

    The rows (512) are not the entry count; sealing them would reserve four
    times the window, 256 GiB where 64.06 GiB is needed.
    """
    from dispatch_joint_quanta import stage_a_spool_window_bytes

    window = stage_a_spool_window_bytes(_campaign(
        tmp_path, {"text_config": GLM}, n_probes=4, n_calib_samples=512,
        calib_seqlen=512, probe_microbatch=4,
        boundary_storage={"prefetch_batches": 64}))
    entry = 4 * 512 * 4096 * 4 * 2 + ENVELOPE
    assert window == 2 * 4 * 128 * entry == 68_786_585_600


def test_a_partial_last_batch_is_one_more_entry_at_the_full_bound(tmp_path):
    """Ten rows at four per batch are three entries: 4, 4 and 2 rows.

    PrismaBuild reserves every entry of a group at the bound entry size
    (``boundary_group_ceiling_bytes``), so the partial batch is priced as a
    full one.
    """
    from dispatch_joint_quanta import stage_a_spool_window_bytes

    window = stage_a_spool_window_bytes(_campaign(
        tmp_path, {"hidden_size": 8, "dtype": "float32"}, n_probes=2,
        n_calib_samples=10, calib_seqlen=3, probe_microbatch=4,
        boundary_storage={"prefetch_batches": 2}))
    assert window == 2 * 2 * 3 * (4 * 3 * 8 * 4 + ENVELOPE)


# -- the dispatcher and the capture's own bind, on the real capture ---------


def _runner():
    from test_layer_major_boundary_capture import fixture

    _, _, runner, _ = fixture()
    runner.context.settle_prefetch_layers = lambda layers: None
    return runner


def _capture(root: Path, runner, *, n_probes, probe_microbatch,
             produced_output=None):
    from test_layer_major_boundary_capture import draw
    from test_streamed_boundary_artifacts import _policy
    from test_streamed_cost_checkpoints import _model_identity
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core

    return run_adjoint_capture_core(
        runner, draw(), execution={
            "n_probes": n_probes, "seed_base": 7000,
            "probe_microbatch": probe_microbatch,
            "boundary_storage": {
                **_policy(root / "b", cap=1 << 24, aux=1 << 22, disk=1 << 26),
                "schema": "prismaquant.aura.boundary_storage.v2",
                "capture_order": "layer_major"}},
        output_root=root / "out", stride=8,
        source_model_identity=_model_identity("joint-source"),
        unit_roster_sha256="a" * 64, plan_sha256="d" * 64,
        prepared_sha256="e" * 64, read_manifest_sha256="f" * 64,
        implementation_sha256="b" * 64, produced_output=produced_output)


def _written_planes(root: Path, monkeypatch, *, n_probes, probe_microbatch):
    """Every cotangent plane the capture writes: its entries and their bytes."""

    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    planes: dict = {}
    sizes: list = []
    real = StreamedBoundaryArtifacts.write

    def recording(self, tensor, **kwargs):
        if kwargs.get("probe_index") is not None:
            planes.setdefault((kwargs["probe_index"], kwargs["boundary_index"]),
                              set()).add(kwargs["batch_index"])
            sizes.append(int(tensor.numel()) * int(tensor.element_size()))
        return real(self, tensor, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "write", recording)
        _capture(root, _runner(), n_probes=n_probes,
                 probe_microbatch=probe_microbatch)
    return planes, sizes


class _Bound(Exception):
    """Stops the capture at its produced-output bind."""


def _capture_bind_arguments(root: Path, monkeypatch, *, n_probes,
                            probe_microbatch) -> dict:
    """The arguments the real capture passes to its produced-output bind."""

    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    seen: dict = {}

    def recording(self, publication, **kwargs):
        seen.update(kwargs)
        raise _Bound

    publication = SimpleNamespace(
        instance={"owner_action_key": "0" * 64}, output_prefix=str(root))
    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, "bind_produced_output",
                      recording)
        with pytest.raises(_Bound):
            _capture(root, _runner(), n_probes=n_probes,
                     probe_microbatch=probe_microbatch,
                     produced_output=publication)
    return seen


def _bind_need(root: Path, monkeypatch, arguments: dict, *, n_probes,
               sealed: int) -> int:
    """The window the real bind derives from ``arguments``, against ``sealed``.

    The local spool stands in with PrismaBuild's sealed bound ``sealed`` and
    a root on this test's disk; the bind refuses a need above it
    (``ProducedWindowRefused``).
    """

    from prismaquant import produced_output_spool
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.stage_a_produced_output import boundary_group_ceiling_bytes
    from test_streamed_boundary_artifacts import _policy

    spool = SimpleNamespace(max_bytes=int(sealed), root=root)
    publication = SimpleNamespace(
        write_only=False, output_prefix=str(root), contains=lambda path: True,
        group_ceiling_bytes=lambda *, entries, max_entry_tensor_bytes: (
            boundary_group_ceiling_bytes(
                group_size=entries,
                max_entry_tensor_bytes=max_entry_tensor_bytes)))
    storage = StreamedBoundaryArtifacts(_policy(root / "replay"))
    storage.bind({"fixture": "replay"}, n_probes=n_probes, published=True)
    with monkeypatch.context() as patch:
        patch.setattr(produced_output_spool.ProducedOutputSpool,
                      "from_publication",
                      staticmethod(lambda publication: spool))
        try:
            storage.bind_produced_output(publication, **arguments)
            return storage._produced_plan["local_window_bytes"]
        finally:
            storage._produced_stop_stager()


def test_the_dispatcher_and_the_capture_bind_agree_on_the_planes_it_writes(
        tmp_path, monkeypatch):
    """At probe_microbatch 4 the 5-row draw is two entries a plane: 4 and 1 rows.

    One run of the real capture counts the entries it writes; a second is
    stopped at its produced-output bind and the bind's own derivation is
    replayed with the arguments the capture passed. The dispatcher's seal,
    the bind's need and two planes of the written entries are one number.
    """

    from dispatch_joint_quanta import stage_a_spool_window_bytes

    n_probes, microbatch = 2, 4
    planes, sizes = _written_planes(tmp_path / "plain", monkeypatch,
                                    n_probes=n_probes,
                                    probe_microbatch=microbatch)
    counts = {len(entries) for entries in planes.values()}
    assert counts == {2}, planes
    two_planes = 2 * n_probes * 2 * (max(sizes) + ENVELOPE)

    runner = _runner()
    config = {"hidden_size": int(runner.base_model.embed_tokens.weight.shape[1]),
              "dtype": _DTYPE_NAMES[runner.dtype]}
    sealed = stage_a_spool_window_bytes(_campaign(
        tmp_path / "plan", config, n_probes=n_probes, n_calib_samples=5,
        calib_seqlen=4, probe_microbatch=microbatch,
        boundary_storage={"prefetch_batches": 2}))
    arguments = _capture_bind_arguments(
        tmp_path / "bound", monkeypatch, n_probes=n_probes,
        probe_microbatch=microbatch)
    need = _bind_need(tmp_path / "replay", monkeypatch, arguments,
                      n_probes=n_probes, sealed=sealed)
    assert (sealed, need) == (two_planes, two_planes), arguments
