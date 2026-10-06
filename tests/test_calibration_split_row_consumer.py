"""CPU behavior cases for the default-off uncapped ``row_consumer`` seam and
the disjoint FIT/HELDOUT moment accumulator (eng-ldlq-split-collector).

Everything runs on the real tiny routed fixture (``_RoutedModel``: a mixed
dense+MoE model whose packed experts are captured through the existing
``derive_per_expert_activations`` path).  No second model forward is ever
performed: both split moments ride the single capture pass, with
``max_rows=0`` and ``want_hessian=False`` as the research caller drives them.
"""
import pytest
import torch

from experiments.indomain_split_stats import (
    FIT, HELDOUT, DisjointRowMoments,
)
from prismaquant.tessera_campaign import _collect_activations
from test_tessera_campaign_packed import (
    EXPERT_PREFIX, _RoutedModel, _routed_tokens,
)

DENSE = "model.layers.0.attention"


def packed_targets():
    return [f"{EXPERT_PREFIX}.{expert}.{projection}"
            for expert in range(2) for projection in ("w1", "w3", "w2")]


class _RecordingMoments(DisjointRowMoments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def consume(self, names, flat):
        self.calls.append((names, flat.detach().clone()))
        super().consume(names, flat)

    __call__ = consume


def _one_sample_capture(moments, tokens_batches, *, max_rows=0,
                        want_hessian=False, shared=False):
    model = _RoutedModel()
    index = {"current": 0}

    def forward_batch(batch):
        # The parent's wrapper: each forward is exactly one sample.
        moments.set_sample(index["current"])
        index["current"] += 1
        model(batch)

    try:
        result = _collect_activations(
            model, [DENSE, *packed_targets()], tokens_batches, max_rows, "cpu",
            want_hessian=want_hessian, shared_packed_inputs=shared,
            forward_batch=forward_batch, row_consumer=moments)
    except BaseException:
        moments.close()
        raise
    return model, result


def test_callback_sees_all_rows_with_zero_cap_and_no_hessian():
    moments = _RecordingMoments(fit_stop=1, total_samples=2, tokens_per_sample=4)
    _, (rows, hessians, seen, _) = _one_sample_capture(
        moments, _routed_tokens(), max_rows=0, want_hessian=False)
    assert all(v is None for v in rows.values()) and hessians == {}
    names = {names for names, _ in moments.calls}
    assert tuple([DENSE]) in names
    for expert in range(2):
        for projection in ("w1", "w3", "w2"):
            assert (f"{EXPERT_PREFIX}.{expert}.{projection}",) in names
    assert seen[DENSE] == 8
    assert all(count == 4 for name, count in seen.items() if name != DENSE)
    finished = moments.finish()
    dense = torch.cat(_routed_tokens()).reshape(-1, 4).float()
    fit = finished[FIT][DENSE]
    held = finished[HELDOUT][DENSE]
    assert fit["count"] + held["count"] == dense.shape[0] == 8


def test_shared_aliases_group_into_one_callback_with_exact_unit_counts():
    moments = _RecordingMoments(fit_stop=1, total_samples=2, tokens_per_sample=4)
    _one_sample_capture(moments, _routed_tokens(), max_rows=0,
                        want_hessian=False, shared=True)
    grouped = [names for names, _ in moments.calls
               if len(names) == 2 and names[0].endswith(".0.w1")]
    assert grouped == [(f"{EXPERT_PREFIX}.0.w1", f"{EXPERT_PREFIX}.0.w3")] * 2
    finished = moments.finish()
    for expert in range(2):
        # 2 routed rows per expert per one-sample batch; sample 0 is FIT.
        w1 = finished[FIT][f"{EXPERT_PREFIX}.{expert}.w1"]
        w3 = finished[FIT][f"{EXPERT_PREFIX}.{expert}.w3"]
        assert w1["count"] == w3["count"] == 2
        assert finished[HELDOUT][f"{EXPERT_PREFIX}.{expert}.w1"]["count"] == 2
        # ONE materialized H per canonical group, handed read-only to both.
        assert w1["hessian"] is w3["hessian"]


def test_observed_h_and_count_equal_independent_xtx_for_both_partitions():
    moments = _RecordingMoments(fit_stop=1, total_samples=2,
                                tokens_per_sample=4, max_prefix_rows=4)
    _one_sample_capture(moments, _routed_tokens(), max_rows=0,
                        want_hessian=False)
    dense = torch.cat(_routed_tokens()).reshape(-1, 4).float()
    fit, held = dense[:4], dense[4:]
    finished = moments.finish()
    assert set(finished) == {FIT, HELDOUT}
    for role, expected in ((FIT, fit), (HELDOUT, held)):
        units = finished[role]
        assert set(units) == {DENSE, *packed_targets()}
        unit = units[DENSE]
        assert unit["role"] == role
        assert unit["count"] == 4
        torch.testing.assert_close(unit["hessian"], expected.t() @ expected)
        torch.testing.assert_close(unit["inputs"], expected)
        assert unit["prefix_sample_ids"].dtype == torch.int64
        # One sample id per retained row, not per batch/chunk.
        assert unit["prefix_sample_ids"].shape[0] == unit["inputs"].shape[0]
        assert unit["prefix_sample_ids"].tolist() == [0 if role == FIT else 1] * 4
        assert unit["max_abs"] == pytest.approx(float(expected.abs().amax()))
    torch.testing.assert_close(finished[FIT][DENSE]["hessian"], fit.t() @ fit)
    torch.testing.assert_close(
        finished[HELDOUT][DENSE]["hessian"], held.t() @ held)


def test_prefix_sample_ids_are_per_retained_row_across_routed_subsets():
    moments = _RecordingMoments(fit_stop=1, total_samples=2,
                                tokens_per_sample=4, max_prefix_rows=5)
    moments.set_sample(0)
    moments.consume(("u",), torch.ones(3, 4))          # 3 rows kept (sample 0)
    moments.set_sample(1)
    moments.consume(("u",), torch.ones(2, 4))          # 2 more rows (sample 1)
    fit = moments.finish()[FIT]["u"]
    held = moments.finish()[HELDOUT]["u"]
    # Each split role keeps its own bounded prefix with per-row ids.
    assert fit["inputs"].shape[0] == 3
    assert fit["prefix_sample_ids"].tolist() == [0, 0, 0]
    assert fit["prefix_sample_ids"].shape[0] == fit["inputs"].shape[0]
    assert held["inputs"].shape[0] == 2
    assert held["prefix_sample_ids"].tolist() == [1, 1]
    assert held["prefix_sample_ids"].shape[0] == held["inputs"].shape[0]
    assert fit["count"] == 3 and held["count"] == 2


def test_split_roles_stay_disjoint_and_boundary_moves_with_fit_stop():
    moments = _RecordingMoments(total_samples=3, fit_stop=1,
                                tokens_per_sample=4)
    _one_sample_capture(moments, _routed_tokens() + [torch.zeros(1, 2, 4)],
                        max_rows=0, want_hessian=False)
    finished = moments.finish()
    assert set(finished[FIT]) == {DENSE, *packed_targets()}
    # Batch 3 is two all-zero tokens; both route to expert 0, so the dense
    # census drops from two 4-row samples to 4 + 2 heldout rows, and expert 1
    # gets a zero-row derivation that never reaches the consumer.
    assert set(finished[HELDOUT]) == {DENSE, *packed_targets()}
    assert finished[FIT][DENSE]["count"] == 4
    assert finished[HELDOUT][DENSE]["count"] == 6
    assert finished[FIT][DENSE]["prefix_sample_ids"].tolist() == [0] * 4
    assert finished[HELDOUT][DENSE]["prefix_sample_ids"].tolist() == [1] * 4 + [2] * 2


def test_zero_row_routed_calls_are_harmless():
    # The fixture's router splits tokens between the two experts (2 rows each
    # per 4-token sample).  Every forwarded group call is a non-empty subset
    # of one sample; zero-row derivations never reach the seam's consumer.
    moments = _RecordingMoments(fit_stop=1, total_samples=2, tokens_per_sample=4)
    _one_sample_capture(moments, _routed_tokens(), max_rows=0,
                        want_hessian=False)
    for _, rows in moments.calls:
        assert 0 < rows.shape[0] <= 4
    expert1_w1 = [rows for names, rows in moments.calls
                  if names[0] == f"{EXPERT_PREFIX}.1.w1"]
    assert [tuple(r.shape) for r in expert1_w1] == [(2, 4), (2, 4)]
    finished = moments.finish()
    for role in (FIT, HELDOUT):
        assert finished[role][f"{EXPERT_PREFIX}.1.w1"]["count"] == 2


def test_mixed_sample_batches_are_refused_not_guessed():
    moments = DisjointRowMoments(tokens_per_sample=4)
    moments.set_sample(0)
    with pytest.raises(ValueError, match="non-empty subset"):
        moments.consume((DENSE,), torch.zeros(8, 4))
    with pytest.raises(ValueError, match="non-empty subset"):
        moments.consume((DENSE,), torch.zeros(0, 4))
    # A routed subset of one sample is legal and lands in the set role.
    moments.consume((DENSE,), torch.ones(3, 4))
    assert moments.finish()[FIT][DENSE]["count"] == 3
    with pytest.raises(RuntimeError, match="set_sample"):
        moments._sample_index = None
        moments.consume((DENSE,), torch.zeros(4, 4))
    with pytest.raises(ValueError, match="sample index"):
        moments.set_sample(512)
    with pytest.raises(ValueError, match="sample index"):
        moments.set_sample(-1)
    # Refusals left no partial state beyond the one legal routed subset.
    assert moments.finish()[HELDOUT] == {}
    assert moments.finish()[FIT][DENSE]["count"] == 3


def test_set_sample_rejects_out_of_range():
    moments = DisjointRowMoments(fit_stop=384, total_samples=512)
    moments.set_sample(0)
    moments.set_sample(383)
    moments.set_sample(511)
    with pytest.raises(ValueError):
        moments.set_sample(384 + 128)
    with pytest.raises(ValueError):
        moments.set_sample(True)


def test_prefix_device_choice_is_validated():
    with pytest.raises(ValueError, match="prefix_device"):
        DisjointRowMoments(prefix_device="npu")
    for choice in ("cpu", "device"):
        assert DisjointRowMoments(prefix_device=choice).prefix_device == choice


def test_moments_stay_on_the_rows_device_until_finish(monkeypatch):
    """No full-row CPU transfer and no per-batch host sync during consume."""
    moments = DisjointRowMoments(fit_stop=1, total_samples=2,
                                 tokens_per_sample=4, prefix_device="device")
    cpu_transfers = []
    original_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target == "cpu" and self.ndim >= 1 and self.numel() >= 16:
            cpu_transfers.append(tuple(self.shape))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    moments.set_sample(0)
    moments.consume((DENSE,), _routed_tokens()[0].reshape(4, 4))
    moments.set_sample(1)
    moments.consume((DENSE,), _routed_tokens()[1].reshape(4, 4))
    # Capture phase: zero device-to-CPU transfers of moment-sized tensors.
    assert cpu_transfers == []
    # The two roles own disjoint buffers on the rows' device.
    assert set(moments._groups) == {
        (FIT, (DENSE,)), (HELDOUT, (DENSE,))}
    state = moments._groups[(FIT, (DENSE,))]
    assert state["hessian"].device == torch.device("cpu")  # rows are CPU here
    expected_fit = _routed_tokens()[0].reshape(4, 4).t() @ _routed_tokens()[0].reshape(4, 4)
    # finish() performs the single bounded transfer.
    finished = moments.finish()
    assert len(cpu_transfers) >= 1
    torch.testing.assert_close(finished[FIT][DENSE]["hessian"], expected_fit)
    torch.testing.assert_close(
        finished[HELDOUT][DENSE]["hessian"],
        _routed_tokens()[1].reshape(4, 4).t() @ _routed_tokens()[1].reshape(4, 4))


def test_single_owned_hessian_buffer_mutates_in_place_per_group():
    moments = DisjointRowMoments(fit_stop=1, total_samples=2,
                                 tokens_per_sample=4, max_prefix_rows=0)
    moments.set_sample(0)
    moments.consume(("u",), torch.ones(2, 4))
    state = moments._groups[(FIT, ("u",))]
    buffer = state["hessian"]
    moments.set_sample(0)  # second batch of the SAME sample-split group
    moments.consume(("u",), torch.ones(2, 4))
    assert state["hessian"] is buffer  # in-place, not a fresh tensor
    assert state["count"] == 4
    # Both aliases map to the same group state; finish hands one read-only H.
    record = moments.finish()
    torch.testing.assert_close(record[FIT]["u"]["hessian"],
                               4 * torch.ones(4, 4))
    assert all(s["hessian"] is None for s in moments._groups.values())


def test_close_clears_ownership_on_failure_and_blocks_reuse():
    moments = _RecordingMoments(fit_stop=1, total_samples=2,
                                tokens_per_sample=4, max_prefix_rows=4)
    moments.set_sample(0)
    moments.consume((DENSE,), _routed_tokens()[0].reshape(4, 4))
    assert moments.finish()[FIT]
    moments.close()
    with pytest.raises(RuntimeError, match="after close"):
        moments.consume((DENSE,), _routed_tokens()[0].reshape(4, 4))
    with pytest.raises(RuntimeError, match="after close"):
        moments.finish()
    moments.close()  # idempotent


def test_resource_check_gates_allocations_and_transfers():
    labels = []
    moments = _RecordingMoments(fit_stop=1, total_samples=2,
                                tokens_per_sample=4, max_prefix_rows=4,
                                resource_check=labels.append)
    moments.set_sample(0)
    moments.consume((DENSE,), _routed_tokens()[0].reshape(4, 4))
    assert "split_moments:before_moment_growth:fit:model.layers.0.attention" in labels
    assert any(label.startswith("split_moments:before_prefix_growth:")
               for label in labels)
    # No transfer label before finish: consume does not leave the device.
    assert not any(label.startswith("split_moments:before_hessian_transfer:")
                   for label in labels)
    finished = moments.finish()
    assert "split_moments:before_hessian_transfer:fit:model.layers.0.attention" in labels
    # finish() is a cached publication snapshot; repeats do not re-transfer.
    assert moments.finish() is finished

    def refusing(label):
        raise RuntimeError("disk admission refused")

    refused = DisjointRowMoments(fit_stop=1, total_samples=2,
                                 tokens_per_sample=4, resource_check=refusing)
    refused.set_sample(0)
    with pytest.raises(RuntimeError, match="disk admission"):
        refused.consume((DENSE,), _routed_tokens()[0].reshape(4, 4))
    assert refused.finish() == {FIT: {}, HELDOUT: {}}
    moments.close()
    refused.close()


def test_off_mode_is_bit_identical():
    tokens = _routed_tokens()
    targets = [DENSE, *packed_targets()]
    baseline = _collect_activations(_RoutedModel(), targets, tokens, 3, "cpu",
                                    want_hessian=True)
    with_callback = _collect_activations(
        _RoutedModel(), targets, tokens, 3, "cpu", want_hessian=True,
        row_consumer=lambda names, rows: None)
    assert baseline[2:] == with_callback[2:]
    for left_out, right_out in zip(baseline[:2], with_callback[:2]):
        assert left_out.keys() == right_out.keys()
        for name in left_out:
            if left_out[name] is None:
                assert right_out[name] is None
            else:
                torch.testing.assert_close(left_out[name], right_out[name])


def test_consumer_owns_its_bytes_despite_reused_mutable_source_buffer():
    moments = _RecordingMoments(fit_stop=1, total_samples=2,
                                tokens_per_sample=4, max_prefix_rows=4)
    source = [batch.clone() for batch in _routed_tokens()]
    model = _RoutedModel()
    index = {"current": 0}

    def forward_batch(batch):
        moments.set_sample(index["current"])
        index["current"] += 1
        model(batch)
        batch.fill_(-999)  # the caller reuses/mutates the source buffer

    try:
        _collect_activations(model, [DENSE, *packed_targets()], source, 0, "cpu",
                             forward_batch=forward_batch, row_consumer=moments)
    except BaseException:
        moments.close()
        raise
    dense = torch.cat(_routed_tokens()).reshape(-1, 4).float()
    finished = moments.finish()
    torch.testing.assert_close(
        finished[FIT][DENSE]["hessian"], dense[:4].t() @ dense[:4])
    torch.testing.assert_close(
        finished[HELDOUT][DENSE]["hessian"], dense[4:].t() @ dense[4:])
    # The seam handed over a live view; the accumulator's recorded snapshot
    # must not alias it (mutation above would otherwise corrupt moments).
    for _, rows in moments.calls:
        assert rows.untyped_storage().data_ptr() != source[0].untyped_storage().data_ptr()
