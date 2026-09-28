"""Dual-teacher scoring: CPU contracts only (no GPU serve)."""
import hashlib
import sys

import numpy as np
import pytest
import torch

from experiments import glm_tr3_full_vocab as exp
from experiments import measure_glm_tr3_vllm as served

ROWS, VOCAB = 6, 37


def np_kl(teacher, candidate):
    """Independent FP64 reference: max-subtracted logsumexp, no torch."""
    t = np.asarray(teacher, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    t = t - t.max(axis=-1, keepdims=True)
    c = c - c.max(axis=-1, keepdims=True)
    lt = t - np.logaddexp.reduce(t, axis=-1, keepdims=True)
    lc = c - np.logaddexp.reduce(c, axis=-1, keepdims=True)
    return (np.exp(lt) * (lt - lc)).sum(axis=-1)


def synthetic(seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    t1 = (rng.standard_normal((ROWS, VOCAB)) * scale).astype(np.float32)
    t2 = (rng.standard_normal((ROWS, VOCAB)) * scale).astype(np.float32)
    cand = (t1 + rng.standard_normal((ROWS, VOCAB)) * 0.5).astype(np.float32)
    return t1, t2, cand


def hook(rank=0, world=1, teacher=None, teacher2=None):
    h = exp.PromptLogitsCapture(rank=rank, world_size=world, rows=ROWS, vocab_size=VOCAB,
                                require_cuda=False)
    if rank == 0:
        h.arm(0, "final-0000", torch.from_numpy(teacher), teacher2=(
            None if teacher2 is None else torch.from_numpy(teacher2)))
    else:
        h.arm(0, "final-0000", None)
    return h


def drive(h, cand):
    # Stock legacy call order: the 1-row sample call, then the full prompt.
    h(None, (), torch.zeros(1, VOCAB))
    h(None, (), torch.from_numpy(cand.copy()))
    return h.finish("final-0000")


def test_each_teacher_kl_matches_independent_numpy_fp64():
    t1, t2, cand = synthetic()
    result = drive(hook(teacher=t1, teacher2=t2), cand)
    v1 = exp.collect_tp_result([result], window_id="final-0000", world_size=1,
                               rows=ROWS, vocab_size=VOCAB)
    v2 = exp.collect_tp_result2([result], rows=ROWS)
    np.testing.assert_allclose(v1, np_kl(t1, cand), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v2, np_kl(t2, cand), rtol=1e-12, atol=1e-14)
    assert not np.allclose(v1, v2)  # the two teachers are genuinely different


def test_teacher1_unchanged_by_presence_of_teacher2():
    t1, t2, cand = synthetic(1)
    solo = drive(hook(teacher=t1), cand)
    dual = drive(hook(teacher=t1, teacher2=t2), cand)
    assert solo["values"] == dual["values"]  # bit-identical, not just close


def test_single_teacher_finish_keys_unchanged():
    t1, _, cand = synthetic(2)
    result = drive(hook(teacher=t1), cand)
    assert set(result) == {"rank", "world_size", "window_id", "calls", "values",
                           "target_logprobs", "logits_layout"}
    with pytest.raises(StopIteration):  # no values2 anywhere: collect2 has nothing to read
        exp.collect_tp_result2([{"rank": 1, "values2": None}], rows=ROWS)


def test_teacher2_state_cleared_after_finish():
    t1, t2, cand = synthetic(3)
    h = hook(teacher=t1, teacher2=t2)
    drive(h, cand)
    assert h.teacher2 is None and h.values2 is None and h.teacher is None
    # next window without teacher2 does not inherit the previous one
    h.arm(1, "final-0001", torch.from_numpy(t1))
    h(None, (), torch.zeros(1, VOCAB))
    h(None, (), torch.from_numpy(cand.copy()))
    assert "values2" not in h.finish("final-0001")


def test_teacher2_geometry_and_ownership_refused():
    t1, t2, _ = synthetic(4)
    h = exp.PromptLogitsCapture(rank=0, world_size=1, rows=ROWS, vocab_size=VOCAB,
                                require_cuda=False)
    with pytest.raises(ValueError, match="teacher2"):
        h.arm(0, "final-0000", torch.from_numpy(t1), teacher2=torch.zeros(ROWS - 1, VOCAB))
    non_owner = exp.PromptLogitsCapture(rank=1, world_size=2, rows=ROWS, vocab_size=VOCAB,
                                        require_cuda=False)
    with pytest.raises(ValueError, match="only TP rank zero"):
        non_owner.arm(0, "final-0000", None, teacher2=torch.from_numpy(t2))


def test_tp_owner_scores_both_teachers_non_owner_none():
    t1, t2, cand = synthetic(5)
    owner = hook(0, 2, t1, t2)
    other = hook(1, 2)
    for h in (owner, other):
        h(None, (), torch.zeros(1, VOCAB) if h.rank == 0 else None)
        h(None, (), torch.from_numpy(cand.copy()) if h.rank == 0 else None)
    results = [h.finish("final-0000") for h in (owner, other)]
    np.testing.assert_allclose(exp.collect_tp_result2(results, rows=ROWS), np_kl(t2, cand),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate TP teacher2"),
    ("missing", "missing/nonfinite"),
    ("short", "missing/nonfinite"),
    ("nan", "missing/nonfinite"),
])
def test_collect_tp_result2_refusals(mutation, match):
    good = [0.1] * ROWS
    results = [{"rank": 0, "values2": list(good)}, {"rank": 1, "values2": None}]
    if mutation == "duplicate":
        results[1]["values2"] = list(good)
    elif mutation == "missing":
        results[0]["values2"] = None
    elif mutation == "short":
        results[0]["values2"] = good[:-1]
    else:
        results[0]["values2"][2] = float("nan")
    with pytest.raises(ValueError, match=match):
        exp.collect_tp_result2(results, rows=ROWS)


def test_host_staging_is_one_window_at_a_time(tmp_path):
    """Two teachers stage sequentially; the host peak stays at one window, not two."""
    import tracemalloc
    arrays = [np.full((256, 4096), v, dtype=np.float32) for v in (1.0, 2.0)]
    paths = []
    for i, a in enumerate(arrays):
        p = tmp_path / f"w{i}.npy"
        with p.open("wb") as f:
            np.save(f, a)
        raw = p.read_bytes()
        paths.append((p, {"path": p.name, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(a.shape)}))
    size = paths[0][1]["bytes"]
    tracemalloc.start()
    try:
        for p, d in paths:  # the _resident_window pattern: load, hand off, drop
            a = served.load_teacher_window(p, d)
            assert a.shape == (256, 4096)
            del a
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.1 * size  # bound: one window (about 1.27 GB at production shape)


BASE_ARGS = ["--model", "m", "--candidate-digest-cache", "c", "--panel", "p", "--teacher", "t",
             "--teacher-sha256", "a" * 64, "--serve-image", "img@sha256:" + "b" * 64,
             "--output", "o", "--kv-cache-dtype", "auto", "--expected-kv-cache-dtype", "auto"]


def run_main(monkeypatch, extra):
    seen = {}
    monkeypatch.setattr(served, "measure", lambda args: seen.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["measure"] + BASE_ARGS + extra)
    served.main()
    return seen.get("args")


def test_cli_single_teacher_defaults_unchanged(monkeypatch):
    args = run_main(monkeypatch, [])
    assert args.teacher2 is None and args.teacher2_sha256 is None


def test_cli_accepts_both_teachers(monkeypatch):
    args = run_main(monkeypatch, ["--teacher2", "t2", "--teacher2-sha256", "c" * 64])
    assert (args.teacher2, args.teacher2_sha256) == ("t2", "c" * 64)


@pytest.mark.parametrize("extra", [
    ["--teacher2", "t2"],
    ["--teacher2-sha256", "c" * 64],
    ["--teacher2", "t2", "--teacher2-sha256", "a" * 64],  # same sha as teacher 1
])
def test_cli_refuses_incomplete_or_duplicate_teacher2(monkeypatch, extra):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, extra)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for resident windows")
def test_arm_capture_arms_teacher2_when_given(tmp_path):
    t1, t2, cand = synthetic(6)
    descriptors = []
    for i, a in enumerate((t1, t2)):
        p = tmp_path / f"w{i}.npy"
        with p.open("wb") as f:
            np.save(f, a)
        raw = p.read_bytes()
        descriptors.append({"path": p.name, "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(a.shape)})

    class Model:
        def __init__(self):
            self.h = exp.PromptLogitsCapture(rank=0, world_size=1, rows=ROWS, vocab_size=VOCAB)

    m = Model()
    served.arm_capture(m.h, index=0, window_id="final-0000", descriptor=descriptors[0],
                       teacher_root=tmp_path, target_ids=None,
                       descriptor2=descriptors[1], teacher2_root=tmp_path)
    assert m.h.teacher2 is not None
