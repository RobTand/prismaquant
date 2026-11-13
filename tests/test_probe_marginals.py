"""The Fisher-marginal merge rule of `probe_statistics.merge_marginals`:
sums add, the activation bound merges by maximum, and partial entries merge
without loss or aliasing."""
from __future__ import annotations

import numpy as np

from prismaquant.probe_statistics import merge_marginals


def test_merge_marginals_sums_add_and_absmax_maxes():
    a = {
        "fisher_row": np.array([1.0, 2.0], dtype=np.float32),
        "fisher_col": np.array([3.0], dtype=np.float32),
        "g_sq_sum": np.array([4.0, 5.0], dtype=np.float32),
        "act_sq_sum": np.array([6.0], dtype=np.float32),
        "act_absmax": np.array([9.0], dtype=np.float32),
    }
    b = {
        "fisher_row": np.array([10.0, 20.0], dtype=np.float32),
        "fisher_col": np.array([30.0], dtype=np.float32),
        "g_sq_sum": np.array([40.0, 50.0], dtype=np.float32),
        "act_sq_sum": np.array([60.0], dtype=np.float32),
        "act_absmax": np.array([2.0], dtype=np.float32),
    }
    merge_marginals(a, b)
    np.testing.assert_allclose(a["fisher_row"], [11.0, 22.0])
    np.testing.assert_allclose(a["fisher_col"], [33.0])
    np.testing.assert_allclose(a["g_sq_sum"], [44.0, 55.0])
    np.testing.assert_allclose(a["act_sq_sum"], [66.0])
    # 9 > 2: a bound merges by MAXIMUM, never by sum (11.0 would be the
    # bug this asserts against).
    np.testing.assert_allclose(a["act_absmax"], [9.0])

    c = {"act_absmax": np.array([1.0], dtype=np.float32)}
    merge_marginals(c, {"act_absmax": np.array([5.0], dtype=np.float32)})
    np.testing.assert_allclose(c["act_absmax"], [5.0])


def test_merge_marginals_asymmetric_presence():
    """A partial stats entry may predate the flag, or carry keys the
    destination lacks; neither side may raise or silently drop."""
    dst: dict = {}
    src = {"fisher_row": np.array([1.0, 2.0], dtype=np.float32)}
    merge_marginals(dst, src)
    np.testing.assert_allclose(dst["fisher_row"], [1.0, 2.0])
    assert "fisher_col" not in dst
    # Copied, not aliased: mutating the source must not move `dst`.
    src["fisher_row"][0] = 99.0
    np.testing.assert_allclose(dst["fisher_row"], [1.0, 2.0])

    before = dict(dst)
    merge_marginals(dst, {})
    np.testing.assert_allclose(dst["fisher_row"], before["fisher_row"])
