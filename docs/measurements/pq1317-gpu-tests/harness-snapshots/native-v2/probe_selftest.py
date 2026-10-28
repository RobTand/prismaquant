"""Synthetic tests for ``native_probe``. The CPU preflight runs them under xdist.

They need no GPU and no Tessera. ``container_entry.py`` checks that the probe
records the first assertion (it holds a float) and the third (it names a launch),
and skips the second (an integer). The fourth test checks the worker's own extension
directory when ``PQ1317_EXT_DIR_ROOT`` is set.
"""
import os


def test_a_float_assertion_is_recorded():
    tolerance = 0.25
    measured = abs(0.5 - 0.375)
    assert measured < tolerance


def test_an_integer_assertion_is_not_recorded():
    assert 3 == 3


def test_a_launch_pair_assertion_is_recorded():
    record = {"symbol": "tessera::fused_window_dense", "decoder": "native_fused_window_dense"}
    assert (record["symbol"], record["decoder"]) == ("tessera::fused_window_dense", "native_fused_window_dense")


def test_the_extension_directory_belongs_to_the_worker():
    root, worker = os.environ.get("PQ1317_EXT_DIR_ROOT"), os.environ.get("PYTEST_XDIST_WORKER")
    if root and worker:
        assert os.environ["TORCH_EXTENSIONS_DIR"] == f"{root}/{worker}"
