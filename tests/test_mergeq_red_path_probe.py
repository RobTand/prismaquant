"""A deliberately failing test: it proves the merge queue's red path.

This file exists only on the probe branch for PQ #1929's merge-queue
acceptance. The queue must report it as a new failing node ID, bisect its
batch to this PR, post a failing pb-tests status with a culprit comment,
and never merge it. The PR that carries it is closed unmerged afterwards.
"""


def test_the_merge_queue_reports_this_pr_as_the_culprit():
    assert False, "red-path probe: this PR must never merge"
