Lifting my own hold on this one. The reason I held it was that the native
gate used `pytest.xfail`, so a plan that undercharged would have reported
green. That is gone, and what replaced it discriminates:
`tests/test_glm_campaign_streaming.py:369` asserts
`max(growths[1:]) <= sum(plan['phases']['resident_anchors'].values())`, with
the per-anchor growth measured by the row's own guard across
`before_selected_anchor_batch` / `after_selected_anchor_batch`, and the first
step excluded and named as the runtime's one-time first-use cost rather than
folded into headroom. `:341` separately asserts the admission arithmetic the
row actually ran, `memory_bytes <= budget - baseline`, against a baseline
measured in that process.

Verified independently rather than taken from the report. The assertion bites:
probe row `034099017411`, sparklina, rc=1, parent `37d86e62635e`, a commit not
on this branch, with five `resident_anchors` terms cut to one byte; it fails on
the second step. The green rows resolve at this head, `checkout_snapshot.parent`
`0da7dff6f39c` on every one: `ed69def83163` sparklina with `accel=1`, which is
the row that actually executes the CUDA branch, and `d83da8ff8fc4`,
`09fc1698bc00` and the rest on dl380g10. CI is CLEAN.

Two terms remain documented gaps rather than derived, and the comments name
them: the export archive's pickle and directory metadata, and the producer's
own working set inside `encode_linear`, which is not traceable from this
repository and keeps its conservative bound.

Ready to merge.

-- claude-triage
