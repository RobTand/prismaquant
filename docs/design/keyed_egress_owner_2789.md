# Stage A keyed egress retirement handshake (PQ #2789)

This is the PQ owner-loop portion of PB #811, not a scheduler, generic wait
service, SDK migration, or GPU performance qualification.

## Local ownership and authority

`BoundaryProducedPublication.retire` is the single gate shared by waited and
nonblocking release, pending-release drain, and refill/window funding. It uses
only the already-bound SDK's `retire_batch` and the publication's selected
public `PoolQueue.current_ending`. No second SDK lookup, action-wait client,
record-path polling, dependency update, or topology/configuration change is
needed.

The first retirement request is unchanged. A typed `own-egress-in-flight`
deferral with a nonempty top-level `egress_action_key` retains the outcome in
the publication. Further calls observe that key rather than mutating retirement
while it is live. Missing, unreadable, ambiguous, malformed, or exception-producing
observations retain the debt. The existing owner pacing and absolute deadline
remain responsible for bounded waiting; this gate neither sleeps nor creates a
fresh timeout, so nonblocking drains still overlap compute.

A usable DONE, FAILED, or WITHDRAWN ending permits one authoritative
`retire_batch` re-evaluation. It does **not** prove receipt completeness,
coverage of a newer publication, or retirement. In particular, withdrawal is
not interpreted as successful generation coverage. PB must still validate its
materialization/egress receipt, and PQ's existing typed refusal handling still
applies. Credit, context, held batches, and pending releases are cleared only
by the existing successful-retirement path.

PB does not return the publication generation from `retire_batch` and may
republish the same content-addressed key. The adapter therefore remembers each
consumed `(key, ending generation)` per batch. A lone public legacy ending with
no generation also gets at most one wakeup. If validation returns another keyed
deferral, the old ending cannot repeatedly wake its successor; a new usable
ending can. The identity is consumed before the retirement call, conservatively
including calls that raise after potentially mutating PB state. This bookkeeping
is process-local wakeup suppression, not durable retirement state. Recovery
continues to derive ownership and materialization from PB's records.

Keyless `own-copy-in-flight` keeps its bounded retry behavior and can transition
to the keyed gate. Foreign pins, promotion handoffs, errors and unknown refusals
retain their existing distinct handling, even if a stale action key is present.

## CPU verification

No acceptance files were changed; the existing fake-SDK/virtual-clock tests
exercise the production adapter and owner, including the post-binding SDK-loader
trap. No GPU or fleet latency claim is made.

- `python -m pytest tests/test_keyed_egress_owner_2789.py -q --basetemp="$TMPDIR/pytest"`
  — **18 passed**.
- `python -m pytest tests/test_stage_a_produced_boundary_chain.py tests/test_stage_a_produced_ownership.py tests/test_stage_a_produced_failfast.py -q --basetemp="$TMPDIR/pytest"`
  — **52 passed** in this run. This is fresh observed evidence, not a claim
  that the separately supplied earlier fixture-admission failures were passes.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b53bc1d4271eeddfb4757f57eb06a1ea0bb86dca3f9adedea2a844187f6f32e3","size":269}},"operation":"177dbc9fe8d37e87760e13dcc202187e6e0f6819c614c1e8dea4c7c53d8957be","receipt":"53829ebfdbdee3eddc5425ee8a5364fe40d75c653e27ef0dccf4a7f25788e6e4"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"929d67ed471cb081dd963c4c186e667cb3818f499597e14b3151e1be5cc8466a","size":281}},"operation":"49a0c0203f65cbbd739acad510a3e348338ae29e729a15f173def2f833e3d1b4","receipt":"540245ec1a191edd011449cc710378df263de65147aea08eab987dfdd9415ed7"}]
