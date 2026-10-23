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
ending can. The identity is consumed only after an authoritative retirement
response returns, including a response that defers again. If the call raises
(for example, PB cannot acquire its ownership lock before mutation), the pending
outcome and ending remain retryable through the existing owner failure handling:
nonblocking releases retain their attempt cap and pacing, and funding retains
the read's original deadline and refusal handling. A later returned keyed
deferral consumes the ending and gates its republished successor, rather than
turning exception recovery into repeated retirement polling. This bookkeeping
is process-local wakeup suppression, not durable retirement state. Recovery
continues to derive ownership and materialization from PB's records.

Keyless `own-copy-in-flight` keeps its bounded retry behavior and can transition
to the keyed gate. Foreign pins, promotion handoffs, errors and unknown refusals
retain their existing distinct handling, even if a stale action key is present.

## CPU verification

No acceptance files were changed; the existing fake-SDK/virtual-clock tests
exercise the production adapter and owner, including the post-binding SDK-loader
trap. The added validation-retry tests reuse that lane to check transient lock
failure recovery, successor gating after recovery, the existing attempt cap,
and funding recovery within the original read deadline. No GPU or fleet latency
claim is made.

- `python -m pytest tests/test_keyed_egress_validation_retry_2789.py tests/test_keyed_egress_owner_2789.py tests/test_docs_staleness.py tests/test_architecture_doc.py tests/test_docs_entrypoints.py tests/test_duplication_alias_scope_audit.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"`
  — **75 passed**, with 14 existing Torch JIT/Python 3.14 deprecation warnings.
- `python -m pytest tests/test_stage_a_produced_boundary_chain.py tests/test_stage_a_produced_ownership.py tests/test_stage_a_produced_failfast.py -q --basetemp="$TMPDIR/pytest"`
  — **52 passed** in this repair run. This is fresh observed evidence, not a
  claim that the separately supplied fixture-admission failures were passes.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"33150a941df029c5a328d49472dc77020925d10e621664703611d1a897c3bf45","size":830}},"operation":"03da1486ad16091810a16e10f04687b0b5829e3efaae0182d271cd4395103b40","receipt":"075823d4c02569d4255d5166519b8941d05bcb577b40f9090b8d61223d5639dd"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3af062a64d5bbf35b7e0670e05e2d63436cbc9254248d78bcf3b4d839af50c36","size":281}},"operation":"c6bb6627a7dcaf1c0149467b760bcbcb168fd7d6cc5a3f67d921d88741f07841","receipt":"8015e77ccd5ec11b1ba24c8ac61aa57ea1579b539c6f1e79cc29be2b10467108"}]
