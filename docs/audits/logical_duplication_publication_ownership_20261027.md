# Completion-receipt staging ownership, PRI-226

This finding arose while reading the publication family at main
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`. Native child #3134 owns the fix.
Tested implementation source is `c025758a6110c0057534bb828d8c9ff066ce6287`.
Only this audit document follows that source. Independent review and merge
remain pending; #3061 and #1295 remain open.

## Defect and smallest repair

`dsv4_campaign_completion._atomic_publish_new` entered its cleanup block
before acquiring its fixed PID stage with exclusive create. If another file
or symlink occupied that name, acquisition raised FileExistsError and the
finally block removed the entry that caused refusal.

The writer now acquires the descriptor before entering cleanup. No new
publisher, cache, loader, format or identity API is introduced. The direct
caller remains `publish_completion_receipt`, and its production caller is
`tools.wait_dsv4_aura_campaign._inner`. Receipt encoding, fixed staging name,
mode 0600, publication by hard link, existing-target and concurrent-target
refusals, file/directory fsync order and post-acquisition cleanup are retained.
This repair establishes ownership at acquisition. It does not establish a
new defense against arbitrary path replacement after acquisition.

The surrounding publication census preserves different policies. Cost-stage
publish_new_bytes returns a Boolean for occupied targets and has a separate
nofollow descriptor mode. Cluster publication uses mkstemp and distinguishes
initial occupation from a concurrent winner. Completion remains independently
loadable from a verified snapshot. A common link/fsync sequence alone does
not justify flattening those contracts.

## Deterministic fleet evidence

All checks used the published PrismaBuild client with the project's pinned
CPU interpreter on dl380g10. They were ordinary test actions, not idle-host
measurements. The retained dispatcher is `publication_ownership_jobs.py`.

| Run | Source | Action and outcome |
| --- | --- | --- |
| Before repair | `edeab1db00c586cd5e9d021ddd66e851534900ca` | `bf756cd4f15267e0f1ecf1a056c36b1d64b87c9f74fce1b6c5b2facb03be3b4b`: two failures reproduce deletion of an occupied file and symlink; three owned-stage controls pass; nine unrelated cases deselected. |
| After repair and prose corrections | `c025758a6110c0057534bb828d8c9ff066ce6287` | `3367f9f2a890232adf39c446e763fbdd3773ed538f1bcc566f71c1a1c8eb927e`: 89 tests pass, no failures or skips; five Python files compile. Fourteen upstream Torch/Python 3.14 deprecation warnings remain. |

The new cases use real filesystem entries and injected link outcomes, without
sleeps, concurrent timing or load thresholds. They pin preservation of the
unowned inode/bytes and symlink, success bytes/mode, cleanup after an ordinary
link error, and preservation of a concurrent target winner. The existing
completion tests exercise public receipt publication/verification and the
standalone waiter import. Adjacent checks cover strict JSON diagnostics,
serving-profile and patch-set contracts, architecture and docs staleness.

`verify_publication_ownership.py` checks both native terminal outcomes, exact
terminal log hashes and every tracked source blob against each retained
snapshot. Successful CAS claim and payload checks pass. The first claim read
reported an absent receipt; a later read passed without re-execution. Both
observations are retained and the cause is unestablished. Worker attestation
is outside this claim checker's validation.

Commands, native endings, sanitized actions, logs, source checks and claims
are retained under `/mnt/shared/paperclip-runs/PRI-226/` with the prefix
`publication-ownership`, alongside the dispatcher and verifier. The uploaded
census/evidence bundle is the reviewable copy for readers without host access.

## Separate prose fixes and remaining finding

A separate commit corrects five encountered claims in four files:

- The serving-lane deletion note describes its unknown-profile fallback in
  past tense and identifies CB layout coverage as archived history.
- The inherited export-lane docstring uses a derived-profile example without
  implying that the retired CB lane is active.
- The Tessera structure docstring calls v21 then pinned, preserving its
  historical gate event without presenting it as the current pin.
- The serving-runtime patch overview distinguishes recorded sets from the
  two unbuilt source candidates and their missing qualification manifests.

The separate exact-activation cleanup concern is recorded, with uncertainty,
in #3135. Supported same-name concurrency has not been proved and no race was
reproduced. A repair must reconcile the ordinary and preallocated writer
layers and their descriptor/serialization contracts, which exceeds this
completion-receipt unit. Its issue records the source evidence and this
reason for deferral.

No performance, GPU, artifact corruption, serving, KL, bpp or residency result
is claimed. Existing #1301/#1303 owners, archive/GGUF exclusions, scientific
gates and all approved PRI-226 source heads remain preserved. No historical
source seal is rewritten. Final program closure still needs the independent
current-main residual census after delivery.
