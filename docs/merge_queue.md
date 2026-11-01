# Land a pull request through the merge queue

Every change reaches `main` through Mergify's merge queue. The configuration
is [`.mergify.yml`](../.mergify.yml). The gate workflow supplies `pr-gate` on
ordinary PRs and `gate` on temporary batch drafts. Development tests run
through PrismaBuild; cite their action keys and results in the PR body.

## Before you queue

1. Name your branch `prismaquant-<issue>` or `prismaquant-<issue>-<word>`.
   Suffixes use lowercase letters, digits and hyphens, for example
   `prismaquant-<issue>-tl3-<node>`.
2. Reference the same issue in the PR body. A partial change uses
   `Refs #<issue>.` with an open parent issue in this repository. A completed
   issue uses a supported closing reference such as `Closes #<issue>`.
   The `linked issue` check verifies both the reference and branch number.
3. Wait for `linked issue` and `pr-gate` to pass.

Classic branch protection requires `linked issue` only. Mergify adds it to
the queue conditions, which also name `linked issue` and `pr-gate`, and never
to the merge conditions, so the batch draft is judged by `gate` alone.

## Queue and dequeue

- Add the `queue` label. Mergify adds `queued` after accepting the PR.
- Remove `queue` to take the PR out of the queue.
- Add `release` alongside `queue` to use the release queue, served first.
- People with write permission and TL3's GitHub App, `tl3-robtand[bot]`, can
  use the comment commands `@mergifyio queue` and `@mergifyio requeue`.
  Put the command at the start of the comment.

The TL3 coordinator owns TL3's separate approval and label path. It is added
separately to the same queue; this guide describes the `queue` label path.

## What happens to a batch

Mergify groups queued PRs into batches. The default queue batches one to seven
PRs and waits at most two minutes to form a batch. The release queue always
checks one PR at a time. Two batches can run checks concurrently, and merges
follow queue order.

For each batch, Mergify opens a temporary draft PR on a
`mergify/merge-queue/` branch. It contains `main`, the batches ahead of it,
and the current batch. Draft checks preserve each original PR's head, which
TL3's approval names. The batch's `gate` check tests the full gate population
on that integrated SHA. A green `pr-gate` on each original PR alone does not
allow the batch to merge.

When `gate` passes, Mergify merges the original PRs into `main` with merge
commits. A failed batch is split to isolate the failing PR. Mergify marks a
removed PR `dequeued` and links the failure. Each batch has a 30-minute checks
timeout and no automatic Mergify check retries.

## Requeue after a failure

A failed PR stays out of the queue even if it still has the `queue` label.
Removing and re-adding that label does not requeue it.

1. Read the failed check and Mergify's comment. Fix the PR, or resolve the
   infrastructure failure, and push any needed changes.
2. Keep the `queue` label and wait for the current head's `pr-gate` and
   required checks to pass.
3. Comment `@mergifyio queue`. Mergify rechecks the conditions and queues the PR.
   The older `@mergifyio requeue` command still works but is deprecated.

Never merge outside the queue.
