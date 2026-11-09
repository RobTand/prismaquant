# Land a pull request through the merge queue

Every change reaches `main` through Mergify's merge queue. The configuration
is [`.mergify.yml`](../.mergify.yml). The gate workflow supplies `pr-gate` on
ordinary PRs and `gate` on temporary batch drafts. Development tests run
as admitted Flux jobs; cite their job IDs and results in the PR body.

## Before you queue

1. Name your branch `prismaquant-<issue>` or `prismaquant-<issue>-<word>`.
   Suffixes use lowercase letters, digits and hyphens, for example
   `prismaquant-<issue>-tl3-<node>`.
2. Reference the same issue in the PR body. A partial change uses
   `Refs #<issue>.` with an open parent issue in this repository. A completed
   issue uses a supported closing reference such as `Closes #<issue>`.
   The `linked issue` check verifies both the reference and branch number.
3. Wait for `linked issue` and `pr-gate` to pass.

Classic branch protection requires a pull request but no status checks.
The `queue` label path enforces `linked issue` and `pr-gate` on your pull
request, and the batch draft is judged by `gate` alone. Mergify would also
apply any classic required check to the batch draft, which can't pass the
issue rule, so keep required status checks out of classic protection.

## Queue and dequeue

- Add the `queue` label. Mergify adds `queued` after accepting the PR.
- Remove `queue` to take the PR out of the queue.
- Add `release` alongside `queue` to use the release queue, served first.
- People with write permission and TL3's GitHub App, `tl3-robtand[bot]`, can
  use the comment commands `@mergifyio queue` and `@mergifyio requeue`.
  Put the command at the start of the comment.

## TL3's pull requests

TL3 uses branches named `prismaquant-<issue>-tl3-<node>` and starts each
pull request body with `Refs #<issue>.`, so `linked issue` passes as it
does for every pull request. TL3 tests each change in admitted jobs and
reviews it, then posts its `TL3 approval` check on the reviewed head and adds
the `tl3-queue` label. That pair queues the pull request in the default queue
in place of `queue` and `pr-gate`; the issue-link check stays required. TL3
follows Mergify's queue check on the PR head and never reads `pr-gate`, so a
`pr-gate` condition would leave a failing TL3 PR unqueued without TL3 noticing.
The batch's full `gate` still decides every merge. A TL3 change that would
fail `pr-gate` can enter a batch, fail `gate`, and delay other pull requests
while the batch splits. Don't push to a TL3 branch: TL3 approved that exact
head. The rule name `Queue TL3-approved PRs` is fixed because TL3 follows its
`Rule: Queue TL3-approved PRs (queue)` check.

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
