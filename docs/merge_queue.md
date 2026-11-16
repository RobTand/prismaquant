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
3. Wait for `pr-gate` to pass.

Classic branch protection requires a pull request but no status checks.
The `queue` label path requires two queue conditions on your PR's text: a body
line `Closes #<N>`, `Fixes #<N>`, `Resolves #<N>`, `Refs #<N>` or `Part of #<N>`,
and a branch `prismaquant-<N>[-suffix]` (`ig/`, `pg/`, `pg-train/` and `release`
branches are exempt). Mergify does not check that the issue exists or is open, or
that the branch number equals the issue number; you keep those rules yourself.
The draft requires `gate` alone. Mergify would also apply any classic required
check to the batch draft, so keep required status checks out of classic protection.

## Queue and dequeue

- Add the `queue` label. Mergify adds `queued` after accepting the PR.
- Remove `queue` to take the PR out of the queue.
- People with write permission and TL3b's GitHub App, `tl3-robtand[bot]`, can
  use the comment commands `@mergifyio queue` and `@mergifyio requeue`.
  Put the command at the start of the comment.

## What happens to a PR

Mergify checks one PR at a time, in `queue` label order. There are no priority lanes.
For each PR, Mergify opens a temporary draft on a `mergify/merge-queue/` branch.
The draft contains the newest `main` and that PR. Its `gate` must pass before merge.
An external merge resets the check. Original PR heads stay unchanged.

Mergify merges each passing PR with a merge commit. A failed PR leaves the queue.
Mergify adds `dequeued` and links the failure. Checks have a 90-minute timeout and no automatic retries.

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
