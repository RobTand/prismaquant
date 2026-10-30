# Draft Mergify fleet CI contract

Scope: GitHub #3140, Paperclip PRI-1065 under PRI-1057. This draft does not
authorize integration, queue activation, runner registration, or protection
changes. The independent review, adversarial design consultation and manager
synthesis remain required.

## Existing queue configuration

Reuse PR #3142 at `e5d0aeed8279000f862709ac2eccb0efb6762389`; do not add a second
queue configuration. It defines serial default and release queues, batch size
one, one check slot, no retries and no automatic enqueue rule. The release
label selects the release queue. Mergify injects branch-protection conditions;
the configuration does not add a review exemption. Its in-place path updates
the original issue-named PR, so the ordinary issue/branch policy still applies.

The synthetic-PR fallback below admits only a single independent batch member,
matching that configuration. Multiple members, batches, parent batches and
split batches fail closed. Increasing queue parallelism or batch size requires
a new reviewed provenance contract, not a branch-prefix exemption.

## Trust boundaries

`linked-issue.yml` remains a metadata-only `pull_request_target` workflow. It
checks out the trusted base commit, reads GitHub metadata and publishes the
existing `linked issue` status on the event head. It never runs contributor
code. Ordinary PR issue and branch rules are unchanged.

For a synthetic queue PR, `queue_policy.cjs` requires all of these facts:

- Fresh same-repository, open PR metadata against `main`.
- The real `mergify[bot]` Bot identity, matched by immutable ID against GitHub's
  user endpoint, and a `mergify/merge-queue/main/` ref. The prefix alone grants
  nothing; PR text and commit messages do not supply membership.
- Authenticated Mergify Merge Queue Status API data binding queue rule name,
  UUID batch ID, queue PR number and a complete nonempty single-member list.
  Fresh GitHub reads bind that queue PR number to its exact ref and head.
- The original member is open, not a draft or fork, passes the existing
  issue/branch validator, and has the right release-label routing.
- The synthetic commit has exactly two parents: current `main`, then the exact
  current member head. A stale base/head or extra parent fails closed.
- Repeated member, queue PR, main and authenticated batch reads agree. HTTP/API
  errors, pagination indicators, missing membership and binding changes fail
  closed. Cross-service reads are not an atomic transaction: final publication
  repeats admission, and any later mutation requires checks for the new SHA.

The Mergify status endpoint is
`GET https://api.mergify.com/v1/repos/RobTand/prismaquant/merge-queue/status?branch=main`.
It documents an unpaginated status object with `batches` and
`waiting_pull_requests`. The adapter rejects Link headers, explicit next-page
indicators and unsupported shapes. It uses a fixed HTTPS origin, refuses
redirects and has a 15-second timeout. No body parser or unauthenticated
fallback is used.

`MERGIFY_API_TOKEN` must be a separately approved, least-privilege Mergify API
read credential. It is referenced only by hosted metadata jobs. This draft
does not install a secret, grant permissions, or register a runner. Missing or
unsupported credentials block the synthetic fallback; the in-place ordinary
PR path needs no Mergify credential. Do not send `GITHUB_TOKEN` to Mergify.

## Fleet execution and check binding

`fleet-ci.yml` starts on completion of `Linked issue policy`. GitHub loads the
`workflow_run` workflow from the default branch. Hosted admission checks the
server's workflow path, event, successful completion, same-repository source,
and exactly one PR/head/base association. A missing association fails closed;
there is no fallback to a user-controlled title or branch search.

Admission repeats the issue and queue checks. For ordinary PRs it resolves
GitHub's `refs/pull/<number>/merge`, requires the returned test-merge SHA and
exact current base/head parents, and tests that integrated SHA. In-place queue
updates take this same path. Synthetic queue PRs test their authenticated
integrated head directly.

Only then does a job run on `[self-hosted, linux, x64, fleet]`. It checks out
that exact immutable SHA with credential persistence disabled. Its token has
only `contents: read`; it receives no repository secrets, write token, OIDC
permission, cache or uploaded artifact. It runs policy tests and the CPU pytest
suite against the installed pinned Tessera dependency. It does not call
PrismaBuild. Candidate code is intentionally executed only inside this isolated
read-only job, not in `pull_request_target` or either hosted write-token job.

Hosted publication repeats admission and compares the complete saved binding.
Only a successful fleet job and unchanged binding complete `fleet integrated CI`
as successful. Failure, cancellation, skipped jobs and verification errors fail
the check. Checks are attached to the tested integrated SHA and, when distinct,
the corresponding PR head; their summary names the actual tested SHA and base.
This lets ordinary/in-place heads carry the same verified integration result.
No check named `pb-tests` is created by this workflow.

## Protection snapshot and eventual transition

CoS supplied the authenticated October 28, 2026, 02:34:08 UTC snapshot for
`main` at `05b62b4742793a2188abd4ed041fc4d46fd6c2c0`:

- Required `linked issue`, GitHub Actions app ID `15368`.
- Required `pb-tests`, unrestricted app ID; strict status mode is false.
- Zero required approving reviews, stale reviews dismissed, no required code
  owner or last-pusher approval. Task-level independent review still applies.
- Admin enforcement is off; there are no repository/effective branch rulesets.

This is a supplied timestamped snapshot, not a claim of a fresh settings read.
Both current required contexts and all actual review settings stay unchanged.
Local green tests do not satisfy either required context or the merge gate.

After independent exact-head review, the existing cross-family consultation
and manager synthesis, CoS must establish disposable, isolated fleet runners
and credential readiness. Each runner must support Node >=22, Python 3.12,
the CPU test dependencies, owned temporary storage and the suite's cgroup
fixtures. Candidate code must not share a persistent host with credentials or
trusted publisher jobs. This draft does not claim these conditions are met.
CoS must also verify the runner's runtime-token and cache isolation: a
`workflow_run` job uses the default-branch workflow context. A disposable disk
alone does not prevent a compromised candidate from attempting main-scoped
cache writes through Actions runtime credentials. Hosted trusted jobs must
never restore candidate-controlled cache entries. No runner may be admitted
until that isolation is established and independently checked.

Only after a real green `fleet integrated CI` check on the actual Mergify
integrated SHA may CoS refresh protection evidence and replace the required
`pb-tests` context with `fleet integrated CI`, explicitly bound to GitHub
Actions app ID `15368`. Retain `linked issue` with that app binding and preserve
every current review/protection setting. The final queue requirements are
`linked issue` plus `fleet integrated CI`, not a permanent dependency on the
retired producer. Mergify must inherit those requirements for both queues.

If that bootstrap check cannot run before the transition, leave reviewed PRs
unmerged and report their exact heads/check URLs to the manager. Do not post a
second `pb-tests` producer, use PrismaBuild as CI, waive checks, delete reviews,
or merge outside Mergify. No live queue success is claimed by the local tests.

Release rehearsal remains the separate PR #3141, exact head
`36e8ec29dd303d192a27febadb1feb1a06828a79`, with its retained local good/bad
build receipts. Do not rerun or publish that completed slice for this draft.

## Local verification

Run `node .github/scripts/linked_issue.test.cjs` and
`node .github/scripts/queue_policy.test.cjs`. The negative matrix covers
identity/ref spoofing, empty/missing/invalid members, stale/fork/draft/unlinked
members, unsupported or truncated batches, pagination, API failure, exact SHA
and parent mismatch, metadata races, untrusted workflow associations and stale
publication. These mocked service tests are local evidence, not a real fleet
run or an atomic cross-service guarantee.
