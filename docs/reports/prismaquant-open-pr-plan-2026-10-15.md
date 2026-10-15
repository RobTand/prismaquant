# Open PR delivery plan

Read-only refresh on 2026-10-15 finds the same 36 open PrismaQuant PRs and the
same head commits as the 2026-10-14 inventory. No merge or closure is authorized
by this plan. Preserve existing issuegraph and merge-train owners.

| Action | PRs | Reason |
| --- | --- | --- |
| Incorporate locally; retain open until replacement delivery lands | [#2669](https://github.com/RobTand/prismaquant/pull/2669) | `add6056aaa` adopts acquisition availability fallback using the current classifier. Before/after PB checks fail one case then pass all 16. |
| Preserve independently | #2693, #2677, #2670, #2665, #2664, #2657, #2647, #2645, #2644, #2643, #2634, #2631, #2593 | Existing prerequisite, independent correctness, evidence or contract work. Rebase against the new owners and verify exact heads before merge. |
| Preserve independently after source adoption audit | #2659, #2653, #2633, #2614, #2611, #2598, #2591, #2586, #2557 | The proposed behavior is absent here. Moving owners does not incorporate these patches. |
| Keep the existing block | #2684, #2492 | D44 delivery/bundle has unresolved acceptance and composed-gate holds. No blanket merge. |
| Preserve unrelated work | #2632, #2619, #2613, #2599, #2589, #2584, #2551 | No refactor ownership overlap justifies taking or retiring it. |
| Conditional supersession only | #2694, #2641, #2489, #2472 | Each needs the replacement proof below. Retain them until that proof and accepted delivery exist. |

The nine source-audited independent changes are concrete. #2659 adds recipe/shape
memory telemetry. #2653 adds explicit GLM owner selection and anchor consistency.
#2598 adds attention census/publication. #2586 adds a held-content close reread
stronger than stat fencing. #2614 adds future pin-update workflow checks. #2611
adds consumer growth reservation. #2591 adds pinned-buffer staging. #2557 adds
joint body/MTP allocation, followed by #2633's solved-target output stamp. The
current refactor does not claim those correctness or performance improvements.

Choose one canonical delivery for #2693/#2694, whose document bytes are identical;
retire the duplicate only after corrected linking and accepted delivery. Replace
#2641's raw-hash AST locks with verified mutation/refusal/read-lifecycle coverage
before retiring it. Preserve #2489 until canonical experiment delivery explicitly
adopts the changed deadline/stall-supervision contract with source provenance.
Preserve #2472 until the chosen D44 entry point proves dry-run publication false,
real publication true and failed execution false.

Delivery order starts with agreed census/profile/golden/duplication prerequisites,
then independent correctness at its new owner. #2557 precedes #2633. Keep #2591's
historical profiler and Netdata evidence unchanged; exact-head acceptance needs
its own current evidence. A prior targeted receipt or missing check rollup never
replaces the final composed gate.

Detailed source citations, cached patches, exact heads, checks and dependency
records remain at `/home/rob/tmp/pq-refactor-20261014/reports/`. This report is a
disposition decision, not an assertion that all retained PRs are merge-ready.
