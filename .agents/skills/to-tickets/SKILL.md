---
name: to-tickets
description: Break approved plans into tracer-bullet tickets for PrismaQuant, PrismaBuild, and Tessera, then publish through the duplicate and evidence intake gate.
---

# To Tickets

Break a plan, spec, or conversation into a set of **tickets**: tracer-bullet vertical slices, each declaring the tickets that **block** it.

The issue tracker and triage label vocabulary should have been provided to you. If not, tell the user to run `/setup-matt-pocock-skills`.

## Process

### 1. Gather context

Work from whatever is already in the conversation context. If the user passes a reference (a spec path, an issue number or URL) as an argument, fetch it and read its full body and comments.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so to understand the current state of the code. Ticket titles and descriptions should use the project's domain glossary vocabulary, and respect ADRs in the area you're touching.

Look for opportunities to prefactor the code to make the implementation easier. "Make the change easy, then make the easy change."

### 3. Draft vertical slices

Break the work into **tracer bullet** tickets.

<vertical-slice-rules>

- Each slice cuts a narrow but COMPLETE path through every layer (schema, API, UI, tests): vertical, NOT a horizontal slice of one layer
- A completed slice is demoable or verifiable on its own
- Each slice is sized to fit in a single fresh context window
- Any prefactoring should be done first

</vertical-slice-rules>

Give each ticket its **blocking edges**: the other tickets that must complete before it can start. A ticket with no blockers can start immediately.

**Wide refactors are the exception to vertical slicing.** A **wide refactor** is one mechanical change (rename a column, retype a shared symbol) whose **blast radius** fans across the whole codebase, so a single edit breaks thousands of call sites at once and no vertical slice can land green. Don't force it into a tracer bullet; sequence it as **expand–contract**. First expand: add the new form beside the old so nothing breaks. Then migrate the call sites over in batches sized by blast radius (per package, per directory), each batch its own ticket blocked by the expand, keeping CI green batch to batch because the old form still exists. Finally contract: delete the old form once no caller remains, in a ticket blocked by every migrate batch. When even the batches can't stay green alone, keep the sequence but let them share an integration branch that all block a final integrate-and-verify ticket; green is promised only there.

### 4. Quiz the user

Present the proposed breakdown as a numbered list. For each ticket, show:

- **Title**: short descriptive name
- **Blocked by**: which other tickets (if any) must complete first
- **What it delivers**: the end-to-end behaviour this ticket makes work

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the blocking edges correct: does each ticket only depend on tickets that genuinely gate it?
- Should any tickets be merged or split further?

Iterate until the user approves the breakdown.

### 5. Publish the tickets to the configured tracker

Publish the approved tickets. **How** depends on the tracker `/setup-matt-pocock-skills` configured; the tickets are the same either way, only the shape of the blocking edges changes:

- **Local files** → write one file per ticket under `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order (blockers first). Each file's "Blocked by" lists the numbers/titles it depends on. Use the per-ticket file template below: one ticket per file, never a single combined file.
- **GitHub in RobTand/prismaquant, RobTand/prismabuild, or RobTand/tessera** → publish one issue per ticket in dependency order through the intake command below. Require a native GitHub parent for every ticket, including plans that began as conversation. If no suitable parent exists, resolve the parent with the requester before publication. Keep the approved breakdown while that question is pending.

For each ticket, save its body in your task worktree or evidence directory. Include a nonempty Reproduction or Measurement section for P0/P1. Run the repo's AGENTS.md intake command without `--apply` first. A likely duplicate refusal names existing issues: reuse that work or revise the scope with the requester. Do not bypass a refusal with `gh issue create` or a tracker write tool.

After the dry run passes, run the same command with `--apply`. Publication checks live open issues again and creates the native child in one mutation. Pass `--label ready-for-agent` only if the repository already defines that label. Never invent a label to satisfy the flow.

In a prismabuild worktree:

```bash
python3 tools/backlog_census/intake.py \
  --repo RobTand/prismabuild --parent PARENT_NUMBER --priority P2 \
  --title 'Ticket title' --body-file BODY_FILE
```

In prismaquant or tessera, use `/home/rob/paperclip-repos/prismabuild/tools/backlog_census/intake.py` and the matching `--repo`. The command must come from the reviewed prismabuild change; if it has not landed, retain the drafts until it does. Record each returned child URL, then set its native blocking relationships using the tracker operation. Parentage and blocking edges are separate relationships. Stop on an uncertain write result and inspect GitHub before retrying.

- **Other real trackers** → use the configured tracker operations in dependency order, with native blocking relationships where supported.

Work the **frontier**: any ticket whose blockers are all done. For a purely linear chain that means top to bottom.

Do NOT close or modify any parent issue.

<local-ticket-template>

# <NN>: <Ticket title>

**What to build:** the end-to-end behaviour this ticket makes work, from the user's perspective, not a layer-by-layer implementation list.

**Blocked by:** the numbers/titles of the tickets that gate this one, or "None (can start immediately)".

**Status:** ready-for-agent

- [ ] Acceptance criterion 1
- [ ] Acceptance criterion 2

</local-ticket-template>

<issue-template>

## Parent

A reference to the native GitHub parent, required for these three repositories. Other trackers use their configured parent convention.

## What to build

The end-to-end behaviour this ticket makes work, from the user's perspective, not layer-by-layer implementation.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Blocked by

- A reference to each blocking ticket, or "None (can start immediately)". Omit this section when blockers were set as native edges.

</issue-template>

In either form, avoid specific file paths or code snippets: they go stale fast. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it and note briefly that it came from a prototype. Trim to the decision-rich parts, not a working demo, just the important bits.
