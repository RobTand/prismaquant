# PrismaQuant Agent Rules

These rules are mandatory for coding agents working in this repository.
Before implementing new functionality, read this file,
`docs/design/design_guidelines.md`, and `docs/ARCHITECTURE.md`.

## Repository change delivery

Every change to `main` must start with an issue in this repository and arrive
through a pull request. Never push commits directly to `main`. When the change
completes an issue, put a supported closing reference such as `Closes #123` in
the pull request description so GitHub links its native closing relationship.
For a partial slice, instead add an explicit `Refs #123` or `Part of #123` line
naming an OPEN parent issue in this repository. The required `linked issue`
check verifies that parent through GitHub without closing it. Do not create a
bookkeeping issue just to deliver a slice. Plain mentions, quoted code examples,
pull request numbers, nonexistent parents, closed non-closing parents and
references to another repository do not satisfy this check.

## Core Principles

1. **GPU-bound by default.** Production probes, cache fills, recache,
   polish, export, and validation must be designed so the GPU is the
   bottleneck. A hot path that is CPU-bound, disk-bound, or NVMe-bound is a
   bug unless the user explicitly requested an offline data-prep step.
2. **Self-contained, deterministic operation.** Agents may design, diagnose,
   review, and improve PrismaQuant, but they must not be required to operate
   it. Ordinary quantization, distributed scheduling, barriers, retries,
   recovery, allocation, export, and validation must run from explicit
   versioned configuration and machine-readable state through deterministic
   code paths. If execution needs intelligence to decide what to do next, the
   application is missing a contract or state-machine transition. Ambiguous
   state must fail closed; an operator or agent may repair the implementation
   or supply a new explicit input, but may not serve as the runtime scheduler.
3. **Use the existing cache and prefetch system.** Do not create a parallel
   cache, preload, or residency mechanism for rendered weights or
   activations. Extend `ProductionWeightCache`, `PerturbedActivationCache`,
   the streaming model prefetch path, or the existing pipeline wiring.
   Production paths should fail fast when required resident data cannot be
   prefetched instead of silently streaming from NVMe.
4. **Right quantization for the right layer.** Preserve PrismaQuant's core
   contract: per-Linear empirical selection with measured quality/cost
   tradeoffs. Avoid model-wide defaults unless they are only a fallback or
   have been validated against the per-Linear path.
5. **Only ship formats the target engine serves performantly.** A format can
   appear in research menus before it is production-ready, but it must not
   become a production default until the engine for its container loads it,
   generates correctly, and uses a performant kernel on representative shapes.
   Three containers, three gates: `compressed-tensors` on vanilla vLLM (no
   PrismaQuant kernels); GGUF on llama.cpp and the vLLM GGUF plugin, gated
   additionally on bit-exactness against `gguf-py`; and the Tessera wire on
   **Tessera's own** vLLM plugin (package `tessera.serving`, entry point
   `tessera = "tessera.serving:register"` under `vllm.general_plugins`,
   registering `quant_method = "tessera"`), pinned by
   `prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`. That lane
   has no enable flag — the checkpoint's `quant_method` selects the plugin, and
   the single operator knob is `TESSERA_SERVE_MODE=resident|streamed` — and it
   is gated on an unforked vLLM (the plugin is installed into the stock image,
   no core patches), on `device_qualified` native cells in Tessera's packaged
   `runtime_contract.json`, and on every such cell declaring
   `requires_plugin: "tessera"`, because stock vLLM has no reader for these
   bytes and the route is plugin-gated rather than merely flag-gated. Its
   admission is pinned to an **exact Tessera commit plus the SHA-256 of the
   `runtime_contract.json` that commit packages** (pin schema
   `prismaquant.tessera_serving_runtime_pin.v2`, 2026-09-04: Rob retired the
   release-tag requirement). `require_pinned_tessera_runtime` refuses unless
   the pin equals the reader's three pinned constants AND the installed
   contract hashes to the pinned digest, so any other Tessera on `PYTHONPATH`
   still answers False for every rung — by the pin, not by an edit.
   `version_is_release` is recorded and advisory; no gate reads it. It is
   limited by each matched cell's published scope and the closed-world TP
   ceiling. Read each matched cell's evidence and lane schema from the
   current pinned runtime's packaged contract. The recorded routed pair from
   v22 was historical, not a claim about today's cell roster. The historical
   v39 GLM-image cells carried route-only evidence with smoke not recorded,
   which the unchanged status-only evidence gate permits; that is not new
   served KL or a shipped PrismaQuant MoE artifact. Preserve
   `lane_eligibility.cell_evidence_admits` and the independent export and
   serving gates; do not infer admission beyond a matched cell. The non-vLLM-native lanes are
   sanctioned, not exceptions; what is forbidden is a forked runtime.
   PrismaQuant must never vendor or import the Tessera *serving* runtime;
   compatibility crosses that repository boundary only through the immutable
   pin and the packaged contract.

   **A fourth container was retired on 2026-09-02.** The codebook lane
   (NVFP4-CB / FP8-CB) served by the separately released
   [`gridbook`](https://github.com/RobTand/gridbook) plugin was removed from
   PrismaQuant by Robert's decision — *"put Tessera in PrismaQuant and remove
   Gridbook"* — and archived whole at `archive/gridbook_lane_2026-09-02/`. Its
   pin, exporter, serving profiles, ship gates and lane documents are gone from
   the live tree; the Tessera wire is its successor. Do not re-add a Gridbook
   pin, a `gridbook_runtime/` directory, or an `EXPORT_CONTAINER=nvfp4_cb`
   path: `run-pipeline.sh` refuses that container with `exit 2`.
6. **Measure on the same calibration contract.** New levers need apples-to-
   apples KL, bpp, and runtime measurements. Compare against the relevant
   shipped or current baseline using the same calibration set, sequence
   length, layer assignment semantics, and production cache behavior.
7. **Report bpp over quantizable parameters only.** Bits-per-parameter
   accounting must exclude immutable BF16 regions that the allocator is not
   allowed to quantize, including `lm_head` and any profile-pinned model
   components. Published uniform-NVFP4 and GGUF k-quant comparisons do not
   average in unquantizable parameters; PrismaQuant reports should follow the
   same convention. (MXFP8 is de-menued — exact-scale FP8 dominates it — so it
   is not the comparison to reach for.)
8. **Reuse local abstractions.** Prefer the existing format registry,
   allocator, production cache, recache, validation harness, and pipeline
   flags. If an abstraction is missing, add it at the shared layer rather
   than building a one-off call site.
9. **Keep cross-layer machinery archived unless explicitly requested.**
   The archived CLADO, propagated-cost, output-Fisher, PrismaSCOUT iteration,
   QUBO, and polish-of-many code is research context, not a production
   shipping lever.
10. **Use the known-good Docker environments.** PrismaQuant has Docker images
   with the required CUDA, PyTorch, Transformers, vLLM, and pipeline
   dependencies already installed. For GPU runs, validation, export, and
   large-model experiments, use those working containers first instead of
   assuming the host Python environment is sufficient or rebuilding ad hoc.
11. **Acquire required dependencies and artifacts.** Agents are authorized to
    download and install task-required files, model checkpoints, CLI tools,
    libraries, packages, and container images without asking solely for
    download/install permission. Prefer scoped, user-local, or containerized
    installs where practical, pin or record versions needed for reproducibility,
    and continue to use the known-good Docker environments for GPU work.
12. **Keep `docs/ARCHITECTURE.md` current in the same commit.** A change to
    pipeline defaults, the stage graph, the format menu, the plugin contract,
    serving-lane defaults, or ship gates is incomplete until
    `docs/ARCHITECTURE.md` (and its diagrams, if topology changed) reflects it
    and its provenance block is re-stamped. `tests/test_docs_staleness.py` and
    `tests/test_architecture_doc.py` enforce the mechanical subset. Dated
    results docs and handovers are append-only history, never a substitute.

13. **Measurement is first-class, and telemetry counts as measurement.** Any
    claim about speed, cost, residency, or "where the time went" is carried by
    a measurement, never by log-line reasoning. Profile BEFORE the change and
    AFTER it -- the delta is the claim, and a bench number without a profile
    does not establish where the time went. Two instruments, both required,
    because they answer different questions: an **in-process profiler**
    (`torch.profiler`, py-spy, `/proc/PID/io`, `nsys`) says where time goes
    *inside* a run; the **Netdata series on both boxes** says whether the box
    was actually loaded, which no in-process tool can see. Attach the evidence
    to the finding, and put it in the acceptance criteria of every perf
    delegation.

    On GB10, **`nvidia_smi.gpu_utilization` is non-diagnostic under load**: it
    means "at least one kernel is resident", not "the SMs are working", and it
    reads 96% for a memory-stalled kernel exactly as for a saturated one
    (measured 2026-08-28: 96% on both sides of a 5.83x throughput change).
    `utilization.memory` is worse -- a fake hard 0. Read **power against the
    ~140 W envelope** instead, and rank implementations by **work per joule**;
    the envelope fraction also estimates remaining headroom, which wall-clock
    cannot. Do not diagnose a GPU hot path from utilization.

14. **Fix the finding where you found it -- file only what you cannot.** A
    defect you noticed and neither fixed nor filed dies with your context. But a
    ticket is not the only record, and it is usually the worse one: whoever
    tripped over the defect understands it better than any fresh agent will, and
    that understanding evaporates the moment the task returns. A filed one-liner
    then costs a brief, a worktree, a fresh-context ramp-up, a report and a
    merge, to re-derive what somebody already knew. (Rob, 2026-09-03, watching a
    dozen tickets arrive in an hour: *"we're just proliferating issues right now
    that may make sense to fix in the context of the way in which they were
    discovered."*)

    **Default: fix it, on your branch, in a separate commit.** One commit does
    one thing so a reviewer can take it or drop it alone -- that constraint is
    satisfied by a second *commit*, and was never a reason to leave a defect
    unfixed. Only the mixed commit is forbidden.

    **File instead when the fix is not yours to make:**
    - it needs a decision only Rob prices -- a default moves, an artifact's bytes
      move, a format menu or serving lane changes, a ship gate is involved;
    - it needs a measurement you are not set up to take -- a served A/B, a
      second model, the other box;
    - it lives in another agent's live branch (read theirs, never edit theirs);
    - it is large enough to swamp the diff of the task you came for.

    Two bounds, so this is not a licence to widen scope: it covers what you
    **trip over** doing the task you came for, never a hunt for adjacent work;
    and every off-task fix is named in the report, one line each, so nothing
    lands unannounced.

    **When you do file, timeliness still binds.** Same working session, before
    starting the next task -- not at the end of the day, not in a handover, not
    in a summary. A finding held in context for "later" has already failed. The
    bar is **"I believe this is wrong"**, not "I have proved it is wrong and
    scoped the fix": over-filing beats losing a finding, and fixing beats both.

    **Say which it was.** An issue filed under this principle records whether the
    filer could have fixed it and chose not to, and why. Without that, a backlog
    stops describing the work and starts hiding it.

    What a filed issue otherwise owes is little: the **evidence at `file:line`**
    (read the line, do not repeat a claim), what breaks and under what inputs, a
    **severity** from the rubric below, and what would fix it -- or, when the fix
    is a judgment call, the options and who decides. Say plainly what you did
    *not* measure, and file an uncertain finding with its uncertainty stated.

    **Severity rubric.** `P0` -- can ship or serve a wrong artifact. `P1` -- a
    gate that cannot catch its own defect, or a wrong or underived number that
    a decision reads. `P2` -- provenance, observability, or a claim beyond its
    evidence. `P2+` -- an important `P2` with a documented objective urgency
    trigger: it blocks a named, already required near-term acceptance or
    handoff from obtaining trustworthy evidence, irreplaceable evidence sits
    at a concrete retention/deletion deadline, or measured recurring resource
    waste threatens an approved campaign's admitted resource window. Name the
    milestone/deadline or measurement and its causal effect; a generic
    blocker, annoyance, or assertion of importance is not a trigger, and
    `P2+` is never a `P1` downgrade -- if missing provenance defeats a safety
    gate or changes a decision number, file `P1` or `P0`. `P3` -- cleanup with
    no decision riding on it. Two orthogonal labels: `measurement-needed` when
    a GPU or served A/B decides it, and `needs-decision` when the answer is a
    trade only Rob prices.

    **One visible canonical priority.** Every newly filed, reopened, or
    materially triaged owned issue carries exactly one leading title prefix --
    `[P0]`, `[P1]`, `[P2+]`, `[P2]`, or `[P3]` -- before handoff; body prose,
    a local ledger, or a status label alone does not satisfy this. Where the
    matching repository priority label exists (`P0`, `P1`, `P2`, `P2+`, `P3`
    all do), apply exactly that one priority label and remove other priority
    labels; the orthogonal labels above may coexist. If the matching label is
    absent, the leading prefix supplies the canonical priority until an
    authorized maintainer adds it: never substitute `P2` for `P2+` and never
    create labels without authority. A missing prefix, multiple priority
    labels, or a title/label mismatch blocks handoff and closure until the
    owner encodes one evidence-backed classification, and the issue routes at
    the highest displayed tier meanwhile. The review parent reviews severity,
    disposition and acceptance within its authority; urgent safe action need
    not wait for that review. Severity stays orthogonal to status: a GPU
    hold, missing owner, or user decision neither lowers it nor by itself
    promotes `P2` to `P2+`. A material reprioritization leaves an audit
    comment (old -> new priority, changed evidence, reason, reviewer), title
    and label updated together. No mass relabel or retrospective
    reprioritization of historical issues; a demonstrated `P0` path stays
    `P0`.

    **One exception, narrower than it was.** A finding in prose -- a doc, a
    comment, a docstring -- is *fixed on sight* and never filed: reading the
    cited line IS the verification, so a stale sentence is a one-line commit.
    (The former second exception, that a delegated worker neither fixes nor
    files, is withdrawn: it described a constraint workers do not have.)

## Code design principles — Rob, 2026-09-30

When two principles collide, pick the one that cuts future cost in THIS codebase.

HARD RULE: refactor to the principle FIRST, then change behavior.

1. Separation of Concerns — one kind of work per part (UI / domain / persistence / infra). Root principle.
2. Encapsulation / Information Hiding — small stable contract; hide internals.
3. High Cohesion + Loose Coupling — change-together lives together; independents talk narrow.
4. DRY — one authoritative representation of each piece of *knowledge* (not every similar line). Avoid over-DRY.
5. KISS — simplest design that works; complexity is the long-term tax.
6. Single Responsibility — one reason to change.
7. Depend on Abstractions — policy doesn't depend on details; both depend on contracts.
8. YAGNI — no speculative features, frameworks, or "later" hooks.
9. Composition over Inheritance — assemble pieces; don't grow fragile hierarchies.
10. Open/Closed (with discipline) — extend at stable boundaries; only where change showed up twice.

Honorable: Law of Demeter · fail fast / illegal states unrepresentable · optimize for deletion · Unix do-one-thing + compose.

Treat as constraints. Violate slogans when judgment says so.

## Implementation Checklist

Before editing:

- Identify the existing mechanism this change should extend.
- Decide how the change stays GPU-bound and resident-prefetched.
- Decide what the before/after measurement is, and which instrument
  produces it (in-process profiler, Netdata series, or both).
- Define the vLLM compatibility gate if formats, export metadata, kernels,
  or compressed-tensors layout are touched.
- Define the KL/bpp/runtime comparison and calibration set.

Before finishing:

- Run targeted tests and compile checks for touched modules.
- Attach before/after profiler evidence for any hot-path or perf change;
  rank GPU work by work-per-joule, never by utilization.
- Add or update tests for new policies, format gates, cache residency, or
  validation behavior.
- Record measured results, commands, and log paths in docs when a claim is
  based on a run.
- Leave experimental methods opt-in until the validation gate in
  `docs/design/design_guidelines.md` is satisfied.
