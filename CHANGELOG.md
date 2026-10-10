# Changelog

## Unreleased

### Added
- Read Tessera lane schema v12 and apply the per-launch rung scope (#2511).
  A launch with `rungs_q256` joins a unit only at a rung its scope covers;
  a launch without the key keeps the scope of its cell. The scope covers
  census rungs plus the allowable run-table rungs those rungs derive.
  The launch scope joins the cell record and the contract answer, so a
  scope-only change moves the dev-pin answer. No pin moves: the serving
  and export pins stay at lane schema v11 until a later change bumps
  them. The v66 routed fixture proves R768 selects only the class
  decoder and R832 to R1088 select only the historical decoders.
- The fleet registry attests both x86 merge-train layers for the SDK4 and SDK5 vehicles (#2467).
  The SDK5 entry names `/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python`, not its base interpreter.
  PB `975efe59c9d63ae5de43a3a001b1b9a8d5369da9a82d1e6bc5dc3dc1a56243cd` verifies the non-editable PB `027103d9` and Tessera `fca4c6ce` installs.
  Its guard admits PR #2216 head `f6960f8`; pytest reports 10 passed.
  Current dependency pins and the existing SDK4 entry stay unchanged.

  PB `df3d153fa68ce8780031596028e8620a8be65042c93bcd7214b621c5ae39cd9f` executes `suite.command` from fleetgraph `1d2d5a6d1bac7a727ee17903b0decd37ec33a248`.
  It uses the captured live interpreter table, SHA-256 `efbbbd1b87c43d9b3d6bd63aff67697352a91361c770fc1127f8cf9598cc2701`.
  The SDK4 vehicle selects `/home/rob/venvs/pq-task-suite-layer-20261008/bin/python`.
  The SDK5 vehicle at PR #2216 head `d08c3ba9` selects `/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python`.
  Each selected interpreter passes the dependency pin guard and all 50 row-class tests.
  Portable placement tests replace the host-dependent SDK4 pin assertions.

  The same action replays the retained private build script from PB `7386f1618cf0739623072428b6da532b48fe7fca4de691caa9cc18dd055407e0`.
  Only private destinations, the temporary directory, and the immutable source transport differ.
  Content manifests cover both train layers, both pin bases, and both source environments, including symlink targets.
  Before and after manifests match: SHA-256 `33b5839e2adef872abffe2edf2ce9ca9b225d7c61479094fe51c14862cc349bc`, with zero changed entries.
  The action retains the manifests and selector output under `/mnt/shared/tessera-measurements/pq2467-a4-ka_6p__1/`.
  This proves preservation during the checks and replay; the original build has no retained before manifest.

- The day-zero model intake tool and new-model runbook reuse existing profile and source metadata interfaces (#2410).
  The central processor path writes a valid structure draft without tensor payload reads or profile registration.
  Unsupported kinds remain explicit, and inconsistent inputs refuse before full-weight download or launch.
  The opt-in brain floating-point degree-two check executes the existing vLLM prompt smoke through Docker.
  Its central processor preflight checks the real runtime arguments without a graphics processor.
  Native serving qualification and all production defaults stay unchanged.
- The digest census now records source commit `30b6231` at `docs/audits/digest_site_census_pq1301_2026-10-04.json` (#2540).
  The refresh retains every historical record. It adds 420 primitive calls across 320 scopes, 275 supplemental calls, and 22 protected layouts.
  Each load-bearing call records its exact algorithm, encoding options, input order, framing, digest width, result type, read boundary, and refusal behavior.
  Exclusive one-call cohorts route eligible work to #2541, #2542, or #2543. The change has no source, scanner, test, or baseline behavior.

### Fixed
- The allocator layer config stamps the solved target with provenance (#2623).
  Each writer call passes its explicit solve target, and the metadata keeps
  the parsed `--target-bits` value beside the stamped one. A candidate write
  stamps its own target instead of the ambient CLI default. The stock and
  Tessera pin digests move by that block only; all allocations stay identical.
- The G3 harness now resides in `tools/g3job` and consumes PB phased residency (#2301).
  It omits a source tensor only when every requested forward uses a complete replacement.
  Used source tensors retain their byte-integrity checks.
  The manifest follows setup, decoder layers, and teacher reads through the profile's layer names.
  The consumers retain Docker mounts, the PB environment, reader leases, and RAM epochs.
  The map cache follows atomic publication, including RAM updates at the same fragment count.
  Phase leases batch published ranges by tier instead of each tensor.
  The real four-layer comparison uses the pinned source loader, expert packing, and original panel window.
  Its logits and both FP64 KL arrays match bitwise with omission on and off.
  This diagnostic does not qualify full-model quality, performance, or promotion.
  Hash batches use the shared IO engine and preserve their digests.
  Development mode retains identity stamps without a new promotion gate.
  The pinned `/pq` package and action `1cb0d64a` remain unchanged.
  The merge with `origin/main` preserves the accepted fix and both documentation additions.
  PB `42ee5b235134` passes 113 CPU cases; PB `07921c3dd917` preserves bitwise equality in the real container.
  The next merge retains main at `3d2841dc9605` and its independent changelog addition.
  An unavailable RAM cover now selects published stage covers for the same phase and expected bytes.
  The shared lease owner classifies each refusal; integrity and unknown refusals still fail.
  PB `35251b1db6f4` reproduces the old RAM refusal; PB `7b2fd0569f8c` passes all 117 affected CPU cases.
  PB `36269dbc3563` completes the real comparison on Sparklina with bitwise logits and FP64 KL equality.
  The next correction uses PB's public client instead of internal modules.
  Both manifest entry points share the existing bounded byte checks and PB decoder.
  The normal production entry point retains its exact SDK pin.
  The host loads the container adapter from its pinned source file, not the checkout's `tools` package.
  PB `064d872a4635` passes 164 affected CPU cases, including both G3 import and PB boundary guards.
  PB `74f166933103` preserves real logits and both FP64 KL arrays bitwise in the producer container.
  The next correction tests the launcher's image seal on the pinned adapter.
  The tests start the real launcher with the real pinned adapter and a replaced docker executable.
  Default dev mode stamps a differing image and starts the container with the observed digest.
  Certified mode refuses the same image before any container starts.
  The pinned adapter binds `image_content_sha256`; the checkout's own adapter binds `_runtime_identity` and is not loaded.
- The `linked issue` check enforces `prismaquant-<issue>` branches with an optional lowercase suffix (#2518).
  The branch issue must match a verified same-repository closing reference or an open parent reference.
  Branches with `ig/` or `release` prefixes and pull requests created before `2026-10-09T17:00:00Z` retain branch exemptions.
  These exemptions do not bypass the issue-link check.
  PrismaBuild action `e0087aafc8cd` passes all 62 behavioral cases on x86.
  Action `4a11ec1d4a68` passes the direct gate smoke and both syntax checks with a simulated GitHub API.
- A hand-built `Glm5NextProfile` now refuses KDA fused-member queries without a declared config (#2456).
  The error names the missing `config.json` document.
  The lane still owns every fused-group result when the config exists.
  Shared-expert MLP, MLA, indexer, and standalone attention queries retain their config-independent behavior.
  Tests cover both model prefixes and both `f_a_proj` spellings.
- The unknown-deferral test starts its release timer after initial staging and tensor access (#2304).
  The real mover retains the fixture's normal staging budget; only release uses the 30-second budget.
  A synthetic 31-second staging delay verifies immediate refusal, zero release retries, zero supported deferrals, and the exact unclassified debt.
  Production deadlines and fail-closed release behavior remain unchanged.
  PB `cc59619e6a893b3778cfd0e8863565a6b6dd98e594631114e5210ee8e444dc0f` passes 11 affected tests, the direct delay smoke, and the compile check on x86.
  The same run retains deterministic staging-timeout and release-budget coverage.
- **Production pilot result fixtures use one real installed SDK5 helper**
  (Refs #2152, #1293). The production Gateway, resolver, CAS, queue and capture
  binder now consume the same Git/RECORD-qualified SDK5 package. The positive
  and source/invocation/attempt negative controls no longer request admission
  from the preserved SDK4 source archive. No installed-SDK injection fallback,
  mixed package tree, runtime publication or GPU qualification is introduced;
  the durable portable SDK5 pack and native helper adoption remain PB #1485
  prerequisites. The fixture reads the install identity with the standard
  library, because the pbtest pin verifier is absent in xdist workers.

- **Original reader authority joins the selected native producer and outer
  target** (Refs #2152). The strict result request uses SDK5's existing public
  native-context owner. Published source controls, the reader's actual snapshot
  and independently accepted target-family transfer, producer reservation,
  complete helper tree and every native delivery identity must agree. A later
  consumer's resource/session tuple and the distinct launch/selected-attempt
  provenance labels are not substituted for those producer observations.
  Seven joins are D32 seals and go through `seal_check`. They compare the
  reader's record of its producer run with the identity the SDK selected. They
  are the helper root, the helper generation, the reservation, the delivery
  claims, their launch labels, the complete helper tree and the family
  restamp. Certified mode (`PRISMAQUANT_DEV_MODE=0`) refuses each one. Default
  dev mode stamps `[DEV-MODE]` and continues. Dev mode hashes no helper tree.
  The target, readset, calibration and config joins refuse in both modes.
  Runtime version validation reuses the existing exact SDK owner. This is
  nonactivating source preparation: current SDK4/DC480 fixtures, c437 helpers,
  deployment and GPU/source admission are unchanged; a qualified SDK5 helper
  generation remains a coupled prerequisite.

- A failed fused-mapping lookup now stops the native export and the artifact completeness check (#2443).
  Both callers used to swallow every exception from `profile.fused_sibling_leaf_mapping()`.
  With the GLM lane lookup failing, the export returned an empty fused mapping and wrote its `ignore` list without the fused siblings.
  The completeness check reported fused units as claimed by no mechanism.
  Both now raise the lookup error. A working lookup gives the same mapping as before.
  No format, default, gate threshold or artifact byte changes when the lookup works.
- The capture observer retains a finished campaign when only profiler or telemetry evidence is incomplete (#2315).
  Development mode writes `status: "complete"` with a `dev_uncertified` stamp and the missing instruments named.
  The retained result carries no speed, energy or residency qualification.
  Certified mode keeps the refusal and the failed status.
  A real campaign error, native profiler teardown failure, and a monitor that never stopped fail in both modes.
  Missing telemetry retains each actual instrument name and its exact error detail.
  Failed rejected-trace deletion also fails in both modes; a successful deletion leaves an evidence-only cap rejection.

### Changed
- **Original consumer repins PrismaBuild to SDK5** (Refs #2152, PB #1481/#1482).
  `staged_lease.PB_READER_LEASE_PIN_COMMIT` moves to merged PB #1482
  `027103d9a8417e06c7f13356e58779a313cd7088` and `PB_CLIENT_SDK_VERSION` to 5 —
  one exact commit/version contract, no dual-SDK probing, alias, editable
  install or fallback. The strict original-source result read now requires the
  selected immutable attempt's native producer context; runtime documents,
  `original_cuda_control`'s launch-owned helper check and the fixture
  provenance pins follow the same owner-resolved identities. The portable SDK5
  pack (PB #1485), fleet helper-runtime selection and live deployment remain
  separate and unchanged; no capability, admission or serving claim rides the
  pin.

- **The shared connected-fixture pin moves to an immutable SDK5 source bundle**
  (Refs #2152, #2455). The reader pin had moved to SDK5 while
  `tests/pb_runtime_generation_pin.json` still named the SDK4 archive. The
  Stage A produced-output, band handoff, spool, retirement and
  render-publication suites then refused their own transport by version.
  The pin now names a read-only `git archive` of PB `027103d9` under
  `qualification/pq-pb-sdk5-20261009`. `tests/test_pb_generation_pin_1084.py`
  requires one commit for both pins. The SDK4 archive stays on the mount, and
  the consumer still refuses it by exact version. No runtime, default,
  numerical or serving behavior changes. The pbtest pin guard refuses a shard
  whose interpreter lacks a non-editable Git install of the reader pin, so
  every PrismaQuant pbtest run now needs an SDK5 interpreter and the SDK4
  interpreters are refused. The x86 interpreter is
  `/home/rob/venvs/pq-task-suite-layer-sdk5-20261009/bin/python` (#2467).
  The required task-suite gate runs on it and reports 76 passes and zero
  skips (PB `20cb79a0504632a78f6412465012eb868f56f5a432b4717cba6ff3c619560bb6`).
  The gate record in `tests/conftest.py` and `docs/ARCHITECTURE.md` follows.

- The paired expert-dominance menu rule applies only to complete routed layers (#2288).
  A subgroup verdict reprices its option without pruning it, and the verdict stays in provenance.
  The complete-assignment exact filter and the final emission guard still refuse dominant trades.
  The price arithmetic, strict-half boundary, cancellation, and all-zero behavior stay unchanged.
- The allocation byte-identity test retains the canonical `layer.json` digest across Tessera pin moves (#2426).
  Its fixture excludes only `contract_version` and `reviewed_contract_sha256` from the pin block.
  The commit, read digest, read-byte comparison, contract path, and all other fields remain in the oracle.
  The retained D13 and v60 outputs produce the same digest in the final CPU check.
  The applicability and Pareto digests remain unchanged.
- The Tessera pin moves to `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60 (Tessera #1033, PQ #2426).
  The serving and development constants, legal-domain provenance, and identity snapshot move in one commit.
  The generated admission answer remains unchanged.
  The new producer supplies the attention projection APIs and the canonical rung-allowability API.
  The contract retains the v56 cells and native-extension rows.

  The isolated x86 interpreter leaves shared defaults, the D13 overlay, and active measurements unchanged.
  Tests derive pin identities and cell rosters from their owners instead of duplicate version, digest, and count literals.
  No new seal, serving result, or performance result is claimed.

- The ship-gates runner uses `digests.file_sha256hex` at every call (#2413).
  Remove the private digest helper and unused import. Keep the digest recipe and read size.
  Consolidate unique operator instructions in ARCHITECTURE and remove the standalone guide.
  State the HF-only task scope. Unmeasured GPU examples use operator-supplied resource demand.
- The report publisher reuses `digests.indent2_json_file_bytes` (#2413).
  Preserve sorted ASCII-escaped JSON, nonfinite refusal and the trailing newline.
  Give independent ship-gates contracts domain-specific names and migrate every caller without aliases.
  Keep the duplication ratchet and its baseline unchanged.
- Consume the shared G3 array verifier through the quality owner (#2413).
  Add verify-only consumer cases for edited summaries, consistent edited gates,
  missing arrays and corrupt bytes. Preserve unconditional numerical checks.
- Preserve the quality owner's dev replay stamp in the aggregate ship result (#2413).
  Keep stored metrics. Add verify-only consumer regressions for owned bytes,
  mathematical comparison and certified source replacement. No new seal gate is added.
- Stamp generic WikiText cache provenance drift under D32 (#2413).
  Preserve exact corpus hashes, sampling, token values and own-byte integrity.
  Keep the certified fixture refusal and the unchanged DSv4 input path.
- Refuse unsafe actual preflight destinations and retained runtime logs (#2413).
  Reuse the destination validator for configured and derived paths.
  Create logs exclusively so a later collision cannot truncate retained bytes.
- Add one configured ship-gates action and CPU preflight (#2413, D50 item 4).
  Keep every lane gate. Replay quality criteria through the shared quality owner.
  Keep offline G3 separate from served KL. Record stage exits, logs and hashes.
  Use the generic gold producer verifier for uniform controls. Remove unused
  model constants. Live GLM measurements and serving admission stay unchanged.
  Multi-host stage lifecycle and real served qualification remain external
  prerequisites. No performance, energy or residency change is claimed.

- The Tessera worker loads the shared partition helper only for preparation
  (#2417, R1). Standalone help and atomic publication retain their standard-library
  contract without that helper. The tests use the shared owner directly.
  The ASCII escapes, final newline, source checks and partition rule stay unchanged.
- Tessera export setup derives architecture wiring, census inputs and partitions
  from profile capabilities and source headers (#2409, D50 item six).
  The plan writer can emit the metadata plan. The model dispatcher exposes it
  through a non-submit mode and replaces the fixed-count shard driver.
  The existing exporter, construction census and native runtime checks remain
  authoritative. This CPU work grants no runtime or serving qualification.
- The public MTP selector now enforces canonical admission through the existing shared owners (#2364).
  Held, unmeasured and outside-menu wires remain unavailable, even with an always-true native callback.
  Missing or unresolved actual structure, shape, regime M or required routing also removes the option.
  The selector retains rank-local scope, the activation build, the recipe and the table provenance.
  The allocator and fixed-selection tool supply this scope through explicit MTP arguments.
  Legacy emulation, anchor-only quality, exact budgets and bound wire export retain their existing contracts.
  Scope cache keys reuse the exact strict and lax JSON profiles from the digest owner.
  Body and MTP topology use the existing allocation lane protocol and shared scope owner.
  Synthetic card fixtures declare a compatible Qwen3 MoE profile. Missing-profile refusals remain strict.
  Constructor repairs remain unchanged. No identity seal is added.
- The PQ #2364 consumer keeps measured joint rows and fractional chords in one scientific quantity.
  Fused sums apply no extra gain or activation transfer.
  Independent MTP selection and export use whole-bit quality anchors with exact fractional wire receipts.
  Canonical logical candidates retain their bound chord records and report absent fractional stderr as null.
  Measured interval consumers refuse this absent uncertainty. Native, byte, scope, and serving gates stay unchanged.
- Canonical chord constructors refuse non-null fractional stderr, including zero (#2364).
  Anchor stderr does not supply fractional uncertainty evidence.
  Point-only menus remain available with null stderr. Scientific and byte checks stay strict.
- Canonical candidates restore their frozen provenance through the existing thaw path before scientific validation (#2364).
  Direct construction and dataclass replacement retain immutable metadata, original anchors, and both scientific currencies.
  The original joint validator stays strict. No identity seal is added.
- Mark the task-suite tests for the required layered gate (#2412, #2413).
  Ordinary runs report explicit skips with the recorded receipt.
  The task-suite gate must run the marked tests and must report zero skips.
  Select it with `-m task_suite` or `PQ_TASK_SUITE_TESTS=1`.
- The forward-recovery proof reader and campaign resolver call the existing
  digest owners (#2398, references #1301). The capsule tail hash calls
  `digests.bytes_sha256hex`; the source-record check calls the same bytes
  owner and the historical trailing-newline roster check calls
  `digests.text_sha256hex`. File reads, stat fences, limits, JSON parsing,
  campaign fields, geometry, identities, containment and refusal behavior
  stay unchanged. The streaming one-MiB reader keeps its own code.
- The forward-recovery chain tools drop the source-text scan test
  (#2398, references #1929). The test read the builder source and its
  string literals instead of consumer behavior. Rendered-launcher
  identity, invalid-field checks, read bounds and pre-load mismatch
  checks stay unchanged. No speed change is claimed.
- The streaming MXFP4 decode builds its lookup from the existing full signed
  table (#2401, references #1303). The lookup reads `mxfp4_widen.E2M1_VALUES`;
  code order, the positive zero at code 8, dtype, device, nibble order,
  scales, chunking, copy behavior and output stay unchanged.

- The allocator and PACT consume canonical v3 scope, qualified chords and
  reconciled class costs through their existing owners (#2364).
  Chord prices retain their actual scientific quantity without another transfer.
  Complete scope binds joint anchors to validated samples and actual coordinates.
  Producer provenance uses the existing development-mode check.
  Actual bytes, native admission and scientific gates remain unchanged.
  An immutable candidate index requires no active index change.
  Legacy metadata contracts retain the original producer interface.
  CPU fixtures do not qualify speed, served loss or serving.

- The shipcard model identity hashes its two canonical JSON texts through the
  existing text owner (#2384, references #1301). The canonical quant-config
  digest and the final canonical payload digest call `digests.text_sha256hex`;
  canonicalization, scope flags, model hashes, errors, auxiliary files,
  content checks and admission policy stay unchanged.

- The shipcard model identity hashes its raw config bytes through the
  existing bytes owner (#2394, references #1301). The config digest calls
  `digests.bytes_sha256hex`; the read position, call count, scope flags,
  model hashes, errors, auxiliary files, content checks and admission
  policy stay unchanged.

- The AURA checkpoint manifest writer uses the existing strict, indented
  UTF-8 profile (#2375, references #1301). Exact-input bytes, identity digests,
  unit order, source diagnostics and atomic publication stay unchanged.
  The writer adds no final newline.

- The AURA checkpoint producer comparison follows development-mode policy
  (#2375, references #1147, CEO D32). A dirty producer file prints the source
  provenance stamp and keeps the resolved commit. Certified mode keeps its
  original refusal. Git errors and timeouts still refuse in both modes.
  The change does not rewrite or recompute checkpoints.

- The allocator byte-budget test reuses the existing safetensors fixture
  writer from test_footprint (#2388, references #1929). Fixture bytes,
  tensor order, costs and semantic assertions stay unchanged. The
  per-unit-rate file drops its unused writer import.
- The allocator serve-constraints and serving-lane tests reuse the existing
  safetensors fixture writer from test_footprint (#2383, references #1929).
  Fixture bytes, tensor order, costs and semantic assertions stay unchanged.
  The serve-constraints introduction no longer cites the retired Gridbook
  lane policy as a live document.

- The native route-histogram test reuses the existing safetensors fixture
  writer from test_footprint (#2390, references #1929). Fixture bytes,
  tensor order, costs, route reports and semantic assertions stay unchanged.

- The PACT scope lives in the existing model profile contract (#2483, part of #2427).
  `ModelStructureSpec` carries a declared `pact` scope (dense end, band width,
  role TP splits, hidden streams); `ModelProfile` exposes it with layer,
  cohort, hidden and TP readers. `Glm5NextProfile` states the exact running
  cohort and layer count; `Qwen3Profile` resolves its declared fields with
  no GLM fallback. The adapter consumes only that public contract and
  refuses undeclared or inconsistent scope. GLM paths stay bit-identical;
  detection, name projection and production behavior stay unchanged.
  The CPU conformance evidence uses the existing isolated x86 interpreter with both reviewed dependency pins.
  All four PrismaBuild shards pass, including the GLM fused-owner case; ARCHITECTURE records the command and evidence.
  Both profiles now use one alias reader in the existing base owner.
  The unchanged helper gate passes without a baseline change.
  The current x86 command uses the isolated environment with the SDK pin from main.
- The PACT frontier profile cohort carries the measured-consumer contract
  (#2423, references #2427). `pact_cohort_from_profile` returns
  `local_prefix_rows="excluded"` and `input_contract="prefixed_514"` through
  the existing adapter, and `glm_paths_identical` rejects a cohort that
  lacks or changes them. Remaining cohort values, bands, TP rules, defaults,
  gates and the active index stay unchanged.

### Added

- **Bounded head walk measurement mode** (#1492, #1247).
  `tools/profile_stage_b_head.py --mode scoped-walk` walks a slice of the
  census roster with one explicit I/O worker count. It is read-only: it writes
  no head checkpoint, verifies no payload and synthesizes no render. One call
  holds at most 2,000 units. A sweep (`--sweep-start`, `--slice-units`,
  `--sweep-workers`) reads and validates the census-wide metadata once, in
  the baseline scope, and walks disjoint slices in one process from that
  state (`metadata_memo` on `load_measured_anchor_input`; a drifted file is
  read and verified again). The mode leaves the candidate overlay out of the
  inputs, so no wire payload is hashed. The whole sweep shares the 2,000-unit
  budget, and the first scope is a one-unit baseline. A guard thread stops the
  run when the NFS READ round trip exceeds `--stop-read-rtt-ms`. Both guard
  limits must be positive and finite. A `SIGTERM` stops the run with one
  report. The tool changes no default. The default worker count stays
  derived from the CPU reservation until a measurement supports a change.

- **Research-only finer-grained FIT pricing and packed reference wire**
  (#2329). Conditional single-block prices retain the complete baseline
  gradient; group residual shrinkage and a deterministic integer-byte solver
  select existing whole-projection parent fragments. The standalone wire
  preserves actual column rates, tables, scales and incoming window states,
  charging tags, framing, checksum and padding exactly. Its reader rejects
  invalid rates, noncanonical planes and nonfinite scales before allocating
  decoded weights; offset scratch is linear in block count. The recorded
  real-A8S CPU proof covers exact mixed-fragment reconstruction and boundary
  refusals, not in-domain held-out gain or a serving qualification. No
  production menu, pipeline default, serving pin or kernel changes.
  The research CLI binds declared inputs through the existing I/O engine
  and validates actual source/FIT shapes, row counts and current canonical
  producer admission. Encode and encode preflight never open HELDOUT
  payloads; CPU scoring verifies their own bytes and complete moments.
  Real L0 source/FIT preflight is recorded by action `fad7fa64ee59`. The
  later complete dense L0 gate `6b5288acd80c` matches the native control
  exactly at 25,191,435 bytes but has 3.475059 times its HELDOUT output-error
  loss. The routed L3 expert0-up contrast `509251c3242d` also matches exactly
  at 4,215,563 bytes, with 2.044249 times the control HELDOUT loss. Both
  early-site contrasts reject this recipe, not fine-grained allocation in
  general. FIT loss is already 4.13x/3.67x control: surrogate additivity fails
  before held-out generalization. L28/L44 are cancelled and unmeasured under
  CEO `dec-1006-223431-2bf3`, not deferred. No sampling interval or G3
  qualification is claimed.
  `pq_block_schedule_cost.py` consumes the real driver's flat row-major
  selection, candidate-order mapping and own-byte stamps, while retaining
  raw source-order input. CPU proof `63a5fc616b98` exercises both CLI forms
  and malformed order, digest, geometry and tag refusals. Stored bytes are
  exact; run, launch and shared-memory estimates remain conditional models,
  not measured traffic, occupancy or speed.

- **Explicit GLM projection owners.** The producer reads fused members from Tessera dense_ownership instead of a duplicate profile table.
  The source config separates KDA queries from standalone MLA queries.
  Existing pin-lift controls can select the actual bare router parameter owner.
  The producer keeps its stock forward and uses the existing input capture hook.
  DSA and MLA direct buffers increase memory_bytes through the shared runtime byte rule.
  Wire prices, default scope, source identities, pins, and serving cells remain unchanged.

- **Direct head arithmetic.** The DSA head screen uses FP32 inputs without the key projection's A8 activation quantization.
  The producer reads its decoded weights with the runtime's direct consumer helper.
  The publication cache keeps the head values in FP32 and charges the actual publication bytes.
  T-16 retains its folded BF16 weight arithmetic before the head cache cast.

- **Direct-consumer journal contracts.** Fresh publication, resume, and seed use the same activation contract as the direct-consumer score.
  The DSA head and MLA projection refuse quantized input observations in both development and certified modes.

- **Declared GLM ownership seam.** The GLM profile reads its fused owner and leaf mapping through the existing lane plugin.
  The Tessera lane keeps the authoritative runtime import. No boundary allowlist or runtime pin changes.

- **Shared-expert roster migration.** The quality population reads census names and the profile fused-owner seam.
  It no longer reads the removed duplicate fused-group table.

- **Behavioral coverage of persisted multimodal calibration provenance**
  (#2237, #2244, Refs #1921). All three visual-probe pickle writes are read
  back after the actual loader blends a partial real dataset with synthetic
  rows. Populated CPU forward/backward paths include a failed synthetic row
  and check nonzero visual Fisher; loaded-row composition stays independent
  of successful forwards. Duplicate synthetic-composition coverage and a
  fixture-only shutdown-call assertion are removed; no blend policy or
  numerical behavior changes.

- **Explicit selected-unit fresh calibration capture**. The shared cache
  identity accepts `unit_names` and declares `unit_scope="selected"`; writers
  and publication retain full-draw H/counts while requiring exactly the
  requested entries. Empty or unknown selections, missing requested units and
  implicit partial full-census captures refuse. Existing full capture identities,
  wire formats and serving kernels stay unchanged. The existing `--units`
  whole-group grammar flows through prep, selected collection, empty-range
  forwarding, join and coverage-checked reuse. Full source-layer traversal and
  full-draw row counts remain required. The immutable-provider safety refusal
  remains enforced; this is not admission of real automatic model capture.

- **Default-off uncapped calibration-row consumer**. The activation collector
  can hand each canonical shared input group to an owner before prefix capping,
  independently of built-in Hessian collection. The D42 Stage 1 research entry
  uses this seam for same-pass disjoint fit/held-out moments and checks their
  row-count sum against this forward's observed routing. It does not publish an
  ordinary production capture or qualify immutable source delivery.

- **Receipt-bound shared research publication and multi-layer actions**. Joins
  bind role counts, paths, digests, lengths and geometry to completed quantum
  receipts, preventing count redistribution or payload substitution. Adjacent
  prepared layers run in one GPU action with bounded per-layer moments and
  durable progress; explicit priority units publish first in the same capture.
  Moments drain after every layer even inside a multi-layer prepared range.
  Completed adoption rechecks scope, persisted split and scoring geometry and
  refuses a changed unit set or scoring prefix. A changed selection digest or
  retained-prefix budget replays the quantum instead of refusing or reusing it,
  and prints one `research_quantum_replay` line that names the changed fields.
  Historical census counts/maxima use D32 stamps with explicit deltas; actual
  routing, per-role projection agreement and tokens-times-top-k stay hard checks.
  `--mode preflight --device cuda` runs its toy control on the GPU; a CUDA-gated
  test pins that the quanta forward and accumulate on the device.

- **CPU-prep/GPU-quantum capture runtime stamps**. In default dev mode, runtime
  version metadata does not reapply an identity seal after source admission.
  Chain writers keep the canonical prep identity; completed initialization
  witnesses retain valid grammar and use the same identity stamp at join and
  finish. Calibration, unit scope/geometry, batch counts and own-byte integrity
  remain refusals; certified mode still refuses identity drift. Selected cache
  readers require coverage rather than output-scope equality, and every research
  quantum binds its actual fit/held-out coordinates before model load. The
  automatic source safety guard is unchanged.

- **Default-off CPU input/readset preflight for joint adjoint capture**
  (#2325). `--cpu-input-preflight` shares the calibration and source metadata
  startup owners, derives expected source phases from the actual meta-model and
  normalized job independently of the manifest, then checks every selected
  tensor span. It reports no capture or price and creates no generation,
  checkpoint or GPU allocation; normal CUDA and staged tier guards are unchanged.
  Missing phases, shard tails and ranges in another phase refuse before capture.
  The shared checkpoint-map and resident-head owners cover indexed/unindexed
  sources and LFM2/DSv4 head extras without skipping tensors. Source metadata
  reads use the same staged owner as the loader, not a pool fallback.
  Resume selection uses the actual adjoint-checkpoint session marker; a stored
  recovery capsule cannot re-enable a resumed forward pass. Bounded diagnostic
  specs are read under their own byte pin and bound to the actual draw before
  their lower-chain stopping boundary selects the source schedule.

- **Use shared digest owners in the research harness** (#2329, PR #2367).
  Byte, file, and tensor digests use their existing owners. Trial, bank, and
  schedule reports share strict JSON output in insertion order with one final
  newline. The wire retains its raw 32-byte checksum. Direct refusals retain
  their exception classes, messages, and check order. The duplication baseline,
  scientific results, production defaults, and serving contracts do not change.

- **Opt-in per-sequence/per-block signed attribution sidecar on joint AURA
  rows** (#1962). `make_joint_aura_entry` can publish a
  `sequence_attribution` block decomposing each projection over whole
  caller-owned blocks of the calibration draw — the streamed dense lease
  notes each streamed partition's batch block; the Stage B quantum retains
  the caller's capture-batch id per spill record and re-reads one window's
  captured rows through the spill's own reader lifecycle right after each
  single candidate's `project`, while that rendered delta is resident. The
  authoritative whole-draw fields and the stats contraction total stay
  bitwise authoritative (flag-off controls byte-identical); the price stays
  `0.5 mean_p(total_p**2)` over the whole draw, never a sum of per-block
  squares. The sidecar reports its own residual against the authoritative
  totals with a stated gate and denominator (default the issue's 1e-3
  relative against `fsum |w|+|a|+|m|`; zero scale must reconcile exactly),
  the attribution `c_i = .5 mean_p a_pi*total_p` against those totals, and —
  for at least two equal whole blocks covering the draw — delete-one
  leaveout prices `Nseq/(Nseq-k) * .5 mean_p((t_p-a_pi)**2)` with their
  jackknife standard error under a stated exchangeability assumption; a
  one-block or unequal-block scope publishes no standard error rather than a
  fabricated one. `validate_joint_aura_entry` recomputes the sidecar from
  its own parts for every reader, refuses any foreign `uncertainty_scope` a
  row publishes (a missing field keeps the legacy conditional reading), and
  refuses cohorts that cut a block or bind a foreign calibration.
  Each W/A/mixed projection separately reconciles with its authoritative
  component; a matching total cannot hide projection redistribution.
  Full block coverage, integer coordinates and recorded calibration/token
  geometry are validated before any reader uses the decomposition.
  Spill callbacks release their borrowed source and delta immediately and
  retain scalar block components only. Each read charges the existing complete
  host and device replay buffer envelopes before allocation, including pinned
  host capacity and aligned device staging; resource refusal prevents read
  and publication. Both collectors reject unknown
  candidates before capture and require complete sidecars only on selected
  rows. The full normalized selector binds the run identity; resume refuses
  a changed requested attribution surface even in dev mode. Missing selected
  probes refuse before unit publication. Existing rows stay
  probe-only; this is not the #1962 estimator fix, a repricing, or an issue
  closure. Gates: `tests/test_joint_sequence_attribution.py`, the spill
  sidecar tests of `tests/test_stageb_one_pass_spill.py`.

### Changed
- **Reuse the shared E2M1 value owner in the MXFP4 widening table**
  (#2380, Refs #1303). `mxfp4_widen.E2M1_VALUES` derives its positive half
  from `_E2M1_POSITIVE` in `nvfp4_activation_contract`, keeping the same
  public tuple type, the same code order, explicit positive zero at index 8
  and a negative half that negates only the nonzero magnitudes. Table bytes,
  widened weight bytes, carried scale bytes, geometry, route status and
  evidence strings are unchanged. The module introduction now describes the
  retired Gridbook MXFP8 dense lane in the past tense. No performance or
  serving qualification follows from this refactor.

- **Reuse the shared E2M1 value owner in RTN and MXFP4 source decode**
  (#2369, Refs #1303). `build_rtn_cache._nvfp4_round_rtn` reads its
  magnitudes, midpoint ties and maximum from `_E2M1_POSITIVE`,
  `E2M1_MIDPOINTS` and `FP4_E2M1_MAX` in
  `nvfp4_activation_contract`, and the MXFP4 nibble decode in
  `layer_streaming._apply_fp8_dequant_inplace` builds its value table
  from the same positive grid. The nested `torch.where` chain, the
  decode order, positive zero at codes 0 and 8, dtypes, scales,
  padding, ties and nonfinite behavior are unchanged. No format,
  default, gate, pin or artifact byte changes. No performance or
  serving qualification follows from this refactor.

- **Share the two GLM MTP capture file-byte recipes** (#2359, Refs #1301).
  Final-hidden manifests and MTP censuses use the existing
  `digests.indent2_json_file_bytes` owner: sorted keys, two-space indentation,
  ASCII escapes, strict non-finite handling, UTF-8 and one final line feed.
  Atomic publication, publish-once checks, native encoding errors, returned
  byte digests and census admission ordering are unchanged. The projection
  tool and its text summaries retain their own recipes.

- **Allocator partition and cost byte digests reuse the shared bytes owner**
  (#2355, Refs #1301). The rank-partition manifest reference and the
  measured-runtime cost payload integrity comparison in `allocator.main` call
  `digests.bytes_sha256hex` instead of inlining
  `hashlib.sha256(...).hexdigest()`. Bytes, digest values, comparison order,
  refusal messages and ownership are unchanged: the partition reference still
  authenticates the exact rank manifest bytes the recomputation consumes, and
  the cost comparison still runs on the owned bytes before `pickle.loads`,
  ahead of any parse or publication. The remaining `hashlib` use in
  `allocator.main` is the assignment-payload dedupe digest, which hashes a
  canonical JSON string, not artifact bytes.

- **Reuse the shared forward-KL owner in final-vocabulary scoring** (#2334,
  Refs #1303). The final `_student` scoring path of
  `tools/measure_vllm_full_kl.py` now calls
  `prismaquant.kl_fisher.forward_kl_per_token` instead of inlining the
  `teacher.exp() * (teacher - student)` sum over the vocabulary axis.
  Operand order, dtypes, broadcasting, the last-axis reduction, the
  non-finite refusal and every published field are unchanged, as are script
  and module bootstrap. The all-position estimator keeps its own convention
  (`_position_kl` / `_student_all_positions`: top-K support plus one tail
  bucket); final-position full vocabulary is not all-position top-K plus
  tail.

- **Caller-declared journal producer seals** (#687, CEO
  dec-1005-212354-dc9d, D32). `prepare_journal` defaults to no seal fields;
  the joint qualification journal declares only `implementation_sha256`;
  the campaign checkpoint and stream journal declare only the PrismaQuant and
  encoder source hashes. Producer-only drift stamps one `[DEV-MODE]` line
  and reuses stored shards without rewriting their manifest or digest.
  The campaign wire-reader wrapper defers only `encoder_source_sha256` and
  still calls the real cached-unit verifier with that stored field; wire bytes,
  every other identity field, filename and measured length remain checked.
  Mixed mismatches, every comparability field and byte-integrity checks still
  refuse, as does certified mode (`PRISMAQUANT_DEV_MODE=0`). This retires
  the freeze-the-tree / `--seed-checkpoint` workaround for producer-hash moves
  only; `--seed-checkpoint` stays the path for real correctness changes,
  and every comparability field still refuses.

- **Stage the exact public Tessera master D13 pin** (#2262): serving JSON,
  all serving/development constants and the complete reviewed answer bind
  `2dbac1910c88254d9c6391f02a34c4b07e516803` / contract v56
  (`47f180ef…d0aed78`), v11 run-table coverage and four extension rows.
  Root graph-receipt v2 retains all nine scope fields. The existing
  fingerprint, provisioning, lane, export and release refusals remain enforced.
  Route-census scope checks use the existing authoritative scoped-schema set
  at both entry points, preserving flat-row refusal and valid scoped replay
  under v11 rather than comparing with the legacy v10 label.
  Isolated installed-package evidence is not performance, compiled-cell
  or ship-card qualification; D13 landing still needs the fresh public-source
  packet and CEO review. No private `608bb` result is transferred.

- **D32 dev-mode metadata stamp-and-continue** (#2302). The existing central
  default (anything except exact PRISMAQUANT_DEV_MODE=0) now also governs
  capture metadata owners, source provenance consumers and paired-trade
  producer/arithmetic/source metadata. Consumers keep stored data without
  archive, rehash, recompute or proof barriers and mark dev results in-band.
  Row-to-row probe coordinates, seed/draw/token/noise alignment and KL units
  (temperature, normalization, distribution) remain mathematical refusals in
  both modes. Own-byte digests, strict publishing fingerprints, same-held-fd
  mutation, finite complete numeric dimensions and wire/kernel/resource safety
  remain. No prices are rescaled or fabricated; no serving qualification is
  claimed.

- **Verified-capture capacity test isolation** (#2308). Scope its global
  `os.open` refusal sentinel to the loader call so failed-only pytest temp
  cleanup does not trip the sentinel after the safety assertion has passed.
  The real pre-open capacity refusal is unchanged.

- **Mixed-rate COST_UCB_Z and paired rate-trade validity** (#2282, Refs #2281).
  The existing allocator accepts an explicit `--cost-baseline-assignment`;
  mixed-rate UCB requires it and matched joint AURA source/probe/currency
  evidence instead of raising NotImplementedError or dropping uncertainty.
  The exact group fold retains complete combinations through common-probe
  paired pricing. One paired arithmetic owner also aggregates projections by
  routed expert and refuses changes exceeding half the signed layer delta;
  exactly half is allowed and nonzero cancellation to zero is refused.
  Signed differences keep their global KL Fisher normalization and are not
  clipped or rescaled; final candidate totals alone clamp nonnegative.
  Legacy unpaired stock/uniform paths and zero-z prices are unchanged,
  apart from the guard on explicitly compared trades. This source correction
  does not establish corrected campaign prices or close measured P0 #2281.

- **Bounded sorted-JSON consumers route to their existing exact
  `JsonProfile` recipes** (Refs #1301, follow-up to the byte-hash routing).
  79 selected scopes / 86 sorted-`json.dumps` calls across 56 package
  modules now call the profile they already spelled out by options —
  `DIRECT_ASCII_SPACED_LAX`, `DIRECT_ASCII_SPACED_STRICT`,
  `DIRECT_ASCII_LAX`, `DIRECT_ASCII_STRICT`, `DIRECT_ASCII_INDENT2_LAX`,
  `DIRECT_UTF8_LAX`, `DIRECT_UTF8_STRICT` or `DIRECT_UTF8_INDENT2_STRICT` —
  via `.text`, `.encoded` or `.sha256` exactly where the previous expression
  consumed text, bytes or the digest of those bytes. Each route is
  byte-identical to the recipe it replaces: the encoder options and UTF-8
  encoding step are unchanged. Flags, bytes, refusals, evaluation order,
  newlines, prefix truncation and source-identity contracts are unchanged,
  and `digests.py` itself is untouched. Five matching scopes in two
  protected loaders stay on raw `json.dumps`: the campaign container test
  loads `prismabuild_progress.py` by run path without package context (its
  four scopes), and the host row profiler loads `io_spans.py` by file-spec
  without package context — `tools/pq_profile_source.py:18-25`,
  `tools/pq_row_profile_observer.py:26` — so `ReadRateReporter._emit` keeps
  its direct recipe; no fallback loaders were added. `dsv4_campaign_completion.py`
  extends its existing same-tree file-spec digests owner binding with the
  two UTF-8 profiles symmetrically in package and no-package mode. The
  committed per-scope census in
  `docs/audits/digest_site_census_pq1301_2026-10-04.json` retains all 501
  historical rows and names the 334 remaining gated scopes; #1301 stays open.

- **The tools serializer correction retains only proved routes** (Refs #1301,
  correction #2320). The original selection was 58 calls in 45 scopes across
  36 files. The CEO-authorized cutoff retains 40 proved profile calls in 32
  scopes across 24 files and restores 18 individual unproved expressions to
  their exact original recipes as explicit #1301 residuals. Proved mixed-scope
  neighbors and all eight actual lightweight bootstrap repairs remain.
  Source-routing/profile-echo assertions were removed in favor of actual
  consumer outcomes and the supported-context launch matrix. Version-three
  evidence distinguishes boundary and unqualified CPU-fixture behavior from
  whole-producer/native qualification; all old red runs and late diagnostics
  remain preserved. The owner-generated gated-scope baseline is 309, a net
  reduction of 25 from the original 334, with residuals explicitly restored.
  The staged standard-library worker and unrelated recipe families stay
  unchanged. #1301 remains open; no speed, serving or full-suite claim.

- **Raw byte-hash constructors route to the digest owners** (Refs #1301).
  83 raw `hashlib.sha256(...).hexdigest()` constructor sites across 47
  package modules now call the existing `prismaquant.digests` byte/text
  owners (`bytes_sha256hex`, `text_sha256hex`), with the base-bound census
  of 485 gated digest scopes (434 load-bearing, 51 ephemeral) recorded in
  `docs/audits/digest_site_census_pq1301_2026-10-04.json`. The slice leaves
  413 gated scopes as explicit remaining work. One migrated module loads
  without a package context — `dsv4_campaign_completion.py` (the campaign
  waiter's file-spec load) — and binds the same named owner functions by
  loading the same tree's stdlib-only `digests.py` by file path; its three
  census rows carry that binding note.

- **Three protected digest contracts stay on raw `hashlib` and are
  excluded from owner routing** (Refs #1301). The packaged KDA capture
  kernel self-source identity (`kernels/kda_chunk.py` `source_sha256`,
  sealed as `identity.source_sha256` in
  `kernels/kda_chunk_qualification.json` and seal-checked by
  `glm_kda_capture_kernel` admission), the deployed single-file
  `container_runtime_identity.py` contract (no-package bootstrap and runpy
  loads; the gold producer byte set lists the file without `digests.py`),
  and `joint_prewarm_phases.py` (executed by file path without a package
  by `experiments/glm_data_manifests.py`) keep raw `hashlib`; their four
  census rows are reclassified as retained protected contracts with
  file:line witnesses, and their four ratchet rows return to the baseline.
  Editing any of these files is an identity or contract change, not a
  mechanical route. No encoding, wire, default or pin change is part of
  this slice.

### Fixed

- **The nested-rotary meta-skeleton test owns its import process** (Refs
  #2279, a residual exposure of #2276). `tests/test_dsv4_nested_rotary_init.py`
  uses the existing `own_process` marker when it shares a pytest session, so a
  native `transformers.models.deepseek_v4` import in an earlier test cannot make
  `register_deepseek_v4()` refuse. On main, running
  `test_streaming_text_only_wrapper_config.py` first failed this test; the
  reverse order and the test alone passed. The regression runs the real
  unsupported-configuration predecessor, this test and the native-import
  refusal control together. Production registration and its native-module
  refusal are unchanged. The other exposures listed in #2279 are not fixed
  here, and the issue stays open.

- **Bound paired-rate-trade diagnostic retention to summaries outside the
  emitted assignment** (#2286). Menu, applicability and diagnostic-trace
  records keep a bounded summary per priced trade -- priced scalars, refusal
  verdict, per-group and per-expert means without per-probe arrays, and the
  canonical digest binding the exact full trade -- instead of storing every
  complete trade. Only the emitted assignment carries the full paired arrays
  and per-expert breakdown; refusal stdout prints the summarized rows.
  Pricing arithmetic, UCB hedging, joint sample/currency/format validation,
  expert-dominance refusal and reproduction diagnostics are unchanged.
  The command tests read the complete selected evidence and the actual refusal text.
  `tools/paired_trade_report_proof.py` measures the real allocation and report paths with synthetic samples.
  The workload uses the routed dimensions from `GLM-5.3-Flash-BF16`.
  The proof records process peaks, output bytes, profiles, and assignment arithmetic.
  It makes no scientific quality, serving, or GPU claim.
  The proof tool now uses the public PrismaBuild client to reserve and
  publish its archive as a retained output batch. It uses the existing
  exact-byte digest and sorted, spaced, ASCII JSON profiles. The allocator
  command profiler has a domain-specific name. Both boundary baselines and
  their scanner rules are unchanged.

- **Correct readiness in the research preflight** (#2329). The census now
  reports not-ready if any bank rung lacks canonical allow status. It keeps
  all admission decisions and names the missing `rung_admission` prerequisite.
  CPU CLI action `53433514eeee` passed the all-allow and valid R768-hold cases.
  Both cases left HELDOUT payloads unopened. The test does not encode, score,
  or capture a model.
- **Authenticated locator aliases survive both real acquisition merges** (#2195,
  PR #2253). The existing control owner derives one locator-independent request
  control identity carried through actual row loading, rendering and journals.
  Checkpoint and scalar-payload joins compare those controls and all real
  scientific inputs before stamping only raw request provenance. Own-file SHA
  reads, schedule/source/scope, calibration and numerical refusals stay strict.

- **Acquisition input byte checks participate in the seal ratchet** (#2195,
  PR #2253). The torch-free input owner is scanned, with only its actual
  own-byte digest check allowlisted as integrity. Injecting a new recorded
  producer wall is a causal regression, not an unscanned escape.

- **Acquisition producer identities and locator spelling follow development mode**
  (#2195, PR #2253). Only recorded live pins and export/grammar source digests
  stamp and continue; source-state schema and actual request/cost, rate, shape,
  atomic scope and calibration correctness still refuse. Planning and manifest
  validation authenticate controls through the existing input owners before
  separating locator spelling, without trusting a declared digest alone.

- **Authenticated acquisition requests reach complete per-row execution and
  strict merge through the existing planner** (#2195). Each active atomic
  cohort keeps the original request/cost/run/probe identity; deferred cohorts
  remain explicit and produce no zero-work jobs. Whole request/cost bindings
  precede captures in the torch-free staged readset, with a bounded fenced
  metadata memo instead of repeated whole-cost reads. Submission and merge
  require disjoint complete active coverage, exact requested scalar cells,
  source proofs and common regime settings. Both merges use authenticated
  checkpoint menus to bind the exact deferred family domain, retaining real
  unrequested families and refusing fabricated extras. The original raw joint evidence,
  normal opt-out paths and production/scientific qualification gates remain;
  no Fisher price, pin/default change or served artifact is inferred.

- **Bind admitted produced output without input residency** (#2339). Queue
  discovery lives in the allowlisted staged-lease seam over the same sealed
  generation; the campaign container carries its launcher-owned
  `PRISMABUILD_QUEUE_ROOT` and refuses spec forgeries. Legacy residency-map
  discovery stays with PB.
  Live-attempt, declared-template, own-byte and output-budget guards remain
  unchanged; no fake map or input staging declaration is introduced.

- **Sealed io buffers open where this interpreter's os lacks
  `memfd_create`** (#1896). A portable CPU venv (`pq-cpu312` on dl380g10)
  has no `os.memfd_create`, and every io engine stream entry died on the
  missing attribute (PrismaBuild action `ded8698fa4d6`). `SealedBuffer` now
  opens through `io_engine._create_memfd`: `os.memfd_create` when present,
  otherwise the runtime libc's `memfd_create` wrapper (glibc 2.27+,
  musl 1.1.20+), with the MFD flags resolved from their ABI-fixed Linux
  UAPI numbers and a named `OSError` when the runtime libc has no wrapper.
  The seal-and-verify guard is unchanged; no default, pin, wire, GPU or
  serving claim.

- **The io engine resolves memfd seal constants without CPython build-time
  fcntl names** (#1896). Portable interpreters whose build headers predate
  glibc 2.27 expose only part of the fcntl seal surface, and reading
  `fcntl.F_SEAL_*` at import crashed the engine there, failing collection of
  every capture and calibration test file that imports it (qualified CPU venv
  `pq-cpu312` on dl380g10; PrismaBuild actions `be3dd1332259`, `5e848d8252be`).
  The numbers now resolve from the Linux UAPI values they denote, ABI-fixed
  since kernel 3.11, and the kernel stays the authority: each seal is
  attempted through fcntl and read back through `F_GET_SEALS`. The
  original-material delivery witness reads seals through
  `io_engine.kernel_seal_bits`, so one home owns the seal grammar. No guard
  moved: a buffer still refuses unless the kernel reports all four seals; no
  default, pin, wire, GPU or serving claim.

- **A joint plan that cannot name its campaign chain is refused by name, at
  admission, before any device** (#1293). `load_joint_anchor_plan` admitted a
  plan with no `inputs` block, and the `prepare` GPU action then died on a
  bare `KeyError: 'inputs'` after the projection prewarm had already
  allocated — preserved in the #1293 non-release pilot's run-01 S3. The plan
  grammar now requires the campaign chain `inputs` mapping, shape-checks
  every bound head-walk key without reading behind the binding, and requires
  the canonical capture binding; the anchor intake names its missing chain
  keys in one refusal; the standalone synthesis census read refuses a missing
  binding by name. A Stage B quantum plan that binds a subset (#1024) still
  loads in both modes; its test explicitly sets the empty subset instead of
  depending on the imported fixture default. No gate weakened, no wire, codec,
  numerical or GPU claim; the named refusal moves the run-01 S3 failure from
  minutes into a GPU action to a plan-load ValueError naming the absent key.
  Policy-refusal tests start from the canonical shape-only plan and name each
  probe, token-scope, temperature and activation-clipping refusal, so an
  earlier missing-input error cannot hide those guards. The standalone
  `synthesize` command also takes this complete plan and its canonical capture
  binding, although it loads neither model nor capture payloads.

- **Real codec CPU fixtures retain their full acceptance at bounded geometry**

- **The RTN FP8 helper keeps finite FP16 zero and tiny rows finite**
  (#2352; `build_rtn_cache._fp8_round`, parent #1303). The 1e-8 max-abs
  floor and the resulting `/448` scale underflow FP16 to `0.0`: an
  all-zero row divides `0/0` to NaN, and a nonzero tiny row divides to
  ±Inf, which the finite-only E4M3FN cast turns into NaN — every
  element of the row comes back NaN either way. FP16 inputs now
  round-trip through FP32 with the dequantized result cast back at the
  output boundary; FP32 and BF16 keep the original arithmetic and
  byte-identical outputs (pinned by
  `test_fp8_round_preserves_fp32_bf16_recipe_bytes`), and the default
  BF16 cache recipe is unchanged.

- **`file_sha256hex` refuses a zero read count instead of hashing no bytes**
  (#2344). `read(0)` never advances, so a `block_size=0` caller — including
  every zero spelling `read` coerces (`False`, any `__index__` zero) — got
  the SHA-256 of no bytes: any **nonempty** file silently returned the
  empty-input digest `e3b0c442…855` instead of a digest of its bytes (an
  empty file already matched that digest, which is how the bug could hide).
  The owner now
  raises a `ValueError` naming `block_size` before the first read, whether
  the file is empty or not. Everything else is unchanged: positive sizes
  stream in that many bytes per read, `-1` and `None` read the whole file,
  other negative sizes keep `read`'s own refusal, non-integer sizes keep
  the `TypeError` from the integer coercion, and a missing path or
  directory still raises what `open` raises before any read-count check.
  Regression tests pin the refusal on real empty and nonempty files plus
  the read-all and type-refusal boundaries.

- **Forward-split relaunches reuse declared producer seals** (#2342). Full
  retained/running bind identities now use the existing chain-resume
  classifier before adopting the original identity for exact-session rebind.
  Calibration/draw/probe/seed/temperature/partition and unknown fields still
  refuse, including an unknown null field; certified mode still refuses
  seal drift. Session hashes and prep/entry bytes are never re-keyed.

- **Restore the shared owned-byte digest comparison after the #2283 port.**
  `read_bound` again routes the acquired-byte hash through the hard `same`
  comparison before memoizing, preserving `owned bytes: identity mismatch`.
  A stat-fence drift cannot adopt changed bytes; dev-mode provenance stamps
  never waive this byte-integrity check. Existing consumer tests stay unchanged.

- **The seed wire filename refusal in `tessera_materialization.finalize()`
  precedes the source read** (#2310). The check that a seed receipt names its
  unit/rung destination is now one helper, `_require_seed_wire_filename`, which
  `_link_seed_wire` and `finalize()` both call; `finalize()` runs it before
  locating or reading the source wire, so a misnamed receipt is refused without
  touching its bytes. A regression builds a misnamed seed receipt and checks that
  nothing reads or links it. No bytes, formats or gates change.

- **Retired interpreter receipts remain history, not active attestation**
  (#2222). The SDK3 entries on `dl380g10`, `sparky` and `sparklina` stay under
  `retired_interpreters`, separate from active placement attestations. The
  inventory reader validates both mappings and refuses active/retired overlap.
  Following D32, missing or retired host attestation now uses the existing
  `dev_mode.seal_check`: default dev mode stamps `[DEV-MODE]` and continues with
  the spec; only `PRISMAQUANT_DEV_MODE=0` keeps the certified refusal. Continuing
  never marks a retired receipt active or qualifies its runtime. Active SDK4
  and ROCm entries, inventory shape, container/device and instruction-set
  checks, actual runtime compatibility and safety remain distinct; no runtime
  probe, re-pin or new identity gate is added.

- **The collection-time PrismaBuild import guard catches dynamic imports**
  (#2299). It scans importlib and built-in import calls and decorator arguments,
  including named local helpers, without entering uncalled test bodies; the
  source-family test documents its static-analysis limit and rename upkeep.

- **Seed wire filenames are checked before linking** (#2273). Campaign
  adoption and selected-wire materialization refuse another coordinate's
  filename without leaving a stray link; a real resume pins registration
  before receipt reads. The architecture note documents off-menu evidence
  and the per-cache roster's pre-publication registration contract.

- **The real DeepSeek V4 model-walk export gate owns its import process**
  (#2276). It uses the existing `own_process` marker when sharing a pytest
  session, so an unsupported native AutoModel configuration in another test
  cannot select the gate's model implementation. The regression runs that
  real unsupported-configuration predecessor, the complete export gate, and
  the native-import refusal control together. Production registration and
  its native-module refusal are unchanged.

- **PrismaBuild test imports are owned by tests or fixtures, not collection**
  (#2265). The control-artifact tests import their installed CAS inside the
  two tests that use it; the optional prefill decomposer harness imports its
  candidate inside an explicit graph-detaching module scope. The three readset
  source consumers also own pinned source scopes, including when pytest
  preloads the installed package through its public PrismaBuild bound plugin.
  Every test module now explicitly owns an initially detached import graph,
  which also covers candidate, published-runtime and imported helper families
  without adding source prerequisites to unrelated tests. The old graph and
  parent edges return at module teardown. A syntax-tree regression discovers
  origin-refusing resolvers across tests, follows imports, fixture parameters
  and named calls, and runs a representative of every family with the real
  public plugin preloaded. All origin assertions remain unchanged.

- **Required domain imports and malformed-header consumer refusals stay
  visible** (Refs #2260, bounded child of #1303).
  `tests/test_container_qname_owner_1303.py` imports
  `prismaquant.measure_quant_cost` directly instead of
  `pytest.importorskip`, so a broken required module now fails the suite
  instead of silently skipping the per-expert name-decomposition check;
  that test's assertion is unchanged. `tests/test_safetensors_reader_owner.py`
  adds one parametrised consumer test that overwrites its small real
  checkpoint shard with genuinely malformed containers — a file shorter
  than the u64 length prefix, and a real 8-byte prefix naming an impossible
  header length — and asserts each consumer's actual visible error
  boundary: `footprint._read_safetensors_header`,
  `artifact_completeness._read_safetensors_header` and
  `autoscale._shard_resident_bytes` propagate the container grammar owner's
  named `ValueError` from `prismaquant.source_read_plan`, and
  `pipeline._safetensors_parameter_count` wraps it as
  `cannot inspect safetensors shard ...` with the owner's reason intact.
  Test-only: no production behavior, guard, default, format, pin or
  geometry change; the added coverage does not claim these readers were
  broken before.

- **Test cost: repeated in-process work runs once; two stale consumer
  fixtures move from 32 rows to the shared fixture's `OUTPUT_FEATURES`=8**
  (#1929). The prefill dry-run tables memoize their pure seeded mandatory-set
  build and legal-domain enumeration per family; three pairs of plan-driver
  tests read one real driver run instead of two identical ones; the scalar
  staged publication campaign is shared by its two read-only consumers behind
  byte-digest guards; and the two stale row-startup consumer fixtures now
  build `stream_fixture`'s `OUTPUT_FEATURES`=8 rows, with the wrong-shape
  admission slice derived from `OUTPUT_FEATURES` so the refusal cell stays
  live. The launcher contract tests share one stdlib-only inspection
  fixture. Ten old test IDs become six merged or renamed IDs; every test
  assertion and execution logic is unchanged, and every removed execution's
  assertions survive on the execution that remains. Production defaults,
  pins, guards, timeouts, markers, production geometry and the fixture-
  owning file's geometry are unchanged; line coverage and encode-regime
  coverage are not measured here and are left to the shared merged batch.

- **The joint prepare's omitted `file_hash_workers` takes `prepare_cache`'s
  own derivation** (#1382). One derivation for one knob:
  `prepare_cache`'s `file_load_workers` default and the joint plan's omitted
  key both resolve through `default_file_load_workers()` — the measured PWC
  load curve (4 threads), bounded by the PB-assigned CPU affinity the same
  way an explicit plan value is fenced — and the regeneration tool's one
  head intake resolves the key through the same resolver. A plan without the
  key no longer loads render files serially, and an explicit plan value
  still wins with its existing refusals unchanged.

- **A producer-source checkpoint mismatch names the first file that moved**
  (#2218, option (c); the writer itself was fixed separately in #2226). The AURA checkpoint
  manifest records one pass over the producer tree — its aggregate digest
  and per-file digests — beside the sealed identity, and when a resume's
  `producer_source_sha256` seal mismatches, the refusal names the first
  differing relative path with both per-file digests; when the per-file
  maps match but the aggregate differs, it says the tree changed between
  the identity's digest and the manifest write. The record is diagnostic
  only — no gate reads it — and manifests written before it existed keep
  the plain refusal.

- **An expert-projection refusal no longer writes into the model source, and
  the carried block's producer entry is described as it is** (#2243).
  `request_expert_projection` ran both inside-source refusals after writing
  the stack-plan request file, so a call with an output path inside the
  checkpoint was refused only after a file had landed in the source tree
  every later source identity hashes; the refusals (output parent and
  digest cache directory) now run before any write, and an output path
  inside the source is refused even when the producer advertises no digest
  cache. Nothing else moved: the returned projection still carries the
  caller's `source_digest_cache_use` record, and because
  `carried_projection` embeds that answer verbatim under `producer`, the
  docstrings and `docs/ARCHITECTURE.md` now say the carried block's
  producer entry carries this one caller-side key instead of restructuring
  the block.

- **The census caller hands the producer's projection a stat-bound digest
  cache** (#2229, Refs RobTand/tessera#790). `request_expert_projection`
  passes `--source-digest-cache` whenever the selected producer's CLI
  advertises it, using a stable directory beside the projection output
  (caller-overridable, created if absent, refused inside the model source
  and when the override is an existing file), so a repeated projection of
  unchanged checkpoint bytes reuses the producer's recorded shard digests
  instead of re-hashing the whole checkpoint. The producer's
  `source_digest_cache` receipt stays the producer's own in the answer and
  invalidation on changed bytes is unchanged; every answer also carries the
  caller's `source_digest_cache_use` record (`used`, `reason`), so a
  producer without the option is named and a consumer never branches on the
  receipt's schema. An explicitly requested cache with such a producer
  refuses by name. No real census row or idle-GPU measurement is claimed;
  the tessera#790 acceptance evidence stays open.


- **Remaining safetensors header decoders and in-file name grammars read
  through their owners** (#1303). The model-profile validator's header check
  and the `chain_roll_bench` / `stage_fed_demonstration` host tools use
  `source_read_plan.read_safetensors_header` instead of restating the 8-byte
  prefix + JSON grammar; well-formed files are byte-identical and corrupt
  prefixes now refuse with the owner's named bound messages. The per-expert
  cost-name grammar (`measure_quant_cost._PER_EXPERT_NAME_RE`) and the
  streaming prefix/layer grammar (`streaming_initialization._prefix_layer_index`)
  are each stated once, and the fused kernel module's unused local
  `_FP4_E2M1_MAX` literal is removed in favor of the activation-contract
  constant. Regex acceptance, spans, wire, defaults and served paths are
  unchanged; the broader domain-numerics census stays open. The benchmark
  imports the header reader only for manifest spans, so historical child
  trees need not provide it merely to import the standalone tool. Coverage
  retains real bytes, spans, grammar results and malformed-header refusals,
  not source-layout or forwarding assertions.

- **Remaining write paths refuse non-injective cache filename sets at the
  write open** (#2231, parent #2219). The shared
  `require_injective_cache_filenames` check now also covers the MTP append —
  over its own `(qname, canonical format)` coordinates and, when it opens a
  real cache directory, over the union with the manifest keys already in
  the cache, before the stale-scope prune or any shard is written — and the
  joint aura head walk's per-owner
  check groups names by the resolved owner root instead of the directory
  string as spelled, so two rows naming one cache directory through
  different spellings are checked as one cache. The same check now shares
  the walk's measured-format set, so costed but unmeasured aliases do not
  refuse distinct render reads (#2231 item 10). The residency docstring now
  states the prefetch recheck honestly: it runs on every call that has
  something to load (the `if not keys: return 0` early return skips it),
  not on every call. Still unrefused at write time, and out of scope here:
  the campaign encode lane (`tessera_campaign.py`), the `.tessera` wire
  family (`_wire_path`), `weight_session.py` source snapshots, a
  `__` separator check on format names, and the packed and dense opens'
  union with the manifest keys already in a directory they append to.

- **Non-injective production cache filenames refuse at open** (#2219).
  `_cache_weight_filename` mangles `.` → `_` and `/` → `__`, so distinct
  qualified names can share one shard leaf while the stored payload is the
  bare tensor. A shared `require_injective_cache_filenames` check now refuses
  — naming both coordinates and the colliding filename — wherever a cache
  directory is opened for a model's rendered coordinate set: the dense fill's
  render-identity destination check (via delegation), the packed-expert
  fill, the streaming dense fill, and every residency read — `prefetch`
  rechecks the whole manifest on every call that has something to load
  (keys can be popped after fill), and the lazy `get()` load path carries
  the same check behind a size memo — plus the joint aura head walk's
  per-owner render reads. Injectivity is filename-level over
  `(qname, canonical format)` coordinates: an alias pair at two different
  formats names two different files and is admitted (#1859); the same pair
  at one format refuses. The mangled filename spelling is unchanged.

- **Campaign, wire and source snapshot writes refuse filename collisions**
  (#2231). Campaign rendered-file checks retain the whole manifest; wire-file
  checks include only the wire-owning roster and each new coordinate, allowing
  dense-only aliases that never wrote a wire. The roster records successful
  publications and reserves resume/seed coordinates before links or receipt
  reads, even when their rendered-manifest entries are absent. Batch admission
  and the ordered writer still refuse real collisions before either write.
  Concurrent producer admission and ordered publication install one shared
  wire roster; a stale bootstrap cannot discard an already published owner.
  Selected-wire materialization reserves the complete group's coordinates
  before any seed link, resumed wire read or fresh publication, including a
  missing-first coordinate whose wire name aliases a later seeded selection.
  The unchanged direct writer in `experiments/pq237_joint_aura_streamed.py:243-260`
  is not covered by these campaign and materialization guards.
  Packed expert appends include existing dense keys.
  Disk-backed weight sessions check the complete snapshot roster before
  capture or reuse and share the existing cache leaf helper. Every refusal
  names both coordinates and their shared filename. Cross-format name aliases
  remain legal when they name distinct files; all on-disk spellings are
  unchanged.
  The memory-only multi-token prediction append now has an entry-point test
  for same-format refusal before rendering and cross-format admission; this
  pins existing behavior rather than changing it (#2231 item 13).

- **Campaign filename indexing is not adopted** (#2231 item 15).
  A filename index would need to own every cache-manifest mutation and
  failed-publication lifetime, not just the campaign writer, to retain the
  rendered-file refusal set. No before/after measurement establishes a material
  cost here, so the index is not worth adding for this low-priority follow-up;
  complete destination checks remain and no speed improvement is claimed.

- **Projected preparation validates effective CUDA allocator settings**
  (#2039, PR #2247 hardening). An in-process allocator setter followed by an
  empty-string reset can leave expandable segments active while the snapshot
  configuration text is empty. The existing default-native guard now also
  requires the qualified PyTorch 2.11 effective defaults: expandable segments
  off, signed SIZE_MAX split bound, zero garbage-collection threshold and the
  complete all-zero rounding table. Public C10 getters also read the sticky
  large-segment and nonsplit-rounding sizes the snapshot omits; both must be
  their derived 20-MiB defaults. The tiny read-only bridge reuses the existing
  locked Torch extension loader/cache and packages its C++ source; no CUDA
  kernel, Torch rebuild, state reset or serving-pin change is involved.
  Missing, nondefault or unpriced fields refuse before source reads. Real
  setter/reset regressions run in isolated PB child processes so global
  settings never leak into the suite, including the large-segment-only reset.
  The full-pass reservation, source ownership, four credits, cap, ordering,
  lifetimes and cancellation are unchanged; no tighter or performance claim.


- **The head-wait credit control holds the second launch's token** (#2039).
  The control test for mid-wait credit reaping decided which launch's
  completion token to hold after the event had already appended itself, so it
  held the FIRST launch's token, stranded that credit for the whole test and
  could never reach the mid-wait admission it exists to prove. The decision
  now happens before the append. Its read-order check also assumed FIFO worker
  starts, which the shared two-thread executor does not promise. The control
  now forces a legal out-of-order read and proves coordinator CUDA launches
  remain ordered, while retaining the mid-wait fifth-read credit gate.
  Control-only; production admission and lifetime semantics are unchanged.

- **Projected preparation reserves comparison and allocator residency**
  (#2039, PR #2247). The previous element-count term was the full-size bool
  comparison-mask allowance, not staged-copy byte pricing: staged bytes were
  already charged by the private-byte bound. Replacing that term removed the
  mask allowance and charged retained verdicts/settle storage as logical bytes
  even though the guard reads CUDA allocator segments. The opt-in path now
  conservatively charges a fresh native allocator segment for every staged
  copy, comparison mask, reduction workspace, retained verdict and per-device
  settle stack across the entire pass, without assuming cache or stream reuse.
  Nondefault or unavailable allocator settings refuse before source reads;
  serial preparation is unchanged. Actual CUDA reserved-growth and early
  refusal regressions replace the arithmetic-only reservation assertion.
  Four credits, the finite private-byte cap, authentication, ordered refusals,
  lifetimes and cancellation remain intact. The historical device timing is
  an unqualified same-host screen, not a host-copy causality, residency saving,
  energy/work-per-joule or campaign qualification claim.

- **Ordered projected preparation reaps freed credits during the head wait**
  (#2039). The coordinator held all four credits until the ordered head's
  staging wait returned at the loop top, so a launch whose completion event
  fired mid-wait idled both read-pool workers instead of admitting the next
  source read. The head wait now reaps completed events and admits through
  the same finite-credit rule before each bounded result poll. Admission
  order, the serial-fallback exclusion, private-buffer lifetimes through
  asynchronous completion and cancellation drains are unchanged; the paired
  device timing comparison is a separate measurement.


- **Real codec CPU fixtures retain their branch/assertion acceptance at bounded geometry**
  (#2213, parent #1929). Streaming/resume controls keep three units, two layers,
  private source/capture identities, full-width Hessians and every existing
  byte/refusal/window assertion while encoding fewer output rows. The preserved
  acceptance is branch and assertion coverage, not encode-regime coverage: at
  eight output rows several derived sweep reps stay inside the trellis start
  transient taller fixtures passed through, while the row-stream window rep
  (E4M3 K1 R1024, L=14 at 4 rows) still shifts past its window and reaches
  steady state. The Hessian
  predicate sweep still derives every family, wire recipe and scale plane,
  with real encodes sized to complete arity/span groups rather than a model-sized
  weight matrix. Production defaults, pins, guards and timeout/skip policy are
  unchanged; this fixture change is not GPU or scientific qualification.

- **Connected PB fixtures own their authenticated import contexts** (#2192).
  Explicit source fixtures detach and restore the canonical PB graph, fleet
  tools and parent-package edges around their existing reviewed source pin.
  Module-scoped movers retain that graph until their work finishes; installed
  SDK controls retain their own provenance. Removed and orphaned module edges
  restore exactly, so differential monkeypatch targets remain identical.
  Production source-origin refusals, dependency pins, framework generation and
  the next-full negative gate are unchanged; no deployment or native GPU claim.
  Candidate qualification retains named refusal details, and real guard
  controls distinguish rejected in-checkout generated executables from
  sibling fixture work preserving the same source HEAD. Qualification places
  pytest scratch outside the authenticated checkout without moving compiler
  TMPDIR, adding ignore rules or changing the cleanliness guard.
  Ticket scenarios send the pinned broker's actual scope-ID intent fields;
  malformed requests remain failures instead of being mislabeled as missing
  creator cgroup membership. Actual membership refusal remains nonqualified.

- **Original-containing guard regressions retain their existing owners and
  refusal semantics** (Refs #2198, #2125). The seal ratchet classifies the
  issued session's real pending-policy and metadata-owner checks, removes the
  obsolete rebind entry and still rejects an extra run seal at every changed
  scope. Original snapshots and publication bytes use the exact shared JSON
  profiles; identity comparison and source-dispatch validation reuse their
  existing owners while retaining distinct Original/native type, backend and
  error policies. Offline Stage A fixtures provide their real legacy context
  instead of relying on a production fallback. Native reader, resource, CUDA,
  source-admission and automatic-capture gates are unchanged; these corrections
  do not establish full-suite, GPU, scientific or serving qualification.
- **Original checkpoint metadata uses its generic public source owner**
  (#2200). Core checkpoint and streamed identities share
  `source_generation.original_checkpoint_description`, rather than reaching
  directly into the PQ-internal Tessera calibration domain. The old private
  core copy is gone; exact class qualification, absolute-root equality,
  descriptor provenance and refusal vocabulary remain unchanged. CPU
  metadata tests do not establish original GLM/native/capture admission.

- **Original CUDA fixture dependencies use the existing source-owned pins**
  (#2188). The finite pure-Python packer resolves the authoritative PB and
  Tessera pins through their existing stdlib owners, and the inner fixture
  reuses that exact mapping instead of retaining a separate SDK3 literal.
  Installed provenance, RECORD/digest and closed extraction checks remain;
  no new pin, dependency artifact, CUDA qualification or deployment follows.
- **Original proper-prefix coverage compares its exact checkpoint roster**
  (Refs #2147). Python set equality preserves the whole required head/layer
  coverage and rejects missing or extra checkpoints without sending sets to
  the JSON identity encoder. CPU copy-history controls distinguish unfinished
  aliases, which must remain alive through their fence, from successfully
  completed aliases that may be released; native fences and retention stay
  unchanged. Superseded native claims retain their public lease-refusal type.
- **Original authority resources use the native claimed reservation**
  (Refs #2148, #2149). The strict join reads PrismaBuild's actual `resources`
  field, not an invented `demand` alias; missing, changed and misleading alias
  claims remain refused before source work. Synthetic native-panel controls
  retain the existing complete full-calibration provenance grammar rather
  than masking their intended refusal with an obsolete tokenizer field.
  SDK installation pins and all original CUDA/adoption guards stay unchanged.
- **Original render-free diagnostics use a real acyclic context and session**
  (#2149). The shared strict original BASE/preparation/execution contracts bind
  full calibration and current source/runtime/resources without borrowing
  pricing PREPARED, old canonical captures, teachers or source caches. The
  explicit PB CPU metadata issuer creates a genuine pending published artifact
  generation; read-only inspection verifies its metadata owner, exact policy
  and cold namespace before runtime rebinds that same session. The guarded
  original entry keeps full-N row0/probe7000/boundary6 and one Fisher operation,
  remains non-bandable and preserves every original CUDA/automatic refusal.
  No source, GPU, numerical, pricing, wire or serving admission follows.
- **Original authority proof sidecars bind to their selected CAS result**
  (#2152, Refs #2148). The strict consumer requires the existing producer's
  one canonical artifact publication and joins actual control/call/ending,
  raw host/trace and native-reader artifacts by exact node, digest and length.
  Independently rebound sidecars and old receipts without publication refuse.
  Unchanged-family transfers bind the actual target package/runtime, not a
  bare new-source commit string. No old16/full64 restamp or source gate waiver.
- **Original copy lifecycle snapshots are atomic** (#2155, Refs #2148).
  Actual stream registration and pending-to-completed fence/alias retirement
  share the existing source-owner receipt lock. Hardware fences and fatal
  recovery drains remain outside it, with charge retained on failure and no
  new event/query/wait/copy/release. Legacy close remains serialized unchanged.

- **Original copy receipts retain every observed completion** (#2154,
  Refs #2148). A later completion or failure-drain on the same held file
  generation and stream no longer overwrites an earlier successful witness.
  Existing hardware fences, aliases, resource ownership and qualification
  refusals remain unchanged; the history behavior control is CPU-spied only.

- **Original-source authority intake is strict and nonactivating** (#2148,
  Refs #2008). The existing source owner joins independently bound
  publisher/producer/map/readset, original runtime, full calibration,
  acyclic base-plan/run/session, active native claim and finite resources.
  Missing/partial actual full64 qualification or root matched-source
  admission refuses before source/profile/device/output work. The existing
  receipt preserves actual native delivery generations and completion/debt
  witnesses without changing fences, retention or any CPU/direct-GPU/
  automatic-capture refusal. Shared issued-session validation composes with
  #2149 through the existing artifact owner; no new provider/cache/registry.

- **Current-original routed capture has an exclusive scoped intake** (#2147;
  integration dependencies #2148 and #2149). The existing CLI/visitor/source
  owner bind independent authority, issued artifact session, full calibration,
  actual source/runtime/initialization and all five raw tensor identities before
  lossless route transport. Layers 3–43 remain proper prefixes; layer 44 requires
  full text-forward initialization. The four source tensors retain their observed
  indexed device; ordered coordinates remain CPU bookkeeping. Required source
  deliveries/copy fences and all pending lookahead debt stay distinct. Missing
  real qualification/root admission and the unchanged original CUDA guard refuse;
  legacy DEV/cache/complete-v2, LFM, bias, format, pricing and serving gates stay fixed.
- **Original quantum source intake refuses before mutable profile discovery**
  (#2143). An explicitly qualified original material owner applies its existing
  device predicate first and supplies the same owned config/profile to the
  streaming builder. Legacy capture behavior and original CUDA/automatic-source
  admission remain unchanged; no provider or GPU qualification is implied.
- **Checkpoint incoming readset slice agreement is explicitly classified by
  the seal ratchet** (#2176). The exact one-site entry follows the existing
  conservative Stage A slice-binding classification. Foreign slice references
  still refuse in both dev and certified modes; a second identity check still
  fails the ratchet. No runtime guard, scanner rule or existing control changes.
- **PACT regression fixtures retain their declared synthetic standing after
  JSON receipt validation** (#2137). The shared constrained/hull/replay fixture
  now writes a JSON legacy digest-bound artifact rather than plain text. Real
  receipt hashes, shape parsing, lane admission, baseline matching and solver
  replay remain active; no checker attestation or GPU price is fabricated.
  The numerics-pair exporter tests require the already-declared `gguf` full
  dependency, not a changed numeric golden or a skipped arithmetic regression.
- **Remaining producer test callers use the qualified public dependency** (#2158).
  GLM census/capture and stack CLI fixtures select the declared producer
  interpreter with verified installed-package provenance. Namespace and
  materialization handoffs exercise the supported PrismaQuant plan writer,
  not retired Tessera experiment scripts. Serving pins and assertions remain.
- **Expert projection uses Tessera's public installed producer CLI** (#2128).
  The campaign invokes `python -m tessera.producer_plan` with the unchanged
  `tessera.expert_projection.v1` contract; it no longer locates an experiment
  through `TESSERA_REPO` or disables `PYTHONSAFEPATH`. Requires a producer
  package containing Tessera #871. Export/serving pins and admission stay fixed.
  `TESSERA_PRODUCER_PYTHON` can select a separate installed producer without
  changing the pinned consumer/serving package; explicit `python=` overrides it.
- Prepare Tessera lane-schema v11 readers without moving the live b40c93cb/v45
  producer or serving pin. Window-rate rules and census-derived run tables are
  validated once per family and retained separately from the census rungs.
  Runtime, render, profile and shape-price lookups share their derived coverage;
  legal-domain reports use the same parser. This is compatibility preparation,
  not new native, compiled, TP2, quality, construction or release qualification.
- **Native MoE source replay binds the geometry's original router bias** (#2144).
  GLM's strict FP32 `correction_bias` identity is checked with the whole
  expert-roster shape; LFM retains its existing `selection_bias` identity.
  Independent source/runtime/calibration and exact tensor-byte checks remain
  mandatory. This CPU protocol repair grants no GPU, capture or serving admission.
- **Matched-byte control uses Tessera's public installed CLI** (#2168).
  The standing plan/verify producer is `python -m tessera.uniform_control`,
  emitting the existing versioned control handoff without experiment checkout
  imports. The lane can name a separate `TESSERA_PRODUCER_PYTHON`; pinned
  serving packages and all existing unserved/byte/KL/shipping refusals stay fixed.
- **Installed GLM evidence consumers have their container identity owner**
  (#2190). The sole stdlib implementation is shipped as
  `prismaquant.container_runtime_identity`, rather than imported from an
  absent checkout-only `tools` package. Live API and bootstrap paths migrate
  together; direct-file bootstrap still authenticates the mount before
  importing PrismaQuant. Image fingerprint bytes, runtime/source identity,
  duplicate-JSON refusals and scientific gates are unchanged.
  Launcher regression fixtures now supply complete Docker inspection metadata
  to that owner instead of patching a removed launcher-level digest export.

- **Authenticated shape-time tables refuse unmeasured admission** (#2094, PR #2112).
  Conversion and reload compare the independently expected panel digest to
  authenticated bytes. Nonempty rate pools and legacy or synthetic digest-only
  receipts now refuse at `--pact-shape-table` intake; only checker-bound rows
  price allocator/frontier options. Receipt reads reuse the bounded checker
  envelope and malformed bindings refuse cleanly. The serving pin, SDK4 source
  contract and independent native/serving qualification are unchanged.

- **Lane roster mirror learns Tessera v45's structure-scoped
  `column_rates_routed_moe`** (#1618; `lane_eligibility`, `tessera_render`).
  A v45 table was refused as an unknown requirement. The parser now reads the
  field as an ascending subset of `column_rates`, `planned_wire_facts` states
  the unit's structure (`dense` | `routed_moe`) from the eligibility cell
  rather than inferring it, and Tessera's decision core decides the field for
  routed units only: a routed unit at rate 7 is refused with the field named
  and the compact adapter as the route it keeps, a dense unit passes, and a
  unit with no structure fact is refused by name. A v44 contract still reads.

- **Campaign resume refuses W4A4 anchors priced under another activation
  contract** (follow-on to #194; `tessera_campaign._require_resumable_anchor`).
  A pre-#194 checkpoint's W4A4 rows carry no `input_global_scale` (dynamic
  pricing) and rows from another calibration carry a different one; merging
  either silently rebuilds the mixed-currency table on the activation axis
  that the Hessian identity guard refuses on its own.
- **The Tessera export arm fails closed on missing priced inputs, and the
  campaign supplies them** (#193; `run-pipeline.sh`,
  `tessera_export_lane.require_priced_export_inputs`,
  `tessera_campaign.write_export_inputs`). The arm forwarded only
  `--plan-json`/`--device` to Tessera's exporter: an H-aware allocation (the
  campaign's default) was re-encoded weights-only — the exporter builds an
  `ActivationSource` only when `--hessian` is present and raises nothing
  without one — and any E2M1 selection died inside the exporter, which
  hard-requires `--input-scales` for NVFP4 routes. The campaign now writes
  `hessian_capture.pt` (the exact un-normalised per-unit XᵀX plus the
  identity triple, with a JSON provenance sidecar) and
  `input_scales.safetensors` beside its cache; the driver threads
  `TESSERA_HESSIAN`/`TESSERA_INPUT_SCALES` to both the lane preflight and the
  exporter; and a fifth lane gate refuses, before the plan translation, an
  H-aware allocation without its identity-matched capture, a weights-only
  allocation handed a stray one, an undeclared allocation, and a W4A4
  selection whose scales file does not cover every selected unit.
- **Tessera W4A4 anchors are priced under the served static UE4M3 activation
  contract** (#194; `tessera_campaign._measure_anchor`). The campaign scored
  every E2M1 anchor with NVFP4's registry callback — a dynamic per-group
  FP32-scale RTN — while Tessera's plugin executes vLLM's static-global-scale
  `scaled_fp4_quant` against the artifact's `trellis_input_global_scale`,
  with UE4M3-stored block scales; underflow and midpoint cases diverge (a
  1e-3 block flushes to exact 0 under the served contract at G=1 and survives
  under the FP32 scale). W4A4 anchors are now scored through the owned served
  oracle (`nvfp4_activation_qdq_served`) at the unit's calibrated,
  fused-sibling-unified static scale — calibrated over every calibration row
  from the campaign's own forward passes under the resolved NVFP4 policy — a
  missing scale refuses (`ActivationScaleContractError`) instead of falling
  back to the dynamic quantiser, and the scale value/policy identity travels
  on every W4A4 row and in the payload provenance. **Re-pricing:** existing
  Tessera cost tables' E2M1 rows were priced under the dynamic A side and are
  stale; re-measure on the same calibration contract (E4M3/BF16 rows are
  unaffected).
- **The Tessera cost-table identity guard compares the required Hessian
  identity triple** (#195; `tessera_menu.assert_uniform_hessian_identity`).
  The guard keyed rows only on the legacy `(supplied, text_sha, token_count,
  kwarg)` projection, so any number of distinct modern identities —
  `text_sha256` / `fit_tokens` / `fit_ids_sha256`, read from
  `tessera_hessian.HESSIAN_IDENTITY_FIELDS`, which every current campaign row
  carries — collapsed to one key and a table merged from two Hessian draws
  allocated as one. Modern rows now key on the triple AND the legacy aliases,
  pre-triple rows are their own `legacy` class that never merges with a
  modern row, a partial triple refuses by name, and the returned stamp
  carries the canonical triple into `__prismaquant__.tessera_hessian` so
  export can bind a capture against the allocation.
- **The group knapsack's per-member rung licence is read from the Tessera
  contract, not from its own docstring** (#132, RobTand/tessera#37;
  `allocator_candidates.tessera_group_composites`,
  `tessera_menu.fused_module_licence`, `tessera_formats.fused_shared_signature`).
  Contract v6's `fused_module.fields` block says which fields one vLLM-fused
  module's roles must share and which are free per member, and the fold now
  reads it: `fused_module` is part of `contract_answer`, so a contract that
  re-tightens `q256` re-stales the pin with a named field instead of leaving
  the allocator to pick rungs the exporter refuses. The same read narrows the
  fold -- `wire_recipe` is a function of `(grid, q256)`, so one family can span
  two bodies and only rungs agreeing on every shared field are summed -- and a
  `shared` field the allocator cannot evaluate refuses rather than being
  skipped. **Behaviour change:** with no Tessera contract pinned, which is
  production, the fold now returns no options where it previously folded, and
  stamps `__licence__` saying so on every group with a Tessera rung on a
  member's menu. A stock-only group never asks the licence question and so
  carries no receipt: its super item is unchanged.
- **The solver's per-member rung relaxation is licensed, and fused-only**
  (#140; `allocator_solver.py`). `_resolve_family_group` moved a serving unit
  onto one Tessera family with a rate per member on the strength of its own
  docstring, and the branch fired for packed-expert groups the contract does
  not cover. The groups are now tagged by kind where `promote_serving_units`
  builds the list, and the branch runs only for fused-kind components under
  the pinned contract's `fused_module` word for `q256` (`per_member`, read
  through `tessera_menu.fused_module_licence`, which #132 added). A
  packed-expert component takes the uniform path, a `shared` (or absent)
  licence collapses every component to one rung, and a component unioning a
  fused group with a packed one refuses.

- **The lane-slot vocabulary is derived and every derived slot names a verifier**
  (#162; `shipcard.py`, `tests/test_lane_gate_recording.py`). `LANE_SCOPED_SLOTS`
  and `ALL_SLOTS` were two enumerated tuples while the declaration side was
  already derived, so a fourth lane declaring a novel gate could not be honoured
  without a code edit -- and admitting it by derivation alone would have given it
  the generic checks and no replay. The vocabulary is now every `shipcard_slot`
  any `lane_specs/<lane>.json` declares, union the base set; each derived slot
  names its replay in `shipcard.LANE_SLOT_VERIFIERS`, and a declaration with no
  verifier is refused at parse time. `route.census`'s entry is the #136
  priced-vs-served replay, dispatched through the registry, so the slot refuses
  a wrong census as well as silence.

- **The Tessera TP gate fails closed on an unknown profile id** (#120's
  fourth seam; `allocator_candidates._tensor_parallel_applicability`). It
  caught `FileNotFoundError` and loaded `research`, pricing every Tessera rung
  under the research world size for a profile the export would then refuse.
  It now answers `profile_mismatch` like `check_serving_format`; `None` still
  means `research`. No `load_serving_profile("research")` fallback remains.

### Added

- **The Tessera pin moves to master's tip `8ed1d9a` (contract v22, lane schema
  v9 — v21 landed at `b8b1cb38` in Tessera #313 and the release `e78959ed`
  carried v20), and the reader consumes the contract's
  lane predicate through Tessera's own decision core** (RobTand/tessera#195,
  #198, #264, #313; `lane_eligibility.py`, `tessera_render.py`,
  `tessera_runtime_contract.py`, `tessera_serving_runtime_pin.py`,
  `tessera_runtime/tessera_serving_runtime_pin.json`). Between the reviewed v17
  answer and v21 exactly four things moved and nothing else — no family, rung,
  route status, launch, image or version, and the same ten cell ids in the same
  order: the lane schema (`v6` → `v8`; v7 adds a greedy smoke's `control` and
  the `attribution` derived from it, v8 the encoder `artifact` a cell was
  measured from); every `native_extensions[]` row gained a `lane {decoder,
  requires}` block (v20); and the two `routed_moe` cells' `smoke.status` moved
  `repetitive` → `recorded` (v21: Tessera re-measured the smoke through the
  checkpoint's own chat template, receipt
  `docs/measurements/moe-smoke-recorded-2026-09-05.md`, control retired). The
  reader parses v7/v8 closed (`LANE_ELIGIBILITY_SCHEMA_TESSERA` is v8; the
  scoped route receipt and the shipcard's census record gate on it), and the
  `lane.requires` predicate — what the window-GEMV kernel reads: column rates
  `[1,2,4]`, a 14-bit window, `window`/`channel`, no diagonals, rotation or
  release overrides, grid arity 1 — is decided at every admission leg (menu, dev
  pin, export) over the wire THIS producer plans at the rung
  (`tessera_render.planned_wire_facts`) by calling
  `tessera.serving.scheme.decide_lane_requirements`, the rule's one home; every
  `extension::symbol` launch a cell names is bound at parse to that extension's
  declared lane. **Admission under the re-pin:** the eight dense cells (E4M3
  1024 resident/streamed, E2M1 896, BF16 1792, decode and batch) are admitted
  on their own evidence exactly as at v17; the two `routed_moe` cells, refused
  from v17 through v20 on `smoke.status: repetitive` (v20's
  `shared_with_reference` control was read and not admitted on), are no longer
  refused by the unchanged status-only rule (`cell_evidence_admits`), because
  v21 publishes `recorded` — #198 option C, the review event being this pin
  move. That is not a promotion: whether routed-MoE Tessera goes on the menu
  is Rob's under principle 9. The tests assert the mechanism — the predicate's
  answer tracks the status the pinned table publishes — rather than typing a
  verdict, so the same tests hold whichever status a re-pin installs.

- **Lane-eligibility schema v9: a smoke's RECORD, re-derived through Tessera's
  own functions** (RobTand/tessera#327; `lane_eligibility.py`,
  `tests/test_tessera_lane_v9.py`). #327 (P1) found that contract v21's
  `smoke.status: recorded` rested on a repetition rule that lived only in a
  dated measurements file — derived and checked by nothing in the contract,
  and satisfiable by an empty completion. Tessera's v22 puts it in the table:
  `smoke` gains one nullable `record` (`{instrument, rule, reference,
  rows[{prompt, form, interface, status, reference_status}]}`; `null` on a
  cell nobody re-ran), and the lane schema moves `v8` → `v9`. This reader
  parses it closed and **delegates the derivation**: on a v9 table the status
  and the attribution are re-derived by calling
  `tessera.serving.contract.derive_smoke_status` /
  `derive_smoke_attribution`, and a published value they do not derive is
  refused as a contract defect — a repetition rule over completion text is
  not a projection this repository can restate, and a second copy of it is
  the same two-homes defect #327 reports one level down. `EVIDENCE_SMOKE_
  INTERFACES`, `EVIDENCE_SMOKE_FORMS`, `EVIDENCE_SMOKE_RECORD_KEYS` and
  `EVIDENCE_SMOKE_ROW_KEYS` are read from Tessera for the same reason, and
  whether a status was derived at all is asked through Tessera's
  `smoke_status_is_derived` rather than inferred from the key — it reaches
  provenance so a shipcard can tell an attested status from an asserted one.
  One refusal is mirrored rather than re-derived: a `record` beside a
  non-null v7 `control` is refused by name, as Tessera's validator refuses
  it. Two shape rules this reader adds, both #327's finding in the grammar:
  a record must name a non-empty `instrument`, `rule` and `reference`, and
  its `rows` must be non-empty — a status derived over zero observations is
  the empty completion wearing a schema. A `record` on a v8 table is refused
  as an unknown field, and v5–v8 stay SCOPED.
  The lane predicate refuses nothing on the pinned table (only the two
  streamed E4M3 cells launch through the lane and their plan is what it reads);
  a BF16 cell that ever claimed the window-GEMV launch at 1792 (column rate 7)
  would be refused by name. Every admission is scoped: a context-free
  `tessera_lane_attested(name)` answers `False` under the pinned table by SCOPE, so
  the allocator admits Tessera rungs only when the operator passes
  `--tessera-platform/--tessera-runtime-image/--tessera-execution-mode/--tessera-residency`.
  Dev pin, serving pin and their reviewed answer moved in one commit.

- **Tessera admission is pinned to an exact commit plus the packaged contract's
  digest, and the lane reader speaks schema v6** (RobTand/tessera#176;
  `tessera_serving_runtime_pin.py`, `lane_eligibility.py`,
  `tessera_runtime_contract.py`, `tessera_export_lane.py`,
  `tessera_runtime/tessera_serving_runtime_pin.json`). Rob retired the
  release-tag requirement ("we won't have to keep cutting releases"), so
  immutability now rests on the pair a producer can actually check: the pin
  schema is `prismaquant.tessera_serving_runtime_pin.v2`, carrying `commit`,
  `version` and `contract_sha256`, and `require_pinned_tessera_runtime` refuses
  unless the pin equals the reader's constants AND the INSTALLED
  `tessera/serving/runtime_contract.json` hashes to the pinned digest. The
  digest is the enforced half because it is the only half that can be attested
  (principle 14): the packaged contract publishes `versions.tessera`,
  `plugin_entry_point` and `default_serve_image` and no commit field at all, so
  a commit is recorded identity and the digest is the gate.
  `version_is_release` is still parsed, still recorded, and still cannot be
  `true` over a PENDING commit — it is advisory, and no gate reads it. A stray
  Tessera checkout on `PYTHONPATH` is refused exactly as the PENDING sentinels
  refused it, and a new gate (`require_producer_repo_is_pinned`) closes the
  hole the change itself opens by hashing the contract inside `$TESSERA_REPO`,
  the checkout that encodes. The reader now parses `tessera.lane-eligibility.v6`
  (v3/v4/v5 still parse): per-cell `runtime.vllm`/`runtime.torch`,
  `versions.default_serve_image` in place of the removed `versions.attested_on`,
  and a required per-cell `evidence {grade, kl, smoke}` whose `grade` must equal
  what its own `kl` entries derive.
  **Refusal, deliberately:** contract v17 declares `routed_moe` for the first
  time, and publishes on both routed_moe cells `evidence.smoke.status
  "repetitive"` — the runtime's own record that a greedy smoke on that route
  degenerated. PrismaQuant does not admit them. The refusal is the runtime's
  measurement read back (`lane_eligibility.cell_evidence_admits`), not a
  structure ban this repository typed, and it fires in the menu, the render
  admission, `native_cells` and the export lane's structure gate. Promoting
  routed MoE is a decision on the evidence under principle 9 and it is Rob's.

- **A lane's declared gates are recorded on a lane-gated ship record** (#119 in
  part, #162 filed; `lane_spec.py`, `lane_shipcard.py`, `shipcard.py`,
  `lane_specs/*.json`, `run-pipeline.sh`). `route.census` — principle 12's
  second leg on the Tessera lane — carried `shipcard_slot: null` and the arm
  opened no card, so every gate the lane declared was enforced by nothing. A
  null slot now requires an `unrecorded_reason`, `lane_shipcard open --lane
  <lane>` opens a record whose slots are that lane's gates, `required_slots`
  unions them with the base set so a lane can add a requirement and never
  subtract one, and a declared slot the shipcard has no name for raises rather
  than being dropped. Two rosters moved into the lane declaration:
  `wired_architectures` and `producer_tools` (with `stability` and a mandatory
  `tracking_issue`), the latter now preflight gate 4. `RETIRED_EXPORT_LANES`
  names the archive wall in the unknown-lane refusal. Nothing runs a gate yet;
  that half stays with #119.

- **A byte-identity gate on the two super-item aggregators** (#128;
  `tests/test_super_item_menu_byte_identity.py`,
  `tests/fixtures/super_item_menu_golden.json`). Rescued from closed PR #92,
  whose digest `033adb45...` still reproduces on this tree unregenerated. The
  golden pins every super item's whole menu -- order, `fmt`, bytes, both
  serialized identities, activation pricing, membership -- exactly, and the
  four accumulated floats to 16 ulps of `sys.float_info.epsilon`. Its
  load-bearing tooth now perturbs the loop that orders the menu today
  (`for spec in formats:`), and `--regen` refuses without
  `PRISMAQUANT_REGEN_GOLDEN_REASON`, which it stamps beside the digest.

- **The shipcard carries the byte-matched uniform control's verdict, and
  refuses a loss** (#121; `shipcard.py`, `shipcard_cli.py fill-control /
  override-control`, `tools/publish_artifact.py`). A rate-axis artifact must
  show its allocation beat spending the same bytes uniformly on the gold lane;
  a loss refuses at `verify` and at publish unless overridden with the
  re-typed-basename ceremony, which stamps the forgiven ratio onto the card.
  The override binds to the card, not the directory name.

- **Grouped-BMM Fisher: DSv4's `attn.wo_a` is now an allocator decision**
  (`sensitivity_probe.py`, `incremental_probe.py`, `measure_quant_cost.py`,
  `model_profiles/*`). The grouped operand — 33.5M params × 43 layers, 17.9%
  of decode read traffic — had never been priced: the probe skipped
  `DeepseekV4GroupedLinear` because the dense accumulator cannot represent a
  `[G,R,D]` consumption (its `chunk_h` comes out `[R,D]` against a `[G*R,D]`
  plane), and the walk held the weight as a named pin. The new accumulator is
  EXACT (the grouped Fisher is block-diagonal in `g`; one batched matmul per
  hook), shares the dense rows' one global-token normalization and wiring
  identity (`sum(fisher_row) == sum(fisher_col) == h_trace_raw` by
  construction), reports flat-plane dims plus `num_groups` (never
  `num_experts`, so packed-expert scoping cannot ride along), and dispatches
  only on the new spec field `probe.grouped_module_class_names` — declared
  classes lacking `n_groups` fail fast. The cost stage prices grouped units
  from probe keys with no new plumbing and stamps their joint-output-MSE
  screen honestly unmeasured (`output_mse_measured=False`): its dense
  `y = X @ W.T` model would inflate the output term ~G-fold with cross-group
  error no token sees. The walk claim for `wo_a` moves from
  `pin(probe skips...)` to `decide`; no shipped artifact changes (the DSpark
  sidecar keeps all three `wo_a` bases on source-FP8 W8A16, and CB export
  still refuses grouped operands). Boundary: the W8A16 handoff's frozen
  source closure pins `model_profiles/base.py`, `model_profiles/deepseek_v4.py`
  and `specs/deepseek_v4.json` byte-for-byte, so the next handoff
  verification refuses until those three files are re-frozen with review.
  (`docs/ARCHITECTURE.md` §8.9; tests `tests/test_grouped_linear_fisher.py`.)

### Fixed

- **FP8-source MoE checkpoints now build end to end** — first wrapped-VLM
  MoE with an FP8 source through the codebook container
  (Qwen3.6-35B-A3B-FP8, fmt e4m3, block-wise `weight_scale_inv`). Two
  independent defect families, both previously fail-closed with named
  errors:
  - `moe_imatrix._load_tensors` refused every float8 tensor instead of
    fulfilling the serialized scale contract it named: it now loads the
    `<name>_scale_inv` companion and dequantizes exactly on the
    checkpoint's declared `weight_block_size` (partial trailing blocks
    handled exactly), read through the streaming loader's
    `_declared_weight_block_size` so the packed-expert replay and the
    layer-streaming load cannot disagree about one checkpoint. An
    undeclared block size REFUSES rather than inferring the grid by
    division: a 200-row weight over a 2-row scale plane divides exactly
    at 100 and is equally a 128-block tiling with a partial trailing
    block, and the two dequants differ on every row from 128 up. The
    original refusal stands for tensors lacking the companion. Tests:
    contract round-trip, declared-block partial-trailing exactness,
    undeclared refusal, non-tiling (transposed) scale-plane refusal.
  - `export_nvfp4_cb_streaming`: for a wrapped source the group planner's
    regex-matched key lands in the LIVE module-tree namespace
    (`language_model.model.*`) — neither the recipe spelling
    `_resolve_target` looks up (the planner's own documented contract)
    nor the checkpoint spelling emission bridges — so the coverage gate
    KeyErrors on uniform groups and 30,720 consumed per-expert sources
    ship verbatim into `ignore`. Live-spelled keys are now normalized to
    the RECIPE spelling derived from the group's own member tensors;
    recipe- and checkpoint-keyed groups (DSv4-class sources) are left
    exactly as planned (`test_nvfp4_cb_streaming` passes in full), and a
    collision on the normalized key or on the recipe-to-checkpoint bridge
    refuses rather than silently declining to normalize.
    Packed expert stack tensors are named by their group's checkpoint
    prefix via a member-derived recipe-to-checkpoint bridge (a
    recipe-spelled stack falls through gridbook's top-level loader to the
    arch loader and dies). Delegated (stock-CT) and source-passthrough
    config-group targets now ship in the CANONICAL namespace in THIS
    container — the profile's vLLM-internal rename with the wrapper
    canonicalized (`language_model.model.` to `model.`,
    `language_model.<rest>` to `model.<rest>`), mirroring the pinned
    consumer's `gridbook/config.py::_canonical_prefix` /
    `_candidate_bases`, which try a serving prefix as given *and* in
    canonical form. A canonical target resolves from every namespace
    vintage; a full live-tree target resolves only from its own, which is
    what left the delegated Linears unquantized until gridbook refused
    them fail-closed at load (measured on gridbook 0.8.11/0.9.0 +
    pristine vLLM 0.27.1). The vanilla compressed-tensors container
    (`export_native_compressed`) is untouched: its shipped wrapper
    artifacts correctly keep live-tree targets, which vanilla vLLM
    matches. `docs/ARCHITECTURE.md` §6.2 records the three namespaces and
    which consumer each emission speaks to.

    Scope note: FP8 sources whose scales are stored as `.scale` siblings
    rather than `<name>_scale_inv` (DSv4-class, handled elsewhere by the
    profile's `fp8_scale_pairs`) still hit the packed-expert replay's
    original refusal — unchanged, and not covered by this fix.

## 0.16.2 — 2026-08-22

### Fixed

- **MTP Lambert-W closed form no longer silently disables itself**
  (`mtp_rung_selection.py`): `exp(g_over_c)` overflowed for
  `(t+d0)/c ≳ 1022.6` — an illustrative ~41% of the parameter range the audit treated as plausible, including the
  recorded Hy3 constants (~1260) — and the bare-`None` fallback was
  indistinguishable from "no scipy" / "no real solution", so the fixed-point
  answer shipped without anyone knowing the closed form had died. The closed
  form is now computed in log space, sub-representable arguments take an
  exact Newton continuation of `W₋₁` on `s − ln s = L`, and provenance
  records which solver answered in `continuous_bstar_lambertw_status`
  (`docs/design/mtp_rung_selection.md` §3 updated). Failing-test-first: six
  new tests including overflow-regime survival and status handover.
- **`solve_allocation`'s true contract is now documented with a proven
  overshoot bound** (`allocator_solver.py` docstring): the DP bounds charged
  bins, not achieved bits; raw results can exceed target by up to
  `bit_precision·(n_units+3)/2` (derivation + 400-instance fuzz + near-worst
  construction at ~97% of bound), feasibility deliberately enforced upstream
  by `solve_with_promotion`'s ratchet and the byte-budget filter. No caller
  change — the only live raw caller already frames output as an overshooting
  projection.

### Added

- **`tests/test_math_reunderwrite_pins.py`** — twelve hard-coded-value pins
  closing audit gaps: charged-bin conservative table incl. the
  clamp-over-rounding interaction, `predicted_dloss` gain semantics, KL-Fisher
  probe covariance law (T²-scaled) and quadratic-form equivalence,
  ceil-first bit splits, two-tier constants, scale-plane/type_size laws, CB
  ladder rate factors, dual-interval emptiness on equal-byte domination, and
  the solve_allocation overshoot bound instance.
- **`docs/audits/math_reunderwrite_2026-08-21.md`** — the full mathematical
  re-underwrite: every load-bearing artifact re-derived and numerically
  verified (cost chain, encoders, selection/accounting), verdicts per
  artifact, new proofs (two-tier scale-code 2-to-1 structure with provably
  empty exception set; rearrangement envelope for the ½·H·MSE collapse under
  correlated error; backtrack sufficiency), and the findings register
  (F1–F7).

### Documentation

- **Paper consistency pass** (`paper/main.tex`, claims tests green):
  Proposition 5's proof sketch replaced by a complete proof; Proposition 6
  restated over its provable segment-LP core with empirical clauses moved to
  prose; §5 reconciliation splice citing the validated Qwen3.8-27B log-linear
  fit (R²=0.9948, no saturation over 4.50–8.25 bpp); one provenance footnote
  covering the dispersion/fidelity constant cluster flagged by the external
  review.

### Changed

- **The producer Gridbook pin advances 0.8.5 → 0.8.11** (commit `187c721`,
  `gridbook.runtime-contract.v4`), in lockstep with the serving pin, and drift
  between the two pins is now a test failure
  (`test_gridbook_runtime_boundary.py::test_producer_and_serving_pins_name_the_same_gridbook_release`).
  The two pins had silently diverged by three releases. Nothing ships on the
  producer pin's release: route status, serving-lane eligibility, the gold KL
  tools, the shipped-artifact certificates and every serve script's runtime all
  resolve through the serving pin. The producer pin's only live jobs — the
  build/export gates, the format-plan provenance, and the closed gold
  measurement environment — were therefore describing a runtime nothing runs.
  CI made it visible from the side: the `gridbook-contract` job installs the
  *producer* commit, and since the route-status merge that job's contract test
  needs an indexed materialized contract for the installed version, which only
  0.8.10 and 0.8.11 have. The fix is the pin, not a back-materialized 0.8.5
  contract.
- **The closed gold measurement environment grows 29 → 31 names** (execution
  19 → 21). Scanning the 0.8.11 source surfaced four identifiers the 0.8.5
  registry did not know. Two are real environment reads and are now registered
  with canonical gold value `"0"`: `PRISMAQUANT_CB_FP8_GEMV_V2` (the routed
  FP8-CB whole-row GEMV sibling) and `PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R`
  (persistent-B's nested direct-to-register experiment). Two are not
  environment variables and join
  `GRIDBOOK_SOURCE_NON_ENVIRONMENT_IDENTIFIERS`: `PRISMAQUANT_CB_W2_` is a
  documentation wildcard for the three registered W2 knobs, and
  `VLLM_MOE_SKIP_PADDING` appears only in a Gridbook docstring — Gridbook never
  reads it and the `-1` sentinel normalization it describes is unconditional,
  so it is not an execution input of this lane. Both classifications match the
  independent 0.8.7 audit already recorded in `dspark_serving_profile.py`.
- **Behaviour note for gold replays.** Both new names default to `auto` in
  Gridbook 0.8.9+, and both are pinned `"0"` here, matching every other
  dispatch selector in the table (`CB_GEMV=inherited`, `MOE_PERSISTENT_B=0`,
  `FP4_FUSED_MIDM=0`, `BF16_SM120=0`). A gold replay therefore now pins the
  FP8-CB GEMV sibling **off** where it previously ran at the runtime's `auto`
  default. This is a determinism choice, not a quality one: the 0.8.9
  default-state served leg measured kl_mean +0.17 % / PPL −0.06 % against the
  gold record, inside the ±0.7 % cross-session envelope. Re-baselining gold
  onto the auto dispatch stays available as a reviewed re-measurement.
- **The frozen gold-environment digest is restated, not moved.** The 29-name
  historical projection still hashes to its original literal
  (`41dd44c5…`), proving no pre-existing canonical value changed; the full
  31-name map carries a new digest. The freeze now proves what it was written
  to protect.
- **The FP8-CB fused mid-M backed set is unchanged.** `rungs_by_runtime_version`
  already carried a `0.8.11` key with the identical `{28,32,36,40,44,48}` rung
  list, so `gridbook_runtime_version()` moving to 0.8.11 changes no rung, no
  route and no codec.
- **Not re-served.** The six launchers that source `gridbook_runtime.sh`
  (hy3 smoke/TEB, laguna smoke, qwen27b smoke, the canary ladder, the NVFP4-CB
  delegation smoke) now install 0.8.11 instead of 0.8.5, which is intended;
  their artifacts have **not** been re-served on 0.8.11 in this change. Neither
  has the 91-pass installed-wheel GB10/sm121 W8A16 GPU gate, which remains
  0.8.5 evidence carried forward on 0.8.11's unchanged
  `source_fp8_block128_w8a16` attestation.
- **Fail-closed consequence for existing gold cards.** Shipcard verification
  compares an artifact's recorded environment against the canonical map by
  exact equality, so re-verifying a card produced before this change now
  refuses on the grown 31-name contract. Historical cards remain valid evidence
  for the release they were verified under; re-verification under the advanced
  pin is expected to refuse.

## 0.16.1 — 2026-08-21

### Changed

- **Gridbook serving pin 0.8.11** (commit `187c721`, wheel sha `3fbd257e…`
  read from `gridbook:0.8.11-clean-187c721`'s PEP 610 record; PyPI archive
  verified member-byte-identical to a tag rebuild, 60/60). 0.8.11 is 0.8.10
  plus two CUDA-graph capture fixes and nothing else. gridbook#46 (smb209):
  the MXFP8 dense lane's swizzled-plane A-side offsets were computed on the
  host and moved with an unpinned copy on first use, which aborted
  `FULL_DECODE_ONLY` capture at load; they are pre-warmed at load now.
  gridbook#47 (smb209): the routed grouped lanes' `_padded_route` read a
  routing-**dependent** trim count on the host — and, for the BF16 grouped
  bridge, the per-expert block offsets — so vLLM 0.27's default
  `FULL_AND_PIECEWISE` capture of prefill sizes above the 16-token GEMV band
  died at engine start ("operation not permitted when stream is capturing").
  That abort was protective: a captured graph would have replayed one
  capture-time routing's tile count on every later routing. Under capture the
  fused FP4/FP8 lanes now launch the static-capacity tile layout
  (`P // tile_m + E`, provable from shapes alone — the
  `PRISMAQUANT_CB_GROUPED_TRIM=0` arm); the one lane that chunks by
  host-read per-expert offsets — the opt-in sm12x bridge,
  `PRISMAQUANT_CB_BF16_SM120=1` — refuses capture naming the flag, while the
  default expand + grouped bridge and the persistent-B lane never host-read
  and capture as-is. Eager and decode-band (≤ 16 tokens) dispatch are
  byte-identical to 0.8.10, so no route, codec, or default changes for any
  published artifact and the backed set is carried forward unchanged; the
  packaged runtime contract is byte-identical to 0.8.10's (materialized as
  `gridbook_runtime_contract.0.8.11.json`, still `lane_eligibility: absent`).
  The fp4-CB lane ADDENDUM, the CB endpoint image digest, the Qwen3.8 smoke
  `BASE_IMAGE`, and the real-pin route-status tests follow. Measured on the
  shipped DSv4 87 GB body under the new image (`perf-b1-0811`): the card
  command (`FULL_DECODE_ONLY [1,2]`) decodes 20.53–20.61 tok/s vs
  20.54–20.63 on 0.8.10 — unchanged, as designed — and vLLM's default
  `FULL_AND_PIECEWISE` with capture sizes up to 64 now starts (11 piecewise
  + 7 full graphs) and decodes 20.56–20.64 single-stream, so the default
  command no longer needs `--compilation-config` to come up. Batch-32 decode
  (32 streams) is the one regime where the capture-safe layout costs
  something: captured static-capacity TPOT 789 ms vs 608 ms with capture
  sizes kept ≤ 16 (batch-32 steps eager, trimmed), because
  `cb_fused_moe_grouped` runs its mainloop on pad tiles and at T=32/E=256
  the static capacity is about twice the trimmed tile count. Only the 11
  per-role FP8-CB layers ride that lane above 16 tokens; the pooled-book
  reburn moves them to persistent-B, and a kernel early-exit for
  `expert_id < 0` tiles is the general fix. Until then, multi-stream
  deployments of the shipped body keep capture sizes ≤ 16 — the card
  command already does. Negative control: the same default-mode command
  on the 0.8.10 image dies at the first capture above 16 tokens
  (`moe.py:207`), so the start is measured on this stack.

## 0.16.0 — 2026-08-21

### Added

- **AQUA prices packed routed experts — 94.5% of an MoE's quantizable
  parameters that the activation term had never reached.** On
  Ornith-1.5-35B-A3B the AQUA merge covered 310 of 402 units; the 91 misses
  were the packed routed-expert tensors. Two causes, the second decisive:
  packed expert params carry no `.weight` suffix so `build_weight_resolver`
  never indexed them, and even once resolved they had no `g_sq_sum`, because
  marginals come from `nn.Linear` backward hooks while a packed `[E, M, N]`
  expert is an `nn.Parameter` on a fused module.
  `install_packed_expert_hooks` already intercepted every expert-slice matmul
  — including `down_proj`, whose input is the post-SwiGLU intermediate — and
  already held `(x, gy)` per slice; nothing read them for the activation side.
  Marginals are carried **per expert, not aggregated**, because routing makes
  both `g` and the activation distribution functions of `e` and
  `sum_e W^2 g var != (sum W^2)(sum g)(sum var)`. Sensitivity card schema 1.1
  adds `expert_g_sq_sum [E, M]`, `expert_act_sq_sum [E, N]`,
  `expert_act_absmax [E, N]` and `expert_tokens [E]`, with the two
  normalizations deliberately different: `g` by global tokens, activation
  variance by routed tokens.

- **`lane_specs/compressed_tensors.json` declares
  `served_activation_quantization`.** AQUA-AURA previously refused on the lane
  every flagship ships through: the key existed only on `nvfp4_cb`, so the
  activation term had never been priced on a vanilla-vLLM artifact and asking
  for it returned a REFUSE rather than a number. The list is **derived, not
  asserted** (principle 14): vLLM packages no runtime contract, but on this
  lane the executed contract is a function of the artifact we write — vLLM's
  compressed-tensors dispatcher picks the scheme from the checkpoint's own
  `config_groups[*].input_activations`, which `export_native_compressed`
  emits from its per-format scheme table. Both ends are readable, so the list
  is their intersection, and each entry cites the producer field and the
  consumer predicate. Dense scheme and fused-MoE method were verified
  **separately per family**. MXFP4 is excluded on purpose: its scheme declares
  no `input_activations` key, so vLLM serves it W4A16 while the registry
  descriptor calls it W4A4 — pricing it off the registry would charge a
  phantom A4, the exact shape of the DSv4 mispricing. The registry is not the
  authority here; the lane is.

### Fixed

- **The A-side test tolerance pinned the pre-GPU host-float64 path.**
  `test_activation_dloss_uses_g_sq_sum_not_fisher_row` asserted `rel=1e-9`,
  exact only while `activation_dloss` reduced in numpy float64 on the host.
  Moving that reduction to the device (principle 7) makes the square and the
  product float32 with a float64 accumulation — a deliberate trade, since
  every term is a square times a variance and nothing cancels. Measured
  aggregate error 3.4e-9 relative; the tolerance is now 8 float32 eps, derived
  from the dtype rather than picked. The discrimination the test exists for is
  untouched.

### Documentation

- **D32: the Fisher probe is not bit-reproducible.** A probe-side change was
  gated on `h_trace` being bit-identical to the previous probe and refused
  twice. Two runs of the *same* code at the *same* pin settled it: 379/402
  units differ, median 2.5e-4, max 1.1e-2 — larger than the old-vs-new
  difference of 1.56e-4. `n_tokens_seen` and the per-expert Fisher support are
  bit-identical on every unit, so the forward and the routing are exactly
  deterministic; only the backward moves, and 30 of 40 layers are Gated
  DeltaNet whose fla Triton kernels reduce over chunks in non-deterministic
  order. Consequence recorded: probe-derived artifacts must be rebuilt
  together from one probe run, or `cost.pkl`'s stamped provenance names a
  probe that produced only some of its numbers.

- `CLAUDE.md` named the wrong `AURA_ADDITIVITY_GATE` default (`auto`);
  `run-pipeline.sh` defaults it to `measure`, which is why a run's `cost.pkl`
  carries a measured additivity residual rather than a predicted sum alone.

## 0.15.3 — 2026-08-18

### Fixed

- **`lane_specs/nvfp4_cb.json`: the CB lane executes BOTH families'
  activation grids.** The 2026-08-17 entry scoped
  `served_activation_quantization.executes` to `["FP8_CB_*"]` by reading
  gridbook's "exact native BF16 bridge" as activations-left-exact; the
  runtime QDQs NVFP4_CB activations to E2M1 group-16 on every served route
  (`linear.py` `fp4_act_qdq_or_codec`, moe.py's three routed sites,
  `codec.py` `fp4_group16_act_qdq`), so the bridge names a GEMM schedule,
  not an activation precision. The premise was retracted the same day; the
  spec now says `["NVFP4_CB_*", "FP8_CB_*"]`, drops the dead
  `selectors_must_be_unset` guard, and cites the runtime call sites.
  Measured on the shipped Qwen3.8-27B 13 GB card: the corrected entry
  re-allocates to the shipped `layer_config.json` byte-for-byte, while the
  stale entry silently moves 337/496 body units (272 FP8_CB → NVFP4_CB
  family flips). Both shipped artifacts were priced with both A-sides and
  are correct as shipped; the fix protects every FUTURE fresh-card CB
  campaign, which consumes this list with no flag and no refusal.

## 0.15.2 — 2026-08-18

### Fixed

- **`tools/measure_served_gold.py` `_tail_logprob`: libm-dependent phantom
  residual.** A fully-tabulated row can re-exponentiate to `1.0 − 1 ulp` on
  some libms, and `log()` of that phantom residual returned ≈ −36.7 instead
  of the documented no-residual answer. Two exact guards: `vocab_size <=
  len(row)` means there is no untabulated set to estimate, and a residual at
  or below `len(row)` ulps of 1.0 is summation rounding, not mass. Real
  residuals still spread max-entropy, clamped at the K-th value. (Numeric
  effect on any real KL was bounded by ~1e-14 nats; the fix is about the
  contract, not a measured delta.)
- **CI: the pinned-contract conformance fixture asked the producer for the
  deleted signed fp4 family** (removed 2026-08-17). The tiny-export leaf
  becomes a second product rung at k13; the reference decoder drops the dead
  signed branch. The test only runs in the pinned-Gridbook CI job, which is
  why local suites never saw it.

## 0.15.1 — 2026-08-18

### Changed

- **Gridbook serving pin 0.8.10** (commit `f4b3274`, wheel sha `7a7c98e1…`
  read from `gridbook:0.8.10-clean-f4b3274`'s PEP 610 record; PyPI archive
  verified member-byte-identical to a tag rebuild, 60/60). 0.8.10 is 0.8.9
  plus a fix for a load regression 0.8.9's own suite could not see: the
  tri-state refactor renamed a `moe_gemv_select` symbol that
  `gridbook/moe_mixed.py` still imported, so any artifact declaring
  `per_expert_format_groups` (a split-bank mixed expert stack) died with an
  ImportError at config dispatch. Uniform stacks — every published artifact —
  were unaffected; the pin supersedes 0.8.9 with zero serving-behaviour delta
  on everything shipped today. `fp8_cb_fused_mid_m` gains the 0.8.10
  backed-set key carried forward unchanged (the packaged runtime contract is
  byte-unchanged since 0.8.6); the fp4-CB lane ADDENDUM, the CB endpoint
  image digest, and the Qwen3.8 smoke BASE_IMAGE follow.

## 0.15.0 — 2026-08-18

Merge of two parallel lines: the DSv4-Flash release campaign
(`fix/aqua-profile-aware-resolver`, 74 commits) and the rescued-work
integration line (`merge/proven-rescues`, 43 commits).

### Added

- **Gridbook 0.8.9 serving pin — the qualified CB kernels default on.** The
  serving runtime moves to gridbook 0.8.9 (wheel digest read from the
  `gridbook:0.8.9-clean-23a3955` image's PEP 610 record), whose three lane
  selectors are tri-state with unset → auto: persistent-B decode-in-mainloop
  for routed CB MoE prefill (both payload families), the CB GEMV v2
  dictionary kernel, and the FP8 whole-row GEMV sibling on its qualified
  cell. Every explicit spelling keeps its 0.8.8 semantics, so the canonical
  gold environment replays recorded routes unchanged. Default-state served
  evidence on the shipped clean 87 GB body: kl_mean +0.17 %, PPL −0.06 % vs
  its gold record. The `fp8_cb_fused_mid_m` backed set gains its omitted
  0.8.8 key plus 0.8.9, and the r1–r7 served-validation instruments are
  tracked under `scripts/pb_validation/`.
- **The CB ship-gate stack generalized off the DSv4 shape** — the gold
  contract is a lane read fail-closed from the artifact's own `config.json`;
  every lane's measurement scripts are tracked; a two-artifact (body +
  DSpark draft) release is a declared topology with per-role coverage.
- **AQUA on CB lanes**: the activation answer is priced per format FAMILY
  under the pinned runtime's executed contract, with anchored dense drivers;
  a lane that executes nothing refuses the merge instead of pricing a
  phantom A-side.
- **Byte-budget partition hardening**: namespace exclusions price exactly
  once, the partition is pinned through the real `allocator.main()`, and a
  frozen approval constant can no longer double as a run target.
- **Artifact-completeness fifth namespace**: the checker reads delegated
  targets, per-expert split-format group tokens, routed per-role claims, and
  the DSpark sidecar's published physical→construction bijection — via both
  the sidecar alias map and the dspark-threaded unit-variant bridge
  (redundant fail-closed paths; unification is a recorded follow-up).
- **Publication chain**: publisher rides Xet for large files with digest
  replay after commit; card figures (`allocation-map.png`, `byte-budget.png`)
  share the README exclusion from `model_sha`, so documenting an artifact no
  longer invalidates its own gate records.

### Removed

- **The signed `NVFP4_CB_S*` family is deleted** (registry, encoder,
  exporter, footprint, serving profile). No native route ever admitted an
  `n_sub = 1` rung and no allocation on disk referenced one; a recipe
  carrying `cb_mode: "signed"` now refuses instead of silently resolving to
  the product rung of the same `k`.

### Fixed

- The streamed DSv4 driver fed the `main` rope table to all compressed
  layers (the perplexity-262 teacher); the mapping now has one definition
  and the silent fallback raises. The teacher forward-fidelity gate enforces
  context-monotonicity on the teacher's own NLL.
- The public-repo `DATASET` default pointed into a home directory; the dense
  CB driver read the sensitivity card as a raw npz; `ALLOW_PINNED` was
  unreachable from the pipeline; a validation rung that doubled as the
  anchor validated nothing.
- Native-execution truth is reported per batch regime (decode vs batch), not
  per unit — the "73.7 % unbacked" DSv4 reading was a regime share.


## 0.14.0 — 2026-08-14

### Added

- `allocator_solver.DualInterval` and `selected_rung_dual_intervals`: each
  selected rung's local weak-Lagrangian support interval, expressed in
  predicted-dloss per DP-charged byte. An **empty** interval is a meaningful
  result rather than an error — it marks an integer-knapsack choice sitting in
  a non-convex pocket that no scalar lambda supports, which is precisely the
  documented reason lambda-bisection was rejected as a selector and kept only
  as a candidate generator. Diagnostic only: no pipeline stage calls it, and
  no allocation changes.


## 0.13.0 — 2026-08-14

### Added

- **The Sensitivity Card** — a shareable, probe-once artifact that carries
  everything an optimizer needs to price an *arbitrary* format menu, so a new
  format costs a registry entry rather than a new probe. `sensitivity_card.py`
  builds and validates it, `format_cost_protocol.py` + `format_cost_registry.py`
  are the plugin seam, and `sensitivity_card_allocate.py` feeds the real
  allocator DP. The card compresses the probe's per-Linear Fisher structure to
  per-channel marginals — 104.2 GB down to 75.2 MB on 27B — because the
  marginals are out+in vectors, not the full outer product.
  See `docs/design/sensitivity_card_contract.md`.
- **AQUA-AURA** — activation-quantization awareness. AURA is provably blind to
  the W4A4/W4A8 choice today: NVFP4 and NVFP4A16 render weights *bit-identically*
  (max `dW` difference 0.0) while differing 9.42% RMS on activations, so the DP
  cannot see the difference no matter how it is weighted. `speed_quality_frontier.py`
  adds the speed/quality lever as a **constraint on a Pareto frontier, never a
  weight** — activation format changes speed and quality but not bytes, so it
  does not belong in the byte-budget objective.
  See `docs/design/aqua_aura_activation_awareness.md`.

### Fixed

- `incremental_probe.py` computed `lm_head`'s `h_trace` through the
  outer-product-norm identity with both reductions in **bf16**. That reduction
  runs over the output dim — the vocabulary, ~152k addends on Qwen3 against
  ~1–5k for a body Linear — so with 8 mantissa bits it cost ~1e-3 relative on
  `lm_head` and ~1e-8 everywhere else. `lm_head` was consequently the only unit
  of 197 failing `SensitivityCard.validate()`. The trace is now taken from the
  same fp32 object the marginals reduce, which both body-layer sites already
  did, so `sum(fisher_row) == sum(fisher_col) == h_trace_raw` holds by
  construction rather than by numerical coincidence. Measured on Qwen3-0.6B
  n=8 T=512: row-vs-trace agreement 1.019e-03 → 2.781e-08, card validation
  failures 1 → 0. `lm_head` sits in the allocator's non-quantizable floor by
  default, so no default allocation changes; a run that opts it back in with
  `--allow-pinned lm_head` now prices it correctly rather than through a bf16
  vocabulary-length reduction.

### Changed

- The probe now emits five per-channel Fisher marginal vectors into `probe.pkl`
  unless told not to. **This is the only shipping default this release changes,
  and it is probe-side.** The card, its three cost tiers and AQUA-AURA are
  additive modules with no `run-pipeline.sh` call site, no `COST_MODE` value,
  and no served validation — they allocate nothing today, by design.

## 0.12.3 — 2026-08-14

### Fixed

- A wheel that fails the pinned-digest check is no longer published into the
  serving-wheel cache. The cache is keyed by the *expected* digest, and the
  materializer's first branch trusts that directory name — it takes the single
  wheel inside and verifies it, never consulting a supplied wheel or a download.
  Caching a rejected wheel was therefore permanent: one
  `pip download gridbook==0.8.6` bricked the DSpark serving lane on that
  machine, and supplying a correct wheel afterwards could not help because the
  supplied path was never reached. The pre-`mv` verification did not abort
  because the materializer's only caller reaches it as
  `wheel="$(_gridbook_serving_materialize_wheel)" || return`, and Bash disables
  `errexit` for a command substitution whose enclosing command is part of a
  `||` list — re-arming `set -e` inside that subshell does not restore it, so
  the function's own `set -euo pipefail` is inert on the one path that runs it.
  All three verification sites now test their status explicitly, and the
  fast-path refusal names the directory to remove instead of reporting only a
  digest mismatch. Regression-tested through the download path (the only path
  where the defect fires) and mutation-proven against the pre-fix file.
  This defect was *armed* by publishing 0.8.6: before it existed on PyPI the
  download failed outright and cached nothing.
  See `docs/audits/serving_wheel_cache_poisoning_2026-08-14.md`.

### Changed

- `docs/ARCHITECTURE.md` no longer describes the DSpark serving pin as pending
  sentinels — it is resolved — and it no longer asserts a rule the code
  contradicts. The doc required the digest "reported by the published PyPI
  file"; `prismaquant/gridbook_serving_runtime_pin.py` requires the digest read
  out of the served image and forbids substituting a PyPI or locally rebuilt
  wheel. Both cannot govern. Measured: the two 0.8.6 wheels are
  content-identical (all 58 archive members byte-for-byte equal, differing only
  in zip container metadata) and the PyPI wheel was built by the release run
  from exactly the pinned commit, so the rules select the same code and differ
  only in which archive's digest is asserted. Which rule governs is Robert's
  call; the doc now records the tension and the operational consequence rather
  than hiding it.
- The serve-environment census fix shipped in 0.12.2 is now **verified
  end-to-end on a live server** rather than by screen: the release-pinned image
  with the ship gate's exact environment block reports `consistent: true` with
  the EngineCore process renamed (`VLLM::EngineCore`) and both PIDs at an
  identical allowlist digest, before and after a completion request.

## 0.12.2 — 2026-08-14

### Fixed

- Serve attestations can now actually read the environment they compare. Both
  Gridbook runtime Docker vectors set `SPT_NOENV=1`. The environment census
  reads `/proc/<pid>/environ`, and vLLM's EngineCore renames itself through
  `setproctitle` (`vllm/v1/engine/core.py` → `set_process_title`), which on
  Linux overwrites the contiguous argv+envp block and destroys that file while
  leaving the process's real `os.environ` intact. The census therefore saw a
  destroyed remnant for the one process that runs the CB kernels, reported
  `consistent: false`, and refused a correct server — structurally, on every
  lane, at every commit, and it had never once been green. Measured in the
  pinned serve image across that single call, `/proc/self/environ` went from all
  six probed variables to zero while `os.environ` kept all six.
  `SPT_NOENV` confines the process title to the argv area (the title truncates,
  which is cosmetic; kernels, memory and numerics are untouched).
  This is not a relaxation: every allowlisted name's value is still compared
  exactly, so a genuinely mismatched EngineCore environment still fails, and
  `SPT_NOENV` is deliberately excluded from every compared allowlist.
  See `docs/audits/serve_env_census_setproctitle_2026-08-14.md`; §5 records why
  this does **not** unblock the already-built DSv4-Flash 0731 artifact, whose
  gate is bound to its own build commit.

## 0.12.1 — 2026-08-13

### Fixed

- Serve fingerprints now attest both supported immutable Gridbook installation
  forms: the existing exact VCS commit and an exact release wheel. Wheel-backed
  images must provide `PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256`; PrismaQuant requires
  the matching PEP 610 archive SHA-256 and versioned wheel filename, then still
  verifies `direct_url.json`, `METADATA`, `RECORD`, every installed Python/CUDA
  source byte, and the actual import origin. Unpinned wheels, digest mismatches,
  editable/bare-directory installs, and same-version import shadows remain
  fail-closed. The v0.12.0 VCS evidence shape remains replay-compatible.
- Endpoint, performance, and shipcard replay accept the optional digest-bound
  wheel identity while continuing to require the tracked Gridbook commit and
  version. This fixes production images that correctly install a reviewed wheel
  instead of synthesizing VCS metadata inside the image.

## 0.12.0 — 2026-08-12

The DSv4Flash export-readiness release. It lands the activation-safe,
identity-bound AURA campaign and replay path merged in PR #81, then closes the
source block-FP8 serving mismatch discovered during final allocation review.
The approved 112.690 GB assignment is unchanged; this release changes the
legality and provenance gates used to reproduce and export it.

- Bind campaign completion, streamed checkpoints, allocator replay, exporter
  handoff, and child-process imports to one clean immutable PrismaQuant source
  snapshot. The CPU-only W8A16 readmission reconstructs the historical AURA
  rows and permits only the audited source-terminal correction; it refuses any
  assignment or byte-accounting drift before export.
- Preserve the production cache/prefetch contract and bounded one-Spark
  residency throughout the DSv4 path. No parallel activation or rendered-weight
  cache was introduced.

- Resolve the external Gridbook 0.8.5 runtime pin to immutable commit
  `e992e5980c96333a48149f96392d6cff56ae9e3f` and promote the raw block-128
  E4M3/UE8M0 source lane to its dedicated W8A16 route. The installed-wheel
  GB10/sm121 gate passed 91 tests with no skips; decode uses native source
  GEMV and prefill uses transient BF16 expansion plus Gridbook's owned grouped
  CUTLASS bridge. Newly exported artifacts no longer carry the obsolete
  route-pending acknowledgement. Full-artifact serving, performance, and
  quality remain independent shipcard gates. The exact command, image, wheel
  identity, logs, and JUnit record are in
  `docs/results/gridbook_0p8p5_w8a16_gate_2026-08-12.md`.
- Make every shipping boundary require that exact Gridbook version and commit,
  not merely a syntactically resolved release pin. Source block-FP8 remains
  W8A16; the distinct direct group-32 MXFP8 re-encode lane remains W8A8 and
  unbacked by default.

Activation-quantization-aware AURA and MTP/DSpark re-optimization are explicitly
post-ship work and are not part of 0.12.0.

## 0.11.0 — 2026-08-11

The learned-codebook release: value-bearing learned books reach the production
allocator and exporter, and the cost surrogate is refactored so that the
*evaluation of a model no longer knows what a codebook is*. A platform-agnostic
anchored-cost core owns anchor planning, shape fitting, hull pruning and
exposure reporting; a mapping plugin supplies the format vocabulary.

MINOR, not patch: the allocator now rejects candidates that price above the
source bit rate and replaces the structural CBL rung ceiling with a measured
per-rung policy, so a CB menu can select a different assignment than 0.10.0
would have. Published artifacts on disk are unaffected; the lattice default
render is byte-identical.

**Nothing here is a served result.** There is no vLLM KL-vs-BF16 or WikiText
PPL for learned codebooks, for LDLQ, or for anchored AURA in this release. The
routed learned-book runtime opt-in ships **default-off**. The DSv4 anchored-AURA
campaign has not been run — its driver's first production run will also be the
first end-to-end exercise of the seams below, every one of which is fail-closed.

### Learned codebooks reach production

- Lift the allocator's hard refusal of `--cb-codebook-source=learned`, replaced
  by scoped learned bundles: an immutable, value-bearing `.pqcb` read by cost,
  cache, KL, allocator and export from one identity, rather than a digest
  manifest plus export-time retraining.
- A bundle reports its basis **per rung** via `codebook_source_by_format()`
  (refusing a non-uniform map), which is what lets a single FP8_CB menu span
  learned K28–K46 alongside lattice K47/K48 instead of forcing one basis on the
  whole family. `load_bundle`'s policy stamp remains the primary guard.
- Wire learned bundles through the production exporter.
- Routed-MoE learned books, **producer side only and refusing by default**:
  version-gate routed learned refs on the Gridbook per-role LUT ABI
  (`>= 0.8.3`) so that old, prerelease, local or malformed pins refuse before
  export. Expert bundle cells are built only from explicitly selected,
  identity-verified K28–K33 burn shards, copying their FP16 books exactly with
  no export-time training, LDLQ, directory search or lattice fallback.
  Independent logical role refs are emitted for fused uniform and
  per-expert-format stacks while their physical `gate_up_proj`/`down_proj`
  payloads are retained.
- **The Gridbook pin stays on the released 0.8.2** (`9f915dd`). An earlier
  revision of this work advanced it to an "0.8.3 preparation commit"
  `032e8158…`; that commit exists nowhere — not on the Gridbook remote, whose
  newest tag is `v0.8.2`, and not in any checkout on the build box, where
  `gridbook.__version__` still reads `"0.8.2"`. CI caught it at
  `pip install gridbook @ git+…@032e8158…` and the pin was reverted before this
  release. Consequence: the routed learned path refuses under the shipped pin,
  which is the correct state while its ABI is unreleased, and the 0.8.2 fused
  mid-M rung table remains the attested backed set.

### Anchored AURA — a platform-agnostic cost core with a format plugin

- `anchored_cost` owns the generic mechanism and imports no format module. A
  plugin declares the candidate ladder, the shape-transfer equivalence
  partition, the renderer hook, the anchor-rung policy and provenance;
  `cb_anchored_cost` is the codebook instance.
- The equivalence partition is load-bearing: segment keys are
  `(family, role, equivalence_class)` with the class declared by the plugin, and
  the core refuses to fit or apply a shape across a declared boundary — so
  pricing a family as one segment when it spans two bases is impossible by
  construction rather than by convention.
- AURA `predicted_dloss` is the sole currency; `weight_mse`, `output_mse`,
  `h_trace` and `cw_m2` are refused as cost inputs so a sensitivity cannot be
  applied twice. Anchors are production-arm renders bound to a render receipt —
  a bare scalar cannot masquerade as one.
- The allocator admits anchored AURA on **provenance**, not on a claimed
  property: three independent stamps, with `fisher_application_count` read
  through `operator.index` so a string cannot forge it.
- **`activation-inclusive` is retired.** Calling anchored AURA an
  activation-inclusive supersurrogate was wrong. "Supersurrogate" is a statement
  about the *currency* — one projection replaced the two-factor
  `h_trace x output_mse` score — not an activation error model: `aura_cost`
  runs its adjoint on unquantized boundary activations and `dW` is a weight
  delta. AURA is activation-**weighted** and activation-quantization-**blind**.
  The activation path is constant across K within a CB family, so the blindness
  moves only the `nvfp4_cb`-vs-`fp8_cb` family-choice margin. Carried as a named
  limitation, reported and not gated; a served A/B arbitrates.
- `dsv4_aura_cb_reprice` is a thin acceptance driver (model profile, CB plugin,
  byte budget) behind the frozen `tools/run_aura_cb_reprice.sh`, defining none
  of the pricing math: one production anchor per legal
  `(unit, family, equivalence_class)` — 66,951 for DSv4, against ~198k cells and
  ~3.3 TB for a full-menu campaign.

### Allocation legality and provenance

- Derive exact source and candidate payloads from the footprint authorities,
  reject above-source candidates, and persist complete elimination provenance.
- Byte-verbatim terminals now speak the allocator's source-passthrough contract
  (`SOURCE_PASSTHROUGH_COST_SOURCE`) instead of a second spelling of it.
  Allocation was already numerically correct, but every terminal had been
  misclassified in provenance and on the activation branch. The two spellings
  are pinned together by a test.
- Correct the DSv4 routed-expert profile declarations.

### Identity-bound resume, checkpointing, and exact accounting

- `production_render_cost` gains `--format-plan` and validates the cache format
  set against it, so a cost run can no longer be scored against a menu the cache
  was not built for; `source_class_format_plan` derives the plan from source
  classes rather than a hardcoded family.
- Streamed cost with identity-bound atomic checkpoints, so a long cost stage
  resumes on the exact model/menu/arm identity instead of trusting file
  presence. Same treatment for CB pair shards and per-linear KL adjoints;
  resumed CB artifact state is verified rather than assumed.
- Producer identity hashes the complete producer package and binds git-less
  producer source bytes.
- Exact per-rung footprint rate accounting, plus an encode-identity regression
  test so a performance change cannot silently move bytes.

### Performance

- Fuse the LDLQ atom candidate search into one compiled region behind
  `PRISMAQUANT_CB_ATOM_COMPILE=1` (unset is a byte-identical no-op). Measured at
  production shapes: `reassign_product_2d` 223.6 ms → 70.0 ms (3.19x),
  `reassign_product_3d_batched` 1994.5 ms → 258.5 ms (7.72x), peak GPU memory
  unchanged at 1.21 GB. The eager route keeps `torch.linalg.solve_triangular` —
  substituting it unconditionally regressed the gate-OFF path to 0.46x.

## 0.10.0 — 2026-08-09

The DeepSeek-V4-Flash campaign release: LDLQ becomes a *certified* encoder path,
the vendored DSV4 forward becomes faithful to the reference, and four silent
correctness defects in the export path are closed.

MINOR, not patch: `layer_config.json` gains a required key, exported CB weight
bytes move for any artifact built with LDLQ on, and DSV4-class exports change
which units they declare quantized. Published artifacts already on disk are
unaffected — nothing here rewrites them — but they cannot be re-produced
byte-for-byte without the reproduction switches named below.

**The LDLQ quality claim is not yet a result.** Everything below is a screen:
local render/activation MSE, holdout-gated cost-table statistics, and block
parity against the reference implementation. There is no served vLLM
KL-vs-BF16 or WikiText-PPL number for LDLQ in this release. The served A/B is
pending.

### LDLQ certified out of sample — and the in-sample gate was anti-correlated

- **The do-no-harm gate was measuring the objective LDLQ had just been
  optimised against.** It built each expert's Hessian from the captured
  activation rows and then scored keep/revert on those same rows, so it could
  not fail — and its error was not a constant. On `L17 gate_proj, K12` the gate
  figure and the truth on held-out rows run in **opposite** directions: at full
  64-row support the gate posted 0.0325 against a true 0.6517 (20x
  overstatement); at 1–3 rows it posted its *best* figure of the study, 0.0196,
  where the true gain was 0.9504 — i.e. 5% — a 48.5x overstatement. Pricing
  from it would have **inverted** the allocator's ranking, not merely inflated
  it. Support is thin and layer-dependent: the median tensor has 21 activation
  rows against 2048–4096 columns, and 8.2% of tensors have exactly one.
- **The replacement certifies on rows no arm of the gate saw.** Scored with the
  gate handed only half of each expert's rows to keep the evaluation
  non-circular, degeneration falls **7/96 → 1/96**; on full-support `down_proj`
  the new gate rejected exactly the one regressing expert and nothing else.
  It is not literally never — at 4 rows the certificate splits 2/2 and has
  little power. Production hands the gate all rows, so 1/96 is an upper bound.
  Two honest limits: the *shipped* assignment is still the all-rows fit, which
  sees strictly more data than the arm that earned the certificate; and the
  authoritative check remains a model-level disjoint-corpus A/B, which is the
  pending served work.
- `PRISMAQUANT_CB_LDLQ_GATE` selects `holdout` (default), `in_sample`
  (reproduction of pre-2026-08-08 artifacts only, never for new ones), or `0`.
  The certifiability floor `LDLQ_GATE_MIN_ROWS` rises **2 → 16** — eight fit and
  eight decision rows, an evidence floor and explicitly *not* a claim that
  sixteen rows give a population-level guarantee. Tensors below it keep raw.
- **`PRISMAQUANT_CB_LDLQ_SCOPE`** (`none|nvfp4|all`, matching the allocator's
  `--cb-ldlq-scope`) replaces the legacy boolean as the authoritative
  per-family switch and is stamped into the serialization context; an
  inconsistent legacy/scope pair refuses. Per-tensor identities now record the
  LDLQ that actually applied to each format, which fixes mismatched identities
  on mixed NVFP4/FP8 assignments.
- **Both CB exporters resolve LDLQ per format rather than from one global
  boolean, and the activation loader is handed over per format with it.** In
  the non-streaming exporter each format resolves through `_ldlq_for_format`
  and an activation is loaded only when that format's family is LDLQ-eligible
  under the active scope. The streaming exporter had two further scope leaks:
  a single global boolean decided the warm path, the recorded identity and the
  loader requirement for every format alike. The actual per-format family scope
  now controls all four — encoding, warm path, identity and loader — so an
  FP8 tensor under `scope=nvfp4` is neither encoded as LDLQ nor charged an
  activation-loader requirement it cannot satisfy.
- **Product-atom E16 LDLQ** (`cb_ldlq_atoms.py`): exact FP8 atom-2 / FP4 atom-4
  exhaustive Mahalanobis assignment, with typed Hessian failures so dead
  channels refuse rather than fabricate an identity. The canonical packed route
  is ABI-fixed — serial, feeder-thread, non-16 expert-batch, shared-Hessian and
  non-divisible routes are now **refused**, where some were previously legal.
  Two concurrency defects are closed with them: a missing caller→worker
  `wait_stream` in both multi-stream arms, and a factor-cache cross-stream race
  (the cache is now event-backed, bounded at 8 GiB, and keyed on
  `Tensor._version`).
- **The gate no longer materializes the whole stack.** Each fp32 reconstruction
  at DSV4 shape is 16 GiB and the gate transiently held two of them; scoring is
  now chunked per expert slice, holding **64 MiB instead of 16 GiB**. This is
  bitwise identical, not approximately equal — reconstruction is row-local and
  the gate MSE is a within-expert reduction, so slicing commutes, and the tests
  assert exact float equality across both scale codings.
- **Fused siblings union their activation rows by global row index**, raising
  Hessian support rather than intersecting it: on DSV4 L0 expert 0, gate and up
  contribute 64 rows each with an intersection of 15 and a union of 113. Rows at
  shared indices must be bit-identical or it raises.
- Never-routed experts take the cold-expert prior on the direct per-expert path
  too — the DSV4 export reports 3,984 never-routed expert projections across 60
  stacks, so raising there would make LDLQ unusable on any MoE with cold
  experts. The substitution is logged, not silent.
- New additive artifact sidecar **`cb_ldlq_gate_telemetry.json`** (schema
  `prismaquant.cb_ldlq_gate_telemetry.v1`) with exact-qname coverage
  enforcement and a validated kernel stamp; and a post-allocation refinement
  contract `cb_ldlq_refinement.v1` recorded in `quant_config` provenance and in
  `layer_config`. Raw exports are unchanged.

### A no-LDLQ cost table from the LDLQ burn, without a second burn

- The CB fields encoded *before* the gated reassignment are the identical-env
  raw render — same encode, codebook, scale sweep and coding, same
  `col_weights`. A gated cost run therefore now also emits
  `weight_mse_raw_render`, `predicted_dloss_raw_render` and
  `weight_mse_per_expert_raw_render` under provenance
  `prismaquant.cb_ldlq_raw_render_sidecar.v1`, which is what makes the
  LDLQ-contribution A/B affordable at all: the isolate would otherwise cost a
  second multi-hour burn. Output-side metrics are **not** re-measured for the
  raw arm — the allocator prices `predicted_dloss`/`weight_mse`, and a raw
  output measurement would need the full-stack forward the sidecar exists to
  avoid. Ladder-rejected slices deliberately record no sidecar.
- **`tools/extract_raw_cost_table.py`** derives the no-LDLQ allocator cost table
  from it, re-stamping the CB context as `ldlq=false, scope=none` and recording
  the source stamp under `derived_from_ldlq_gated_cost`. It fail-closes on error
  rows, missing or partial sidecars, and already-raw input, and its output must
  pass `validate_cb_cost_provenance` before it is written.
- When LDLQ is off this is a strict no-op: no sidecar keys are emitted and cost
  pickles stay byte-identical, asserted key-for-key against the legacy row
  schema. Banked pickles are never rewritten.
- Never-routed experts have no calibration activations by construction (51 on
  the production capture), which crashed the weight-only row path under an LDLQ
  context. The fix is an explicit call-site opt-in used *only* by that path; the
  default still raises, so a broken activation loader can never silently produce
  an all-raw table stamped as LDLQ.

### Band-interpolated rungs are priced from their own `output_mse`

- A row stamped `band_interpolated`/`mixed` carries `output_mse_measured:
  false`, so it fell to the weight-only branch and was priced as `weight_mse ×`
  a per-family activation constant while its measured neighbours on the same
  ladder were priced from `output_mse` directly — one family, two bases. The
  true output/weight ratio is **not** family-wide: 187–320 on gate/up_proj
  against 9.4–22 on down_proj, a ~20x spread. The constant over-priced
  interpolated down_proj rungs ~12x and under-priced gate/up_proj ones ~1.6x,
  **breaking rung order inside the family**: the higher rung cost more than the
  one below it on ~85% of down_proj experts, so K13/K17 could never be selected.
- This retracts a design claim of `activation_fair_pricing`: "a per-family
  constant cannot reorder rungs inside a family" holds only while every rung of
  the family takes the same branch, and these did not. On the 32 banked layers
  down_proj violations fall from **84.8%/0%/84.6% to 0.014%/0.014%/0.043%**.
  `cost_source` and `output_mse_measured` keep their meanings and no banked
  pickle is rewritten; a new branch label `interpolated_output_mse` records in
  the shipped artifact which selected prices were predictions.
- Ladder interpolation was separately checked to survive the LDLQ identity, on
  the layer-9 pilot (18 exact rungs, 103 cells): gated `output_mse` is
  log-linear in K per family, LOO median 0.9–2.3% and p99 3–7.6%, and the gate
  branch is a cell property (~28% raw across all rungs) rather than a
  K-crossover — so the per-tensor anchors+holdout law applies unchanged.

### The vendored DeepSeek-V4 forward is now faithful to the reference

- The vendored forward carried a half-split `rotate_half` RoPE instead of the
  release's interleaved complex rotation, and an **amputated compressor/indexer
  branch**. Both are restored, with per-layer YaRN tables and the HCA/CSA path
  per the reference, and probe mode becomes an explicit flag with a loud
  one-time warning — the silent architectural degradation is gone.
- **Recorded for honesty: every prior DSV4 probe output — `h_trace`, the
  KL-adjoint, the activation cache, and every cost derived from them — was
  measured through the defective forward, and is scheduled for re-derivation
  before the next allocation.**
- **The first certification was itself wrong, and is retracted.** The "29/29
  parity, zero divergence" claim never executed the vendored module: its
  HyperConnection check was a source-text search and its block check compared
  the reference against itself. `HyperConnection.__getattr__` in fact raised for
  every non-alias name, so the `fn`/`base`/`scale` Parameters were unreachable
  and every `self.fn` access crashed.
- True block parity is now certified by executing the *vendored*
  `DecoderLayer.forward` against the vendor's own Block on real DSV4-Flash-Base
  weights, with forward-hook counters proving execution rather than source-text
  inspection: `max|diff|` **1.0133e-06** (sliding), **1.5348e-06** (CSA),
  **1.6689e-06** (HCA), every intermediate boundary ≤ 6.2e-06 and the mHC
  boundaries exactly 0.0. Six further defects fell out of it, two of them
  silent: the CSA top-k was used as a `torch.gather` index still carrying the
  reference's `-1` sentinel — which CUDA does not bounds-check, so it **read out
  of bounds on every CSA layer** — and pooled-entry mask columns were padded
  visible, a causality break on every compressed layer. The compressor was also
  skipped whenever no cache was passed, so the Fisher probe silently got
  window-only attention with no error.
- Profile fixes: the indexer lives on the CSA compressor, not on attention, and
  its checkpoint tensors sit flat on it — the corrected mapping resolves **all
  72,317 checkpoint keys** with zero misses. A new plugin accessor
  `ModelProfile.probe_linear_exclude_extra()` makes the probe's Linear-exclusion
  regex profile-owned; DSV4 excludes `self_attn.{compressor,indexer}`, restoring
  the 33,325-selectable-Linear inventory the byte accounting assumes.
  `transformers` signature drift between 5.6.0 and 5.12.1 is bound by keyword.

### Streaming

- Nested rotary instances are materialized on meta skeletons.
  `ModelProfile.init_rotaries` gains an optional `base_model` kwarg (the Gemma4
  override is updated in step) and the DSV4 override walks the skeleton, so no
  `inv_freq` buffer is left on meta.
- Eviction no longer meta-izes non-persistent buffers, which `_fast_install`
  deliberately skips and therefore could not restore. Harmless while all rotary
  state lived at the model root; fatal once the faithful forward puts
  compressor/indexer caches inside the layers.

### Export and allocator correctness

- **`layer_config.json` now records `cb_serialized_payload`.** The allocator
  wrote per-tensor CB identities but never the context they were written under,
  so on DSV4-Flash **24,851 of 24,851 CB tensors mismatched** at export. This is
  the root cause of four earlier "CB per-layer serialization identity mismatch"
  failures, and it **retracts** the earlier attribution of those to a
  packed-expert/LDLQ-scope mismatch. The assignment itself is unchanged. A
  `layer_config.json` produced before this release lacks the key and must be
  re-produced by the allocator.
- **21 `attn.indexer.wq_b` units were being shipped as unquantized floats.** The
  floor block-FP8 scan iterated a live-model-name map in which the indexer had
  0 of 33,368 entries, so those units fell through to the verbatim-copy loop and
  were declared `ignore`. A consumer honouring that allocates bf16, the element
  counts match, and the FP8 bytes are cast **with no scale applied** — every
  block wrong by its own power of two, and nothing raised. The scan now reads
  the checkpoint. Pre-existing and LDLQ-independent.
- A failed export **preserves** its partial `.tmp-<token>` root and prints
  resume/discard commands instead of deleting it; a 21-tensor mis-declaration
  threw away ~6 hours of tensor writes that the retry reproduced bit-for-bit.
  The destination is still never created on failure and never clobbered, and the
  preserved root carries no completeness stamp.
- New `dspark_source_metadata.py` bridges the released three-stage DSpark
  topology (4,705 tensors) into the exporter — MXFP4/FP8-block source unit
  classification, physical-to-construction MTP layer mapping, refusal of partial
  stages, and an atomic hardlink-sibling sidecar publisher that provably
  rewrites zero tensor bytes.
- Gridbook runtime pin advances **0.8.0 → 0.8.1 → 0.8.2** (`9011a19` →
  `c9c1265` → `9f915dd`), each with its matching backed-rung key
  `[28,32,36,40,44,48]`. The 0.8.1 bump had landed without its key, which
  emptied the fail-closed resolver's backed set and silently degraded every
  `k%4==0` rung to the expand+GEMM fallback.
  The advance to **0.8.2** is what makes the released pin describe the runtime
  that actually serves: the DSV4-Flash serving images are built from Gridbook
  0.8.2, so shipping a pin that declares `0.8.1` with `version_is_release: true`
  would have asserted a released runtime the artifact does not run on — the
  exact confusion `tests/test_gridbook_runtime_boundary.py` exists to prevent.
  The rung set carries over verbatim, verified more strongly than by reading the
  constant: `gridbook/codec.py` is **byte-unchanged** across the entire
  v0.8.1..v0.8.2 range, so `FP8_FUSED_KBITS` cannot have moved; 0.8.2's content
  is loader and dispatch work above the codec. **This release therefore depends
  on Gridbook v0.8.2 being tagged and its commit pushed** — CI installs the
  runtime from the pinned git commit and asserts the reported version matches.

### Campaign tooling

- Thirteen DSV4 A-FAST campaign tools under `tools/`, plus the burn runner. The
  CBL audit gate no longer escapes `_measure_projection` as an `AssertionError`
  before the layer-wide fallback can run — that turned a recoverable verdict
  into an aborted campaign at 42/43 layers. Per-expert uplifts are now recorded,
  because the aggregate hides the spread: one expert 8.51% worse and one 0.93%
  worse in a sample whose median *gained* 8.45%.
- A Pareto-point writer `KeyError` on every research-cost run emitting a CB
  format is fixed (it dereferenced `cb_render_identity` while guarding on a
  different field).

### Known state at release

- Suite: **3123 passed, 0 failed, 86 skipped, 3 xfailed, 151 subtests passed.**
- **One piece of disclosed debt.**
  `tests/test_dsv4_campaign_tools.py::test_dual_holdout_fit*` is
  `xfail(strict=True, raises=KeyError)`. These two tests were *introduced in
  this cycle* by 41b5c62 and have failed since — they did not exist at v0.9.0,
  so they are not pre-existing debt inherited from an earlier release. The
  cause is a live defect in the campaign tool rather than in the tests: the
  2026-08-06 demand-driven revision narrowed the priced domain to `RUNGS =
  K28–K38` without moving `ANCHORS = (28, 38, 48)` or `HOLDOUTS = (33, 43)`
  with it, so `_fit_slices` indexes its rung-name map with K48 and raises
  before evaluating anything. A real dual-holdout fit takes the same path.
  It is marked rather than repaired because choosing the replacement anchor and
  holdout rungs changes which rungs are measured and which are predicted — a
  campaign-design decision — and the 43-layer burn in flight was launched
  against these exact constants. The marker is conditional on the
  inconsistency and strict, so the suite fails the moment the domain is made
  consistent and the marker must then be deleted.
- `docs/ARCHITECTURE.md` §0 conformance for this range was checked and closed in
  the same release: the LDLQ certifiability floor (documented as 2, actually
  16), the raw-render sidecar and its extractor, three omitted defaults, and the
  two new profile-plugin accessors. `docs/design/runtime_flags.md` gains rows for
  `PRISMAQUANT_CB_LDLQ_SCOPE` and `PRISMAQUANT_CB_LDLQ_GATE`.

## 0.9.0 (2026-08-05)

- **Monotone Min-Chain encoder mode** (`PRISMAQUANT_CB_MINCHAIN=1`): per-rung min over the free LDLQ fit and the previous rung's embedded solution — error curves monotone non-increasing by construction at zero representational tax. Winning-arm/solution/predecessor digests enter the serialization context; mismatches refused at export. Validated: pilot-2 PASS on pre-declared DSV4 layer (zero violations, PCHIP held-out 2.7-2.9% median / <14% p95, 1.003x overhead). (#76)
- **Per-expert-slice ladder gating** with sliced measured fallback for packed-expert cost measurement; per-slice cost provenance (`cost_source_per_expert`). (#74)
- **Batched LDLQ encode** across identical-shape expert units (bit-identical to serial; ~5x on 256-expert stacks). (#73)
- **Pipelined export** (`PRISMAQUANT_EXPORT_PIPELINE=1`): read/encode/write three-stage overlap, byte-identical artifacts (GPU-verified). (#75)
- Qwen3.6-35B-A3B model profile + campaign machinery; amendment-v2 interpolation semantic (5-anchor monotone PCHIP, accept-all + CV outlier backstop + per-layer audit rung). (#74, #76)

## 0.8.0 — 2026-08-03

This release advances the immutable runtime boundary to Gridbook **0.8.0** at
exact commit `9011a19228ddb96b8a49e11a20ac75c99c83998e` — the released
`v0.8.0` tag commit — and adds the source-passthrough format family, its
measured serving verdicts, and the MXFP8 re-quantization rung.

MINOR, not patch: the allocator menu gains formats, `route_backed` is replaced
by a three-valued `route_status`, and the runtime pin file's schema moves to
`v2`. Persisted assignments and shipped artifacts are unaffected.

### Source-passthrough format family (#53)

- **`SOURCE_PASSTHROUGH_CONTRACTS` makes "keep the checkpoint's own bytes" a
  first-class rung** instead of two hand-written special cases. Two formats,
  both from a header-only census of the released checkpoint: `MXFP4_SOURCE`
  (routed experts, nibble-packed E2M1 + E8M0 group scales, **4.25 bpw**) and
  `FP8_BLOCK_UE8M0_SOURCE` (body, E4M3 + one-byte UE8M0 block exponents at
  128×128, **8.00049 bpw**). The census also corrected two live defects: the
  source-kind scan keyed on "has a scale sibling" and stamped 33,024 MXFP4
  experts as `FP8_SOURCE`-compatible — an 8.002 bpw format declared legal on a
  4.25 bpw unit — and the body is not `FP8_SOURCE` at all.
- **Zero-cost candidates, with bytes pinned to the checkpoint.** Shipping the
  source bytes is the identity transform on the reference, so `predicted_dloss`
  is `0.0` with provenance `cost_source="source_passthrough"`; the allocator
  *synthesizes* the candidate because no cost table will ever carry a column
  for a byte-copy contract. The claim cannot be forged — the provenance string
  is honoured only for a declared passthrough format whose activation path is
  the identity. `assignment_artifact_bytes` charges the unit's real header span
  and refuses an allocation where the span and the closed-form byte count
  disagree, rather than letting the artifact budget drift.
- **Measured route verdicts replace assumed ones, and both inverted.**
  `route_backed: bool` could not distinguish "nobody looked" from "we looked
  and it is dead", so it becomes `route_status` (backed / pending / blocked)
  plus `route_requirement` and `route_evidence`. Measured on GB10/sm121:
  `MXFP4_SOURCE` is **backed but only via vLLM's Marlin MoE backend**
  (requirement: `--moe-backend marlin`), and `FP8_BLOCK_UE8M0_SOURCE` was
  **blocked** — every stock route dead. The second finding is load-bearing: CB
  re-encoding of the body is the only way DSV4-Flash serves on this box by
  default, so the body's CB rungs are architectural, not merely economical.
- **Exporter verbatim stream-copy lane.** Passthrough tensors are stream-copied,
  never materialized, so a 3.4 GB expert layer moves through the 16 MiB chunked
  path without becoming a tensor. E8M0 scale planes are copied **verbatim**.
  `quant_config.json` gains `source_passthrough` (schema v1); absence of the key
  means "legacy all-CB artifact", so it is omitted rather than emitted empty.
  `build_quant_config` round-trips its own output through the consumer's parser
  before the file exists.
- **Fixes a pre-existing silent corruption.** `_StreamWriter.write` built its
  header as a dict, so two emit paths claiming one tensor name kept only the
  last span while both blobs were written — a file whose offsets are wrong from
  that point on, with no error. It now raises.

### MXFP8_UE8M0_G32 menu format (#54)

- **A second MX-FP8 format, because it is a different on-disk contract.**
  `MXFP8_E4M3` defers to compressed-tensors, which rounds the group amax to a
  power of two and can scale a group *up* — losing small values off the E4M3
  subnormal ladder. This rung picks the smallest non-clipping shared exponent
  and serializes `float8_e8m0fnu` rather than `uint8`. The `MXFP8` alias still
  points at `MXFP8_E4M3`: repointing it would reinterpret every persisted
  assignment that uses it.
- **W8A8, with the A side measured.** The exactness claim is `weight_mse == 0.0`,
  **not** `output_mse == 0.0`; declaring `act_bits=8` makes the cost stage apply
  the activation closure before measuring, and a row with no measured
  `output_mse` is masked off the menu rather than priced at the global minimum.
- **A latent codec divergence fixed on the way.** `_batched_quantize` dispatches
  on `weight_element_dtype`, which this rung shares with `MXFP8_E4M3` and
  `FP8_CB`, so the batched render would have priced it with the codebook
  replica's E8M0 snap — a different codec in the batched and unbatched paths.
  `_EXPORT_ALIGNED_BATCH_FORMATS` routes it to the registry closure and a test
  holds the two paths to each other.

### Gridbook runtime pin advanced to 0.8.0, and a ratchet so it cannot drift

- The pin now names Gridbook **0.8.0** at `9011a19228ddb96b8a49e11a20ac75c99c83998e`.
  It reached that value through two intermediate bumps during development
  (`7c0b527`, then `4d7292c`), and that is exactly what exposed the gap below.
- **Pin schema `v1` → `v2`: the new `version_is_release` boolean.** A gridbook
  feature merge does not bump `gridbook.__version__`, so a pin can advance to a
  post-release master commit while still self-reporting the same version —
  three distinct commits self-reported `0.7.0` during this work. The version
  string alone therefore cannot say whether a runtime was ever released; the
  commit is the identity and this flag records the rest.
- **`rungs_by_runtime_version` is ratcheted against it**
  (`tests/test_gridbook_runtime_boundary.py`): no key may name a runtime newer
  than the one actually pinned, and the pinned version may appear as a key only
  when its commit *is* that version's release. An unreleased pin backs nothing —
  the fail-closed direction the spec's own "when the pin advances, ADD the
  version key" rule already asked for, now enforced rather than trusted.
- `serving_profile_specs/nvfp4_cb.json` gains the `0.8.0` key, carrying the same
  fused mid-M rung set forward after verifying `FP8_FUSED_KBITS` is still
  `tuple(range(28, 49, 4))` at the v0.8.0 tag commit. The block-FP8 body lane
  moves from **BLOCKED** to **OPT-IN, NOT BACKED**: Gridbook 0.8.0 serves it
  through its own sm120 block-scaled MXFP8 collective, correctness-audited at
  worst rel-Frobenius 5.9e-5 over the seven real-checkpoint body shapes, but
  behind `GRIDBOOK_MXFP8_DENSE=1` with the served timing bench still pending.
  Available is not backed, so it prices no rung.

### Also

- Stage the dual-variant DP drivers for the mid-rung and body-route questions
  (#55): `run_dsv4_mxfp4_dual_alloc.sh` and `summarise_dual_alloc.py`, which
  produce the contingent allocation by *widening the menu* rather than editing
  the route table — pricing a future is fine, declaring it is not.
- Docs now cite the current pin rather than `746473c`; the `0.7.0` section below
  deliberately still names that commit, because it records what 0.7.0 pinned.
- `ci.yml`'s header claimed the suite was 967 tests in ~51 s (reference box,
  2026-07-28). It is ~2762 tests in 10.5–14 min on the runner; the note now says
  so, and names the real margin against the job's 30-minute timeout.

## 0.7.0 — 2026-08-02

This release advances the immutable runtime boundary to Gridbook 0.7.0 at exact
commit `746473c8459acd24c71e7602d1c982da2f8fa80e`, and carries the
DeepSeek-V4-Flash-0731 92 GB pipeline work, the cross-repository K0.2
stage-attestation interop test, and a verified docs-truth batch.

MINOR, not patch: the cost stage prices packed experts differently (per-Linear
activation rows, declared never-routed experts) and the CB encoder was rewritten
for speed, so a production run's numbers and provenance move even though the
producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged. This release makes no DeepSeek-V4
(DSV4) qualification or support claim; the DSv4 CB export lane remains
unbuilt (`docs/lanes/nvfp4-cb/dsv4_readiness.md`) and that work stays paused.

### The DSv4-Flash-0731 92 GB pipeline (merged from `dsv4/flash-0731-92gb`)

- Ported the 2026-08-01 92 GB study work onto the 0.6.0 line: the Stage-0
  format screen (`scripts/ab_nvfp4_vs_k36_dense.py`) now loads through
  `_LazySkeleton.dequant_weight` — the same profile-aware decoder the CB cost
  stage and the exporter use — instead of reading raw safetensors bytes, which
  compared *storage codes* against values for a checkpoint that ships packed
  I8-MXFP4 experts and F8_E4M3 dense tensors. It hard-fails on an unscaled
  float8 tensor rather than silently mis-measuring, resolves its checkout from
  `__file__` instead of a hard-coded `sys.path.insert`, and indexes through
  `detect_profile().checkpoint_to_live_name` so profile-rewritten names
  resolve. Pinned by `tests/test_format_choice_stage0_source.py`.
- Added the production calibration driver `scripts/run_dsv4_flash_92gb.sh`,
  runnable against the `gridbook:test` image, plus the exclusive-GPU
  old-vs-new validation harness (`tools/cb_encode_exclusive_bench.sh` and the
  `tools/cb_encode_*` / `tools/cost_*` probes behind it). The harness refuses
  to benchmark a contended GPU, stops orphaning driver containers, and
  distinguishes "no output" from "different output" rather than scoring a
  silent failure as a pass.
- **Cost-stage correctness.** Every Linear is now measured on its own
  activation rows and fails loud instead of borrowing a sibling's; declared
  never-routed routed experts get weight-only cost rows (Option A) under the
  new never-routed rule; and the CB cost encode is ~2x faster
  **bit-identically**, pinned by
  `tests/test_nvfp4_cb_encode_perf_identity.py` and the v1-vs-v2 K14-excess
  rank-stability cross-check.
- Corrected the DeepSeek-V4-Flash parameter count: **~285 B total** by
  checkpoint arithmetic (281,263,734,784 probe-measured quantizable
  parameters), not the 671 B DeepSeek-V3-family headline; and the 172 GB
  "below the floor" figure is the Pareto grid, not a fixture. The three
  dsv4-branch cost-stage flags are documented in
  `docs/design/runtime_flags.md`.

### K0.2 stage attestation executed across the repository boundary

- `tests/test_gridbook_attestation_interop.py` is the first place one process
  runs **both** halves of the producer/consumer stage attestation. It builds a
  routed-MoE record through the real emitter — a synthetic two-stage FusedMoE
  checkpoint through `synthesize_packed_expert_activation_samples`,
  `calibrated_input_global_scales_with_sources` and `build_execution_contract`
  — then parses it with Gridbook's own v2 parser and verifies it
  `attested_and_verified`, including the artifact-level K0.2 verdict read off a
  tmp `quant_config.json` + `model.safetensors` exactly as
  `k02_readiness_verdict` reads a real artifact.
- It closes a real trap rather than adding tidiness. The attestation was held
  together by 4 pinned digest hexes, 3 schema literals, and a Gridbook-side
  fixture that hand-mirrors this emitter; neither suite ever executed the
  other's code. Gridbook's parser requires a stage entry to declare *exactly*
  `_STAGE_ENTRY_FIELDS` (extra keys rejected), while every digest is framed
  over those same five fields **by name** — so an "additive, backwards-
  compatible" producer-side field moves no hex on either side, leaves every
  pinned-hex test green, and first surfaces at vLLM model load. The new tests
  demonstrate that mutation end to end and name the exact load-time error.
- Wired into the required `pinned Gridbook contract` CI job's file list, which
  already installs the exact VCS pin. Skip semantics match the three tests
  already there: gated on `PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT` with
  `importorskip` behind it.

### Docs-truth batch

- Fixed eight verified defects from a corpus review — docs and one provenance
  data file, no code and no behaviour change. The load-bearing ones: §9.2's
  claim that the constrained Pareto formulation was "normative future work"
  had been false since P5c shipped; `docs/design/runtime_flags.md` carried 13
  ghost env vars left standing by the 2026-07-30 L2/L3 wall (retired into a new
  §9.1 ledger recording each token's last reader, rather than silently
  deleted); and 18 `file.py:NNN` citations in `docs/` pointed past EOF, 14 of
  them citing `kl_measurement.py` lines 2054-5516 against a 1,246-line file.
  Citations inside dated audit records were **annotated in place** per the D21
  ledger convention rather than rewritten.
- `docs/lanes/nvfp4-cb/STANDARDS.md` now states that its FP8-CB fused rung set
  is a hand-maintained mirror whose machine-readable form is this producer's
  own `nvfp4_cb.json`, and that PrismaQuant CI *cannot* catch it drifting,
  because Gridbook's packaged `runtime_contract.json` does not carry the fused
  rung set at all.
- The example serve-dispatch table's `provenance.source` strings were
  normalized to byte-match the Gridbook headings they cite (U+00D7 and U+2014
  had been ASCII stand-ins), so a `grep` against the source now resolves them.
  No claim changed — only the citations became resolvable.

### Gridbook runtime pin advanced to 0.7.0

- `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` now names Gridbook
  **0.7.0** at exact commit `746473c8459acd24c71e7602d1c982da2f8fa80e` — the
  peeled `refs/tags/v0.7.0^{}` commit, not the annotated tag object
  (`c23393750fc53d0463892571a8854457a091ea0c`).
- **The producer/runtime lane gate is now directional.** Gridbook 0.7.0's D0.1
  registers `deepseek_v4` in the packaged `runtime_contract.json`, so for the
  first time the runtime *leads* this producer on an architecture: it declares
  it can serve DSV4, while the DSV4 CB export lane remains unlanded here
  (`docs/lanes/nvfp4-cb/dsv4_readiness.md` gaps 1-3 — the exporter still loads
  the whole model in `_load_skeleton`, has no fp8-block dequant-on-read, and
  does no per-expert to packed-expert stacking). The required contract job had
  asserted strict set EQUALITY between declared CB lanes and the runtime's
  `producer_profiles.supported_ids`, which made the intended landing order —
  serving contract first, exporter second — unrepresentable. It now asserts
  CONTAINMENT (`declared ⊆ supported`). The dangerous direction is unchanged
  and still fails closed: declaring a lane the runtime cannot serve does not
  crash, it ships an artifact that serves uninitialised memory, so a new
  negative-control test fabricates exactly that case and requires the check to
  fail and name the architecture. PrismaQuant makes no DSV4 export or
  qualification claim on the strength of the runtime's registration.
- The `nvfp4_cb` serving profile's FP8-CB fused mid-M lane now declares
  `"0.7.0": [28, 32, 36, 40, 44, 48]` — the SAME set as 0.5.0 and 0.6.0, and
  added as a NEW version key rather than by editing an existing one. An
  artifact produced under an older pin therefore stays resolvable at the route
  it actually shipped on, and a pin with no key here still backs nothing
  (fail-closed).
- The set did not move because it **cannot**: 0.6.0's K1.2 resolution proved
  `k % 4 == 0` is a format + TMA law, and `FP8_FUSED_KBITS` is still
  `range(28, 49, 4)` at 0.7.0. The five off-law rungs of the published 27B
  K36..K47 ladder stay permanently expand+GEMM-served and the allocator keeps
  pricing them on the fallback row.
- **No FP4-CB backed set was added.** Gridbook 0.7.0's contract-preserving
  FP4-CB v2 fused mid-M kernel is *still* opt-in behind
  `PRISMAQUANT_CB_FP4_FUSED_MIDM=1`, pending its served NATIVE-PARITY gate;
  with the flag unset the dispatch is byte-for-byte the BF16 bridge. Available
  is not backed, so the honest backed set for the DEFAULT contract stays empty.

## 0.6.0 — 2026-08-02

This release advances the immutable runtime boundary to Gridbook 0.6.0 at exact
commit `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1`, and lands the producer-side
items **P5a**–**P5d** of the cross-repo performance ultraplan,
[gridbook
`docs/audits/ultraplan_perf_2026-08-01.md`](https://github.com/RobTand/gridbook/blob/master/docs/audits/ultraplan_perf_2026-08-01.md)
§6 ("Producer-side allocation: NVFP4 vs FP8-CB at matched bytes"), together with
the producer half of gridbook **K0.2**.

P5a and P5b change how candidates are **priced and described**;
`solve_allocation`'s DP semantics are untouched. P5c adds a **second hard
constraint axis** — served latency and device memory — at assignment level,
still without changing the DP and still without any λ: latency never enters
the objective. P5d adds the D0.3 exact-rate experiment harness.

The producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged by the runtime advance. This release
makes no DeepSeek-V4 (DSV4) qualification or support claim; that work remains
paused.

### Activation-fair pricing on the weight-only cost branches (P5a)

- Fixed the audit's first cost-model asymmetry: W4A4-vs-W8A8 activation cost
  was priced **only** on the measured `output_mse` branch, so packed experts
  under `PRISMAQUANT_EXPERT_COST_SAMPLE` and ladder-interpolated rungs under
  `PRISMAQUANT_CB_LADDER_INTERP` — most rows of a production run — were
  priced weight-only, crediting NVFP4-CB with its cheaper index stream and
  none of its A-side cost. The allocator now calibrates one per-format-family
  correction per run (geometric mean of the measured-over-weight-only Δloss
  ratio, over the rows that carry both estimators) and applies it to that
  family's weight-only-priced rows.
- The correction is multiplicative, so it cannot reorder rungs within a
  family (the holdout-gated ladder shape is untouched) and cannot lift an
  exactly-0.0 price off the DP's global minimum — the existing
  `activation_cost_unmeasured` candidate removal keeps full strength.
- Fail-closed: a run that would hand the DP a **mixed** scale (one family
  calibrated, another still uncorrected) refuses by name. A run with no
  measured activation rows anywhere corrects nothing, prints the verdict, and
  stamps it — no currently-legal run becomes illegal.
- Added `PRISMAQUANT_ACTIVATION_FAIR_PRICING` (default on; `0` reproduces
  prior pricing bit-for-bit), wired as the pipeline knob
  `ACTIVATION_FAIR_PRICING` and documented in `docs/design/runtime_flags.md`.
- Every candidate now records which estimator priced its activation contract,
  and the fit — sample, digest, residual band, per-rung dependence — is
  stamped into `format_applicability.json` and `selection.json`.

### Cross-family CB-ladder symmetry verdict (P5a)

- Fixed the audit's second asymmetry: the per-family RD-law ladders were never
  cross-calibrated. The expert cost stage now records each ladder's family and
  its **signed** holdout residual, and computes a family-symmetry verdict over
  held-out units with a tolerance derived the way `_cb_ladder_holdout_tol`
  derives its own — the sampling noise of the difference, floored at each
  family's declared resolution. No taste constant.
- A failure does not abort: it publishes
  `cross_family_comparison_publishable: false` with the numbers into the cost
  provenance, and the allocator republishes it in its diagnostics and
  selection provenance.

### Gridbook serving eligibility as candidate metadata (P5b)

- Fixed the audit's third asymmetry: the producer modelled exactly one
  gridbook kernel gate (`in_features % 256`). The `nvfp4_cb` serving profile
  now also declares the N-dimension load gates — `out_features % 8` for the
  fp4-CB families, `out_features % 16` for fp8-CB — per grid from the
  `cb_layout` family table.
- Added a declarative `serving_lanes` block: per CB format family, the served
  activation contract (`w8a8-dynamic-e4m3` vs `w4-bf16-bridge`), the fused
  mid-M rung set **as data keyed by the pinned Gridbook runtime version**, and
  the fallback route. Gridbook 0.5.0 backs FP8-CB fused mid-M for
  K ∈ {28,32,36,40,44,48}, and **Gridbook 0.6.0 — the version this release
  pins — backs the same set** (`nvfp4_cb.json` declares both keys; 0.6.0's
  K1.2 resolution proved `k % 4 == 0` is a format+TMA law, so the set is
  complete rather than pending); an undeclared runtime version backs nothing.
- Candidates carry the resolved route, and `selection.json` records which
  selected rungs ride a backed fused lane versus the expand+GEMM fallback —
  the producer-side mirror of gridbook K1.2, so neither repo can price an
  unbacked fast path.

### The constrained Pareto solver (P5c)

`docs/lanes/nvfp4-cb/format-speed-policy.md` §1 specified this solver and
deferred it ("not yet implemented"). It exists now; that paragraph has been
replaced with what it does and what still gates promotion.

- Added `prismaquant/serve_dispatch_table.py`: a torch-free declarative schema
  (`prismaquant.serve_dispatch_table.v1`) for measured per-(format-family,
  phase, M-regime, serving-lane) serving costs. **Provenance is mandatory on
  every row** — source document, date, GPU identity, measured quantity, units,
  and the derivation from the published number to the ratio — and a row
  without a source is a load error, not a defaulted field.
- Each `(phase, M-regime)` **arena** names exactly one reference route, so
  ratios measured against different denominators can never be silently
  composed (the 27B 1.44× is against a native artifact; the fused mid-M
  1.04×/1.26×/1.45× are against FP8-CB's own expand+GEMM route — multiplying
  them would manufacture a measurement). Isolated-operator (`operator_ms`)
  arenas and arenas with no published absolute are kept as evidence but are
  never SLO-eligible: policy §5, "raw standalone kernel timing is never served
  evidence".
- Shipped ONE example table,
  `prismaquant/serve_dispatch_tables/gridbook_gb10_2026-08-01.example.json`,
  populated **only** from measurements already published in Gridbook, each row
  citing its source. It is marked proposal data in both the file and the
  module docstring. It deliberately has **no whole-model NVFP4_CB row**: none
  is published, so an assignment containing NVFP4_CB cannot be certified
  against a latency SLO from it, and the evaluator refuses rather than
  interpolating.
- Added `prismaquant/serve_constraints.py`: policy §1's hard constraints
  (p95 TTFT, p95 ITL, p05 TPS, `resident + KV + peak_scratch`) evaluated on the
  exact expanded assignment. **No λ-blended objective anywhere.** Prefill and
  decode stay separate constraints. An assignment that misses an SLO is
  INFEASIBLE — removed from the candidate set, never re-ranked — and the
  objective and its tie-break (min predicted Δloss, ties toward the larger
  footprint) are unchanged among the survivors.
- Enforced at **assignment level**, in the byte-budget ratchet beside the
  exact byte filter, not inside `solve_allocation`. The DP is unchanged for
  the unconstrained case and that is pinned by test; the filter also sees the
  promoted, expanded assignment that actually ships, which the DP does not.
  The solver claims no global optimality it does not have: it stamps that
  every ACCEPTED assignment is feasible on both axes, not that the feasible
  set was enumerated.
- The aggregation model is explicit and stamped
  (`additive_layer_time__param_share_weighted__table_driven_proposal`) with
  **eight named assumptions** — additivity, parameter-share weighting, route
  locality, regime uniformity, baseline transfer, resident bytes, the
  single-stream `p05_TPS = 1000 / p95_ITL_ms` identity, and statistic
  transfer — carried in every artifact, along with policy §1's
  fastest-globally-feasible-assignment rule for any relative-tax denominator.
- Fail-closed: a unit with no dispatch row, an arena with no absolute
  reference, and an operator-microbenchmark arena all make the phase UNPRICED
  and therefore infeasible. "We could not price it" is never "it passed".
- Lane-aware pricing consumes P5b: a rung whose fused mid-M lane the pinned
  Gridbook version does not instantiate is priced with its **fallback** route's
  row, never the fused lane's. `FP8_CB_K36` (backed by 0.5.0) and
  `FP8_CB_K37` (not) therefore take different table rows despite sharing a
  family and a bpw class.
- `selection.json` records which constraints were active, which probed
  assignments the SLO axis rejected and the limit that rejected each, and
  which constraint binds at the shipped optimum. With no table and no SLOs
  supplied, every code path is byte-identical to the pre-P5c allocator apart
  from a stamp saying constraints were absent — pinned by an end-to-end test
  that compares `selection.json` and `layer_config.json` across a run with the
  feature absent and a run with it present-but-unused.
- New allocator flags: `--serve-dispatch-table`, `--serve-workload-mix`,
  `--slo-prefill-p95-ttft-ms`, `--slo-decode-p95-itl-ms`,
  `--slo-decode-p05-tps`, `--serve-device-budget-bytes`, `--serve-kv-bytes`,
  `--serve-peak-scratch-bytes`. Wired into `run-pipeline.sh` as
  `SERVE_DISPATCH_TABLE`, `SERVE_WORKLOAD_MIX`, `SLO_*`,
  `SERVE_DEVICE_BUDGET_BYTES`, `SERVE_KV_BYTES`, `SERVE_PEAK_SCRATCH_BYTES`
  and recorded in `STAGE_SETTINGS_ENV`. There is **no default workload mix**:
  policy §1 forbids one hidden in the allocator, and a latency SLO with no
  table or mix is refused by name.
- Design note: `docs/design/constrained_pareto_allocation.md`.

### D0.3 exact-rate experiment harness (P5d)

- Added `prismaquant/d03_exact_rate.py` and `scripts/run_d03_exact_rate.sh`:
  the two experiments gridbook ROADMAP **D0.3** names, run against a model's
  existing probe/cost artifacts. (i) `FP8_CB_K36` vs vanilla `NVFP4` on dense
  units at matched **exact whole-artifact bytes**, using the same non-additive
  accounting as the allocator's exact filter (shared CB codebook sidecars
  charged once per physical identity). (ii) Below 4.5 bpw, byte-neutral sweeps
  whose vanilla-NVFP4 promotions are **funded** by demoting other units down
  their own CB ladder, with a reclaim pass so each point sits at the baseline
  rate rather than under it.
- Each arm reports its assignment, exact bytes, predicted Δloss under the new
  activation-fair pricing, the P5c constraint verdict, and the serving-lane
  provenance (which selected rungs ride a backed fused lane).
- **Two refusals.** No cross-family verdict is printed when P5a's band check
  failed — suppressing it is that check's entire purpose, and printing it with
  a caveat would defeat it. No quality verdict follows when the two arms miss
  the ≤0.1% whole-artifact byte-match target policy §5 already names (the
  threshold the published 0.6B endpoint pair missed at +0.154%).
- **The harness prepares release-gate evidence; it does not claim it.** Every
  output is labelled proposal data pending the served NATIVE-PARITY protocol.
- Packed-expert vanilla NVFP4 is **excluded** from the contest and the
  exclusion is recorded explicitly in every report, citing gridbook **D0.2**:
  the producer profile denies stock NVFP4/FP8 on packed expert stacks because
  no stock-compressed-tensors packed-expert emit path exists, and building one
  is out of scope under the one-payload / no-new-packer rule.

### Routed-MoE stage attestation in the execution contract (gridbook K0.2)

- The NVFP4 W4A4 execution-contract record now carries a per-packed-FusedMoE
  stage section under `routed_moe_stages`
  (`prismaquant.nvfp4_w4a4_activation_stages.v1`). Each module attests BOTH
  stages — `w13` (the experts-module input) and `w2` (the routed intermediate)
  — with the stage label, the exact serialized physical target prefix, the
  input-global-scale policy, the calibration source that produced the scalar,
  and a per-stage value digest. A section digest covers the whole set. The
  scales were already stage-specific by construction (distinct physical
  targets; `unify_fused_sibling_input_global_scales` never joins across
  stages); what was missing was an attestation making that verifiable by a
  consumer.
- **Deliberate record-schema bump.** A record carrying the stage section
  declares `prismaquant.nvfp4_w4a4_activation.v2`; a dense-only record still
  declares `...v1` and is byte-identical to before. `target_values_sha256` is
  still framed with the **v1** literal, so the whole-model digest fields never
  move under the bump and an old reader verifies exactly what it always
  verified. The bump exists so a reader that cannot check stage attestation
  fails closed on a routed-MoE artifact instead of accepting a fused-readiness
  claim it cannot verify.
- `calibrated_input_global_scales_with_sources` reports which mechanism
  produced each scalar (target cache, parent experts-module cache, supplemental
  module-input sample, supplemental routed-intermediate replay, supplemental
  max-abs, packed-expert render max-abs). The stage attestation refuses an
  illegal pairing in either direction: `w2` can never be calibrated from the
  experts-module input, and `w13` can never be calibrated from a routed-
  intermediate replay.
- Fails closed exactly as before on a missing calibration input, and
  additionally makes it impossible to emit a routed-MoE artifact whose contract
  claims fused readiness with only one stage attested, or with no calibration
  source at all.
- All three emit paths build the section through one shared builder: the
  resident CB exporter, the streaming CB exporter (same inputs → byte-identical
  contract, as the existing resident-vs-streaming identity test requires), and
  the legacy native-compressed packed-expert path. The native container still
  publishes no `execution_contracts` record — its activation scalars remain
  optional/defaultable — but it now refuses to render a packed FusedMoE stage
  whose sibling stage has no calibrated max-abs.

### Gridbook runtime pin advanced to 0.6.0

- `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` now names Gridbook
  **0.6.0** at exact commit `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1` — the
  peeled `refs/tags/v0.6.0^{}` commit, not the annotated tag object. The
  required `pinned Gridbook contract` CI job installs that exact VCS revision
  and checks PEP 610 provenance, the packaged `runtime_contract.json`, producer
  profiles, rungs, layouts and emitted artifacts against it; all of it passes
  unchanged at 0.6.0, so no producer surface moved under this advance.
- The `nvfp4_cb` serving profile's FP8-CB fused mid-M lane now declares
  `"0.6.0": [28, 32, 36, 40, 44, 48]` — the SAME set as 0.5.0, and added as a
  NEW version key rather than by editing the old one. An artifact produced
  under the 0.5.0 pin therefore stays resolvable at the route it actually
  shipped on, and a pin with no key here still backs nothing (fail-closed).
- That set is now known to be **complete, not partial**. Gridbook 0.6.0
  resolved ROADMAP K1.2: `k % 4 == 0` is a format + TMA **law**, not a build
  option — `type_size = 4k` is the packed-B TMA box's contiguous extent and
  must be a 16-byte multiple, and the fused mainloop decodes with a single
  sub-table width `CbSubW = k/4` while the format splits `k` over `n_sub = 4`
  raggedly, so at k37 the true widths are `(10,9,9,9)` and a uniform decode
  would be *wrong*, not merely unaligned. The five off-law rungs of the
  published 27B K36..K47 ladder are therefore permanently expand+GEMM-served,
  and the allocator prices them on the fallback row for good rather than
  pending coverage that cannot arrive. The compiled set is queryable
  (`cb_fused_kbits()`), so the declaration is checkable against the runtime
  rather than transcribed from it.
- **No FP4-CB backed set was added.** Gridbook 0.6.0's contract-preserving
  FP4-CB v2 fused mid-M kernel exists only as an opt-in behind
  `PRISMAQUANT_CB_FP4_FUSED_MIDM=1`, pending its served NATIVE-PARITY gate;
  with the flag unset the dispatch is byte-for-byte the BF16 bridge. The honest
  backed set for the DEFAULT contract is therefore still empty, and the lane's
  `detail` records the available-versus-backed distinction and cites the flag.
  Pricing a rung on a lane the default serve never takes is exactly the P5b
  defect this data exists to prevent.

## 0.5.2 — 2026-08-01

This patch release advances the immutable runtime boundary to Gridbook 0.5.0
at exact commit `593f524e0a5d73b18e56d290a7b1355e66b2f9ce`.

Gridbook serving is now native CUDA/CUTLASS-only. Required native kernels are
attested at model load and missing or ineligible kernels fail closed instead of
falling back to Triton or another serving implementation.

The PrismaQuant producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged. This release makes no DeepSeek-V4
(DSV4) qualification or support claim; that work remains paused.

## 0.5.1 — 2026-08-01

This patch release makes fused NVFP4 W4A4 artifact eligibility explicit and
auditable while keeping fused serving opt-in. It does not claim default
enablement or a served-quality promotion.

### Versioned fused-activation contract

- Added one versioned NVFP4 W4A4 execution contract for production FP4-CB
  exports, including calibrated per-target `input_global_scale` tensors,
  fused-sibling scale unification, and a digest binding the serialized mapping.
- Added fail-closed coverage and provenance checks plus a serve-faithful
  activation-QDQ oracle. Legacy and unstamped research artifacts remain
  readable by their baseline paths but are not eligible for static fused
  dispatch; Gridbook's explicit rowwise fused research path remains available.

### Streaming and exact accounting

- Made resident and streaming exporters share the same activation-contract and
  served-target namespace rules, including packed-expert calibration synthesis.
- Accounted for FP4-CB activation-scale tensors and stock NVFP4 sidecars in
  whole-artifact bytes and bit totals, including weight-only W4A16 targets.

### One scale policy owner

- Consolidated activation-scale formulas, fused-unit grouping, calibration,
  and legacy compatibility behavior in one producer-owned module so native and
  CB exporters cannot silently drift into different contracts.

## 0.5.0 — 2026-08-01

This release establishes the production boundary between PrismaQuant and
Gridbook. PrismaQuant owns quantization, allocation, serialized-byte accounting,
and artifact export; Gridbook alone owns serving code, kernels, runtime flags,
tests, packaging, and releases.

### One runtime, one producer contract

- Deleted the complete vendored Gridbook runtime, CUDA/HIP sources, runtime
  tests, and source-sync machinery (38,216 lines removed).
- Added one immutable Gridbook commit pin, PEP 610 provenance checks, a packaged
  consumer contract, and tiny real-artifact compatibility tests.
- Consolidated producer-owned CB layout and export metadata in
  `cb_layout.py` and `cb_export_config.py`, shared by resident and streaming
  exporters.
- Moved the sole Gridbook pin and resolver into packaged assets under
  `prismaquant/gridbook_runtime/`. This is an intentional 0.x interface change:
  external scripts that sourced `scripts/lib/gridbook_runtime.sh` must source
  `prismaquant/gridbook_runtime/gridbook_runtime.sh` instead.

### Exact accounting and constrained selection

- Unified serialized-payload accounting across candidate construction,
  allocation, reporting, and exporter assertions, including FP4-CB layout-v2
  scale planes, FP8 per-row scales, and shared codebook sidecars.
- Replaced blended latency scoring with quality minimization under exact whole-
  artifact bytes, phase-specific serving SLOs, memory, backend, shape, TP, and
  serving-unit constraints.
- Excluded signed S13-S16 rungs from production menus while retaining research
  export and decoder compatibility.
- Fixed partial LFM packed-expert CB export layouts.

### Packaging correctness

- Wheels and sdists now include the canonical IQ grids and NVFP4/FP8-CB lattice
  tables. Earlier distributions omitted them: IQ failed at first use and CB
  could silently regenerate expensive lattices.
- The shipcard CLI is now installed as
  `python -m prismaquant.shipcard_cli`; the packaged pipeline no longer points
  at a checkout-only `tools/shipcard.py`.
- Distribution and clean installed-wheel gates now exercise every model,
  serving and lane spec, both tensor-table assets, the exact Gridbook pin and
  resolver, the pipeline, and the shipcard CLI.

### Fused NVFP4 safety decision

Gridbook's installed-wheel CUDA operator gate passed, but the teacher-backed
LFM2.5 A/B rejected promotion (exact full-vocabulary KL 0.247178, delta NLL
+0.054964, perplexity +5.65%). Dense and grouped fused-NVFP4 paths therefore
remain explicit opt-ins and default off in Gridbook 0.4.1.

## 0.4.1 — 2026-07-30

Tied-embedding models could not be quantized at all. Found by running the
pipeline on a real checkpoint rather than by reading code.

### A tied `lm_head` is structurally non-quantizable

On `google/gemma-4-31b-it` the cost stage cleared all 60 body layers, skipped the
vision-tower shards, then died on the `lm_head` shard with
`NotImplementedError: Cannot copy out of meta tensor`. Cause: the config declares
`tie_word_embeddings: True` and the checkpoint ships **no `lm_head` tensor at
all** — only `model.language_model.embed_tokens.weight` — so `lm_head.weight` is
a tied alias that nothing materialized. `tie_word_embeddings` appeared nowhere in
the streaming or cost path; there was no weight-tying support. Every
tied-embedding model hit this, which is most of the Gemma family; it went
unnoticed because every shipped artifact (Qwen3.6-27B, Qwen3.5-35B-A3B, Hy3) is
untied.

The head is now **materialized** (phase-2's CE backward runs through it, so meta
is never acceptable) via transformers' own `get_output_embeddings()` /
`get_input_embeddings()` accessors, so no embedding path is hardcoded and the
VL-prefixed name resolves like the plain one. Detection is from the config
declaration plus the index's absence of a head tensor — never a name guess. A
meta head with **no** declared tie now raises immediately instead of surfacing
thousands of lines later.

And a tied head is **excluded from probe, cost and the DP**, rather than
measured. Tying means one `Parameter`: quantizing the head would quantize the
embedding, and the surrogate cannot see that cost — probe and cost measure only
the head's output MSE, while the identical perturbation enters every token
embedding and thus layer 0's input for the whole forward, which no surrogate and
not even the L2 perturbed-X fixed point observes. There is also nothing to
re-encode: a tied source has no `lm_head.weight` bytes, so the footprint would
either fail to resolve the name or subtract the embedding from the floor while it
still ships verbatim. The codebase had already reached this conclusion in one
place — `aura_cost.py` hard-raises on a tied head with the same argument — so
this makes automatic what was an operator instruction, and extends it to the
L1/L2 path AURA does not cover. The exclusion deliberately ignores
`--allow-pinned lm_head`, because the tie is a property of the checkpoint rather
than of the serving profile.

Also removed: an ad-hoc repair in the probe that hardcoded three embedding names
inside a `try/except Exception` that only warned.

### Measured end to end

With this fix, Gemma4-31B completes **probe → cost → allocate → export** for the
first time. The probe was already passing (411 rows, all nonzero `h_trace`, 60
layers); cost now completes with zero errors; the allocator hits
`achieved_bits=6.000` with a genuinely heterogeneous 244 NVFP4 / 119 FP8 / 27
BF16 assignment; and the export writes a 27.18 GB compressed-tensors artifact
whose `config_groups` carry 4-bit `tensor_group` and 8-bit `channel` schemes,
with `tie_word_embeddings` preserved and **no `lm_head` tensor** — the embedding
ships once, so the tie is not silently materialized into duplicated bytes.

That run used a deliberately tiny calibration (2 samples, seqlen 512) to reach
failures fast. **It is an enablement result, not a quality claim** — the artifact
has not been served and no KL/PPL has been measured.

## 0.4.0 — 2026-07-30

Closes #29 and lands the KV-cotangent path, which removes the default-off guard
that was blocking KV-sharing architectures. Minor rather than patch because the
Fisher measurement for KV-sharing models changes (it was wrong), and because an
export that previously succeeded by silently demoting FP8_SOURCE now behaves
differently. Shipped allocations are unchanged (35B: 0 of 500).

### Fisher: the KV-cotangent path (part of #9, closes MINOR-M33)

Gemma4-style architectures share K/V across layers: a "storing" layer computes
K/V and later "sharing" layers consume them. Phase-3 forwards each layer in
isolation and handed the consumer a **detached** K/V, so its backward stopped
at that boundary and the storing layer's `k_proj`/`v_proj` Fisher never saw any
consumer's contribution — an under-count on precisely the layers that feed other
layers. That is why `num_kv_shared_layers > 0` was blocked behind
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER`.

Consumers are now handed grad-enabled leaf clones; their `.grad` is the cotangent
each contributes, accumulated per storing layer and used to seed that layer's
backward alongside its own output cotangent. Phase-3 sweeps in reverse and
`kv_shared_layer_index` is derived from layers strictly below the sharing point,
so every consumer is harvested before its producer is forwarded — one pass, no
disk state. Both facts are pinned against the installed modeling source.

**Verified by exact equivalence, not plausibility:** on an fp64 synthetic model,
`h_trace` through the isolated protocol is bit-identical to a single end-to-end
autograd backward (relative error 0.00e+00), while the pre-fix protocol
under-counts `k_proj` by 85.1% and `v_proj` by 38.5%.

Three things the equivalence surfaced that the design did not predict:

- **The under-count was never confined to k/v_proj.** Phase-3 chains each layer's
  input gradient downward, so the producer's truncated input gradient was
  inherited by every layer *below* it — all of `layers.0.*` moves without the fix.
- **The Fisher hook must fire exactly once.** These hooks pop their saved forward
  input, so a backward hook firing once per root would silently drop half the
  Fisher; both roots go through one `torch.autograd.backward` so autograd
  accumulates at the shared node first. Pinned by counting hook invocations.
- **A borrowed leaf must never be seeded as a root.** In reverse order a consumer
  is handed a container keyed identically to the entry the previous consumer just
  filled; seeding it would inject one consumer's cotangent into another's harvest.

The guard is inverted rather than deleted: it now fires only when the cotangent
path is unavailable (`PRISMAQUANT_KV_COTANGENT=0`), and
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` still reproduces a pre-fix probe. Models
without KV sharing are bit-for-bit unaffected with the accumulator on or off.

**Honest limit:** no real `num_kv_shared_layers > 0` checkpoint has been probed.
The percentages above are a correctness demonstration on a toy; the real-model
magnitude is unmeasured. Three conditions would still make such a probe unsafe
and are documented in code: non-differentiable shared state (no cotangent
exists), cotangent left unclaimed at sweep end, and an architecture whose
consumer sits below its producer — the last two surface as diagnostics rather
than wrong-but-silent numbers.

### Export: passthrough source integrity (#29)

The runtime coercion never passed `source_kind`, so passthrough-integrity judged
**every** `FP8_SOURCE` Linear illegal and rewrote it to BF16. The bytes were fine
(the config overlay restored it, materialization copied verbatim) but every
FP8-source artifact's `runtime_coercions` was full of demotions that never
happened — making a real coercion invisible — and it forced a passthrough
exemption in 0.3.1's serving-group escalation.

The source dtype now comes from `_scan_source_dtype_manifest`, the same
recipe-keyed map `allocator.main` feeds `build_candidates` to gate passthrough
candidates, so the exporter judges legality against exactly the vocabulary the
gate that admitted the allocation used. It is scanned lazily, so a BF16-source
export does no extra header IO. Bogus rows went from 4-of-4 to 0 on a synthetic
fp8 checkpoint, and the exemption is gone, so a genuine passthrough mismatch
inside a serving unit now escalates like any other illegality.

No bespoke raise was added, on a measurement: a genuinely non-fp8 `FP8_SOURCE`
assignment is repaired by the coercion rather than the overlay, so it already
ships as BF16 today and a hard raise would be the only change turning a
succeeding export into a failure. The 0.3.1 policy decides instead — refuse
inside a serving unit (naming the legal rungs and the byte cost), coerce alone
when dense, now with a true `delta_bytes` and a passthrough-specific banner.

One required side-fix: with `FP8_SOURCE` surviving the guard it reached
`_production_cache_expected_keys` for the first time, and since its emit branch
returns before the packer, that check would have demanded a render entry nothing
reads — newly failing a valid FP8-source export. All passthrough formats are now
skipped there, not just BF16.

### Streaming: text-only skeletons for vision-language wrapper configs

`CALIBRATION_MODALITY=text-only` decided *what to calibrate on*, but it also
silently decided *how the skeleton is built*: the multimodal path instantiates
via the declared architecture specifically to bypass `AutoModelForCausalLM`'s
text-only downgrade, while the text-only path passed the top-level config
straight to `AutoModelForCausalLM` — which fails on any model whose wrapper
config is not in that mapping (reported on MiniMax-M3 in #12, where the error's
own accepted-class list contains only the model's *text* config).

The text-only path now falls back to the text sub-config **class** rebuilt from
the staged top-level keys. That distinction matters: `stage_text_only` pops the
nested `text_config` and lifts its keys up, so the sub-config object still
hanging off the wrapper is default-constructed and reading dimensions from it
would silently build a wrong-sized skeleton. Detection asks the same two
questions `from_config` asks itself (remote-code `auto_map`, then membership in
the mapping the call consults), so it is config-only and contains no model-type
or class name; a config the auto class can resolve returns the identical object
and takes the original path unchanged.

**This makes no architecture supported.** MiniMax-M3 still needs a
model-structure profile and a serving profile, and the mechanism was validated
against two unrelated real wrapper families since no M3 checkpoint exists here.
The next wall for any VL checkpoint is tensor-name matching (a text skeleton
expects `model.layers.*` where a VL checkpoint often ships
`model.language_model.layers.*`), which is per-architecture profile work.

## 0.3.1 — 2026-07-30

Closes #28: a serving-atomic group could end up with a quantized + BF16 mix
inside it, reported only by the fused-coherence gate at the very end of export.
Fixed at both the cause and the safety net. Allocations on the shipped
Qwen3.6-27B and Qwen3.5-35B-A3B are unchanged (0 of 614 and 0 of 500).

### Cause: promotion now picks a format the whole unit can run

`_promote_group_components` took the highest-**rank** format assigned to any
member of a serving-atomic component and wrote it to all of them, with no check
that the format was legal for the rest — it only received `assignment`,
`format_rank` and `groups`, so it had no way to know. Members of one unit do not
share a shape (gate_up vs down differ on the reduce dim; an odd
`moe_intermediate_size` makes one projection's group/scale-block divisibility
fail while the other's passes), so the promoted format could be illegal for a
subset.

Promotion now takes per-row legal-format sets (`legal_formats_from_candidates`)
derived from the candidate lists, which already encode source-passthrough
integrity, serving-profile rules, group/scale-block divisibility and kernel
shape rules. It picks the cheapest legal-for-all format at or above the max
rank — preserving promotion's non-degrading contract, which
`solve_with_promotion`'s tightening loop is built around — and only downgrades
to the highest legal-for-all when nothing above is common. In the illegal case
every member is written unconditionally, since a member on an equal-rank but
different format would otherwise survive and leave the unit mixed. No common
legal format raises, naming every member with its legal set and the three
upstream causes.

The argument is optional: omit it and the legacy max-rank path runs verbatim, so
callers that cannot supply legality (auxiliary MTP/visual pins, hand-built
assignments) keep today's behaviour rather than acquiring a new failure.

Two paths were genuinely reachable and are now covered: the un-aggregated
(`--no-packed-aggregation` / `--no-fused-aggregation`) solve path, where
promotion is the only coherence mechanism — there the pre-fix symptom was
actually an aborted run, since `compute_achieved` refuses to price an
unpriceable member — and the Pareto seed-JSON promotion, which is **not** priced
by `compute_achieved` and so could let an illegal member format escape silently.
The aggregated path already intersected member candidate sets and needed nothing;
that is now pinned by a test rather than assumed.

### Safety net: export coercion is group-aware

The per-Linear shape/policy coercion (deliberately preserved in 0.3.0) could
rewrite a single member of a unit to BF16. It now resolves whole serving-atomic
components, unioning overlapping units — on the split per-expert representation
a Linear can be both a fused sibling and a packed-expert member — using the same
profile accessors the fused-coherence gate uses, never by parsing names.

The resolution is deliberately asymmetric. If some emittable quantized format is
legal for every member, export **raises** and names it: coercing would ship the
whole unit at 16 bpp (for a packed-expert unit, `num_experts ×` the per-Linear
cost), the dimension that made one member illegal is model-wide so it recurs in
every layer, and a re-solve lands the unit on that legal format for free.
Export must not substitute a format itself — the format the allocator picked is
the one the production weight cache holds a deliberate render for, so a
substitute is a cache miss at best and an RTN render at worst. Only when no
quantized format is legal for every member is BF16 the sole representable
answer; then the whole unit is coerced, loudly, with every member and the byte
delta recorded in `runtime_coercions` and the BF16 audit.

Since the cause is fixed upstream, this path should be unreachable in normal
operation, and the report is written to make any firing look like the upstream
regression it would be. One case it must keep catching regardless: rank-1 legacy
probe stats carry no shape, so `check_stats_format_applicability` admits a
shape-illegal format legitimately and the exporter is the only gate.

## 0.3.0 — 2026-07-30

Closes the three open issues that were ours (#27, #19, #9 item 1). Minor rather
than patch because an export that previously "succeeded" can now raise, and
because a vendored-modelling override that cannot take effect now stops the run
instead of silently continuing on the wrong code.

### Export refuses what it cannot emit (#27)

`export_native_compressed` now declares `EXPORTABLE_FORMATS`, derived from
`FORMAT_SCHEME` plus the container passthrough rather than hand-listed, and the
vLLM serving lane reads its menu from that one place. A format with no
compressed-tensors emit path is a **hard error** naming the Linear, the format
and the resolved profile — it used to be silently rewritten to BF16 with only a
`print`, so a Linear allocated at ~4.25 bpp would ship at 16, blowing the byte
budget and leaving the artifact's real bpp disagreeing with its own
`layer_config.json`.

The *legitimate* coercion is unchanged: a format the exporter can emit but which
is shape-illegal or profile-denied still falls back to BF16 and is still audited
into `mixed_native_manifest.json`. Two facts corrected by reading the exporter:
`FP8_SOURCE` **is** emittable (verbatim-copy path, no packer branch) and
`FP8_E5M2` is **not** (packer branch, no scheme entry) — so the set cannot be
derived from the packer branches. Menu unchanged for every lane.

Consequence worth knowing: allocating under the `research` profile and then
exporting compressed-tensors now fails loudly instead of shipping ~16 bpp.

### Vendored modelling overrides verify or die (#19)

`register_qwen3()` returned cleanly, set its "registered" flag, and on
transformers ≥ 5.13.0 did nothing — after which a probe ran **upstream** Qwen3
modelling code, on the architecture family behind most shipped artifacts, with
no exception anywhere. Root cause is upstream: `_LazyAutoMapping.register`
returns early whenever the config key's `__module__` starts with
`transformers.`, so no override of a natively-supported `model_type` can land
through that call.

- The override now genuinely applies, via public API only: a PrismaQuant-owned
  subclass of the native config (same `__name__`, non-`transformers.`
  `__module__`, picklable) registered through `AutoConfig.register`, which
  applies no such filter. No transformers internals are patched, and the
  fallback engages only when the direct route is verified dead.
- Every registration is **verified** by resolving it config-only, and a failure
  raises with the transformers version, the resolved class, the upstream
  file/function and the remedy. The "registered" flag is set only after
  verification, so a failure stays retryable rather than caching as done.
- `register_deepseek_v4` had a second silent no-op of its own and was resolving
  correctly only by module-path hijack; it now gets the same verification plus a
  guard against a foreign module occupying its path.
- `detect_profile` no longer loses that verdict: it consults the recorded
  override failures and refuses to hand back a profile whose vendored path is
  known dead. The surrounding `except Exception: pass` is correct for keeping
  detection alive, but it cannot be allowed to re-hide a silent no-op.
- The version boundary is now measured, not guessed: healthy through 5.12.1,
  broken from 5.13.0. The old `xfail` threshold of 5.7 was six minor versions
  pessimistic, and the `xfail` is gone — the suite goes red on the wrong
  modelling path.

### Gemma4 KV-sharing pass state (#9, item 1)

The per-forward-pass state hook had already landed; what was missing is that
`_save_precompute_cache` never persisted it and the load path omitted the field
entirely. Since the precompute cache is the normal path for a sharded or
resumed probe, the first checkpoint with `num_kv_shared_layers > 0` would
capture the shared K/V in phase 1, silently drop it on save, and `KeyError`
inside attention for every sharing layer in phase 3 — after hours of phase-1
work, untested in either direction. Fixed both ways, an old cache now hits a
loud error rather than a `KeyError` deep in attention, and a sharing layer with
no captured source K/V raises naming the layer and the remedy instead of
handing back an empty dict. The merge of pass state into per-layer kwargs is now
one shallow-by-design function that raises on key collision instead of silently
overriding.

Item 2 of that issue is unchanged and still needs a GPU run. Two caveats worth
carrying: `google/gemma-4-31b-it` has `num_kv_shared_layers = 0`, so the sharing
path is covered only synthetically until a genuinely KV-sharing checkpoint is
probed; and KV-sharing probes remain default-off because phase-3's isolated
forward detaches the borrowed K/V, under-counting the storing layers'
`k_proj`/`v_proj` Fisher — a cost-model gap, not a flag.

### Also

`F8_E8M0` reporting, the verified DSv4 source layout, and the routed-only
expert-declaration scope all shipped in 0.2.1 and are unchanged here.

## 0.2.1 — 2026-07-30

Corrects one thing that shipped in 0.2.0 on an unverified assumption, settled by
pulling the real `deepseek-ai/DeepSeek-V4-Flash` config and safetensors headers
(a few hundred KB — no weights) plus the authors' `inference/convert.py`.

- **Declared-MXFP4 expert scope is routed-only again.** 0.2.0 widened it to
  `shared_experts.*` on the reasoning that `expert_dtype` describes all of a
  layer's experts. The headers refute that: routed-expert weights are `I8`
  nibble-packs (2304/2304 sampled) while shared-expert weights are `F8_E4M3`
  block-FP8 (9/9), and the authors' converter gates its fp4 path on
  `"experts" in name and dtype == torch.int8`. With the widening in place a real
  DSv4 load would have pushed block-FP8 into the nibble decode and hard-failed
  the packed-grid assertion. No other model is affected — the widening only ever
  applied to a checkpoint declaring `expert_dtype: fp4`.
- `F8_E8M0` added to the safetensors dtype table (it fell to the unknown-dtype
  default of 2 bytes in `dominant_source_bytes_per_param`; span-based accounting
  was already exact).
- The verified DSv4 source layout is recorded in
  `model_profiles/specs/deepseek_v4.json`, including the accounting trap that
  its scales are 1-byte E8M0 planes named `.scale` rather than fp32
  `.weight_scale_inv`, so DSv4 byte accounting must use the per-tensor manifest.

Everything else confirmed as the code already assumed: the `expert_dtype` key
and value, `scale_fmt: ue8m0`, the E8M0 exponent bias, the packed grid, and — the
question that previously could only be guessed — the nibble order and E2M1 table,
which match the authors' reference decode value-for-value.

## 0.2.0 — 2026-07-30

First published release. 0.1.0 existed only as a version string in
`pyproject.toml` and was never uploaded anywhere.

Everything below is allocator/solver/footprint/profile logic. **No shipped
artifact is affected:** re-solving the shipped Qwen3.6-27B and Qwen3.5-35B-A3B
probe/cost pairs at `TARGET_BITS` produces the same assignments as before (0 of
614 and 0 of 500 changed), and the byte-budget floors are byte-identical on both
real checkpoints (27B 6.012 GB, 35B 4.661 GB).

### Allocator

- **Fisher `h_trace` is normalized by the global calibration token count**, for
  every row. Per-routed-token normalization inflated a rarely-routed
  per-expert-`nn.Linear` row by `global/routed` (typically `n_experts/top_k`),
  i.e. inverted importance weighting. Tokens never routed to an expert
  contribute a genuine zero that belongs in the mean-Δloss average. Dense and
  packed-3D probes are numerically unchanged; existing probes are corrected at
  load time from their stored raw accumulators, and a probe carrying raw
  accumulators without token metadata now hard-fails
  (`--allow-legacy-fisher-norm` restores the old warning path). This reverses
  the convention audit M4 documented; the reversal is recorded in `CLAUDE.md`
  §3 and `docs/prismaquant_design.md` §2.2.
- **Packed serving groups are first-class DP units.** A packed-MoE serving
  group is atomic at serve time but the DP priced upgrades per row while
  promotion charged the whole group — a systematic mispricing that starved
  cheap dense rows. `aggregate_packed_serving_groups` collapses each group into
  one multi-choice item, so the DP and the serving constraint price identical
  moves and post-DP promotion is a validated no-op. `--no-packed-aggregation`
  restores per-row pricing.
- **Solver termination is feasible-only.** A rung either returns an iterate
  satisfying `achieved <= target + tolerance` or reports INFEASIBLE; deep
  undershoot is recovered by bisection rather than shipped. Among feasible
  iterates the solver keeps **minimum predicted Δloss** (ties to denser), which
  is its actual objective — denser is not monotonically better. `--target-bits`
  runs that previously emitted an over-budget config now exit, with the format
  floor, what the floor solve promotes to, and the closest achieved bits in the
  message.
- **Byte-budget "fit the card" selection ships minimum predicted Δloss among
  the rungs that fit** (ties to the larger footprint), matching the solver's
  objective. Filling the card is a proxy that can select a denser artifact with
  worse predicted loss than a sparser one that also fits. `selection.json` is
  self-describing (schema `…byte_budget_selection.v2`): objective, feasibility
  test, the tightened search ceiling, whether bisection ran and why not, the
  full ratchet trace, and the max-bytes pick for comparison.
- **Bit-exact re-encode pricing is gated on an identity activation path.** A
  measured `weight_mse == 0.0` proves `W' == W`, but for W·A· formats the
  measured `output_mse` is real activation-side error, so pricing such an entry
  at zero Δloss handed the DP an unbeatable global minimum for a W4A4
  assignment. The short-circuit now requires
  `FormatSpec.act_quant_changes_input` to be false — a dtype-level declaration,
  pinned registry-wide. Relatedly, a W·A· candidate whose activation cost was
  never measured and prices at exactly 0.0 is excluded from the menu with a
  counted, logged reason instead of winning every budget.
- **Fused-sibling and packed-group UCB hedges aggregate in quadrature**
  (`z·√Σ(stderr·gain)²`) instead of linearly, which over-hedged an N-member
  group by up to √N. Byte-for-byte identical at the default `COST_UCB_Z=0`.
- **`--packed-role-split` hard-errors unless the resolved serving profile
  declares `supports_per_role_expert_schemes`** (GGUF only). It could otherwise
  emit gate_up=NVFP4 with down=FP8 in one MoE layer — a checkpoint vLLM cannot
  load. Role grouping now comes from the model profile rather than a projection
  table inside the allocator.
- A fused group whose members have disjoint format menus, and an assignment
  row whose format has no candidate to price it, are now hard errors naming the
  group/row instead of silently vanishing from the DP or scoring as free.

### Footprint

- **Source bytes are priced from an exact per-tensor safetensors-span
  manifest.** The regime-wide accounting charged every re-encoded Linear at the
  FP8_SOURCE layout as soon as any fp8 dtype appeared, which on a mixed source
  removed more bytes than the checkpoint holds and drove the non-quantizable
  floor negative — letting an artifact twice the budget "fit". A negative floor
  is now always a hard error.
- Tensors whose live-name mapper declines them (MTP sidecars, visual towers)
  keep their source bytes: the mapper answers "is this in the live graph", not
  "does this have bytes on disk". Without this, every `--target-disk-gb` run on
  an MTP-carrying model failed.
- Two re-encoded names resolving to the **same** source span is rejected
  structurally (the manifest carries per-entry span provenance), not by
  docstring convention. Charging both a per-expert name and its packed parent
  would subtract the expert mass twice.
- One accounting path: the byte-budget selector calls
  `footprint.assignment_artifact_bytes` rather than reimplementing the identity,
  and `source_manifest` is a required keyword so the legacy regime
  approximation cannot be reached by omission.

### Serving profiles

- **A profile's format menu is bounded by its lane's exporter**, read from the
  exporter's own declaration (`export_native_compressed:FORMAT_SCHEME`,
  `gguf_formats:GGUF_BLOCK_BYTES`) rather than a duplicated list. Weight-only
  A16 rungs were legal for dense Linears on the vLLM lane while the exporter
  cannot emit them. No production format was narrowed; GGUF is unchanged.
  `research` declares `emulation_only` instead of a lane, deliberately, so
  unserved rungs stay measurable.

### Probe / streaming (DeepSeek-V4-Flash enablement)

- MXFP4-packed routed experts dequant on a dedicated vectorized path, triggered
  by the checkpoint's `expert_dtype` declaration rather than a tensor-shape
  heuristic, with the packed grid and the E8M0 scale-plane dtype as assertions.
  Shared experts are covered by the same declaration. E8M0 `0xFF` decodes to
  NaN. Bit-exactness is pinned against an independent scalar reference.
- Nibble-packed `I8`/`U8` expert tensors are sized at 2 logical elements per
  disk byte by both pre-load cache estimators; sizing them verbatim under-counts
  the resident tensor 4× and makes prefetch silently refuse layers.
- Compressed-sparse-attention layer types and the rope-axis `layer_types` dict
  are handled; phase-1 activations stream to host per layer
  (`PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER=1` restores the batched transfer).
- Per-expert cost rows resolve, and `PRISMAQUANT_EXPERT_COST_SAMPLE` works on
  the default `COST_MODE=production-render-score` path.
- h-detail blobs record their normalization denominator and a stale directory is
  refused rather than mixed with differently-normalized scalars.

### Packaging / CI

- CI runs the test suite on every push and pull request (Python 3.11 and 3.12,
  CPU torch) plus an import-surface job.
- Tag-driven release pipeline (`docs/RELEASING.md`): builds, asserts the tag
  matches the built version, asserts the runtime JSON specs and
  `run-pipeline.sh` are packaged in both wheel and sdist, verifies a
  non-editable install resolves those specs from site-packages, then publishes
  to PyPI via Trusted Publishing (no API token) and creates the GitHub Release.
- `prismaquant.__version__` is resolved from installed metadata, so
  `pyproject.toml` stays the single source of truth.
- Three tests that drive repo-root `tools/` scripts skip cleanly when run
  against an installed package instead of failing collection.
