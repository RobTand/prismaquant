# Unary test recorder consolidation

PRI-226 continues under open #1295. This bounded slice removes duplicated
recording behavior from twelve test modules. It does not close the program.

## Shared contract and scope

`tests/delegation_recording.py:4` owns `record_unary_calls(monkeypatch, module,
name)`. It captures the current unary target, appends the original argument
before calling that target, and returns the same result or propagates the same
exception. An absent target or explicit `None` returns `None` after recording.
The returned list stays mutable. Pytest owns installation and restoration,
including temporary attributes and nested recorders.

Six S1 helpers and six S2 dictionary factories repeated this behavior. The
migration deletes their recording bodies and uses the one ordinary test
support owner. S2 callers keep their `hash` and `dumps` dictionary keys and
installation order. These subjects are ordinary modules with independent
named attributes. Capturing the second owner after installing the first does
not change those modules' bindings. No general proxy/object protocol is added.

The model-walk test's recorder delegates to a separately imported owner, so
its binding contract stays separate. Other recorder and file-loader candidates
in the support census remain unadjudicated. This is not an all-test-helper
closure claim. Independent JSON/hash expected-value expressions and existing
refusal assertions in all twelve routing modules are unchanged.

An explicit support function was chosen over a conftest fixture factory.
Callers name the patched module and owner; no autouse fixture, cache, mock
framework, production digest migration or source-loader API is introduced.
The shared abstraction was committed before migrating its callers.

## Complete bounded caller roster

The source before migration is `05b62b4742793a2188abd4ed041fc4d46fd6c2c0`.
These are all twenty calls of the twelve removed private helpers. There are
23 direct shared-owner invocations afterward because three dictionary
factories each installed two recorders.

| Old helper path and definition | Direct local calls before migration |
| --- | --- |
| `tests/test_census_compat_routing_1703.py:20` | `test_publish_load_execution_routes` at :33; `test_publish_load_execution_routes` at :34; `test_create_compat_routes` at :43; `test_create_compat_routes` at :44 |
| `tests/test_catalog_ext_routing_1710.py:26` | `test_fence_hashes_route_to_file_owner` at :44; `test_create_extension_envelope_routes_to_owners` at :63; `test_create_extension_envelope_routes_to_owners` at :64 |
| `tests/test_chain_seed_routing_1705.py:19` | `test_envelope_routes_to_owners` at :32; `test_envelope_routes_to_owners` at :33 |
| `tests/test_glm_oracle_routing_1707.py:19` | `test_oracle_routes_to_owners` at :32; `test_oracle_routes_to_owners` at :33 |
| `tests/test_slice_envelope_routing_1699.py:28` | `test_manifest_bytes_routes_to_indent2_owner` at :39; `test_hash_sites_route_to_bytes_owner` at :46 |
| `tests/test_prismasnap_routing_1713.py:19` | `test_probe_hashes_route_to_owner` at :32 |
| `tests/test_adjoint_band_routing_1735.py:17` | `test_json_file_hash_routes_to_owner` at :30 |
| `tests/test_layer_quanta_routing_1733.py:17` | `test_layer_quanta_hash_routes_to_owner` at :58 |
| `tests/test_stage_retirement_routing_1737.py:16` | `test_successor_pinned_check_routes_to_owner` at :37 |
| `tests/test_checkpoints_routing_1723.py:21` | `test_checkpoint_hashes_route_to_owners` at :50 |
| `tests/test_handoff_routing_1720.py:28` | `test_handoff_hashes_route_to_owners` at :48 |
| `tests/test_stage_a_routing_1731.py:19` | `test_stage_a_hashes_route_to_owners` at :52 |

The rerunnable `migrate_recorders.py` under the evidence directory binds the
explicit roster to the old commit, refuses unexpected source, and verifies
an already migrated file without changing it. One file was migrated by hand
first, then compared with the tool's expected output. A second tool pass
recognized all twelve migrated files. `recorders-migration.json` retains
before/after hashes and caller locations; `recorders-callers.json` lists the
new direct calls. These scripts are review evidence rather than production
entrypoints.

## Fleet evidence

All commands use published PrismaBuild. Evidence lives under
`/mnt/shared/paperclip-runs/PRI-226/` and in the uploaded recorder bundle.
`recorders_jobs.py red|green` retains the exact client argv and timeout.
The absolute pinned interpreter is available on the x86 worker. The initial
untagged submission was refused because that interpreter is absent on the
submitting box; the retained x86-tagged action uses the existing environment.
No idle-host measurement was submitted.

| Check | Action | Result |
| --- | --- | --- |
| Frozen old helper and missing shared owner | `18bbeb280c3a201acd91c480cd040a6f66be285dc21e5a8b55d73cf14172ac9d` | Six frozen-path cases pass; six shared cases error solely on missing module |
| Final contracts, all twelve routing modules and compile checks | `9f0073d6f8872b17f4e3667e81c7517368fce03ce17eadfb6938f1929aa9bf9b` | 39 passed, zero failures/skips, 14 Torch/Python 3.14 deprecation warnings; all 15 touched Python files compile |

The frozen helper is an unchanged copy of the general S1 helper from
`tests/test_census_compat_routing_1703.py:20-30` at the old commit.
Its function AST matches that source. The independent observable assertions
run against both implementations: argument/result identity, append-before-call
with argument mutation, captured target after rebinding, absent/None target,
exact exception identity and nested restoration. No sleeps or timing limits
appear in these tests.

Final tested source is `f8b5b639e6a9b4528ffd0c702783c3b132d0ff60`; its immutable PB snapshot is `9d270137035fdbae1fdf335fd5cace330712739f`. All tracked blobs match. The only snapshot-only path is `.pbrun-closure.3368101089081684.json`. Only this audit document is added after testing.

Native terminal exit is 0 on `dl380g10`. Both terminal log hashes and the result payload were checked. Receipt `18f41917078a61016d76187c38e5ce51f01f7a4ce1254831cd8345560ec9b96e` binds result `d547f139e372ae74dc82916534a540f6045ddd2106cdc6f5a3c783f7aa03bff7`, 821 bytes. `verify_recorders.py` reruns terminal, log, snapshot and claim/payload checks. The claim checker does not verify worker attestation against the full action manifest.

The first claim read reported an absent receipt. A subsequent read-only
action observation found the native receipt, and the repeated evidence check
passed without rerunning tests. The initial observation is retained; this
pass did not establish its cause or a persistent receipt defect.

## Separate prose corrections

A separate commit corrects four findings encountered in the configuration
census. No behavior or declared interpreter floor changes.

- `pyproject.toml` no longer claims every venv uses Python 3.12; hosted CI and
  release cover the floor while PB uses a newer interpreter.
- `examples/launchers/README.md` removes three nonexistent launcher bullets
  and labels its remaining list as selected examples.
- `docs/RELEASING.md` distinguishes main/manual hosted full-suite checks from
  the required composed PrismaBuild PR gate, matching the workflows.
- `README.md` states that native export prints open shipcard slots; the
  serving lane must complete serving/quality gates before shipping.

Source reading verifies these prose corrections. The historical results,
archive and deprecated GGUF bytes remain preserved. The pipeline, format menu,
plugin contract and ship gates do not change, so architecture provenance does
not need a new contract entry for this slice.

## Program limits

No production code or production hot path, numerical recipe, cache/prefetch,
source identity API, artifact bytes, plugin pin, default or serving gate is
migrated. KL/bpp/runtime comparisons and GPU/serving measurements are
inapplicable to this test-plumbing change; no speed, cost, residency or served
quality claim follows.

PR #3065 retains its separate approved head and PR #3126 its separate scanner
review. The new allocation, orchestration and configuration census documents
on PRI-226 record additional bounded candidates, preserved differences and
unread areas. #1301/#1303 retain their owner boundaries. Current-main merge
proof and the final independent semantic residual census remain required.
