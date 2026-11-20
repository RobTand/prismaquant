# #2575 vision CPU evidence

Flux CPU job `f3cTbFU6s27` on `aecdd5285` passed 689 tests and 59 subtests across 36 files. Exit status was 0.
The gate environment used Python 3.12.12 and torch 2.15.0+cpu.
The run covered `test_vision_*.py`, `test_visual_allocator_*.py`, and `test_allocator_*.py`.
It also covered `test_tessera_census_cache.py`, `test_tessera_export_lane.py`, `test_tessera_expert_projection.py`,
`test_glm_mtp_selection.py`, `test_prismaquant_visual_format.py`, and `test_prismaquant_visual_phase2.py`.

| #2575 criterion | Verdict | Evidence |
| --- | --- | --- |
| 1. Give each measured vision row a legal DP rank. Refuse an incomplete set. | MOOT | `7786cfbf2` removed the multimodal probe. The allocator now applies one `VISUAL_FORMAT` outside the DP. Real multimodal capture is a product decision under #2812. |
| 2. Keep invented text-only Fisher, costs, rates, and uniform swaps out of the DP. | COVERED | The CPU controls exercise missing visual cost rows, source-only visual rows, and body-only accounting. |
| 3. Record each canonical source name and profile root with a pinned serving cell or explicit refusal. | PARTLY | The roster and export audit cover names and roots. Per-entry audit coverage still needs a serving cell or refusal for each entry. |
| 4. Match each emitted vision name to a pinned cell or refuse export. | PARTLY | Tests cover BF16 shard names and explicit refusal. The pinned-cell check for emitted low-bit vision names remains open. |
| 5. Refuse incomplete or invalid rows and wire proof. | COVERED | The cache checks the exact unit set and original wire receipts. The #2809 refusal cases cover invalid inputs. |
| 6. Keep the body and MTP plans equivalent with vision off and on. | COVERED (body) | `tests/test_vision_alloc_failclosed_1921.py::test_body_plan_is_the_same_with_and_without_vision` runs `allocator.main` with the visual rows (VISUAL_FORMAT BF16 and NVFP4) and without them, and checks that every body Linear gets the same format. The MTP selection inputs do not depend on vision by construction. The export projection of body names is covered by `test_checkpoint_body_names_survive_export_projection`. |

| #2809 item | Verdict | Evidence |
| --- | --- | --- |
| Absent units | COVERED | The cache refuses a missing vision, merger, or body assignment. |
| Wrong shapes | COVERED | Intake refuses shape differences for vision and merger units, including differences in checkpoint headers. |
| Unsupported names and roles | COVERED | Cache and projection refuse out-of-roster wires, unowned names, role tags, and missing BF16 source names. |
| Missing or mismatched original wire receipts | COVERED | Tests cover missing receipts and files, unmeasured rows, wrong sizes, and changed bytes, including same-size substitutions. |
| Source and recipe identity mismatches | COVERED | Intake refuses mismatched source and encoder hashes, recipe ranks, and receipt names. |
| BF16 passthrough and the independent GLM source-precision pin | COVERED | Tests read the BF16 shard names, types, and values. GLM preflight preserves source precision and refuses low-bit vision. |

Flux CPU job `f3ceCsa3gCf` on `36bb06619`: the five vision test files, 92 passed, exit 0; `tools/ci/static_checks.sh --merge-main` exit 0 (graph impact: new edges 0). The new test alone (job `f3cfRZPub8b`): 4 passed (two synthetic profiles, BF16 and NVFP4).
