# Remaining domain-numeric owners: bounded census, 2026-10-02

Refs #1303 and #1301. Source inspected at `693a38f3ae3467ac3234cd9a7d756e41b69c3242`;
branch `sol/pq-dedup-domain-20261002`. This is a per-site semantic census of
selected live families, not evidence that either parent is exhausted. The
older `ws-dedup/out/primitives.json` includes retired codebook paths and
keyword matches for column means/second moments that are not RMSNorm.

## Forward KL slice (#2099)

| Site | Authoritative expression | Local contract preserved |
| --- | --- | --- |
| `expert_empirical_cost._unit_kl` | `kl_fisher.forward_kl_per_token` | FP32 log-softmax, baseline transfer, calibration batch, token sum/count, packed/sampled restoration |
| `expert_empirical_cost._unpacked_unit_kl` | same | packed render/scatter, member restoration, token sum/count |
| `expert_empirical_cost.measure_expert_unit_costs_forked` window reduction | same | per-window FP32 log-softmax, total token mean, window means and cost splitting |
| `kl_measurement._replay_lane_kl_totals` | same | teacher regrouping/scope, row-count refusals, lane broadcast, position mean then row sum |
| `build_rtn_cache.kl_divergence` | existing shared owner | widens students only, caller mean |
| `emu_forward_kl._KLAccumulator` | existing shared owner | selected/confident positions and accumulator normalization |

The shared owner accepts log probabilities and performs only exp, subtraction,
multiplication and the vocabulary sum. There is no log-softmax, scope or final
normalization policy in it. No numeric disagreement was found among the four
replaced expressions. The routing tests expose absent delegation; they do not
claim a prior wrong measured value. CPU raw-byte compatibility is checked on
finite, nonfinite, strided, empty and broadcast operands, plus all four real
consumers. GPU and served-artifact measurements have not been taken here.

## Distinct numeric contracts retained

| Live site/owner | Evidence and disposition |
| --- | --- |
| `build_rtn_cache._fp8_round` vs `fp8_dynamic.fp8_dynamic_weight_qdq` | legacy input-dtype amax/scales, clamp floor `1e-8`, cast-back before multiplication vs FP32 compressed-tensors qparams/dequant; retained as distinct, not silently migrated |
| `fp8_dynamic.fp8_dynamic_activation_qdq_vllm` | per-token scale, on-device denominator and negative-zero-preserving direct cast; distinct from weight QDQ and already shared by format closures |
| `mx_formats` / `nvfp4_activation_contract` | shared exponent vs exact-scale and UE4M3/E2M1 rules remain separate format contracts; no format/profile change authorized |
| `kernels/nvfp4_fused._tl_quantize_e2m1_dequant` | Triton midpoint/tie reference is device code; any consolidation requires compiler/device compatibility and qualification, not host-helper substitution |
| `glm_mtp._fused_eh_norm.rms` | fused FP32 concatenation/norm with final cast after weight multiplication; retained separately from upstream/vendored norm modules |
| `vendored/transformers_*` RMSNorm/hyperconnection references | upstream model implementations and generated modular/modeling pairs; source provenance and model-specific precision/offset contracts must be audited before changing |

These are source-level distinctions, not measured equality or disagreement.
The remaining numeric-family inventory needs representative device/serving
qualification where shipped paths would change.

## Safetensors headers and qnames

`source_read_plan.read_safetensors_header` owns the bounded ordinary-JSON
header reader and existing footprint/artifact/pipeline/autoscale/tool callers.
`source_read_plan.safetensors_prefix_length` already owns exact eight-byte
little-endian decoding and bounds for the stream and residency readers
(#1803). Acquisition/stat/body/object/refusal policies remain local.

The #2166 continuation routes `calibration_data._decode_calibration_buffer`
and `layer_streaming._advise_consumed_safetensors_pages` to the existing
`safetensors_prefix_length` owner. Both require exactly eight little-endian
unsigned bytes and a positive length bounded by their existing cap and file
extent. Calibration keeps its two refusal texts, provenance parsing and public
whole-buffer decoder. Page release keeps its single refusal text, pread/stat
fence, page alignment, selected-span merging and descriptor cleanup.

Remaining inspected sites are `shipcard._verify_open_safetensors_fd` and
`export_structure._read_metadata` (their diagnostics include the decoded
length), `model_profiles.validate._safetensors_header` (legacy unbounded
diagnostic), and `tools/chain_roll_bench._safetensors_spans` (unbounded benchmark
reader). These intentional acquisition/refusal contracts are not silently
flattened; they remain outside this bounded two-consumer slice.

`qnames.py` is the shared grammar owner. Dotted-key work is separately owned
under the coordinator's qname slice; no change is included here. Worktree
existence alone is not an active claim.

## Digest/JSON remaining sites (#1301)

`digests.JsonProfile`, `bytes_sha256hex`, `tensor_digests` and named framing
owners already exist. The next inspected compatible candidate is the two
final raw SHA constructors in `expert_empirical_cost._expert_checkpoint_identity`
and `expert_empirical_cost.main`. Both are load-bearing calibration/provenance
identities. Their input-byte acquisition is intentionally different:
checkpoint detach/CPU/contiguous/uint8 view versus CLI CPU/NumPy C-order bytes.
Only the final constructor may be shared; replacing both with a tensor
normalizer would alter the CLI's accepted inputs/refusals.

Distinct JSON profiles retain Unicode/ASCII escaping, compact/spaced/pretty
separators, strict/lax nonfinite values, fallback serializer, round-trip
normalization, caller framing and refusal order. No profile collapse or identity
migration is authorized. Census publisher and source-capture sites are outside
this writer's scope. Both parent issues remain open.

## CPU evidence for #2099

PrismaBuild OLD action `fcb94170979e54306afa431204c6285c675d32cb3ae6b3c2349d3e2c016e5b38`
produced 13 raw-bit/refusal compatibility passes and 13 expected missing-route
failures, zero skips, with all 26 outcomes reconciled. GREEN used
`pbtest.py --checkout <owned-worktree> --python
/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python --shards 4
--workers-per-shard 2 --threads-per-shard 1 --cpus-per-shard 2 --mem-gb 6
--priority -10 --timeout-s 300 --wait-s 900` on the two new consumer files
and `test_expert_empirical_cost`, `test_kl_measurement_teacher_pairing`,
`test_kl_fisher`, `test_kl_per_sequence_tail`, `test_duplication_baseline`.
All 74 tests passed and reconciled; no failures/skips/missing collection.
The duplication baseline is unchanged. Four-module `compileall -q` passed
through PB action `bf68d57882a2da9328e918e54e2d14ba4ad1001c24bf65a17cd116c3af9a1d15`.

Environment: dl380g10, CPU, Python 3.14.4, torch 2.11.0+cpu, Transformers
5.16.1; pbtest verified installed PB `95a59051...` and Tessera `b40c93cb...`
Git provenance. Terminal records, logs and hashed CAS claim/payload checks
were consumed; the claim verifier does not verify worker attestation.
Test snapshots contain the same four changed source/test blobs and shared
owner/pin/baseline blobs as the delivered head. Cgroup CPU/memory and host
Netdata are retained in the PB endings; missing dl380g10 pqteld CSV is recorded.
No performance, GPU, served quality or full-suite qualification is inferred.

Receipts and commands: `/home/rob/tmp/astra-resume-20261002/pq_dedup/kl-green.json`
(4 action keys, 74 reconciled outcomes), `kl-red.json`, `kl-compile-action.json`,
`kl-source-binding.json` and the corresponding `.stdout`/`.stderr` files.
An initial unsupported `-q` pbtest option and an absent-local-executable compile
submission were refused before execution. A second compile submission detected
a concurrent documentation commit during snapshot and refused; only the frozen
retry above establishes compilation. No rejected submission is a test result.
