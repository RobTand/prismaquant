# Selected capture loading diagnosis and opt-in wiring — 2026-09-08

Selected anchor reuse read its 49.53 GB canonical X/H selection twice before
encoding. The existing verified activation loader was unreachable from that
CLI path. Commit `ae781f4ea5` exposes it through the existing explicit
`--capture-load-policy`, with complete resource accounting and separate bound
execution evidence. Commit `5f12978a8b` adds direct policy/source resume-boundary
tests. The option remains unset by default. Native after-change measurement is
pending the coordinator's GPU window; this report claims no speedup.

## Attributable before measurement

Original expert action
`ab1c32b1a4c8fabc1bc27161cd5e8690b106503cd868e3640e1339a487d11e38`
ran the frozen `8d8a293ffbcadf6a40ec91fde03d8f690eafd2e5` checkout on Sparky,
original producer image content
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`,
Torch 2.13.0+cu130, CUDA 13.0. Selected row 0076 is layer-4 routed experts,
864 entries: 576 files of 75,499,429 bytes and 288 of 20,973,477 bytes.
Total selected file size is 49,528,032,480 bytes. The original complete capture
manifest remains SHA256
`f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c`.
The unchanged calibration is 512 samples × 512 tokens, seed 0, full-census H,
with a maximum 512-row FP32 scoring prefix.

`before-read-audit.json` binds fixed snapshots of the actual main-thread
sampler and both-host Netdata. There are 607 prefetch samples from Unix
1788909253.506646 through 1788909859.848720 (606.342 seconds). The enclosing
samples span 1788909252.506191–1788909861.158911 and account for
99,056,222,208 physical read bytes. Twice the selected files is
99,056,064,960 bytes: the difference is 157,248 bytes. The narrower sampled
window reads 98,716,471,296 physical bytes and 101,440,741,694 `rchar` bytes.
Its physical rate is 162.81 MB/s and selected payload/time is 81.68 MB/s
(decimal MB). These are sampled process/window figures, not per-file syscall
traces; `rchar` includes metadata/other reads and cannot be labeled payload.

In those 607 samples, 499 end at `tessera_calibration_cache.py:40` (file read),
25 at line 49 (digest update), 38 at Torch `load_tensor`, and 28 at finite
validation. Only three stacks include `CaptureMemoryGuard.check`; the evidence
does not identify guard overhead as the dominant cost. Whole-process counters
also include selected source-weight preparation: its separate 51-sample
window has 35,947,208,704 physical read bytes. Attributing the process's entire
~160 GB counter to the 49.53 GB capture selection would be wrong.

Both-host Netdata contributes 121 samples during prefetch. Sparky's power is
12.98 W mean (12–13 W), CPU idle 92.42%, and iowait 5.07%; Sparklina's power is
19.55 W mean (4–43 W), CPU idle 84.37%, and iowait 9.16%. This supports a
waiting GPU during selected prefetch. GPU utilization is not used as saturation
evidence. The metrics retain their host/window scope and are not attributed
exclusively to this action.

## Cause and bounded change

At frozen source `8d8a293f`, campaign lines 3962–3965 reject
`--capture-load-policy` outside new bounded capture generation; selected reuse
at lines 4292–4298 does not forward it. This was missing call-path wiring,
not an invocation flag that the coordinator forgot.

The legacy path at `tessera_calibration_cache.py:906–913` hashes each selected
artifact while advising consumed pages away, then `torch.load` reads the file
again. Its page release at lines 934–935 is another boundary, not another
payload verification read. The existing `_verified_capture_entry` at line 485
uses `load_verified_activation_cache_entry` in `perturbed_x_cache.py:579`:
one held regular nonsymlink object, original expected payload SHA256, actual
before/after file signature checks, explicit F/S/M bounds, one hashed private
buffer, bounded archive and tensor validation, and buffer expiry before CUDA
transfer. No stat signature substitutes for the checksum.

The change allows that policy only for existing bounded capture generation or
selected streaming reuse with units and a SHA-bound complete capture. The
existing complete-manifest, census, selected-source descriptor and runtime
authentication still run. No new capture, source projection probe, cache,
scheduler, format, default, or numerical method is introduced.

The dispatcher and runtime use the same new `capture_prefetch` resource phase:
selected resident weights/X/H, one decoded storage S, private serialized F,
full-file source-page F, scratch M and existing headroom. Existing source,
export and encoding phase terms are retained. The physical guard still checks
actual read/decode/transfer boundaries. A digest-named execution sidecar records
actual loader byte counts and load digest chain, original capture binding,
resource plan and guard; selected-source provenance carries its path/SHA.
Immutable receipt names prevent an interrupted or refused later resume from
replacing evidence referenced by an older cost output.

The original capture identity and file bytes do not change. The CLI policy
remains in checkpoint settings, and changed package source SHA remains a resume
boundary. Existing explicit seed intake has its ordinary per-row gates; this
change does not select that path or authorize inheriting old priced anchors.
The coordinator must review the source transition before a new pricing freeze.

## Post-prefetch observation and disposition

The initial anchor profiler rejected its 9,222,546,186-byte CUDA trace against
the 536,870,912-byte cap. It recorded `observation_failed` and no accepted CUDA
trace. The observer preserved the completed anchor return so the campaign could
journal it. Profiler failure is not a native qualification pass.

A fixed post-observer slice contains 299 samples over 298.383 seconds: 266 end
at Tessera `window_viterbi.py:771`, `sse += float(final.sum())`, which waits for
preceding GPU work; 11 end at graph replay and two at `tensor_identity`.
Sparky's 59 Netdata samples show 58.08 W mean / 61 W maximum, CPU idle 92.11%
and iowait 0.44%; Sparklina has 4.12 W mean and CPU idle 97.63%. This identifies
the observed host wait, not the responsible GPU instruction/kernel bottleneck
or saturation. The coordinator owns the replacement bounded CUDA profiler.

The coordinator withdrew this exact action after the observation failure.
At the final read-only audit, its owned PID 2467709 was gone; 88 unit envelopes
contained 88 anchors, with each serialized payload digest checked. There is no
completed `cost.pkl`. Journals, existing wires, profiler evidence and canonical
capture remain at their original paths. Wires were not independently requalified
by this disposition audit. The records are retained partial evidence and may
only pass the normal explicit resume/seed/source-transition gates; this agent
did not stop or restart the action. `withdrawn-expert-disposition.json` binds
the retained journal and profile files.

## Validation and remaining native gate

Regression-first PB `fd3ae6aab1bc06c59b19b1699dac43e56618c5585e7658d9598998371a9ff08d`
ran the new tests against unchanged implementation: three expected failures
(CLI refusal, missing admission policy, missing dispatcher forwarding), four
refusal cases passed. The positive implementation has 235 completed unique
CPU tests and one existing CUDA-only encoder-memo test skipped. This includes
15 new selected-loader cases, two existing pre-acceptance resume refusals,
verified loader/capture/source authentication, admission, fanout and architecture
tests. Compile checks passed for all touched Python modules. CPU workers ran
through PB on dl380g10, one native thread each, with independent files fanned
out (up to nine actions). No new GPU work ran for this change.

The broader `test_tessera_campaign_resume.py` was not completed: its 4 GiB action
`7814eecd02a9fb7aef948103fe8aeb1537b9ec066b5a5073cb02657bc091e9e0` OOMed after two
progress dots. A 12 GiB attempt
`f2e22ec4352190831f0f27a618a4669601bb771092e68fbb43da84d5fad7de75` reached 13 dots,
then the 300-second deadline (4,685,869,056-byte peak, no OOM). Neither has a
completed pytest summary or counts toward the passing total. This file performs
real CPU encodes; the focused journal policy/source tests cover the changed
reuse boundary without claiming those unfinished wire/encoding cases passed.

`cpu-action-audits.json` verifies actual terminal cleanup, logs, canonical CAS
receipt and producer attestation digests, CAS payload bytes, and actual source
bundles. Successful snapshots differ from the tested code/test commits only by
PB's generated closure. Expected failures, OOM and timeout are retained with
actual status and missing CAS where applicable. All successful actions exited
zero with complete cleanup and no OOM/live processes; the compile CAS payload
is correctly empty.

The coordinator's next same-row expert qualification can supply the after
measurement with the explicit policy, original capture and corrected observer.
It must retain loader execution receipt, main-thread/process-I/O profile,
both-host Netdata, exact resident/selected bytes, and native tensor parity.
Compare the prefetch phase, not whole campaign duration, and retain original
calibration/runtime/resource conditions. Until that completes, reduced physical
I/O, speed and work-per-joule for this selected route remain unmeasured.
