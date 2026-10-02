# Internal original CPU bootstrap qualification — 2026-10-02

PR #2076, child #2074; parents #2010/#2008/#1896/#1887 remain open.
The tested executable/test head is
`48e9f5a7f89899d23335e51c590b664eae715b4d`. Later delivery changes are
documentation only and are enumerated in the source audit.

This is a tiny synthetic native-safetensors CPU context qualification, not an
original GLM run, GPU capture, full-provider admission, corrected global
cotangent claim, served result or performance measurement. Automatic capture
remains closed. Original context/head/layer/direct FP8 GPU loads refuse before
material or CUDA work. Exact a6 adoption, higher-level runner/profile/calibration
propagation, tokenizer consumers and whole-source/chain qualification remain
open. Header estimation still uses whole-shard windows and can reacquire released
header-only material; single-pass and GPU-bound production operation are not
established.

The independent fixture constructs a real two-layer Llama from a supported
Transformers config, saves its native FP32 tensors, declares exact native Git/LFS
publisher objects plus producer/readset bindings, and acquires actual PB leases.
The logical pool config/index deliberately conflict with the admitted originals.
Every installed tensor and a complete CPU forward's logits equal the same-byte
legacy baseline exactly. Owned derived text config also matches existing legacy
staging. Additional controls cover config mutation after sealing, dynamic config
refusal, caller-vs-construction modality, cancellation before material, finite
windows with previous-shard consumers, detached aliases and a second-payload
failure whose traceback retains its native view until explicitly released.
Qwen MoE namespace lookup consumes owned complete index evidence, refuses
ambiguity and resets its prior namespace memo when explicit index intake changes.

## Executed evidence

All tests and compile checks ran through PrismaBuild on dl380g10, CPU-only:
Python 3.14.4, Torch 2.11.0+cpu, Transformers 5.16.1, PB commit
`95a59051d48cda82eea7927f31870c6c862d7174`, Tessera commit
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`.
The PB test wrapper checked the installed dependency pins before collection.

The final suite collected and ran **129 tests, all passed, no skips** across
14 independently admitted files, including the new bootstrap 15, owned-profile
6, selected-source 33, strict JSON 4 and automatic-capture refusal 14 cases.
Legacy metadata/wrapper/staging, duplication, IO site, architecture and staleness
controls are included. Ten touched/prerequisite Python files compiled, exit 0.

| Result | Action key |
| --- | --- |
| Predecessor original bootstrap/loading: two failures, 11 deselected | `954d309ead18e2c1ce4e11e70388f517717db7c51d868d4a061f192a76bfe778` |
| Predecessor Qwen pool-index reopen: two failures, three deselected | `c86708ca79a14865c14680d938f914dcdeef07c30047a3bf7d206d8cdfeddcba` |
| Final CPU bootstrap: 15 passed | `4edf5f89fe9f040c393cfa39211dad2252e58a23c8a789faefc1151683840192` |
| Final owned profile: six passed | `2e3484f8739342c342d64bdbf13e5bd0075e5c80ce8e2a5474e9314047a9bef4` |
| Ten-file compile: exit 0 | `56eb7fc224e9ff49a7e7c1d515903679f70b785d9102ea709f0309309139e9cf` |

An earlier source-family candidate ran 143 nodes: 141 passed and two CUDA
single-pass-source cases skipped because CUDA was unavailable. This does not
establish their GPU behavior. Two later files were not published because source
edits overlapped lazy snapshotting; the frozen final suite covers both. The first
bootstrap candidate retained test loop aliases and correctly failed owner close;
the fixture now releases those aliases explicitly. A construction spy initially
treated skeleton inspection as a visual payload read; it was corrected to allow
one meta inspection and refuse a later materialization call. These negative
results remain in the audit packet.

Root temporarily held submissions for fleet maintenance. After release, the
fresh dl380 offer denied queued actions with `measurement_census_unavailable`
and `scratch filesystem type not supported: tmpfs`; the complete status snapshot
recorded 28 ready and zero claimed, with available host capacity. Root rolled the
admission runtime back through the normal publisher. The existing sealed requests
completed without resubmission/withdrawal or a bypass. Their sealed worker script
and actual receipts name `829a1afb8acd-1790951491-60e7c9cfbead`; these application
test results are not qualification of PB #1419 or its deployment.

## Commands and artifact paths

Coordinator evidence root:
`/home/rob/tmp/astra-resume-20261002/pq_capture/`.
`original-bootstrap-final-green.json` carries all final file outcomes and
collection reconciliation; `original-bootstrap-compile.log` carries compile
exit/CAS publication. `original-bootstrap-red.json`, `original-profile-red.json`,
candidate reports, `bootstrap-admission-start.json` and the prepublication
`original-bootstrap-compile-refused-anywhere-tag.log` retain the negative evidence.

`audit_bootstrap.py` reads canonical CAS receipts and queue terminal records,
checks receipt/request/result/source-bundle digests and sizes, producer input
binding, exit status, scope cleanup and OOM status, then imports immutable source
bundles into a bare audit repo and compares actual blobs to delivery HEAD.
`BOOTSTRAP-REVIEW.json` records every candidate, failed, unpublished and final
outcome plus compile/source differences. The packet makes no speed or energy claim.

Final test command uses published `tools/pbtest.py` with the checkout above,
`--python /home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`,
`--threads-per-shard 1 --max-clients 14 --mem-gb 4 --priority -10
--timeout-s 300 --wait-s 900`. Exact file paths are in
`bootstrap-final-checks.json` and the result report; PB owns placement/fanout.
Compile uses published `tools/pbrun.py --anywhere --cpus 1 --demand mem_gb=2
--priority -10 --timeout-s 120 --wait-s 600`, native OMP/MKL/OpenBLAS threads one,
the same pinned interpreter and `-m py_compile` on the ten paths in that plan.
No `--tag` accompanies `--anywhere`; the interpreter-path requirement selects
an eligible worker. The original contradictory flag attempt refused before
publication and was corrected without changing a sealed request.

## Follow-up direct FP8 metadata ownership — PQ #2086

The internal context already supplies its owned config. A direct FP8-map caller
that supplied or inferred a profile but omitted config could nevertheless choose
the logical pool's block/MXFP4 declarations. The local map config is now acquired
through the existing owned JSON/window seam when absent; explicit already-owned
config and legacy routes retain their behavior. No GPU or format gate changes.

The real PB-bound CPU fixture declares original block `[2,4]` while its mutable
logical pool declares `[128,128]`. Both direct-profile variants returned the wrong
pool block before the fix: action
`8c8532f0a7cf87130b961001a7b873e0c8914145e3e0ca52f95fd83fb224319d`,
two failures. Fixed source/test head
`cd7a320c877dfe9ac60974c3996c734888e94eac` passes **88 collected/run tests,
all passed, no skips** across eight PB files. The focused regression action is
`6877264354d71ac65f3a1fe095b6f02f8e082122f3a3f4f5b073a13b4f61a1a2`;
the adjacent FP8, strict staged reads, tiny original context, selected-source,
architecture/staleness and IO controls also pass. Two files compile with exit 0,
action `6f81dac07368917af3bd6b40614bff057d2d1ee3254772893bb221fac440ad64`.

The pinned CPU environment and native thread limits above are unchanged. Test
command uses `pbtest.py --checkout .../wt-fp8-original --max-clients 8` with the
same interpreter/memory/priority/time bounds; compile uses `pbrun.py --anywhere`
with the same CPU/memory/native thread bounds on `prismaquant/layer_streaming.py`
and `tests/test_original_fp8_map_config_2010.py`. The coordinator evidence root
above contains `original-fp8-map-{red,green}.json`,
`original-fp8-map-compile.log` and `FP8-MAP-REVIEW.json`. The shared read-only
auditor's `--fp8-map` mode binds canonical receipts, terminal cleanup and actual
source/test blobs to final delivery HEAD; this appended evidence is its sole
post-validation source delta. No new original payload or GPU run was performed.
