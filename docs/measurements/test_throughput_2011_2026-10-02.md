# CPU side-byte test throughput — 2026-10-02 (#2011, refs #1929)

The four existing deterministic writer cases now each perform one genuine
Tessera encode, then check both container coverage and fused-wire tightness.
Previously the two parametrized consumers encoded each identical case again.
All four family/rung/shape tuples, the mixed-rate manifest assertions, caller
member-name accounting and packed-stack member-name accounting remain.

The two former four-case tests consolidate into one four-case test. The module
therefore has 6 pytest nodes after, compared with 10 before; this is an explicit
consolidation, not an unchanged node census. No case, unique assertion, writer
input geometry, production code, pin, wire or serving gate was removed.

## Paired measurement

Both arms ran sequentially inside one admitted PB measurement action on
dl380g10, one CPU, four GiB, affinity [0], native thread ceilings 1, no GPU.
Both used Python 3.14.4, torch 2.11.0+cpu, Transformers 5.16.1, PB SDK
95a59051d48cda82eea7927f31870c6c862d7174 and Tessera
b40c93cb73745097e57a1ba4cf5b9eee166c759a. The unchanged worker pin guard
verified installed Git/RECORD/import ownership before each arm.

| Measurement | Before | After |
|---|---:|---:|
| Subprocess wall, cProfile + package branch coverage | 251.648 s | 176.277 s |
| Pytest reported wall under those instruments | 248.08 s | 172.21 s |
| Observed outcomes | 10 passed, 0 failed, 0 skipped | 6 passed, 0 failed, 0 skipped |
| Genuine fixture encodes | 8 | 4 |
| cProfile cumulative fixture encode time | 195.081 s | 115.724 s |
| cProfile cumulative window Viterbi time | 184.115 s | 104.275 s |
| Footprint executed statements / branches | 131 / 32 | 131 / 32 |

The measured instrumented module wall decreases 30.0%, or 1.43x throughput.
This is one paired module run. It is not an uninstrumented timing result or a
full-suite speed claim. Producer identity initialization remains about
17–19 seconds per fresh process; reducing calls does not remove that cost.

Every original and candidate (family, rung, shape) produced the same container
SHA-256, byte count and plane byte count. Both complete package coverage
reports have identical executed/missing statement and branch populations;
the footprint subset above is reported explicitly. The before/after profiles
attribute the change to fewer genuine encodes, without replacing the encoder.

## Host telemetry and action resources

Netdata series cover each arm on dl380g10, sparky and sparklina, in two-second
average buckets. CPU busy averages exclude idle and iowait.

| Host | Before busy | After busy |
|---|---:|---:|
| dl380g10 (80 logical CPUs) | 5.31% | 9.77% |
| sparky | 3.62% | 7.09% |
| sparklina | 1.76% | 4.26% |

Other host activity increased during the after arm. The admitted action itself
used 427.676 CPU seconds over a 434.184-second contained window, with a
734,134,272-byte cgroup peak and 28,069,291 sampled write bytes. This includes
both arms and telemetry collection. The PB box window retains its explicit
missing-pqteld diagnostic and its independently available Netdata CPU data.
There is no GPU saturation, energy or work-per-joule claim for this CPU change.

The first attempt collected zero tests: coverage's dotted-module lookup
prematurely imported the package and Torch refused a second import. That
failure is retained. The corrected harness covers the package, propagates
pytest's actual exit code, and reconciles it with recorded outcomes.
The worker's DNS did not resolve the Spark aliases during the successful
action; retained original telemetry records say so. Supplementary reads used
the existing SSH addresses 192.168.1.180 and 192.168.1.110 and recovered the
same before/after time windows without rerunning work.

## Reproducibility and evidence

Source parent: a7b4dbaede7c8362d334c996072d564eaee0be80, based on
6707ee494a504195babd355e77d76343233d738d. Baseline bytes came from that exact
base ancestor inside the sealed Git bundle. The actual snapshot is
26848dddef13efe8a30afe741ab4d4005d760466 and differs from its actual parent only
by the synthetic closure stamp. Candidate test SHA-256:
f25c1056b63631471dc6169575e505c9e13dfbebf53ca317fde35ac542b0171e.

- Paired action: 0358ac6f9bdb700ed34764a80aae8b433639cb193391d9ffdf5884eb05e40056.
- Receipt: /mnt/shared/prismabuild-fleet/cas/actions/v3/03/0358ac6f9bdb700ed34764a80aae8b433639cb193391d9ffdf5884eb05e40056.json.
- Receipt digest: ee9c3d3b9c1a082b6613b7093e90e582e05596de1b3572d631da6d8a33ce2f49.
- Payload: 7ec6a8ba5b6ab17537c90322883cde292e5bdd4f15e64c12b658fbbae72d2a53, 34,299 bytes.
- Artifact root: /mnt/shared/prismabuild-fleet/artifacts/sol-throughput-2011/f41f9f7b37dadfb52d6b2648a736e1d9e84d38ccdb205d642c68c285cd925ecc/.
- Driver and proof root on dl380g10: /home/rob/tmp/claude-campaign-20260926/pi/native-throughput/evidence/.
- Driver: pair-2011-v2.py, submitted as target Python -c source through published pbrun with --measurement --priority -10 --cpus 1 --demand mem_gb=4 --timeout-s 1200 --wait-s 1800. Exact source/argv is retained in the sealed request.
- Retained artifacts include both raw .pstats, pytest logs, outcome/encode audits, package branch coverage, JUnit, Netdata series and summary.json. All 18 action-manifest artifact byte/hash records were reread and verified; supplemental Netdata hashes are recorded separately.
- Proof: pair-2011-verified.json. Published core CAS.lookup(request) checked the complete request-bound receipt/producer and payload. The actual bundle input bytes/digest, actual snapshot parent and candidate test bytes were independently read. This is not a hardware-authenticity claim.

A subsequent portable admitted action performed the syntax check (1 module,
0 errors) and the duplication ratchet (7 passed, 0 failed, 0 skipped):
f6a9fd402c109afb2afde7e0349b811e3555d17e6f58ccc364a92ba69e5b0ebf.
Receipt digest: 1268bc065ea3296baa49f90910bcfe2df1e94157fb4ba349c7c73db52cb38889.
It reserved one CPU / three GiB with --anywhere, priority -10 and threads 1.
The actual interpreter dependency determined eligible placement.

## Integration batching

Keep #1929 open. This bounded change can share an existing pbmergeq candidate
with compatible reviewed fixes. The live queue's batch cap is eight; one full
suite qualifies the composed candidate. A failing candidate retains existing
node-ID comparison with its compatible base, immediate failed-file rerun and
prefix attribution. Candidate/base runtime identity must be bound by the
PB #1433 implementation before integrating a dev-pin bump. Inconclusive
shards and runtime refusals retain their distinct outcomes.

The demonstrated whole-suite critical path remains the two long Viterbi-heavy
GLM CLI cases: recent b00022 used 977 files, 850.2 seconds testing and
860.2 seconds total. Its checkout/build phase was about two seconds.
Tessera #798 and a reviewed development-pin/environment integration are the
existing measured lever for those cases; this report does not substitute
module savings for that campaign acceptance.
