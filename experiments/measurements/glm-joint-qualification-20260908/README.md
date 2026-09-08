# Native bounded joint qualification, 2026-09-08

The opt-in qualification window preserves all 26 actual H-aware anchor
verification records for 13 selected tiny GLM units. Across 50 fresh qualifier
invocations, peak simultaneously live X/H/PWC backing storage fell from
7,077,888 bytes (6.75 MiB) to 1,441,792 bytes (1.375 MiB), a 79.6% reduction.
This result qualifies ownership and exact input/record parity for this fixture.
It does not establish full GLM fit, production throughput, serving, KL or bpp.

The source is the genuinely initialized original-layout two-layer GLM fixture,
with hidden size 256, dense intermediate size 512, and expert intermediate size
256. Full native census and `shared-inputs-bounded-v1` capture precede selected
pricing. The selected scope is the largest eligible dense group plus one complete
four-expert packed group, comprising 13 units. Both `TESSERA_E4M3_K1_R768` and
`TESSERA_E4M3_K1_R1024` are real measured H-aware anchors, with positive measured
cost in every unit. No source/capture/wire/render verifier is replaced. Selected
intake is explicit and makes no full-campaign completion claim.

Calibration is one frozen synthetic 257-token sequence, seed 1917, with 64
activation prefix rows: `arange(257).remainder(126).add(2)`. Fixture generation
uses the existing tiny tests' upstream reference KDA and causal-convolution
implementations, recorded by qualified function name in `fixture.json`.
Qualification performs no model forward; forward hooks fail immediately if one
is attempted. Every invocation reads each decoder layer exactly once. Visual
tower initialization is separately recorded once per runner construction.

The ABBA comparison uses the same original source, canonical capture, measured
journals, PWC files and wire bytes. Ninety input files, including every journal
part, were hashed before/after every arm and independently rechecked after the
run. All 50 calls have identical candidate rosters, verified records and source
read counts. The verified-record digest is
`beb8b575b5fddbe90e11e7d11f343679ef3885bbe07b0a90fc09ada54f46c090`.
Weak backing-storage references prove that all prior X/H/PWC owners expire
before each next unit, every PWC donor expires after release, and the returned
cache holds paths without live capture or PWC tensor backing storage.

| Arm | Calls including one profiled call | H peak | X peak | PWC peak | Median unprofiled qualifier time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy A1 | 16 | 3 MiB | 0.75 MiB | 3 MiB | 0.2895 s |
| Window B1 | 9 | 1 MiB | 0.125 MiB | 0.25 MiB | 0.9980 s |
| Window B2 | 9 | 1 MiB | 0.125 MiB | 0.25 MiB | 1.0134 s |
| Legacy A2 | 16 | 3 MiB | 0.75 MiB | 3 MiB | 0.2935 s |

The window deliberately admits one serialized PWC donor per quantum. This tiny
instrumented workload is slower with windows. Unprofiled calls still contain
ownership observations and process/cgroup sampling, so the table is not a
production throughput estimate. CPU+CUDA traces contain 1,099 actual CUDA kernel
events per profiled arm. Capture-prefetch observer ranges increase from two
per legacy call to 13 per window call. The dominant CPU range includes Python
qualification and observer work; this trace does not isolate their individual
costs. CUDA peak allocated/reserved bytes are unchanged at 83,589,632/98,566,144
for all four profiles. Total-process physical memory is dominated by fixture,
runtime, profiler and allocator state; no total-process memory reduction is
claimed from these X/H/PWC ownership measurements.

PrismaBuild selected Sparklina for the complete measurement, reserving four
physical CPU cores (affinity 5–8), 8 GiB total DRAM and a 4 GiB GPU subset. The
container content digest is
`eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd`, with
PyTorch 2.13.0+cu130, CUDA 13.0 and NVIDIA GB10, driver 595.84. Native math,
source read and source prefetch use one thread/worker; PWC file loading uses two
workers. Source residency requires prefetched delivery with two slots. The
window's physical guard remains active: 8 GiB cap, 2 GiB margin and 8 GiB host
free-memory floor. There are no qualifier gate or guard bypasses.

Both Sparks have 83 Netdata samples. During the measured arms, Sparklina CPU
busy averages 5.57–5.83% of the box, while reported power is 11 W against the
140 W envelope. Power has only two distinct, lagged observations per arm;
work-per-joule ranking is therefore not established. The diagnostic does not
demonstrate GPU saturation. Both-host raw telemetry and the PB host-level
resource profile are retained. The complete action's cgroup peak is
2,447,802,368 bytes, with no OOM and verified cleanup.

The first native action `170046b125d4` is retained as a failed instrumentation
attempt: it verified all 26 legacy cells but incorrectly counted visual tower
initialization as a decoder read. Commit `dc601fb935` separates initialization
from qualification counts and includes nested journal files in the input
inventory. It changes only the harness and helper tests. No runtime change was
needed for the native retry.

Validation and attributable artifacts:

- Native action `ed222243e82e664e044e43632632980216a068690275501a363e569b74dcba90`
  exited zero, with receipt, CAS payload and local claim verified. Its snapshot
  differs from source `dc601fb935d86789c680fa19eb13e2a0d68fd2d9` only by PB's
  closure declaration. Runtime base is `84b0419877`, with the authorized source
  page-release fix `e22a0286a8` merged into the harness branch.
- CPU action `5a13b11d2cb3f25689dea9b235ff5682a42beb3e7f1cc7850f0d4a39d87d4064`
  passed 10 helper tests, zero skips, using the scoped CPU Python 3.12
  environment on DL380G10. Tests cover weak alias lifetime, complete group
  selection and required parity fields; these CPU checks do not qualify native
  anchor bytes. The native execution imports and runs the touched modules.
- [Invocation](invocation.json), [native audit](native-audit.json),
  [CPU audit](cpu-audit.json), and [summary](summary.json) preserve commands,
  source identities, receipt checks and machine-readable results.
- Complete raw artifacts:
  `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-qualification-windows-01/`.
  Successful result, four traces, raw both-host Netdata and immutable fixture
  are under `native-02/`; the first failure is under `native-01/`.

The harness never changes the runtime default. The bounded policy remains
opt-in, pending representative production qualification and profiling.
