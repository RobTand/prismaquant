# Fixed-resource design verification — 2026-09-08

This is source inspection and CPU verification of the retained refusal for
PrismaQuant #420. No new producer schema, admitted runtime table, GPU/served
run, capacity measurement, latency result or quality result was produced.

The [design](../design/runtime_fixed_resource_admission.md) identifies the raw
observation and accounting prerequisites and explains why an arbitrary positive
JSON receipt cannot close them. It also records the existing route for six
independently allocated variants from one common probe: ordinary byte/quality
allocation followed by valid exports, fixed-teacher held-out quality, actual
served prefill and existing publication gates. That route does not require the
optional runtime-v2 prediction contract.

Source evidence is retained at
`/mnt/shared/prismaquant-measurements/runtime-fixed-resource-design-20260908/source-evidence.json`
(SHA-256 `5d951086dcf5c7f170549f6100b47d0ae995dd2e73d872580fc456699bbad97d`).
Its 19 files were read from immutable Git objects and checked byte-for-byte
against the inspected worktrees. PrismaQuant source is
`47aec941d382ba419666a653fa2436a9ae8d6d33`; Tessera producer source is
`98aa317a06e55731c36c53f509adc04260668cc1`.

The existing runtime-provenance, measured-table, architecture and staleness
suites passed **124 tests in 9.55 s**, with 56 upstream Torch deprecation
warnings and no skips reported. This includes refusal of incomplete, opaque
and forged fixed-resource claims. No tests were added to restate the prose;
there is no implementation change requiring a new regression.

PrismaBuild chose Sparklina with four physical CPU cores `[5, 6, 7, 8]` and
12 GiB aggregate memory. The existing portable helper used CPU mode in
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
without GPU exposure (`NVIDIA_VISIBLE_DEVICES=void`, PB's empty
`CUDA_VISIBLE_DEVICES`). OMP/MKL/OpenBLAS threads were one. The image's presence
is not evidence of a vLLM run; this action executed pytest CPU contracts only.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-runtime-fixed-resources --anywhere \
  --cpus 4 --demand mem_gb=12 --priority -10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --detach -- \
  bash experiments/pq322_cpu_checks.sh -n 4 --dist worksteal \
  tests/test_runtime_provenance.py tests/test_measured_runtime_prices.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py \
  -q -p no:cacheprovider
```

Action:
`8b3ab965cd63f7cdfcefbb8d3eaf75e3986efbda0dcdf5f5dd55b85b5f9993c1`.
Tested checkout snapshot:
`3f113aad7f031d6d4235dc676ae3b9df7579c3f3`.
The terminal record's actual exit was 0. Its stdout, checkout identity,
resources, affinity and CAS receipt are retained in `cpu-audit.json` beside the
source evidence (audit SHA-256
`2bad74de790ac48f511e67ea790fcbbffbaa20b4aeb6fedb849defe69abe5c4d`).
The CAS result payload's size and SHA-256 were independently verified:
`53705431db3d007474a813dd9f048283976bdbb55e4d9fe44dce781919f0f3ec`.
CAS receipt SHA-256:
`1aed0ad2dfe89388a0a50346847a5f179b86b1fb5e11e476fb4665a7702e6dcd`.

PrismaQuant #420 and Tessera #399 remain open for implementation and qualified
measurements. Their incomplete fixed-resource model is not a new mandatory
shipping stage for the direct six-variant comparison.
