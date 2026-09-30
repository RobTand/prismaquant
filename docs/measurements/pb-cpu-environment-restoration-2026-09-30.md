# PB CPU test environment restoration

Issues: #1458 (dependencies), #1454 (full-suite coverage).

## Scope

The original b40c93cb test environments on dl380g10, sparky, and sparklina
now include the project runtime/test direct dependency versions from the
reference green CI run, apart from the deliberately preserved Torch and
PrismaBuild/Tessera identities. This repairs the named served-gold/numerics
collection, source-bound Transformers tests, and wider inventory gaps in
datasets, coverage, compressed-tensors, and xxhash. It does not make the
Python/Torch stacks identical to CI or establish a whole-suite pass. #1454
remains open for full-suite coverage.

The environments are `/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb` and its
`-tf516` sibling on each worker. The missing x86 sibling was created by the
existing provisioning recipe: copy the environment with `cp -a`, then repoint
local base-shadow `.pth` entries. No hardlinks or older runtime pin were used.

## Dependency and runtime identities

Reference: green CI run
[36287954876](https://github.com/RobTand/prismaquant/actions/runs/36287954876).
The test dependency/source closure was installed at these versions:

| Distribution | Version |
|---|---|
| requests | 2.34.2 |
| gguf | 0.19.0 |
| pillow | 12.3.0 |
| transformers | 5.16.1 |
| accelerate | 1.15.0 |
| datasets | 5.0.1 |
| compressed-tensors | 0.19.0 |
| xxhash | 4.0.1 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| pytest-timeout | 2.4.0 |
| pytest-xdist | 3.8.0 |
| tokenizers | 0.23.2 |
| huggingface-hub | 1.33.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |
| regex | 2026.9.10 |
| filelock | 3.32.3 |
| tqdm | 4.70.1 |

Torch stayed at `2.11.0+cpu` on x86 and `2.11.0+cu130` on the Sparks.
PrismaBuild stayed at `059953bc3793f539600d333cd3311773e592b0e6` and Tessera
at `b40c93cb73745097e57a1ba4cf5b9eee166c759a`. Before/after distribution
metadata, direct URLs, and packaged contract hashes were compared. The
Tessera contract SHA-256 remains
`0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa`.

The Transformers GLM module source was checked against the existing 5.16.1
contract, SHA-256
`2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b`.
A package version alone was not treated as source qualification.

## Execution and receipts

All repairs ran in admitted PB CPU actions with priority `-10`, explicit
execution deadlines, one CPU, bounded native threads, and no visible GPU.
Host placement was required because each action repaired a host-local
interpreter. Qualified absolute-venv users were checked before mutation;
active environments refused repair. No process was killed and PB privileges
were not relaxed. This check does not certify arbitrary private PYTHONPATH
aliases outside the supported interpreter protocol.

The missing timeout plugin was installed through the original interpreter's
pip, at the same version as the 83460680 source environment:

```text
/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python -m pip install --no-deps pytest-timeout==2.4.0
```

Action `c550ea92019150bdbaa269e0919e2ece0ef8432a760a67c128947bdae644ebaf`
then ran the architecture/staleness checks with `--timeout=300`: 19 passed,
zero failures, zero skips. This repairs the missing plugin that caused
pq-stageb action `dddcf24ad0c70af1bb718d4ab65bfd69bf24e20df442c13fccfb619c5510d264`
to exit 4 before collection; that old failure was not a behavioral RED test.

The initial named dependency closure used the same interpreter-local pip,
`--no-deps`, and `--no-build-isolation`, installing only mismatched
distributions. The admitted stdout contains the exact `INSTALL` argv and
`BEFORE`/`AFTER` inventories. Both interpreter roots were checked in each
initial successful action:

| Worker | Final action | Result |
|---|---|---|
| dl380g10 | `ab0eb5175221972fe9920288ffd8b8546c0d8e9398cb096c7db14cfe783c42fb` | exit 0; imports/source hash match |
| sparky | `ba8a59ca92b61640b6b557693ac0c828ad3725c73f56ede414c00b5fc1836434` | exit 0; imports/source hash match |
| sparklina | `666d4405ae175e29023a93e32f58c57f4b8d688fc4fe886b9297dc599c44a0ba` | exit 0; imports/source hash match |

Receipt SHA-256 values, respectively:

- `e67c07b99e05182149610354b79d08f8f9d2e0d6b7d23961b3888d0e1e54fbd8`
- `596660721023a9ccaebc9fea3663efc931d5c49cc98b64a4c6f5d37b24fe9c0c`
- `3e4afd0a4227d461127a43c3bef4903dfe71b64721073b22e4d51c4c6e22d97a`

Each result blob's digest and length were verified against its CAS receipt.
These are package/import qualifications, not pytest pass counts.

The original x86 interpreter then ran the formerly blocked files
`tests/test_measure_served_gold.py`, `tests/test_numerics_pairs_1394.py`, and
`tests/test_glm_derivative_reach.py`. Action
`b263ffb3b28ef62bb8f9d22766a9e4abf7063c0988f194d58c1f04114a1c403c`
completed exit 0: 36 collected, 32 passed, four skipped, no failures, with a
reconciled roster. All four skips are explicitly CUDA-only variants, not
missing Python dependencies. Receipt SHA-256
`5db0ab89af02d29f151785d334450c3d125e6b7f9ba76984235445bdf9099fd4`
and its 10,271-byte result blob were verified against CAS. These passes
qualify the named CPU cases, not the skipped GPU cases or the entire suite.

## Complete direct dependency follow-up

A wider inventory found missing x86 datasets, missing coverage tooling, and
older compressed-tensors/xxhash distributions. A second guarded repair used
CI-derived constraints for the support dependencies required by the missing
direct distributions. Installed Torch, PrismaBuild, and Tessera versions were
explicit constraints; their distribution metadata and runtime contract were
checked unchanged afterward. The resolver was allowed to install required
support packages, rather than use `--no-deps` to leave an unusable closure.

The actual command shape was:

```text
<qualified-root>/bin/python -m pip install --no-build-isolation -c <action-local-constraints.txt> <mismatched-distribution==CI-version> ...
```

Each action's result blob records the complete `CONSTRAINTS`, exact `INSTALL`
argv for both roots, and `BEFORE`/`AFTER` inventories. The constraints file was
created under `/home/rob/venvs` and removed after pip completed. No `/tmp`,
Docker, GPU, or active-venv mutation was used.

| Worker | Final complete-closure action | CAS receipt SHA-256 |
|---|---|---|
| dl380g10 | `6e8ce5561342a69e35082e5d01a9caa15d20ba4564b1d88aa9a631166036912e` | `47aff5810458c7a573e835e80c147afad275b827070710ca5999198a14997d78` |
| sparky | `2c0cebf85e4a54773e7c3b071ab90f4a38b9a95cdece01ba6fabbb771019b796` | `5b57c3d0e3960020e87366bd2a060ac922bd7dd1126eeb2ae1ed241d13b6f350` |
| sparklina | `5e2de54f4299e29414deb95dbb29863561ebe3a979683f17cea08a3777ce15d1` | `248222349248b2d6f5490c16a6db4e7e84a5734b1c020e0c2166af5dc32616cc` |

All three actions completed exit 0 with unambiguous terminal records. Each
imported the direct dependencies in both roots and rechecked the GLM source
hash. Each printed two `CI-DEPENDENCIES-IMPORT-OK` markers and its final
`CI-ENVIRONMENT-REPAIR-OK`. The 49,732-, 54,181-, and 54,055-byte result blobs
were verified against their CAS digests and lengths. These are six qualified
package/import closures, not six pytest suites.

## Final CPU qualification

After the complete closure repair, action
`ab11ef836284ef0075ca61b583086e6a271fa3ad5255b0c3db74e76c7ada6c9c`
ran the three named prerequisite files, the streaming text-only wrapper
configuration tests (including the formerly missing accelerate skeleton
cases), and the architecture/staleness checks. It completed exit 0 on
x86: 69 collected, 65 passed, four skipped, zero failures. All 69 outcomes
reconciled. The four skips are explicitly CUDA-only numerics variants;
no dependency skip was certified. Receipt self-digest
`799a1a19f585de6908c09167b42f4a93059877804e77a8394cc9f942feaa02a7`
and its 17,458-byte result blob were verified. This is targeted CPU
qualification, not a whole-suite pass.

## Full-suite limitation

An x86 validation-only companion with the same runtime pins completed one
full-suite shard: action
`54080f54566bf622dd8d5ae53d19bbc4f0e6f01d712763243dc693b2f077556d`,
8,135 passed, 183 skipped, one xfail, and 10 passed subtests. Its outcome
roster reconciled; skipped cases are not certified.

The other shard, action
`cd803edb5f2b3eb00b7550b73c5b241ae9e85a2725d3221e7371dfb58d7d28f5`,
reached approximately 82% before its 3600-second action deadline. It printed
failure indicators, including a 300-second call-bound failure in the MTP
source-scope campaign test. No final counts or complete roster exist for that
shard. The test environment is repaired; full-suite coverage is still open
under #1454. PB's two-shard subdivision of that incomplete bucket subsequently
completed with reconciled rosters: action
`095e649e1fbf7648cf92e279efcc820c470e98d938209ac9289be2e4508ac18c`
reported 3,737 passed, seven failed, 123 skipped and 65 passed subtests;
action `67e05877a0183c35a1a57cba8023fb5c23f36b8a48500fed4791b1a94ee1e1d4`
reported 3,724 passed, 14 failed, 44 skipped and 13 passed subtests. These
failed actions have no success CAS receipt. The 21 failures include 16 mock
fanout fixtures missing the new explicit pilot prerequisite, two call-bound
failures, two private mover timeouts, and a duplicate-helper gate subsequently
fixed by PR #1788. This is diagnostic RED evidence, not a same-snapshot
whole-suite result combined with the earlier shard.

No production default, format, export bytes, kernel, serving numerical
behavior, runtime pin, GPU performance, or GPU qualification changed.
