# Commands that ran

Candidate: Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, contract v60.
Image X: `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Target: sm_121 on NVIDIA GB10. Priority 0 for every measurement action.
All commands ran through PrismaBuild from the PrismaQuant worktree at harness commit
`dffc6a6b72949a5d149406c66bcb16ccfd74e4e7`. The sealed snapshot was clean:
the `.pbrun-closure` file in each `run-manifest.json` shows `dirty_sha256` equal to the hash of empty input.

`IMG` below stands for the image reference above. `<checkout>` is the worktree path.

## What the harness does

`harness/run_suite.py` runs on the Spark. It fetches the candidate commit into a fresh git checkout and refuses any other HEAD.
It starts image X with that checkout mounted read-only and the output directory mounted read-write.
Inside, `harness/container_entry.py`:

1. installs the pinned pytest tools into the output directory;
2. binds the source: Tessera's `measured_source` must say `verified` at the candidate, and the packaged contract must hash to the pinned digest;
3. records Torch, the CUDA device, vLLM, Triton and Tessera;
4. self-tests the probe plugin on four synthetic tests under xdist;
5. collects the suite's nodes and compares them with `roster.json`, exactly;
6. in `run` mode: runs pytest with `-n 2 --dist loadfile --strict-cuda`, a surface file, a junit file,
   assertion-pass recording (`enable_assertion_pass_hook`, `verbosity_assertions=2`), INFO logs and `-rP`.

The probe plugin `harness/native_probe.py` only observes. Each xdist worker builds extensions into its own directory.

## D38 preflight on CPU (no GPU attached)

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd <checkout> --cpus 2 --demand mem_gb=8 --tag gb10 \
  --container-image "$IMG" --priority 0 --timeout-s 1800 -- \
  python3 docs/measurements/pq1317-gpu-tests/harness/run_suite.py --suite moe --mode collect
```

The same command with `--suite dense` covers the dense suite.

| Suite | Action | Host | Result |
|---|---|---|---|
| MoE | `fc2fb0448f13f5f3b5472c8a54e825410c8c5ca502d5dcb8e4ddc1c414410676` | sparklina | done, 71 nodes collected, roster equal, source `verified`, probe self-test ok |
| Dense | `e51a6b63c2676f5a68566ced603a4b458af5f6b3550245fd1fd7a648bd32a207` | sparklina | done, 65 nodes collected, roster equal, source `verified`, probe self-test ok |

The preflight proves collection, imports, the candidate and identity checks, output paths and the probe plugin.
It gives no GPU qualification.

## GPU runs

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd <checkout> --cpus 2 --demand mem_gb=12 --gpu --tag gb10 \
  --container-image "$IMG" --priority 0 --timeout-s 5400 -- \
  python3 docs/measurements/pq1317-gpu-tests/harness/run_suite.py --suite moe --mode run

python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd <checkout> --cpus 2 --demand mem_gb=24 --gpu --tag gb10 \
  --container-image "$IMG" --priority 0 --timeout-s 5400 -- \
  python3 docs/measurements/pq1317-gpu-tests/harness/run_suite.py --suite dense --mode run
```

| Suite | Action | Host | Elapsed | Peak memory | Result |
|---|---|---|---|---|---|
| MoE (tessera#610) | `7a3dac4d75437bb570b7c27379112cf65bff1fcc8bc6728d72a6b6c07e19db4d` | sparky | 48 s | 4.1 GB | 71 passed |
| Dense (tessera#611) | `ec7d9d959d863db83a37063203f0d34d32dc4a588230c7f4856ad25921c8cbd6` | sparky | 130 s | 10.0 GB | 65 passed |

Sizing (D30). Attempt 2's runs of the same suites peaked at 4.5 GB and 11.16 GB. The 12 GB cap was close for the dense run.
This attempt asked for 12 GB and 24 GB. Placement was left to PrismaBuild (`--tag gb10`). No host was pinned.
The timeout is a backstop of 5400 s for runs of about one to two minutes.

`results/pb-receipts.json` has the attempt record, memory peak, GPU power peak and CAS receipt digests of each action.
`results/logs/run2-<suite>/` holds `pytest.log`, `junit.xml`, `surface.json`, `run-manifest.json`, `container-report.json`,
the probe records, `native-so.sha256`, `triton-kernels.json` and `collect.txt`.

## Run 1: the earlier runs that corroborate

Attempt 2 ran the same five files from the shared pin directory with one inline script per action.
The scripts are in `results/logs/run1-*/command.sh`.

| Role | Action | Host | Result |
|---|---|---|---|
| Preflight (collect) | `57575b6ee1afe4f3c41496acdfaf46501186e2c25126da36760e901734467a22` | sparky | done, 136 collected |
| MoE | `fe1ae1f2402b865d7dc84a6f86e56745f4767db4b8aee0bddeec73d964d03462` | sparky | 71 passed |
| Dense | `370fead21f36ad81db81b7f6b536e85c09ccc97baae95f297105510dd2931fac` | sparklina | 65 passed |

Tessera's instrument could not bind the source of those runs: the directory has no git metadata, so it reported `unknown`.
The directory digests to the same value as a tarball of the commit (`candidate.json`, `source_tree`).

## Rules

* Run every GPU action in image X. Run the D38 preflight first for any new or changed entry point.
* Keep every node outcome, device fact and receipt. Mark any skip, failure or unexecuted required node as incomplete qualification.
* CPU skips and old receipts give no qualification.
* After any change to the candidate, the image or the harness, requalify (see `README.md`).
