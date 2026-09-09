# Reviewed Tessera producer pin, 2026-09-09

The development and serving source pins now name
`b1eb1dccc9df6773ab94e1c98f316f45bb18ab4c`. This includes upstream
Tessera #434/#436/#437/#438: cached packed export, rank-local TP2 intake,
opt-in best-form trellis and explicit opt-in tile controls. The packaged
contract remains SHA256
`a688f8de244f936ec3a63a782e20af7985733e7a6fb0b4b981b5fe4c44112212`.
No producer default, format menu, admission answer or serving qualification
changes. The source package identity in the GPU experiments is
`bcad2ef2a7fdec2aab51b30d59f1f5e10b4933637ca4ffc74ee05e20816c1822`.

Actual GLM row-0076 runs retain exact wire bytes and pricing scores for the
first 32 experts at R832 and R1088, with the original capture reused and
864 selected entries resident. No source forwards occurred. B32 vs B8 improved
throughput/work per joule by 1.24504x/1.24089x at R832, and
1.12470x/0.880061x at R1088. A separate first-16-expert R1088 comparison
favoured B16 vs B8 by 1.15141x/1.02186x. Comparisons are within each run's
host and environment; they do not establish a blanket batch policy or
full-model quality. Sustained 90–100 W remains unmet. Full performance
records, raw profiles, both-host telemetry, actual-file parity and CAS/source
audits are retained in [the experiment branch at 82a312bb1b](https://github.com/RobTand/prismaquant/blob/82a312bb1b/docs/experiments/glm_best_form_screen_20260909/README.md).
The shared evidence root is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/`.

Pin integration tests: **249 CPU tests passed; zero skipped or uncollected**,
ten modules, eight independent PB shards with two workers each and one native
thread per worker. Torch 2.10.0+cpu, Python 3.12, dl380g10; no device coverage
is claimed from these tests. The interpreter is
`/home/rob/venvs/pq-cpu312-glm-best-form-b1eb1dccc/bin/python`, separately
provisioned from the frozen reviewed producer without changing the fleet base
venv. Provisioning action
`8a21bbc82233d7228ae45cd1d891eb8c4f386d748b113f22d402f7c51ffa10ec`
built outside the immutable source and passed a second check-only verification.
The retained provisioner script fails if the scoped environment already exists.

`cpu-receipts.json` gives all eight action keys, payload/receipt hashes,
executed snapshots, module assignments and summaries. Root checked actual
terminal exits, cleanup, receipt/payload hashes and snapshot sources against
`2a30407878`. Differences were the PB closure and then-untracked provisioning
script; tested production source matches. The 28 warnings per shard are
Torch JIT deprecation warnings. Run with the published `pbtest.py`,
`--checkout` this branch, `--python` the scoped interpreter, `--tag x86`,
`--shards 8 --workers-per-shard 2 --threads-per-shard 1 --mem-gb 4`,
`--timeout-s 600 --wait-s 1200 --priority -10`, and the ten modules listed in
the receipt file.

Original capture identity is retained. Priced anchors from the old producer
keep their old source seal; this pin change does not relabel or automatically
adopt them. New pricing binds the new producer. Full pricing remains paused
for publication-overlap implementation and measurement at this checkpoint.
