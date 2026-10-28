# Historical fresh-text LDLQ validation

The eight original `ldlq-fresh/` files are retained byte-for-byte here. They
describe the retired Gridbook codebook lane, including its lattice codebooks,
CB scale policy and `prismaquant.cb_layout` dependency. This harness is historical
evidence, not a current PrismaQuant command or validation gate.

The 2026-10-16 Claude review found a stale calibration-loader import. An admitted
PrismaBuild collection of the original test file first failed on the already
removed `prismaquant.cb_layout`; that dependency was also absent at the refactor's
base commit `138069b799323435fb5f0b3cec3ee339f6264a62`. Migrating one import would
not restore a supported lane. The harness, corpus, manifests and original report
therefore move together, without rewriting their historical bytes or results.

The original failed action is
`a466ce114c4016376ec82b6933dc35793ff78dec17b47befb373c74441473f78`.
The review's collection record and byte-comparison proof are retained under
`/home/rob/tmp/pq-claude-review-20261016/`.
