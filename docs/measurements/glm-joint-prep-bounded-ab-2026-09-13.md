# GLM joint preparation: bounded dense and expert qualification

This measures 18 original Tessera cells, nine rungs each for
`model.language_model.layers.0.mlp.down_proj` and
`model.language_model.layers.3.mlp.experts.0.down_proj`. It is a before/after
qualification experiment, **not** a prepared artifact or a full-campaign
throughput qualification. The full local report `AB_REPORT.md`,
the raw `results.json`, `results.pstats`, Netdata series and PB terminal/CAS
records are retained under
`/home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/`.

Frozen source `c7a376689c1ae9a92cfa72380fafa449dc26f8e2` and candidate
`b80faca03d616007ff8219dc5697b3359478fa3a` used the same canonical plan
SHA `d88ff538532c3acd51df17d261c3f59fb2ac235d75fc2a5be4dbbaa2b273c0d0`,
selected-record fixture SHA
`ee697d7a15661ca613bb111e0cb29b2c8d797b3646236160f20b3a224ed2dafe`,
harness SHA `76ecc60273d5f39221e21f9c49ba88a5b4e499d0768cc50b63ab79dc1d7d217a`,
and 52-entry PB data-manifest SHA
`54b944368b1e400648202bc563ed73acea27ee67e7e3ee0f5dfe32de843570d5`.
They ran in the same pinned ARM image on Sparky's GB10 with six assigned CPUs,
104 GiB declared shared/GPU memory, exclusive PB measurement admission, and
`head=1800,dense=900,expert=900` semantic progress. Source proof was local to
Sparky's `st_dev=64`; a PB preflight matched all 120 full-file source SHAs.
The CPU-only fixture build ran under PB action
`33a626f804e7a2b2bf572f76b913eb7f07c8896b04133ad5cfd2a0ae60b0ce6a`
(rc0/CAS). It derived all 197,990 real PWC key/path entries from the bound
merged campaign and authenticated the two selected unit envelopes; it did
not validate every unselected journal cell. Both arms verified selected live
source/H, capture, wire, render and activation receipts. The 18 normalized
verification records had identical SHA
`c7f55ca2ac1a5ebf94637ccaf87dd0c32074e9cca9d41f43b0db8cdb9bb29756`.
The selected wires total 252,829,605 bytes and their serialized render files
1,057,008,847 bytes. The complete roster's BF16 render tensor payload sums
to 3,328,515,768,320 bytes, a lower bound on serialized archives; it was
derived from shapes, not read during this bounded experiment.

| Measured interval | Frozen c7 | Candidate b80 |
|---|---:|---:|
| PB action key | `55ba891ed34ae9e9de9b05597d0e6b2d7933a0d5097c4c67244c9cc55414e65d` | `0599e3a503a22f8ee18a2a84a9a56a30f17c7380206f1bb63ad8d7986bc0b318` |
| Verified cells | 18/18 | 18/18 |
| Selected-unit hot time (source install excluded) | 16.248 s | 14.938 s |
| Selected-unit throughput | 1.108 cells/s | 1.205 cells/s |
| Source layer installs | 17.391 s | 15.290 s |
| Complete bounded qualification interval | 33.669 s | 30.262 s |
| Client `/proc/self/io` `read_bytes` | 17.334 GB | 17.336 GB |
| CUDA allocated/reserved peak | 28.707/33.544 GB | 28.707/33.544 GB |
| Candidate selected journal setup/two fsynced writes | — | 1.241/0.0047 s |

Both actions exited rc0 and published CAS receipts
`f341f161e2dd97fd0ededa2d375c1cb825d828fe1ea8eb2834ae0b6eb0d86311`
and `b77feb82067b8fef3ed8f334329ab0f2c1a35e898d52e532567a69a484648515`.
The candidate was 8.8% faster in selected-cell throughput and the complete
bounded interval was 10.1% shorter. Source installation, outside the edited
hot path, accounts for 2.10 seconds of the 3.41-second whole-interval
difference. In-process cProfile shows `_window_resident_storages` (14 calls)
0.4055→0.0328 s, window planning (8 calls) 0.3188→0.0911 s, and capture
`_load_execution` (2 calls) 0.5426→0.0001 s. The source/H encoding-identity
bind stayed 7.534→7.550 s. These cumulative function times are diagnostic;
they are not additive wall-time fractions. The wire read-ahead's isolated
effect was not measured, and no benefit is claimed for it.

Phase-aligned one-second Netdata on both boxes recorded Sparky GPU power
12.706 W mean over 34 baseline samples and 14.850 W over 30 candidate samples,
against its approximately 140 W envelope. Mean power times interval is an
estimated 427.8 versus 449.4 joules, or 0.0421 versus 0.0401 cells/joule:
the candidate was about **4.8% worse** in this short energy sample despite its
lower latency. The separate PB whole-action pqteld means were 10.5508 and
10.5407 W over different windows and must not be mixed with phase energy.
Sparklina carried more unrelated load during baseline than candidate. There
is no robust full-campaign work-per-joule ranking from one short pair.

Both PB prewarm sidecars report completed reads of the same 52 entries and
20.487 GB before qualification. Server-side ARC/disk samples over the
observed baseline window showed 247,438 demand hits, 2 misses and 36,864
bytes read from sdb–sde; candidate showed 273,508 hits, 0 misses and 155,648
disk bytes. The first roughly seven seconds of baseline lacked server samples,
so fully matched ARC residency is not established. Client NFS `read_bytes`
must not be interpreted as server disk traffic. Source/H identity work and
source installation kept the GPU near 10% of its power envelope despite
server-RAM delivery on observed intervals.

Run the same local frozen/candidate harness through
`python3 /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/submit_bounded_ab.py baseline --phased --label baseline-phased`
and
`python3 /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/submit_bounded_ab.py candidate --phased --checkout /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/code --head b80faca03d616007ff8219dc5697b3359478fa3a --label candidate-phased`.
The wrapper verifies the image, fixture and harness hashes and submits the
identical PB resource, phase and data-manifest policies. The candidate's
journal measurement uses an 18-cell selected identity with the full qname
roster, so neither the production 197,990-cell journal-manifest startup nor
full preparation memory/runtime is established. A full metadata walk was
withdrawn after 15m21s at unit 23,079/36,423 with about 40 GiB peak memory
and NFS metadata waits, without publishing any fixture. This negative result
is retained in the full local report and ruled out using it as A/B input.

## Steady routed-expert role panel

The two-unit panel above included once-per-process Tessera encoder-fixture
setup. A second bounded panel warms that memo and installs the full source
layer 3 *before* timing real qualification of nine distinct experts
`0,35,71,107,143,179,215,251,287`, each with gate/up/down projections.
There are 27 exact source/H journal units and 241 measured original wire cells.
The identical PB-selected fixture SHA is
`c87108e9c56c85e853cdc3443a9c2ae8a8b8f6e05cc95a5f0ee80271ac18595b`
(126,311,954 bytes), with all 197,990 PWC key/path entries. CPU-only PB
builder `84913bd588c6a4557055c8250f88ac37f40c0c1afa45cb105ef8db054b1a5afd`
and source/fixture preflight
`835a1e69f42f909986ecbb91233efd5e725ffdf723f298c973848b93b8adeda5`
both exited rc0 with CAS receipts. The same harness SHA
`c6166b17514280cd910ff00dadc2e7aef053c66ed1318e26ab335da5b0b95609`
and 522-entry, 23.910 GB head→expert manifest SHA
`dc187f3a7ca221be5e85ce353e080ab1156b61fe19ff498bc753cb4bb9cd3009`
were sealed for both Sparky GB10 arms. All 241 normalized source, render,
encoding, wire, origin and activation records matched exactly (SHA
`46a61630668238dfb8191b4e7da9d1b2a446f0052751ff5073276b865635f944`).
The candidate's 27 journal unit files were independently read back and
SHA-verified; its final PB progress count 27 followed the last durable write.

| Steady role measurement | Frozen c7 | Candidate 85dd |
|---|---:|---:|
| PB action | `05e18e45ac8694086175e0374c5b895e783a1e77a0417e2d006be4c45f71d0c7` | `b15ff2c1b50471a48c62da85925c8fa1d549a4a4cc190dcee40cb9410fe43b80` |
| Verified cells | 241/241 | 241/241 |
| Timed steady qualification | 42.646 s | 27.127 s |
| All-role steady rate | 5.651 cells/s | 8.884 cells/s |
| Gate rate, 80 cells | 5.120 cells/s | 7.814 cells/s |
| Up rate, 80 cells | 5.559 cells/s | 8.568 cells/s |
| Down rate, 81 cells | 6.414 cells/s | 10.873 cells/s |
| Encoder fixture memo warm (excluded) | 5.138 s | 4.956 s |
| Source layer 3 install (excluded) | 17.140 s | 16.615 s |
| Client steady `read_bytes` | 6.54311 GB | 6.54331 GB |
| Candidate selected journal setup / 27 writes | — | 1.284 s excluded / 0.102 s included |
| CUDA allocated peak | 19.010 GB | 19.010 GB |

Both PB actions reached terminal rc0/cleanup and published CAS receipt SHAs
`caed24edfe60e64d02470c58a5dc132136592cc4a30e69391ca02704f9205b9d`
and `fc005eeaf1ef316bf573598229f90687f8fc3c4349e53c77e212ab811bc3a43f`.
The candidate completed steady work **36.4% sooner**, or **57.2% more
cells/s**. The in-process profiles isolate two repeated costs:
`_window_resident_storages` (185 calls) 5.360→0.034 s and capture
`_load_execution` (27 calls) 7.343→0.0006 s; PWC window planning (106 calls)
3.955→0.799 s includes some of the resident scan, so those cumulative times
must not be summed. Tessera's encoder identity bind stayed 2.235→2.216 s
after memo warming. The wire read-ahead effect has not been isolated against
the other changes. Candidate journal setup is for these selected 241 cells;
full production journal setup and source-authentication startup remain
unmeasured.

Phase-aligned one-second Netdata on both boxes recorded Sparky power at
12.735 W mean over 43 baseline samples and 13.848 W over 27 candidate
samples, against ~140 W. Mean power times exact steady duration estimates
543.1→375.7 joules, or **0.4438→0.6415 cells/joule (+44.6%)** on this
workload. The separate PB whole-action pqteld means were 11.9366→11.7825 W
over longer intervals and are not phase-energy readings. Sparklina stayed
near 4 W, with CPU user means ~0.66→0.71%; Sparky user/system/iowait means
were 4.73/1.42/2.24%→4.39/1.42/3.41%. The GPU remained near 9–10% of
its power envelope, so this is improved useful work/J within a still
underfed hot path, not GPU saturation.

Server ARC/disk samples during both timed steady windows showed **zero ARC
demand misses** and only 16 KiB baseline / 1.64 MB candidate global disk
reads despite ~6.543 GB client NFS `read_bytes` in each. Prewarm timing was
not identical: candidate's manifest completed while READY, whereas frozen's
completed after claim and incurred ~657 MB server disk reads before steady.
Source install is excluded precisely because its starting cache state differs.
The timed windows were both observed ARC-hot; the counters are global, not
per-action bytes. The 1-second energy sample and one-layer selection do not
establish full-model energy or latency.

The sealed merged cost declares routed populations: gate 12,096 units/
65,552 measured cells, up the same, and down 12,096/65,664. Weighting each
role's observed per-cell *steady* rate by those counts yields a diagnostic
34,833→22,078 seconds of pure routed qualification, a modeled 12,754-second
(3.54-hour) difference. It excludes the remaining dense units, changing
per-layer H/content and cache state, full journal identity setup, source
installs, startup, failures/retries and the later COST stage; it is not a
full-preparation ETA or a shipping throughput guarantee. The complete local
steady report and raw files are under
`/home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/steady-{baseline,candidate}/`.
The submitted commands used the fixed benchmark checkout's wrapper:
`python3 /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/benchmark-code/tools/submit_bounded_ab.py baseline --expert-steady --label steady-baseline`
and
`python3 /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/benchmark-code/tools/submit_bounded_ab.py candidate --expert-steady --checkout /home/rob/dq-runs/glm-campaign-takeover-20260913/prepare-rewrite/code --head 85dd0d78f9de7ac18ca70f3eaa334309c9ce51b4 --label steady-candidate`.
The wrapper's `submission.json` records exact PB argv and image/mount source
identity for each arm.
