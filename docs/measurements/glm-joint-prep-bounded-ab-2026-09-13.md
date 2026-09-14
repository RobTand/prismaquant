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
