# Render read-bound metadata: controlled CPU evidence

Refs #1247. This is a metadata-only prerequisite, not the issue's full NFS
preparation, concurrency or GPU acceptance.

## Change and invariants

`_prepare_file_read_bound` reads each distinct render path's size once per
invocation, in first-seen order. Each new invocation still reads current sizes.
The maximum and declared PWC read-buffer limit are unchanged. Missing files,
zero maxima and oversized maxima still refuse. No file-size cache persists
between calls; no weight cache, preloader, body-read path, quantization,
export/serving format, pin or stage topology changed.

## Regression and syntax checks

All execution used PrismaBuild on eligible x86, with the main-matching
`/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python`, priority -10,
one CPU/native thread and one test shard. The U4 sentinel was independently
checked before each client. Tests reserved 4 GiB and 120 s for RED or 300 s for
GREEN; profiles reserved 4 GiB/120 s and compilation 2 GiB/120 s. No GPU ran.

| Check | PB action key | Outcome |
| --- | --- | --- |
| RED, unchanged production code | `a78b246478417f43cd4a728be4364a269ef40afd10f85bde88ea3b54577f19e7` | 1 failed, 3 passed, 0 skipped; 4 collected/ran; 5.78 s |
| GREEN, six CPU test modules | `d8905b356763998e74adc79453dc937f52bed9a99af2dc41c9430152767526af` | 124 passed, 0 failed, 0 skipped; 124 collected/ran; 3 deselected; 161.21 s |
| Two-module compilation | `6afd3ed993bd7e9524caa12897786048e6fbd70c59fa64b526fbb8dc23e73aa1` | executed, exit 0 |

RED fails because two cells referencing each file each issue a stat, while the
regression expects one per distinct path. The other three tests establish the
unchanged fresh-size, missing-path and zero-size guards. GREEN also includes
the existing preparation/budget tests, architecture/staleness checks and
seal/duplication policy checks. The CUDA-only existing decode test was explicitly
deselected (three parameterized cases), not skipped or certified.

Public terminal outcomes and CAS v3 action/worker identities were checked.
GREEN receipt `e9544dcf23a04c64db923e31279023e7bebdcb8dbaeea7626be818f85295ab11`
has a 30103-byte result with SHA-256
`aecd017f79b9783f83293c40c00ecf7941eee148fa110da5fa02205e0864d4ed`.
Compile receipt `92102ed7892162e621821f8ca5a6a9c93fae320ba7ed673b802d6be4b88ef053`
has a 24-byte result with SHA-256
`bf6b34fc4c4629aab19f60c39623107d0a0a0eb44694fdee11d662df37994b1b`.
Payload lengths and SHA-256 values were independently verified.

## In-process before/after profiles

The identical cProfile fixture makes 16 helper calls over 32768 cells sharing
32 files, with the same 16384-byte maximum. Both admitted runs executed on
dl380g10's local btrfs filesystem, not NFS. PrismaBuild assigned affinity `[1]`
BEFORE and `[5]` AFTER; neither was reset or widened.

| Metric | BEFORE | AFTER |
| --- | ---: | ---: |
| `Path.stat` calls | 524288 | 512 |
| `Path.stat` cumulative seconds | 4.863694 | 0.005581 |
| Profiled helper wall seconds | 7.069069 | 0.147319 |
| Maximum bytes, every call | 16384 | 16384 |

This proves removal of duplicate metadata operations in the controlled fixture.
The wall numbers include cProfile overhead and different assigned cores; they
are not a production speedup, NFS floor or model throughput claim.

- BEFORE action: `12ef2e72becdb16e1c035d8906acc7f8ca94b3079ad2ade6f9f5feca3ea02e11`;
  receipt `cdfa43f0574604e76a76bd614310efb841e8bdac62ab92fadc1e9f9b3a160663`;
  result 5508 bytes, SHA-256
  `b134215cfb2a202f4e3d573276684404b20b8d04d4a96a831b319a89536b3672`.
- AFTER action: `f58658c2a775fcd56b7dfb7e8afaf2924ae7e3c3dac3b1703662961b5c6de337`;
  receipt `6515cbae2bc859c1d282fc1aec71e48448449eb098865324804616b767303e8f`;
  result 5729 bytes, SHA-256
  `6a0526b4488b95e8e810443b804818043d318ac749a13ba1aaa14505ac54a632`.

Both public terminals were executed/exit 0. Receipt identities and payload
lengths/hashes were independently checked.

## Box telemetry and limits

Netdata CPU, IO, RAM and CPU/IO pressure series were captured on sparky,
sparklina and dl380g10 for both actual in-process windows:

- BEFORE: Unix 1790742553.8214893–1790742560.8905587, seven overlapping CPU
  samples per host.
- AFTER: Unix 1790743637.9336624–1790743638.0809813, one overlapping CPU
  sample per host. Surrounding samples are retained; the sub-second window
  limits attribution at Netdata's sampling resolution.

Fetch/coverage errors were zero for all fifteen series in each capture. On
dl380g10, BEFORE mean CPU user/system were 8.418786%/2.780151%; the single
AFTER sample was 11.132099%/2.757839%. Other work was present on the boxes;
these measurements do not establish an isolated throughput comparison.
There was no GPU measurement, power/energy comparison, real model preparation,
KL/bpp comparison or changed concurrency.

## Commands and retained artifacts

Campaign directory:
`/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/prismaquant/`.

- `prepare-bound-red-profile.sh`: sequential PB regression RED and BEFORE.
- `prepare-bound-green-profile.sh`: CPU GREEN, identical AFTER and compilation.
- `prepare-bound-profile.py`: full deterministic cProfile fixture, submitted
  inline through pbrun rather than executed locally.
- `capture-prepare-bound-netdata.py`: read-only HTTP capture aligned to the
  profile timestamps; no tests or GPU work execute in this utility.
- `prepare-bound-red.json`, `prepare-bound-green.json`: reconciled test results.
- `prepare-bound-profile-before.log/json` and
  `prepare-bound-profile-after.log/json`: complete profile results.
- `prepare-bound-netdata-before.json`, `prepare-bound-netdata-after.json`:
  full host series and coverage records.
- `prepare-bound-before-evidence.json`, `prepare-bound-final-evidence.json`,
  `prepare-bound-before-public-terminal.txt`,
  `prepare-bound-final-public-terminal.txt`: verified receipts and terminals.

#1247 remains open for representative NFS preparation/concurrency measurement
and any GPU acceptance requiring the coordinator's approval.
