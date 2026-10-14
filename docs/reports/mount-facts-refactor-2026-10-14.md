# Kernel mount facts

The staged-range reader and portable source identity now read mount facts
through `mount_facts.find_mount`. It decodes the kernel's two table formats,
selects the longest containing path and selects the last row at an equal
mount point. A frozen record exposes filesystem and options. Reader-specific
read shape, cache and device-number policy stay with their existing owners.
The identity adapter still resolves the path before comparing it.

The alternative was one parser per proc table, with three caller-specific
selection loops. One parser supports both actual kernel formats without
adding another residency cache or an I/O engine. The pin-release reader's
different equal-path behavior is corrected in a separate commit after the
ownership move.

PB `e6766b2dfec13cdb0d4c7d3f2b637d3d05d1e46007c566113995e8feaf065ff9`
passed all 380 baseline probe, mount, staged-read and runtime-constraint cases,
with 11 subtests and no skips. Its result CAS digest is
`c7047db2448ec2184f595dd44ebd6abb48851cb86ad38d1eac7365680fe580be`.
PB `d72aef90ee15761b242e42ce97758a4ba8ec46d777bbcbb5133791549afa5cef`
compiled all four touched modules and passed 221 mount, reader, identity and
architecture cases with no skips. It failed only the composed campaign digest
duplication check. The independent shared-validator correction passed that
guard in PB `1d77dee590e483c34e86ae5ca588bcd4231b6e452508e374899ea4b1ceeab24e`
and is included in this branch. The failed terminal remains recorded as a failure.

No read layout, tensor arithmetic, production defaults, admission or
performance claim changes in the ownership move.

## Pin-release correction

PB `3ca3b91e8aa57c84975131b16ac0905b557afc155fd299afc7eb30101783892b`
reproduced all three equal-path NFS/autofs failures: `/mnt/shared`,
`/stage/prewarm` and `/ram/prewarm` returned zero instead of180 seconds.
Pin release now uses the shared mount facts and the existing NFS timeout
formula. The duplicate parser is deleted.

PB `0ea3bb68c57738d946ee56069cd231668d37383001348937f2fb415316b26109`
compiled both touched owners and passed all112 targeted mount, pin, lease
lifetime and duplication cases with no skips. Exit0; result CAS
`d2ab7f356fbac2f46c90e655f9781bb67e8ce2afed309d0b9a23d788a7fa66b0`.
