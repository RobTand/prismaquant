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
