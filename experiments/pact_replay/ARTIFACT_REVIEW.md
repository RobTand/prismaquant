# Corrected replay artifact recovery (#2482)

## What is and is not verified

The PR reports `corrected-de90c563e2.bundle`, SHA-256
`33e890ddea0624fe22ee2c27bac143178c6f58aa1f4d7ff1fa113537feb23aa7`.
Its bytes and an exact-digest publication record were not found in the accessible
retained evidence. That historical claim remains **unverified**. In particular,
the expected `cas/blobs/33/<digest>` object was absent. Searches of publication
records and accessible evidence found no match; some broad CAS searches timed
out, so this is not a claim that the artifact never existed.

There is an independently inspectable, already retained **different bundle**.
`IMPORT_PROVENANCE.json` now records its digest, byte size, fleet-relative
locators, embedded commit identities, and complete snapshot delta under
`corrected_artifact_recovery`. No archive was created or published in this repair.
Do not rename this bundle or attribute the missing bundle's digest to it.

The retained bundle advertises snapshot `272ccc34b135c57230ad1e1b26932516e7eef764`.
It contains the actual corrected commit
`de90c563e245f483ab54a7e668c41aec2d05fc41`, not merely a parent string in a receipt.
Unbundling into an empty object database succeeded without prerequisites. The
embedded corrected commit object equals the reviewed checkout's commit object,
and its tree is `87ead35047d8a6597347965e39a23cc0dcb0cb1a`. All 31 imported files
in that commit match `imported_files`. The snapshot's only change over that
commit is the added 153-byte `.pbrun-closure.7e4f8c8f1e579c2f.json` file; no source
file differs. Recover **the corrected commit**, not the snapshot tip.

The retained action `0c3694302889…` independently records the same bundle digest,
size, snapshot commit and parent. Its complete record digest is bound in the
provenance. This supplies a concrete artifact for exact-head review; it does not
establish the missing `33e890dd…` artifact's identity.

## Reproduce the read-only inspection

Set `fleet_root` to the deployment's retained fleet-evidence root. The locators
are relative to that root, so no application code depends on a particular mount.
Use an empty temporary object database, not the working checkout's history:

```sh
bundle="$fleet_root/cas/blobs/b3/b3d790efbc698b0d05989bfb8242504605d883266e48b741651e848adce10d7d"
record="$fleet_root/pb-queue/done/0c3694302889899de93e3b426bc4402751830afdf2a7fdeae536822f6ad29a94.json"
sha256sum "$bundle" "$record"
wc -c "$bundle" "$record"
objects="$(mktemp -d "$TMPDIR/pact-artifact.XXXXXXXX")"
git init --bare "$objects"
git --git-dir="$objects" bundle verify "$bundle"
git --git-dir="$objects" bundle unbundle "$bundle"
git --git-dir="$objects" show -s --format='%H%n%T%n%P' 272ccc34b135c57230ad1e1b26932516e7eef764
git --git-dir="$objects" show -s --format='%H%n%T%n%P' de90c563e245f483ab54a7e668c41aec2d05fc41
git --git-dir="$objects" diff-tree --no-commit-id --name-status -r de90c563e245f483ab54a7e668c41aec2d05fc41 272ccc34b135c57230ad1e1b26932516e7eef764
```

Observed bundle SHA-256: `b3d790efbc698b0d05989bfb8242504605d883266e48b741651e848adce10d7d`,
43,702,471 bytes. Observed record SHA-256:
`d1630e6085b24dd70a76f6a47657ef6ade19747f66ba92a342ad7ef09efb49b6`, 15,503 bytes.
The bundle has one advertised ref, `refs/heads/prismabuild-snapshot`, and no
prerequisites. The snapshot tree is `3267afaa6322863dc8cea0287c6d3f1ff8b98e13`;
its sole parent is the corrected commit. The corrected commit's parent is
`c9260081766ec8a47e7a0f357432833a4d3b1cb6`. The diff is exactly:

```text
A       .pbrun-closure.7e4f8c8f1e579c2f.json
```

After inspection, the temporary object database can be removed. Normal CPU
checks do not require fleet data or historical git objects. This metadata repair
is not part of the historical bundle; it does not relabel that bundle as a new
repair head.

## Review and execution boundary

The supplied PR discussion reports source approval at `de90c563…` in
[the exact-head review](https://github.com/RobTand/prismaquant/pull/2489#issuecomment-6084535272).
The subsequent independent TrustLadder review also confirmed the source and CPU
behavior, but withheld acceptance because the named artifact was unavailable.
Neither report is a new artifact approval by this implementer.

The pipeline owns independent final artifact review and any publication or
execution adoption. The CEO decision IDs and replacement conditions in
`replacement_authority` remain unchanged. Neither the frozen bundle nor the
private execution copy is modified. No failed D38 was repeated, no GPU replay
was run, and the separate PrismaBuild deadline defect remains out of scope.

## CPU regression verification

```sh
python -m pytest tests/test_pact_replay_deadline_2482.py tests/test_pact_replay_dependency_manifest_2482.py tests/test_capture_pact_replay_imports.py tests/test_duplication_baseline.py tests/test_io_site_freeze.py tests/test_prismabuild_boundary.py tests/test_tessera_core_boundary.py tests/test_architecture_doc.py tests/test_docs_staleness.py -q --basetemp="$TMPDIR/pytest"
```

Observed on the admitted x86 CPU arm: **70 passed, no skips**, with 14 existing
Torch deprecation warnings. This checks deadline behavior, byte-bound source
inventories, the CLI observer, unchanged production boundary gates, and document
consistency. It does not check fleet artifact availability or provide a new D38
or GPU result. Artifact inspection above is separate, read-only host forensics.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ac5d552f0cb1419648b1b019e27bb0f8a2da2ac6209f42657030e9e87c1e64cb","size":769}},"operation":"2731123c725e240a606079d15f272ac13cf2c82cf1f05b8d244ea08f5567ad44","receipt":"f4608c99fd48fcd923541e2b15cd9128df0e5a3bf1d071c3467728a6bf2deb39"}]

After the final metadata wording change, repeated the manifest and documentation
checks with:

```sh
python -m pytest tests/test_pact_replay_dependency_manifest_2482.py tests/test_architecture_doc.py tests/test_docs_staleness.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **22 passed, no skips**, with the same 14 Torch deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b78b648047400b70f5866d9422b087415283763076b1fe0abc0b08ff767084af","size":769}},"operation":"f5a49888a7f69829f5f232fff155ed5618d95835b0ef3c45c65268a17b1dcbe8","receipt":"5c5744ef55df3ed8c6693f37692f90245b161ae4051f263aa674f0805d569de5"}]

## Main integration conflict repair

Resolved the architecture-document conflict with main
`a794607fec04941bf944b2597bb0477b75457983`. Main's current audit entries and
architecture contracts are retained, with the complete #2482 replay note kept
among the October 9 historical entries. No replay source, test, manifest,
provenance identity, or boundary allowlist changes in this repair.

```sh
python -m pytest tests/test_pact_replay_deadline_2482.py tests/test_pact_replay_dependency_manifest_2482.py tests/test_capture_pact_replay_imports.py tests/test_observer_join_timeout_1367.py tests/test_duplication_baseline.py tests/test_io_site_freeze.py tests/test_prismabuild_boundary.py tests/test_tessera_core_boundary.py tests/test_package_tools_boundary.py tests/test_architecture_doc.py tests/test_docs_staleness.py -q --basetemp="$TMPDIR/pytest"
```

Observed on the default CPU arm: **72 passed, no skips**, with 14 Torch
deprecation warnings, in 33.81 seconds. The retained stdout was read back and
verified. These checks cover the integrated checkout's parser/resolver,
byte-bound inventory, observer, production boundaries and documentation; they
are not a new fleet D38, external-source equivalence run, artifact publication,
or GPU replay. The original failed D38 remains untouched, and execution
adoption still requires the pipeline's final review. An initial explicit
`arm="x86"` request was refused before execution; the default-arm call above
completed successfully.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7286108c2b31fabbd9db16c82f82394879cbbd3825df0ab469d4ccc5328a700a","size":750}},"operation":"3e22f84aec5cf74166e0803c7d5e2f2c1d88f7d6099fb1934cccb2333adb0b7c","receipt":"233815475b021271ba55cec6f0553e0280ebc378d99bf111628385e88d5cc55f"}]

After adding the integration evidence above, repeated the manifest and
documentation checks:

```sh
python -m pytest tests/test_pact_replay_dependency_manifest_2482.py tests/test_architecture_doc.py tests/test_docs_staleness.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **21 passed, no skips**, with the same 14 Torch deprecation warnings.
Retained stdout was read back and verified.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e6fa6f2ae10e1e3d692568489a7796dcd84c6c8de21b5632a577b7d0b340a208","size":749}},"operation":"e888d56bc5a66cfe199d8efbf1c202d7c55e9b05fcc905882e92dded8ef1e680","receipt":"f3f0a5017425419fb0c7365f780dc9aa4b5ca7c7ce809f00f3f0886ea112caf7"}]
