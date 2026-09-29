# Legacy NUL source records — #1774 / #1386 / #1301

`LegacyNulSourceSha256` names the inherited incremental SHA-256 recipe:

```
for name, payload in caller_order:
    sha.update(name.encode("utf-8"))
    sha.update(b"\0")
    sha.update(payload)
    sha.update(b"\0")
```

UTF-8 is strict; there is no length prefix, header, count or final trailer.
The owner does not discover, filter, sort or retain files. Caller traversal,
validation, source exclusions and ordering remain caller responsibilities.

**This is not an unambiguous file-map encoding.** #1762 records a measured
counterexample: `a.py: b"first\0b.py\0second"` and the two entries
`a.py: b"first"`, `b.py: b"second"` contribute identical bytes. The compatibility
suite retains this fact rather than silently hardening the qualification
protocol. A migration changes producer/consumer qualification bytes and belongs
to the coordinator/Tessera owner; #1774 changes neither protocol nor pin.

## Callers

- `runtime_provenance._source_digest`: Path-order keys and source suffix filter
  are unchanged; only incremental record encoding delegates.
- `tessera_reader._source_tree`: Path-order discovery, regular-file and symlink
  fences, complete-package refusal, and resolved per-file identity map are
  unchanged. Existing raw-byte identities use `bytes_sha256hex`.
- Associated `ArtifactReader.bytes`, `_source_tree_identity` member digests,
  `_package_source` installed-file map and `_ReaderSourceLoader.get_code` use that same plain-byte owner. Declared source
  comparisons, artifact mismatch refusals, loader AST restrictions and JSON
  serialization profiles are unchanged. No serving runtime is imported.

## Verification

`tests/test_nul_source_framing_1774.py` freezes literal legacy vectors and exact
owner inputs, including Unicode names, binary/empty payloads, caller ordering,
source exclusions, member maps and source/artifact/loader refusal paths. The
#1762 counterexample is an explicit compatibility case, not a security claim.
Existing provenance and reader-namespace suites, compile, the source-primitive
ratchet and integrity inventory run through PrismaBuild. No production speed,
quality, residency or served-artifact claim accompanies this byte delegation.
