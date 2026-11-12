# Immutable #3017 acceptance fixtures

These files are test inputs, not repaired executable code or new GPU evidence.
Research recovered them once from the supplied Git object store. Acceptance tests
never read Git history, fetch source, use Docker, or open external measurement
roots.

- `run2/*.py`: original four-file harness at PrismaQuant
  `dffc6a6b72949a5d149406c66bcb16ccfd74e4e7`. The digests match the preserved
  run2 manifests.
- `native-v2/*.py`: preserved five-file harness at
  `4e5380f573dff3319fd687a26b82678eaead7dcd`. It contains the bug under test.
- `snapshot-digests.json`: byte identities of those two snapshots.
- `preserved-files.json`: SHA-256 and size of 107 preserved measurement artifacts,
  correction/input documents, per-suite records and receipt identities from
  `4e5380f573dff3319fd687a26b82678eaead7dcd`. Derived publication manifests,
  README and live harness are deliberately not frozen by this inventory.
- `requalification.json`: retained native-v2 index from that same commit. Its
  measurement-bearing fields must survive integration. Additive validation
  bindings do not relabel those launches.

The behavioral acceptance tests use `native-v2` only if the application package
is entirely absent, to expose the preserved control-flow failure rather than an
import error. Package integration is independently mandatory. Once the package
exists, behavior tests exclusively load its live `harness/` modules.

## Additive integration contract

`results/harness-bindings.json` in the recovered package maps
`historical_run2`, `retained_native_v2` and `current_validation` to objects with
`root` (package-relative directory) and `sha256` (exact filename-to-digest map).
The two recorded snapshots must be separate immutable directories inside the
package; `current_validation` points to `harness`. The requalification index adds
`harness_binding: retained_native_v2` and
`validation_binding: current_validation`. This keeps measured launch hashes
unchanged while explicitly selecting recorded versus current validation bytes.
Historical validators must consume the historical binding; qualification
validators select retained runs through the index and the retained binding.
These bindings alone do not establish GPU execution of repaired code.
