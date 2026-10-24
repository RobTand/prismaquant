# Artifact owner-route fragment

Production scope: only `tools/publish_artifact.py`, `tools/provision_tessera_pin.py`, and `tools/tessera_campaign_container.py`.

`ledger.json` records nine replaced original calls, using the exact IDs and source blobs from the acceptance registry `tests/fixtures/digest_parent_completion_1301.json`, source commit `6d8288adc94b3c4b624c143ed372fb6513f32784`. The hash and JSON serializer at the manifest site have separate original IDs, despite sharing one replacement expression. Test IDs have no parameter suffixes.

Original fixtures are small extracted original function/class bodies, with standalone stdlib imports and the original dataclass decorator. The publisher additionally freezes the original manifest digest expression, canonical-byte expression, and path-quote expression as callable witnesses. Fixtures do not import the migrated owner. Dependencies/constants required by extracted functions are supplied explicitly by tests.

Retained algorithms: descriptor-bound incremental scanning and replay (including chunking and refusals), and the complete isolated installed-target probe. `_scan_file` accumulates its whole-file hash during the same held-descriptor `_pread_exact` pass that builds block digests, sample, bounded inline content and read-call accounting; a separate whole-file/path read would not preserve that integrated scan. `_replay_frozen_digests` instead rereads non-inline entries through held descriptors until EOF and checks both size and digest after upload, refusing divergence. These are distinct descriptor-integrated retention reasons, not tree-framing arguments. Source-witness tests compare unchanged replay/probe ASTs, and compare scan/tree ASTs after reversing only the nested bytes-call replacement.

The provisioning outer path/NUL/ASCII-hex-content-digest/LF hash is unchanged but is **GAP**, not retention. Its unusual recipe does not justify retention: it needs an exact shared owner in follow-up shared-owner work, outside this evidence-only slice. The tree AST/behavior witnesses establish preservation only, not owner completion.

### Host bootstrap audit boundary

Inspected invocation witnesses: `tools/accept_tessera_pin_provisioner.py` sets `ROOT` from its checkout location and invokes `ROOT/tools/provision_tessera_pin.py`; the dated `docs/experiments/tessera_best_form_pin_20260909/provision_cpu.py` invokes checkout-relative `tools/provision_tessera_pin.py`; `tests/test_tessera_pin_provisioner.py` also anchors `TOOL` under `ROOT` and checks that the provisioner imports neither PrismaQuant nor Tessera. Those witnesses support checkout context, **not a tool copied alone**.

`test_provision_host_checkout_bootstrap_isolated` stages only the actual host tool, `prismaquant/digests.py`, and the actual default pin source with their checkout-relative layout and no package `__init__.py`. From a different working directory, `python -I -S` loads the host via `runpy`, parses its default pin and invokes the bytes helper, asserting the helper came from the staged `digests.py` and no PrismaQuant, Tessera, Torch or site module was loaded. A second isolated invocation exercises script `--help`. This proves stdlib-only bootstrap in that layout; the new host dependency on `digests.py` is explicit. It does not establish support without that file, standalone copied-tool support, real provisioning/install behavior, or arbitrary deployment layouts. The embedded installed-target `-I` probe is a separate boundary: its unchanged source and mocked invocation remain witnessed, with no new target-side PrismaQuant requirement.

`audit.json` distinguishes retained descriptor algorithms and the target probe from the outer-tree and supplemental JSON gaps. Insertion-order CLI JSON and the indent=1 insertion-order provisioning manifest have no exact existing profile and remain unchanged. Gaps carry no invented behavioral witness claims. No owner, profile, census, baseline, scanner, deployment roster, authority, or qualification source was edited. Pre-existing changes in other tools and the sibling images fragment were left untouched.

## Checks

Final command (run through `run_tests`):

```
python -m pytest -q tests/test_digest_1301/tessera-supply/test_artifacts.py tests/test_publish_artifact.py tests/test_tessera_pin_provisioner.py tests/test_tessera_campaign_container.py --basetemp="$TMPDIR/pytest"
```

Latest review-fix result: **117 passed**. Includes six original-vs-new behavioral/source-witness tests, the new isolated checkout-layout host bootstrap test, and the existing three tool suites. Tests exercise bytes and JSON failures, Unicode, empty/multiblock data, truncation/mutation refusals, framing sensitivity, target subprocess isolation, archive-hash refusal before Docker load, and Docker inspect/load ordering. Docker calls are mocked; no live deployment or target installation was attempted. The full repository suite was not run.

An initial six-test run failed two checks due to incorrect owner API names (`stream_sha256hex` and profile `.bytes`); corrected to the inspected existing `file_digest_sha256hex` and `.encoded` APIs. Subsequent targeted runs passed. `git diff --check` also passed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"408e8b5624292d720b3e5402337f26accacd9740ef59f0306b8e05e2b4931aa2","size":350}},"operation":"d2818f4455e94d95712383191dcf49907ce60bbd5fee9d954cccc35783f0c17c","receipt":"dc9a5b23bd09e6267e41bc4962a2de47af0625c53136a569cd240a7bcc960fc4"}]
