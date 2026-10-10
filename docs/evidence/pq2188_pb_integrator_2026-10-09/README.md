# prismaquant#2188: Python 3.12 pure-Python SDK5 artifact, built by pb-integrator (2026-10-09)

Routed to pb-integrator by issuegraph (effect `81ebe16dacfc`) because no repository change could meet it. No repository file was changed. CEO decision `rep-1009-232238-0eff` allowed one CPU-only image check.

## Result
- Artifact: `/mnt/shared/astra-pq-2188-evidence-20261009/original-cuda-test-deps-02/pure-python-installs.tar`, sha256 `9923cb74caa5e5fa1c41c9aef809941bf21c4687e53fd0b30ed781a0ce49023c`, 16,885,760 bytes, 744 members. **Not in git (16 MB); it stays on the mount.**
- prismabuild `027103d9a8417e06c7f13356e58779a313cd7088` (SDK5) and tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`, both non-editable git installs; pytest 9.1.1 and its pure-Python packages.
- Build: PB action `cc8658368b74c2f3b842a83933740c7e97caf350f3c1da2b75f3108f84f0ddb6`, CPU only, dl380g10. Python 3.12.11.
- Image check: PB action `c736f9b072a0681454387a8a6203880060f73546543755ac954ab5dfaebc60b9` on sparky: image `prismaquant-glm-derivative@sha256:c0e532d2...`, CPython 3.12.3 aarch64, PASS.
- Posted on the issue: comments 6090942708 (build) and 6091027946 (image check).

## Files here
| file | what |
|---|---|
| `build.sh` | the build script, byte for byte from `/mnt/shared/fleet-ceo/pb-2188/` |
| `run1.log`, `run2.log` | the build submissions. Run 1 was refused: a helper script must be inside the snapshotted repository. Run 2 built the artifact. |
| `run3-image.log` | the image check output |
| `check_in_image.py` | **reconstruction** of the image-check script (the original was lost with `/home/rob/tmp`; see its header) |
| `BUILD-PROVENANCE.json`, `.sha256`, `receipt.json`, `extraction-check.json`, `pip-freeze.txt`, `SHA256SUMS` | the metadata files that sit beside the artifact, byte for byte. `receipt.json` is the packer's, unedited; its fixed text says "not new install/provenance", which does not describe this artifact, so `BUILD-PROVENANCE.json` says what it is. |

## Not changed
The old artifact (`original-cuda-test-deps-01`, sha256 `4de08759...`), the SDK guard, every existing venv, and every repository file.
