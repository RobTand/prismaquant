# Provisioning the Tessera pin: what a real install proves that a mock cannot

Issue [PrismaQuant #455](https://github.com/RobTand/prismaquant/issues/455).
Branch `claude/pq455-provision-tessera-pin`.

`tools/provision_tessera_pin.py` installs into a fleet interpreter the Tessera
the pin names, and then checks that it took. Its tests place a package
directory into site-packages and ask what the tool decides, which is the right
shape for a decision and blind to two things only a build backend does: leave
the pinned tree's bytes alone, and install a subset of them.

Both turned out to matter, and one of them had already happened.

## The digest could never have matched a real install

Tessera's `pyproject.toml` carries
`[tool.setuptools.packages.find] exclude = ["tessera._dev*"]`. That subpackage
is the repository's own tooling, and a wheel does not ship it. The pin's tree
holds 76 files under `src/tessera`; a correct install of that tree holds 71.

The tool compared the two whole. On a correctly provisioned interpreter it
would have found drift, installed, re-probed, found the same drift and exited
1, on every run, forever. Every synthetic test passed because the fixture
copied the whole source directory into site-packages, so the two sides were
equal by construction.

The exclusion is now read from the pinned tree's own build configuration
rather than known in the tool, and applied on both sides. Tessera's
`tools/check_wheel.py` proves the same exclusion on the built artifact.

## The shared pins tree is not the commit, and pip is why

`/mnt/shared/tessera-pins/07ad344c3275bb2fa7ce2432f93d89945d66f4c2` digests to
`adb6511e...` over 1256 files. A `git archive` of that commit digests to
`926f2f03...` over 1179. The 77 extra files are 71 under `build/` and 6 under
`src/tessera_quant.egg-info/`: output of an earlier `pip install` run with the
frozen tree as its build directory. An earlier draft of this page named a
`_pb_native_moe_measure` directory among them, read off a top-level listing
rather than the diff; it is in the commit, and the egg-info is what was
missed.

Nothing served from that tree is wrong -- `build/lib` is not a package under
`src/` and never reaches an install -- but the directory is a working copy
wearing a commit's name, which is the state a pin exists to exclude. The tool
now builds from a copy under `--staging-root` and re-checks the tree against
its manifest afterwards, so the property is proved rather than arranged.

**The live directory has not been repaired**, and root has since ruled it
retained as bounded evidence. Its replacement is a rename, and the repair path
is two of them: the old name is vacated, then the new tree takes it, and
between them the pinned path does not exist. Nothing makes that atomic, so the
ordering is what there is to get right -- the staged tree is hashed and
attested first, and a failure there leaves the old tree untouched. Run a repair
when nothing is serving out of the path.

## What the acceptance runs

`tools/accept_tessera_pin_provisioner.py` makes a throwaway venv, provisions
it for real, and reads the artifacts rather than the log. Seeded from a copy
of the live shared tree, so the repair path runs on the real bytes:

| check | result |
|---|---|
| the install did not write into the published tree | pass, `926f2f03...`/1179 before and after |
| the tool says the tree is intact | pass |
| the first run installed the pin | pass, `installed the reviewed source` |
| the second run is a no-op | pass, `none, already the reviewed source`, drift `[]` |
| `--check-only` reports clean | pass |
| the tree it replaced was kept | pass, `.quarantine-07ad344c...-20260909T050854Z-j_ilktv2` |

`unshipped_packages` reads `["tessera._dev*"]` off the pinned tree, and the
wheel it built (`tessera_quant-0.1.0-py3-none-any.whl`, 725628 bytes) installs
to a package the digest matches. That equality is the thing no placed fixture
could show.

## Receipts

| what | action key | where |
|---|---|---|
| RED, `9fe2f18f`: 6 failed, 12 passed | `3fd2bbb5d18d` | dl380g10, `pq-cpu312`, `--tag x86` |
| GREEN, `45a2ed6d`: 18 passed | `b7619de49cf6` | same |
| acceptance, unseeded | `9fbb480fd14f` | same |
| acceptance, seeded from the live tree | `df2e22218e8d` | same |
| GREEN, `86e60a2f`: 19 passed | `2fc2f307ca5e` | same |
| the ordering mutation: 1 failed, 18 passed | `b961932c705e` | same, `pb-queue/failed/` |

The RED is a committed intermediate, not a scratch tree: `9fe2f18f` changes
tests only and `45a2ed6d` is the fix.

The mutation row is the same tree with one statement moved: `_write_manifest`
back after the quarantine rename, where it sat until this round. The only test
that fails is the one that asserts the ordering, which is what makes that test
worth having -- a check that passes on the code it is meant to catch is not a
check.

## What this does not establish

One interpreter, one commit, one architecture. The GB10 venv is not
provisioned by this and should not be until root rules on the PQ #455 test
environment contract. The exclusion is read from setuptools' `packages.find`;
a build backend that excluded files another way would need reading another
way. And a lock serialises publication per commit within one filesystem's
`flock`, which is what the shared NFS mount offers and not a distributed lease.

`/mnt/shared/tessera-source.git` is a mirror taken once, so the default clone
holds the commits that existed when it was made. A re-pin to a newer commit
fails inside `git archive` with an error that names the object and not the
mirror; `git --git-dir /mnt/shared/tessera-source.git fetch --prune origin`
before a re-pin, or pass `--clone` a local checkout that has it.
