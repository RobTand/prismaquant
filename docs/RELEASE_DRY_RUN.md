# Rehearse a release without publishing

This local check builds an exact committed candidate, validates its wheel and
sdist, and compares both artifact versions with a temporary local tag. It does
not push a tag, publish a package, upload an artifact, or deploy anything.
It does not replace GitHub checks, independent review, or the Mergify merge gate.

## Run

Install `build`, `twine`, `packaging`, `setuptools>=77`, and `wheel` in a clean
virtual environment first. The build uses `--no-isolation` and disables the pip
index, so install the candidate's build requirements before running it.

From the repository root, use a new output directory for each run:

```sh
python .github/scripts/release_dry_run.py \
  --revision HEAD --tag v0.16.2 --output /tmp/pq-release-good
python .github/scripts/release_dry_run.py \
  --revision HEAD --tag v0.16.3 --output /tmp/pq-release-bad
```

For the current `0.16.2` package, the first command must exit zero and the second
must exit nonzero with a tag/version mismatch. Adjust the good tag when the
package version changes. Both commands build and validate real artifacts; the
bad command must not pass solely because a subprocess failed earlier.

Each output directory contains `validation.log`, `receipt.json`, and local
`dist/` artifacts when the build reaches that stage. The receipt binds the
candidate commit, tag, result, and successful artifact SHA-256 digests.
Existing output directories are rejected rather than reusing stale evidence.

The script fetches the candidate only from the supplied local repository into a
temporary repository. It creates the tag there and removes that repository on
exit. It neither creates nor changes tags or branches in the source repository.
Uncommitted and untracked source changes do not enter the candidate.

The subprocess environment excludes inherited tokens, publication credentials,
OIDC variables, Git overrides, and Python import overrides. It uses a temporary
home and the system executable path. The build interpreter is the same Python
executable that runs the script. The trusted caller's `check_dist.py` validates
the candidate's runtime data alongside `twine check --strict`.

Build backends execute candidate code. Run this only for trusted local or
independently admitted same-repository candidates in a disposable environment.
Environment cleanup is not a sandbox: it does not restrict network access or
prevent candidate code from reading other files accessible to the OS account.
Never run it on a fork with credentials or a persistent privileged fleet runner.

## Focused tests

```sh
python -m unittest discover -s .github/scripts -p release_dry_run_test.py -v
node .github/scripts/linked_issue.test.cjs
```

This slice does not change `.github/workflows/release.yml`, branch protections,
`pb-tests`, queue admission, or review requirements. Queue provenance and fleet
CI remain separate work under #3140 and Paperclip PRI-1057; activation requires
the manager's design synthesis and CoS's verified protection transition.
