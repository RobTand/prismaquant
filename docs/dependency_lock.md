# Dependency lock maintenance

`pyproject.toml` declares project dependencies; `uv.lock` is the generated
resolution, including the editable project's version, Python floor and optional
extras. Update them together. Naming a dependency only in root metadata is not
sufficient: the corresponding resolved package and transitive references must
also exist.

Adding an already-resolved base dependency to an optional extra still needs a
lock refresh: the extra's root reference and its `requires-dist` marker must
both be present. The Pillow test extra is covered by the same per-declaration
checks as every other extra.

The #1756 refresh uses **uv 0.10.10** for both generation and verification.
The generator version is intentionally recorded here because `uv.lock` does not
record it. Use the same version when reproducing this refresh. Do not use
`--upgrade` for a metadata-only refresh: preserve existing resolutions wherever
they still satisfy the declarations.

With that version and a matching project interpreter, the relevant commands are:

```sh
uv --version  # must report uv 0.10.10
uv lock --python "$PY" --no-python-downloads --no-progress
uv lock --check --offline --python "$PY" --no-python-downloads --no-progress
uv sync --locked --extra test --dry-run --no-install-project \
  --python "$PY" --no-python-downloads --no-progress
```

Run these checks through PrismaBuild, obeying its admission rules and any active
campaign window. Verification must leave the lock byte-identical. A dry-run
checks the declared test-extra install plan; it is **not** an actual dependency
installation, GPU validation or a serving-compatibility measurement.

`tests/test_dependency_metadata_1756.py` checks the Python floor and project
version, each declared dependency's resolved package and root requirement
metadata, and the `pytest-xdist` → `execnet` edge. It does not call a resolver or
require network access. The uv check above independently verifies resolver
freshness rather than trusting this structural check alone.

## Canonical metadata tests

The D41, version-dispatch, canonical-consumer and MTP allocator tests use
`tests/_rung_allowability_producer.py`. Its authoritative declaration is
`tests/fixtures/rung_allowability_v3/producer.json`: repository `RobTand/tessera`,
commit `d9ec5291791ea09592f96c7e75a45d8f5cf23b1c`, source path
`src/tessera/rung_allowability.py`, SHA-256
`3e2b904b18231a7283cce581fbc45d824f8191a3a011d104f342ed90ac7d6821`.

With `TESSERA_RUNG_ALLOWABILITY_MODULE`,
`TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256` and `TESSERA_PRODUCER_PYTHON` unset,
the adapter fetches that exact source from GitHub's raw-content endpoint,
checks its declared digest and runs it with `sys.executable`.
Only authenticated downloads are cached, by declaration identity, in a temporary
directory. Each resolution reads the current environment selection; each call
rechecks the selected source bytes. A failed fetch or a downloaded digest
mismatch fails explicitly with the declared repository, commit, path and digest.
There is no skip, alternate producer or synthetic admission fallback.

For offline execution, set `TESSERA_RUNG_ALLOWABILITY_MODULE` to a readable
local copy and `TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256` to the declaration's
digest. Both must be supplied together, and both the companion digest and the
actual source bytes must match the declaration. Missing or empty companions,
conflicting digests, unreadable files and changed bytes fail before execution
with the dependency identity and override information. An authenticated override
performs no network request. `TESSERA_PRODUCER_PYTHON` may select an explicit
interpreter; otherwise the current interpreter is used.

Canonical validation and admission stay in the child-process adapter. The child
runs Python in isolated mode, hashes the source again and executes the bytes it
hashed, without using cached bytecode. Import guards reject other Tessera
modules, Torch, Triton and vLLM during module loading and producer calls;
caught forbidden import attempts still fail the invocation. Successful calls
and canonical `ValueError` refusals retain source and interpreter SHA-256
evidence and the empty forbidden-import list. Canonical refusal messages remain
`ValueError` messages in the parent. Failed invocations clear previous evidence.
The publication fixture restores the consumer's producer binding on teardown.
This test dependency does not alter production admission, the serving pin,
format menus, PrismaBuild or the installed serving package.

`tests/test_rung_allowability_producer.py` uses controlled bytes and transport
failures to check provisioning and isolation. Those controls establish no
canonical admission qualification. The operations follow-up must run the actual
declared source through the composed consumer tests on the production-pinned
interpreter, then repeat through the normal PrismaBuild runner without injected
producer environment. For the direct composed run, with `PY` naming that
interpreter:

```sh
env -u TESSERA_RUNG_ALLOWABILITY_MODULE \
    -u TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256 \
    -u TESSERA_PRODUCER_PYTHON \
    "$PY" -m pytest -q -rA \
    tests/test_rung_allowability_producer.py \
    tests/test_rung_allowability.py tests/test_rung_allowability_versions.py \
    tests/test_canonical_consumer_join.py tests/test_glm_mtp_selection.py \
    tests/test_mtp_formats_menu_1692.py
```

Retain interpreter identity and digest, actual producer source digest, the
command and the complete pass/fail/skip report for both runs. Mocked provisioning
results alone do not establish compatibility with the normal runner.

Fleet test commands can use `tools/provision_tessera_pin.py` to provision the
unchanged development pin in a scoped environment under `/tmp`.
An older qualified interpreter is not a substitute for that current source.
The command records both the installed source and its byte identity.
The metadata adapter and the serving runtime remain separate dependencies.

