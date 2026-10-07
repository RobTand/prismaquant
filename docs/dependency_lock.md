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

The v3 consumer tests declare one pure metadata module in
`tests/fixtures/rung_allowability_v3/producer.json`.
The declaration names the merged Tessera commit, source path and actual module
SHA-256. The existing isolated producer adapter uses this declaration by default.
It verifies the fetched module bytes before a child process imports them.
The child imports no Torch, Triton or serving module.
An explicitly supplied module and its byte hash remain supported.
The dependency does not alter the serving pin or the installed serving package.

Fleet test commands can use `tools/provision_tessera_pin.py` to provision the
unchanged development pin in a scoped environment under `/tmp`.
An older qualified interpreter is not a substitute for that current source.
The command records both the installed source and its byte identity.
The metadata adapter and the serving runtime remain separate dependencies.

