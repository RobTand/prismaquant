# Prepare a new artifact for publication

The publisher enforces the PrismaQuant Weights License for the **one artifact
submitted to it**. It does not rewrite previously published cards or bulk-edit
Hub repositories. Do not run a real publish without the operator's explicit go.

## Stage policy files before measurements

1. Confirm the destination `rdtand/<repository>` identifier.
2. Copy `licenses/PRISMAQUANT-WEIGHTS-LICENSE-1.0.md` into the new artifact as
   `LICENSE`, replacing both `<repository>` placeholders with the destination's
   repository basename. Preserve every other byte, including the final newline.
3. Add these fields at the start of `README.md`, replacing `<repo-id>`:

   ```yaml
   ---
   license: other
   license_name: prismaquant-weights-license-1.0
   license_link: https://huggingface.co/<repo-id>/blob/main/LICENSE
   ---
   ```

4. Include the canonical license's first `> **TL;DR.**` block in the card body.
   Markdown reflow is accepted; different legal terms are not. Preserve required
   upstream notices separately. Do not claim measured quality or serving
   qualification until the corresponding evidence exists.
5. Open the authoritative shipcard and collect/fill the required evidence only
   after staging the final LICENSE. **LICENSE participates in `model_sha`.**
   Adding or changing it after measurement changes the artifact identity;
   neither this tool nor an ingest step can repair that by relabeling receipts.

Policy files must be ordinary UTF-8 regular files, not symlinks, and at most
16 MiB each. Front matter must be a mapping with explicit, unique string keys;
YAML merge keys cannot disguise the license fields. The source and frozen
snapshot both undergo the same policy check. `--force-unverified` never waives
this policy and never stamps the card when the policy check fails.

## Rehearse without changing the Hub

Through the normal admitted CPU validation path, run:

```sh
python tools/publish_artifact.py /path/to/final-artifact \
  --repo-id 'rdtand/<repository>' --dry-run
```

The existing shipcard verifier and complete-file freeze must pass. A successful
dry-run performs **no Hub request**, does not import Hub API bindings, and does
not verify that the remote access setting has been configured. Missing evidence
is still a refusal. No raw upload command is emitted.

## Publish only after explicit go

The destination repository must already exist; `--private` remains an assertion,
not a visibility change. A real publisher invocation first freezes and verifies
the local artifact, resolves/enumerates one remote parent, sets `gated="auto"`,
and reads the setting back **before any LFS preupload or commit**. Unsupported
API versions, API failures, and any value other than the exact string `auto`
refuse without uploading. The existing parent-commit CAS, byte replay and final
file-set audit remain in force.

Before announcing success the publisher reads the gate again. Failure after a
commit requires repository inspection, not an automatic retry or a success
claim. Successful publication output records the observed auto-approve gate
alongside the commit and snapshot identities. Tests exercise these calls with
a mocked Hub; they neither configure a real repository nor upload model bytes.
