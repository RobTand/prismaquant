# Offline public producer oracle fixture for PQ #2894

This directory is TEST DATA, never an application verifier or serving adapter.
It archives the unchanged public, standard-library-only producer verifier so
CPU consumer tests are hermetic even when the installed Tessera predates this
public surface. Do not import these files from PrismaQuant application code.
The task configuration must name the deployed producer's verifier; this test
path is not an application default.

Source: RobTand/tessera main `17b3fdb396410698c9a142615345f046d2f94a5d`,
inspected from the served read-only checkout during research. Files are verbatim:

| File | SHA-256 |
| --- | --- |
| `src/tessera/endpoint_witness.py` | `d5dc6b1ee9aba42a2a803cb9ac1a0dc4ee0dae8ab2e5b7c5cd3a060917bdc14a` |
| `tools/verify_endpoint_witness.py` | `e08369a7021bf3fed3548e6a7544ed422305f55679d199acb5b21fca4c6c8c2d` |
| `LICENSE` | `6587d5df36a03f1b10b098a36928f7ff7f63bdd28a706850618de02d0eff3160` |

The minimal `src/tessera/__init__.py` is a test-only package shell. No serving
runtime, observer, network client, torch or vLLM is included. The executable
fixture is the real producer CLI, not a transcription of its validation logic.
It is always invoked with `--offline`. Tests must not download sources or look
up these git objects. The SHA values identify provenance, not signatures.

`tests/_served_task_fixture.py` ports PR #2470's tiny synthetic witness
constructors (approved head prefix `80f95d0271`). It independently selects tiny
artifact/tokenizer files, computes expected byte facts from those files, and
freezes expected loaded facts before witness mutation. Expected facts must NOT
be derived from an untrusted real witness in deployment.

## What the tests establish

- Synthetic two-rank CPU receipts exercise the public versioned contract,
  independent expected files and deployment labels, same-size substitutions,
  complete ranks, and supported ByteLevel source-to-loaded normalization.
- Small stub processes test process exit status and machine-verdict intake,
  including a valid-looking verdict from a failed process. Stubs establish no
  producer byte-binding fact.
- HF metadata providers are test adapters. Existing reader/replay tests retain
  their coverage; no checkpoint or inference is involved here.

## W1 evidence boundary

The inspected producer has the required public schemas and ByteLevel support.
Its committed `docs/measurements/tessera-endpoint-witness-live-2026-10-09.md`
reports action `25ea6519e540a7f8b30bead59a3895b45c519941e5b5310252cb01b8347c5240`,
a real tiny untied Qwen3 one-rank run, fingerprint
`9df7e024c5be2253b73b07dfa0ab822687baa3b0cc81b01d571fbe07b8148c5c`,
25 loader inputs, 19 resident tensors, 315889664 payload bytes. Research did not
obtain/replay that raw receipt and does not claim a W1 execution pass. No real
multi-rank, D50, tied-weight, task quality or current availability result follows.
That report is producer-attributed historical evidence, not a new verification.

The admitted installed pin `aaa417003f0ee690e8ace138acd18df7f9531369` lacks these
public witness modules in its source tree. The archived oracle avoids an
unrelated dependency upgrade in this package; existing producer reconciliation
work such as supplied open issue #2995 must not be duplicated. Deployment still
requires W1's admitted standalone verifier and independent expected inputs.

Every offline success is **recorded launch byte-binding evidence**, never proof
of a currently available endpoint, numerical equivalence, or served task quality.
Parent #2430 owns actual task measurement. #2469 and PR #2470 are the existing
consumer work to reconcile, not additional delivery lanes.
