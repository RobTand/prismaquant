# Bound Shape-Time Checker Acceptance

The retained singleton was converted, reloaded and admitted against PQ's
independent pinned TP1 scope through the public SDK4 reader. The row is
`dense|256x256|TESSERA_E4M3_K1|R896|M512`, with 30 original CUDA-event samples,
five warmups and original median `0.07507199794054031` ms. There are no rate
pools. This reuses the recorded GPU measurement; the new work is CPU only.

The original panel remains SHA-256
`fd5c2b4f24ffdae8058d28c39f38ada9f943eb8836d0433ecdcc584e36836129`;
the original request remains
`3d252977205a7938b641a0215ac59f185526ba5f400f65b7c7010f9acbbfa577`.
Their measurement producer is `8eb3c05174c292c5c1cf1e0d32f72802e6d2c8cf`.
The installed runtime remains Tessera `b40c93cb73745097e57a1ba4cf5b9eee166c759a`,
raw contract `0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa`,
and image `f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.

## Execution Evidence

| Work | PB action | Result | CAS receipt SHA-256 |
| --- | --- | --- | --- |
| Actual installed CPU preflight and full Tessera checker | `cf0484b46363408ced3c50c3b1f39bd7e064735f3cc2d940126a164e5d0122c4` | rc0, Sparky, no GPU attached | `a205b6544942dc696976499ca1f84528dad6ef1b95309dffcfa753da643b51c6` |
| SDK4 inspection of full snapshot and working directories | `9812f104ebbf5a21c8c7cf35cfe95633a20731a49e9014f6ee4a1b295ab7c66f` | rc0, dl380g10 | `174ece32812f09c01daf3835af758c0590677fbeac5634e5576475d00d3a59a4` |
| Final real convert/load/admit and fully forged negative controls | `49e3447f8af8ebcc353790a9fe3ea2a80af6f405cdc6609d30b9b7230f6f57eb` | rc0, dl380g10 | `0f9beeff61a31133d518a7f60931327a19d3d49df3d685fd8836a60e7a84298a` |
| Targeted shape prices, shared provenance and architecture controls plus compile checks | `c510916d0477d8407d4cad796fb1f2458ecf357f789ea4bad07a0aad3d8a4621` | 177 passed, dl380g10 | `6c4a5e7afa90477c73827ce4d0a96b079cc456022348ceca595fed14b4fbeb7f` |

The exact checker selector is publication `1790970739.694331`, attempt `1`.
Its source snapshot commit is `d28f1dfef1c0187ab5744dcd7c65685168ded3f7`,
parent `c7f6d1aa1a1a57d31936271931d8b859823d94be`, schema v2, subdirectory
`.`, empty refs, and input `pbrun.checkout-snapshot` of 12,945,026 bytes at
`718f4e907470eb580cff8d7468de3fd58b75ace75f4b5bbbecb201e6bad72d1d`.
The independent candidate checker config also pins both working directories,
the entire sealed command, environment and observation output. Its bytes hash
to `7a8c58b951b0c55a4244ac97c1f24d87a9a528eb86d72b461589d5aa1cbc5e25`.
Root retains final source acceptance and merge authority.

Artifacts are retained under
`/mnt/shared/tessera-measurements/pact-observation-bound-20261002-sol/`.
`r2/observation.json` is the genuine checker output;
`sdk-final-trace/acceptance.json`, `table.json` and `checker-receipt.json`
record the final trace. The table hashes to
`d1a4e3792db89ce7f379debe0893c722d4316df494515f1cae361f58aed81722`.
The forged observation, panel, samples, proof and table are bounded negative
evidence in that same trace directory. TP2 admission and both forged conversion
and reload refused through the real public SDK reader.

## Controls And Mode

Causal PB action `c26d1422cb35f1765106133b511a505c8b23b52351d176447feaa4b68856a0df`
demonstrated acceptance of a fully fabricated panel with matching hashes and
no checker. Action `50d439f3f5d452f1391ab30b41ba8175b285c8776ead4a0485e34498e3da210e`
demonstrated a changed family escaping comparison through a relative receipt
path. Both now refuse. The latter fix resolves row receipt paths against the
table's parent, preserving valid relative observations and proofs.

Domain controls use an explicit fixture SDK; they do not certify PB execution.
They cover missing or wrong selectors, source input drift, older selected
commits in the same bundle, subdirectory/refs/working-directory drift,
command/environment/wrapper substitution, duplicate checker output,
oversized observation bytes and changed table context/key/lane/samples.
The separate actual trace uses the real SDK4 reader and capture binder.

The CPU tests used Python 3.14.4, CPU torch 2.11.0, Transformers 5.17.0,
PB commit `dc4803daaf09b6426083d2d36bd2a2da3d6832fe` and unchanged installed
Tessera b40 in `/home/rob/venvs/pq-pbdc4803da-tessera-b40c93cb/bin/python`.
The final targeted command compiled the four touched Python files and ran
`pytest -q -n 8 tests/test_shape_runtime_prices.py tests/test_runtime_provenance.py
tests/test_architecture_doc.py tests/test_docs_staleness.py` through published
`pbrun`, reserving eight CPUs and 10 GiB, with native threads one and CUDA
withheld. No tests skipped. PyTorch emitted its Python 3.14 JIT deprecation
warnings. The actual checker used the original known-good container and its
original installed runtime; test interpreter versions are not a new runtime
qualification.

The acceptance command is
`experiments/pact_checker_receipt_acceptance.py --sdk-root <reviewed SDK4 tree>
--action-key <checker key> --published-unix 1790970739.694331 --attempt 1
--observation <r2/observation.json> --output <owned trace directory>` inside
an admitted CPU action. It calls the existing SDK helper override, public
result reader/binder and PQ's ordinary conversion, load and admission owners.

This is one operator-sum proposal row. Full PACT, TP2/MoE coverage, model
quality, compiled serving, placement, energy and served-latency qualification
remain outside this acceptance. The PQ branch is based on `693a38f3` and
retains the #2092 descriptor-lifetime and #2082 owned-quantum-read fixes.
