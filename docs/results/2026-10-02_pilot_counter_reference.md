# Quantum counter references: CPU provenance prerequisite for PQ #1293

Current fan-out admission verifies the caller's counter document digest and a
successful PB action key. It does not bind those bytes to the action's result.
The original producer final stdout carried only quantum/status/unit fields;
its counters existed only in the external output tree. Root Astra independently
confirmed the P1 gap and approved this producer prerequisite.

The existing publisher now encodes counters once, atomically publishes those
same bytes, then records their SHA-256, byte length and path in results.json.
Its existing final stdout includes that reference under the typed completion
schema `prismaquant.joint_layer_quantum.completion.v1`. A failure before counter
publication exposes no reference or success files. The completion owner was
extracted first, preserving the old output fields before adding the reference.

The shared pilot-domain validator compares an authenticated completion with
the supplied counters digest/length, quantum ID and complete integer unit
counts. Changed bytes, foreign IDs, gapped/nonboolean success, inconsistent
units and malformed references refuse. Paths are provenance, so an exact copy
may be consumed. The validator does not authenticate its own argument.

Remaining consumer work is explicit: use the public bounded PB reader for the
chosen execution's verified CAS result AND sealed request; verify the actual
quantum invocation, source closure, plan/regime/shape and chosen ending. A
self-asserted document, arbitrary successful action or canonical CAS winner is
insufficient. No weak/private CAS reader was added. This prerequisite does not
close #1293, replace the genuine GPU pilot, change numerical/default/serving
contracts, or supply a performance/energy claim.

Validation uses the pinned project environment on dl380g10: Python3.14.4,
torch2.11CPU, Transformers5.16.1, PB95a59051/Tesserab40c93cb, CPU1/4GiB per
shard, native threads1, priority-10 and CUDA disabled. All tests/compile ran
through the published PrismaBuild tools. Reports and complete unambiguous
terminal/CAS metadata are in `/home/rob/tmp/astra-resume-20261002/pq_spill/`.

## pilot-producer-red.json

- tests/test_pilot_counter_reference_1293.py: 2 failed, 1 passed, 14 warnings in 5.11s; action `0bd44315d759fb24a4fd44f2876ed05df25f386b217d7fc42dbd589355c254fd`.

## pilot-producer-green.json

- tests/test_pilot_counter_reference_1293.py: 3 passed, 14 warnings in 4.60s; action `6717a0eebae0cab7466203f8ff9206a9141fd0c0cae0da1ff384eac29fda5563`.
- tests/test_joint_cost_quantum_runtime.py: 37 passed, 14 warnings in 25.48s; action `e88a5a6546080558b4a5947d510e79b0a6ddab7c2d5ad7745f71305e8e0a1f33`.
- tests/test_joint_dispatch_pilot.py: 17 passed, 14 warnings in 5.46s; action `d28e9a83d92c8a8b2c51bbce4e504b2ba7fcb2fc0f22342a4ccab9fe45795094`.

## pilot-reference-green.json

- tests/test_pilot_counter_reference_1293.py: 12 passed, 14 warnings in 4.85s; action `8d4a192e914a2b2f64a1d623ad975c96598e46560257248aaad9a9e49594ed1a`.
- tests/test_joint_dispatch_pilot.py: 17 passed, 14 warnings in 5.69s; action `cc6daad9bbecb68da2ace2607380d98a2591bf5b859b15b4d98cf5d7be6a242e`.

## pilot-structure-green.json

- tests/test_architecture_doc.py: 13 passed, 14 warnings in 4.68s; action `1766bd47095e12950f7339c43e207d2c2a2eae518cc09b5e605702224d71ed13`.
- tests/test_docs_staleness.py: 6 passed, 14 warnings in 4.35s; action `6f123990fe1dbc94683a96961acaff65d47892bca653dc4561eb50f5c5665b5c`.
- tests/test_duplication_baseline.py: 7 passed, 14 warnings in 33.55s; action `d1cca49d5972e69b5747dd63165689eedf54c35471b60fdf3e73071576d110bc`.

The RED baseline retained the new publisher tests on unchanged8811ad5f: two missing-reference failures and one preexisting failure-publication pass. Reference-stage tests cover the final Python blobs (12new+17pilot, no skips); the preceding37quantum cases cover the identical quantum publisher blob before the separately defined domain validator was added. Structure26cases all pass. None of these CPU runs is a real GPU pilot.

Three-module compile: action `db65c6cc16aa1c9c7fd85505a9f6e8fc84719c304a1e95f3bc5257a8cac0161d`, exit0, CAS receipt `8b783f07129fed929854d2ed47c19b3feb70d3d672e8a7788b930bc9cff37858`.

Both reference-test snapshots and the compile snapshot were independently loaded from hash-verified PB source bundles: all three delivered Python file hashes match. Each successful CAS result payload was hashed and compared with the receipt; this is not a claim of full worker attestation or future consumer-request qualification. Exact bindings: `pilot-source-evidence.json`.
