# D50 retained CPU API proof

This record recovers the already accepted protocol evidence for #2913 under
#2430 and #2411. It establishes historical replay only. No server was restarted.

The original source slices #2424 and #2420 landed together in #2433. Both
approved source heads are ancestors of source base
`05b62b4742793a2188abd4ed041fc4d46fd6c2c0`. This change reuses their task metric
and population owners and leaves the held production serving adapter unchanged.

## Original observation

The original evidence root is
`/mnt/shared/prismabuild-fleet/qualification/testcost-d50-cpu-task-api-proof-20261008/`.
The source record is
`/home/rob/fleet/records/testcost-d50-cpu-task-api-proof-20261008.json`.
It records independent parent acceptance on 2026-10-08 at 04:43:42 UTC.

| Identity | Recorded value |
|---|---|
| Action | `dbcc9990d780b5909ba83d462c95a829453ab68e007e7399c80c1db9bf6686c3` |
| Source parent | `2ea7000848a128f4dfd4abef894302001166a5cb` |
| Snapshot | `e8b99455876a72f6127d5dd0b0d5f9898fec36dd` |
| CAS receipt SHA-256 | `de8cb365b83ecadb9cf0b6b946c22387dcb2a9ede6ae20bad792d8d494956a6b` |
| Native outcome | dl380g10, done, exit 0, owned scope released |
| Client | lm-eval 0.4.12, public `LocalCompletionsAPI` |
| Server | llama-cpp-python 0.3.36, CPU, explicit beginning token |
| Task population | Two documents, four scoring choices, two generation requests |
| Sampling | Limit 2, no few-shot examples, all four seeds 17 |

The recipe retains exact server arguments, actual model-open observations,
model and tokenizer identities, client configuration, capabilities and cleanup.
The source helper records the original tokenizer-backed position check.
The historical explicit beginning token changes the served input. The proof
makes no comparison between different servers or artifacts.

The original successful model was SmolLM2-135M converted to f16 GGUF. Its
identity is historical evidence. This recovery neither reads that artifact nor
adds or changes GGUF implementation. The fixture contains no model weights.

## Replay

Run this command inside an ordinary admitted CPU PrismaBuild action:

```bash
python -m tools.replay_task_api_proof \
  --evidence-dir tests/fixtures/task_api_proof
```

The fixture contains exact original files. Its manifest joins portable paths to
their original paths and digests. Separate request and response bytes remain
authoritative over the transcript's parsed copies. Replay joins scoring prompts
to actual task arguments, derives the continuation boundary from token offsets,
and compares the sum with the retained public-client result. It replays task
counts and seeds through `task_suite.task_population`, and reads metrics through
`task_suite.task_metrics`. Generation replies also join to the raw samples.

Replay does not run the historical source helper, tokenizer, model or endpoint.
A successful replay is not a new live API smoke or a current deployment witness.
It supplies no quality promotion criterion or shipcard pass.

## Remaining root prerequisites

The original source hold on #2911 still requires the intended launch's
accessible producer-owned runtime witness, including loaded artifact and
tokenizer bytes, endpoint, alias, launch attempt and actual complete ranks.
The existing #2458 supplier remains the owner. No replacement producer or
server was created. The current PrismaQuant Tessera pin does not contain
`src/tessera/endpoint_witness.py`; later Tessera source cannot silently replace
that pin. A compatible public consumer handoff must be established explicitly.

Dependent #2912 still owes serving-backed owned-byte replay and artifact binding.
The aggregate G3 artifact-equivalence refusal and multi-host lifecycle
prerequisites documented in architecture section 7.0 remain in force. All current
lane slots, explicit criteria and independent review remain required. No GPU,
native Tessera quality, performance, energy, residency or next-model readiness
result follows from the retained CPU evidence.
