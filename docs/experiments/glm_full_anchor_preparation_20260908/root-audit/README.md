# Root integration review — 2026-09-08

Reviewed the observer, adaptive loop and regression tests, original-roster
preflight, draft generator, spec and both qualification rows. The draft artifact
matches the committed bytes; its generator and dispatcher match the executed
PB source snapshot. All positive actions below have zero terminal return codes,
completed scope cleanup, checked receipt-body hashes, CAS payload hashes and
source bundle hashes. These checks do not assert signer attestation verification.

- Observer: 26 passed, one native CUDA capture skip; compile succeeded.
- Adaptive progress: reproduced the permanent interior-member failure on the
  original loop, including repeated encoding of the successful sibling. The
  fixed branch passed 31 campaign tests and 19 architecture/document tests;
  compile succeeded. Resume retains prices and reaches missing interior work.
- Integrated architecture/doc checks: 19 passed, no skips. Production/test
  bytes were unchanged by the later preparation-only documentation merge.
- CPU draft generation: exit zero, exact reviewed generator/spec/dispatcher.

All actions used PB x86 CPU workers with native threads bounded to one.
The linked JSON files record source comparisons and attributable payloads.
No native anchor qualification, full-model serving or performance improvement
is claimed by these CPU checks. The 132-row campaign remains an unfrozen draft.

A separate prose commit corrects the Tessera gold-validation examples to the
actual offline LLM CLI. This is the only incidental change in this integration.
