# Retained dense pricing completed before the performance pause

Three additional selected dense groups completed on the frozen pricing source
before withdrawal reached the remaining work. Their actual outputs were
checked under PB action
`3cd38d77fba877f32f57421abcd844eca2070e24b37efd3f33d2371204b88f41`.
The verifier source is `experiments/glm_completed_dense_audit.py`.

| Group | Units | Measured anchors / wire hashes checked | Interpolated rows |
|---|---:|---:|---:|
| row-0003 | 2 | 14 | 1016 |
| row-0004 | 2 | 14 | 1016 |
| row-0118 | 1 | 7 | 508 |

Each reached three rounds, reports no early stop and performed zero source
forwards. Verification binds selected unit membership, original capture,
pricing package identity, checkpoint identities, measured scores and actual
wire bytes. Interpolated rows are explicitly distinguished from measurements.
The three native action keys are:

- `74b588f7464a431c8eff2420589319ead099d2add5f5a4901a0b1f32f3b32481`
- `83db680aad9daf6ea7aadc3564bd3c6e7b79769b6cfa065dcea1313b181b5714`
- `90d0fdfa4e7ddb28494c2e62ce69bde4f741f617437f85a91c844d52434d7668`

All native terminal records confirm exit 0 and cleanup. The CPU verifier's
CAS payload hash is
`6890addaf085f3b4447f0b9cf0e450c865aaa7f6acb3f922fe43b975eb2b2259`;
root independently checked the payload, source snapshot and receipt.

Evidence lives under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/`
in `root-three-more-dense-artifact-audit.json`,
`root-three-more-dense-cas-source-audit.json` and
`root-three-more-dense-verification-cas-audit.json`.
Retain these completed results; this audit neither resumes incomplete groups
nor establishes full-model quality or a shipping qualification.
