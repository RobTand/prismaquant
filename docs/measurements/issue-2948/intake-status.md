# #2948 existing-owner artifact intake: unresolved

Refs #2948. Delivery remains with the existing #2999/#2427 owner/pipeline;
this note is a consumer-side intake status, not an original producer record,
publication receipt, artifact approval, or replacement implementation request.

## Available evidence and limits

The repository does not currently contain the acceptance handoff at
`docs/measurements/issue-2948/handoff.json`. No original selection/capture or
historical recorded 29-file roster was identified in the inspected repository
and targeted shared-measurement searches. Those searches were not exhaustive.
The earlier intake reported denied shared fleet storage listing. On this
follow-up, the shared measurement, evidence, results, investigations and CAS
roots were listable. Bounded targeted searches still did not identify the
required owner-provided artifact set; neither result establishes global absence
or corruption. Artifacts may remain with the existing owner elsewhere. No
credentials or specific host-privilege remedy have been established.

Follow-up candidate inspection did not establish a historical selection/capture
join: the pin-named `pq503-tessera-pin-7dbbacbd` measurement directory contains a
source tree, not an identified original 29-file roster;
`fused-selected-config-cpu.json` is a test-surface measurement reporting 1,146
files; `ig-ts-1053-dependency-surface.json` is another test-surface result,
reporting 2,082 files. The provision record
`glm-resident-hessian-provision-cas-source-20260909.json` contains an install
audit for 71 package files at pin
`7f8aef0dd1ba4ef6a6ff5cf00ee9c2bc06c73e5d`, but no demonstrated join to this
issue's historical selected image. None was retained as a substitute. Searches
were bounded, some stopped at tool limits, and CAS was not exhaustively scanned.

The supplied issue discussion attributes the delivery requirement to Rob's
2026-10-24 comment on #2948, explicitly reusing #2999/#2427. That discussion
identifies ownership and required evidence; it is not the original evidence.
Archive integrity and agreement on the historical runner hash remain bounded
bridges, not authentication of all installed paths, source bytes, dependencies,
or authoritative historical image selection. Historical build evidence remains
unavailable to this intake and has not been reconstructed or counted as success.

## Objective status

- **O1 unmet:** original selection and capture records, their unchanged bytes,
  and the test index have not been delivered here. Immutable selected-image
  identity and its exact capture join cannot yet be verified.
- **O2 unmet:** original absolute installed-path observations, source bytes and
  hashes, and original image-bound dependency observations/bytes cannot yet be
  verified. Today's checkout or installed environment is not a substitute.
- **O3 unmet:** existing #2999/#2427 owner must supply original provenance for
  authoritative historical selection and capture, the complete recorded 29-file
  tessera-pin roster with original relative paths, pin identity and hashes, and
  the complete unchanged dependency closure. No consumer-created roster or
  arbitrary subset has been introduced.
- **O4 unresolved:** canonical retention/delivery and independent exact-head
  artifact verdict have not been established by evidence available to this
  intake. No publication or independent approval is claimed or performed.

On owner delivery, retain original bytes/layouts without restamping. Only then
add the small repo-relative test index locating original JSON fields and byte
references, verify sizes/hashes and image/source/dependency joins, and refer
historical authenticity/completeness and the delivered exact head to the existing
independent review lane. Mechanical acceptance alone cannot establish those
properties. This note deliberately does not create a placeholder `handoff.json`.

## Verification

Command:

```sh
python -m pytest tests/test_issue_2948_selected_image_capture.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **2 failed in 0.66s**, exit 1. Both tests collected and failed at the
explicit missing-handoff assertion in `_records`, before any artifact joins.
This is an unmet evidence requirement, not a demonstrated acceptance-fixture or
application defect. Retained stdout was read back using `read_test_artifact`.
The acceptance tests and application code were not modified.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5eeba67f1f32f4669807eff8be49be0ee82432fcc46ba9c191e05477154f95e8","size":3123}},"operation":"10543fb4154495512b035504ce2e60cfa9316eac08369481de65aafdc53f8815","receipt":"ff7e014be8b33da77dc1a5f02c73066e7a15a6a70010ffd49f1c375489b727f1"}]
