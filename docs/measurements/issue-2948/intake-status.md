# #2948 existing-owner artifact intake: unresolved

Refs #2948. Delivery remains with the existing #2999/#2427 owner/pipeline;
this note is a consumer-side intake status, not an original producer record,
publication receipt, artifact approval, or replacement implementation request.

## Available evidence and limits

The repository does not currently contain the acceptance handoff at
`docs/measurements/issue-2948/handoff.json`. No original selection/capture or
historical recorded 29-file roster was identified in the inspected repository
and targeted shared-measurement searches. Those searches were not exhaustive.
Shared fleet storage listing was denied. This establishes neither global
absence nor corruption; artifacts may remain with the existing owner elsewhere.
No credentials or specific host-privilege remedy have been established.

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

Observed: **2 failed in 0.69s**, exit 1. Both tests collected and failed at the
explicit missing-handoff assertion in `_records`, before any artifact joins.
This is an unmet evidence requirement, not a demonstrated acceptance-fixture or
application defect. Retained stdout was read back using `read_test_artifact`.
The acceptance tests and application code were not modified.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"92b3a4a7aacf04891db46b9e5713946334863eacb62941d50433639a99240777","size":3123}},"operation":"247b5282b24526e7179e4b1c9606ac73574004d735e906b4ef5bc0f1f437d26e","receipt":"d8ab3401962f8cdf797a9785de890c998e6326e8e716c1b36caadd8b3e4ba194"}]
