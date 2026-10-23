# Digest #1301 serial integration readiness

## Verdict: prerequisite not met; no closure claim

Inspected checkout HEAD: `8bc0b5af8c69af352b559dccda99bd4e6fa8b418`.
Its parent is research base `6d8288adc94b3c4b624c143ed372fb6513f32784`.
The working tree was clean at inspection. No integrated consumer ledger fragments
exist under `docs/audits/digest_1301/` or consumer contract files under
`tests/test_digest_1301/`. The census contains `refresh_2540`, but no
`current_1301`. These observations establish that the supplied tree is not the
post-consumer integration tree required for this serial piece; they do not
establish the status of work in other branches.

A concrete pending handoff is allocation publication: at
`prismaquant/allocator.py:5361`, `main` still executes
`hashlib.sha256(digest_src.encode()).hexdigest()[:12]`.
The acceptance publication comparison reaches its final owner-routing assertion
and fails there. The allocation consumer owner must supply the exact owner
route and its per-call original-source fixture/test ledger references. Other
consumer owners must likewise integrate their implementations and Agreement S
ledger fragments before this piece can aggregate and reconcile them. Missing
fragments are not evidence for retention.

No consumer, owner, scanner, baseline, protected source, historical census or
acceptance file was edited. In particular no placeholder current census or
fabricated retention was published. A current validator and final census/baseline
reconciliation remain pending the integrated source and evidence. Independent O6
sign-off was not performed. No GPU, numerical, serving or performance claim.

## Checks

Executed the O5 command, which includes all four individually named acceptance
nodes for O1–O4, and the existing profile/file/guarded/standalone/ratchet checks:

```sh
python -m pytest tests/test_digest_parent_completion_1301.py tests/test_digest_profiles_1301.py tests/test_digest_profiles_1301_files.py tests/test_guarded_read_digest_retention_2637.py tests/test_stdlib_tools_no_package_1605.py tests/test_duplication_baseline.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **4 failed, 202 passed**, 14 torch deprecation warnings. Failures:

- O1: `census is not current (historical refresh_2540 must remain separate)`.
- O2: `parent closure still has eligible/unresolved calls; migrate exactly or prove retention` (this check falls back to historical rows; it is not a current eligibility audit).
- O3: `no completed per-call migrations recorded for the remaining parent work`.
- O4: `allocator.main's remaining Pareto identity must use its exact digest owner`.

These are unmet implementation/integration prerequisites, not acceptance defects.
The remaining 202 regression cases passed; this is not a parent completion pass.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6a46d0a22c306d045eaca29d266b44d205fa45922be1b507742b3afbe639668f","size":11545}},"operation":"cab4f89b73be5e2e07f765182f03144c594deb7fd4e5c9cb2298c26411b40740","receipt":"24d503f3f9018c56b196fa062b821fb9421085038204b971251c8d0e63698f38"}]

Executed the separately requested dynamic completed-call consumer command:

```sh
python -c 'import json, os, pytest; from pathlib import Path; d=json.loads(Path("docs/audits/digest_site_census_pq1301_2026-10-04.json").read_text()); rows=d["current_1301"]["completed_calls"]; nodes=sorted({n for r in rows for n in r["tests"]}); assert nodes, "no consumer compatibility tests"; raise SystemExit(pytest.main(["-q", "--basetemp="+os.path.join(os.environ["TMPDIR"], "consumer-pytest"), *nodes]))'
```

Observed exit 1, `KeyError: 'current_1301'`, before pytest invocation. No dynamically
referenced consumer tests ran; no compatibility evidence is inferred from this
submission. Both retained outputs were inspected (combined suite stdout and
dynamic-command traceback).

[Test evidence {"artifacts":{"stderr":{"digest":"8f41bc2bcd94613a401e4e806240cc36fcb116bd04e7134e29f73f56afbb0c80","size":670},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"96befc9d52ea6551a82074cba8fded3e271a12c770723de1f70cbadb42e84243","receipt":"da92f1f2d65f4a11eb39669dbd1b10ec1e1a5f390f1ea5969181f847a9068f72"}]
