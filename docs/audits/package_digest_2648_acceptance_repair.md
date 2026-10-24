# Package digest #2648 acceptance repair

## Changes and boundaries

The package route, existing 22 golden rows, CHANGELOG entry and one-row baseline
reduction from `166ea0238679e2b60c457ff56479eebf3f88d912` remain unchanged.
No `digests.py`, file-reading, Path-ordering, frontier policy or pin changes.

The `refresh_2540` census call
`hashlib:tools/generate_transition_rewrites.py::_package_digest@40:13` now records
completion and its owner route, fixture and implementation evidence. Its pending
cohort and ratchet membership are retired; the summary records 651 eligible
cohorts and 303 ratcheted scopes. Historical source identity, blob, expression,
recipe and consumer evidence remain intact, including the original scan snapshot.
The previous disposition and membership are preserved in `migration_evidence`.

The four reported selector failures reproduced before repair: **55 passed,
4 failed**, all on the absent Tessera `_evidence_smoke_record` name. The installed
producer publishes `tessera.metadata.parse_smoke_record` with
`METADATA_API_VERSION == "tessera.producer-metadata.v1"`. The bounded compatibility
adapter retains the legacy four-positional-argument API when present; only its
absence selects the exact public-v1 API with keyword-only `control` and `where`.
Missing, noncallable or incompatible capabilities refuse. Validation remains in
Tessera, including null records; a rejection or invocation error never retries
another API. This grants no admission and changes no runtime identity checks.

The v9 tests retain all grammar/refusal assertions, patch vocabularies in the
selected producer owner, and cover both signatures, null records, incompatible
versions, missing/noncallable validators and no retry after producer rejection.

Reproduction command:
`python -m pytest tests/test_select_validated_frontier.py -q --basetemp="$TMPDIR/pytest"`

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"93232aeb7a23729be5fdf72cbd41cbb82e6659363a95462f4b931a794bf94b68","size":31228}},"operation":"8c2af8c384d8e977d5f690a9be48ba345c495a8d0136c8821aa7da714fda2af4","receipt":"fd3349e990f2ed4409e0390e5d44bcd3c7129877e92a0a3c0677347c83a7d520"}]

## Final verification

All checks used the admitted CPU environment. No GPU or serving qualification is
claimed or needed for this metadata/digest repair.

```sh
python -m pytest tests/test_digest_profiles_1301_files.py tests/test_select_validated_frontier.py tests/test_prefill_frontier.py tests/test_joint_replay_frontier_607.py tests/test_campaign_digest_profiles_2174.py tests/test_stage_a_chain_resume.py tests/test_duplication_baseline.py tests/test_generate_transition_rewrites.py tests/test_tessera_lane_v9.py tests/test_source_framing_1752.py tests/test_framed_digest_owner_1301.py tests/test_tool_imports_1304.py tests/test_digest_profiles_1301.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py tests/test_doc_claims_driver_truth.py tests/test_architecture_doc.py -q --basetemp="$TMPDIR/pytest"
```

**636 passed**, no skips or failures; only existing Torch/Python deprecation
warnings. This includes all eight requested modules and the broader guards.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e25147dc78f6bbe58fd89b665976202fe1ac4af1a0fe4c74e19acc79ad927fad","size":1474}},"operation":"874bfc19c0bdffbfddc951b1e8b9d3fd244f101b4afad7940213e21114d90b08","receipt":"5d2fd1304cbf0df397a6011dc94f03cdc2dc7cf2992e1108b3a1e3c66d718679"}]

```sh
python -m pytest tests/test_tessera_lane_admission.py tests/test_tessera_export_lane.py tests/test_tessera_lane_v10.py tests/test_tessera_lane_v12.py tests/test_tessera_serving_pin.py -q --basetemp="$TMPDIR/pytest"
```

**99 passed**, no skips or failures; only the same deprecation warnings.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5d1fe34e82d236999da008a5a4e8a81f2250c6dddd4e97c732a46a6faec6b1ff","size":902}},"operation":"bde592c3f904c3cab335217d56ab22c943b6329003db902f312a9afd14b889a2","receipt":"ff236e98af950997ee7f38253f7deb504730a4e3bf180404d9e41272f6c75afb"}]

`python3 tools/duplication_inventory.py --write-baseline` reported **303** gated
primitive sites and produced no further baseline diff: the PR's single removed
raw package-digest scope remains the only reduction.

A small `python -c` check (reproduced below as a heredoc) built the frozen
sealed/executing inputs under `$TMPDIR`, invoked the standalone generator from an
unrelated working directory, and compared stdout exactly to
`test_package_digest_generator_output_matches_frozen#1`. It matched, with sealed
digest `f5f9a8a75a78ab06fde1e0c22ee0fd388e6fe41d5e983c54e641effafd0278fa`.
The same check verified historical call identity, completed route, disposition
counts, exclusive pending cohort membership and equality of the census ratchet
with the recomputed baseline.

```sh
python - <<'PY'
import ast, json, os, subprocess, sys, tempfile
from pathlib import Path
from collections import Counter
root = Path.cwd()
fixture = json.loads((root / "tests/fixtures/digest_profiles_1301_files.json").read_text())
expected = ast.literal_eval(fixture["test_package_digest_generator_output_matches_frozen#1"]["returned"])
with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
    work = Path(scratch)
    sealed, executing = work / "sealed", work / "executing"
    sealed.mkdir(); executing.mkdir()
    files = {"a.py": b"one\ntwo\nthree\nfour\n", "b.py": b"keep\n", "c.py": b"same\n"}
    for name, payload in files.items():
        (sealed / name).write_bytes(payload); (executing / name).write_bytes(payload)
    (executing / "b.py").write_bytes(b"keep\nchanged\n")
    (executing / "new.py").write_bytes(b"verifier\n")
    result = subprocess.run([sys.executable, str(root / "tools/generate_transition_rewrites.py"), "--sealed-dir", str(sealed), "--executing-dir", str(executing), "--new-file", "new.py"], cwd=work, text=True, capture_output=True, check=True)
    assert result.stdout == expected, (result.stdout, expected)
    assert "f5f9a8a75a78ab06fde1e0c22ee0fd388e6fe41d5e983c54e641effafd0278fa" in result.stderr
    print("Standalone CLI output matches frozen fixture byte-for-byte from unrelated cwd.")
    print(result.stderr.strip())
census = json.loads((root / "docs/audits/digest_site_census_pq1301_2026-10-04.json").read_text())["refresh_2540"]
id_ = "hashlib:tools/generate_transition_rewrites.py::_package_digest@40:13"
row = next(row for row in census["calls"] if row["id"] == id_)
assert row["source_blob"] == "51c0b1b4031c34939e35067c163a9500076b9cf2"
assert row["expression"] == "hashlib.sha256()"
assert row["migration_evidence"]["census_id"] == id_
assert row["owner"]["route"] == "prismaquant.digests.LengthFramedSourceSha256"
assert not row["eligible"] and not row["ratcheted"] and row["cohort"] is None
assert row["disposition"] == "migrated-by-pq2648"
assert all(id_ not in cohort["calls"] for cohort in census["contract_cohorts"])
summary = census["summary"]
assert dict(Counter(row["disposition"] for row in census["calls"])) == summary["calls_by_disposition"]
assert len(census["contract_cohorts"]) == summary["eligible_cohorts"] == 651
baseline = json.loads((root / "tests/fixtures/duplication_baseline.json").read_text())
assert set(census["ratchet_sites"]) == set(baseline["primitive_digest_sites"])
assert len(census["ratchet_sites"]) == summary["ratcheted_scopes"] == 303
eligible = {row["id"] for row in census["calls"] + census["supplemental_calls"] if row["eligible"]}
assigned = [id_ for cohort in census["contract_cohorts"] for id_ in cohort["calls"]]
assert set(assigned) == eligible and len(assigned) == len(set(assigned))
print("Census completion, historical identity, exclusive pending cohorts, disposition counts and 303-site baseline agree.")
PY
```

An initial version of this additional checker omitted `supplemental_calls` from
its eligible population and failed its final membership assertion; the generator
comparison and preceding count assertions had passed. Correcting the checker to
include both census populations passed without changing repository data.

[Test evidence {"artifacts":{"stderr":{"digest":"441baab72dc50f3cf25749bbc14b01e48ac48d0769229138fc5a79fa0a05559b","size":244},"stdout":{"digest":"633e4e905f7647a4342710e3a2367494225d249740b792574975c71bd36f6d20","size":210}},"operation":"2bb912ea4f623d5c456cd6f1bc66a00a4c9aa6367858a96c9aee3474b7cd9b65","receipt":"8e78eb857ffa2e41dca8cddca9d75c8d583d0f47c3eae63c610243ee81e7ae72"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f849c1721e77dbee5091130c29cfbf4a6b8de9e8627e1da94140e3a648a0ecd9","size":325}},"operation":"9d0f939e78f312f97e4b6067248c71e19c8b62ba9f443c62c2413e2432042a49","receipt":"0a2512511cc34250104442f5e96e7407ed523e04927c8deb1f31e143f6365be5"}]

After final documentation edits, reran the four documentation guard modules
above (**25 passed**) and executed the heredoc extracted directly from this
document. Both the CLI comparison and census/baseline assertions passed again.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fd9597b0ca53d8b16581f5bca080d04f527ae0ce559cbd379c6fda8f8db988e8","size":1072}},"operation":"c405c4cfc5b2d54d1909cd4184ecd754bb94823f4736bbf0fc53f9ed7380b23e","receipt":"54547c5f2faf281832a7d61ddb886e3c290f7bdb53ed79748b3e6c1a858c64ab"}]
