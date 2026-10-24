"""Portable accounting checks for #2989's operational evidence handoff.

The new measurement record is docs/measurements/nonrelease-lfm2.5-1.2b-inputs-2989.md
with a same-stem JSON appendix. This is a report appendix, NOT a new runtime
manifest/API. The appendix contains bindings, commands, prerequisite observations,
units, candidates, segments and exposure. Bindings are {path, sha256}; small
metadata copies can additionally name a repository-relative `retained` file.
Only the streamed_identity_cache requires such a copy, so CPU checks can replay
its existing validator without opening model weights or live fleet mounts.

These checks cannot authenticate a run, certify mount health, infer unrecorded
exposure, or prove corpus disjointness. Review must verify the operational
receipts, current producer/reader legality, census and serving-unit mapping.
Missing inputs must remain explicit refusals, never fabricated successful runs.
An honest blocked report can satisfy documentary checks but is NOT completion
of the operational objective. No GLM/LFM8B qualification follows from this record.
"""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "docs/measurements/nonrelease-lfm2.5-1.2b-inputs-2989"


def _record():
    # Append, rather than replace a suffix in the decimal model version.
    report = Path(str(STEM) + ".md")
    appendix = Path(str(STEM) + ".json")
    assert report.is_file(), "#2989 needs its new operational measurement report"
    assert appendix.is_file(), "#2989 needs its frozen, machine-readable input/population appendix"
    text = report.read_text(encoding="utf-8")
    value = json.loads(appendix.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return text, value


def _binding(value):
    assert isinstance(value, dict)
    assert isinstance(value.get("path"), str) and value["path"].strip()
    assert re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
    if "retained" in value:
        relative = Path(value["retained"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file(), f"Missing retained metadata: {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]
        return path
    return None


def test_prerequisites_commands_and_refusals_are_retained():
    text, record = _record()
    for heading in ("Scope", "Prerequisites", "Source and data identities",
                    "Frozen population", "Commands and outcomes", "Missing inputs"):
        assert f"## {heading}" in text, f"Missing measurement section: {heading}"
    assert record["model"] == "LFM2.5-1.2B"
    assert record["nonrelease"] is True
    prerequisites = record["prerequisites"]
    assert prerequisites
    for item in prerequisites:
        assert item["checked_at"] and item["subject"]
        assert item["outcome"] in {"passed", "refused", "unavailable"}
        _binding(item["evidence"])
    commands = record["commands"]
    assert commands
    for item in commands:
        assert item["command"].strip() and item["started_at"] and item["finished_at"]
        assert item["outcome"] in {"succeeded", "failed", "refused", "unavailable"}
        if item["exit_code"] is None:
            assert item["outcome"] in {"refused", "unavailable"}
            assert item["reason"], "A command not executed has no invented exit status"
        else:
            assert type(item["exit_code"]) is int
            assert (item["exit_code"] == 0) == (item["outcome"] == "succeeded")
        _binding(item["evidence"])
    assert isinstance(record["missing_inputs"], list)
    for item in record["missing_inputs"]:
        assert item["input"] and item["reason"] and item["blocked_claim"]
    # Historical STOP is not current health; no human permission gate is added.
    assert "1398" in text and "October 1" in text
    assert "#2913" in text and "#2950" in text and "#2959" in text


def test_source_and_data_bindings_include_replayable_streamed_metadata():
    _, record = _record()
    bindings = record["bindings"]
    roles = {"checkpoint", "config", "tokenizer", "calibration_tokens",
             "calibration_corpus", "heldout_corpus", "census", "checkpoint_map",
             "source_digest_cache", "streamed_identity_cache", "adoption_receipt",
             "producer_contract", "reader_contract", "prior_exposure"}
    assert roles <= bindings.keys()
    for role in roles:
        _binding(bindings[role])
    assert bindings["calibration_corpus"]["sha256"] != bindings["heldout_corpus"]["sha256"]
    tokens = record["calibration"]
    assert tokens["rows"] > 0 and tokens["sequence_length"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", tokens["token_content_sha256"])
    assert record["heldout"]["disjointness_evidence"]
    _binding(record["heldout"]["disjointness_evidence"])
    proof_path = _binding(bindings["streamed_identity_cache"])
    assert proof_path is not None, "Retain the small real streamed proof JSON for CPU replay"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    from prismaquant.streamed_model_identity import validate_streamed_model_identity
    identity = validate_streamed_model_identity(proof["identity"], where="#2989 retained proof")
    assert identity["checkpoint_weight_map"]
    assert set(identity["weight_map"].values()) <= set(identity["checkpoint_weight_map"])
    assert record["coverage"]["checkpoint_tensor_count"] == len(identity["checkpoint_weight_map"])
    assert record["coverage"]["census_unit_count"] > 0
    assert record["coverage"]["serving_unit_count"] == len(record["units"])


def test_population_freeze_uses_distinct_serving_units_and_explicit_shortfalls():
    _, record = _record()
    units = record["units"]
    assert units
    by_id = {unit["id"]: unit for unit in units}
    assert len(by_id) == len(units)
    members = []
    for unit in units:
        assert unit["id"] and unit["members"] and unit["recipe"] and unit["activation"]
        members.extend(unit["members"])
    assert len(set(members)) == len(members), "A member cannot belong to two serving units"
    assert record["candidates"]
    candidates = {row["id"]: row for row in record["candidates"]}
    assert len(candidates) == len(record["candidates"])
    for row in candidates.values():
        assert row["family"] and type(row["rate_q256"]) is int and row["structure"]
        assert row["producer_legal"] is True and row["reader_supported"] is True
        assert row["shape"] and row["contract_evidence"]
    assert record["segments"]
    for segment in record["segments"]:
        assert segment["recipe"] and segment["activation"]
        assert segment["status"] in {"supported", "unsupported", "undersized"}
        pilot, validation = segment["pilot_units"], segment["validation_units"]
        assert len(set(pilot)) == len(pilot) and len(set(validation)) == len(validation)
        assert not set(pilot) & set(validation), "Do not pool pilot and validation roles"
        selected = set(pilot) | set(validation)
        assert selected <= by_id.keys()
        for uid in selected:
            assert by_id[uid]["recipe"] == segment["recipe"]
            assert by_id[uid]["activation"] == segment["activation"]
        assert set(segment["candidate_ids"]) <= candidates.keys()
        assert segment["minimum_distinct_units"] >= 2
        if segment["status"] == "supported":
            assert pilot and validation and segment["validation_rungs"]
            assert len(selected) >= segment["minimum_distinct_units"]
            assert segment["candidate_ids"]
        else:
            assert segment["reason"], "Name unsupported/undersized populations instead of pooling"


def test_untouched_validation_is_disjoint_from_all_recorded_prior_exposure():
    _, record = _record()
    units = {unit["id"] for unit in record["units"]}
    exposure = record["exposure"]
    assert {row["unit"] for row in exposure} == units, "Unknown history is not untouched"
    assert len(exposure) == len(units)
    untouched = set()
    for row in exposure:
        assert row["state"] in {"untouched", "exposed", "unknown"}
        assert isinstance(row["events"], list)
        if row["state"] == "untouched":
            assert not row["events"]
            assert row["evidence"], "Untouched needs an affirmative ledger basis"
            untouched.add(row["unit"])
        for event in row["events"]:
            assert event["purpose"] and event["rungs"] and event["evidence"]
    pilot = {uid for segment in record["segments"] for uid in segment["pilot_units"]}
    validation = {uid for segment in record["segments"] for uid in segment["validation_units"]}
    assert not pilot & validation, "A pilot in any segment is not untouched validation elsewhere"
    assert validation <= untouched, "A new rung on an exposed unit is not a new validation unit"
