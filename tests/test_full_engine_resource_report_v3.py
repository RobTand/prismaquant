"""Real producer-path CPU fixtures prove protocol/refusals, not GPU admission."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import full_engine_resource_report as consumer
from prismaquant.measured_runtime_prices import RuntimePriceError


FIXTURES = Path(__file__).parent / "fixtures/tessera_full_engine_tp2"


def report(rank=0):
    return json.loads((FIXTURES / f"report-rank{rank}.json").read_text())


def seal(root, value, name="report.json"):
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("rank", [0, 1])
def test_real_producer_v3_is_readable_but_does_not_certify_placement(tmp_path, rank):
    document = report(rank)
    reference = seal(tmp_path, document)
    parsed = consumer.read_full_engine_resource_report(reference, root=tmp_path)
    assert parsed == document
    verdict = consumer.consume_full_engine_resource_report(reference, root=tmp_path)
    assert verdict.schema == "tessera.full_engine_resource_report.v3"
    assert verdict.recomputed_placement_obligation_bytes is None
    assert verdict.refusals
    assert any("fixture provenance" in message for message in verdict.refusals)
    assert any("placement" in message for message in verdict.refusals)


@pytest.mark.parametrize("mutation", ["schema", "topology", "world", "duplicate_rank", "own_device",
                                     "peer_device", "host", "digest", "ref_digest", "missing_world",
                                     "obligation", "placement", "unknown_observation"])
def test_v3_malformed_identity_or_placement_claim_refuses(tmp_path, mutation):
    document = report()
    roster = document["observations"]["rank_world"]
    if mutation == "schema":
        document["schema"] = "tessera.full_engine_resource_report.v999"
    elif mutation == "topology":
        document["execution"]["topology"] = "tp1"
    elif mutation == "world":
        roster["world_size"] = 1
    elif mutation == "duplicate_rank":
        roster["ranks"][1]["rank"] = 0
    elif mutation == "own_device":
        roster["ranks"][0]["device_uuid"] = "GPU-other"
    elif mutation == "peer_device":
        roster["ranks"][1]["device_uuid"] = roster["ranks"][0]["device_uuid"]
    elif mutation == "host":
        roster["ranks"][0]["host"]["ip"] = "192.0.2.1"
    elif mutation == "digest":
        roster["run_identity"]["configuration_sha256"] = "0" * 64
    elif mutation == "ref_digest":
        roster["raw_plan"]["sha256"] = "not-a-digest"
    elif mutation == "missing_world":
        del document["observations"]["rank_world"]
    elif mutation == "obligation":
        document["derived"]["placement_obligation"] = consumer.PLACEMENT_OBLIGATION
    elif mutation == "placement":
        document["derived"]["certifies_placement"] = True
    else:
        document["observations"]["unregistered_v3_observation"] = {}
    with pytest.raises(RuntimePriceError):
        consumer.read_full_engine_resource_report(seal(tmp_path, document), root=tmp_path)


def test_each_rank_uses_its_own_capture_ceiling_not_the_mean(tmp_path):
    documents = [report(rank) for rank in range(2)]
    references = [seal(tmp_path, document, f"rank-{rank}.json")
                  for rank, document in enumerate(documents)]
    peaks = [document["observations"]["torch_observed_live_peak_bytes"] for document in documents]
    assert min(peaks) > 0
    ceilings = [peaks[0] - 1, peaks[1] + 2]
    assert sum(ceilings) > sum(peaks)  # The average would pass; rank zero must not.
    verdicts = consumer.consume_full_engine_rank_reports(
        references, root=tmp_path, device_ceilings=ceilings)
    assert any("own device ceiling" in message for message in verdicts[0].refusals)
    assert not any("own device ceiling" in message for message in verdicts[1].refusals)
    assert all(verdict.recomputed_placement_obligation_bytes is None for verdict in verdicts)


def test_rank_report_swap_cannot_borrow_a_peers_ceiling(tmp_path):
    references = [seal(tmp_path, report(rank), f"rank-{rank}.json") for rank in range(2)]
    with pytest.raises(RuntimePriceError, match="rank"):
        consumer.consume_full_engine_rank_reports(
            references[::-1], root=tmp_path, device_ceilings=[10**9, 10**9])


def test_whole_peak_is_a_checked_observation_never_a_fixed_term(tmp_path):
    document = report()
    document["derived"]["off_step_torch_live_peak_bytes"] = 10**9
    verdict = consumer.consume_full_engine_resource_report(seal(tmp_path, document), root=tmp_path)
    assert any("off_step_torch_live_peak_bytes" in message for message in verdict.disagreements)
    assert verdict.recomputed_placement_obligation_bytes is None
    assert 10**9 not in verdict.recomputed_terms.values()


def test_optional_false_placement_marker_is_explicit_and_non_admitting(tmp_path):
    document = copy.deepcopy(report())
    document["derived"]["certifies_placement"] = False
    verdict = consumer.consume_full_engine_resource_report(seal(tmp_path, document), root=tmp_path)
    assert verdict.recomputed_placement_obligation_bytes is None
    assert verdict.refusals


@pytest.mark.parametrize("flag", [True, 0, 1, None, "false"])
def test_only_literal_false_certifies_placement_is_permitted(tmp_path, flag):
    document = report()
    document["derived"]["certifies_placement"] = flag
    with pytest.raises(RuntimePriceError, match="certifies_placement"):
        consumer.read_full_engine_resource_report(seal(tmp_path, document), root=tmp_path)


@pytest.mark.parametrize("ceiling", [True, -1, 1.5, None])
def test_rank_ceiling_is_a_nonnegative_integer(tmp_path, ceiling):
    refs = [seal(tmp_path, report(rank), f"rank{rank}.json") for rank in range(2)]
    with pytest.raises(RuntimePriceError):
        consumer.consume_full_engine_rank_reports(refs, root=tmp_path,
                                                  device_ceilings=[ceiling, 10**9])


def test_a_forged_claim_cannot_hide_a_rank_over_its_ceiling(tmp_path):
    documents = [report(rank) for rank in range(2)]
    documents[0]["observations"]["torch_observed_live_peak_bytes"] = 0
    refs = [seal(tmp_path, doc, f"rank{rank}.json") for rank, doc in enumerate(documents)]
    verdicts = consumer.consume_full_engine_rank_reports(refs, root=tmp_path,
                                                         device_ceilings=[1, 10**9])
    assert any("own device ceiling" in message for message in verdicts[0].refusals)
    assert any("observed live peak" in message for message in verdicts[0].disagreements)


def whole_peak_observations(begin=5, end=11):
    observations = report()["observations"]
    observations["step_coverage"] = {"state": "complete", "declared": 1, "executed": 1,
                                      "reason": None, "scope": "synthetic step"}
    observations["step_intervals"] = [{"step_id": "step", "begin_index": begin, "end_index": end}]
    observations["owner_views"] = {
        "schema": "tessera.full_engine_ownership_observation.v1",
        "views": {"views": [{"allocation_id": row["allocation_id"], "class": "fixed"}
                            for row in observations["torch_allocations"]]},
    }
    return observations


@pytest.mark.parametrize("begin,end,expected", [(5, 11, 1024), (3, 5, 1536), (2, 10, 512)])
def test_whole_off_step_peak_is_same_instant_and_includes_residents(begin, end, expected):
    assert consumer._whole_off_step_peak(whole_peak_observations(begin, end)) == expected


def test_whole_peak_excludes_observer_only_with_complete_views():
    observations = whole_peak_observations()
    observations["owner_views"]["views"]["views"][1]["class"] = "observer"
    assert consumer._whole_off_step_peak(observations) == 512
    observations["owner_views"]["views"]["views"].pop()
    assert consumer._whole_off_step_peak(observations) is None


def test_whole_peak_requires_complete_step_coverage():
    observations = whole_peak_observations()
    observations["step_coverage"]["state"] = "partial"
    assert consumer._whole_off_step_peak(observations) is None


def test_world_rosters_cannot_mix_different_configurations(tmp_path):
    documents = [report(rank) for rank in range(2)]
    for identity in (documents[1]["identity"]["run"], documents[1]["partition"]["identity"],
                     documents[1]["observations"]["rank_world"]["run_identity"]):
        identity["configuration_sha256"] = "0" * 64
    refs = [seal(tmp_path, doc, f"rank{rank}.json") for rank, doc in enumerate(documents)]
    with pytest.raises(RuntimePriceError, match="world roster"):
        consumer.consume_full_engine_rank_reports(refs, root=tmp_path,
                                                  device_ceilings=[10**9, 10**9])
