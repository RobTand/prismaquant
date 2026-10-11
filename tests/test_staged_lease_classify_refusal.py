"""The public refusal classifier keeps its three outcomes (PQ #2616)."""
from prismaquant.staged_lease import _classify as legacy
from prismaquant.staged_lease import classify_refusal


def test_public_name_pins_availability_integrity_and_unknown():
    assert classify_refusal("unpublished") == "availability"
    assert classify_refusal("source-coverage-gap") == "integrity"
    assert classify_refusal("not-a-real-refusal") == "integrity"


def test_private_name_stays_an_alias_for_one_release():
    assert legacy is classify_refusal
