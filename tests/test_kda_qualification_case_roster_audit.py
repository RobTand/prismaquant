"""Metadata replay cannot qualify a strict subset of the numerical case roster."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments import build_kda_kernel_qualification as qualification


FIXTURES = Path(__file__).parent / "fixture-data" / "kda-production-numerics"


@pytest.mark.parametrize("case_mode", ["full", "subset", "extra"],
                         ids=["full-retained-roster", "selected-case-only", "unexpected-case"])
def test_qualification_requires_complete_numerical_case_roster(case_mode):
    raw = (FIXTURES / "numerics.json").read_bytes()
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance["sha256"]
    observed = json.loads(raw)
    assert len(observed["cases"]) == 9
    assert observed["summary"] == {"pass": True, "failures": [], "repeat_unequal": []}
    record = deepcopy(observed)
    if case_mode == "subset":
        record["cases"] = {"glm_gate_random": record["cases"]["glm_gate_random"]}
        assert record["cases"]["glm_gate_random"] == observed["cases"]["glm_gate_random"]
    elif case_mode == "extra":
        record["cases"]["unexpected_case"] = deepcopy(record["cases"]["glm_gate_random"])
    if case_mode != "full":
        with pytest.raises(qualification.QualificationRefused, match="case|roster"):
            qualification.qualify(record)
    else:
        admitted = qualification.qualify(record)
        assert set(admitted["cases"]) == set(observed["cases"])
        assert admitted["cells"] == 108
        assert admitted["failures"] == []


@pytest.mark.parametrize("path", ["fp32", "bf16"])
def test_complete_case_roster_still_requires_both_numerical_paths(path):
    record = json.loads((FIXTURES / "numerics.json").read_bytes())
    del record["cases"]["glm_gate_random"][path]
    with pytest.raises(qualification.QualificationRefused, match="without both paths"):
        qualification.qualify(record)


def test_complete_case_and_path_roster_still_requires_each_tensor():
    record = json.loads((FIXTURES / "numerics.json").read_bytes())
    del record["cases"]["glm_gate_random"]["fp32"]["kernel"]["o"]
    with pytest.raises(KeyError, match="o"):
        qualification.qualify(record)
