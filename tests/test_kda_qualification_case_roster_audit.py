"""Metadata replay cannot qualify a strict subset of the numerical case roster."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments import build_kda_kernel_qualification as qualification


FIXTURES = Path(__file__).parent / "fixture-data" / "kda-production-numerics"


@pytest.mark.parametrize("subset", [False, True], ids=["full-retained-roster", "selected-case-only"])
def test_qualification_requires_complete_numerical_case_roster(subset):
    raw = (FIXTURES / "numerics.json").read_bytes()
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance["sha256"]
    observed = json.loads(raw)
    assert len(observed["cases"]) == 9
    assert observed["summary"] == {"pass": True, "failures": [], "repeat_unequal": []}
    record = deepcopy(observed)
    if subset:
        record["cases"] = {"glm_gate_random": record["cases"]["glm_gate_random"]}
        assert record["cases"]["glm_gate_random"] == observed["cases"]["glm_gate_random"]
    if subset:
        with pytest.raises(qualification.QualificationRefused, match="case|roster"):
            qualification.qualify(record)
    else:
        admitted = qualification.qualify(record)
        assert set(admitted["cases"]) == set(observed["cases"])
        assert admitted["cells"] == 108
        assert admitted["failures"] == []
