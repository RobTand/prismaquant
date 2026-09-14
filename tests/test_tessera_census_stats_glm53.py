"""GLM-5.3 acceptance: the bf16 probe expands onto the r1024 census roster.

Reads the real probe and census cost table by their recorded SHA-256.  It
skips only where those files do not exist (hosted CI); on the fleet a skip is
a failure to run, and its record must show the test passed.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path

import pytest

from prismaquant.allocator_candidates import _has_measured_output_mse
from prismaquant.model_profiles.structure import load_structure_spec
from prismaquant.schemas import validate_probe_payload
from prismaquant.tessera_census_stats import (
    _FLOAT32_EPS, META_KEY, expand_probe_onto_census,
)
from prismaquant.tessera_expert_projection import PROJECTION_KEY, carried_units

PROBE = Path("/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/artifacts/probe.pkl")
PROBE_SHA256 = "d206363dd585176496e8c645f1e00e0b3086723be35c638ac3939812f43c5e9b"
COSTS = Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
             "activation-runtime-allocation-20260911/extension-r1024-02/workspace/merged/cost.pkl")
COSTS_SHA256 = "cd21541019cb670876fcd3501cba8c58e3473044090e54f16726ad26b0eb27e1"


def _load(path: Path, digest: str):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, path
    return pickle.loads(raw)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not (PROBE.is_file() and COSTS.is_file()),
                    reason="GLM-5.3 probe and census are fleet-local inputs")
def test_glm53_probe_expands_onto_the_census_roster():
    probe = _load(PROBE, PROBE_SHA256)
    cost = _load(COSTS, COSTS_SHA256)
    out = expand_probe_onto_census(probe, cost, structure=load_structure_spec("glm5_next"))
    stats = out["stats"]
    census = set(cost["costs"])
    _source, units, stack_of = carried_units(cost["provenance"][PROJECTION_KEY])

    assert len(stats) == 36_423
    assert set(stats) - census == set() and census - set(stats) == set()
    assert sum(row["n_params"] for row in stats.values()) == 305_915_756_544
    assert not [n for n, row in stats.items() if any(k.startswith("_packed_") for k in row)]
    assert out["meta"][META_KEY]["routed_units"] == 36_288
    assert out["meta"][META_KEY]["dense_units"] == 135
    assert out["meta"][META_KEY]["packed_rows"] == 84
    validate_probe_payload(out)

    for parent_name, parent in probe["stats"].items():
        if "_packed_param" not in parent:
            continue
        stack = parent["_packed_experts_module"]
        roles = ({"gate_proj", "up_proj"} if parent["_packed_param"] == "gate_up_proj"
                 else {"down_proj"})
        children = [n for n, s in stack_of.items()
                    if s == stack and units[n]["projection"] in roles]
        assert len(children) == (576 if len(roles) == 2 else 288), parent_name
        total = math.fsum(stats[n]["h_trace"] for n in children)
        bound = parent["num_experts"] * _FLOAT32_EPS * abs(parent["h_trace"])
        assert abs(total - parent["h_trace"]) <= bound, parent_name

    unmeasured = [(name, fmt) for name in units for fmt, cell in cost["costs"][name].items()
                  if not _has_measured_output_mse(stats[name], cell)]
    assert unmeasured == []
