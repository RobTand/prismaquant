"""Native placement obligation of the shared-tag rule (PQ #831), with no PrismaBuild matcher.

The dispatcher shipped ``--tag sparky --tag sparklina``; the placement conjunction admitted neither Spark. Native
placement is one declared property or one host (``FluxResources.placement``), so the dispatcher's tags must be
exactly one native property, and the historical two-host pair must be inexpressible rather than silently
unplaceable. This is a placement-vocabulary proof, not a fleet placement proof.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from dispatch_joint_quanta import CONSUMER_TAGS  # noqa: E402
from prismaquant.flux_jobs import FluxResources  # noqa: E402


def _resources(placement):
    return FluxResources(cores=1, memory_gib=1, gpus=0, walltime_s=60, placement=placement)


def test_dispatcher_tags_are_one_native_property():
    assert CONSUMER_TAGS == ("gb10",)
    assert _resources(CONSUMER_TAGS[0]).placement == "gb10"


@pytest.mark.parametrize("placement", ["host:sparky", "host:sparklina", "host:dl380g10", "x86_64", "aarch64"])
def test_a_class_or_a_single_host_is_a_native_placement(placement):
    assert _resources(placement).placement == placement


@pytest.mark.parametrize("placement", ["sparky,sparklina", "sparky sparklina", "host:sparky,host:sparklina",
                                       ("sparky", "sparklina"), "sparky", "gpu"])
def test_the_historical_host_pair_and_unknown_tags_are_not_native_placements(placement):
    with pytest.raises(ValueError, match="unsupported placement"):
        _resources(placement)
