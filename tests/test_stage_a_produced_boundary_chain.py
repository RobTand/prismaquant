"""Behavioral RED for the produced-boundary chain (Astra review correction).

The adapter-import failure is not a RED. This is: the real writer publishes a
boundary entry, the strict prefetch expects to serve the tensor back, and the
unchanged guard refuses ``staged-not-serving`` -- the live 8ca8952cc651…
failure class, on pristine main. The repeat-read and restage scenarios are
declared as the passing-after acceptance surface, gated on the plan's GAP-1
resolution; they are skipped (not faked) until the adapter lands.
Characterization limits stated: the SDK acquire path is NOT exercised (the
refusal fires before acquire), and the tamper case shows a length mismatch,
not an isolated digest proof.
"""
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prismaquant.residency_map import ENV_VAR as RESIDENCY_MAP_ENV  # noqa: E402


def _storage(tmp_path):
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 1})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    return storage


def test_strict_prefetch_of_the_own_published_entry_refuses(tmp_path, monkeypatch):
    """RED (behavioral): expect the tensor, get staged-not-serving."""
    torch = pytest.importorskip("torch")
    from prismaquant.staged_lease import LeaseRefused
    from prismaquant.staged_tier_policy import (
        activate_staged_tier_policy, deactivate_staged_tier_policy_for_tests)

    storage = _storage(tmp_path)
    reference = storage.write(torch.arange(8, dtype=torch.float32),
                              batch_index=0, boundary_index=0)
    monkeypatch.setenv(RESIDENCY_MAP_ENV, str(tmp_path / "map-absent.json"))
    activate_staged_tier_policy("ram,ssd")
    try:
        with pytest.raises(LeaseRefused, match="staged-not-serving"):
            with storage.prefetch([reference]):
                pass
    finally:
        deactivate_staged_tier_policy_for_tests()


def test_without_the_policy_the_same_read_serves_the_tensor(tmp_path, monkeypatch):
    """The identity-checked direct read exists; only the tier decision fails."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(RESIDENCY_MAP_ENV, raising=False)
    storage = _storage(tmp_path)
    payload = torch.arange(8, dtype=torch.float32)
    reference = storage.write(payload, batch_index=0, boundary_index=0)
    with storage.prefetch([reference]) as window:
        assert torch.equal(storage.get(window, reference), payload)


@pytest.mark.skip(reason="passing-after surface: repeat read through the "
                         "produced batch namespace; gated on GAP-1 resolution "
                         "and the adapter -- never faked green")
def test_repeat_read_after_retire_restage_within_one_action(tmp_path):
    raise AssertionError("declared acceptance, not implemented")


def test_pinned_pb_source_provenance_is_recorded():
    import os

    pin = os.environ.get("PRISMA_STAGEA_PB_PIN", "")
    assert pin, "PRISMA_STAGEA_PB_PIN must name the pinned PB checkout"
    import prismabuild

    assert Path(prismabuild.__file__).resolve().is_relative_to(
        Path(pin).resolve()), (prismabuild.__file__, pin)
