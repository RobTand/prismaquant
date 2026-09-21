"""SSD entry windows adopt minimal per-key covers, not every lead (PQ #876).

Stage A ``f9951e60``: the input head passed all 36,439 entries, then every
model-87..90 SSD entry refused ``staged-tier-forbidden: unpublished`` in
``_range_for`` — while that entry's own forward mover (004/6feb) was
complete at 1789953014 and its phase-past head (3dd/egress 2f579) had
legitimately retired at 3060. The SSD leg of ``acquire_entry_window``
named EVERY composed-map lead as covers, and PB's ``acquire`` requires
every named mover's fragment to exist: one retired, unrelated head
poisoned every later entry. The RAM leg already resolved minimal
per-key covers through ``covers_for_keys``.

Every case here runs the real chain on tiny real files — the pinned
installed SDK, a real claim row, fragments and material written by the
real PB writers, a real composed map, the real resolver — and retirement
is the real egress end state (fragment and material unlinked, exactly
what ``stage_release.evict`` leaves behind), never a fabricated refusal.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.residency_map import (  # noqa: E402
    ENV_VAR, bind_residency_manifest, residency_map_key,
    residency_resolver, reset_residency_resolver_for_tests)
from prismaquant.staged_lease import (  # noqa: E402
    LeaseRefused, acquire_entry_window)
from prismaquant.staged_tier_policy import (  # noqa: E402
    activate_staged_tier_policy, deactivate_staged_tier_policy_for_tests)

from test_strict_reader_tier_enforcement import (  # noqa: E402
    MANIFEST, _hex64, _launch_env, _pb, _pb_publish, _pb_queue, _write_map)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_NONCE", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_SCOPE", raising=False)
    monkeypatch.delenv("PRISMABUILD_READER_HELPER_ROOT", raising=False)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    from prismaquant.staged_lease import (
        _ACQUIRE_CONTEXT, clear_injected_sdk_for_tests, set_lease_helper_root)
    set_lease_helper_root(None)
    _ACQUIRE_CONTEXT.clear()
    clear_injected_sdk_for_tests()
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    _ACQUIRE_CONTEXT.clear()
    clear_injected_sdk_for_tests()


def _two_entries(tmp_path, monkeypatch, *, retire_head=False):
    """Two real SSD movers over two real files; the map leads name both.

    Returns ``(resolver, consumer, head, current, declared, entry)``
    where ``entry`` is the composed map's real entry for the current
    mover's file, fetched through the resolver. ``retire_head`` removes
    the head's fragment and material — the end state of a legitimate
    phase-past egress.
    """
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    head = _hex64(f"mover-head-{tmp_path}")
    current = _hex64(f"mover-current-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    stage = tmp_path / "stage" / "prewarm"
    stage.mkdir(parents=True, exist_ok=True)
    payloads = {
        "head-model.safetensors": b"\x01head-bytes\x00" * 64,
        "current-model.safetensors": b"\x02current-bytes" * 64,
    }
    declared = {}
    for name, blob in payloads.items():
        source = pool / name
        source.write_bytes(blob)
        staged = stage / name
        staged.write_bytes(blob)
        declared[name] = (source, staged)
    root = tmp_path / "residency"
    head_key = residency_map_key(str(declared["head-model.safetensors"][0]), 0)
    current_key = residency_map_key(
        str(declared["current-model.safetensors"][0]), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, head, MANIFEST,
                {head_key: declared["head-model.safetensors"]})
    _pb_publish(rl, map_mod, root, stage, consumer, current, MANIFEST,
                {current_key: declared["current-model.safetensors"]})
    if retire_head:
        _retire(root, consumer, head)
    rows = {
        "h": (declared["head-model.safetensors"][0],
              declared["head-model.safetensors"][1], None),
        "c": (declared["current-model.safetensors"][0],
              declared["current-model.safetensors"][1], None),
    }
    map_path = _write_map(tmp_path, rows, leads=[head, current])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ssd")
    resolver = residency_resolver()
    source, _staged = declared["current-model.safetensors"]
    entry = resolver.staged_range(source, 0, len(payloads["current-model.safetensors"]))
    assert entry is not None, "the composed map covers the current entry"
    return resolver, consumer, head, current, root, source, entry


def _retire(root: Path, consumer: str, mover: str) -> None:
    """The legitimate egress end state: ``stage_release.evict`` unlinks the
    retired mover's fragment and material sidecar. The staged FILE
    survives (shared, content-addressed) — only the proof documents go,
    which is exactly the poisoning condition."""

    (root / consumer / f"{mover}.json").unlink()
    (root / "material" / consumer / f"{mover}.json").unlink()


def _acquire_and_read(resolver, declared, entry):
    """The reader's real sequence: window, enter, open, read, release."""
    window, key = acquire_entry_window(resolver, declared, entry)
    with window:
        fd, serving = window.open(key)
        try:
            return os.read(fd, int(entry["bytes"])), serving
        finally:
            window.close_fd(fd)


def test_a_retired_unrelated_lead_does_not_poison_a_current_entry(
        tmp_path, monkeypatch):
    """The exact Stage A f9951e60 shape, on tiny real files.

    The head's proof retired legitimately; the current entry's own mover
    is complete. Naming every lead refused the current entry
    ``unpublished``; the minimal per-key cover must acquire and serve.
    """
    (resolver, consumer, _head, _current, _root, source,
     entry) = _two_entries(tmp_path, monkeypatch, retire_head=True)

    payload, serving = _acquire_and_read(resolver, source, entry)

    assert payload == b"\x02current-bytes" * 64
    assert serving["tier_id"].startswith("prismabuild-stage:")
    # The direct window API serves without any pool leg; byte accounting
    # per tier is the shard reader's, exercised in the adjacent suites.
    assert resolver.report()["bytes_from_pool"] == 0
    # The window released exactly: no live pin remains for the consumer.
    leases = tmp_path / "residency" / "leases" / consumer
    assert not list(leases.glob("*.lease.json"))


def test_a_current_entry_also_acquires_while_the_head_is_live(
        tmp_path, monkeypatch):
    """Unretired heritage never regresses: same read, both movers live."""

    (resolver, _consumer, _head, _current, _root, source,
     entry) = _two_entries(tmp_path, monkeypatch, retire_head=False)
    payload, _serving = _acquire_and_read(resolver, source, entry)
    assert payload == b"\x02current-bytes" * 64
    assert resolver.report()["bytes_from_pool"] == 0


def test_missing_requested_coverage_still_refuses_without_pool_bytes(
        tmp_path, monkeypatch):
    """An entry no mover's material actually covers refuses fail-closed."""

    (resolver, _consumer, _head, current, root, source,
     entry) = _two_entries(tmp_path, monkeypatch, retire_head=True)
    # The map entry exists, but with the current mover's proof retired too
    # nothing at all covers the requested key: absence, never a pool read.
    _retire(root, _consumer, current)

    with pytest.raises(LeaseRefused) as refusal:
        _acquire_and_read(resolver, source, entry)
    assert refusal.value.kind == "availability"
    assert "unpublished" in str(refusal.value)
    # The refusal is the reader's too: no pool bytes on any path.
    assert resolver.report()["bytes_from_pool"] == 0


def test_contradictory_actual_proof_refuses_as_integrity(
        tmp_path, monkeypatch):
    """Two movers vouching the requested key with different bytes refuse."""

    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover_one = _hex64(f"mover-one-{tmp_path}")
    mover_two = _hex64(f"mover-two-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    stage = tmp_path / "stage" / "prewarm"
    stage.mkdir(parents=True, exist_ok=True)
    blob = b"\x03shared-name\x00" * 64
    other = b"\x04different-bytes" * 64
    source = pool / "model.safetensors"
    source.write_bytes(blob)
    staged_one = stage / "model.safetensors"
    staged_one.write_bytes(blob)
    staged_two = stage / "model-copy.safetensors"
    staged_two.write_bytes(other)
    root = tmp_path / "residency"
    key = residency_map_key(str(source), 0)
    _pb_publish(rl, map_mod, root, stage, consumer, mover_one, MANIFEST,
                {key: (source, staged_one)})
    _pb_publish(rl, map_mod, root, stage, consumer, mover_two, MANIFEST,
                {key: (source, staged_two)})
    rows = {"m": (source, staged_one, None)}
    map_path = _write_map(tmp_path, rows, leads=[mover_one, mover_two])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ssd")
    resolver = residency_resolver()
    entry = resolver.staged_range(source, 0, len(blob))
    assert entry is not None

    with pytest.raises(LeaseRefused) as refusal:
        _acquire_and_read(resolver, source, entry)
    assert refusal.value.kind == "integrity"
    assert "contradictory" in str(refusal.value) or "ownership-uncertain" in str(
        refusal.value)
    assert resolver.report()["bytes_from_pool"] == 0
