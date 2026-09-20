"""Calibration input through the strict staged tiers (PQ #865).

The real calibration reader served GPU-consumed token draws straight off
the pool path while the staged-tier policy was active: the executable
manifest declares the calibration file, but ``load_calibration_input``
never asked the stage. These tests run the real chain on a real tiny
safetensors calibration artifact — real resolver, real composed map,
real PB writers, the accepted pinned SDK, real claim rows — never a
test-only calibration format or writer:

- unmapped calibration under active policy refuses with zero pool bytes
  (the RED that fails on the pool-reading main);
- published stage serves pinned with exact release, byte-equal ids and
  both identity hash conventions unchanged versus the offline load;
- RAM serves first with real RAM-mover covers; stale RAM falls honestly
  to allowed SSD; RAM-only policy refuses before payload;
- corrupt staged bytes fail clear (integrity), wrong pins refuse, pins
  release exactly and outputs stay valid after release.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.residency_map import (
    ENV_VAR, bind_residency_manifest,
    residency_map_key, residency_resolver, reset_residency_resolver_for_tests,
)
from prismaquant.staged_tier_policy import (
    TierPolicyRefused, activate_staged_tier_policy,
    deactivate_staged_tier_policy_for_tests,
)
from prismaquant.staged_lease import LeaseRefused, set_lease_helper_root

from test_strict_reader_tier_enforcement import (
    _announce,
    _hex64,
    _launch_env,
    _leased_fixture,
    _pb,
    _pb_publish,
    _pb_publish_ram,
    _pb_queue,
    _pins_live,
    _promote_ram,
    _stage_root,
    _stage_whole,
    _strict,
    _write_map,
    EPOCH,
    MANIFEST,
    STALE_EPOCH,
)


@pytest.fixture(autouse=True)
def _forget_state(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_NONCE", raising=False)
    monkeypatch.delenv("PRISMABUILD_ACTION_SCOPE", raising=False)
    monkeypatch.delenv("PRISMABUILD_READER_HELPER_ROOT", raising=False)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    from prismaquant.staged_lease import (
        _ACQUIRE_CONTEXT, clear_injected_sdk_for_tests)
    _ACQUIRE_CONTEXT.clear()
    clear_injected_sdk_for_tests()
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    _ACQUIRE_CONTEXT.clear()
    clear_injected_sdk_for_tests()


def _artifact(tmp_path, *, n_samples=4, seqlen=8):
    ids = torch.arange(n_samples * seqlen).reshape(n_samples, seqlen)
    provenance = {"source": "fixture", "fit_tokens": ids.numel(),
                  "nsamples": n_samples, "seqlen": seqlen,
                  "fit_ids_sha256": hashlib.sha256(
                      ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    path = tmp_path / "pool" / "tokens.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), ids


def _legacy_load(path, sha):
    from prismaquant.calibration_data import load_calibration_input
    return load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)


def _guard_pool(monkeypatch, pool_path):
    """Fail on any pool payload open; the staged path must not need it."""
    opened = []
    real_open = os.open
    target = os.fspath(pool_path)

    def counting(given, *args, **kwargs):
        if os.fspath(given) == target:
            opened.append(os.fspath(given))
        return real_open(given, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting)
    import safetensors
    real_safe_open = safetensors.safe_open
    calls = []

    def guarded(*args, **kwargs):
        calls.append(args[0] if args else None)
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", guarded)
    return opened, calls


def _stage_calibration(tmp_path, monkeypatch, path):
    """Full honest stack for one whole-file calibration entry (SSD leg)."""
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    return staged, _leased_fixture(
        tmp_path, monkeypatch, {'cal': (path, staged, None)})


# -- RED: the pool-reading hole ----------------------------------------------

def test_strict_unmapped_calibration_refuses_without_pool_bytes(tmp_path, monkeypatch):
    """Active policy, calibration absent from the readset: refuse, no pool.

    Fails on the pool-reading main (the load succeeds off the pool path);
    passes once the reader honors the staged tiers.
    """
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    other = tmp_path / "pool" / "unrelated.safetensors"
    other.write_bytes(b"unrelated")
    root = _stage_root(tmp_path)
    _strict(monkeypatch, _write_map(tmp_path, {'o': (other, _stage_whole(root, other), None)}))
    activate_staged_tier_policy("ram,ssd")
    opened, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert residency_resolver().report()['bytes_from_pool'] == 0


# -- positive: published stage serves pinned -----------------------------------

def test_strict_calibration_stage_serves_pinned_with_exact_release(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    want_ids, want_receipt = _legacy_load(path, sha)
    assert torch.equal(want_ids, ids)
    _staged, (resolver, consumer, _mover) = _stage_calibration(tmp_path, monkeypatch, path)
    opened, safe_opens = _guard_pool(monkeypatch, path)
    got_ids, receipt = load_calibration_input(
        path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert torch.equal(got_ids, ids)
    assert receipt == want_receipt
    assert receipt["artifact_sha256"] == sha
    assert receipt["calibration_sha256"] == hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    assert receipt["provenance"]["fit_ids_sha256"] != receipt["calibration_sha256"]
    # Owned storage: exact bytes, contiguous, valid after lease release.
    assert got_ids.is_contiguous()
    assert got_ids.untyped_storage().nbytes() == got_ids.numel() * 8
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == path.stat().st_size
    assert any(row.get('pin_id') for row in report['serving_tiers'])
    assert _pins_live(tmp_path, consumer) == []


def test_strict_calibration_ram_serves_first_pinned(tmp_path, monkeypatch):
    """A live tmpfs copy with real RAM-mover covers serves; SSD never opens."""
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'cal': staged})
    _announce(tmp_path)
    rows = {'cal': (path, staged, ram['cal'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover_ssd = _hex64(f"mover-ssd-{tmp_path}")
    mover_ram = _hex64(f"mover-ram-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    root_dir = tmp_path / 'residency'
    key = residency_map_key(str(path), 0)
    _pb_publish(rl, map_mod, root_dir, root, consumer, mover_ssd, MANIFEST,
                {key: (path, staged)})
    _pb_publish_ram(rl, map_mod, root_dir, ram_root, consumer, mover_ram,
                    MANIFEST, {key: (path, ram['cal'])}, EPOCH)
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, leads=[mover_ssd])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    opened, safe_opens = _guard_pool(monkeypatch, path)
    got_ids, receipt = load_calibration_input(
        path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert torch.equal(got_ids, ids)
    assert receipt["artifact_sha256"] == sha
    report = resolver.report()
    assert report['bytes_from_ram'] == path.stat().st_size
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == 0
    row = report['serving_tiers'][-1]
    assert row['serving_tier'] == 'ram' and row['pin_id'] and row['range_ref']
    assert _pins_live(tmp_path, consumer) == []


def test_strict_calibration_stale_ram_falls_to_allowed_stage(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    ram_root, ram = _promote_ram(tmp_path, {'cal': staged})
    _announce(tmp_path, epoch=STALE_EPOCH)
    rows = {'cal': (path, staged, ram['cal'])}
    rl, pool_mod, map_mod = _pb()
    consumer = _hex64(f"consumer-{tmp_path}")
    mover = _hex64(f"mover-{tmp_path}")
    _pb_queue(tmp_path, pool_mod, consumer)
    _pb_publish(rl, map_mod, tmp_path / 'residency', root, consumer, mover,
                MANIFEST, {residency_map_key(str(path), 0): (path, staged)})
    map_path = _write_map(tmp_path, rows, ram_root=ram_root, epoch=EPOCH,
                          leads=[mover])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _launch_env(monkeypatch, consumer)
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")
    resolver = residency_resolver()
    opened, safe_opens = _guard_pool(monkeypatch, path)
    got_ids, _receipt = load_calibration_input(
        path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert torch.equal(got_ids, ids)
    report = resolver.report()
    assert report['bytes_from_stage'] > 0 and report['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


# -- refusals: forbidden tier, corruption, wrong pin -----------------------------

def test_strict_calibration_ram_only_refuses_before_payload(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'cal': (path, staged, None)}), tiers="ram")
    opened, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused, match="ssd-not-allowed"):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_calibration_corrupt_stage_fails_clear(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver, consumer, _mover = _leased_fixture(
        tmp_path, monkeypatch, {'cal': (path, staged, None)})
    blob = staged.read_bytes()
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    opened, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(LeaseRefused) as excinfo:
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert excinfo.value.kind == "integrity"
    assert opened == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0
    assert _pins_live(tmp_path, consumer) == []


def test_strict_calibration_wrong_pin_refuses_without_pool_bytes(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, _sha, _ids = _artifact(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _strict(monkeypatch, _write_map(
        tmp_path, {'cal': (path, staged, None)}))
    activate_staged_tier_policy("ram,ssd")
    opened, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused):
        load_calibration_input(path, expected_sha256="0" * 64, n_samples=4, seqlen=8)
    assert opened == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0
