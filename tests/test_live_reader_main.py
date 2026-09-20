"""live_reader_qualify.main end-to-end on real calibration + SDK records.

Calls ``tools.live_reader_qualify.main`` (argv + environment, stdout
JSON) instead of only its helpers: a tiny real safetensors calibration
artifact, fragments/material/map composed through the real installed
SDK writers into isolated tmp dirs, and the accepted test-only SDK
injection. Covers the refuse-mode pass with proven zero pool reads,
pool-read exit 1, uncertain counters, forbidden-pool and wrong-digest
typed refusals, the no-claim block (pins never forged), the public
(window, key) unpack contract, and fallback evidence on a real
resolver. Only platform mount observations are simulated; no live
queue mutation, no fake SDK, no CLAIMED rows. Run via published pbtest
at -10 in the scoped PQ environment.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.live_reader_qualify as probe  # noqa: E402

TIER = "prismabuild-stage:fixture"
CONSUMER = "c" * 64
MOVER = "e" * 64
MANIFEST = "a" * 64
N_SAMPLES = 4
SEQLEN = 8
MOUNT = "/mnt/shared"


@pytest.fixture(autouse=True)
def _isolated_probe_env(monkeypatch):
    """Each main run starts with no identity, map, policy, or injection."""

    from prismaquant.staged_lease import (
        clear_injected_sdk_for_tests,
        set_lease_helper_root,
    )
    from prismaquant.residency_map import reset_residency_resolver_for_tests
    from prismaquant.staged_tier_policy import (
        deactivate_staged_tier_policy_for_tests,
    )

    def scrub() -> None:
        for name in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                     "PRISMABUILD_ACTION_SCOPE",
                     "PRISMABUILD_READER_HELPER_ROOT",
                     "PRISMABUILD_RESIDENCY_MAP"):
            monkeypatch.delenv(name, raising=False)
        set_lease_helper_root(None)
        clear_injected_sdk_for_tests()
        reset_residency_resolver_for_tests()
        deactivate_staged_tier_policy_for_tests()

    scrub()
    yield
    scrub()


def _artifact(tmp_path: Path):
    """A real tiny calibration artifact: bytes, sha, and draw hex."""

    import torch
    from safetensors.torch import save_file

    ids = torch.arange(N_SAMPLES * SEQLEN, dtype=torch.int64).reshape(
        N_SAMPLES, SEQLEN)
    draw = hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest()
    provenance = {"fit_ids_sha256": draw, "fit_tokens": ids.numel(),
                  "nsamples": N_SAMPLES, "seqlen": SEQLEN}
    path = tmp_path / "calib.safetensors"
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), draw


def _sdk():
    from prismaquant.staged_lease import inject_installed_sdk_for_tests
    return inject_installed_sdk_for_tests()


def _composed(tmp_path: Path, artifact: Path, digest: str):
    """Fragment/material/map through the real SDK writers (isolated)."""

    import prismabuild.pool as pool_mod
    import prismabuild.residency_map as sdk_map

    rl = _sdk()
    queue = pool_mod.PoolQueue(tmp_path / "queue")
    queue.ensure_layout()
    root = queue.root / "residency"
    stage = tmp_path / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / "calib.safetensors"
    staged.write_bytes(artifact.read_bytes())
    declared = tmp_path / "declared.safetensors"
    declared.write_bytes(artifact.read_bytes())
    key = sdk_map.residency_map_key(str(declared), 0)
    fragment = {
        "schema": sdk_map.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": CONSUMER, "mover_action_key": MOVER,
        "tier_id": TIER, "stage_root": str(stage),
        "manifest_sha256": MANIFEST,
        "entries": {key: {"stage_path": str(staged),
                          "bytes": staged.stat().st_size,
                          "sha256": digest, "offset": 0}}}
    sdk_map.write_fragment(root, fragment)
    rl.write_material(
        root, consumer_action_key=CONSUMER, mover_action_key=MOVER,
        tier_id=TIER, stage_root=str(stage), manifest_sha256=MANIFEST,
        generation=rl.mint_generation(),
        entries={key: {"stage_path": str(staged),
                       "bytes": staged.stat().st_size, "sha256": digest,
                       "file_id": rl.stat_identity(str(staged))}})
    composed = sdk_map.compose([fragment])
    map_path = tmp_path / "residency.map.json"
    sdk_map.write_map(map_path, composed)
    return queue, root, declared, key


def _stats(mount_reads):
    return {MOUNT: {"client_read": mount_reads, "server_read": 0}}


def _run_main(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["live_reader_qualify.py", *argv])
    code = probe.main()
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def _read_argv(declared, digest, draw, *, tiers="ram,ssd", extra=()):
    return ["--allowed-tiers", tiers, "--declared", str(declared),
            "--expect-sha256", digest, "--expect-draw", draw,
            "--n-samples", str(N_SAMPLES), "--seqlen", str(SEQLEN), *extra]


def test_refuse_mode_pass_proves_zero_pool_reads(
        tmp_path, monkeypatch, capsys) -> None:
    """Forbidden origin refuses typed with observed zero pool payload reads."""

    declared = tmp_path / "pool-only.safetensors"
    declared.write_bytes(b"\x07" * 64)
    monkeypatch.setattr(probe, "_mountstats", lambda: _stats(41))
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys,
        _read_argv(declared, "b" * 64, "c" * 64, extra=["--refuse"]))
    assert code == 0
    assert "staged-tier-forbidden" in result["refusal"]
    assert result["pool_reads_observed"] is True
    assert result["pool_client_read_delta"] == 0


def test_refuse_mode_pool_read_is_exit_1(tmp_path, monkeypatch, capsys) -> None:
    """A nonzero pool delta on the forbidden path fails exit 1, never 0."""

    declared = tmp_path / "pool-only.safetensors"
    declared.write_bytes(b"\x07" * 64)

    calls = {"n": 0}

    def stats():
        calls["n"] += 1
        return _stats(41 + calls["n"] * 7)

    monkeypatch.setattr(probe, "_mountstats", stats)
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys,
        _read_argv(declared, "b" * 64, "c" * 64, extra=["--refuse"]))
    assert code == 1
    assert "POOL WAS READ" in result["refusal"]
    assert result["pool_client_read_delta"] > 0


def test_refuse_mode_unobserved_counters_unqualified(
        tmp_path, monkeypatch, capsys) -> None:
    """Missing counter evidence never proves zero reads: exit 2."""

    declared = tmp_path / "pool-only.safetensors"
    declared.write_bytes(b"\x07" * 64)
    monkeypatch.setattr(probe, "_mountstats", lambda: {})
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys,
        _read_argv(declared, "b" * 64, "c" * 64, extra=["--refuse"]))
    assert code == probe.UNQUALIFIED == 2
    assert "UNOBSERVED" in result["refusal"]


def test_read_mode_forbidden_pool_typed_refusal(
        tmp_path, monkeypatch, capsys) -> None:
    """Read mode with no staged map fails clear with the actual refusal."""

    declared = tmp_path / "pool-only.safetensors"
    declared.write_bytes(b"\x07" * 64)
    monkeypatch.setattr(probe, "_mountstats", lambda: _stats(41))
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys, _read_argv(declared, "b" * 64, "c" * 64))
    assert code == probe.UNQUALIFIED == 2
    assert result["ok"] is False
    assert "readset-not-staged" in result["finding"]["refusal"]
    assert result["finding"]["identity_present"] == {
        "PRISMABUILD_ACTION_KEY": False, "PRISMABUILD_ACTION_NONCE": False,
        "PRISMABUILD_ACTION_SCOPE": False,
        "PRISMABUILD_READER_HELPER_ROOT": False}
    assert result["pool_client_read_delta"] == 0


def test_read_mode_wrong_digest_refuses(tmp_path, monkeypatch, capsys) -> None:
    """A digest the map does not vouch is not this read's entry: exit 2."""

    artifact, digest, draw = _artifact(tmp_path)
    _queue, _root, declared, _key = _composed(tmp_path, artifact, digest)
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP",
                       str(tmp_path / "residency.map.json"))
    monkeypatch.setattr(probe, "_mountstats", lambda: _stats(41))
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys, _read_argv(declared, "0" * 64, draw))
    assert code == probe.UNQUALIFIED == 2
    assert result["ok"] is False
    assert "readset-not-staged" in result["finding"]["refusal"]


def test_read_mode_no_claim_blocks_without_forged_pin(
        tmp_path, monkeypatch, capsys) -> None:
    """Composed bytes but no live claim: the strict read blocks typed.

    The pin/open/release path is never forged around the missing claim;
    the finding names the actual context refusal and no pin record
    appears under the isolated queue.
    """

    artifact, digest, draw = _artifact(tmp_path)
    queue, _root, declared, _key = _composed(tmp_path, artifact, digest)
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP",
                       str(tmp_path / "residency.map.json"))
    monkeypatch.setattr(probe, "_mountstats", lambda: _stats(41))
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)
    code, result = _run_main(
        monkeypatch, capsys, _read_argv(declared, digest, draw))
    assert code == probe.UNQUALIFIED == 2
    assert result["ok"] is False
    assert "lease-context-unavailable" in result["finding"]["refusal"]
    leases = queue.root / "residency" / "leases"
    leftovers = list(leases.rglob("*.lease.json")) if leases.is_dir() else []
    assert leftovers == []


def test_acquire_returns_window_and_key_tuple(tmp_path, monkeypatch) -> None:
    """The public contract main unpacks: (LeaseWindow, key), entered once.

    Entering the whole tuple (the R5 defect shape) raises TypeError, so
    only the unpacked window may serve as the context manager; main's
    source unpacks the public return and opens the returned key.
    """

    from prismaquant.residency_map import residency_resolver
    from prismaquant.residency_map import bind_residency_manifest
    from prismaquant.staged_lease import LeaseWindow, acquire_entry_window
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    artifact, digest, _draw = _artifact(tmp_path)
    _queue, _root, declared, key = _composed(tmp_path, artifact, digest)
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP",
                       str(tmp_path / "residency.map.json"))
    activate_staged_tier_policy("ram,ssd")
    bind_residency_manifest(MANIFEST)
    resolver = residency_resolver()
    assert resolver is not None
    entry = resolver.staged_read(str(declared), expected_sha256=digest)
    assert entry is not None
    produced = acquire_entry_window(resolver, str(declared), entry)
    assert isinstance(produced, tuple) and len(produced) == 2
    window, map_key = produced
    assert isinstance(window, LeaseWindow)
    assert map_key == key
    with pytest.raises(TypeError):
        with produced:  # noqa: F841 -- the exact R5 defect shape
            pass
    source = Path(probe.__file__).read_text()
    tree = ast.parse(source)

    def _calls_acquire(call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Name):
            return func.id == "acquire_entry_window"
        return getattr(func, "attr", "") == "acquire_entry_window"

    unpacked = any(
        any(isinstance(target, ast.Tuple) for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _calls_acquire(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign))
    assert unpacked, "main must unpack (window, key)"
    assert "entered.open(map_key)" in source


def test_fallback_evidence_reads_real_resolver(tmp_path, monkeypatch) -> None:
    """SSD over offered RAM passes only with the resolver's own record.

    The record comes from the existing ``record_ram_fallback`` evidence
    API on a real composed resolver -- a bare ``ram_path`` string never
    suffices, and an unrecorded leg violates RAM-first.
    """

    import torch
    from prismaquant.residency_map import residency_resolver
    from prismaquant.residency_map import bind_residency_manifest
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    _ = torch.zeros(1).sum().item()  # scoped-venv torch import check
    artifact, digest, _draw = _artifact(tmp_path)
    _queue, _root, declared, _key = _composed(tmp_path, artifact, digest)
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP",
                       str(tmp_path / "residency.map.json"))
    activate_staged_tier_policy("ram,ssd")
    bind_residency_manifest(MANIFEST)
    resolver = residency_resolver()
    assert resolver is not None
    assert probe.ram_availability_fallback_recorded(
        resolver, str(declared)) is False
    stage = {"tier_id": TIER, "epoch": "", "pin_id": "p", "range_ref": "k"}
    assert probe.expect_serving_tier(
        ram_offered=True, allowed={"ram", "ssd"}, lease_tier_id=TIER,
        serving=stage)[0] is False
    resolver.record_ram_fallback(
        str(declared), "staged-tier-forbidden: ram-covers-unresolved")
    assert probe.ram_availability_fallback_recorded(
        resolver, str(declared)) is True
    assert probe.ram_availability_fallback_recorded(
        resolver, str(tmp_path / "elsewhere.bin")) is False
    assert probe.expect_serving_tier(
        ram_offered=True, allowed={"ram", "ssd"}, lease_tier_id=TIER,
        serving=stage, ram_fallback_recorded=True) == (True, "ssd-served")
