"""Live injected reader qualification scenario (PQ live-reader lane).

Runs as a real admitted action (native or container) with genuinely
injected PB identity. Reads one tiny calibration artifact through the
strict staged-tier path (merged 866 consumer), proves RAM-first /
allowed-SSD behavior with no pool fallback via mountstats, proves the
exact lease is held while the window is open and released after, and
prints one result JSON on stdout. A refusal mode proves forbidden
origins fail clear with zero pool payload reads.

Usage (argv sealed by the submitter):
  live_reader_qualify.py --allowed-tiers ram,ssd --declared PATH
      --expect-sha256 HEX --expect-draw HEX --n-samples N --seqlen L
      [--refuse]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mountstats() -> dict:
    out, cur = {}, None
    try:
        lines = open("/proc/self/mountstats").read().splitlines()
    except OSError:
        return out
    for line in lines:
        if line.startswith("device "):
            parts = line.split()
            cur = parts[parts.index("on") + 1] if " on " in parts else None
        elif cur is not None and line.strip().startswith("bytes:"):
            fields = [int(x) for x in line.split()[1:]]
            out[cur] = {"client_read": fields[0], "server_read": fields[4]}
            cur = None
    return out


def _mount_of(path: str) -> str | None:
    best, real = None, os.path.realpath(path)
    try:
        mounts = open("/proc/self/mounts").read().splitlines()
    except OSError:
        return None
    for line in mounts:
        parts = line.split()
        if len(parts) > 1 and (real == parts[1] or real.startswith(parts[1] + "/")):
            if best is None or len(parts[1]) > len(best):
                best = parts[1]
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-tiers", required=True)
    parser.add_argument("--declared", required=True)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--expect-draw", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--refuse", action="store_true")
    ns = parser.parse_args()

    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.residency_map import residency_resolver, bind_residency_manifest
    from prismaquant.staged_lease import (
        LeaseRefused, acquire_entry_window, lease_helper_root,
    )
    from prismaquant.residency_map import residency_map_key

    result: dict = {
        "schema": "pq.live_reader_qual.v1",
        "mode": "refuse" if ns.refuse else "read",
        "identity": {
            "key": os.environ.get("PRISMABUILD_ACTION_KEY"),
            "nonce": os.environ.get("PRISMABUILD_ACTION_NONCE"),
            "scope": os.environ.get("PRISMABUILD_ACTION_SCOPE"),
            "helper_root": os.environ.get("PRISMABUILD_READER_HELPER_ROOT"),
            "residency_map": os.environ.get("PRISMABUILD_RESIDENCY_MAP"),
        },
    }
    try:
        result["allowed_tiers"] = sorted(activate_staged_tier_policy(ns.allowed_tiers))
        try:
            import prismabuild.reader_lease as sdk
            result["sdk_file"] = str(Path(getattr(sdk, "__file__", "")).resolve())
            result["sdk_importable"] = True
        except ImportError as exc:
            sdk = None  # type: ignore[assignment]
            result["sdk_importable"] = f"unavailable: {exc}"
        result["helper_root_auto"] = lease_helper_root()
        map_path = os.environ.get("PRISMABUILD_RESIDENCY_MAP") or ""
        bound = json.load(open(map_path)).get("manifest_sha256", "") if map_path else ""
        if bound:
            bind_residency_manifest(bound)
        result["manifest_sha256"] = bound
        before = _mountstats()
        pool_mount = _mount_of(ns.declared)
        result["pool_mount"] = pool_mount
        if ns.refuse:
            try:
                load_calibration_input(
                    ns.declared, expected_sha256=ns.expect_sha256,
                    n_samples=ns.n_samples, seqlen=ns.seqlen)
            except LeaseRefused as exc:
                result["refusal"] = f"{exc.kind}: {exc}"
            else:
                result["refusal"] = "MISSING: forbidden read was served"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            after = _mountstats()
            delta = ((after.get(pool_mount, {}).get("client_read", 0)
                      - before.get(pool_mount, {}).get("client_read", 0))
                     if pool_mount else 0)
            result["pool_client_read_delta"] = delta
            if delta != 0:
                result["refusal"] += "; POOL WAS READ"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        try:
            ids, _provenance = load_calibration_input(
                ns.declared, expected_sha256=ns.expect_sha256,
                n_samples=ns.n_samples, seqlen=ns.seqlen)
        except Exception as exc:  # noqa: BLE001 -- LeaseRefused is the finding
            from prismaquant.staged_lease import LeaseRefused as _LR
            after = _mountstats()
            delta = ((after.get(pool_mount, {}).get("client_read", 0)
                      - before.get(pool_mount, {}).get("client_read", 0))
                     if pool_mount else 0)
            result["pool_client_read_delta"] = delta
            if isinstance(exc, _LR):
                result["finding"] = (
                    f"strict read refused fail-closed ({exc.kind}: {exc}); "
                    "no injected nonce/scope/helper in live run-local env; "
                    "pins/leases/capture unexercised, pool unread")
                result["ok"] = False
                print(json.dumps(result, sort_keys=True), flush=True)
                return 0
            raise
        import torch
        draw = hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest()
        result["draw_sha256"] = draw
        result["draw_match"] = draw == ns.expect_draw
        result["shape"] = list(ids.shape)
        # Hold proof on the real composed-map entry (RAM leg when offered).
        # Without injected identity the strict SDK cannot bind, so a live
        # run records the block instead of forging a pin.
        resolver = residency_resolver()
        entry = resolver.staged_read(ns.declared, expected_sha256=ns.expect_sha256)
        if entry is None:
            result["error"] = "composed entry vanished after strict read"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1
        result["ram_offered"] = entry.get("ram_path")
        result["stage_path"] = entry.get("stage_path")
        if sdk is None or result.get("helper_root_auto") is None:
            result["finding"] = (
                "strict SDK unbindable: no injected nonce/scope/helper; "
                "live run-local actions receive key+map only "
                "(core._residency_environment); pins/leases unexercised")
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        window = acquire_entry_window(resolver, ns.declared, entry)
        from prismabuild.pool import PoolQueue
        queue = PoolQueue("/mnt/shared/prismabuild-fleet/pb-queue")
        with window as entered:
            fd, serving = entered.open(residency_map_key(str(Path(ns.declared)), 0))
            result["serving"] = serving
            owners, tainted = sdk.live_for(
                queue, {os.path.normpath(str(serving.get("stage_path", "")))})
            result["pinned_while_open"] = owners
            result["tainted_while_open"] = tainted
            entered.close_fd(fd)
        owners_after, tainted_after = sdk.live_for(queue, None)
        result["pins_after_release"] = owners_after
        result["tainted_after_release"] = tainted_after
        after = _mountstats()
        deltas = {m: (after.get(m, {}).get("client_read", 0)
                      - before.get(m, {}).get("client_read", 0)) for m in after}
        result["mount_deltas"] = deltas
        result["pool_client_read_delta"] = deltas.get(pool_mount, 0) if pool_mount else 0
        ok = (result["draw_match"] and result["pinned_while_open"]
              and not result["tainted_while_open"]
              and result["pool_client_read_delta"] == 0)
        result["ok"] = bool(ok)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 -- scenario reports, never hides
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
