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


#: Exit status: 0 qualified full pass; 1 infrastructure error; 2 the
#: action executed but is NOT qualified (findings preserved in JSON).
#: Only --refuse mode may exit 0 on an intended typed refusal.
UNQUALIFIED = 2

IDENTITY_NAMES = ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                  "PRISMABUILD_ACTION_SCOPE",
                  "PRISMABUILD_READER_HELPER_ROOT")


def identity_presence(env=None) -> dict:
    """Which identity names the environment actually carries (pure)."""

    source = os.environ if env is None else env
    get = source.get if hasattr(source, "get") else {}.get
    return {name: bool(get(name)) for name in IDENTITY_NAMES}


def eval_pool_delta(before, after, mount):
    """(observed, delta) for pool client reads (pure).

    Unknown evidence never proves zero reads: observed is False unless
    both snapshots exist and the mount's client_read field is present
    in both.
    """

    if not isinstance(before, dict) or not isinstance(after, dict) \
            or not mount:
        return False, None
    first = before.get(mount, {}).get("client_read") \
        if isinstance(before.get(mount), dict) else None
    second = after.get(mount, {}).get("client_read") \
        if isinstance(after.get(mount), dict) else None
    if not isinstance(first, int) or not isinstance(second, int):
        return False, None
    return True, second - first


def expect_serving_tier(*, ram_offered, ram_allowed, ssd_allowed, serving):
    """(ok, reason) for the observed serving tier (pure).

    Valid offered RAM is selected first; SSD serves only when explicitly
    allowed and no RAM was offered. A missing offer or tier never passes.
    """

    tier = serving.get("tier_id") if isinstance(serving, dict) else None
    if not isinstance(tier, str) or not tier:
        return False, "serving names no tier"
    family = tier.split(":")[0]
    if ram_offered and ram_allowed:
        if family == "ram":
            return True, "ram-served"
        return False, f"ram offered but served {tier}"
    if ssd_allowed and not ram_offered:
        if family == "ram":
            return False, "ram served without a recorded offer"
        return True, "ssd-served"
    return False, "no appropriate tier allowed for this offer"


def _find_pin_refs(sdk, queue, pin_id):
    """(owner, refs) for one pin id, or (None, None) when fully released.

    Scans owner directories (pin ids are globally unique); reads through
    the SDK's own pin validation. Unreadable pins raise -- unknown is
    never an empty census here.
    """

    import os as _os
    root = sdk.leases_root(queue)
    for owner_dir in sorted(_os.scandir(root), key=lambda e: e.name):
        if not owner_dir.is_dir():
            continue
        path = Path(owner_dir.path) / f"{pin_id}.lease.json"
        if not path.is_file():
            continue
        pin = sdk.validate_pin(json.loads(path.read_text()))
        refs = pin["refs"]
        assert isinstance(refs, dict)
        return str(pin.get("owner_action_key") or owner_dir.name), refs
    return None, None


def attempt_refs(refs, nonce, scope_id):
    """Ref ids in a pin refs map bound to exactly this attempt (pure)."""

    return [ref_id for ref_id, ref in refs.items()
            if isinstance(ref, dict)
            and isinstance(ref.get("attempt"), dict)
            and str(ref["attempt"].get("nonce") or "") == nonce
            and str(ref["attempt"].get("scope_id") or "") == scope_id]


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
        try:
            from prismaquant.residency_map import residency_resolver as _rr
            _probe = _rr().staged_read(ns.declared, expected_sha256=(
                ns.expect_sha256 if not ns.refuse else None))
            if _probe is not None:
                result["composed_entry"] = {
                    "stage_path": _probe.get("stage_path"),
                    "ram_path": _probe.get("ram_path"),
                    "bytes": _probe.get("bytes")}
        except Exception as exc:  # noqa: BLE001 -- diagnostic only
            result["composed_entry"] = f"unreadable: {type(exc).__name__}: {exc}"
        if ns.refuse:
            try:
                load_calibration_input(
                    ns.declared, expected_sha256=ns.expect_sha256,
                    n_samples=ns.n_samples, seqlen=ns.seqlen)
            except Exception as exc:  # noqa: BLE001 -- refusal is the finding
                from prismaquant.staged_lease import LeaseRefused as _LR
                from prismaquant.staged_tier_policy import (
                    TierPolicyRefused as _TR)
                after = _mountstats()
                observed, delta = eval_pool_delta(before, after, pool_mount)
                result["pool_reads_observed"] = observed
                result["pool_client_read_delta"] = delta
                if not isinstance(exc, (_LR, _TR)):
                    raise
                result["refusal"] = f"{type(exc).__name__}: {exc}"
            else:
                result["refusal"] = "MISSING: forbidden read was served"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            if not result.get("pool_reads_observed", False):
                result["refusal"] += "; pool reads UNOBSERVED, not proven zero"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            if result.get("pool_client_read_delta", 1) != 0:
                result["refusal"] += "; POOL WAS READ"
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        try:
            ids, _provenance = load_calibration_input(
                ns.declared, expected_sha256=ns.expect_sha256,
                n_samples=ns.n_samples, seqlen=ns.seqlen)
        except Exception as exc:  # noqa: BLE001 -- refusal is evidence
            from prismaquant.staged_lease import LeaseRefused as _LR
            from prismaquant.staged_tier_policy import (
                TierPolicyRefused as _TR)
            after = _mountstats()
            observed, delta = eval_pool_delta(before, after, pool_mount)
            result["pool_reads_observed"] = observed
            result["pool_client_read_delta"] = delta
            if not isinstance(exc, (_LR, _TR)):
                raise
            # Report the actual refusal and the actual identity present;
            # never a hardcoded cause.
            result["finding"] = {
                "refusal": f"{type(exc).__name__}: {exc}",
                "identity_present": identity_presence(),
            }
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
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
            result["finding"] = "composed entry vanished after strict read"
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        result["ram_offered"] = entry.get("ram_path")
        result["stage_path"] = entry.get("stage_path")
        if sdk is None or result.get("helper_root_auto") is None:
            result["finding"] = {
                "blocked": "strict SDK unbindable in this run",
                "identity_present": identity_presence(),
            }
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        allowed = set(ns.allowed_tiers.split(","))
        window = acquire_entry_window(resolver, ns.declared, entry)
        from prismabuild.pool import PoolQueue
        queue = PoolQueue("/mnt/shared/prismabuild-fleet/pb-queue")
        nonce = os.environ.get("PRISMABUILD_ACTION_NONCE") or ""
        scope_id = os.environ.get("PRISMABUILD_ACTION_SCOPE") or ""
        with window as entered:
            fd, serving = entered.open(residency_map_key(str(Path(ns.declared)), 0))
            result["serving"] = serving
            tier_ok, tier_reason = expect_serving_tier(
                ram_offered=bool(entry.get("ram_path")),
                ram_allowed="ram" in allowed, ssd_allowed="ssd" in allowed,
                serving=serving)
            result["serving_tier_check"] = tier_reason
            if not tier_ok:
                result["finding"] = f"serving tier refused: {tier_reason}"
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            # Bind THIS exact attempt's refs: the pin must hold at least
            # one ref for our (nonce, scope); another attempt's lease
            # never counts.
            _owner, refs = _find_pin_refs(
                sdk, queue, str(serving.get("pin_id") or ""))
            mine = attempt_refs(refs or {}, nonce, scope_id)
            result["held_refs_for_attempt"] = mine
            owners, tainted = sdk.live_for(
                queue, {os.path.normpath(str(serving.get("stage_path", "")))})
            result["pinned_while_open"] = owners
            result["tainted_while_open"] = tainted
            if not mine or not owners or tainted:
                result["finding"] = (
                    "held proof incomplete: exact-attempt refs, live pin, "
                    "and clean census all required")
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            entered.close_fd(fd)
        # After release THIS exact lease must be gone; unrelated readers
        # may remain and are not disturbed.
        _owner_after, refs_after = _find_pin_refs(
            sdk, queue, str(serving.get("pin_id") or ""))
        lingering = attempt_refs(refs_after or {}, nonce, scope_id)
        result["lingering_refs_for_attempt"] = lingering
        if lingering:
            result["finding"] = "exact lease survived release"
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        owners_after, tainted_after = sdk.live_for(queue, None)
        result["pins_after_release"] = owners_after
        result["tainted_after_release"] = tainted_after
        after = _mountstats()
        observed, delta = eval_pool_delta(before, after, pool_mount)
        result["pool_reads_observed"] = observed
        result["pool_client_read_delta"] = delta
        result["mount_deltas"] = {
            m: ((after.get(m, {}).get("client_read", 0)
                 - before.get(m, {}).get("client_read", 0))
                if observed else None) for m in after} if observed else None
        ok = (result["draw_match"] and observed and delta == 0)
        result["ok"] = bool(ok)
        if not ok:
            result["finding"] = "draw, observed pool accounting, or both failed"
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if ok else UNQUALIFIED
    except Exception as exc:  # noqa: BLE001 -- scenario reports, never hides
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
