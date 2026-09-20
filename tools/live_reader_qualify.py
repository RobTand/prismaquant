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


def expect_serving_tier(*, ram_offered, allowed, lease_tier_id, serving):
    """(ok, reason) for the observed serving tier (pure).

    A valid offered RAM leg serves first when allowed; an explicitly
    SSD-only run serves exactly the lease tier it acquired. Anything
    else -- arc/pool/unknown families, RAM without a recorded offer,
    RAM served while disallowed -- fails. A missing offer or tier
    never passes.
    """

    tier = serving.get("tier_id") if isinstance(serving, dict) else None
    if not isinstance(tier, str) or not tier:
        return False, "serving names no tier"
    if tier.split(":")[0] == "ram":
        if ram_offered and "ram" in allowed:
            return True, "ram-served"
        return False, f"ram served without offer+allowance: {tier}"
    if "ssd" in allowed and isinstance(lease_tier_id, str) \
            and lease_tier_id and tier == lease_tier_id:
        return True, "ssd-served"
    return False, f"tier not served as allowed: {tier}"


def _read_pin_record(sdk, queue, pin_id):
    """(owner, pin) for one pin id, or (None, None) on clean absence.

    Error-visible reads: anything but ENOENT propagates, so unknown
    census never reads as released. Only a fully missing file is
    absence; a present-but-unreadable or unparseable pin is uncertainty.
    """

    import os as _os
    root = sdk.leases_root(queue)
    try:
        with _os.scandir(root) as entries:
            owners = sorted(entry.name for entry in entries
                            if entry.is_dir())
    except FileNotFoundError:
        return None, None
    for owner in owners:
        path = Path(root) / owner / f"{pin_id}.lease.json"
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        pin = sdk.validate_pin(json.loads(raw))
        refs = pin["refs"]
        assert isinstance(refs, dict)
        return str(pin.get("owner_action_key") or owner), pin
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
                return UNQUALIFIED
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
        try:
            lease_tier = resolver.lease_identity().get("tier_id")
        except Exception:  # noqa: BLE001 -- diagnostic only
            lease_tier = None
        result["lease_tier_id"] = lease_tier
        window = acquire_entry_window(resolver, ns.declared, entry)
        try:
            from prismabuild.pool import PoolQueue
        except ImportError as exc:
            result["finding"] = {
                "blocked": f"live queue unreachable: {exc}",
                "identity_present": identity_presence(),
            }
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        queue = PoolQueue("/mnt/shared/prismabuild-fleet/pb-queue")
        nonce = os.environ.get("PRISMABUILD_ACTION_NONCE") or ""
        scope_id = os.environ.get("PRISMABUILD_ACTION_SCOPE") or ""
        with window as entered:
            fd, serving = entered.open(residency_map_key(str(Path(ns.declared)), 0))
            result["serving"] = serving
            tier_ok, tier_reason = expect_serving_tier(
                ram_offered=bool(entry.get("ram_path")),
                allowed=allowed, lease_tier_id=lease_tier,
                serving=serving)
            result["serving_tier_check"] = tier_reason
            if not tier_ok:
                result["finding"] = f"serving tier refused: {tier_reason}"
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            # Bind THIS exact attempt through public SDK shapes: the pin
            # record for the served pin_id must name our (nonce, scope)
            # among its refs, and its selected entry must be the composed
            # path actually served. Another attempt's lease never counts.
            pin_id = serving.get("pin_id")
            if not isinstance(pin_id, str) or not pin_id:
                result["finding"] = "serving names no pin"
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            try:
                _owner, pin = _read_pin_record(sdk, queue, pin_id)
            except Exception as exc:  # noqa: BLE001 -- unknown, not empty
                result["finding"] = (
                    "pin record unreadable while open: "
                    f"{type(exc).__name__}: {exc}")
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            if pin is None:
                result["finding"] = "served pin has no record while open"
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            selected = [str(e.get("stage_path", ""))
                        for e in pin.get("entries", [])
                        if isinstance(e, dict)
                        and str(e.get("key") or "")
                        == str(serving.get("range_ref") or "")]
            composed_paths = {os.path.normpath(str(entry.get("stage_path") or ""))}
            if entry.get("ram_path"):
                composed_paths.add(os.path.normpath(str(entry["ram_path"])))
            result["selected_paths"] = selected
            if len(selected) != 1 or os.path.normpath(
                    selected[0]) not in composed_paths:
                result["finding"] = "served pin entry is not the composed path"
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            refs = pin.get("refs", {})
            mine = attempt_refs(
                refs if isinstance(refs, dict) else {}, nonce, scope_id)
            result["held_refs_for_attempt"] = mine
            owners, tainted = sdk.live_for(queue, None)
            pinned_here = owners.get(os.path.normpath(selected[0]), [])
            result["pinned_while_open"] = {selected[0]: pinned_here}
            result["tainted_while_open"] = tainted
            if not mine or pin_id not in pinned_here or tainted:
                result["finding"] = (
                    "held proof incomplete: exact-attempt refs, live pin "
                    "for the served path, and clean census all required")
                result["ok"] = False
                entered.close_fd(fd)
                print(json.dumps(result, sort_keys=True), flush=True)
                return UNQUALIFIED
            entered.close_fd(fd)
        # After release THIS exact lease must be gone: uncertainty fails,
        # while unrelated healthy readers may remain undisturbed.
        try:
            _owner_after, pin_after = _read_pin_record(sdk, queue, pin_id)
        except Exception as exc:  # noqa: BLE001
            result["finding"] = (
                f"post-release pin census uncertain: "
                f"{type(exc).__name__}: {exc}")
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        lingering = attempt_refs(
            (pin_after.get("refs", {}) if isinstance(pin_after, dict)
             else {}), nonce, scope_id)
        result["lingering_refs_for_attempt"] = lingering
        if lingering:
            result["finding"] = "exact lease survived release"
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        owners_after, tainted_after = sdk.live_for(queue, None)
        result["pins_after_release"] = owners_after
        result["tainted_after_release"] = tainted_after
        if tainted_after:
            result["finding"] = "post-release census tainted: release unproven"
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
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
