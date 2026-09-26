"""Live injected reader qualification scenario (PQ live-reader lane).

Runs as a real admitted action (native or container) with genuinely
injected PB identity. Reads one tiny calibration artifact through the
strict staged-tier path (merged 866 consumer), proves RAM-first /
allowed-SSD behavior with no pool fallback via mountstats, proves the
exact lease is held while the window is open and released after, and
prints one result JSON on stdout. A refusal mode proves forbidden
origins fail clear with zero pool payload reads.

``--functional-only`` is the dev read mode: it qualifies the functional
reader path -- strict staged read, byte/draw checks, allowed serving
tier, exact live lease held and cleanly released -- and reports
pool-origin I/O qualification separately instead of failing closed on
missing mount-wide counters (the live cold-read result: everything
passed, ``pool_reads_observed: false`` alone forced exit 2; that
``false`` came from this module's own mountstats parser, which tested
``" on "`` against the split tokens and so never found a mount, PQ #1299). Unknown
telemetry stays unknown in every mode: the result always carries
``pool_reads_observed``/``pool_client_read_delta`` as observed, plus
``functional_success``, ``io_qualification`` and ``validation_scope`` --
mount-wide counters observe the mount, never this payload, so absent
counts can never read as proof of zero origin reads and nonzero counts
never read as proof of reads. Default mode (no flag) still requires
observed zero pool reads for success.

Usage (argv sealed by the submitter):
  live_reader_qualify.py --allowed-tiers ram,ssd --declared PATH
      --expect-sha256 HEX --expect-draw HEX --n-samples N --seqlen L
      [--functional-only] [--refuse]
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
    from prismaquant.io_spans import nfs_read_bytes

    try:
        reads = nfs_read_bytes()
    except OSError:
        return {}
    return {mount: {"client_read": client, "server_read": server}
            for mount, (client, server) in reads.items()}


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
#: Only --refuse mode may exit 0 on an intended typed refusal, and only
#: --functional-only may exit 0 while pool I/O is unqualified (the
#: result names the validation scope and the I/O verdict either way).
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


#: Tier families the probe accepts as staged service. ``ram`` serves the
#: live tmpfs copy; ``prismabuild-stage`` (the SDK's ``STAGE_POOL_PREFIX``)
#: serves the stage-pool copy. Any other family (arc/pool/unknown) never
#: passes, even when it textually equals the lease tier: the lease tier
#: proves which acquire this serving claims, the family proves it is a
#: tier the readers lane actually serves.
_STAGE_FAMILIES = ("ram", "prismabuild-stage")


def expect_serving_tier(*, ram_offered, allowed, lease_tier_id, serving,
                        ram_fallback_recorded=False):
    """(ok, reason) for the observed serving tier (pure).

    RAM serves first when offered and allowed. An SSD-family tier serves
    only when explicitly allowed, exactly equal to the lease tier the
    window acquired, and -- when RAM was offered -- only with recorded
    typed RAM-availability fallback evidence from the resolver (a bare
    ``ram_path`` string never proves availability, and a RAM integrity
    refusal never falls back: ``acquire_entry_window`` propagates those
    with no alternate adoption). Anything else -- arc/pool/unknown
    families even under a matching lease id, RAM without offer plus
    allowance, RAM served while disallowed, SSD served over a validly
    offered RAM leg -- fails. A missing offer or tier never passes.
    """

    tier = serving.get("tier_id") if isinstance(serving, dict) else None
    if not isinstance(tier, str) or not tier:
        return False, "serving names no tier"
    family = tier.split(":")[0]
    if family == "ram":
        if ram_offered and "ram" in allowed:
            return True, "ram-served"
        return False, f"ram served without offer+allowance: {tier}"
    if family not in _STAGE_FAMILIES:
        return False, f"unsupported serving family: {tier}"
    if ("ssd" not in allowed or not isinstance(lease_tier_id, str)
            or not lease_tier_id or tier != lease_tier_id):
        return False, f"tier not served as allowed: {tier}"
    if ram_offered and not ram_fallback_recorded:
        return False, (f"ssd served while ram offered with no recorded "
                       f"availability fallback (ram-first violated): {tier}")
    return True, "ssd-served"


def ram_availability_fallback_recorded(resolver, declared) -> bool:
    """Whether the resolver recorded a typed RAM-availability fallback.

    Reads the resolver's own ``ram_fallbacks`` evidence (written only on
    the availability leg of ``acquire_entry_window``; integrity refusals
    propagate without recording). A bare ``ram_path`` string is never
    consulted here, so map-offered-but-unproven RAM cannot pass.
    """

    try:
        fallbacks = resolver.report().get("ram_fallbacks", [])
    except Exception:  # noqa: BLE001 -- no evidence is no fallback
        return False
    if not isinstance(fallbacks, list):
        return False
    want = os.path.normpath(str(declared))
    return any(isinstance(row, dict) and os.path.normpath(
        str(row.get("path") or "")) == want for row in fallbacks)


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
    parser.add_argument("--functional-only", action="store_true",
                        help="dev read mode: qualify the functional reader "
                             "path (strict read, draw, allowed serving tier, "
                             "exact lease held+released) and report pool I/O "
                             "qualification separately; unknown counters "
                             "stay unknown")
    ns = parser.parse_args()
    if ns.functional_only and ns.refuse:
        parser.error(
            "--functional-only has no refuse-mode meaning: it qualifies "
            "the read path only")

    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.residency_map import residency_resolver, bind_residency_manifest
    from prismaquant.staged_lease import (
        LeaseRefused, acquire_entry_window, lease_helper_root, resolve_context,
    )

    result: dict = {
        "schema": "pq.live_reader_qual.v1",
        "mode": "refuse" if ns.refuse else (
            "functional-read" if ns.functional_only else "read"),
        "functional_only": bool(ns.functional_only),
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
        # No direct ``prismabuild.reader_lease`` import here: in the scoped
        # PQ venv that preimport resolves the installed package, and the
        # sealed-tree coherence check then refuses the divergence the tool
        # itself created. The strict reader below resolves the sealed SDK
        # first; diagnostic access follows through that same runtime.
        sdk = None
        result["sdk_importable"] = "sealed SDK resolves at first strict read"
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
        # The SDK this run actually served: the module the strict read
        # above resolved (sealed tree in production, injected install in
        # tests) -- never a fresh environment import that could resolve
        # elsewhere. Absent means the read path never bound, so the hold
        # proof is blocked rather than forged.
        sdk = sys.modules.get("prismabuild.reader_lease")
        if sdk is not None:
            result["sdk_file"] = str(
                Path(getattr(sdk, "__file__", "")).resolve())
            result["sdk_importable"] = True
        else:
            result["sdk_importable"] = "strict read bound no SDK module"
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
        # The public contract returns (window, key): unpack both and open
        # the returned key. Storing the whole tuple as the window cannot
        # enter.
        window, map_key = acquire_entry_window(resolver, ns.declared, entry)
        try:
            # Ordered after the strict read on purpose: the sealed tree's
            # ``src`` sits on ``sys.path`` ahead of the install only once
            # the SDK above resolved it, so this names the sealed pool.
            from prismabuild.pool import PoolQueue
        except ImportError as exc:
            result["finding"] = {
                "blocked": f"live queue unreachable: {exc}",
                "identity_present": identity_presence(),
            }
            result["ok"] = False
            print(json.dumps(result, sort_keys=True), flush=True)
            return UNQUALIFIED
        # The hold/release census reads the queue the claim actually lives
        # on: the SDK's own injected context names it (queue_root), the same
        # identity the window above acquired under. A hardcoded fleet path
        # read the wrong (live) queue from any other queue root.
        _sdk_mod, lease_ctx = resolve_context()
        result["census_queue_root"] = str(lease_ctx["queue_root"])
        queue = PoolQueue(str(lease_ctx["queue_root"]))
        nonce = os.environ.get("PRISMABUILD_ACTION_NONCE") or ""
        scope_id = os.environ.get("PRISMABUILD_ACTION_SCOPE") or ""
        with window as entered:
            fd, serving = entered.open(map_key)
            result["serving"] = serving
            tier_ok, tier_reason = expect_serving_tier(
                ram_offered=bool(entry.get("ram_path")),
                allowed=allowed, lease_tier_id=lease_tier,
                serving=serving,
                ram_fallback_recorded=ram_availability_fallback_recorded(
                    resolver, ns.declared))
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
        # Mount-wide counters observe the mount, never this payload: absent
        # counters are unknown (never proof of zero reads), and a nonzero
        # count is an observation, not attribution. Both modes report this
        # honestly; only the requested verdict differs.
        functional_success = bool(result["draw_match"])
        io_qualified = bool(observed and delta == 0)
        result["functional_success"] = functional_success
        result["io_qualification"] = (
            "qualified" if io_qualified else
            "nonzero-observed" if observed else "unobserved")
        result["validation_scope"] = {
            "staged_read_draw_tier_lease_release": functional_success,
            "pool_origin_reads_proven_zero": io_qualified,
        }
        if ns.functional_only:
            result["ok"] = functional_success
            if not functional_success:
                result["finding"] = "functional read failed: draw mismatch"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0 if functional_success else UNQUALIFIED
        ok = functional_success and io_qualified
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
