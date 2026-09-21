"""Per-window stage-token accounting for the produced-boundary rollover.

Evidence tool, not a test. The rollover fixture was raised from a one-token
tier to a two-token tier to stop a ``tier-reservation-unavailable``, and the
reason given -- "headroom" -- does not follow from the prepaid exact-transfer
contract: one token should move owner -> mover and come back to free on
retirement, so one group should never need a second token.

So this measures instead of arguing. It drives the SAME fixture as
``test_a_rollover_longer_than_the_credit_bound_keeps_making_progress`` and
snapshots, at every transition, the tier ledger's own statement: total held on
the tier, each holder's tokens, free tokens, plus the producer's funded-window
report, its release debt and its durable charge. Run it at ``--gib 1`` and
``--gib 2``; the difference between the two names the second consumer, or
shows a per-window loss.

Prints JSON on the last line.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def _snapshot(q, label: str, *, storage=None, publication=None) -> dict:
    from prismabuild import pool as pool_mod
    import test_stage_a_produced_boundary_chain as fixture

    ledger = q.tier_ledger(fixture.TIER)
    free = len(list(ledger.free_dir.glob(f"{fixture.KIND}-*")))
    holders = {key: ledger.holder_tokens(key).get(fixture.KIND, 0)
               for key in ledger.held_keys()}
    out = {"at": label, "free": free,
           "held_total": ledger.held().get(fixture.KIND, 0),
           "holders": {key: count for key, count in holders.items() if count},
           "queue": {state: len(list((Path(q.root) / state).glob("*.json")))
                     for state in (pool_mod.READY, pool_mod.CLAIMED)
                     if (Path(q.root) / state).is_dir()}}
    if storage is not None:
        out["release_debt"] = storage.produced_release_debt()
        out["groups"] = [{"batch_id": record["batch_id"],
                          "retired": record["retired"],
                          "origin_reclaimed": record["origin_reclaimed"]}
                         for record in storage.produced_group_records()]
    if publication is not None:
        try:
            out["durable_charge"] = publication.durable_charge()
        except Exception as exc:                       # evidence, not a gate
            out["durable_charge_error"] = repr(exc)
    return out


def _census(q, label: str) -> dict:
    """Every token NAME on the tier, by holder, plus queue and lease state.

    The aggregate counts are what showed the contradiction (held 0,
    outstanding 0, free 0); naming the individual token files is what can
    say WHO holds the missing one.
    """

    from prismabuild import pool as pool_mod
    import test_stage_a_produced_boundary_chain as fixture

    ledger = q.tier_ledger(fixture.TIER)
    holders = {}
    for holder in sorted(ledger.held_dir.glob("*")):
        if holder.is_dir():
            names = sorted(path.name for path in holder.glob("*-*"))
            if names:
                holders[holder.name] = names
    rows = {}
    for state in (pool_mod.READY, pool_mod.CLAIMED, pool_mod.DONE,
                  getattr(pool_mod, "FAILED", "failed")):
        directory = Path(q.root) / str(state)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            row = pool_mod._read_json(path)
            if not isinstance(row, dict):
                continue
            rows[path.stem] = {
                "state": str(state), "attempts": row.get("attempts"),
                "owner": row.get("owner"), "claimed_at": row.get("claimed_at"),
                "heartbeat": row.get("heartbeat"),
                "lease_expires": row.get("lease_expires"),
                "resources": row.get("resources")}
    return {"at": label, "now": time.time(),
            "free_names": sorted(path.name
                                 for path in ledger.free_dir.glob("*-*")),
            "held_by_holder": holders,
            "available": ledger.available(),
            "rows": rows}


def _poll_for_free(q, *, budget_s: float = 30.0) -> dict:
    """Does the missing token come back on its own, and when?

    This is the measurement that separates a release that has not happened
    YET from one that never will: if the token appears in free shortly
    after the refusal, the release is not complete when the producer's
    retirement returns; if it never appears, it is held by whoever the
    census names.
    """

    import test_stage_a_produced_boundary_chain as fixture

    ledger = q.tier_ledger(fixture.TIER)
    start = time.perf_counter()
    while time.perf_counter() - start < budget_s:
        names = sorted(path.name for path in ledger.free_dir.glob("*-*"))
        if names:
            return {"appeared": True, "after_s": time.perf_counter() - start,
                    "free_names": names}
        time.sleep(0.05)
    return {"appeared": False, "after_s": time.perf_counter() - start}


def _cycle(args, index: int) -> dict:
    """One four-window rollover, instrumented at the failing transition."""

    import pytest
    import torch
    import test_stage_a_produced_boundary_chain as fixture
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublication, BoundaryProducedPublicationFailed)

    monkeypatch = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp(prefix=f"produced-token-audit-{index}-",
                                dir=str(Path.cwd())))
    trace: list[dict] = []
    retirements: list[dict] = []
    out = {"iteration": index, "gib": args.gib, "window_gib": args.window_gib,
           "groups": args.groups, "group_size": fixture.GROUP_SIZE,
           "tier": fixture.TIER, "kind": fixture.KIND, "tmp": str(tmp),
           "trace": trace, "retirements": retirements}

    # When does the producer's retirement RETURN, and what does it say? The
    # ordering against the physical token release is the open question, so
    # both ends are timestamped rather than assumed.
    original_retire = BoundaryProducedPublication.retire

    def timed_retire(self, batch_id, **kwargs):
        started = time.perf_counter()
        wall = time.time()
        try:
            result = original_retire(self, batch_id, **kwargs)
        except BaseException as exc:
            retirements.append({"batch_id": batch_id, "raised": repr(exc),
                                "entered_wall": wall,
                                "elapsed_s": time.perf_counter() - started})
            raise
        retirements.append({"batch_id": batch_id, "entered_wall": wall,
                            "returned_wall": time.time(),
                            "elapsed_s": time.perf_counter() - started,
                            "result": result})
        return result

    monkeypatch.setattr(BoundaryProducedPublication, "retire", timed_retire)
    q = None
    try:
        storage, publication, q, env, pb_repo = fixture._bound_owner(
            tmp, n_batches=args.groups * fixture.GROUP_SIZE,
            payload_max_bytes=1 << 22, window_gib=args.window_gib,
            gib=args.gib)
        out["owner_action_key"] = env["PRISMABUILD_ACTION_KEY"]
        from prismabuild import pool as pool_mod
        for state in (pool_mod.CLAIMED, pool_mod.READY):
            row = pool_mod._read_json(
                q.item_path(state, env["PRISMABUILD_ACTION_KEY"]))
            if isinstance(row, dict):
                out["owner_declared_demand"] = row.get("resources")
                out["owner_row_state"] = state
                break
        trace.append(_snapshot(q, "after bind + admit_window",
                               publication=publication))
        groups = [fixture._write_group(storage, count=fixture.GROUP_SIZE,
                                       first=i * fixture.GROUP_SIZE)
                  for i in range(args.groups)]
        trace.append(_snapshot(q, "after all prewrites and writes",
                               storage=storage, publication=publication))
        with fixture._fleet(q, tmp):
            fixture._strict(monkeypatch, env, pb_repo, q)
            previous = None
            for index_w, references in enumerate(groups):
                with storage.prefetch(references) as window:
                    trace.append(_snapshot(
                        q, f"window {index_w} open (staged)",
                        storage=storage, publication=publication))
                    torch.equal(storage.get(window, references[0]),
                                torch.arange(8, dtype=torch.float32)
                                + index_w * fixture.GROUP_SIZE)
                    if previous is not None:
                        for reference in previous:
                            storage.retire(reference)
                previous = references
                trace.append(_snapshot(q, f"window {index_w} closed",
                                       storage=storage,
                                       publication=publication))
        out["ok"] = True
    except BoundaryProducedPublicationFailed as exc:
        # THE transition. Census before anything else touches the ledger.
        out["ok"] = False
        out["refusal"] = {"type": type(exc).__name__, "text": str(exc),
                          "refusal": getattr(exc, "refusal", None)}
        out["census_at_refusal"] = _census(q, "at refusal")
        out["free_poll"] = _poll_for_free(q)
        out["census_after_poll"] = _census(q, "after poll")
    except BaseException as exc:
        out["ok"] = False
        out["failed_with"] = f"{type(exc).__name__}: {exc}"
        if q is not None:
            try:
                out["census_at_failure"] = _census(q, "at failure")
            except Exception:
                pass
    finally:
        monkeypatch.undo()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--stop-on-refusal", action="store_true")
    ap.add_argument("--window-gib", type=int, default=1)
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args()

    runs = []
    for index in range(args.repeat):
        result = _cycle(args, index)
        runs.append(result)
        print(f"# iteration {index}: ok={result.get('ok')} "
              f"{result.get('refusal', {}).get('refusal') if not result.get('ok') else ''}",
              flush=True, file=sys.stderr)
        if args.stop_on_refusal and not result.get("ok"):
            break
    print(json.dumps({"runs": runs}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
