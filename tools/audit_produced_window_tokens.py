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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=int, default=1)
    ap.add_argument("--window-gib", type=int, default=1)
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args()

    import pytest
    import torch
    import test_stage_a_produced_boundary_chain as fixture

    monkeypatch = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp(prefix="produced-token-audit-",
                                dir=str(Path.cwd())))
    trace: list[dict] = []
    out = {"gib": args.gib, "window_gib": args.window_gib,
           "groups": args.groups, "group_size": fixture.GROUP_SIZE,
           "tier": fixture.TIER, "kind": fixture.KIND, "trace": trace}
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
                                       first=index * fixture.GROUP_SIZE)
                  for index in range(args.groups)]
        trace.append(_snapshot(q, "after all prewrites and writes",
                               storage=storage, publication=publication))
        with fixture._fleet(q, tmp):
            fixture._strict(monkeypatch, env, pb_repo, q)
            previous = None
            for index, references in enumerate(groups):
                with storage.prefetch(references) as window:
                    trace.append(_snapshot(
                        q, f"window {index} open (staged)",
                        storage=storage, publication=publication))
                    torch.equal(storage.get(window, references[0]),
                                torch.arange(8, dtype=torch.float32)
                                + index * fixture.GROUP_SIZE)
                    if previous is not None:
                        for reference in previous:
                            storage.retire(reference)
                        trace.append(_snapshot(
                            q, f"window {index} open, previous disposed",
                            storage=storage, publication=publication))
                previous = references
                trace.append(_snapshot(q, f"window {index} closed",
                                       storage=storage,
                                       publication=publication))
        out["ok"] = True
    except BaseException as exc:                        # evidence, not a gate
        out["ok"] = False
        out["failed_with"] = f"{type(exc).__name__}: {exc}"
        try:
            out["failed_at"] = _snapshot(q, "at failure")
        except Exception:
            pass
    finally:
        monkeypatch.undo()
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
