"""The Tessera lane's ship-record evidence: ``route.census`` and ``route.trace``.

Decoupling step 6, part 3 (PQ #1553). The lane declares both slots in
``lane_specs/tessera.json`` ``gates[]``. This module holds the code that fills
them and the code that replays them, and ``tessera_lane`` registers both with
core:

- ``shipcard_slot_verifiers`` gives ``shipcard.verify`` the two replays.
- ``shipcard_cli_commands`` adds ``fill-route-census`` and
  ``fill-route-trace`` to ``python -m prismaquant.shipcard_cli``.

``shipcard`` and ``shipcard_cli`` import no lane module. The code moved here
unchanged from those two modules. Each replay takes
``(slot, record, *, card, model_dir)`` and owns its slot's whole verdict, so
``verify`` routes these slots past its generic ``passed`` check.

Module scope imports only the standard library and the two core modules. The
census, trace and contract modules load when a function needs them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from prismaquant.shipcard import (
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
)
from prismaquant.shipcard_cli import EXIT_NOT_VERIFIED

#: The priced-vs-served route census verdict (PrismaQuant #136).  Tessera's
#: plugin stamps which decoder ran on every route record, and a serve whose
#: extension did not build keeps serving on a named substitute -- so a KL
#: priced as TESSERA_NVFP4 but measured on torch_materialize_stock is a
#: different number wearing the right name.  The slot carries the priced
#: routes, the served records and the known substitute set, and `verify`
#: replays the comparison from those carried values rather than trusting the
#: carried boolean.
ROUTE_CENSUS_SLOT = "route.census"

#: Principle 14's serve-side leg (RobTand/prismaquant#575): the activation
#: contracts a live serve dispatched, read from every rank's
#: ``TESSERA_ROUTE_TRACE`` file, against the contracts the artifact's own
#: ``config.json`` prices on its platform.  A separate slot from
#: `route.census`: the census is a dedicated offline run judged per cell; the
#: trace is what the served process itself counted.  The record carries the
#: traces and the config text, and `verify` replays the comparison against the
#: current packaged contract (`tessera_route_trace_gate`).
ROUTE_TRACE_SLOT = "route.trace"


# ---------------------------------------------------------------------------
# The priced-vs-served route census receipt (#136)
# ---------------------------------------------------------------------------
def make_route_census_record(
    *,
    tool: str,
    model_sha: str | None,
    priced_routes: Sequence[str] = (),
    route_records: Any,
    substitute_decoders: Sequence[str] = (),
    serve_fingerprint: str | None = None,
    git_commit: str | None = None,
    binding: Mapping[str, str] | None = None,
    build: Mapping[str, Any] | None = None,
    model_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Close `route.census` from the priced routes and the served records.

    The comparison runs HERE, at fill time, and its verdict is what the
    record carries -- but `verify` replays it from the carried values
    (`verify_route_census_record`), so filling a `passed=true` over
    substitute-decoder records still refuses at publication.
    """
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError, check_route_receipt, check_scoped_route_receipt,
        current_table_refuses_flat_census,
    )

    if isinstance(route_records, Mapping):
        from copy import deepcopy
        verdict = check_scoped_route_receipt(route_records, binding, build=build, model_dir=model_dir)
        if priced_routes and sorted(set(priced_routes)) != verdict["served_routes"]:
            raise TesseraRouteReceiptError("caller priced routes differ from independently bound price projections")
        return make_record(slot=ROUTE_CENSUS_SLOT, tool=tool, passed=True, model_sha=model_sha,
            metrics={"n_records": verdict["n_records"], "n_served_routes": len(verdict["served_routes"])},
            detail=verdict["detail"], serve_fingerprint=serve_fingerprint, git_commit=git_commit,
            extra={"route_census": deepcopy(route_records), "census_binding": deepcopy(binding),
                   "scoped_verdict": verdict, "served_routes": verdict["served_routes"],
                   "served_decoders": verdict["served_decoders"]})
    if binding is not None or (build or {}).get("tessera_serving_scope") is not None:
        raise TesseraRouteReceiptError("scoped artifact cannot fill route.census from an unbound legacy flat list")
    # The rule `verify` replays, applied here first (#214): a producer that
    # says passed=True where the verifier on the same box then refuses is two
    # homes for one decision.
    try:
        refusal = current_table_refuses_flat_census()
    except (ValueError, OSError) as exc:
        raise TesseraRouteReceiptError(f"cannot inspect current census contract: {exc}") from exc
    if refusal is not None:
        raise TesseraRouteReceiptError(refusal)

    verdict = check_route_receipt(
        priced_routes=list(priced_routes),
        route_records=[dict(row) for row in route_records],
        substitute_decoders=list(substitute_decoders),
    )
    return make_record(
        slot=ROUTE_CENSUS_SLOT,
        tool=tool,
        passed=bool(verdict["passed"]),
        model_sha=model_sha,
        metrics={
            "n_records": verdict["n_records"],
            "n_served_routes": len(verdict["served_routes"]),
            "n_substitute_hits": len(verdict["substitute_hits"]),
        },
        detail=verdict["detail"],
        serve_fingerprint=serve_fingerprint,
        git_commit=git_commit,
        extra={
            "priced_routes": verdict["priced_routes"],
            "route_records": [
                {"route": row["route"], "decoder": row["decoder"],
                 "count": row["count"]}
                for row in parse_route_records_for_card(route_records)
            ],
            "served_routes": verdict["served_routes"],
            "served_decoders": verdict["served_decoders"],
            "substitute_decoders": sorted(set(substitute_decoders)),
        },
    )


def parse_route_records_for_card(
    route_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The carried rows in canonical form (parse once, replay forever)."""
    from prismaquant.tessera_route_receipt import parse_route_records

    return parse_route_records(list(route_records),
                               where="route.census route_records")


def verify_route_census_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    card: Mapping[str, Any] | None = None,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Replay the priced-vs-served comparison from the carried values."""
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError,
        check_route_receipt,
        check_scoped_route_receipt,
    )

    problems: list[str] = []
    build = (card or {}).get("build") or {}
    if "route_census" in record or "census_binding" in record or "scoped_verdict" in record:
        try:
            replay = check_scoped_route_receipt(record.get("route_census"), record.get("census_binding"),
                                               build=build, model_dir=model_dir)
        except TesseraRouteReceiptError as exc:
            return [f"{slot}: scoped census REFUSED: {exc}"]
        if record.get("passed") is not True or record.get("scoped_verdict") != replay:
            problems.append(f"{slot}: carried scoped verdict differs from current-contract replay")
        for key in ("served_routes", "served_decoders"):
            if record.get(key) != replay[key]:
                problems.append(f"{slot}: carried {key} differs from scoped replay")
        return problems
    if build.get("tessera_serving_scope") is not None:
        return [f"{slot}: scoped artifact carries an unbound legacy census; retain the producer's v2 receipt"]
    # The same rule fill applies (`make_route_census_record`, #214), from its
    # one home; live since the pin moved to a table of the scoped schema (v8,
    # first packaged at b8b1cb38), dead before it when the pinned table was
    # v4.  With no
    # runtime installed there is no current table to refuse on and the flat
    # rows keep their historical v4 comparison.
    from prismaquant.tessera_route_receipt import current_table_refuses_flat_census
    try:
        refusal = current_table_refuses_flat_census()
    except (ValueError, OSError) as exc:
        return [f"{slot}: cannot inspect current census contract: {exc}"]
    if refusal is not None:
        return [f"{slot}: {refusal}"]
    priced = record.get("priced_routes")
    rows = record.get("route_records")
    substitutes = record.get("substitute_decoders")
    if not isinstance(priced, list) or not priced:
        problems.append(
            f"{slot}: record carries no priced_routes; the receipt must say "
            "which routes the artifact priced")
    if not isinstance(rows, list) or not rows:
        problems.append(
            f"{slot}: record carries no route_records; an absent census is "
            "not a clean bill")
    if not isinstance(substitutes, list) or not substitutes:
        problems.append(
            f"{slot}: record carries no substitute_decoders; a gate that "
            "knows no substitute detects nothing")
    if problems:
        return problems
    try:
        verdict = check_route_receipt(
            priced_routes=priced,
            route_records=rows,
            substitute_decoders=substitutes,
        )
    except TesseraRouteReceiptError as exc:
        return [f"{slot}: carried census is malformed: {exc}"]
    if not verdict["passed"]:
        problems.append(f"{slot}: FAILED — {verdict['detail']}")
    if record.get("passed") is not True and verdict["passed"]:
        problems.append(
            f"{slot}: record carries passed={record.get('passed')!r} but "
            "its own records replay to agreement; re-fill the slot")
    if record.get("passed") is True and not verdict["passed"]:
        problems.append(
            f"{slot}: record claims passed=true but its own records replay "
            f"to refusal: {verdict['detail']}")
    for key, carried_key in (("served_routes", "served_routes"),
                             ("served_decoders", "served_decoders")):
        if record.get(carried_key) != verdict[key]:
            problems.append(
                f"{slot}: carried {carried_key} "
                f"{record.get(carried_key)!r} disagrees with the replay "
                f"{verdict[key]!r}")
    return problems


# ---------------------------------------------------------------------------
# The served route trace (#575)
# ---------------------------------------------------------------------------
def make_route_trace_record(
    *,
    tool: str,
    model_sha: str | None,
    traces: Sequence[tuple[str, Any]],
    expected_ranks: int,
    config_json: str,
    build: Mapping[str, Any] | None = None,
    platform: str | None = None,
    serve_fingerprint: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Close `route.trace` from every rank's served trace (#575).

    Raises ``RouteTraceNotVerified`` when no usable observation exists and
    ``TesseraRouteTraceError`` when the observation disagrees with the price:
    neither produces a record, so the slot stays unfilled and publication
    refuses.  Only an agreeing verdict is written, and `verify` replays it.
    """
    from copy import deepcopy

    from prismaquant import tessera_route_trace_gate as gate

    executes, formats = gate.load_trace_contract()
    resolved = gate.resolve_platform(build, platform)
    try:
        config = json.loads(config_json)
    except ValueError as exc:
        raise gate.TesseraRouteTraceError(f"config.json is not JSON: {exc}") from exc
    verdict = gate.compare_route_traces(
        list(traces), expected_ranks=expected_ranks, config=config,
        platform=resolved, executes_by_platform=executes, formats=formats)
    if verdict["status"] == gate.NOT_VERIFIED:
        raise gate.RouteTraceNotVerified(verdict["detail"])
    if verdict["status"] != gate.AGREE:
        raise gate.TesseraRouteTraceError(verdict["detail"])
    carried = []
    for label, payload in traces:
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        carried.append({"rank": label, "trace": deepcopy(payload)})
    return make_record(
        slot=ROUTE_TRACE_SLOT,
        tool=tool,
        passed=True,
        model_sha=model_sha,
        metrics={
            "n_ranks": len(carried),
            "n_priced_modules": sum(verdict["priced"].values()),
            "n_served_modules": sum((verdict["served"] or {}).values()),
        },
        detail=verdict["detail"],
        serve_fingerprint=serve_fingerprint,
        git_commit=git_commit,
        extra={
            "route_traces": carried,
            "expected_ranks": expected_ranks,
            "platform": resolved,
            "config_json": config_json,
            "trace_verdict": verdict,
        },
    )


def verify_route_trace_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    card: Mapping[str, Any] | None = None,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Replay the served-vs-priced contract histogram from the carried traces."""
    from prismaquant import tessera_route_trace_gate as gate

    traces = record.get("route_traces")
    config_json = record.get("config_json")
    expected = record.get("expected_ranks")
    problems: list[str] = []
    if not isinstance(traces, list) or not traces or not all(
            isinstance(row, Mapping) and isinstance(row.get("rank"), str)
            for row in traces):
        problems.append(
            f"{slot}: record carries no route_traces; an absent observation is "
            "not a clean bill")
    if not isinstance(config_json, str) or not config_json:
        problems.append(f"{slot}: record carries no config_json to price against")
    if type(expected) is not int:
        problems.append(f"{slot}: record carries no expected_ranks")
    if problems:
        return problems
    if model_dir is not None:
        try:
            on_disk = (Path(model_dir) / "config.json").read_bytes().decode("utf-8")
        except (OSError, ValueError) as exc:
            return [f"{slot}: cannot read the artifact's config.json: {exc}"]
        if on_disk != config_json:
            problems.append(
                f"{slot}: carried config_json differs from the artifact's "
                "config.json; the traces were compared against another price")
    try:
        executes, formats = gate.load_trace_contract()
        platform = gate.resolve_platform((card or {}).get("build"), record.get("platform"))
        verdict = gate.compare_route_traces(
            [(row["rank"], row.get("trace")) for row in traces],
            expected_ranks=expected, config=json.loads(config_json),
            platform=platform, executes_by_platform=executes, formats=formats)
    except (gate.TesseraRouteTraceError, ValueError) as exc:
        return problems + [f"{slot}: REFUSED on replay: {exc}"]
    if verdict["status"] != gate.AGREE:
        problems.append(f"{slot}: {verdict['detail']}")
    if record.get("passed") is not True:
        problems.append(f"{slot}: record carries passed={record.get('passed')!r}")
    if record.get("trace_verdict") != verdict:
        problems.append(
            f"{slot}: carried trace_verdict differs from the replay against "
            "the current packaged contract")
    return problems


# ---------------------------------------------------------------------------
# The commands that fill both slots (python -m prismaquant.shipcard_cli)
# ---------------------------------------------------------------------------
def _cmd_fill_route_census(args: argparse.Namespace) -> int:
    """Close `route.census` from the priced routes and the served records."""
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError,
        parse_census_json,
        substitute_decoders_from_contract_answer,
    )

    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    try:
        records = parse_census_json(Path(args.census).read_bytes().decode("utf-8"), where=str(args.census))
    except (OSError, ValueError) as exc:
        print(f"[shipcard] ERROR: cannot read census rows from "
              f"{args.census}: {exc}", file=sys.stderr)
        return 2
    card = load_shipcard(args.shipcard)
    scoped = isinstance(records, dict)
    binding = None
    if scoped:
        try:
            if not args.layer_config:
                raise TesseraRouteReceiptError("v2 census requires --layer-config exact allocation input")
            binding = {"layer_config_json": Path(args.layer_config).read_bytes().decode("utf-8"),
                       "config_json": (Path(model_dir) / "config.json").read_bytes().decode("utf-8"),
                       "manifest_json": (Path(model_dir) / "tessera_serving_manifest.json").read_bytes().decode("utf-8")}
        except (OSError, ValueError) as exc:
            print(f"[shipcard] REFUSED: cannot bind scoped census: {exc}", file=sys.stderr)
            return 2
    substitutes = list(args.substitute_decoder or ())
    if not substitutes and not scoped:
        try:
            from prismaquant import tessera_runtime_contract as trc

            contract = trc.load_tessera_contract()
            if contract is not None:
                substitutes = list(
                    substitute_decoders_from_contract_answer(
                        trc.contract_answer(contract)))
        except trc.TesseraContractError as exc:
            print(f"[shipcard] ERROR: {exc}", file=sys.stderr)
            return 2
    if not substitutes and not scoped:
        print("[shipcard] ERROR: no substitute decoder is known -- pass "
              "--substitute-decoder explicitly (repeatable) or set "
              "PRISMAQUANT_TESSERA_DEV_PIN so the pinned contract answer "
              "can be read. A gate that knows no substitute detects "
              "nothing.", file=sys.stderr)
        return 2
    try:
        record = make_route_census_record(
            tool=args.tool or f"route-census:{Path(args.census).name}",
            model_sha=compute_model_sha(model_dir),
            priced_routes=list(args.priced_route),
            route_records=records,
            substitute_decoders=substitutes,
            binding=binding,
            build=card.get("build"),
            model_dir=model_dir,
        )
    except TesseraRouteReceiptError as exc:
        print(f"[shipcard] REFUSED: {args.census} cannot be a census "
              f"receipt: {exc}", file=sys.stderr)
        return 2
    # Lane-scoped, not optional: the slot exists on cards the lane opened
    # (`lane_shipcard open --lane tessera`).  Filling a card that never
    # opened it is a refusal, not an auto-added key -- a slot the card does
    # not owe is a slot the receipt does not belong on.
    if ROUTE_CENSUS_SLOT not in (card.get("slots") or {}):
        print(f"[shipcard] REFUSED: {args.shipcard} has no "
              f"{ROUTE_CENSUS_SLOT} slot; open a Tessera lane card first "
              f"(python -m prismaquant.lane_shipcard open --lane tessera "
              f"--artifact {model_dir})", file=sys.stderr)
        return 2
    fill_slot(args.shipcard, ROUTE_CENSUS_SLOT, record)
    print(f"[shipcard] filled {ROUTE_CENSUS_SLOT} from {args.census} "
          f"(passed={record['passed']})")
    print(f"[shipcard]   {record['detail']}")
    return 0


def _cmd_fill_route_trace(args: argparse.Namespace) -> int:
    """Close `route.trace` from every rank's served route trace (#575)."""
    from prismaquant.tessera_route_trace_gate import (
        RouteTraceNotVerified,
        TesseraRouteTraceError,
    )

    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    card = load_shipcard(args.shipcard)
    if ROUTE_TRACE_SLOT not in (card.get("slots") or {}):
        print(f"[shipcard] REFUSED: {args.shipcard} has no {ROUTE_TRACE_SLOT} "
              "slot; open a Tessera lane card first (python -m "
              f"prismaquant.lane_shipcard open --lane tessera --artifact "
              f"{model_dir})", file=sys.stderr)
        return 2
    try:
        config_json = (Path(model_dir) / "config.json").read_bytes().decode("utf-8")
    except (OSError, ValueError) as exc:
        print(f"[shipcard] REFUSED: cannot read the artifact's config.json: "
              f"{exc}", file=sys.stderr)
        return 2
    traces = []
    for rank, path in enumerate(args.trace):
        label = f"rank{rank}:{Path(path).name}"
        try:
            traces.append((label, Path(path).read_bytes().decode("utf-8")))
        except FileNotFoundError:
            traces.append((label, None))
        except (OSError, ValueError) as exc:
            print(f"[shipcard] NOT VERIFIED: cannot read {path}: {exc}",
                  file=sys.stderr)
            return EXIT_NOT_VERIFIED
    try:
        record = make_route_trace_record(
            tool=args.tool or "fill-route-trace",
            model_sha=compute_model_sha(model_dir),
            traces=traces,
            expected_ranks=args.expected_ranks,
            config_json=config_json,
            build=card.get("build"),
            platform=args.platform,
        )
    except RouteTraceNotVerified as exc:
        print("[shipcard] NOT VERIFIED -- route.trace stays unfilled and the "
              f"card stays unpublishable: {exc}", file=sys.stderr)
        return EXIT_NOT_VERIFIED
    except TesseraRouteTraceError as exc:
        print(f"[shipcard] REFUSED -- route.trace: {exc}", file=sys.stderr)
        return 1
    fill_slot(args.shipcard, ROUTE_TRACE_SLOT, record)
    print(f"[shipcard] filled {ROUTE_TRACE_SLOT} from {len(traces)} rank "
          f"trace(s) (passed={record['passed']})")
    print(f"[shipcard]   {record['detail']}")
    return 0


def register_cli(sub: argparse._SubParsersAction) -> None:
    """Add ``fill-route-census`` and ``fill-route-trace`` to the shipcard CLI."""
    p_census = sub.add_parser(
        "fill-route-census",
        help="close route.census from the priced routes and the serve's "
             "route records (Tessera lane: priced-vs-served decoder gate)",
    )
    p_census.add_argument("shipcard")
    p_census.add_argument(
        "--census", required=True,
        help="Complete Tessera route_census/2 JSON (scoped), or historical unscoped row array")
    p_census.add_argument("--layer-config", default=None,
                         help="Exact allocation JSON bound by card.build.layer_config_sha; required for v2")
    p_census.add_argument(
        "--priced-route", action="append", default=[],
        help="legacy priced route (repeatable, required for flat rows); optional cross-check for v2")
    p_census.add_argument(
        "--substitute-decoder", action="append", default=[],
        help="a decoder a serve falls back to (repeatable; default: derived "
             "from the pinned Tessera contract answer, which needs "
             "PRISMAQUANT_TESSERA_DEV_PIN)")
    p_census.add_argument("--model-dir", default=None)
    p_census.add_argument("--tool", default=None)
    p_census.set_defaults(func=_cmd_fill_route_census)

    p_trace = sub.add_parser(
        "fill-route-trace",
        help="close route.trace from every rank's TESSERA_ROUTE_TRACE file "
             "(Tessera lane: priced-vs-served activation-contract gate). "
             "Exit 0 agree, 1 refused, 2 usage, 3 not verified",
    )
    p_trace.add_argument("shipcard")
    p_trace.add_argument(
        "--trace", action="append", required=True,
        help="one rank's tessera.route_trace/1 JSON, in rank order "
             "(repeatable; a path that does not exist is a missing rank)")
    p_trace.add_argument(
        "--expected-ranks", type=int, required=True,
        help="the serve's world size; fewer traces than this is NOT VERIFIED")
    p_trace.add_argument(
        "--platform", default=None,
        help="serving platform to price on (default: the card's "
             "tessera_serving_scope target; both, when present, must agree)")
    p_trace.add_argument("--model-dir", default=None)
    p_trace.add_argument("--tool", default=None)
    p_trace.set_defaults(func=_cmd_fill_route_trace)
