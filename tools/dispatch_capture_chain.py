"""Seal and dispatch a streamed calibration capture as native Flux rows.

Prep, layer quanta and join retain the campaign class, source/read bindings,
resource demands, storage, progress and priority intent. Every predecessor
must have native success and independently verified scientific outputs.
Resume validates the same saved launch; only an explicit terminal failure
permits a requested retry, with the earlier attempt retained. Unsupported
production requirements refuse before the prep launches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from dispatch_stage_a_split import (  # noqa: E402
    SplitDispatchRefused,
    _write_json,
    native_ending,
    native_result,
    saved_submission,
    submit_row,
    retire_attempt,
    submission_path,
)
from prismaquant.tessera_campaign_rows import (  # noqa: E402
    load_spec, campaign_row, row_memory_gb,
)

ROUND_SCHEMA = "prismaquant.capture_chain_round.v2"
ROUND_NAME = "round.json"
#: The agent band, as ``dispatch_tessera_campaign --priority``: agent and
#: post-campaign work never displaces campaign rows. A gate-critical chain
#: passes ``--priority 0`` (PQ #1918).
PRIORITY = -10
#: The phase a capture chain quantum reports its journalled layers in.
QUANTUM_PROGRESS_PHASE = "capture"


class ChainDispatchRefused(SplitDispatchRefused):
    """The chain cannot be sealed or a row cannot be submitted yet."""


def round_directory(workspace) -> Path:
    return Path(workspace).resolve() / "capture-chain"


def seal(spec_path, workspace, *, ranges, boundary_storage, timeout_s, bookend_timeout_s,
         bookend_mem_gb=16, progress_phases=(), priority=PRIORITY) -> dict:
    """Write the chain's rows once; nothing is submitted."""
    from prismaquant.capture_layer_chain import (parse_layer_ranges, range_label,
                                                 require_layer_tiling)
    from prismaquant.streamed_boundary_artifacts import check_boundary_storage
    from prismaquant.digests import DIRECT_ASCII_SPACED_LAX
    from prismaquant.tessera_calibration_cache import sha256
    for value, flag in ((timeout_s, "--timeout-s"), (bookend_timeout_s, "--bookend-timeout-s")):
        if type(value) is not int or value <= 0:
            raise ChainDispatchRefused(f"every chain row needs an explicit positive {flag}")
    if type(priority) is not int:
        raise ChainDispatchRefused("a chain's priority is an integer queue band")
    spec = load_spec(Path(spec_path))
    if "--streaming" not in spec["campaign_argv"]:
        raise ChainDispatchRefused("a capture chain is a streamed capture: the spec's "
                                   "campaign_argv names no --streaming")
    workspace = Path(workspace).resolve()
    census_path = workspace / "census.json"
    census = json.loads(census_path.read_text())
    if census.get("model") != spec["model"]:
        raise ChainDispatchRefused("capture census and spec name different models")
    pairs = require_layer_tiling(parse_layer_ranges(ranges))
    check_boundary_storage(boundary_storage)
    round_dir = round_directory(workspace)
    if (round_dir / ROUND_NAME).exists():
        raise ChainDispatchRefused(f"{round_dir / ROUND_NAME} exists: a chain is sealed once")
    capture_root = workspace / "calibration-cache"
    common = ["--model", spec["model"], "--calibration-census", str(census_path),
              "--capture-calibration-out", str(capture_root), *spec["campaign_argv"]]
    rows = []

    def add(name, kind, extra, *, mem_gb, timeout, phases, gpu, layers=None):
        argv = ["--out", str(round_dir / "out" / f"{name}-unused.pkl"),
                "--cache-dir", str(round_dir / "cache" / name), *common,
                "--capture-chain", kind, *extra]
        row = campaign_row(spec, argv, mem_gb=mem_gb, timeout_s=timeout, progress_phases=phases)
        row["priority"] = priority
        if not gpu:
            row["demand"] = {**row["demand"], "gpu": 0}
        rows.append({"name": name, "kind": kind, "layers": layers, "row": row})

    text = ",".join(f"{start}:{stop}" for start, stop in pairs)
    add("prep", "prep", ["--capture-chain-ranges", text, "--capture-chain-boundary-storage",
                         DIRECT_ASCII_SPACED_LAX.text(boundary_storage)],
        mem_gb=bookend_mem_gb, timeout=bookend_timeout_s, phases=(), gpu=False)
    quantum_mem_gb = row_memory_gb(spec, sorted(census["counts"]), census)
    for start, stop in pairs:
        add(range_label(start, stop), "quantum", ["--capture-layer-range", f"{start}:{stop}"],
            mem_gb=quantum_mem_gb, timeout=timeout_s, phases=tuple(progress_phases),
            gpu=True, layers=[start, stop])
    add("join", "join", [], mem_gb=bookend_mem_gb, timeout=bookend_timeout_s,
        phases=(), gpu=False)
    document = {"schema": ROUND_SCHEMA, "capture_root": str(capture_root),
                "census": {"path": str(census_path), "sha256": sha256(census_path)},
                "ranges": [list(pair) for pair in pairs], "rows": rows}
    _write_json(round_dir / ROUND_NAME, document)
    return document


def load_capture_round(round_dir) -> dict:
    from prismaquant.tessera_calibration_cache import sha256
    document = json.loads((Path(round_dir) / ROUND_NAME).read_text())
    if document.get("schema") != ROUND_SCHEMA:
        raise ChainDispatchRefused(f"{round_dir} holds no sealed capture chain")
    if sha256(document["census"]["path"]) != document["census"]["sha256"]:
        raise ChainDispatchRefused("the census changed since the chain was sealed")
    return document


load_round = load_capture_round


def require_outputs(document, entry) -> None:
    """What a finished row leaves on disk, checked as its successor's reader checks it."""
    from prismaquant import capture_layer_chain as chain
    from prismaquant.tessera_calibration_cache import require_capture_contract
    root = document["capture_root"]
    try:
        prep = chain.read_prep(root)
        if prep["ranges"] != document["ranges"]:
            raise ChainDispatchRefused("the chain's prep is not this round's")
        if entry["kind"] == "join":
            try:
                receipt = json.loads(chain.join_path(root).read_bytes())
            except (OSError, ValueError) as exc:
                raise ChainDispatchRefused("the join left no readable receipt") from exc
            if (receipt.get("schema") != chain.JOIN_SCHEMA
                    or receipt.get("prep_sha256") != prep["prep_sha256"]):
                raise ChainDispatchRefused("the join's receipt is not this chain's")
            manifest = receipt["manifest"]
            try:
                require_capture_contract(manifest["path"], manifest["sha256"])
            except (OSError, ValueError, RuntimeError) as exc:
                raise ChainDispatchRefused(f"the joined capture does not check: {exc}") from exc
            return
        if entry["kind"] == "prep":
            status = json.loads((chain.generation_directory(prep) / "owners" /
                                 f"{chain.PREP_OWNER_LABEL}.json").read_text())
            if status.get("status") != "complete" or status.get("session") != prep["session"]:
                raise ChainDispatchRefused("the prep's owner did not complete")
            return
        chain.require_owner_complete(prep, *entry["layers"])
        chain.read_fragment(root, prep, *entry["layers"])
    except chain.CaptureChainRefused as exc:
        raise ChainDispatchRefused(str(exc)) from exc


def _native_ending(round_dir, name) -> dict:
    """Native completion barrier, with this caller's refusal type."""
    try:
        return native_ending(round_dir, name)
    except ChainDispatchRefused:
        raise
    except SplitDispatchRefused as exc:
        raise ChainDispatchRefused(str(exc)) from exc


def require_capture_ready(round_dir, document, index) -> None:
    """Refuse row ``index`` until the row before it finished and its outputs check."""
    if index == 0:
        return
    before = document["rows"][index - 1]
    _native_ending(round_dir, before["name"])
    require_outputs(document, before)


require_ready = require_capture_ready


def submit_capture_chain(round_dir, *, wait=False, retry=None) -> list[dict]:
    """Submit/resume in order, binding native IDs and independent output barriers."""
    from prismaquant.tessera_campaign_rows import validate_row
    round_dir = Path(round_dir)
    document = load_round(round_dir)
    names = [entry["name"] for entry in document["rows"]]
    # Refuse unsupported downstream requirements before even the prep launches.
    try:
        for entry in document["rows"]:
            validate_row(entry["row"])
        if retry is not None:
            if retry not in names:
                raise ChainDispatchRefused(f"the chain has no row {retry}")
            if any(submission_path(round_dir, name).exists() for name in names[names.index(retry) + 1:]):
                raise ChainDispatchRefused("cannot retry a producer with submitted downstream rows")
            retire_attempt(round_dir, retry)
        results = []
        for index, entry in enumerate(document["rows"]):
            existed = submission_path(round_dir, entry["name"]).exists()
            if not existed:
                require_ready(round_dir, document, index)
            producers = [] if index == 0 else [saved_submission(round_dir, names[index - 1])]
            saved = submit_row(round_dir, entry, afterok=producers)
            result = native_result(round_dir, entry["name"], wait=wait)
            if not existed or result["status"] == "running":
                results.append(saved)
            if not wait and result["status"] == "running":
                break
            _native_ending(round_dir, entry["name"])
            require_outputs(document, entry)
        return results
    except (ValueError, KeyError, SplitDispatchRefused) as exc:
        raise ChainDispatchRefused(str(exc)) from exc


submit = submit_capture_chain


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sealing = sub.add_parser("seal", help="write the chain's rows")
    sealing.add_argument("--spec", required=True)
    sealing.add_argument("--workspace", required=True)
    sealing.add_argument("--ranges", required=True, help="A:B,B:C,... tiling every source layer")
    sealing.add_argument("--boundary-storage", required=True, type=json.loads,
                         help="boundary storage policy JSON the quanta pass hidden states through")
    sealing.add_argument("--timeout-s", required=True, type=int, help="each quantum's deadline")
    sealing.add_argument("--bookend-timeout-s", required=True, type=int,
                         help="the prep's and the join's deadline")
    sealing.add_argument("--bookend-mem-gb", type=int, default=16)
    sealing.add_argument("--progress-grace-s", type=int, default=None,
                         help="opt-in: declare each quantum's progress phase capture=SECONDS")
    sealing.add_argument("--priority", type=int, default=PRIORITY,
                         help="every row's queue band; -10 is the agent band, and a chain "
                              "that feeds a gate is campaign work and runs at 0")
    planning = sub.add_parser("plan", help="print the sealed rows")
    planning.add_argument("--workspace", required=True)
    submitting = sub.add_parser("submit", help="submit the next row once its predecessor finished")
    submitting.add_argument("--workspace", required=True)
    submitting.add_argument("--wait", action="store_true",
                            help="wait for native completion and walk the whole chain")
    submitting.add_argument("--retry", default=None, help="re-submit this row after a failure")
    args = ap.parse_args(argv)
    from prismaquant.digests import DIRECT_ASCII_SPACED_LAX

    try:
        if args.command == "seal":
            phases = () if args.progress_grace_s is None else (
                (QUANTUM_PROGRESS_PHASE, args.progress_grace_s),)
            document = seal(args.spec, args.workspace, ranges=args.ranges,
                            boundary_storage=args.boundary_storage, timeout_s=args.timeout_s,
                            bookend_timeout_s=args.bookend_timeout_s,
                            bookend_mem_gb=args.bookend_mem_gb, progress_phases=phases,
                            priority=args.priority)
            print(f"[capture-chain] sealed {len(document['rows'])} rows in "
                  f"{round_directory(args.workspace) / ROUND_NAME}")
        elif args.command == "plan":
            document = load_round(round_directory(args.workspace))
            for entry in document["rows"]:
                print(DIRECT_ASCII_SPACED_LAX.text({"name": entry["name"], "row": entry["row"]}))
        else:
            for result in submit(round_directory(args.workspace), wait=args.wait,
                                 retry=args.retry):
                print(DIRECT_ASCII_SPACED_LAX.text(
                    {"name": result["name"], "submissions": result["submissions"]}))
    except SplitDispatchRefused as exc:
        print(f"[capture-chain] refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
