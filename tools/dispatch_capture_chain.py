"""Dispatch a streamed calibration capture as a layer chain (PQ #1885).

``prismaquant.capture_layer_chain`` cuts the streamed capture into a prep
row, one quantum row per layer range and a join row. Each quantum reads the
boundary its predecessor wrote, so the rows run one after another, and
``pbrun --after`` cannot order them: a prep or a quantum commits no produced
output for a consumer to wait on. This tool orders them by what they leave
on disk, as ``dispatch_stage_a_split`` does for the Stage A split:

* **seal** writes ``<workspace>/capture-chain/round.json`` from the campaign
  spec: every row's PrismaBuild row (built by
  ``dispatch_tessera_campaign._row``, so the class, container, environment
  and demand are the capture row's), at priority -10 with an explicit
  ``timeout_s``. The prep and the join run no forward and claim no GPU.
  A quantum's progress phase (``capture``, reporting each layer whose units
  are journalled) is declared only when asked for (``--progress-grace-s``),
  and ``pbrun`` refuses it unless the fleet announces the progress contract.
* **plan** prints the sealed rows.
* **submit** submits the next unsubmitted row, detached, once the row before
  it ended ``executed`` with exit 0 and its outputs check: the prep's sealed
  record and complete owner, each quantum's sealed fragment and complete
  owner. ``--wait-s`` waits for each submitted row through ``pbwait`` and
  walks the whole chain. ``--retry NAME`` re-submits a row whose last
  submission did not finish with exit 0; its record is kept beside the new one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from dispatch_stage_a_split import SplitDispatchRefused, pb_ending, submission_path  # noqa: E402
from dispatch_tessera_campaign import PBCAMPAIGN, _row, _row_memory_gb, load_spec  # noqa: E402

ROUND_SCHEMA = "prismaquant.capture_chain_round.v1"
ROUND_NAME = "round.json"
PRIORITY = -10
PBWAIT = PBCAMPAIGN.parent / "pbwait.py"
#: The phase a capture chain quantum reports its journalled layers in.
QUANTUM_PROGRESS_PHASE = "capture"


class ChainDispatchRefused(SplitDispatchRefused):
    """The chain cannot be sealed or a row cannot be submitted yet."""


def round_directory(workspace) -> Path:
    return Path(workspace).resolve() / "capture-chain"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seal(spec_path, workspace, *, ranges, boundary_storage, timeout_s, bookend_timeout_s,
         bookend_mem_gb=16, progress_phases=()) -> dict:
    """Write the chain's rows once; nothing reaches PrismaBuild."""
    from prismaquant.capture_layer_chain import (parse_layer_ranges, range_label,
                                                 require_layer_tiling)
    from prismaquant.cost_streaming import check_boundary_storage
    from prismaquant.tessera_calibration_cache import sha256
    for value, flag in ((timeout_s, "--timeout-s"), (bookend_timeout_s, "--bookend-timeout-s")):
        if type(value) is not int or value <= 0:
            raise ChainDispatchRefused(f"every chain row needs an explicit positive {flag}")
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
        row = _row(spec, argv, mem_gb=mem_gb, timeout_s=timeout, progress_phases=phases)
        row["priority"] = PRIORITY
        if not gpu:
            row["demand"] = {**row["demand"], "gpu": 0}
        rows.append({"name": name, "kind": kind, "layers": layers, "row": row})

    text = ",".join(f"{start}:{stop}" for start, stop in pairs)
    add("prep", "prep", ["--capture-chain-ranges", text, "--capture-chain-boundary-storage",
                         json.dumps(boundary_storage, sort_keys=True)],
        mem_gb=bookend_mem_gb, timeout=bookend_timeout_s, phases=(), gpu=False)
    quantum_mem_gb = _row_memory_gb(spec, sorted(census["counts"]), census)
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


def load_round(round_dir) -> dict:
    from prismaquant.tessera_calibration_cache import sha256
    document = json.loads((Path(round_dir) / ROUND_NAME).read_text())
    if document.get("schema") != ROUND_SCHEMA:
        raise ChainDispatchRefused(f"{round_dir} holds no sealed capture chain")
    if sha256(document["census"]["path"]) != document["census"]["sha256"]:
        raise ChainDispatchRefused("the census changed since the chain was sealed")
    return document


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


def _pb_ending(round_dir, name) -> dict:
    """``dispatch_stage_a_split.pb_ending``, refusing as this tool refuses."""
    try:
        return pb_ending(round_dir, name)
    except ChainDispatchRefused:
        raise
    except SplitDispatchRefused as exc:
        raise ChainDispatchRefused(str(exc)) from exc


def require_ready(round_dir, document, index) -> None:
    """Refuse row ``index`` until the row before it finished and its outputs check."""
    if index == 0:
        return
    before = document["rows"][index - 1]
    _pb_ending(round_dir, before["name"])
    require_outputs(document, before)


def _submit_row(round_dir, entry, *, run) -> dict:
    manifest = Path(round_dir) / "manifests" / f"{entry['name']}.json"
    _write_json(manifest, [entry["row"]])
    command = [sys.executable, str(PBCAMPAIGN), "--detach", str(manifest)]
    completed = run(command, capture_output=True, text=True)
    detach = None
    for line in (completed.stdout or "").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "action_key" in value:
            detach = value
    result = {"name": entry["name"], "argv": command, "returncode": completed.returncode,
              "stdout": completed.stdout, "stderr": completed.stderr, "detach": detach}
    _write_json(submission_path(round_dir, entry["name"]), result)
    if completed.returncode != 0 or detach is None:
        raise ChainDispatchRefused(
            f"row {entry['name']} was not submitted (exit {completed.returncode}): "
            f"{(completed.stderr or '').strip()[-400:]}")
    return result


def _retire_attempt(round_dir, name) -> None:
    """Keep a failed submission's record beside the retry's."""
    path = submission_path(round_dir, name)
    attempt = 1
    while path.with_name(f"{name}.attempt-{attempt}.json").exists():
        attempt += 1
    path.rename(path.with_name(f"{name}.attempt-{attempt}.json"))


def submit(round_dir, *, wait_s=None, retry=None, run=subprocess.run) -> list[dict]:
    """Submit the next row (with ``wait_s``, every remaining row) in chain order."""
    round_dir = Path(round_dir)
    document = load_round(round_dir)
    names = [entry["name"] for entry in document["rows"]]
    if retry is not None:
        if retry not in names:
            raise ChainDispatchRefused(f"the chain has no row {retry}")
        if not submission_path(round_dir, retry).exists():
            raise ChainDispatchRefused(f"row {retry} was never submitted")
        try:
            _pb_ending(round_dir, retry)
        except ChainDispatchRefused:
            _retire_attempt(round_dir, retry)
        else:
            raise ChainDispatchRefused(f"row {retry} already ended with exit 0")
    results = []
    for index, entry in enumerate(document["rows"]):
        if submission_path(round_dir, entry["name"]).exists():
            continue
        require_ready(round_dir, document, index)
        results.append(_submit_row(round_dir, entry, run=run))
        if wait_s is None:
            break
        key = results[-1]["detach"]["action_key"]
        run([sys.executable, str(PBWAIT), "--wait-s", str(wait_s), key],
            capture_output=True, text=True)
        _pb_ending(round_dir, entry["name"])
        require_outputs(document, entry)
    return results


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
    planning = sub.add_parser("plan", help="print the sealed rows")
    planning.add_argument("--workspace", required=True)
    submitting = sub.add_parser("submit", help="submit the next row once its predecessor finished")
    submitting.add_argument("--workspace", required=True)
    submitting.add_argument("--wait-s", type=float, default=None,
                            help="wait for each row through pbwait and walk the whole chain")
    submitting.add_argument("--retry", default=None, help="re-submit this row after a failure")
    args = ap.parse_args(argv)
    try:
        if args.command == "seal":
            phases = () if args.progress_grace_s is None else (
                (QUANTUM_PROGRESS_PHASE, args.progress_grace_s),)
            document = seal(args.spec, args.workspace, ranges=args.ranges,
                            boundary_storage=args.boundary_storage, timeout_s=args.timeout_s,
                            bookend_timeout_s=args.bookend_timeout_s,
                            bookend_mem_gb=args.bookend_mem_gb, progress_phases=phases)
            print(f"[capture-chain] sealed {len(document['rows'])} rows in "
                  f"{round_directory(args.workspace) / ROUND_NAME}")
        elif args.command == "plan":
            document = load_round(round_directory(args.workspace))
            for entry in document["rows"]:
                print(json.dumps({"name": entry["name"], "row": entry["row"]}, sort_keys=True))
        else:
            for result in submit(round_directory(args.workspace), wait_s=args.wait_s,
                                 retry=args.retry):
                print(json.dumps({"name": result["name"], "detach": result["detach"]},
                                 sort_keys=True))
    except SplitDispatchRefused as exc:
        print(f"[capture-chain] refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
