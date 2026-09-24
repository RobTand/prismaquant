"""Dispatch one Stage A chain split round as plain PrismaBuild rows (PQ #738).

A split round (``prismaquant.stage_a_chain_split``) runs as a prep row, one
quantum row per sample range, a join row per stride checkpoint the round
seals, and a band row per joined checkpoint. Until ``pbcampaign`` carries
the residency fields (PB #1082) the rows are plain ``pbrun`` submissions,
``--tag gb10 --priority -10`` and never a host. ``pbrun --after`` does not
order them: it defers a consumer on a producer's write-only template and
builds the consumer's manifest from the batches it committed, and a prep
commits nothing. So this tool orders the rows by what they leave on disk:

* **seal** writes the round directory from one clean checkout: every row's
  produced-output template, and ``round.json``, which holds every row's full
  ``pbrun`` argv and the digest of every input it names. Nothing reaches
  PrismaBuild.
* **plan** prints the sealed rows after checking the pins (the dry run).
* **submit** runs rows, each only once the rows before it have finished:
  the quanta once the prep's receipt and resume record are on disk and its
  PrismaBuild ending is ``executed`` with exit 0; a join once every quantum
  has; a band once its join has. Each submission's argv and ``pbrun``
  output land in ``submissions/``.
* **compare** holds each quantum's payload digests at its digest layer
  (``split/quanta/<label>.digests.json``, written as the layer rolls) to a
  baseline: the tripwire for a numeric defect, minutes into a round.

Every row runs the checkout the round was sealed from: ``plan_chain_resume``
requires the running implementation to be the one the latest resume record
stamps, so the prep, the quanta and the joins must be one tree. Only the
prep declares an implementation switch.

The Stage A rows are built by ``dispatch_joint_quanta.stage_a_argv``: the
full residency path (``--residency stage --residency-ram auto``, the paced
produced-output spool), with two per-row numbers. A quantum's spool window
is two planes of its own batches, and its template's stage window funds at
most the groups it can hold across two consecutive boundaries. The reader
declaration (``--residency-prefetch-depth-gib``, ``--residency-read-mb-s``)
is derived from each row's manifest: the depth is the furthest source byte
the chain's prefetch reaches past the phase it reads, and the rate is an
estimate recorded with its derivation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

ROUND_SCHEMA = "prismaquant.stage_a.split_round.v1"
ROUND_NAME = "round.json"
PRIORITY = "-10"
GIB = 1 << 30
_SOURCE = re.compile(r"\.safetensors$")


class SplitDispatchRefused(RuntimeError):
    """The round cannot be sealed or a row cannot be submitted yet."""


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path) -> dict:
    return {"path": str(path), "sha256": _sha(path)}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def implementation_sha256(checkout) -> str:
    """The implementation a row from ``checkout`` runs, as the capture computes it."""
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    return _production_cache_source_sha256(Path(checkout) / "prismaquant")


def checkout_commit(checkout) -> str:
    """``checkout``'s HEAD; refuses a tree with any change or untracked file."""
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain"):
        raise SplitDispatchRefused(
            f"{checkout} has uncommitted changes: every row of a round runs one commit")
    return git("rev-parse", "HEAD")


# -- per-row numbers -------------------------------------------------------------

def readahead_depth(manifest: dict, lookahead: int) -> dict:
    """How far past the phase it reads the chain's source prefetch reaches.

    PrismaBuild's refill horizon extends from the end of the phase the
    reader is in by the declared depth (#903/#909), so the depth is the
    furthest source byte the reader touches past that end. While ``chain-L``
    is read, each install schedules layers ``L-1 .. L-lookahead`` inside the
    walk (``source_read_plan.chain_prefetch_window``, which stops at the
    walk's end); boundary rows and planes are read only inside their own
    phase. The head's opening window is reported, not declared: it is
    scheduled while the head is still the accepted phase. Returns the rows
    and ``declared_gib``, the chain rows' reach rounded up to whole GiB
    (PrismaBuild takes an integer).
    """
    entries = manifest["entries"]
    table, position = {}, 0
    for phase in manifest["read_plan"]["phases"]:
        start, source, end = position, 0, None
        for index in phase["entry_indices"]:
            position += int(entries[index]["bytes"])
            if _SOURCE.search(entries[index]["path"]):
                source += int(entries[index]["bytes"])
                end = position
        table[phase["name"]] = {"start": start, "end": position, "source_bytes": source,
                                "source_end": end}
    chains = [name for name in table if name.startswith("chain-")]
    rows = []
    for name, row in table.items():
        if name == "head":
            ahead = chains[:lookahead]
        else:
            layer = int(name.removeprefix("chain-"))
            ahead = [f"chain-{layer - k:03d}" for k in range(1, lookahead + 1)
                     if f"chain-{layer - k:03d}" in table]
        ahead = [phase for phase in ahead if table[phase]["source_end"] is not None]
        furthest = max((table[phase]["source_end"] for phase in ahead), default=row["end"])
        rows.append({"reading": name, "ahead": ahead,
                     "reach_bytes": max(0, furthest - row["end"])})
    declared = max((row["reach_bytes"] for row in rows if row["reading"] != "head"),
                   default=0)
    return {"lookahead": int(lookahead), "rows": rows, "declared_gib": -(-declared // GIB),
            "rounding": "ceil(reach_bytes / GiB): PrismaBuild takes whole GiB"}


def quantum_read_rate(described: dict, *, samples, n_batches, seconds_per_layer) -> dict:
    """An estimate of the rate a quantum reads its steady chain phases, in MB/s.

    The largest chain phase after the first (the first also carries the
    resume checkpoint's plane) over the single owner's measured seconds per
    layer scaled by the quantum's share of the samples. The scaling assumes
    a layer's roll time is proportional to the samples it rolls; round 1
    measures it. PrismaBuild prices ``max(measured, declared)``.
    """
    phases = [phase for phase in described["phases"] if phase["name"].startswith("chain-")]
    steady = max(phases[1:] or phases, key=lambda phase: phase["bytes"])
    share = (int(samples[1]) - int(samples[0])) / int(n_batches)
    seconds = float(seconds_per_layer) * share
    return {"read_mb_s": math.ceil(steady["bytes"] / 1e6 / seconds),
            "kind": "estimate", "phase": steady["name"], "phase_bytes": steady["bytes"],
            "seconds_per_layer_single_owner": float(seconds_per_layer),
            "sample_share": share, "seconds_per_layer_estimated": seconds,
            "rule": "steady phase bytes / (single-owner s/layer x sample share), "
                    "ceil decimal MB/s"}


def window_groups(base_template: dict, *, tier: str, n_probes: int,
                  range_batches: int, group_size: int) -> dict:
    """The stage window a row's template funds, in groups and GiB.

    The base template's minimum is one group's whole-GiB ceiling and its
    window a number of such groups (``stage_a_produced_output.
    build_boundary_template``). A split row holds at most the groups of two
    consecutive boundaries of its own range: at each, one boundary group and
    one cotangent group per probe per read window. Credit past that funds
    nothing, so the row's window is the smaller of the two.
    """
    demands = base_template["working_demands"][tier]
    per_group = int(demands["minimum_gib"])
    base = int(demands["window_gib"]) // per_group
    held = 2 * (1 + int(n_probes)) * -(-int(range_batches) // int(group_size))
    groups = min(base, held)
    return {"groups": groups, "window_gib": groups * per_group, "minimum_gib": per_group,
            "base_groups": base, "range_groups": held}


def row_template(base_template: dict, *, template_id: str, tier: str,
                 window: dict, artifact_budget_bytes: int) -> dict:
    """A row's template: the base's prefix, tier and slots under its own id."""
    template = json.loads(json.dumps(base_template))
    maxima = template["durable_maxima"]
    if int(maxima["payload_max_bytes"]) != int(artifact_budget_bytes):
        raise SplitDispatchRefused(
            f"the base template's payload maximum {maxima['payload_max_bytes']} is not "
            f"the artifact budget {artifact_budget_bytes}")
    template["template_id"] = template_id
    template["working_demands"] = {tier: {"minimum_gib": window["minimum_gib"],
                                         "window_gib": window["window_gib"]}}
    return template


# -- sealing -------------------------------------------------------------------------

def _stage_row(*, name, kind, checkout, manifest, template_path, campaign, spec,
               prefetch_override, artifact_budget_bytes, tag, payload_extra, reader,
               batch_range=None) -> dict:
    import dispatch_joint_quanta as dispatch

    dispatch.SPEC_PATH = Path(spec)
    argv = dispatch.stage_a_argv(
        Path(manifest), campaign, tag=tag, prefetch_override=Path(prefetch_override),
        artifact_budget_bytes=artifact_budget_bytes,
        produced_output_template=Path(template_path), batch_range=batch_range)
    first = argv.index("--")
    inner = argv.index("--", first + 1)
    envelope, payload = argv[:first], argv[inner + 1:]
    if "--forward-recovery" in payload or "--chain-seed" in payload:
        raise SplitDispatchRefused("a split row takes no forward recovery and no seed")
    head = [*envelope[:2], "--cwd", str(checkout), *envelope[2:], "--priority", PRIORITY,
            "--residency-prefetch-depth-gib", str(reader["depth"]["declared_gib"]),
            "--residency-read-mb-s", str(reader["rate"]["read_mb_s"])]
    tags = [head[i + 1] for i, word in enumerate(head) if word == "--tag"]
    if tags != [tag] or "--host-class" in head or "--measurement" in head:
        raise SplitDispatchRefused(f"a split row carries exactly the tag {tag}: {tags}")
    return {"name": name, "kind": kind,
            "argv": [*head, *argv[first:inner + 1], *payload, *payload_extra],
            "data_manifest": _pin(manifest), "template": _pin(template_path),
            "reader": reader}


def _cpu_row(*, name, kind, checkout, tag, python, module_argv, mem_gb=8, cpus=2) -> dict:
    import dispatch_joint_quanta as dispatch

    return {"name": name, "kind": kind,
            "argv": [sys.executable, str(dispatch.PBRUN), "--cwd", str(checkout),
                     "--tag", tag, "--priority", PRIORITY, "--demand", f"mem_gb={mem_gb}",
                     "--cpus", str(cpus), "--detach", "--", str(python), *module_argv]}


def seal_round(*, round_dir, checkout, split_package, campaign, spec, prefetch_override,
               base_template, artifact_budget_bytes, chain_regime, seconds_per_layer,
               digest_layer, band_request, band_references, python, tag="gb10",
               template_prefix, tier) -> dict:
    """Write ``round_dir`` (templates and ``round.json``); submit nothing."""
    from prismaquant.joint_adjoint_checkpoints import adjoint_space
    from prismaquant.stage_a_chain_resume import load_chain_state, resume_records
    from prismaquant.stage_a_chain_split import quantum_label

    round_dir = Path(round_dir)
    if (round_dir / ROUND_NAME).exists():
        raise SplitDispatchRefused(f"{round_dir / ROUND_NAME} exists: a round is sealed once")
    commit = checkout_commit(checkout)
    running = implementation_sha256(checkout)
    package_path = Path(split_package)
    package = json.loads(package_path.read_text())
    plan = json.loads(Path(campaign["plan_path"]).read_text())
    output_root = Path(plan["output_root"])
    space = adjoint_space(output_root)
    state = load_chain_state(space, package["chain_state"]["sha256"])
    records = resume_records(space, state["boundary_storage"]["session"])
    sealed_by = (records[-1]["implementation_sha256"] if records
                 else state["run_identity"]["implementation_sha256"])
    declaration = None if sealed_by == running else f"{sealed_by}:{running}"
    base = json.loads(Path(base_template).read_text())
    n_probes = int(state["run_identity"]["n_probes"])
    n_batches = int(state["n_batches"])
    group_size = int(state["boundary_storage"]["policy"]["prefetch_batches"])
    lookahead = int(json.loads(Path(prefetch_override).read_text())
                    ["source_prefetch"]["prefetch_lookahead"])
    boundary, through = int(package["from"]), int(package["through"])
    if digest_layer is not None and not through <= int(digest_layer) < boundary:
        raise SplitDispatchRefused(
            f"the quanta roll layers {boundary - 1} down to {through}, not {digest_layer}")
    resume = ["--chain-batch-size", str(chain_regime["chain_batch_size"]),
              "--chain-probe-fusion", str(chain_regime["chain_probe_fusion"]),
              "--resume-chain-state-sha256", package["chain_state"]["sha256"],
              "--resume-from-checkpoint", str(boundary)]
    common = dict(checkout=checkout, campaign=campaign, spec=spec,
                  prefetch_override=prefetch_override,
                  artifact_budget_bytes=artifact_budget_bytes, tag=tag)
    templates = round_dir / "templates"
    templates.mkdir(parents=True, exist_ok=True)

    def template(name, samples):
        window = window_groups(base, tier=tier, n_probes=n_probes,
                               range_batches=samples[1] - samples[0], group_size=group_size)
        body = row_template(base, template_id=f"{template_prefix}-{name}", tier=tier,
                            window=window, artifact_budget_bytes=artifact_budget_bytes)
        path = templates / f"{name}.json"
        _write_json(path, body)
        return path, window

    rates = [quantum_read_rate(described, samples=described["samples"], n_batches=n_batches,
                               seconds_per_layer=seconds_per_layer)
             for described in package["quanta"]]
    rows = []
    ranges = [list(item["samples"]) for item in package["quanta"]]
    prep_manifest = gzip.decompress(Path(package["prep"]["data_manifest"]["path"]).read_bytes())
    # The prep writes no entry, but the wrapper binds its produced output
    # before the core: it is sized as the round's largest quantum.
    widest = tuple(max(ranges, key=lambda pair: pair[1] - pair[0]))
    path, window = template("prep", widest)
    rows.append({**_stage_row(
        name="prep", kind="prep", manifest=package["prep"]["data_manifest"]["path"],
        template_path=path, batch_range=widest, reader={
            "depth": readahead_depth(json.loads(prep_manifest), lookahead),
            "rate": {**max(rates, key=lambda rate: rate["read_mb_s"]),
                     "why": "the round's quanta's rate; the prep reads its head only"}},
        payload_extra=[*resume, "--chain-split-prep", str(through), "--chain-split-ranges",
                       ",".join(f"{start}:{stop}" for start, stop in ranges),
                       *(["--resume-implementation-compatibility", declaration]
                         if declaration is not None else [])],
        **common), "window": window})
    labels = []
    for described, rate in zip(package["quanta"], rates):
        start, stop = described["samples"]
        label = quantum_label(boundary, through, start, stop)
        labels.append(label)
        manifest = json.loads(gzip.decompress(
            Path(described["data_manifest"]["path"]).read_bytes()))
        path, window = template(label, (start, stop))
        rows.append({**_stage_row(
            name=label, kind="quantum", manifest=described["data_manifest"]["path"],
            template_path=path, batch_range=(start, stop),
            reader={"depth": readahead_depth(manifest, lookahead), "rate": rate},
            payload_extra=[*resume, "--chain-split-quantum", f"{through}:{start}:{stop}",
                           *(["--chain-split-digest-layer", str(digest_layer)]
                             if digest_layer is not None else [])],
            **common), "window": window, "label": label, "samples": [start, stop]})
    for mark in package["boundaries"]:
        receipt = round_dir / "joins" / f"join-{int(mark):03d}.json"
        rows.append({**_cpu_row(
            name=f"join-{int(mark):03d}", kind="join", checkout=checkout, tag=tag,
            python=python, module_argv=[
                "-m", "prismaquant.stage_a_chain_split", "--output-root", str(output_root),
                "--boundary", str(int(mark)), "--receipt", str(receipt)]),
            "boundary": int(mark), "receipt": str(receipt)})
        band = round_dir / "bands" / f"band-{int(mark):03d}.json"
        rows.append({**_cpu_row(
            name=f"band-{int(mark):03d}", kind="band", checkout=checkout, tag=tag,
            python=python, module_argv=[
                "-m", "prismaquant.joint_adjoint_band", "--request", band_request["path"],
                "--request-sha256", band_request["sha256"], "--boundary", str(int(mark)),
                "--output", str(band)]),
            "boundary": int(mark), "band": str(band)})
    bands = [*(reference["path"] for reference in band_references),
             *(row["band"] for row in rows if row["kind"] == "band")]
    rows.append({**_cpu_row(
        name="band-set", kind="band-set", checkout=checkout, tag=tag, python=python,
        module_argv=["-c", _BAND_SET, *bands]), "bands": bands})
    document = {
        "schema": ROUND_SCHEMA, "checkout": str(checkout), "commit": commit,
        "implementation_sha256": running, "implementation_declaration": declaration,
        "output_root": str(output_root), "from": boundary, "through": through,
        "chain_state_sha256": package["chain_state"]["sha256"],
        "boundaries": list(package["boundaries"]), "ranges": ranges, "labels": labels,
        "digest_layer": digest_layer, "tag": tag,
        "pins": {"split_package": _pin(package_path), "spec": _pin(spec),
                 "prefetch_override": _pin(prefetch_override),
                 "base_template": _pin(base_template),
                 "band_references": [_pin(ref["path"]) for ref in band_references]},
        "campaign": dict(campaign), "rows": rows}
    _write_json(round_dir / ROUND_NAME, document)
    return document


#: The band set check: every band shares one run header (``band_set``).
_BAND_SET = ("import json, sys; "
             "from prismaquant.joint_adjoint_slices import band_set, stage_a_run_header, "
             "stage_a_run_header_sha256; "
             "bands = [json.load(open(p)) for p in sys.argv[1:]]; "
             "indexed = band_set(bands); "
             "print(json.dumps({'boundaries': sorted(indexed), 'run_header_sha256': "
             "stage_a_run_header_sha256(stage_a_run_header(bands[0]))}))")


def load_round(round_dir) -> dict:
    """The sealed round, every pinned input unchanged."""
    document = json.loads((Path(round_dir) / ROUND_NAME).read_text())
    if document.get("schema") != ROUND_SCHEMA:
        raise SplitDispatchRefused(f"{round_dir} holds no sealed split round")
    pins = document["pins"]
    for pin in [pins["split_package"], pins["spec"], pins["prefetch_override"],
                pins["base_template"], *pins["band_references"],
                *(row[key] for row in document["rows"]
                  for key in ("data_manifest", "template") if key in row)]:
        if _sha(pin["path"]) != pin["sha256"]:
            raise SplitDispatchRefused(f"{pin['path']} changed since the round was sealed")
    return document


# -- ordering ------------------------------------------------------------------------

def submission_path(round_dir, name) -> Path:
    return Path(round_dir) / "submissions" / f"{name}.json"


def pb_ending(round_dir, name) -> dict:
    """A submitted row's PrismaBuild ending: refuses unless it executed with exit 0."""
    path = submission_path(round_dir, name)
    if not path.is_file():
        raise SplitDispatchRefused(f"row {name} was never submitted")
    detach = json.loads(path.read_text()).get("detach")
    if not detach:
        raise SplitDispatchRefused(f"row {name}'s submission printed no detach line")
    done = Path(detach["done"])
    if not done.is_file():
        failed = Path(detach["failed"])
        raise SplitDispatchRefused(
            f"row {name} ({detach['action_key'][:12]}) has "
            f"{'failed' if failed.is_file() else 'not finished'}")
    record = json.loads(done.read_text())
    returncode = (record.get("detail") or {}).get("returncode")
    if record.get("status") != "executed" or returncode != 0:
        raise SplitDispatchRefused(
            f"row {name} ended {record.get('status')} with exit {returncode}")
    return record


def require_ready(document, round_dir, row) -> None:
    """Refuse ``row`` until every row it follows has finished (see the module doc)."""
    from prismaquant.joint_adjoint_checkpoints import adjoint_space, checkpoint_directory
    from prismaquant.stage_a_chain_split import quantum_directory, split_root

    space = adjoint_space(document["output_root"])
    kind = row["kind"]
    if kind == "prep":
        from prismaquant.stage_a_chain_resume import chain_state_path
        if _sha(chain_state_path(space)) != document["chain_state_sha256"]:
            raise SplitDispatchRefused("the run's chain state is not the one the round pins")
        return
    if kind == "quantum":
        pb_ending(round_dir, "prep")
        preps = sorted((split_root(space) / "preps").glob("resume-*.json"))
        receipt = json.loads(preps[-1].read_text()) if preps else None
        stamp = None if receipt is None else receipt["resume"].get("split")
        if (stamp is None or (stamp["from"], stamp["through"], stamp["ranges"])
                != (document["from"], document["through"], document["ranges"])):
            raise SplitDispatchRefused("the latest prep receipt is not this round's")
        if receipt["resume"]["implementation_sha256"] != document["implementation_sha256"]:
            raise SplitDispatchRefused("the round's prep ran another implementation")
        return
    if kind == "join":
        for label in document["labels"]:
            pb_ending(round_dir, label)
            receipt = quantum_directory(space) / f"{label}.json"
            if not receipt.is_file():
                raise SplitDispatchRefused(f"quantum {label} left no receipt")
        return
    if kind == "band":
        pb_ending(round_dir, f"join-{row['boundary']:03d}")
        if not (checkpoint_directory(space, row["boundary"]) / "checkpoint.json").is_file():
            raise SplitDispatchRefused(f"checkpoint {row['boundary']} is not joined")
        return
    if kind == "band-set":
        for other in document["rows"]:
            if other["kind"] == "band":
                pb_ending(round_dir, other["name"])
        return
    raise SplitDispatchRefused(f"unknown row kind {kind}")


def submit(round_dir, names, *, run=subprocess.run) -> list[dict]:
    """Submit the named rows in order, each once its predecessors finished."""
    round_dir = Path(round_dir)
    document = load_round(round_dir)
    if checkout_commit(document["checkout"]) != document["commit"]:
        raise SplitDispatchRefused(f"{document['checkout']} is not at {document['commit']}")
    if implementation_sha256(document["checkout"]) != document["implementation_sha256"]:
        raise SplitDispatchRefused("the checkout is not the sealed implementation")
    rows = {row["name"]: row for row in document["rows"]}
    results = []
    for name in names:
        if name not in rows:
            raise SplitDispatchRefused(f"the round has no row {name}")
        row = rows[name]
        if submission_path(round_dir, name).exists():
            raise SplitDispatchRefused(f"row {name} was already submitted")
        require_ready(document, round_dir, row)
        completed = run(row["argv"], capture_output=True, text=True)
        detach = None
        for line in (completed.stdout or "").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and "action_key" in value:
                detach = value
        result = {"name": name, "argv": row["argv"], "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr, "detach": detach}
        _write_json(submission_path(round_dir, name), result)
        results.append(result)
        if completed.returncode != 0 or detach is None:
            raise SplitDispatchRefused(
                f"row {name} was not submitted (exit {completed.returncode}): "
                f"{(completed.stderr or '').strip()[-400:]}")
    return results


# -- the digest tripwire -------------------------------------------------------------

def compare_digests(document, baseline_path) -> dict:
    """Each landed quantum's payload digests at the digest layer against a baseline.

    The baseline maps ``cotangent-{probe}-{batch}-at-{layer}.pt`` to the
    payload digest (``tensor_payload_sha256``) the reference run wrote.
    """
    from prismaquant.joint_adjoint_checkpoints import adjoint_space
    from prismaquant.stage_a_chain_split import quantum_directory

    baseline = json.loads(Path(baseline_path).read_text())
    baseline = baseline.get("payload_sha256", baseline)
    space = adjoint_space(document["output_root"])
    layer = document["digest_layer"]
    report = {"layer": layer, "quanta": {}, "mismatches": 0, "compared": 0}
    for label in document["labels"]:
        path = quantum_directory(space) / f"{label}.digests.json"
        if not path.is_file():
            report["quanta"][label] = {"status": "pending"}
            continue
        digests = json.loads(path.read_text())
        if digests["layer"] != layer:
            raise SplitDispatchRefused(f"{path} is layer {digests['layer']}, not {layer}")
        bad = []
        for key, digest in sorted(digests["payload_sha256"].items()):
            want = baseline.get(f"cotangent-{key}-at-{layer}.pt")
            if want != digest:
                bad.append({"entry": f"cotangent-{key}-at-{layer}.pt", "got": digest,
                            "want": want})
        report["quanta"][label] = {"status": "match" if not bad else "mismatch",
                                   "compared": len(digests["payload_sha256"]),
                                   "mismatches": bad[:8], "mismatch_count": len(bad)}
        report["compared"] += len(digests["payload_sha256"])
        report["mismatches"] += len(bad)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal", help="write the round directory; submit nothing")
    seal.add_argument("--round-dir", required=True)
    seal.add_argument("--checkout", required=True)
    seal.add_argument("--split-package", required=True)
    seal.add_argument("--campaign-record", required=True,
                      help="a layer record whose campaign block names the plan and prepared")
    seal.add_argument("--spec", required=True)
    seal.add_argument("--prefetch-override", required=True)
    seal.add_argument("--base-template", required=True)
    seal.add_argument("--tier", required=True)
    seal.add_argument("--template-prefix", required=True)
    seal.add_argument("--artifact-budget-bytes", type=int, required=True)
    seal.add_argument("--chain-batch-size", type=int, required=True)
    seal.add_argument("--chain-probe-fusion", choices=("on", "off"), required=True)
    seal.add_argument("--seconds-per-layer", type=float, required=True,
                      help="the single owner's measured roll seconds per layer")
    seal.add_argument("--digest-layer", type=int, default=None)
    seal.add_argument("--band-request", required=True)
    seal.add_argument("--band-request-sha256", required=True)
    seal.add_argument("--band-reference", action="append", default=[],
                      help="an earlier band of the run the band set must accept")
    seal.add_argument("--python", required=True, help="the CPU rows' interpreter")
    seal.add_argument("--tag", default="gb10")
    for name in ("plan", "status"):
        sub.add_parser(name).add_argument("--round-dir", required=True)
    go = sub.add_parser("submit")
    go.add_argument("--round-dir", required=True)
    go.add_argument("rows", nargs="+", help="row names; 'quanta' names every quantum")
    compare = sub.add_parser("compare")
    compare.add_argument("--round-dir", required=True)
    compare.add_argument("--baseline", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            document = seal_round(
                round_dir=args.round_dir, checkout=Path(args.checkout).resolve(),
                split_package=args.split_package,
                campaign=json.loads(Path(args.campaign_record).read_text())["campaign"],
                spec=args.spec, prefetch_override=args.prefetch_override,
                base_template=args.base_template, tier=args.tier,
                template_prefix=args.template_prefix,
                artifact_budget_bytes=args.artifact_budget_bytes,
                chain_regime={"chain_batch_size": args.chain_batch_size,
                              "chain_probe_fusion": args.chain_probe_fusion},
                seconds_per_layer=args.seconds_per_layer, digest_layer=args.digest_layer,
                band_request={"path": args.band_request, "sha256": args.band_request_sha256},
                band_references=[{"path": path} for path in args.band_reference],
                python=args.python, tag=args.tag)
            print(json.dumps({"round": str(Path(args.round_dir) / ROUND_NAME),
                              "rows": [row["name"] for row in document["rows"]]}))
        elif args.command == "plan":
            document = load_round(args.round_dir)
            for row in document["rows"]:
                print(json.dumps({"name": row["name"], "argv": row["argv"]}))
        elif args.command == "status":
            document = load_round(args.round_dir)
            for row in document["rows"]:
                try:
                    pb_ending(args.round_dir, row["name"])
                    state = "executed"
                except SplitDispatchRefused as exc:
                    state = str(exc)
                print(f"{row['name']}: {state}")
        elif args.command == "submit":
            document = load_round(args.round_dir)
            names = [label for name in args.rows
                     for label in (document["labels"] if name == "quanta" else [name])]
            for result in submit(args.round_dir, names):
                print(json.dumps({"name": result["name"], "detach": result["detach"]}))
        else:
            document = load_round(args.round_dir)
            report = compare_digests(document, args.baseline)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 4 if report["mismatches"] else 0
    except SplitDispatchRefused as exc:
        print(f"dispatch_stage_a_split: refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
