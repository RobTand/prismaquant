#!/usr/bin/env python3
"""Qualify GLM MTP layer 45's original capture and Stage A/B prices (PQ #2530).

Read-only. The audit re-derives, from the original files, every claim of the
qualification record and prints one JSON document on stdout. A PrismaBuild
action's CAS payload is exactly that stdout, so the committed record can be
checked byte for byte against its CAS receipt. Progress goes to stderr.

Scope: the 867 layer-45 units (864 routed experts, 3 shared dense units), the
v2 capture, the producer projection, Stage A (M3 campaign, rates r1024 and
r896), Stage B (M4 joint AURA prepare and run, same rates), the merged MTP
price, and every PrismaBuild action that produced them.

Sections (``--sections``): ``structure`` reads rosters, plans, prices and the
receipt chain through the existing owners. ``io`` re-reads the capture
entries, wire blobs and rendered shards. ``pb`` checks PrismaBuild outcomes
and CAS payloads. ``trees`` recovers the executing source trees from their
snapshots. ``refusals`` runs body-price admission on the MTP payloads.

The tool changes no budget, default, menu, runtime pin or gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "prismaquant.pq2530.mtp_layer45_audit.v1"
WORKSPACE = Path("/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/ws-mtp-20260925")
PB_ROOT = Path("/mnt/shared/prismabuild-fleet")
TESSERA_PIN_SRC = Path("/mnt/shared/tessera-pins/07bfcc0e9b7da13276938cb722bc7dcd893e6c63/src")
SECTIONS = ("structure", "io", "pb", "trees", "refusals")

CALIBRATION_TOKENS = Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
                          "exact-calibration-input-01/calibration_tokens.safetensors")
LAYER = 45
PREFIX = f"model.language_model.layers.{LAYER}.mlp."
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
ROUTED_NAMES = frozenset(f"{PREFIX}experts.{e}.{p}" for e in range(288) for p in PROJECTIONS)
SHARED_NAMES = frozenset(f"{PREFIX}shared_experts.{p}" for p in PROJECTIONS)
GEOMETRY = {"down_proj": (4096, 2048), "gate_proj": (2048, 4096), "up_proj": (2048, 4096)}
RATES = ("r1024", "r896")
RUNGS = {"r1024": ("TESSERA_BF16_K1_R1024", "TESSERA_E4M3_K1_R1024"),
         "r896": ("TESSERA_BF16_K1_R896", "TESSERA_E4M3_K1_R896")}
RUNG_ORDER = RUNGS["r1024"] + RUNGS["r896"]
CAPTURE_POLICY = {"max_buffer_bytes": 629147557, "max_scratch_bytes": 2097152,
                  "schema": "prismaquant.verified_activation_load.v1"}
MTP_BODY_REFUSAL = "joint AURA requires matching aura/joint provenance"
STAGE = "Tessera campaign"
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
#: The scope of the recorded M6 selection and of action 370bdce3b3c2 (tools/reselect_mtp_fixed.py).
SERVING_SCOPE = ("--tessera-platform", "sm_121", "--tessera-runtime-image",
                 "localhost/prismaquant/spark-vllm-nccl230@sha256:"
                 "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
                 "--tessera-execution-mode", "eager", "--tessera-residency", "resident")
TARGET_PROFILE = "glm_packed_research_sm121"


#: Qualifying executions outside the M3 rows (those come from each receipts.json).
ACTIONS = {
    "capture-v2": "7c78d20519be1a13b57daaf9c323ab08c0819c26360d9311c70bf7e995de7005",
    "projection": "89369de5afb2457ff188fff4692661a1f0acdce540e3cda626ec32078168a3c2",
    "final-hidden": "83dc1e8371bb8cc1efe1c5262d83858a2396db456cb7e769cb4e498c5a106667",
    "m4-r1024-prepare": "c3dc42c2ae32773fc516701dbf736b9b72df7e36c5ab1b4323c0bbd5f3db3f4b",
    "m4-r1024-run": "6d839e49e7c3d10e6b31665fbc1f048fd550753648cbd24746afd3d6fa551865",
    "m4-r896-prepare": "64969134bf06a2eebcaeee8963d76acb66b87f3628c22d25f59ce4b3dea416ad",
    "m4-r896-run": "b91a4554c0034915c3d133c073502b54602249e8e59e13d91dc124bf492a0898",
}
#: Support: the owner's receipt join ran on the real merged payload (PQ #1413).
SUPPORT_ACTIONS = {
    "reselect-owner-join": "370bdce3b3c2b59c19bc6a01fedcae9ee257a9826d4ace3fa9a0b639691d7213",
}
#: History that does not qualify: superseded or never executed.
EXCLUDED = {
    "capture-v1-superseded": "d1e1681bd0711f6801b134d752af4cc9ab77fe292f7c8036238e69b7472708b9",
    "m4-r896-attempt1-withdrawn": "1f51251ce30ed6bcd8fdb2a8ab6db5c9c69560380cb3415c774a5e3f39fb9aba",
}
#: Failed M3 rows of the two earlier r1024 attempts (their own receipts.json files).
EXCLUDED_M3_ATTEMPTS = {
    "m3-r1024-attempt1-derivative-guard": "m3/r1024/workspace-attempt1-derivative-guard/receipts.json",
    "m3-r1024-attempt2-no-datasets": "m3/r1024/workspace-attempt2-no-datasets/receipts.json",
}


# --------------------------------------------------------------------------- helpers
def log(message, *, always=False):
    """Progress goes to stderr only on request, so a passing run's stdout is exactly its JSON record."""
    if always or os.environ.get("PQ2530_AUDIT_VERBOSE") == "1":
        print(f"[audit] {message}", file=sys.stderr, flush=True)


def sha256_bytes(raw) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def unit_class(name):
    return "routed" if ".mlp.experts." in name else "shared"


def projection_of(name):
    return name.rsplit(".", 1)[-1]


class Ledger:
    """Named checks. A failed check never stops the audit: every defect is reported."""

    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:600]})
        if not ok:
            log(f"FAIL {name}: {str(detail)[:300]}", always=True)
        return bool(ok)


class Inputs:
    """Every file the audit parses, with the digest of the bytes it parsed."""

    def __init__(self):
        self.rows = {}

    def read(self, role, path) -> bytes:
        raw = Path(path).read_bytes()
        self.rows[(role, str(path))] = {"role": role, "path": str(path),
                                        "sha256": sha256_bytes(raw), "bytes": len(raw)}
        return raw

    def json(self, role, path):
        return json.loads(self.read(role, path))

    def pickle(self, role, path):
        return pickle.loads(self.read(role, path))

    def bind(self, role, path) -> str:
        """Record a file by streaming digest, for files the audit does not parse whole."""
        path = Path(path)
        digest = sha256_file(path)
        self.rows[(role, str(path))] = {"role": role, "path": str(path), "sha256": digest,
                                        "bytes": path.stat().st_size}
        return digest

    def sha(self, role, path) -> str:
        return self.rows[(role, str(path))]["sha256"]

    def table(self):
        return [self.rows[key] for key in sorted(self.rows)]


class Progress:
    """PrismaBuild progress-v1 reporting; a no-op outside an admitted action."""

    def __init__(self):
        import runpy
        import threading
        helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
        self._commit = runpy.run_path(helper)["commit"] if helper else None
        self._lock = threading.Lock()
        self.done = 0

    def advance(self, phase, units=1):
        with self._lock:
            self.done += units
            if self._commit is not None:
                self._commit(self.done, phase)


class Identities:
    """One identity value per stage, so a kind that should be single-valued shows any split."""

    def __init__(self):
        self.seen = {}

    def add(self, kind, where, value):
        if value is not None:
            self.seen.setdefault(kind, {}).setdefault(json.dumps(value, sort_keys=True), []).append(where)

    def table(self, ledger, allowed_split=()):
        rows = []
        for kind in sorted(self.seen):
            values = [{"value": json.loads(value), "where": sorted(set(where))}
                      for value, where in sorted(self.seen[kind].items())]
            split = len(values) > 1
            ok = not split or kind in allowed_split
            ledger.check(f"identity.{kind}", ok, f"{len(values)} distinct value(s)"
                         + (" (recorded split, see seals)" if split and ok else ""))
            rows.append({"kind": kind, "distinct": len(values), "split_allowed": kind in allowed_split,
                         "values": values})
        return rows


def flat_diff(left, right, path=""):
    """Leaf paths where two JSON values differ."""
    out = {}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            out.update(flat_diff(left.get(key), right.get(key), f"{path}/{key}"))
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for index, (a, b) in enumerate(zip(left, right)):
            out.update(flat_diff(a, b, f"{path}[{index}]"))
    elif left != right:
        out[path] = [left, right]
    return out


def roster_row(ledger, name, keys, expected, *, scope="census"):
    """Compare one roster with the authenticated census (or a named subset of it)."""
    keys, expected = set(keys), set(expected)
    missing, extra = sorted(expected - keys), sorted(keys - expected)
    ledger.check(f"roster.{name}", not missing and not extra,
                 f"{len(keys)} units against {len(expected)}; missing {len(missing)}, extra {len(extra)}")
    return {"name": name, "scope": scope, "count": len(keys), "expected": len(expected),
            "equals": not missing and not extra, "missing": missing[:3], "extra": extra[:3]}


# --------------------------------------------------------------------------- PrismaBuild
def resolve_key(pb_root, key):
    """A full action key from a prefix of at least twelve characters."""
    if len(key) == 64:
        return key
    found = sorted(path.stem for path in (Path(pb_root) / "cas" / "requests" / key[:2]).glob(f"{key}*.json"))
    if len(found) != 1:
        raise ValueError(f"action prefix {key} names {len(found)} requests")
    return found[0]


def _spec_of(request):
    """The container spec and inner command of a campaign row's sealed shell command."""
    argv = (request.get("task") or {}).get("argv") or []
    text = argv[-1] if argv else ""
    start = text.find("--spec '")
    if start < 0:
        return None, None
    start += len("--spec '")
    end = text.find("' -- ", start)
    try:
        spec = json.loads(text[start:end])
    except ValueError:
        return None, None
    return spec, text[end + len("' -- "):].split(" 2>&1")[0]


def verify_action(pb_root, key, role) -> dict:
    """One action's terminal outcome and CAS receipt, checked against each other.

    Chain: queue ``done`` record -> adopted attempt record -> stdout log (hash)
    -> CAS payload (the stdout before the worker's receipt line) -> CAS receipt
    (self hash, result, manifest hash) -> sealed request (canonical hash).
    """
    pb_root = Path(pb_root)
    key = resolve_key(pb_root, key)
    queue, cas = pb_root / "pb-queue", pb_root / "cas"
    row = {"role": role, "key": key, "problems": []}

    def problem(text):
        row["problems"].append(text)

    row["queue_state"] = [name for name in ("done", "failed", "ready", "claimed")
                          if (queue / name / f"{key}.json").is_file()]
    if (queue / "done" / f"{key}.json").is_file():
        done = json.loads((queue / "done" / f"{key}.json").read_bytes())
        row["resources"] = done.get("resources")
        row["tags"] = done.get("tags")
    if row["queue_state"] != ["done"]:
        problem(f"queue state is {row['queue_state']}, not done")
    adopted = None
    for path in sorted((queue / "attempts" / key).glob("*/*.json")):
        record = json.loads(path.read_bytes())
        if record.get("disposition") == "done":
            adopted = record
    if adopted is None:
        problem("no attempt with disposition done")
        return row
    detail = adopted.get("detail") or {}
    profile = detail.get("resource_profile") or {}
    row.update({"host": adopted.get("claimed_host"), "status": adopted.get("status"),
                "returncode": detail.get("returncode"), "disposition": adopted.get("disposition"),
                "elapsed_s": round(float(detail.get("elapsed_s") or 0.0), 1),
                "wall_seconds": round(float(profile.get("wall_seconds") or 0.0), 1),
                "memory_peak_bytes": (profile.get("scope") or {}).get("memory_peak_bytes")})
    if adopted.get("status") != "executed" or detail.get("returncode") != 0:
        problem(f"attempt status {adopted.get('status')!r} returncode {detail.get('returncode')!r}")
    logs = (adopted.get("logs") or {}).get("stdout") or {}
    log_path = queue / logs.get("path", "missing")
    raw = log_path.read_bytes() if log_path.is_file() else b""
    row["stdout"] = {"sha256": sha256_bytes(raw), "bytes": len(raw)}
    if (row["stdout"]["sha256"], row["stdout"]["bytes"]) != (logs.get("sha256"), logs.get("bytes")):
        problem("stdout log differs from the attempt record")
    marker = raw.rfind(b'\n{"local_result_claim_sha256"')
    if marker < 0:
        problem("stdout has no worker receipt line")
        return row
    payload, trailer = raw[:marker + 1], json.loads(raw[marker + 1:])
    row["payload"] = {"sha256": sha256_bytes(payload), "bytes": len(payload)}
    receipt_path = cas / "actions" / "v3" / key[:2] / f"{key}.json"
    receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
    body = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    result = receipt.get("result") or {}
    row["receipt"] = {"sha256": receipt.get("receipt_sha256"), "schema": receipt.get("schema"),
                      "result_sha256": result.get("sha256"), "result_bytes": result.get("bytes"),
                      "recomputed_sha256": sha256_bytes(canonical(body)) if receipt else None}
    if not receipt or receipt.get("action_key") != key:
        problem("CAS receipt is absent or names another action")
    if row["receipt"]["sha256"] != row["receipt"]["recomputed_sha256"]:
        problem("CAS receipt does not hash to its own receipt_sha256")
    if (row["payload"]["sha256"], row["payload"]["bytes"]) != (result.get("sha256"), result.get("bytes")):
        problem("stdout payload differs from the CAS receipt result")
    if (trailer.get("receipt") or {}).get("receipt_sha256") != receipt.get("receipt_sha256"):
        problem("worker receipt line names another receipt")
    if trailer.get("status") != "published":
        problem(f"worker receipt status {trailer.get('status')!r}")
    blob_path = cas / "blobs" / str(result.get("sha256", "00"))[:2] / str(result.get("sha256"))
    blob = blob_path.read_bytes() if blob_path.is_file() else b""
    row["blob"] = {"sha256": sha256_bytes(blob), "bytes": len(blob)}
    if (row["blob"]["sha256"], row["blob"]["bytes"]) != (result.get("sha256"), result.get("bytes")):
        problem("CAS blob differs from the receipt result")
    request_path = cas / "requests" / key[:2] / f"{key}.json"
    request = json.loads(request_path.read_bytes()) if request_path.is_file() else {}
    row["request"] = {"action_key": request.get("action_key"),
                      "manifest_sha256": sha256_bytes(canonical(request)) if request else None}
    if request.get("action_key") != key or row["request"]["manifest_sha256"] != receipt.get("action_manifest_sha256"):
        problem("sealed request differs from the receipt's action manifest")
    snapshot = (request.get("params") or {}).get("checkout_snapshot") or {}
    row["snapshot"] = {"commit": snapshot.get("commit"), "parent": snapshot.get("parent"),
                       "bundle_sha256": (snapshot.get("input") or {}).get("sha256")}
    spec, inner = _spec_of(request)
    if spec is not None:
        mounts = (spec.get("container") or {}).get("mounts", [])
        row["spec"] = {"dev_mode": (spec.get("env") or {}).get("PRISMAQUANT_DEV_MODE"),
                       "container_content_sha256": (spec.get("container") or {}).get("content_sha256"),
                       "container_image": (spec.get("container") or {}).get("image"),
                       "container_admission_reference": spec.get("container_admission_reference"),
                       "tessera_pin": next((m["source"] for m in mounts if m.get("target") == "/tessera-pin"), None),
                       "inner_command": inner}
    row["ok"] = not row["problems"]
    return row


def excluded_state(pb_root, key, role):
    """State of an action that does not qualify: recorded, never counted."""
    pb_root = Path(pb_root)
    key = resolve_key(pb_root, key)
    queue = pb_root / "pb-queue"
    states = [name for name in ("done", "failed", "ready", "claimed", "intent")
              if (queue / name / f"{key}.json").is_file()]
    outcomes = []
    for path in sorted((queue / "attempts" / key).glob("*/*.json")):
        record = json.loads(path.read_bytes())
        outcomes.append({"status": record.get("status"), "disposition": record.get("disposition"),
                         "returncode": (record.get("detail") or {}).get("returncode")})
    return {"role": role, "key": key, "queue_state": states, "attempts": outcomes,
            "cas_receipt": (pb_root / "cas" / "actions" / "v3" / key[:2] / f"{key}.json").is_file()}


def audit_actions(pb_root, ledger, ws, extra=()):
    """Verify every qualifying action; list, without counting, the ones that do not qualify."""
    named = dict(ACTIONS)
    for rate in RATES:
        receipts = json.loads((ws / "m3" / rate / "workspace" / "receipts.json").read_bytes())
        for index, row in enumerate(receipts["rows"]):
            named[f"m3-{rate}-row-{index:04d}"] = resolve_key(pb_root, row["key"])
    named.update(SUPPORT_ACTIONS)
    rows = [verify_action(pb_root, key, role) for role, key in named.items()]
    rows += [verify_action(pb_root, key, role) for role, key in extra]
    for row in rows:
        ledger.check(f"action.{row['role']}", row.get("ok", False),
                     "; ".join(row["problems"]) or f"{row['key'][:12]} rc {row.get('returncode')} on {row.get('host')}")
    excluded = [excluded_state(pb_root, key, role) for role, key in EXCLUDED.items()]
    for role, relative in EXCLUDED_M3_ATTEMPTS.items():
        for item in json.loads((ws / relative).read_bytes())["rows"]:
            try:
                excluded.append(excluded_state(pb_root, item["key"], role))
            except ValueError:
                excluded.append({"role": role, "key": item["key"], "queue_state": [], "attempts": [],
                                 "cas_receipt": False, "note": "sealed request no longer retained"})
    qualifying = {row["key"] for row in rows}
    ledger.check("excluded.attempts_are_not_qualifying",
                 not qualifying & {item["key"] for item in excluded} and len(excluded) == 8,
                 f"{len(excluded)} excluded attempts, none among the {len(qualifying)} qualifying actions")
    return rows, excluded


# --------------------------------------------------------------------------- dev-mode ledger
DEV_KINDS = (
    ("prepared_plan", re.compile(r"seal prepared plan_sha256 differs")),
    ("prepared_implementation", re.compile(r"seal prepared implementation_sha256 differs")),
    ("checkpoint_seal", re.compile(r"campaign checkpoint seal not recomputed")),
    ("projection_shape", re.compile(r"seal joint projection qualified shape differs")),
    ("source_rehash", re.compile(r"source rehash of \d+ bytes")),
)


def dev_lines(pb_root, key):
    """Every ``[DEV-MODE]`` line an action printed: a gate a certified run would have enforced."""
    found = []
    for path in sorted((Path(pb_root) / "pb-queue" / "attempts" / key).glob("*/*.stdout.*.log")):
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "[DEV-MODE]" in line:
                kind = next((name for name, pattern in DEV_KINDS if pattern.search(line)), "unclassified")
                found.append({"line": number, "kind": kind, "text": line[:420]})
    return found


# --------------------------------------------------------------------------- trees
def package_profile(root):
    """Aggregate and per-file digests of a package tree, framed as the producer stamp frames them."""
    root = Path(root)
    paths = sorted(p for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.relative_to(root).parts
                   and p.suffix not in {".pyc", ".pyo"})
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(4, "big") + encoded + len(payload).to_bytes(8, "big") + payload)
        files[relative] = sha256_bytes(payload)
    return digest.hexdigest(), files


def recover_tree(pb_root, bundle_sha256, scratch):
    """The package tree a PrismaBuild snapshot executed, extracted from its git bundle."""
    bundle = Path(pb_root) / "cas" / "blobs" / bundle_sha256[:2] / bundle_sha256
    if sha256_file(bundle) != bundle_sha256:
        raise RuntimeError(f"snapshot bundle {bundle_sha256} does not hash to its address")
    clone = Path(scratch) / f"clone-{bundle_sha256[:12]}"
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", str(bundle), str(clone)], check=True,
                   capture_output=True)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "origin/prismabuild-snapshot"], check=True,
                          capture_output=True, text=True).stdout.strip()
    target = Path(scratch) / f"tree-{bundle_sha256[:12]}"
    target.mkdir()
    archive = subprocess.run(["git", "-C", str(clone), "archive", "--format=tar", head, "prismaquant"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)
    return head, target / "prismaquant"


def audit_trees(pb_root, ledger, actions, ws):
    """Prove which source tree each Stage B action ran, and what differs between them."""
    by_role = {row["role"]: row for row in actions}
    out, profiles = {}, {}
    with tempfile.TemporaryDirectory(prefix="pq2530-trees-", dir=os.environ.get("TMPDIR", "/tmp")) as scratch:
        for rate in RATES:
            for phase in ("prepare", "run"):
                role = f"m4-{rate}-{phase}"
                results = json.loads((ws / "m4" / rate / phase / "results.json").read_bytes())
                stamp = results["dev_mode"]["producer_source_sha256"]
                bundle = by_role[role]["snapshot"]["bundle_sha256"]
                if bundle not in profiles:
                    head, root = recover_tree(pb_root, bundle, scratch)
                    aggregate, files = package_profile(root)
                    profiles[bundle] = (head, aggregate, files)
                head, aggregate, _files = profiles[bundle]
                out[role] = {"snapshot_commit": head, "snapshot_parent": by_role[role]["snapshot"]["parent"],
                             "bundle_sha256": bundle, "recorded_producer_source_sha256": stamp,
                             "reproduced_producer_source_sha256": aggregate}
                ledger.check(f"trees.{role}.digest", aggregate == stamp,
                             f"recorded {stamp[:12]} reproduced {aggregate[:12]}")
        prepare, run = (profiles[by_role[f"m4-r1024-{phase}"]["snapshot"]["bundle_sha256"]] for phase in ("prepare", "run"))
        changed = sorted(name for name in set(prepare[2]) | set(run[2]) if prepare[2].get(name) != run[2].get(name))
        out["r1024_prepare_vs_run"] = {"changed_files": changed, "files_compared": len(set(prepare[2]) | set(run[2]))}
    ledger.check("trees.r896_prepare_and_run_share_one_tree",
                 out["m4-r896-prepare"]["reproduced_producer_source_sha256"]
                 == out["m4-r896-run"]["reproduced_producer_source_sha256"], "r896 prepare and run")
    ledger.check("trees.r1024_split_is_identity_proof_code",
                 out["r1024_prepare_vs_run"]["changed_files"] == [
                     "cost_streaming.py", "tessera_calibration_cache.py", "tessera_joint_aura.py",
                     "tessera_source_digest_adoption.py"], str(out["r1024_prepare_vs_run"]["changed_files"]))
    return out


# --------------------------------------------------------------------------- structure: census and capture
def checkpoint_head(path):
    """``identity_sha256`` and the unit list of a campaign checkpoint, without parsing its identity block."""
    import mmap
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        seal_at = view.rfind(b'"identity_sha256": "')
        seal = view[seal_at + 20:seal_at + 84].decode()
        units_at = view.rfind(b'"units": [')
        tail = view[units_at + 9:].decode()
    units = json.loads(tail[:tail.rindex("]") + 1])
    return seal, [item["qname"] for item in units]


def census_facts(census, ledger):
    shapes, counts, maxima = census["unit_shapes"], census["counts"], census["max_abs"]
    units = set(shapes)
    routed = {name for name in units if unit_class(name) == "routed"}
    shared = units - routed
    ledger.check("census.units_867", len(units) == 867, f"{len(units)} units")
    ledger.check("census.routed_864", routed == ROUTED_NAMES, f"{len(routed)} routed units")
    ledger.check("census.shared_3", shared == SHARED_NAMES, f"{len(shared)} shared units")
    ledger.check("census.targets_match_classes", set(census["expert_targets"]) == routed
                 and set(census["dense_targets"]) == shared, "expert and dense targets")
    ledger.check("census.counts", set(counts) == units and all(type(v) is int and v > 0 for v in counts.values()),
                 "one positive integer count per unit")
    ledger.check("census.maxima", set(maxima) == units and all(
        isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in maxima.values()),
        "one finite positive maximum per unit")
    off = [name for name in units if tuple(shapes[name]) != GEOMETRY[projection_of(name)]]
    ledger.check("census.geometry", not off, f"{len(off)} units off the layer geometry")
    groups = census["anchor_groups"]
    ledger.check("census.anchor_groups_partition", sum(map(len, groups.values())) == 867
                 and set().union(*map(set, groups.values())) == units, f"{len(groups)} groups")
    routed_counts = [counts[name] for name in routed]
    return {"units": len(units), "routed": len(routed), "shared": len(shared),
            "count_min": min(routed_counts), "count_max": max(routed_counts),
            "shared_counts": sorted({counts[name] for name in shared}),
            "max_abs_min": min(maxima.values()), "max_abs_max": max(maxima.values()),
            "geometry": {name: list(shape) for name, shape in GEOMETRY.items()},
            "draw": {key: census[key] for key in ("nsamples", "seqlen", "seed")},
            "text_sha256": census["text_sha256"], "fit_ids_sha256": census["fit_ids_sha256"]}


def audit_projection(ws, inputs, ledger, census, ids):
    projection = inputs.json("producer-projection", ws / "m1/projection/mtp-projection.json")
    inputs.bind("producer-answer", ws / "m1/projection/producer-answer.json")
    inputs.bind("projection-run", ws / "m1/projection/projection-run.json")
    names, geometry = set(), Counter()
    for stack in projection["stacks"].values():
        for name, unit in stack.items():
            names.add(name)
            geometry[f"{projection_of(name)} {unit['rows']}x{unit['cols']}"] += 1
    ledger.check("projection.routed_864", names == ROUTED_NAMES, f"{len(names)} projected units")
    ledger.check("projection.geometry", dict(geometry) == {
        f"{p} {r}x{c}": 288 for p, (r, c) in GEOMETRY.items()}, str(dict(geometry)))
    ledger.check("projection.equals_census_copy", census["expert_projection"] == projection,
                 "the census carries the producer projection unchanged")
    files = projection["producer"]["source"]["files"]
    ids.add("source.shard_digests_sha256", "producer-projection", sha256_bytes(canonical(files)))
    return projection, {"units": len(names), "geometry": dict(sorted(geometry.items())),
                        "source_shards": len(files)}


def audit_calibration(inputs, ledger, census, ids):
    """The calibration draw: the token file, its shape and bytes, and the census it was prepared for."""
    raw = inputs.read("calibration-tokens", CALIBRATION_TOKENS)
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + length])
    metadata = header.pop("__metadata__")
    tensor, data = header["calibration_ids"], raw[8 + length:]
    provenance = json.loads(metadata["calibration_provenance"])
    ledger.check("calibration.draw_is_512_by_512_int64", tensor["dtype"] == "I64" and tensor["shape"] == [512, 512]
                 and len(data) == 512 * 512 * 8, f"{tensor['dtype']} {tensor['shape']} {len(data)} bytes")
    ledger.check("calibration.provenance_equals_the_census_draw", all(
        provenance[key] == census[key] for key in ("text_sha256", "fit_ids_sha256", "nsamples", "seqlen", "seed"))
        and provenance["fit_tokens"] == 262144 and metadata["original_census_sha256"]
        == census["mtp_extension"]["base_census"]["sha256"], "text, fit ids, draw size and seed; prepared for the base census")
    ids.add("calibration.artifact_sha256", "calibration-tokens-file", sha256_bytes(raw))
    ids.add("calibration.calibration_sha256", "calibration-tokens-file", sha256_bytes(data))
    ids.add("calibration.text_sha256", "calibration-tokens-file", provenance["text_sha256"])
    ids.add("calibration.fit_ids_sha256", "calibration-tokens-file", provenance["fit_ids_sha256"])
    return {"artifact_sha256": sha256_bytes(raw), "token_bytes_sha256": sha256_bytes(data),
            "shape": tensor["shape"], "dtype": tensor["dtype"], "provenance": provenance}


def audit_census_authentication(inputs, ledger, census, census_sha, report, projection):
    """What makes the census authenticated: its base census, the canonical capture, and the shards read."""
    extension = census["mtp_extension"]
    base, canonical_capture = extension["base_census"], extension["canonical_capture"]
    ledger.check("census.base_census_file", inputs.bind("base-census", base["path"]) == base["sha256"],
                 f"base census {base['sha256'][:12]}")
    ledger.check("census.canonical_capture_file", inputs.bind("canonical-capture-manifest", canonical_capture["path"])
                 == canonical_capture["sha256"], f"canonical capture manifest {canonical_capture['sha256'][:12]}")
    auth = report["source_authentication"]
    ledger.check("census.derivation_authenticated", auth["census_sha256"] == base["sha256"]
                 and auth["capture_manifest_sha256"] == canonical_capture["sha256"]
                 and auth["derived_census_sha256"] == [census_sha]
                 and auth["authentication"].startswith("fresh SHA256 through held")
                 and sum(f["bytes_hashed"] for f in auth["verified_files"]) == auth["payload_bytes_hashed"],
                 f"{len(auth['verified_files'])} files, {auth['payload_bytes_hashed']} payload bytes hashed")
    seal = projection["producer"]["source"]["files"]
    shards = [f for f in auth["verified_files"] if f["name"] in seal]
    ledger.check("census.authenticated_shards_equal_the_producer_seal",
                 bool(shards) and all(seal[f["name"]] == f["sha256"] for f in shards),
                 f"{len(shards)} shards read by the capture carry the producer's digests")
    return {"base_census_sha256": base["sha256"], "canonical_capture_sha256": canonical_capture["sha256"],
            "shards_read_by_capture": len(shards), "payload_bytes_hashed": auth["payload_bytes_hashed"]}


def audit_capture_manifest(ws, inputs, ledger, census, census_sha, ids):
    path = ws / "m1/v2/capture/capture_manifest.json"
    manifest = inputs.json("capture-manifest", path)
    report = inputs.json("capture-report", ws / "m1/v2/capture/capture-run.json")
    final_path = ws / "m1/final-hidden/manifest.json"
    final = inputs.json("final-hidden", final_path)
    inputs.bind("final-hidden-run", ws / "m1/final-hidden/final-hidden-run.json")
    ledger.check("capture.complete_v2", manifest["status"] == "complete"
                 and manifest["schema"] == "prismaquant.tessera_calibration_cache.v2", manifest["status"])
    ledger.check("capture.identity_names_census", manifest["identity"]["census_sha256"] == census_sha,
                 "capture identity census digest equals the authenticated census")
    keys = ("units", "identity_units", "routed_units", "projection_checked_units")
    ledger.check("capture.report_unit_counts", tuple(report[k] for k in keys) == (867, 867, 864, 864),
                 str({k: report[k] for k in keys}))
    ledger.check("capture.report_names_census_and_manifest",
                 report["census"]["sha256"] == census_sha
                 and report["capture"]["sha256"] == inputs.sha("capture-manifest", path), "report digests")
    routed_counts = [census["counts"][n] for n in ROUTED_NAMES]
    ledger.check("capture.report_routed_rows", (report["routed_rows"]["min"], report["routed_rows"]["max"])
                 == (min(routed_counts), max(routed_counts)), str(report["routed_rows"]))
    ledger.check("final_hidden.layer44_512_sequences", final["layer"] == 44 and len(final["records"]) == 512
                 and len(final["head_check"]) == 512, f"layer {final['layer']}, {len(final['records'])} records")
    ledger.check("final_hidden.census_binding", census["model_load_contract"]["input"]["sha256"]
                 == inputs.sha("final-hidden", final_path), "census names the final-hidden manifest digest")
    identity = manifest["identity"]
    ids.add("calibration.fit_ids_sha256", "capture-identity", identity["calibration"]["fit_ids_sha256"])
    ids.add("runtime.capture_torch", "capture-identity", identity["capture_runtime"]["torch"])
    row = roster_row(ledger, "capture_entries", manifest["entries"], census["unit_shapes"])
    return manifest, row, {
        "entries": len(manifest["entries"]), "status": manifest["status"],
        "final_hidden_sequences": len(final["records"]),
        "final_hidden_mean_top1": round(sum(h["top1"] for h in final["head_check"]) / 512, 4),
        "capture_runtime": identity["capture_runtime"], "source_authentication_bytes": report[
            "source_authentication"]["payload_bytes_hashed"]}


# --------------------------------------------------------------------------- structure: stage A
def audit_stage_a(ws, rate, inputs, ledger, facts, ids, *, recompute_seal=True):
    """One M3 rate: the plan, its row receipts, the merged price, and every unit's own checkpoint."""
    from prismaquant import tessera_expert_projection as tep
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path

    census, census_sha, census_units = facts["census"], facts["census_sha"], set(facts["census"]["unit_shapes"])
    root = ws / "m3" / rate / "workspace"
    ledger.check(f"stage_a.{rate}.census_is_authenticated", inputs.bind(f"m3-{rate}-census", root / "census.json")
                 == census_sha, "the campaign's census file is byte-identical to the authenticated census")
    plan = inputs.json(f"m3-{rate}-plan", root / "plan.json")
    receipts = inputs.json(f"m3-{rate}-receipts", root / "receipts.json")
    inputs.bind(f"m3-{rate}-submitted-manifest", root / "manifest.submitted.json")
    cost_path = root / "merged" / "cost.pkl"
    cost = inputs.pickle(f"m3-{rate}-cost", cost_path)
    anchors_path = root / "merged" / "cost.anchors.json"
    inputs.bind(f"m3-{rate}-checkpoint", anchors_path)
    seal, listed = checkpoint_head(anchors_path)
    rungs = RUNGS[rate]
    rows = plan["rows"]
    ledger.check(f"stage_a.{rate}.plan_rows", plan["schema"] == "prismaquant.tessera_campaign_plan.v1"
                 and sorted(len(r["members"]) for r in rows) == [1, 2, 864], f"{len(rows)} campaign rows")
    ledger.check(f"stage_a.{rate}.row_receipts", receipts["returncode"] == 0 and len(receipts["rows"]) == len(rows)
                 and all(r["status"] in {"executed", "cache_hit"} and (r["status"] == "cache_hit" or r["rc"] == "0")
                         for r in receipts["rows"]), str([(r["key"], r["status"], r["rc"]) for r in receipts["rows"]]))
    prov = cost["provenance"]
    ledger.check(f"stage_a.{rate}.cost_header", cost["schema"] == "prismaquant.tessera_campaign_cost.v1"
                 and cost["currency"] == "output_mse_under_route_activation_contract"
                 and list(cost["formats"]) == list(rungs) and prov["stopped_early"] is False
                 and prov["cost_mode"] == "production-render-score" and prov["rate_band"] == [int(rate[1:])] * 2
                 and prov["nsamples"] == prov["seqlen"] == 512 and prov["no_admitted_rung"] == [],
                 f"formats {cost['formats']}")
    ledger.check(f"stage_a.{rate}.capture_binding", prov["calibration_cache"]["sha256"] == facts["capture_sha"],
                 "the campaign priced the v2 capture")
    ledger.check(f"stage_a.{rate}.projection_binding", prov[tep.PROJECTION_KEY] == facts["projection"],
                 "the campaign carries the producer projection unchanged")
    roster = roster_row(ledger, f"m3_{rate}_costs", cost["costs"], census_units)
    bad = []
    hessian = Counter()
    for name, by_rung in cost["costs"].items():
        if set(by_rung) != set(rungs):
            bad.append(f"{name}: rungs {sorted(by_rung)}")
            continue
        for rung, row in by_rung.items():
            h = row["hessian_identity"]
            hessian[(h["supplied"], h["applied"], h["token_count"], h["capture_sha256"])] += 1
            if not (row["output_mse_measured"] is True and row["cost_source"] == "tessera_campaign_measured"
                    and row["tessera_provenance"] == "measured" and row["tessera_body_rate_q256"] == int(rate[1:])
                    and math.isfinite(row["output_mse"]) and row["output_mse"] >= 0
                    and type(row["wire_bytes"]) is int and row["wire_bytes"] > 0):
                bad.append(f"{name}@{rung}")
    ledger.check(f"stage_a.{rate}.every_cell_measured", not bad, f"{len(bad)} cells off: {bad[:3]}")
    ledger.check(f"stage_a.{rate}.one_hessian_identity", len(hessian) == 1 and next(iter(hessian))[:3] == (
        True, True, 262144), str(dict(hessian))[:300])
    ids.add("calibration.hessian_capture_sha256", f"m3-{rate}-rows", next(iter(hessian))[3])
    wires = cost[tep.EXPERT_WIRES_KEY]
    wire_roster = roster_row(ledger, f"m3_{rate}_expert_wires", wires, ROUTED_NAMES, scope="routed")
    ledger.check(f"stage_a.{rate}.expert_wire_rungs", all(set(w) == set(rungs) for w in wires.values()),
                 "two wire receipts per routed unit")
    ledger.check(f"stage_a.{rate}.checkpoint_roster", set(listed) == census_units and len(listed) == 867,
                 f"{len(listed)} listed units")
    parts = anchors_path.with_name(anchors_path.name + ".parts")
    states, mismatch = {}, []
    for name in sorted(census_units):
        state = _load_unit(unit_path(parts, name), stage=STAGE, qname=name, identity_sha256=seal)
        states[name] = state
        anchors = {a["format_name"]: a for a in state["anchors"]}
        records = state["wire_records"]
        if set(anchors) != set(rungs) or set(records) != set(rungs):
            mismatch.append(f"{name}: rungs")
            continue
        for rung in rungs:
            a, r, c = anchors[rung], records[rung], cost["costs"][name][rung]
            if not (a["dloss"] == c["output_mse"] and a["wire_bytes"] == r["blob_bytes"] == c["wire_bytes"]
                    and a["hessian_applied"] is True and r["identity"]["unit"] == name
                    and r["identity"]["recipe"]["q256"] == int(rate[1:])
                    and (name not in ROUTED_NAMES or wires[name][rung] == r)):
                mismatch.append(f"{name}@{rung}")
    ledger.check(f"stage_a.{rate}.checkpoint_matches_price", not mismatch,
                 f"{len(mismatch)} cells off: {mismatch[:3]}")
    ids.add("producer.encoder_source_sha256", f"m3-{rate}-receipts", sorted({
        r["identity"]["encoder_source_sha256"] for s in states.values() for r in s["wire_records"].values()}))
    ids.add("producer.encoder_fixture_id", f"m3-{rate}-receipts", sorted({
        r["identity"]["encoder_fixture_id"] for s in states.values() for r in s["wire_records"].values()}))
    ids.add("source.model_path", f"m3-{rate}-cost", prov["model"])
    recomputed = recompute_checkpoint_seal(anchors_path, rate, seal, ledger, ids) if recompute_seal else None
    return {"cost": cost, "wires": wires, "states": states, "seal": seal, "seal_recomputed": recomputed,
            "wire_dir": Path(prov["wire_dir"]),
            "plan": plan, "receipts": receipts, "cost_path": cost_path, "anchors_path": anchors_path,
            "rows": [{"row_id": r["row_id"], "groups": r["groups"], "members": len(r["members"])} for r in rows],
            "rosters": [roster, wire_roster]}


def recompute_checkpoint_seal(path, rate, declared, ledger, ids):
    """The canonical seal of a merged campaign checkpoint, which dev mode skips (about 14 GiB of parse)."""
    import gc

    from prismaquant.digests import canonical_json_sha256_normalized
    from prismaquant.interned_json import load_json_file

    manifest = load_json_file(path)
    identity = manifest["identity"]
    recomputed = canonical_json_sha256_normalized(identity, where="PQ #2530 audit")
    ledger.check(f"stage_a.{rate}.checkpoint_seal_recomputed", recomputed == declared == manifest["identity_sha256"],
                 f"declared {declared[:12]} recomputed {recomputed[:12]}")
    where = f"m3-{rate}-checkpoint-identity"
    ids.add("producer.encoder_source_sha256", where, [identity["encoder_source_sha256"]])
    ids.add("producer.m3_campaign_source_sha256", where, identity["prismaquant_source_sha256"])
    ids.add("calibration.fit_ids_sha256", where, identity["calibration"]["fit_ids_sha256"])
    ids.add("source.shard_digests_sha256", where, sha256_bytes(canonical(identity["expert_projection"]["source"]["files"])))
    del manifest, identity
    gc.collect()
    return recomputed


def pinned_encoder_source(ledger, ids):
    """The producer's encoder source digest, recomputed from the pinned Tessera tree that encoded the wires."""
    environment = dict(os.environ, PYTHONPATH=str(TESSERA_PIN_SRC))
    digest = subprocess.run([sys.executable, "-c", "from tessera import cached_unit; print(cached_unit.encoder_source_sha256())"],
                            check=True, capture_output=True, text=True, env=environment).stdout.strip()
    ids.add("producer.encoder_source_sha256", "pinned-tree-07bfcc0e", [digest])
    pin = json.loads((TESSERA_PIN_SRC.parent / ".pinned-source.json").read_bytes())
    ledger.check("producer.pin_tree_is_commit_07bfcc0e", pin["commit"] == "07bfcc0e9b7da13276938cb722bc7dcd893e6c63",
                 f"pinned tree {pin['commit'][:12]}, {pin['files']} files, encoder source {digest[:12]}")
    return {"commit": pin["commit"], "tree_sha256": pin["tree_sha256"], "encoder_source_sha256": digest}


# --------------------------------------------------------------------------- structure: stage B
def audit_stage_b(ws, rate, inputs, ledger, facts, a, ids):
    """One M4 rate: plan bindings, prepared evidence, results and the Stage B price."""
    census, census_sha = facts["census"], facts["census_sha"]
    census_units = set(census["unit_shapes"])
    root = ws / "m4" / rate
    rungs = RUNGS[rate]
    plan_path = root / "plan.json"
    plan = inputs.json(f"m4-{rate}-plan", plan_path)
    run_plan_path = root / "plan-run.json" if (root / "plan-run.json").is_file() else plan_path
    run_plan = inputs.json(f"m4-{rate}-plan-run", run_plan_path) if run_plan_path != plan_path else plan
    prepared = inputs.json(f"m4-{rate}-prepared", root / "prepare" / "prepared.json")
    prepare_results = inputs.json(f"m4-{rate}-prepare-results", root / "prepare" / "results.json")
    run_results = inputs.json(f"m4-{rate}-run-results", root / "run" / "results.json")
    price_path = root / "run" / "joint-cost.pkl"
    price = inputs.pickle(f"m4-{rate}-price", price_path)
    production_path = root / "prepare" / "production.pkl"
    production_sha = inputs.bind(f"m4-{rate}-production-cache", production_path)
    identity_files = [inputs.bind(f"m4-{rate}-{phase}-source-identity", root / phase / "source-identity.json")
                      for phase in ("prepare", "run")]
    bindings = plan["inputs"]
    expected = {"merged_cost": inputs.sha(f"m3-{rate}-cost", a["cost_path"]),
                "merged_checkpoint": inputs.sha(f"m3-{rate}-checkpoint", a["anchors_path"]),
                "census": census_sha,
                "campaign_plan": inputs.sha(f"m3-{rate}-plan", a["cost_path"].parents[1] / "plan.json"),
                "campaign_receipts": inputs.sha(f"m3-{rate}-receipts", a["cost_path"].parents[1] / "receipts.json")}
    ledger.check(f"stage_b.{rate}.plan_binds_stage_a", all(bindings[k]["sha256"] == v for k, v in expected.items()),
                 "plan inputs equal the Stage A files by digest")
    ledger.check(f"stage_b.{rate}.plan_binds_capture_and_calibration",
                 plan["canonical_capture"]["sha256"] == facts["capture_sha"]
                 and plan["calibration_input"]["sha256"] == prepared["calibration_input"]["artifact_sha256"],
                 "capture manifest and calibration tokens")
    execution = plan["execution"]
    ledger.check(f"stage_b.{rate}.plan_draw_and_probes", (execution["n_calib_samples"], execution["calib_seqlen"],
                 execution["n_probes"], execution["seed_base"], execution["token_scope"]) == (512, 512, 4, 7000, "all")
                 and plan["source_scope"] == "mtp" and bindings["required_source_units"] == 867, str(
                     {k: execution[k] for k in ("n_probes", "seed_base", "token_scope")}))
    plan_diff = flat_diff(plan, run_plan)
    plan_sha, run_plan_sha = inputs.sha(f"m4-{rate}-plan", plan_path), inputs.sha(
        f"m4-{rate}-plan-run" if run_plan_path != plan_path else f"m4-{rate}-plan", run_plan_path)
    ledger.check(f"stage_b.{rate}.plans_bind_their_passes", prepare_results["plan_sha256"] == plan_sha
                 and run_results["plan_sha256"] == run_plan_sha and prepared["plan_sha256"] == plan_sha,
                 f"prepare {plan_sha[:12]} run {run_plan_sha[:12]}")
    cache_binding = run_plan.get("source_identity_cache")
    ledger.check(f"stage_b.{rate}.run_plan_binds_the_identity_cache",
                 cache_binding is not None and cache_binding["sha256"] == identity_files[0]
                 and all(path.startswith("/source_identity_cache") for path in plan_diff),
                 f"run plan differs from the prepare plan at {sorted(plan_diff) or 'nowhere'}")
    ledger.check(f"stage_b.{rate}.source_identity_files_equal", len(set(identity_files)) == 1,
                 f"{len(set(identity_files))} distinct digests over prepare and run")
    ledger.check(f"stage_b.{rate}.prepared_complete", prepared["status"] == "complete"
                 and prepared["schema"] == "prismaquant.tessera_joint_aura.prepared.v3"
                 and prepared["measured_cells"] == 1734 and prepared["production_cache"]["sha256"] == production_sha
                 and prepared["render_origins"] == {"encoded": 1734, "synthesized_from_wire": 0}
                 and prepared["render_comparisons"] == {"independent_render_vs_wire": 1734, "wire_round_trip_only": 0},
                 f"{prepared['measured_cells']} cells, cache {production_sha[:12]}")
    ledger.check(f"stage_b.{rate}.prepared_roster_3_formats", set(prepared["formats_by_qname"]) == census_units
                 and all(sorted(f) == sorted(rungs + ("BF16",)) for f in prepared["formats_by_qname"].values()),
                 "every unit lists its two Tessera rungs and the BF16 control")
    ledger.check(f"stage_b.{rate}.results_passed", prepare_results["passed"] is True and run_results["passed"] is True
                 and prepare_results["units"] == run_results["units"] == 867
                 and run_results["measured_cells"] == 1734 and run_results["cost"]["sha256"] == inputs.sha(
                     f"m4-{rate}-price", price_path), "prepare and run results")
    from prismaquant import glm_mtp_selection as selection
    ledger.check(f"stage_b.{rate}.price_header", price["schema"] == selection.SCHEMA and price["mtp_layer"] == 45
                 and set(price["costs"]) == census_units and all(set(r) == set(rungs) for r in price["costs"].values()),
                 "867 units with exactly this part's two rungs")
    roster = roster_row(ledger, f"m4_{rate}_price", price["costs"], census_units)
    ledger.check(f"stage_b.{rate}.groups_params_dtype", {k: len(v) for k, v in price["groups"].items()} == {
        "g:" + PREFIX + "shared_experts.gate_up_proj": 2, "s:" + PREFIX + "experts": 864,
        "u:" + PREFIX + "shared_experts.down_proj": 1} and all(
            price["params"][n] == math.prod(census["unit_shapes"][n]) for n in census_units)
        and set(price["source_dtype"].values()) == {"bfloat16"}, "groups 2/864/1, params from census shapes")
    anchors = price["provenance"]["tessera_joint_anchors"]["inputs"]["merged_cost"]
    ledger.check(f"stage_b.{rate}.price_names_its_stage_a_cost", anchors["sha256"] == expected["merged_cost"]
                 and anchors["path"] == str(a["cost_path"]), "M4 price provenance binds the M3 cost by path and digest")
    ledger.check(f"stage_b.{rate}.price_wire_bytes_equal_stage_a", all(
        price["wire_bytes"][n][r] == a["cost"]["costs"][n][r]["wire_bytes"] for n in census_units for r in rungs),
        "serialized byte count per cell equals the M3 price's")
    production = pickle.loads(production_path.read_bytes())
    verified = production.metadata["verified_cells"]
    cells = {(n, r) for n in census_units for r in rungs}
    ledger.check(f"stage_b.{rate}.prepared_cells_cover_roster", set(verified) == cells == set(production.weights),
                 f"{len(verified)} verified cells, {len(production.weights)} render paths")
    meta_inputs = production.metadata["inputs"]
    ledger.check(f"stage_b.{rate}.cache_binds_stage_a", all(meta_inputs[k]["sha256"] == v for k, v in expected.items()),
                 "prepared cache metadata names the Stage A files by digest")
    auth = production.metadata["source_authentication"]
    fresh = auth["authentication"].startswith("fresh SHA256")
    ledger.check(f"stage_b.{rate}.source_authenticated", len(auth["verified_files"]) == 127
                 and (auth["payload_bytes_hashed"] == 642652070880 if fresh else (
                     auth["authentication"].startswith("cached full-file SHA256")
                     and auth["payload_bytes_hashed"] == 0
                     and auth.get("streamed_identity_cache_sha256") == identity_files[0]))
                 and auth["census_sha256"] == census_sha and auth["capture_manifest_sha256"] == facts["capture_sha"],
                 f"{auth['authentication']}: {len(auth['verified_files'])} files, {auth['payload_bytes_hashed']} payload bytes")
    mismatch = []
    for (name, rung), cell in verified.items():
        row = price["costs"][name][rung]
        record = a["states"][name]["wire_records"][rung]
        operator = row["joint_operator_identity"]
        if not (cell["wire_sha256"] == record["blob_sha256"] and cell["source_weight"] == operator["source_weight"]
                and cell["rendered_weight"] == operator["rendered_weight"]
                and cell["render_origin"] == "encoded" and cell["render_comparison"] == "independent_render_vs_wire"
                and tuple(cell["source_weight"]["shape"]) == tuple(census["unit_shapes"][name])):
            mismatch.append(f"{name}@{rung}")
    ledger.check(f"stage_b.{rate}.prepared_evidence_binds_receipts_and_prices", not mismatch,
                 f"{len(mismatch)} cells off: {mismatch[:3]}")
    probe_sha = {row["probe_identity_sha256"] for r in price["costs"].values() for row in r.values()}
    ledger.check(f"stage_b.{rate}.one_probe_identity", len(probe_sha) == 1, f"{len(probe_sha)} probe identities")
    probe = next(iter(next(iter(price["costs"].values())).values()))["probe_identity"]
    ledger.check(f"stage_b.{rate}.probe_draw", (probe["n_probes"], probe["seed_base"], probe["token_scope"],
                 probe["calibration_shape"], probe["objective"]["objective"], probe["objective"]["mtp_layer"]) == (
                     4, 7000, "all", [512, 512], "mtp_head_self_kl", 45), str(
                         {k: probe[k] for k in ("n_probes", "seed_base", "token_scope")}))
    for kind, value in (("probe.identity_sha256", next(iter(probe_sha))),
                        ("probe.producer_source_sha256", probe["producer_source_sha256"]),
                        ("probe.calibration_sha256", probe["calibration_sha256"]),
                        ("probe.objective", probe["objective"])):
        ids.add(kind, f"m4-{rate}-price", value)
    for phase, results in (("prepare", prepare_results), ("run", run_results)):
        where = f"m4-{rate}-{phase}"
        ids.add("runtime.container_content_sha256", where, results["env"]["container_content_sha256"])
        ids.add("runtime.torch", where, results["env"]["torch"])
        ids.add("runtime.cuda", where, results["env"]["cuda"])
        ids.add("runtime.projection_backend_qualification_sha256", where,
                results["projection_backend"]["qualification_sha256"])
        ids.add("runtime.projection_backend_binary_sha256", where, results["projection_backend"]["build"]["binary_sha256"])
        ids.add("runtime.gpu", where, results["projection_backend"]["runtime"]["device"])
        ids.add("source.model_content_sha256", where, results["source_model_identity"]["content_sha256"])
        ids.add("source.model_path", where, results["source_model_identity"]["source"])
        ids.add("calibration.calibration_sha256", where, results["calibration_input"]["calibration_sha256"])
        ids.add("calibration.artifact_sha256", where, results["calibration_input"]["artifact_sha256"])
        ids.add("calibration.fit_ids_sha256", where, results["calibration_input"]["provenance"]["fit_ids_sha256"])
        ids.add("calibration.text_sha256", where, results["calibration_input"]["provenance"]["text_sha256"])
        ids.add("producer.prismaquant_source_sha256", where, results["dev_mode"]["producer_source_sha256"])
    ids.add("source.shard_digests_sha256", f"m4-{rate}-prepared-authentication", sha256_bytes(canonical(
        {f["name"]: f["sha256"] for f in auth["verified_files"] if f["name"].startswith("model-")})))
    ids.add("source.identity_file_sha256", f"m4-{rate}", identity_files[0])
    return {"price": price, "production": production, "verified": verified, "prepared": prepared,
            "prepare_results": prepare_results, "run_results": run_results, "price_path": price_path,
            "probe_sha256": next(iter(probe_sha)), "plan_diff": sorted(plan_diff), "plan_sha": plan_sha,
            "run_plan_sha": run_plan_sha, "rosters": [roster], "source_identity_path": root / "run" / "source-identity.json"}


def audit_source_immutability(identity_path, ledger, ids):
    """Compare the live source shards with the stat fingerprints recorded when they were hashed."""
    document = json.loads(Path(identity_path).read_bytes())
    changed = []
    for fingerprint in document["fingerprints"]:
        observed = os.stat(fingerprint["path"])
        current = {"inode": observed.st_ino, "size": observed.st_size,
                   "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns}
        if any(current[k] != fingerprint[k] for k in current):
            changed.append(fingerprint["path"])
    shards = {Path(s["path"]).name: s["sha256"] for s in document["identity"]["shards"]}
    ledger.check("source.fingerprints_unchanged", not changed and len(document["fingerprints"]) == 120,
                 f"{len(document['fingerprints'])} shards, {len(changed)} changed since they were hashed")
    ids.add("source.shard_digests_sha256", "source-identity-file", sha256_bytes(canonical(shards)))
    ids.add("source.model_content_sha256", "source-identity-file", document["identity"]["content_sha256"])
    return {"shards": len(shards), "changed": changed, "content_sha256": document["identity"]["content_sha256"]}


# --------------------------------------------------------------------------- owners: chain, offered menu, refusals
def audit_chain(ws, inputs, ledger, parts_by_rate, ids, out_dir):
    """The merged price through the owners: merge, receipt join, probe validation.

    The original merged payload (M6) predates ``wire_rungs`` in the merge provenance, which the
    current receipt join requires. The same two bound parts, merged by the current owner, give
    the payload the current owners accept; every price, byte count and group must equal the original.
    """
    from prismaquant import glm_mtp_selection as selection

    original_path = ws / "m6" / "layer45" / "merged-cost.pkl"
    original = inputs.pickle("merged-price-original", original_path)
    original_sha = inputs.sha("merged-price-original", original_path)
    sources = [part["source"] for part in original["provenance"]["parts"]]
    ledger.check("chain.original_sources_are_the_stage_b_prices", [s["sha256"] for s in sources] == [
        inputs.sha(f"m4-{rate}-price", parts_by_rate[rate]["price_path"]) for rate in RATES]
        and [s["path"] for s in sources] == [str(parts_by_rate[rate]["price_path"]) for rate in RATES],
        "the original merge names the r1024 and r896 prices by path and digest")
    current = selection.merge_mtp_costs([parts_by_rate[rate]["price"] for rate in RATES], sources=sources)
    core = ("schema", "mtp_layer", "costs", "wire_bytes", "params", "source_dtype", "groups")
    same = {key: current[key] == original[key] for key in core}
    ledger.check("chain.current_merge_prices_equal_the_original", all(same.values()),
                 str({k: v for k, v in same.items() if not v}) or "costs, wire_bytes, params, source_dtype, groups equal")
    stripped = {**current["provenance"], "parts": [{k: v for k, v in part.items() if k != "wire_rungs"}
                                                   for part in current["provenance"]["parts"]]}
    ledger.check("chain.original_provenance_lacks_only_wire_rungs", stripped == original["provenance"]
                 and all("wire_rungs" not in part for part in original["provenance"]["parts"]),
                 "the original merge provenance equals the current one without wire_rungs")
    try:
        selection.enrich_mtp_cost_wires(original)
        original_join = "accepted"
    except ValueError as exc:
        original_join = f"refused: {exc}"
    ledger.check("chain.current_owner_refuses_the_original_merge_format", original_join.startswith("refused: MTP M4 part 0 wire roster"),
                 original_join)
    enriched = selection.enrich_mtp_cost_wires(current)
    receipts = enriched["mtp_expert_wires"]
    bound = sum(len(by_rung) for by_rung in receipts.values())
    ledger.check("chain.owner_join_binds_every_routed_cell", bound == 3456 and {
        n for n, v in receipts.items() if v} == ROUTED_NAMES and all(
            set(v) == set(RUNG_ORDER) for v in receipts.values() if v), f"{bound} routed cells bound")
    ledger.check("chain.owner_join_leaves_dense_unbound", all(not receipts[n] for n in SHARED_NAMES),
                 "the receipt join has no path for the 3 shared units (their wires are checkpoint receipts)")
    probe_sha, probe = selection._mtp_probe(current)
    ledger.check("chain.one_probe_for_all_cells", probe_sha == parts_by_rate["r1024"]["probe_sha256"]
                 == parts_by_rate["r896"]["probe_sha256"], probe_sha[:12])
    cells = sum(len(v) for v in current["costs"].values())
    ledger.check("chain.merged_cells", cells == 3468 and sum(len(v) for v in current["wire_bytes"].values()) == 3468,
                 f"{cells} priced cells")
    ids.add("probe.identity_sha256", "merged-price", probe_sha)
    raw = pickle.dumps(current, protocol=4)
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="pq2530-merged-", dir=os.environ.get("TMPDIR")))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "merged-cost-current.pkl").write_bytes(raw)
    current_record = {"sha256": sha256_bytes(raw), "bytes": len(raw), "file": "merged-cost-current.pkl",
                      "path": str(out_dir / "merged-cost-current.pkl")}
    return {"original": original, "original_sha256": original_sha, "original_join": original_join,
            "current": current, "current_record": current_record, "enriched": enriched, "probe": probe}


def offered_menus(payload):
    """The menu the installed Tessera contract offers each unit, from the owners' own selection code."""
    from importlib.resources import as_file

    from prismaquant import glm_mtp_selection as selection
    from prismaquant import tessera_runtime_contract as contract
    from prismaquant.allocator import _mtp_rung_attestation, _mtp_scope_inputs
    from prismaquant.model_profiles import detect_profile
    from prismaquant.serving_profiles import load_serving_profile
    from prismaquant.tessera_lane import allocation_allowability_arguments
    from prismaquant.tessera_serving_scope import add_serving_scope_arguments, serving_target_from_args

    parser = argparse.ArgumentParser()
    add_serving_scope_arguments(parser)
    selection.add_mtp_scope_arguments(parser)
    allocation_allowability_arguments(parser)
    args = parser.parse_args(list(SERVING_SCOPE))
    profile = detect_profile(MODEL)
    target = serving_target_from_args(args, target_platform=load_serving_profile(TARGET_PROFILE).target_platform)
    stats, contexts = _mtp_scope_inputs(payload, target, profile, routing=None)
    # The public record counts unattested rungs but names no unit; the private row builder is the owner's menu.
    rows, unattested = selection._unit_rows(
        payload, _mtp_rung_attestation(target, profile), stats=stats, context_by_unit=contexts,
        target_profile=TARGET_PROFILE)
    with as_file(contract.contract_path()) as path:
        contract_sha256 = sha256_file(path)
    return {"contract_sha256": contract_sha256,
            "unattested": {rung: len(units) for rung, units in unattested.items()},
            "menus": {name: sorted(rung for rung in menu if rung != "BF16") for name, menu in rows.items()}}


def m6_offered(merged):
    """The menu the recorded M6 selection offered, read from the group rungs of its own menu rows."""
    record = json.loads((WORKSPACE / "m6/layer45/selection.json").read_bytes())["record"]
    memory = record["selection"]["memory"]
    per_group = {}
    for row in memory["excluded"] + memory["passing"]:
        for part in (row["name"] if isinstance(row, dict) else row).split("|"):
            group, rung = part.split("=", 1)
            per_group.setdefault(group, set()).add(rung)
    menus = {name: sorted(per_group[group] - {"BF16"}) for group, members in merged["groups"].items() for name in members}
    return menus, record["unattested_rungs"]


def audit_offered(merged, ledger):
    """Which priced cells are offered: by the repo-pinned Tessera contract now, and by the recorded M6 menu."""
    pin = json.loads((Path(__file__).resolve().parents[1] / "prismaquant/tessera_runtime/"
                      "tessera_serving_runtime_pin.json").read_bytes())
    current = offered_menus(merged)
    ledger.check("offered.installed_contract_is_the_repo_pin", current["contract_sha256"] == pin["contract_sha256"],
                 f"installed {current['contract_sha256'][:12]} pin {pin['contract_sha256'][:12]}")
    m6_menus, recorded = m6_offered(merged)
    priced = {rung for by_rung in merged["costs"].values() for rung in by_rung}
    derived = {rung: sum(rung not in menu for menu in m6_menus.values()) for rung in sorted(priced)}
    derived = {rung: count for rung, count in derived.items() if count}
    ledger.check("offered.m6_menu_rows_equal_its_unattested_counts", derived == recorded,
                 f"derived {derived} recorded {recorded}")
    summary = {"scope": list(SERVING_SCOPE), "target_profile": TARGET_PROFILE,
               "repo_pin": {"commit": pin["commit"], "contract_sha256": pin["contract_sha256"]},
               "current": {"contract_sha256": current["contract_sha256"], "unattested": current["unattested"]},
               "recorded_m6": {"unattested": recorded, "note": "decided by the Tessera contract of the M6 run; its "
                                                               "tree is not re-evaluable with the current owners"}}
    for label, menus in (("current", current["menus"]), ("recorded_m6", m6_menus)):
        by_class = {}
        for name, menu in menus.items():
            by_class.setdefault(unit_class(name), Counter())["+".join(menu)] += 1
        ledger.check(f"offered.{label}_menu_is_uniform_per_class", all(len(v) == 1 for v in by_class.values())
                     and set(by_class) == {"routed", "shared"}, str({k: dict(v) for k, v in by_class.items()}))
        summary[label]["menus"] = {cls: dict(counter) for cls, counter in by_class.items()}
    offered = {"current": {(n, r) for n, menu in current["menus"].items() for r in menu},
               "m6": {(n, r) for n, menu in m6_menus.items() for r in menu}}
    return offered, summary


def audit_refusals(merged, original, parts_by_rate, ledger):
    """Body-price admission on the MTP payloads, plus one probe of what its refusal rests on."""
    import copy

    from prismaquant import cost_currency
    from prismaquant.joint_aura import JOINT_AURA_COST_CURRENCY

    out = []
    payloads = {"r1024_part": parts_by_rate["r1024"]["price"], "r896_part": parts_by_rate["r896"]["price"],
                "merged_original": original, "merged_current": merged}
    for label, payload in payloads.items():
        try:
            cost_currency.require_run_currency(payload)
            message = None
        except cost_currency.CostCurrencyError as exc:
            message = str(exc)
        ledger.check(f"refusal.body_currency.{label}", message == MTP_BODY_REFUSAL, str(message))
        out.append({"payload": label, "refused": message is not None, "message": message})
    source = (Path(cost_currency.__file__)).read_text().splitlines()
    line = next(i for i, text in enumerate(source, 1) if MTP_BODY_REFUSAL in text)
    relabelled = copy.copy(merged)
    relabelled["provenance"] = {"cost_mode": "aura", "joint_activation": True,
                                "cost_currency": JOINT_AURA_COST_CURRENCY}
    try:
        cost_currency.require_run_currency(relabelled)
        probe = "admitted"
    except Exception as exc:  # the probe records whatever the gate answers
        probe = f"refused: {type(exc).__name__}: {exc}"[:300]
    return {"raised_at": f"prismaquant/cost_currency.py:{line}", "reached_from": "prismaquant/allocator.py (--costs)",
            "payloads": out,
            "probe_relabelled_provenance": {
                "what": "the merged MTP rows under body-style provenance (cost_mode aura, joint_activation true)",
                "outcome": probe}}


# --------------------------------------------------------------------------- io
def _pin_worker_init(pin_src):
    sys.path.insert(0, str(pin_src))


def _verify_wire_batch(batch):
    """Producer-side check of wire blobs: the pinned tessera's own ``verify_cached_unit``."""
    from tessera.cached_unit import verify_cached_unit

    out = []
    for path, record in batch:
        blob = Path(path).read_bytes()
        try:
            verify_cached_unit(blob, record, record["identity"])
            error = None
        except Exception as exc:  # the producer names its own refusal
            error = f"{type(exc).__name__}: {exc}"[:200]
        out.append((path, sha256_bytes(blob), len(blob), error))
    return out


def audit_io(ws, facts, a_by_rate, b_by_rate, ledger, progress, workers, limit):
    """Re-read the capture entries, every wire blob and every rendered shard."""
    from prismaquant import tessera_calibration_cache as cc

    census = facts["census"]
    capture_root = ws / "m1" / "v2" / "capture"
    entries = facts["capture_manifest"]["entries"]
    names = sorted(entries)[:limit] if limit else sorted(entries)
    capture_rows = {}

    def one(name):
        path = capture_root / entries[name]["path"]
        payload, _receipt = cc._verified_capture_entry(
            path, name, expected_sha256=entries[name]["sha256"], census=census, max_rows=512,
            policy=CAPTURE_POLICY, execution=None)
        row = {"bytes": path.stat().st_size, "sha256": entries[name]["sha256"],
               "hessian": list(payload["hessian"].shape), "activations": list(payload["inputs"].shape),
               "count": int(payload["count"]), "max_abs": float(payload["max_abs"])}
        progress.advance("capture")
        return name, row

    log(f"capture: verifying {len(names)} entries through the owner")
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        for name, row in pool.map(one, names):
            capture_rows[name] = row
    bad = [n for n, r in capture_rows.items() if (r["count"], r["max_abs"]) != (
        census["counts"][n], float(census["max_abs"][n]))]
    ledger.check("io.capture_entries_verified", len(capture_rows) == len(names) and not bad,
                 f"{len(capture_rows)} entries hashed and validated (hessian, activations, count, maximum); {len(bad)} off")

    cells = {}
    for rate in RATES:
        a, b = a_by_rate[rate], b_by_rate[rate]
        for name in sorted(census["unit_shapes"]):
            for rung in RUNGS[rate]:
                cells[(name, rung)] = {"rate": rate, "record": a["states"][name]["wire_records"][rung],
                                       "wire_dir": a["wire_dir"], "verified": b["verified"][(name, rung)],
                                       "render": b["production"].weights[(name, rung)]}
    keys = sorted(cells)[:limit * 4] if limit else sorted(cells)
    log(f"wires: producer verification of {len(keys)} blobs")
    batches = [[(str(cells[k]["wire_dir"] / cells[k]["record"]["file"]), cells[k]["record"]) for k in keys[i:i + 16]]
               for i in range(0, len(keys), 16)]
    wire_rows = {}
    import multiprocessing
    with ProcessPoolExecutor(max_workers=min(workers, 16), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_pin_worker_init, initargs=(str(TESSERA_PIN_SRC),)) as pool:
        for result in pool.map(_verify_wire_batch, batches):
            for path, digest, size, error in result:
                wire_rows[path] = (digest, size, error)
            progress.advance("wires", len(result))
    for key in keys:
        cell = cells[key]
        digest, size, error = wire_rows[str(cell["wire_dir"] / cell["record"]["file"])]
        cell.update(wire_sha256=digest, wire_file_bytes=size, producer_error=error)
    log(f"renders: hashing {len(keys)} rendered shards")

    def render(key):
        digest = sha256_file(cells[key]["render"])
        progress.advance("renders")
        return key, digest

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, digest in pool.map(render, keys):
            cells[key]["render_sha256"] = digest
    bad_wire = [k for k in keys if cells[k]["producer_error"] or cells[k]["wire_sha256"] != cells[k]["record"]["blob_sha256"]
                or cells[k]["wire_file_bytes"] != cells[k]["record"]["blob_bytes"]
                or cells[k]["wire_sha256"] != cells[k]["verified"]["wire_sha256"]]
    bad_render = [k for k in keys if cells[k]["render_sha256"] != cells[k]["verified"]["render_file_sha256"]]
    ledger.check("io.wires_match_receipts_and_prepared_evidence", not bad_wire,
                 f"{len(keys)} wire blobs re-hashed, producer-verified, equal to receipt and prepared wire_sha256; {len(bad_wire)} off: {bad_wire[:2]}")
    ledger.check("io.renders_match_prepared_evidence", not bad_render,
                 f"{len(keys)} rendered shards re-hashed against prepared render_file_sha256; {len(bad_render)} off")
    return capture_rows, cells


# --------------------------------------------------------------------------- tables
UNIT_COLUMNS = ("unit", "class", "out_features", "in_features", "params", "count", "max_abs", "capture_bytes",
                "capture_sha256", "hessian_shape", "activation_shape", "projection_geometry")
CELL_COLUMNS = ("unit", "class", "rung", "stage_b_part", "offered_current", "offered_m6", "stage_a_output_mse",
                "stage_b_predicted_dloss",
                "wire_bytes", "wire_file_bytes", "wire_sha256", "receipt", "producer_verified",
                "prepared_wire_match", "prepared_render_match")


def unit_rows(facts, capture_rows, projection):
    census = facts["census"]
    geometry = {name: f"{u['rows']}x{u['cols']}" for stack in projection["stacks"].values() for name, u in stack.items()}
    rows = []
    for name in sorted(census["unit_shapes"]):
        out_f, in_f = census["unit_shapes"][name]
        c = capture_rows.get(name)
        rows.append({"unit": name, "class": unit_class(name), "out_features": out_f, "in_features": in_f,
                     "params": out_f * in_f, "count": census["counts"][name], "max_abs": repr(float(census["max_abs"][name])),
                     "capture_bytes": c["bytes"] if c else "", "capture_sha256": c["sha256"] if c else "",
                     "hessian_shape": "x".join(map(str, c["hessian"])) if c else "",
                     "activation_shape": "x".join(map(str, c["activations"])) if c else "",
                     "projection_geometry": geometry.get(name, "")})
    return rows


def cell_rows(facts, a_by_rate, b_by_rate, cells, offered):
    rows = []
    for rate in RATES:
        a, b = a_by_rate[rate], b_by_rate[rate]
        for name in sorted(facts["census"]["unit_shapes"]):
            for rung in RUNGS[rate]:
                cell = cells.get((name, rung))
                routed = name in ROUTED_NAMES
                record = a["states"][name]["wire_records"][rung]
                rows.append({
                    "unit": name, "class": unit_class(name), "rung": rung, "stage_b_part": rate,
                    "offered_current": int((name, rung) in offered["current"]),
                    "offered_m6": int((name, rung) in offered["m6"]),
                    "stage_a_output_mse": repr(a["cost"]["costs"][name][rung]["output_mse"]),
                    "stage_b_predicted_dloss": repr(b["price"]["costs"][name][rung]["predicted_dloss"]),
                    "wire_bytes": b["price"]["wire_bytes"][name][rung],
                    "wire_file_bytes": cell["wire_file_bytes"] if cell and "wire_file_bytes" in cell else "",
                    "wire_sha256": record["blob_sha256"],
                    "receipt": "expert_wires+checkpoint" if routed else "checkpoint",
                    "producer_verified": ("" if not cell or "producer_error" not in cell else int(cell["producer_error"] is None)),
                    "prepared_wire_match": int(b["verified"][(name, rung)]["wire_sha256"] == record["blob_sha256"]),
                    "prepared_render_match": ("" if not cell or "render_sha256" not in cell else int(
                        cell["render_sha256"] == b["verified"][(name, rung)]["render_file_sha256"]))})
    return rows


def rung_matrix(rows):
    """One row per (rung, class): priced, offered, receipts, wire bytes."""
    out = []
    for rung in RUNG_ORDER:
        for cls in ("routed", "shared"):
            group = [r for r in rows if r["rung"] == rung and r["class"] == cls]
            out.append({"rung": rung, "class": cls, "priced_units": len(group),
                        "offered_current_units": sum(r["offered_current"] for r in group),
                        "offered_m6_units": sum(r["offered_m6"] for r in group),
                        "expert_wire_receipts": sum(r["receipt"] == "expert_wires+checkpoint" for r in group),
                        "checkpoint_receipts": len(group),
                        "wire_bytes": sum(int(r["wire_bytes"]) for r in group)})
    return out


def as_text(rows):
    """Table rows as a CSV reader returns them: every value a string."""
    return [{key: str(value) for key, value in row.items()} for row in rows]


def write_csv(path, columns, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    Path(path).write_bytes(raw)
    return {"file": Path(path).name, "sha256": sha256_bytes(raw), "bytes": len(raw), "rows": len(rows),
            "columns": list(columns)}


# --------------------------------------------------------------------------- the offline validator
def table_problems(summary, units, cells):
    """Problems in the unit and cell tables, found without any external file.

    ``units`` and ``cells`` are the CSV rows as strings. The tests also call this
    on deliberately damaged copies, so every rule has a failing input.
    """
    problems = []

    def need(ok, text):
        if not ok:
            problems.append(text)

    need(len({u["unit"] for u in units}) == len(units) == 867, "867 distinct units")
    need(Counter(u["class"] for u in units) == {"routed": 864, "shared": 3}, "864 routed and 3 shared units")
    need(all(u["capture_sha256"] and u["hessian_shape"] and u["activation_shape"] for u in units),
         "every unit has a verified capture entry")
    need(all(int(u["count"]) > 0 and math.isfinite(float(u["max_abs"])) and float(u["max_abs"]) > 0 for u in units),
         "positive count and finite positive maximum per unit")
    need(all(int(u["hessian_shape"].split("x")[0]) == int(u["in_features"])
             and int(u["activation_shape"].split("x")[1]) == int(u["in_features"]) for u in units),
         "hessian and activation geometry follow the unit's input width")
    need(len(cells) == 3468 and len({(c["unit"], c["rung"]) for c in cells}) == 3468, "3468 distinct cells")
    need({c["unit"] for c in cells} == {u["unit"] for u in units}, "cells cover exactly the units")
    need(Counter(c["rung"] for c in cells) == dict.fromkeys(RUNG_ORDER, 867), "867 cells per rung")
    need(all(c["producer_verified"] == "1" and c["prepared_wire_match"] == "1" and c["prepared_render_match"] == "1"
             and int(c["wire_file_bytes"]) == int(c["wire_bytes"]) for c in cells),
         "every cell: producer-verified wire, prepared evidence match, file bytes equal serialized bytes")
    need(all(c["receipt"] == ("expert_wires+checkpoint" if c["class"] == "routed" else "checkpoint") for c in cells),
         "receipt kind follows the unit class")
    need(sum(c["receipt"] == "expert_wires+checkpoint" for c in cells) == 3456,
         "3456 routed cells carry expert-wire receipts")
    need(sum(c["class"] == "shared" for c in cells) == 12, "12 shared cells carry checkpoint receipts only")
    need(Counter((c["class"], c["rung"]) for c in cells if c["offered_current"] == "1") == {
        ("routed", "TESSERA_BF16_K1_R1024"): 864, ("routed", "TESSERA_E4M3_K1_R1024"): 864,
        ("routed", "TESSERA_E4M3_K1_R896"): 864, ("shared", "TESSERA_BF16_K1_R1024"): 3,
        ("shared", "TESSERA_BF16_K1_R896"): 3, ("shared", "TESSERA_E4M3_K1_R1024"): 3,
        ("shared", "TESSERA_E4M3_K1_R896"): 3}, "2604 cells offered by the repo-pinned contract, by class and rung")
    need(Counter((c["class"], c["rung"]) for c in cells if c["offered_m6"] == "1") == {
        ("routed", "TESSERA_BF16_K1_R1024"): 864, ("routed", "TESSERA_E4M3_K1_R896"): 864,
        ("shared", "TESSERA_BF16_K1_R1024"): 3, ("shared", "TESSERA_E4M3_K1_R1024"): 3},
        "1734 cells offered by the contract of the recorded M6 menu, by class and rung")
    sums = Counter()
    for c in cells:
        sums[c["rung"]] += int(c["wire_bytes"])
    matrix = summary.get("rung_matrix", [])
    need(all(sums[r] == sum(m["wire_bytes"] for m in matrix if m["rung"] == r) for r in RUNG_ORDER),
         "rung matrix byte sums equal the cell table")
    for table, rows in (("units", units), ("cells", cells)):
        need(summary.get("tables", {}).get(table, {}).get("rows") == len(rows), f"{table} row count in the summary")
    return problems


def summary_problems(summary):
    """Problems in the summary alone: checks, execution record, ledgers."""
    problems = []

    def need(ok, text):
        if not ok:
            problems.append(text)

    need(summary.get("schema") == SCHEMA, "schema")
    need(summary.get("limit_units") is None and summary.get("skip_seal_recompute") is False,
         "a smoke run is not a record")
    checks = summary.get("checks", [])
    need(checks and all(c["ok"] for c in checks) and summary.get("failed") == 0
         and summary.get("passed") == len(checks), "every check passed")
    need(not summary.get("performance_claims"), "no performance claim")
    for row in summary.get("actions", []):
        need(row.get("ok") and not row.get("problems") and row["receipt"]["sha256"] == row["receipt"]["recomputed_sha256"]
             and row["payload"]["sha256"] == row["receipt"]["result_sha256"] == row["blob"]["sha256"]
             and row["receipt"]["sha256"] != row["stdout"]["sha256"], f"action {row.get('role')} CAS chain")
    need(summary.get("actions"), "an execution record")
    need(all(s["kind"] != "unclassified" and s.get("resolution") for s in summary.get("dev_ledger", [])),
         "every [DEV-MODE] line is resolved")
    need(all(r["equals"] for r in summary.get("rosters", [])) and summary.get("rosters"),
         "every roster equals the authenticated census or its named subset")
    return problems


def validate_record(summary, units, cells):
    """All problems in a record: the summary, then its two tables. Empty means consistent."""
    return summary_problems(summary) + table_problems(summary, units, cells)


# --------------------------------------------------------------------------- dev ledger resolutions
RESOLUTIONS = {
    "prepared_plan": ("explained", "the r1024 prepare and run plans differ only in source_identity_cache, the path and "
                                   "digest of the identity file the prepare wrote (stage_b.r1024.run_plan_binds_the_identity_cache)"),
    "prepared_implementation": ("explained", "both executing trees reproduce their stamps from the snapshot bundles and "
                                             "differ in four identity-proof files (trees.*)"),
    "checkpoint_seal": ("verified_here", "the canonical seal is recomputed from the checkpoint identity and equals the "
                                         "declared one (stage_a.*.checkpoint_seal_recomputed)"),
    "projection_shape": ("limitation", "routed (4096,2048) and (2048,4096) are outside the fused kernel's qualified "
                                       "shapes, so the reference arithmetic ran; a certified run refuses these shapes"),
    "source_rehash": ("verified_here", "the prepare hashed all 120 shards inside its GPU reservation; the live shards "
                                       "still match their recorded stat fingerprints (source.fingerprints_unchanged)"),
}


def build_dev_ledger(pb_root, actions, ledger):
    rows = []
    for action in actions:
        if not action["role"].startswith(("m4-", "m3-")):
            continue
        for line in dev_lines(pb_root, action["key"]):
            status, evidence = RESOLUTIONS.get(line["kind"], (None, None))
            rows.append({"role": action["role"], "key": action["key"], "line": line["line"], "kind": line["kind"],
                         "text": line["text"], "resolution": {"status": status, "evidence": evidence} if status else None})
    ledger.check("dev_ledger.every_line_classified", all(r["resolution"] for r in rows), Counter(
        r["kind"] for r in rows).__repr__())
    return rows


def limitations(summary):
    """What the record does not establish, derived from its own findings so it cannot be left out."""
    out = []
    stamps = summary.get("dev_stamps", {})
    dev_actions = sorted(a["role"] for a in summary.get("actions", []) if (a.get("spec") or {}).get("dev_mode") == "1")
    if stamps and all(v["dev_uncertified"] for v in stamps.values()):
        out.append({"id": "dev_uncertified",
                    "text": "All four Stage B results carry the dev_uncertified stamp, and "
                            f"{len(dev_actions)} sealed action specs set PRISMAQUANT_DEV_MODE=1. This record re-checks "
                            "the seals those runs skipped. It does not turn the runs into certified runs."})
    if any(line["kind"] == "projection_shape" for line in summary.get("dev_ledger", [])):
        out.append({"id": "certified_run_refuses_routed_shapes",
                    "text": "The routed expert shapes (4096,2048) and (2048,4096) are outside the fused projection "
                            "kernel's qualified shapes. The original Stage B runs used the reference arithmetic. A run with "
                            "PRISMAQUANT_DEV_MODE=0 refuses these shapes."})
    out.append({"id": "dense_wires_have_no_selection_receipt_path",
                "text": "The 12 shared-unit cells have verified wires and checkpoint receipts. The owner's receipt join "
                        "(enrich_mtp_cost_wires) binds routed cells only, so a selected dense Tessera rung is refused "
                        "(tests/test_glm_mtp_priced_wires_1413.py)."})
    probe = (summary.get("body_admission") or {}).get("probe_relabelled_provenance", {}).get("outcome")
    if probe == "admitted":
        out.append({"id": "body_refusal_rests_on_payload_provenance",
                    "text": "Body-price admission refuses the MTP payloads because their provenance is not body-aura "
                            "provenance. It does not read the row-level MTP objective: the same rows under "
                            "body-style provenance are admitted."})
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--pb-root", type=Path, default=PB_ROOT)
    parser.add_argument("--out-dir", type=Path, help="write units.csv and cells.csv here")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--limit-units", type=int, default=None,
                        help="smoke slice: the io section reads only the first N units (D38 dry run)")
    parser.add_argument("--extra-action", action="append", default=[], metavar="ROLE=KEY",
                        help="also verify this action's outcome and CAS receipt")
    parser.add_argument("--verify-actions-only", action="store_true",
                        help="print the PrismaBuild verification of --extra-action keys and stop")
    parser.add_argument("--skip-seal-recompute", action="store_true",
                        help="smoke only: skip the 14 GiB checkpoint seal recompute; the result is not a record")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sections = set(args.sections.split(","))
    unknown = sections - set(SECTIONS)
    if unknown:
        raise SystemExit(f"unknown sections {sorted(unknown)}")
    ws, pb = args.workspace, args.pb_root
    ledger, inputs, ids, progress = Ledger(), Inputs(), Identities(), Progress()
    extra = [tuple(item.split("=", 1)) for item in args.extra_action]
    if args.verify_actions_only:
        rows = [verify_action(pb, key, role) for role, key in extra]
        print(json.dumps({"schema": SCHEMA + ".actions", "actions": rows}, sort_keys=True))
        return 0 if all(r.get("ok") for r in rows) else 1
    summary = {"schema": SCHEMA, "issue": "RobTand/prismaquant#2530", "parent": "RobTand/prismaquant#1271",
               "layer": LAYER, "limit_units": args.limit_units, "sections": sorted(sections),
               "skip_seal_recompute": args.skip_seal_recompute,
               "interpreter": {"path": sys.executable, "python": sys.version.split()[0]},
               "argv": [str(a) for a in (argv if argv is not None else sys.argv[1:])],
               "tool": {"path": "tools/audit_glm_mtp_layer45.py", "sha256": sha256_file(__file__)}}
    census_path = ws / "m1/v2/mtp-census.json"
    census = inputs.json("authenticated-census", census_path)
    census_sha = inputs.sha("authenticated-census", census_path)
    facts = {"census": census, "census_sha": census_sha}
    summary["census"] = {"sha256": census_sha, **census_facts(census, ledger)}
    ids.add("calibration.text_sha256", "census", census["text_sha256"])
    ids.add("calibration.fit_ids_sha256", "census", census["fit_ids_sha256"])
    ids.add("source.model_path", "census", census["model"])
    projection, summary["projection"] = audit_projection(ws, inputs, ledger, census, ids)
    facts["projection"] = census["expert_projection"]
    manifest, capture_roster, summary["capture"] = audit_capture_manifest(ws, inputs, ledger, census, census_sha, ids)
    capture_report = json.loads((ws / "m1/v2/capture/capture-run.json").read_bytes())
    summary["capture"]["census_authentication"] = audit_census_authentication(
        inputs, ledger, census, census_sha, capture_report, projection)
    summary["calibration"] = audit_calibration(inputs, ledger, census, ids)
    facts["capture_manifest"] = manifest
    facts["capture_sha"] = inputs.sha("capture-manifest", ws / "m1/v2/capture/capture_manifest.json")
    rosters = [capture_roster]
    a_by_rate, b_by_rate = {}, {}
    if "structure" in sections:
        for rate in RATES:
            a_by_rate[rate] = audit_stage_a(ws, rate, inputs, ledger, facts, ids,
                                            recompute_seal=not args.skip_seal_recompute)
            b_by_rate[rate] = audit_stage_b(ws, rate, inputs, ledger, facts, a_by_rate[rate], ids)
            rosters += a_by_rate[rate]["rosters"] + b_by_rate[rate]["rosters"]
        summary["source"] = audit_source_immutability(b_by_rate["r1024"]["source_identity_path"], ledger, ids)
        chain = audit_chain(ws, inputs, ledger, b_by_rate, ids, args.out_dir)
        merged, probe = chain["current"], chain["probe"]
        summary["merged_price"] = {
            "original": {"sha256": chain["original_sha256"], "owner_join": chain["original_join"]},
            "current": chain["current_record"], "cells": 3468,
            "probe_identity_sha256": b_by_rate["r1024"]["probe_sha256"],
            "probe": {k: probe[k] for k in ("n_probes", "seed_base", "token_scope", "distribution",
                                            "normalization", "objective")}}
        rosters += [roster_row(ledger, "merged_price", merged["costs"], census["unit_shapes"]),
                    roster_row(ledger, "merged_wire_bytes", merged["wire_bytes"], census["unit_shapes"]),
                    roster_row(ledger, "merged_params", merged["params"], census["unit_shapes"])]
        offered, summary["offered"] = audit_offered(merged, ledger)
        summary["stage_a"] = {rate: {"rows": a_by_rate[rate]["rows"], "seal": a_by_rate[rate]["seal"],
                                     "seal_recomputed": a_by_rate[rate]["seal_recomputed"]} for rate in RATES}
        summary["dev_stamps"] = {
            f"m4-{rate}-{phase}": {"dev_uncertified": results["dev_uncertified"],
                                   "dev_mode": results["dev_mode"]["PRISMAQUANT_DEV_MODE"],
                                   "producer_source_sha256": results["dev_mode"]["producer_source_sha256"]}
            for rate in RATES for phase, results in (("prepare", b_by_rate[rate]["prepare_results"]),
                                                    ("run", b_by_rate[rate]["run_results"]))}
        summary["stage_b"] = {rate: {"plan_sha256": b_by_rate[rate]["plan_sha"], "run_plan_sha256": b_by_rate[rate][
            "run_plan_sha"], "plan_difference": b_by_rate[rate]["plan_diff"]} for rate in RATES}
        if "refusals" in sections:
            summary["body_admission"] = audit_refusals(merged, chain["original"], b_by_rate, ledger)
    summary["rosters"] = rosters
    capture_rows, cells = {}, {}
    if "io" in sections and "structure" in sections:
        capture_rows, cells = audit_io(ws, facts, a_by_rate, b_by_rate, ledger, progress, args.workers,
                                       args.limit_units)
    if "pb" in sections:
        actions, excluded = audit_actions(pb, ledger, ws, [tuple(item) for item in extra])
        summary["actions"], summary["excluded_attempts"] = actions, excluded
        summary["dev_ledger"] = build_dev_ledger(pb, actions, ledger)
        summary["trees"] = audit_trees(pb, ledger, actions, ws) if "trees" in sections else None
        summary["producer_pin"] = pinned_encoder_source(ledger, ids) if "trees" in sections else None
        by_role = {a["role"]: a for a in actions}
        for role, row in by_role.items():
            if row.get("spec"):
                ids.add("runtime.container_content_sha256", f"pb-{role}", row["spec"]["container_content_sha256"])
                ids.add("runtime.container_admission_reference", f"pb-{role}", row["spec"]["container_admission_reference"])
                ids.add("producer.tessera_pin_mount", f"pb-{role}", row["spec"]["tessera_pin"])
        for rate in RATES:
            ids.add("producer.prismaquant_source_sha256", f"pb-m4-{rate}-run-tree",
                    (summary["trees"] or {}).get(f"m4-{rate}-run", {}).get("reproduced_producer_source_sha256"))
    summary["identities"] = ids.table(ledger, allowed_split={"producer.prismaquant_source_sha256"})
    summary["inputs"] = inputs.table()
    summary["performance_claims"] = []
    summary["limitations"] = limitations(summary)
    if args.out_dir and "io" in sections and "structure" in sections:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        urows = unit_rows(facts, capture_rows, projection)
        crows = cell_rows(facts, a_by_rate, b_by_rate, cells, offered)
        summary["tables"] = {"units": write_csv(args.out_dir / "units.csv", UNIT_COLUMNS, urows),
                             "cells": write_csv(args.out_dir / "cells.csv", CELL_COLUMNS, crows)}
        summary["rung_matrix"] = rung_matrix(crows)
        if args.limit_units is None:
            for item in table_problems(summary, as_text(urows), as_text(crows)):
                ledger.check(f"record.{item}", False, "offline validator on the tables just written")
    failed = [c for c in ledger.checks if not c["ok"]]
    summary.update({"checks": ledger.checks, "passed": len(ledger.checks) - len(failed), "failed": len(failed)})
    sys.stdout.write(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    sys.stdout.flush()
    log(f"{summary['passed']} checks passed, {summary['failed']} failed")  # verbose only
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
