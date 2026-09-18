#!/usr/bin/env python3
"""Compute a per-row data manifest for each row of a pbcampaign manifest.

Why this exists
---------------
PrismaBuild issue #487 gives an action a second content-addressed input, a
*data manifest*: the exact files and byte ranges the action will read off the
shared mount, so a storage-role fleet loop can pull them into the dl380g10 ARC
before the row is claimed. W3 measured the payoff -- an ARC-resident row reads
at 3298.7 MB/s against 306.9 MB/s cold, 10.75x -- but only if the *right* bytes
are resident, and only the producer knows which those are.

This tool is the producer side. It reuses the read-set expansion that the W3
prewarm daemon already validated (``glm_arc_prewarm.Campaign``): the 864
capture files a row's members name, in the order ``prefetch_capture`` consumes
them (sorted member name), plus the coalesced byte ranges of those members'
weight tensors inside the safetensors shards. It writes one manifest per row
and a NEW pbcampaign manifest carrying ``data_manifest`` per row. The input
manifest is never rewritten.

Exact identity, not globs
-------------------------
Every entry names one file and, for a weight shard, one byte range. Nothing is
expanded at consume time, so the prewarm loop reads exactly what the producer
priced and a reviewer can diff the manifest against the row.

Per-file ``sha256`` is deliberately ``null``. Hashing this campaign's 1023 GB of
capture bytes would cost more than the prewarm saves, and the contract does not
need it: the *manifest file* is content-addressed in the PrismaBuild CAS, which
is what binds the action key to this byte list. The manifest is a residency
hint, not an integrity claim about the data, and it says so in its own schema.

Capture sizes come from a cached inventory (``--sizes-cache``, one
``<basename> <bytes>`` line per file) so that generating 16 manifests does not
stat 36,423 files over NFS. Pass ``--stat`` to stat instead.

Usage::

    python3 experiments/glm_data_manifests.py \
      --campaign-manifest BASE/transition-optimization-20260910/resume-overlap-manifest.json \
      --out-dir          BASE/transition-optimization-20260910/data-manifests \
      --out-manifest     BASE/transition-optimization-20260910/resume-overlap-manifest.with-data.json \
      --sizes-cache      /home/rob/tmp/glm-perf-20260910/w3/evidence/capture-file-sizes.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm_arc_prewarm import (  # noqa: E402
    CAMPAIGN_BASE, Campaign, RECORD_SIZE, SEED_WIRE_DIR_FLAG, UNITS_RE,
    argv_value, to_pool)

SCHEMA = "prismaquant.prismabuild.data_manifest.v1"
SCHEMA_V2 = "prismaquant.prismabuild.data_manifest.v2"
SHARED_MOUNT = "/mnt/shared"

# Load this torch-free module without importing prismaquant.__init__, whose
# registry import requires the GPU image during a CPU-only manifest build.
_phase_spec = importlib.util.spec_from_file_location(
    "joint_prewarm_phases",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "prismaquant",
                 "joint_prewarm_phases.py"))
_phase_module = importlib.util.module_from_spec(_phase_spec)
_phase_spec.loader.exec_module(_phase_module)

#: The resumed read order and the annotations that seal it (#607). Loaded the
#: same way for the same reason: a CPU manifest build has no GPU image, and
#: the preparing action reads this exact file to walk the order it sealed.
_replay_spec = importlib.util.spec_from_file_location(
    "joint_replay_frontier",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "prismaquant",
                 "joint_replay_frontier.py"))
_replay_module = importlib.util.module_from_spec(_replay_spec)
_replay_spec.loader.exec_module(_replay_module)

#: The run's own layer phase name, loaded the same torch-free way. The joint
#: run reports its committed units under exactly these names, and the window
#: advances by matching the name it reports against the phase table sealed
#: here; a second spelling of the convention is how a consumer ends up
#: committing a name the plan cannot match, which reads as no progress at all.
_run_progress_spec = importlib.util.spec_from_file_location(
    "joint_run_progress",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "prismaquant",
                 "joint_run_progress.py"))
_run_progress_module = importlib.util.module_from_spec(_run_progress_spec)
_run_progress_spec.loader.exec_module(_run_progress_module)

#: ``prismabuild.core._DATA_MANIFEST_KEYS`` and ``_DATA_MANIFEST_ENTRY_KEYS``,
#: restated because PrismaBuild is not importable from the environments that
#: build a manifest.  ``validate_data_manifest`` builds both through
#: ``_exact_mapping``: an unknown key is a refusal, not an ignored extra.
MANIFEST_KEYS = frozenset({"schema", "produced_by", "mount_prefix", "entries",
                           "entry_count", "total_bytes", "annotations"})
ENTRY_KEYS = frozenset({"path", "offset", "bytes", "sha256"})

#: ``prismabuild.core.DATA_MANIFEST_MAX_ENTRIES`` and
#: ``DATA_MANIFEST_MAX_BYTES``: ``validate_data_manifest`` refuses a longer
#: entry list and ``load_data_manifest`` refuses a larger file.  A routed GLM
#: row is about 870 capture and weight entries plus a 5,920-file seed wire, so
#: roughly 6,800 entries and 2 MB -- well inside both, but the ceilings belong
#: here so a future read set that crosses one is refused by the producer
#: rather than by the fleet.
MAX_ENTRIES = 1_000_000
MAX_READS = 4_000_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def check_manifest(manifest: dict, *, where: str = "data manifest") -> dict:
    """Refuse here what PrismaBuild would refuse at submission.

    The producer is in this repo and the validator is in PrismaBuild, so a
    manifest that drifts breaks nothing in either tree: it surfaces as a
    refused submission, after the campaign was laid out.  These are the same
    four rules ``prismabuild.core.validate_data_manifest`` applies, and they
    are cheap enough to apply to every row at submit time.
    """
    if set(manifest) != set(MANIFEST_KEYS):
        raise SystemExit(
            f"{where}: fields differ: "
            f"missing={sorted(MANIFEST_KEYS - set(manifest))}, "
            f"extra={sorted(set(manifest) - MANIFEST_KEYS)}")
    if manifest["schema"] != SCHEMA:
        raise SystemExit(f"{where}: schema must be {SCHEMA}")
    for field in ("produced_by", "annotations"):
        if not isinstance(manifest[field], dict):
            raise SystemExit(f"{where}: {field} must be an object")
    prefix = manifest["mount_prefix"]
    if not prefix.startswith("/") or prefix != os.path.normpath(prefix):
        raise SystemExit(f"{where}: mount_prefix must be a normalized absolute path")
    if prefix == "/":
        raise SystemExit(f"{where}: mount_prefix must name a mount, not the root")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{where}: entries must be a non-empty array")
    if len(entries) > MAX_ENTRIES:
        raise SystemExit(f"{where}: entries exceed {MAX_ENTRIES}")
    seen: set[tuple[str, int]] = set()
    total = 0
    for index, entry in enumerate(entries):
        at = f"{where} entries[{index}]"
        if set(entry) != set(ENTRY_KEYS):
            raise SystemExit(f"{at}: fields differ")
        path, offset, size = entry["path"], entry["offset"], entry["bytes"]
        if not path.startswith(prefix + "/"):
            raise SystemExit(f"{at}: {path} is outside {prefix}")
        if os.path.normpath(path) != path:
            raise SystemExit(f"{at}: {path} is not normalized")
        if not isinstance(offset, int) or offset < 0:
            raise SystemExit(f"{at}: offset must be a non-negative integer")
        if not isinstance(size, int) or size <= 0:
            raise SystemExit(f"{at}: bytes must be positive")
        if (path, offset) in seen:
            raise SystemExit(f"{at}: repeats a (path, offset)")
        seen.add((path, offset))
        total += size
    if manifest["entry_count"] != len(entries):
        raise SystemExit(f"{where}: entry_count disagrees with entries")
    if manifest["total_bytes"] != total:
        raise SystemExit(f"{where}: total_bytes disagrees with entries")
    return manifest


def check_manifest_bytes(blob: bytes, *, where: str = "data manifest") -> bytes:
    """Refuse a manifest file PrismaBuild would refuse to read at all.

    ``load_data_manifest`` stats the file before it parses it, so an oversized
    manifest fails at submission with nothing validated.  Checking the bytes
    the producer is about to write keeps that refusal here.
    """
    if len(blob) > MAX_MANIFEST_BYTES:
        raise SystemExit(
            f"{where}: manifest file is {len(blob)} bytes, over the "
            f"{MAX_MANIFEST_BYTES}-byte limit")
    return blob


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(tree: str) -> str:
    try:
        out = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


#: The source files whose code derives a row's read set: this module's
#: ``build_manifest`` and the ``Campaign`` expansion it walks.  Relative to the
#: repository root.
PRODUCER_SOURCES = ("experiments/glm_data_manifests.py",
                    "experiments/glm_arc_prewarm.py")


def producer_source_sha256() -> str:
    """One digest over the producer's source, the same on every box.

    Each file contributes its path and length before its bytes, so moving
    code from one file to the other is a different digest.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256()
    for relative in PRODUCER_SOURCES:
        with open(os.path.join(root, relative), "rb") as fh:
            data = fh.read()
        digest.update(relative.encode() + b"\0" + str(len(data)).encode() + b"\0")
        digest.update(data)
    return digest.hexdigest()


class CachedSizeCampaign(Campaign):
    """``Campaign`` whose capture sizes come from a cached inventory.

    The capture files are immutable once written -- the campaign wrote them
    during its load phase and only ever reads them back -- so a basename->bytes
    inventory taken from the same tree is as good as a stat, and 36,423 stats
    over NFS are not.
    """

    def __init__(self, workspace: str, sizes: dict[str, int] | None) -> None:
        super().__init__(workspace)
        self._sizes = sizes

    def capture_files_for(self, names):
        if self._sizes is None:
            return super().capture_files_for(names)
        out = []
        missing = []
        for name in names:
            rec = self.entries.get(name)
            if rec is None:
                continue
            path = os.path.join(self.capture_root, rec["path"])
            size = self._sizes.get(os.path.basename(path))
            if size is None:
                missing.append(path)
                size = 0
            out.append((path, size))
        if missing:
            raise SystemExit(
                f"{len(missing)} capture files absent from the size cache, "
                f"first {missing[0]}; rerun with --stat")
        return out


def load_sizes(path: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            f = line.rsplit(None, 1)
            if len(f) == 2:
                try:
                    sizes[f[0]] = int(f[1])
                except ValueError:
                    pass
    return sizes


def row_id_of(row: dict) -> str | None:
    found = set(UNITS_RE.findall(" ".join(str(a) for a in row.get("argv", []))))
    return found.pop() if len(found) == 1 else None


def build_manifest(campaign: Campaign, row_id: str, produced_by: dict,
                   argv: "list | None" = None) -> dict:
    """One row's read set, in the order the row consumes it.

    Captures come first because ``prefetch_capture`` runs before the layer's
    weights are touched; the seed wire the row re-verifies comes last, for the
    same reason -- within each group the order is the consumer's own, and the
    prewarm reader walks ``entries`` in order, so a warm cut short by ARC
    headroom is cut at the end of the row's own read, not in the middle of its
    captures.

    ``argv`` is the row's own command line.  It is what names
    ``--seed-wire-dir``, and reading the plan instead is what made every
    manifest of ``extension-r1024-02`` declare ``seeds: 0`` while the row read
    9.4-19 GB of wire off cold spindles at 41 MB/s.
    """
    plan = campaign.row_plan(row_id, argv)
    entries = []
    for path, size in plan["_captures"]:
        entries.append({"path": path, "offset": 0, "bytes": int(size), "sha256": None})
    for path, offset, length in plan["_extents"]:
        entries.append({"path": path, "offset": int(offset), "bytes": int(length),
                        "sha256": None})
    phases = [{"name": "captures", "bytes": plan["capture_bytes"],
               "cumulative_bytes": plan["capture_bytes"]},
              {"name": "weight_extents", "bytes": plan["weight_bytes"],
               "cumulative_bytes": plan["capture_bytes"] + plan["weight_bytes"]}]
    for path, size in plan["_seeds"]:
        entries.append({"path": path, "offset": 0, "bytes": int(size), "sha256": None})
    phases.append({"name": "seeds", "bytes": plan["seed_bytes"],
                   "cumulative_bytes": plan["total_bytes"]})
    named_seed_dir = None if argv is None else argv_value(argv, SEED_WIRE_DIR_FLAG)
    # The wire directory's own files, not the row's seed total: a row carries
    # ``--seed-checkpoint`` as well, and counting both together would let a
    # present checkpoint mask an empty wire directory -- the same silent zero
    # with one extra file in it.
    wire_root = None if not named_seed_dir else os.path.normpath(named_seed_dir)
    from_wire = [] if wire_root is None else [
        path for path, _ in plan["_seeds"]
        if path == wire_root or path.startswith(wire_root.rstrip("/") + "/")]
    if named_seed_dir and not from_wire:
        # The defect this gate exists for produced exactly this shape: a row
        # that reads 9.4-19 GB of wire, and a manifest that says ``seeds: 0``.
        # A miss count of zero against a directory the row names is a broken
        # read set, not an empty one, so it fails closed here rather than
        # warming nothing at 41 MB/s.
        raise SystemExit(
            f"{row_id}: argv names {SEED_WIRE_DIR_FLAG} {named_seed_dir} but no "
            "readable file was found there; refusing to declare a read set "
            "that omits the row's seed wire")
    for e in entries:
        if not e["path"].startswith(SHARED_MOUNT + "/"):
            raise SystemExit(f"{row_id}: entry outside the shared mount: {e['path']}")
        if e["bytes"] <= 0:
            raise SystemExit(f"{row_id}: zero-length entry: {e['path']}")
    # The key set is the one ``prismabuild.core.validate_data_manifest``
    # accepts exactly; everything this campaign knows and PrismaBuild does not
    # goes under ``annotations``, which the contract carries but never reads.
    manifest = {
        "schema": SCHEMA,
        "produced_by": produced_by,
        "mount_prefix": SHARED_MOUNT,
        "annotations": {
            "row_id": row_id,
            "group": plan["group"],
            "sha256_present": False,
            "sha256_absent_reason": (
                "hashing 1023 GB of capture bytes costs more than the prewarm "
                "saves; the manifest file itself is content-addressed in the "
                "CAS, which is what binds it to the action key"),
            "counts": {
                "captures": plan["capture_files"],
                "weight_extents": plan["weight_extents"],
                "seeds": plan["seed_files"],
            },
            "bytes": {
                "captures": plan["capture_bytes"],
                "weight_extents": plan["weight_bytes"],
                "seeds": plan["seed_bytes"],
            },
            # The directory the row's argv named, so a reader can tell a row
            # that declared no seeds from one whose seed directory was empty
            # or unreadable when the manifest was built.
            "seed_wire_dir": (None if argv is None
                              else argv_value(argv, SEED_WIRE_DIR_FLAG)),
            # Where one phase of the row's read ends and the next begins, as a
            # running byte sum over ``entries``.  The prewarm reader walks the
            # list in order and can stop at a byte budget, so a consumer that
            # warms only what fits has a boundary to stop on that is a
            # property of the row's read order rather than a guess.  Carried,
            # not read: PrismaBuild reads only ``annotations.row_id`` today.
            "phases": phases,
        },
        "entry_count": len(entries),
        "total_bytes": plan["total_bytes"],
        "entries": entries,
    }
    return check_manifest(manifest, where=row_id)


def deterministic_provenance(workspace: str, campaign: Campaign,
                             size_source: str) -> dict:
    """``produced_by`` that two submissions of one campaign agree on, byte for byte.

    ``pbrun`` ingests the manifest as a content-addressed input and seals its
    digest into the action key, so a field that changes between runs -- a
    hostname, a clock reading -- gives the same row a new key on every submit.
    That is not a cosmetic loss: a finished row stops being a cache hit and is
    re-run, which is the opposite of what re-running ``submit`` is for.  Every
    field here is a property of the campaign and the producer, not of the run.

    The producer is named by the digest of its own source, not by ``git
    rev-parse HEAD``.  ``plan`` publishes these bytes and runs as a
    PrismaBuild action, where ``HEAD`` is pbrun's snapshot commit rather than
    the submitter's, and inside a container, where git exits 128 on a checkout
    another uid owns and ``git_commit`` reads that as ``"unknown"``.  A
    host-side ``submit`` re-deriving the same row would then write a different
    commit, and a different action key, for the same read set.  The source
    digest is the same wherever the producer runs.
    """
    return {
        "tool": "prismaquant/experiments/glm_data_manifests.py",
        "producer_sha256": producer_source_sha256(),
        "workspace": workspace,
        "capture_manifest": campaign.capture_manifest_path,
        "size_source": size_source,
    }


# ---------------------------------------------------------------------------
# The post-campaign GPU passes
# ---------------------------------------------------------------------------
#
# A campaign row is not the only GPU work that reads the shared mount. After
# the rows merge, three more passes run: the joint AURA preparation and cost
# passes (``prismaquant.tessera_joint_aura prepare`` / ``run``), the allocation
# handoff (``prismaquant.tessera_joint_allocation``) and the serving export.
# The joint pass reads about 4.75 TB against dl380g10's 240 GiB ARC, so its
# manifest has to name a *consumption order* the prewarm loop can window on:
# warming the whole read set is not possible, and warming an arbitrary 240 GiB
# of it warms the wrong bytes. The per-layer phase boundaries below are that
# order, taken from the code that reads them rather than from prose.
#
# These builders are deliberately torch-free and import nothing from the
# ``prismaquant`` package: they run in the same CPU environment the campaign
# manifests are built in, and importing ``prismaquant`` pulls in
# ``format_registry`` and therefore torch. The few constants and filename
# rules they need are restated with the module and line that owns each.

#: ``prismaquant.tessera_joint_aura.SCHEMA``.
JOINT_PLAN_SCHEMA = "prismaquant.tessera_joint_aura.plan.v1"
#: ``prismaquant.tessera_joint_aura.PREPARED_SCHEMA``.
JOINT_PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v3"
#: ``prismaquant.cost_stage_checkpoint.MANIFEST_SCHEMA`` / ``UNIT_SCHEMA``.
CHECKPOINT_MANIFEST_SCHEMA = "prismaquant.cost_stage_checkpoint.manifest.v1"
CHECKPOINT_UNIT_SCHEMA = "prismaquant.cost_stage_checkpoint.unit.v1"
#: ``prismaquant.layer_config.LAYER_CONFIG_META_KEY``: the one reserved
#: non-qname key an assignment file may carry.
LAYER_CONFIG_META_KEY = "__prismaquant__"

JOINT_ENTRY_POINT = "prismaquant.tessera_joint_aura"
ALLOCATION_ENTRY_POINT = "prismaquant.tessera_joint_allocation"
AQUA_ENTRY_POINT = "prismaquant.aqua_activation_cost"
EXPORT_ENTRY_POINT = "tessera.experiments.export_tessera_serving"

#: ``prismaquant.cost_streaming.layer_index_for_qname`` without the runner:
#: the runner asks its profile for the layers prefix and matches the index
#: that follows it. A roster that does not resolve to exactly one prefix is
#: refused rather than grouped by a guess.
LAYER_QNAME_RE = re.compile(r"^(?P<prefix>.*\.layers\.)(?P<index>\d+)(?:\.|$)")

#: The reason ``sha256`` is null on every entry of a post-campaign manifest.
SHA256_ABSENT_REASON = (
    "hashing the pass's read set costs more than the prewarm saves; the "
    "manifest file itself is content-addressed in the CAS, which is what "
    "binds it to the action key")


def _cache_weight_filename(qname: str, fmt: str) -> str:
    """``prismaquant.production_weight_cache._cache_weight_filename``.

    Restated because importing the package pulls in torch, and the joint
    manifest has to name the render file a cell will open. It is a pure string
    rule; ``tests/test_glm_joint_data_manifest_at_submit.py`` holds it against
    a real render filename from the frozen campaign.
    """
    safe = qname.replace("/", "__").replace(".", "_")
    return f"{safe}__{fmt}.pt"


def _read_json(path: str, label: str) -> dict:
    try:
        with open(path) as handle:
            return json.load(handle)
    except OSError as exc:
        raise SystemExit(f"{label}: unreadable {path}: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"{label}: {path} is not JSON: {exc}") from exc


def _bound(record, label: str) -> str:
    """The path of a ``{"path", "sha256"}`` binding, or a refusal."""
    if not isinstance(record, dict) or "path" not in record:
        raise SystemExit(f"{label}: a bound path is required, got {record!r}")
    return str(record["path"])


def _dir_sizes(directory: str) -> dict:
    """``{basename: bytes}`` for one named directory, not a tree.

    One ``scandir`` of a directory the plan names, never a walk of the mount:
    a recursive scan of ``/mnt/shared`` is the RPC storm that stalls the
    fleet's GPU clients. The reply already carries the attributes, so a
    36,423-entry directory costs one round of directory reads rather than one
    ``stat`` per file.
    """
    local = to_pool(directory)
    out = {}
    try:
        with os.scandir(local) as scan:
            for item in scan:
                try:
                    if not item.is_file(follow_symlinks=False):
                        continue
                    out[item.name] = item.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return {}
    return out


def _file_size(path: str) -> int:
    try:
        return os.stat(to_pool(path)).st_size
    except OSError:
        return 0


def _required_size(path: str, label: str) -> int:
    """The size of a file the pass will open, or a refusal naming it.

    A post-campaign pass runs behind a chain of earlier steps, and the ordinary
    way for one of its inputs to be missing is that the step that writes it has
    not finished. Saying which file and which step is the difference between a
    submission refused in a second and an admitted action that dies twenty
    seconds in with no retry.
    """
    size = _file_size(path)
    if size <= 0:
        raise SystemExit(
            f"{label}: {path} is absent or empty, so the pass's read set "
            "cannot be derived; the step that writes it has not finished")
    return size


def _unit_state(path: str, size_hint: int = 0) -> dict:
    """The ``{anchors, wire_records}`` state inside a checkpoint unit shard.

    ``prismaquant.cost_stage_checkpoint._load_unit`` restated without the
    identity comparison it makes against a live run: this reads the envelope,
    checks the schema and the payload digest, and decodes the inner pickle.
    Both pickles are plain JSON-shaped data -- the shards carry no class
    references at all -- so this stays torch-free.
    """
    try:
        with open(to_pool(path), "rb") as handle:
            envelope = pickle.load(handle)
    except OSError as exc:
        raise SystemExit(f"checkpoint unit shard is unreadable: {path}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - a corrupt shard is a refusal
        raise SystemExit(f"checkpoint unit shard is corrupt: {path}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != CHECKPOINT_UNIT_SCHEMA:
        raise SystemExit(f"{path}: not a {CHECKPOINT_UNIT_SCHEMA} envelope")
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise SystemExit(f"{path}: checkpoint unit shard has no byte payload")
    if hashlib.sha256(payload).hexdigest() != envelope.get("payload_sha256"):
        raise SystemExit(f"{path}: checkpoint unit payload_sha256 differs")
    state = pickle.loads(payload)
    if not isinstance(state, dict) or "anchors" not in state or "wire_records" not in state:
        raise SystemExit(f"{path}: incomplete measured anchor journal")
    del size_hint
    return state


def _layer_index_of(names) -> dict:
    """``{qname: layer}`` for a whole roster, or a refusal.

    ``prepare_cache`` walks ``range(runner.num_layers)`` and groups the units
    by ``runner.layer_index_for_qname``; the phase boundaries are that
    grouping. A roster whose names do not all sit under one ``.layers.``
    prefix has no such grouping, and a manifest that guessed one would declare
    a warm order the pass does not follow.
    """
    prefixes, index = set(), {}
    for name in names:
        match = LAYER_QNAME_RE.match(str(name))
        if match is None:
            raise SystemExit(
                f"{name} names no transformer layer, so the joint pass's "
                "per-layer read order cannot be derived for it")
        prefixes.add(match.group("prefix"))
        index[name] = int(match.group("index"))
    if len(prefixes) != 1:
        raise SystemExit(
            "the campaign roster spans more than one layers prefix "
            f"({sorted(prefixes)}); refusing to group its read set by layer")
    return index


def _backbone_layer_count(model_path: str) -> int:
    """The actual streamed backbone depth, excluding GLM's MTP passthrough."""
    config = _read_json(os.path.join(to_pool(model_path), "config.json"),
                        "source model config")
    text = config.get("text_config")
    source = text if isinstance(text, dict) else config
    count = source.get("num_hidden_layers")
    if type(count) is not int or count <= 0:
        raise SystemExit("source config has no positive backbone num_hidden_layers")
    return count


def _source_layer_indices(weight_map: dict, prefix: str, count: int) -> list[int]:
    """Backbone layers actually installed by the streaming runner."""
    layers = set()
    for tensor in weight_map:
        match = LAYER_QNAME_RE.match(tensor)
        if (match is not None and match.group("prefix") == prefix
                and int(match.group("index")) < count):
            layers.add(int(match.group("index")))
    ordered = sorted(layers)
    if ordered != list(range(count)):
        raise SystemExit("source checkpoint has noncontiguous or absent layers")
    return ordered


class _Phases:
    """Entries plus the running byte sum PrismaBuild's prewarm loop windows on.

    Every file lands in the phase that reads it *first*: the contract refuses a
    repeated ``(path, offset)``, so a byte range that is read again later is
    counted in ``annotations.reread_bytes_by_phase`` instead of appearing
    twice. A phase object itself stays exactly ``{name, bytes,
    cumulative_bytes}``, which is what the loop reads.
    """

    def __init__(self) -> None:
        self.entries = []
        self.phases = []
        self.counts = {}
        self.bytes = {}
        self.reread = {}
        self._seen = set()
        self._total = 0
        self._phase_bytes = 0
        self._phase = None

    def begin(self, name: str) -> None:
        self.end()
        self._phase = name
        self._phase_bytes = 0

    def end(self) -> None:
        if self._phase is None:
            return
        self.phases.append({"name": self._phase, "bytes": self._phase_bytes,
                            "cumulative_bytes": self._total})
        self._phase = None

    def add(self, path: str, offset: int, size: int, kind: str) -> None:
        if self._phase is None:
            raise SystemExit("internal: an entry was added outside a phase")
        offset, size = int(offset), int(size)
        if size <= 0:
            raise SystemExit(f"zero-length {kind} entry: {path}")
        key = (path, offset)
        if key in self._seen:
            # Read again in a later phase. The contract carries it once; the
            # second read is accounted for where a consumer can see it.
            self.reread[self._phase] = self.reread.get(self._phase, 0) + size
            return
        self._seen.add(key)
        self.entries.append({"path": path, "offset": offset, "bytes": size,
                             "sha256": None})
        self._phase_bytes += size
        self._total += size
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.bytes[kind] = self.bytes.get(kind, 0) + size

    @property
    def total_bytes(self) -> int:
        return self._total


class _ReadPlanV2:
    """One unique file-range roster with explicit, repeatable phase reads."""

    def __init__(self) -> None:
        self.entries = []
        self._indices = {}
        self._phases = []
        self._current = None

    def begin(self, name: str) -> None:
        self.end()
        self._current = {"name": name, "entry_indices": [], "seen": set()}

    def end(self) -> None:
        if self._current is not None:
            self._phases.append(self._current)
            self._current = None

    def add(self, path: str, offset: int, size: int, kind: str) -> None:
        if self._current is None:
            raise SystemExit("internal: a V2 read was added outside a phase")
        offset, size = int(offset), int(size)
        if offset < 0 or size <= 0:
            raise SystemExit(f"invalid {kind} file range: {path}@{offset}+{size}")
        key = (path, offset)
        index = self._indices.get(key)
        if index is None:
            index = len(self.entries)
            self._indices[key] = index
            self.entries.append({"path": path, "offset": offset,
                                 "bytes": size, "sha256": None})
        else:
            # PB V2 admits one entry for a (path, offset). If independently
            # coalesced source layers cover different lengths at that offset,
            # warm their conservative union on every reference.
            self.entries[index]["bytes"] = max(self.entries[index]["bytes"], size)
        if index not in self._current["seen"]:
            self._current["seen"].add(index)
            self._current["entry_indices"].append(index)

    def manifest(self, produced_by: dict, annotations: dict, *, where: str) -> dict:
        self.end()
        phases, cumulative = [], 0
        for phase in self._phases:
            size = sum(self.entries[index]["bytes"] for index in phase["entry_indices"])
            cumulative += size
            phases.append({"name": phase["name"],
                           "entry_indices": phase["entry_indices"],
                           "bytes": size, "cumulative_bytes": cumulative})
        result = {"schema": SCHEMA_V2, "produced_by": produced_by,
                  "mount_prefix": SHARED_MOUNT, "annotations": annotations,
                  "entries": self.entries, "entry_count": len(self.entries),
                  "total_bytes": sum(entry["bytes"] for entry in self.entries),
                  "read_plan": {"phases": phases, "read_bytes": cumulative}}
        return check_manifest_v2(result, where=where)


def check_manifest_v2(manifest: dict, *, where: str = "data manifest V2") -> dict:
    """Mirror the published PB V2 unique-entry and ordered-read gates."""
    if set(manifest) != MANIFEST_KEYS | {"read_plan"} or manifest.get("schema") != SCHEMA_V2:
        raise SystemExit(f"{where}: complete {SCHEMA_V2} fields required")
    common = {key: value for key, value in manifest.items() if key != "read_plan"}
    common["schema"] = SCHEMA
    check_manifest(common, where=where)
    if "phases" in manifest["annotations"]:
        raise SystemExit(f"{where}: V2 uses read_plan, not annotations.phases")
    plan = manifest["read_plan"]
    if not isinstance(plan, dict) or set(plan) != {"phases", "read_bytes"}:
        raise SystemExit(f"{where}: complete read_plan required")
    phases = plan["phases"]
    if not isinstance(phases, list) or not phases:
        raise SystemExit(f"{where}: read_plan needs nonempty phases")
    names, used, cumulative, reads = set(), set(), 0, 0
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {
                "name", "entry_indices", "bytes", "cumulative_bytes"}:
            raise SystemExit(f"{where}: malformed read phase")
        name, indices = phase["name"], phase["entry_indices"]
        if (not isinstance(name, str) or not name or name.strip() != name
                or "\x00" in name or name in names or not isinstance(indices, list)):
            raise SystemExit(f"{where}: invalid or repeated read phase name/indices")
        names.add(name)
        reads += len(indices)
        if reads > MAX_READS:
            raise SystemExit(f"{where}: read references exceed {MAX_READS}")
        local, size = set(), 0
        for index in indices:
            if (type(index) is not int or index < 0 or index >= len(manifest["entries"])
                    or index in local):
                raise SystemExit(f"{where}: invalid or repeated phase entry index")
            local.add(index)
            used.add(index)
            size += manifest["entries"][index]["bytes"]
        cumulative += size
        if (type(phase["bytes"]) is not int or phase["bytes"] != size
                or type(phase["cumulative_bytes"]) is not int
                or phase["cumulative_bytes"] != cumulative):
            raise SystemExit(f"{where}: phase byte accounting differs")
    if used != set(range(len(manifest["entries"]))) or type(plan["read_bytes"]) is not int or plan["read_bytes"] != cumulative:
        raise SystemExit(f"{where}: read_plan coverage or read_bytes differs")
    return manifest



def _campaign_with_cached_capture_sizes(workspace: str):
    """A ``Campaign`` whose capture sizes came from directory reads, not stats.

    The joint pass touches every one of the campaign's capture files, and
    stating them one at a time is 36,423 NFS round trips for a number the
    directory reply already carried. The capture manifest names the
    directories, so each is read once with ``scandir``; nothing walks the
    mount.
    """
    campaign = Campaign(workspace)
    directories = {os.path.dirname(os.path.join(campaign.capture_root, record["path"]))
                   for record in campaign.entries.values()}
    sizes = {}
    for directory in sorted(directories):
        sizes.update(_dir_sizes(directory))
    return CachedSizeCampaign(workspace, sizes or None)


def _add_capture(track: "_Phases", found: list) -> None:
    for path, size in found:
        track.add(path, 0, size, "captures")


def _render_file(owner: str, render_sizes: dict, name: str, fmt: str):
    """The cell's PWC shard and the bytes the campaign wrote, or zero.

    Zero means the campaign adopted this rung rather than encoding it, so it
    has a wire and no ``.pt``. The head synthesizes one from that wire
    (``_resolve_render_origin``), which is why the caller reads the wire
    early and leaves the shard out of the read set: declaring a file that is
    not there is the defect that made every #524 warm finish ``partial``.
    """
    filename = _cache_weight_filename(name, fmt)
    return (os.path.join(owner, "cache", filename),
            render_sizes.get(owner, {}).get(filename, 0))


def _add_render(track: "_Phases", owner: str, render_sizes: dict, name: str,
                fmt: str) -> None:
    """Declare a cell's render where the campaign wrote one."""
    path, size = _render_file(owner, render_sizes, name, fmt)
    if size:
        track.add(path, 0, size, "renders")


def _is_resume(argv) -> bool:
    """Whether this submission forwards the pass's own resume switch."""
    return bool(argv) and "--resume" in [str(item) for item in argv]


def _canonical_sha256(value) -> str:
    """``cost_stage_checkpoint.canonical_json_sha256`` without the import.

    The acting image computes the journal's identity digest with that module.
    This reproduces its canonical encoding (sorted keys, no spaces) so a
    manifest built on a CPU box can check the journal it is about to seal. A
    disagreement is a refusal, never a repair: the sealed digest is what the
    preparing action compares against at run time.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qualification_journal_state(plan: dict, *, roster):
    """``(completed units, journal identity sha256)``, or ``None`` if none.

    ``None`` means the pass never journaled a unit, so a resume has nothing to
    replay and seals an empty frontier. Everything else that is not exactly
    the journal this campaign wrote is a refusal: a journal whose identity does
    not hash to its own digest, one qualified against another campaign
    checkpoint, or one whose unit roster is not this campaign's. Sealing a
    prefix against any of those would name reads the resumed action is not
    going to make.
    """
    journal = os.path.join(plan["output_root"], "prepare", "qualification")
    manifest_path = os.path.join(journal, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    manifest = _read_json(manifest_path, "qualification journal manifest")
    if (manifest.get("schema") != _replay_module.JOURNAL_MANIFEST_SCHEMA
            or manifest.get("stage") != _replay_module.QUALIFICATION_STAGE):
        raise SystemExit(
            f"{manifest_path}: not a {_replay_module.QUALIFICATION_STAGE!r} "
            "journal; refusing to seal a replay frontier against it")
    identity = manifest.get("identity")
    identity_sha256 = manifest.get("identity_sha256")
    if (not isinstance(identity, dict) or not isinstance(identity_sha256, str)
            or len(identity_sha256) != 64):
        raise SystemExit(f"{manifest_path}: journal identity is missing or malformed")
    if _canonical_sha256(identity) != identity_sha256:
        raise SystemExit(
            f"{manifest_path}: journal identity does not hash to its own digest; "
            "refusing to seal a replay frontier")
    checkpoint = _bound(plan["inputs"]["merged_checkpoint"],
                        "plan inputs.merged_checkpoint")
    seal = _read_json(checkpoint, "merged checkpoint").get("identity_sha256")
    if identity.get("campaign_checkpoint_sha256") != seal:
        raise SystemExit(
            f"{manifest_path}: journal campaign checkpoint "
            f"{identity.get('campaign_checkpoint_sha256')!r} is not the merged "
            f"checkpoint's seal {seal!r}; re-plan rather than resume against a "
            "different campaign checkpoint")
    units = manifest.get("units")
    listed = [row.get("qname") for row in units] if isinstance(units, list) else None
    files = [row.get("file") for row in units] if isinstance(units, list) else None
    if (listed is None or any(not isinstance(name, str) for name in listed)
            or any(not isinstance(name, str) for name in files)
            or len(set(listed)) != len(listed)):
        raise SystemExit(f"{manifest_path}: journal unit roster is malformed")
    if sorted(listed) != sorted(roster):
        raise SystemExit(
            f"{manifest_path}: journal roster of {len(listed)} unit(s) is not "
            f"this campaign's {len(roster)}-unit roster; refusing to seal a "
            "replay frontier over a different unit set")
    completed = sorted(row["qname"] for row in units
                       if os.path.isfile(os.path.join(journal, row["file"])))
    return completed, identity_sha256


def _checkpoint_unit_shards(track: "_Phases", checkpoint: str, roster) -> dict:
    """Declare one shard of the merged checkpoint per unit, or refuse.

    The roster comes from the campaign plan's own members rather than from the
    checkpoint manifest: that manifest is the campaign's whole identity record
    -- 7.2 GB of JSON on the GLM census -- and the only thing this needs from
    it is which shard belongs to which unit. That mapping is not a guess.
    ``cost_stage_checkpoint.unit_path`` names a shard by the SHA-256 of the
    unit qname, and ``load_measured_anchor_input`` refuses a manifest whose
    ``file`` differs from it, so the canonical path *is* the contract. The
    manifest is still required to be present, because the merge publishes it
    last and a pass cannot be submitted before it exists.
    """
    parts = checkpoint + ".parts"
    unit_sizes = _dir_sizes(os.path.join(parts, "units"))
    if not unit_sizes:
        # Step 2's merge publishes the shards and then the manifest that names
        # them. A joint pass submitted before that has no read set at all, and
        # a manifest built from a half-written merge would warm the wrong
        # bytes, so this fails closed on the directory it looked in.
        raise SystemExit(
            f"the merged checkpoint's unit shards are absent: {parts}/units; "
            "the joint pass cannot be submitted before the campaign merge "
            "publishes them")
    states = {}
    for qname in roster:
        filename = hashlib.sha256(str(qname).encode("utf-8")).hexdigest() + ".pkl"
        size = unit_sizes.get(filename)
        if not size:
            raise SystemExit(
                "the merged checkpoint's shard for "
                f"{qname} is absent: {os.path.join(parts, 'units', filename)}; "
                "the joint pass cannot be submitted before the campaign merge "
                "publishes it")
        path = os.path.join(parts, "units", filename)
        track.add(path, 0, size, "checkpoint_units")
        states[qname] = path
    return states


def _campaign_roster(campaign_plan_path: str) -> list:
    """Every unit the campaign priced, in the order the head reads them.

    ``load_measured_anchor_input`` walks ``sorted(names)``; the names are the
    union of the plan rows' members, which it also checks against the census.
    """
    plan = _read_json(campaign_plan_path, "campaign plan")
    names = [name for row in plan["rows"] for name in row["members"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"{campaign_plan_path}: a unit belongs to two rows")
    return sorted(names)


def _joint_source_authentication_schedule(model_path: str, roster: list):
    """Full-shard SHA at the first streamed source read, by actual layer.

    Joint preparation owns a complete canonical capture descriptor owner.
    The streaming loader hashes a shard through that owner before its first
    head, visual or body tensor read; completion authenticates only shards no
    streamed tensor needed. The index is the source map the loader consumes,
    so no filename or arbitrary shard-number heuristic decides the phase.
    """
    model = os.path.abspath(model_path)
    index = _read_json(os.path.join(model, "model.safetensors.index.json"),
                       "joint source checkpoint index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("joint source checkpoint index has no weight_map")
    prefix = LAYER_QNAME_RE.match(roster[0]).group("prefix")
    layers = _source_layer_indices(weight_map, prefix,
                                   _backbone_layer_count(model))
    base = prefix.removesuffix("layers.")
    head_prefixes = (base + "embed_tokens.", base + "norm.",
                     base + "rotary_emb.", "lm_head.", "model.visual.")
    first_use = {}
    for tensor, filename in weight_map.items():
        if (not isinstance(tensor, str) or not isinstance(filename, str)
                or os.path.basename(filename) != filename):
            raise SystemExit("joint source checkpoint index has an unsafe tensor or shard")
        match = LAYER_QNAME_RE.match(tensor)
        if match is not None and match.group("prefix") == prefix:
            layer = int(match.group("index"))
            phase = layer if layer in layers else None  # MTP passthrough is not installed
        elif tensor.startswith(head_prefixes):
            phase = -1  # head/visual are materialized before layer 0
        else:
            phase = None  # e.g. an auxiliary tower the runner never installs
        prior = first_use.get(filename)
        if prior is None or phase == -1 or (phase is not None and phase < prior):
            first_use[filename] = phase
        elif filename not in first_use:
            first_use[filename] = None
    sizes = _dir_sizes(model)
    schedule = {"metadata": [], "head": [],
                "layers": {layer: [] for layer in layers}, "completion": []}
    for filename, size in sorted(sizes.items()):
        if filename.endswith((".json", ".model", ".txt", ".jinja", ".py")):
            schedule["metadata"].append((os.path.join(model, filename), size))
    for filename, phase in sorted(first_use.items()):
        size = sizes.get(filename)
        if not size:
            raise SystemExit(f"joint source shard is absent: {model}/{filename}")
        row = (os.path.join(model, filename), size)
        if phase == -1:
            schedule["head"].append(row)
        elif phase is None:
            schedule["completion"].append(row)
        else:
            schedule["layers"][phase].append(row)
    return schedule


def _source_tensor_extents(campaign: Campaign, prefixes, *, required=True
                           ) -> list[tuple[str, int, int]]:
    """Every indexed tensor matching the streaming loader's installed prefixes.

    ``Campaign.weight_extents_for`` owns record rounding/coalescing for the
    campaign's quantizable ``.weight`` units. The joint runner installs the
    complete source layer, including norms and GLM's named buffers. Use that
    same header cache and 1 MiB record geometry for the complete prefix.
    """
    by_shard = {}
    for tensor, shard in campaign.weight_map.items():
        if not tensor.startswith(tuple(prefixes)):
            continue
        header = campaign._header(shard)
        meta = header.get(tensor)
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            raise SystemExit(f"{tensor}: source index has no safetensors extent")
        lo, hi = meta["data_offsets"]
        start = header["__data_start__"]
        by_shard.setdefault(shard, []).append((start + lo, start + hi))
    if not by_shard and required:
        raise SystemExit(f"source checkpoint has no tensors for {prefixes}")
    out = []
    for shard, spans in sorted(by_shard.items()):
        path = os.path.join(campaign.model_dir, shard)
        eof = _required_size(path, "source layer shard")
        # safe_open reads a shard header even when its first tensor lies much
        # later in the file. Declare its ZFS record at this shard's first use.
        spans.append((0, campaign._header(shard)["__data_start__"]))
        merged = []
        for lo, hi in sorted((a - a % RECORD_SIZE,
                              min(eof, b + (-b) % RECORD_SIZE))
                             for a, b in spans):
            if hi <= lo:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out.extend((path, lo, hi - lo) for lo, hi in merged)
    return out


def _full_source_layer_extents(campaign: Campaign, layer: int,
                               prefix: str) -> list[tuple[str, int, int]]:
    """All streamed source tensors installed for this layer."""
    return _source_tensor_extents(campaign, (f"{prefix}{layer}.",))


def _joint_source_identity_cache_for_manifest(plan: dict, schedule: dict):
    """Declare a verified existing cache instead of a conditional 643 GB read.

    This is only the torch-free submission gate: execution replays the full
    streamed-model validator, including semantic config and the complete
    checkpoint map, before it adopts a proof. A changed cache or source file
    therefore refuses rather than falling back to reads the manifest omitted.
    """
    binding = plan.get("source_identity_cache")
    cache = (_bound(binding, "source identity cache") if binding is not None else
             os.path.join(plan["output_root"], "prepare", "source-identity.json"))
    if not os.path.isfile(to_pool(cache)):
        return None
    if binding is not None and sha256_file(to_pool(cache)) != binding["sha256"]:
        raise SystemExit("bound joint source identity cache checksum changed")
    record = _read_json(to_pool(cache), "joint source identity cache")
    if (record.get("schema") != "prismaquant.streamed_model.identity_cache.v1"
            or record.get("source") != plan["model"]):
        raise SystemExit("joint source identity cache is foreign or malformed")
    identity = record.get("identity")
    fingerprints = record.get("fingerprints")
    if not isinstance(identity, dict) or not isinstance(fingerprints, list):
        raise SystemExit("joint source identity cache has no full-shard proof")
    by_path = {row.get("path"): row for row in fingerprints if isinstance(row, dict)}
    shards = identity.get("shards")
    expected = {path for path, _size in (
        schedule["head"] + schedule["completion"] +
        [row for rows in schedule["layers"].values() for row in rows])}
    if (not isinstance(shards, list) or len(by_path) != len(fingerprints)
            or {row.get("path") for row in shards} != expected
            or set(by_path) != expected):
        raise SystemExit("joint source identity cache lacks exact shard coverage")
    capture = _read_json(_bound(plan["canonical_capture"], "canonical capture"),
                         "canonical capture")
    source_files = (capture.get("identity") or {}).get("source_files")
    if not isinstance(source_files, dict):
        raise SystemExit("canonical capture has no source-file SHA roster")
    for row in shards:
        path = row["path"]
        fingerprint = by_path[path]
        stat = os.stat(to_pool(path))
        # The proof is mount-instance strict: ``device`` is compared with the
        # four stat fields, because those four identify an object *within* a
        # filesystem and not across filesystems (see docs/ARCHITECTURE.md
        # §"source identity"). A host that reaches the same bytes through
        # another mount instance therefore refuses here, and the manifest's
        # ``source_identity_cache_host`` keeps that as a placement constraint
        # rather than a hint. The refusal names both sides, so a reader can
        # tell "another mount" from "another object" without re-statting.
        if (row.get("sha256") != source_files.get(os.path.basename(path))
                or any(fingerprint.get(key) != getattr(stat, attribute)
                       for key, attribute in (('device', 'st_dev'), ('inode', 'st_ino'),
                                              ('size', 'st_size'), ('mtime_ns', 'st_mtime_ns'),
                                              ('ctime_ns', 'st_ctime_ns')))):
            raise SystemExit(
                f"joint source identity cache no longer proves {path}: "
                f"recorded device {fingerprint.get('device')!r} with "
                f"inode {fingerprint.get('inode')!r}, size "
                f"{fingerprint.get('size')!r}; this mount instance reports "
                f"device {int(stat.st_dev)} with inode {int(stat.st_ino)}, "
                f"size {int(stat.st_size)}")
    return cache


def _joint_head(track: _Phases, plan_path: str, plan: dict, *, roster,
                prepared: str | None, source_transition: str | None = None):
    """The head phase: everything read before the first layer installs.

    Order is ``tessera_joint_aura.main`` and ``load_measured_anchor_input``:
    the census, the campaign plan and its receipts, the merged cost payload,
    the merged checkpoint manifest and every one of its unit shards, then the
    calibration input, the capture compatibility record, the projection
    backend and the canonical capture manifest. ``run`` additionally binds the
    prepared completion and the production cache it names.
    """
    inputs = plan["inputs"]
    track.add(plan_path, 0, _required_size(plan_path, "joint plan"), "plan")
    for key in ("census", "campaign_plan", "campaign_receipts", "merged_cost",
                "merged_checkpoint"):
        path = _bound(inputs[key], f"plan inputs.{key}")
        track.add(path, 0, _required_size(path, f"plan inputs.{key}"), "head")

    checkpoint = _bound(inputs["merged_checkpoint"], "plan inputs.merged_checkpoint")
    parts = checkpoint + ".parts"
    states = _checkpoint_unit_shards(track, checkpoint, roster)
    for key in ("calibration_input", "source_capture_compatibility", "canonical_capture"):
        record = plan.get(key)
        if record is None:
            continue
        path = _bound(record, f"plan {key}")
        track.add(path, 0, _required_size(path, f"plan {key}"), "head")
    backend = ((plan.get("execution") or {}).get("projection_backend") or {}).get("binary")
    if backend is not None:
        path = _bound(backend, "plan execution.projection_backend.binary")
        track.add(path, 0, _required_size(
            path, "plan execution.projection_backend.binary"), "head")
    prepared_cache = None
    if prepared is not None:
        track.add(prepared, 0, _required_size(prepared, "prepared completion"), "head")
        completion = _read_json(prepared, "prepared completion")
        if completion.get("schema") != JOINT_PREPARED_SCHEMA:
            raise SystemExit(
                f"{prepared}: schema is {completion.get('schema')!r}, expected "
                f"{JOINT_PREPARED_SCHEMA}")
        prepared_cache = _bound(completion["production_cache"], "prepared production_cache")
        track.add(prepared_cache, 0,
                  _required_size(prepared_cache, "prepared production_cache"), "head")
    if source_transition is not None:
        # ``tessera_joint_aura.execute`` loads the receipt before it reads the
        # prepared record it admits, so the receipt is a head read too.
        track.add(source_transition, 0,
                  _required_size(source_transition, "source transition receipt"), "head")
    return parts, states, prepared_cache


def _joint_cells(states: dict, wire_dir: str):
    """``{qname: [(fmt, wire path, wire bytes)]}`` for the measured rungs only.

    A unit's journal carries one anchor and one wire record per rung the
    campaign *measured*. The menu it was drawn from is larger: the row caches
    hold wire blobs for rungs no anchor priced, and the joint pass never opens
    them. Declaring them would warm bytes the pass does not read, at the
    expense of bytes it does.
    """
    cells = {}
    for qname, path in states.items():
        state = _unit_state(path)
        records = state["wire_records"]
        measured = sorted(anchor["format_name"] for anchor in state["anchors"])
        if set(measured) != set(records):
            raise SystemExit(
                f"{qname}: measured anchors and wire records differ in "
                f"{path}; refusing to declare a read set from it")
        out = []
        for fmt in measured:
            record = records[fmt]
            filename = record["file"]
            if os.path.basename(filename) != filename:
                raise SystemExit(f"{qname}@{fmt}: escaping wire filename {filename!r}")
            out.append((fmt, os.path.join(wire_dir, filename), int(record["blob_bytes"])))
        cells[qname] = out
    return cells


def build_joint_pass_manifest(plan_path, *, command, produced_by, argv=None,
                              prepared=None, source_transition=None):
    """The joint AURA pass's read set, in the order the pass consumes it.

    ``command`` is ``prepare`` or ``run``. ``source_transition`` is the
    closed receipt a resumed run consumes a prepared record under; it is a
    head read.

    The head phase is everything read before the first layer installs. Then
    bounded complete-unit ``layer-<L>-part-<P>`` phases for windowed prepare,
    in ascending layer order, because ``prepare_cache`` walks
    ``range(runner.num_layers)`` and journals each completed unit. Run keeps
    one ``layer-<L>`` phase per layer. Within a layer the order is the
    runner's: the layer's source byte extents first (the streaming context
    prefetches them on ``install``), then, for each of the layer's units in
    sorted name order, that unit's capture file and then each measured rung's
    render and, for preparation, wire. A plan that declares no
    ``qualification_window`` runs the
    older whole-layer window instead -- all of the layer's captures, then its
    renders and wires -- and the order here follows the plan.

    A rung the campaign adopted has a wire and no decoded shard, and the head
    decodes one from that wire before any layer installs
    (``_resolve_render_origin``). Those wires are declared in the head for
    that reason, not in the layer that later verifies them; the shard itself
    is left out, because it does not exist when the manifest is built.

    Preparation authenticates each whole source shard by SHA256 through a
    held descriptor on its first streamed tensor read. Head and visual shards
    go in the head; body shards go before the earliest declared prefetch of
    one of their tensors; shards the runner never installs go in a final
    phase. The index supplies this exact tensor-to-shard map. The later
    source extents are separate reads, not a substitute for whole-shard
    authentication. A windowed prepare must have a bounded prefetch window
    that the runner settles before the next phase; otherwise this linear ARC
    prefix cannot represent an in-flight future-layer read safely.

    ``run`` consumes prepared renders through PWC's verified load, so their
    first reads belong to their layers. It consumes no wire bodies and does
    not synthesize missing renders. Wire identities in its metadata are the
    historical preparation evidence; selected export authenticates current
    wire bytes. There is no whole-roster payload hash phase.
    The inherited layer ordering covers these reads; it does not yet express
    COST's forward/reverse execution or rewarm repeated source extents. The
    global (path, offset) deduplication records rereads only as annotations.

    ``sha256`` is null for the same reason it is null on a campaign row: the
    manifest is a residency hint whose own bytes are content-addressed, not an
    integrity claim about 4.75 TB of payload.
    """
    command = str(command)
    if command not in ("prepare", "run"):
        raise SystemExit(f"joint pass command must be prepare or run, not {command!r}")
    plan_path = os.path.abspath(plan_path)
    plan = _read_json(plan_path, "joint plan")
    if plan.get("schema") != JOINT_PLAN_SCHEMA:
        raise SystemExit(
            f"{plan_path}: schema is {plan.get('schema')!r}, expected "
            f"{JOINT_PLAN_SCHEMA}")
    if command == "run" and prepared is None:
        raise SystemExit(
            "the run command consumes a prepared completion; pass its path so "
            "its bytes and the production cache it names are declared")
    if source_transition is not None and command != "run":
        raise SystemExit("a source transition receipt is consumed by the run command only")

    inputs = plan["inputs"]
    campaign_plan_path = _bound(inputs["campaign_plan"], "plan inputs.campaign_plan")
    roster = _campaign_roster(campaign_plan_path)

    track = _Phases()
    track.begin("head")
    _parts, states, _prepared_cache = _joint_head(
        track, plan_path, plan, roster=roster,
        prepared=None if command == "prepare" else prepared,
        source_transition=source_transition)

    merged_cost = _bound(inputs["merged_cost"], "plan inputs.merged_cost")
    try:
        with open(to_pool(merged_cost), "rb") as handle:
            payload = pickle.load(handle)
    except OSError as exc:
        raise SystemExit(f"merged cost payload is unreadable: {merged_cost}: {exc}") from exc
    wire_dir = str(payload["provenance"]["wire_dir"])
    # The merged payload is the campaign's whole cost table, several GB once
    # unpickled, and the only field this needs is the wire directory the
    # journals' filenames are relative to. Drop it before the entry list is
    # built rather than holding both.
    del payload
    cells = _joint_cells(states, wire_dir)

    workspace = os.path.dirname(campaign_plan_path)
    campaign = _campaign_with_cached_capture_sizes(workspace)
    owners = {}
    for row in campaign.plan["rows"]:
        for name in row["members"]:
            owners[name] = row["dir"]
    layer_of = _layer_index_of(roster)
    by_layer = {}
    for name in roster:
        by_layer.setdefault(layer_of[name], []).append(name)
    layers = sorted(by_layer)
    render_sizes = {owner: _dir_sizes(os.path.join(owner, "cache"))
                    for owner in sorted(set(owners.values()))}

    # Still the head phase: a cell whose shard the campaign never wrote is
    # synthesized from its wire inside ``load_measured_anchor_input``, before
    # any layer installs, so that wire is read here and not first in the layer
    # that verifies it. Declaring it in the layer phase would leave the pass
    # to read it cold, which is the one thing this manifest exists to prevent.
    absent, synthesized_wire_bytes = [], 0
    for name in roster:
        for fmt, wire, wire_bytes in cells[name]:
            path, size = _render_file(owners[name], render_sizes, name, fmt)
            if size:
                continue
            if command == "run":
                raise SystemExit(f"prepared render is missing; COST will not synthesize it: {path}")
            absent.append(path)
            synthesized_wire_bytes += wire_bytes
            track.add(wire, 0, wire_bytes, "wires")

    source_schedule = None
    source_cache = None
    if command == "prepare":
        source_schedule = _joint_source_authentication_schedule(plan["model"], roster)
        source_cache = _joint_source_identity_cache_for_manifest(plan, source_schedule)
        for path, size in source_schedule["metadata"]:
            track.add(path, 0, size, "source_metadata")
        if source_cache is not None:
            track.add(source_cache, 0, _required_size(source_cache,
                "joint source identity cache"), "source_identity_cache")
        else:
            for path, size in source_schedule["head"]:
                track.add(path, 0, size, "source_authentication")

    prefix = LAYER_QNAME_RE.match(roster[0]).group("prefix")
    all_source_layers = (_source_layer_indices(campaign.weight_map, prefix,
                                               _backbone_layer_count(campaign.model_dir))
                         if source_schedule is None else sorted(source_schedule["layers"]))
    base = prefix.removesuffix("layers.")
    for path, offset, length in _source_tensor_extents(campaign, (
            base + "embed_tokens.", base + "norm.", base + "rotary_emb.",
            "lm_head."), required=False):
        track.add(path, offset, length, "source_head")

    if command == "run":
        binding = plan.get("source_identity_cache")
        if binding is not None:
            cache_path = _bound(binding, "source identity cache")
            if sha256_file(to_pool(cache_path)) != binding["sha256"]:
                raise SystemExit("bound joint source identity cache checksum changed")
            track.add(cache_path, 0, _required_size(cache_path,
                "source identity cache"), "source_identity_cache")

    # A plan that declares a qualification window runs one unit per capture
    # window; without one the whole layer's captures are loaded together.
    per_unit_window = plan.get("qualification_window") is not None
    # A resumed prepare reads its journal's units before the layer walk:
    # ``_qualification_replay`` re-authenticates each one's X/H capture, wires
    # and renders, and only then does the walk qualify the rest. Those are not
    # the layer/part reads a fresh manifest declares, so they get their own
    # phases in front of the walk -- and the roster, its exact order and the
    # journal identity are sealed beside them, because the resumed action
    # refuses to report a transition for a journal that moved after
    # submission (RobTand/prismaquant#607).
    replay, replayed = None, frozenset()
    if command == "prepare" and per_unit_window and _is_resume(argv):
        state = _qualification_journal_state(plan, roster=roster)
        completed, journal_identity_sha256 = state if state is not None else ([], None)
        replayed = frozenset(completed)
        # The replay block is windowed exactly as the walk is. One phase per
        # unit would be a phase table as long as the journal -- tens of
        # thousands of entries on the census -- and the submission declares one
        # PB progress phase per manifest phase, up to 2048. A part therefore
        # starts before a unit only once the current one has reached the
        # budget, so a part is never split inside a unit and never starts empty.
        replay_starts, part = {}, 0
        for name in completed:
            if not replay_starts:
                track.begin(_replay_module.replay_phase_name(part))
            elif track._phase_bytes >= _phase_module.MAX_PHASE_BYTES:
                part += 1
                track.begin(_replay_module.replay_phase_name(part))
            replay_starts[name] = track._phase
            _add_capture(track, campaign.capture_files_for([name]))
            for fmt, wire, wire_bytes in sorted(cells[name], key=lambda row: row[0]):
                track.add(wire, 0, wire_bytes, "wires")
                _add_render(track, owners[name], render_sizes, name, fmt)
        replay = _replay_module.seal_frontier(
            completed,
            {name: [fmt for fmt, _wire, _bytes in cells[name]] for name in roster},
            phase_start_units=replay_starts,
            journal_identity_sha256=journal_identity_sha256)
    source_lookahead = 0
    if command == "prepare" and per_unit_window:
        prefetch = plan.get("source_prefetch") or {}
        source_lookahead = prefetch.get("prefetch_lookahead")
        slots = prefetch.get("max_cache_slots")
        if (type(source_lookahead) is not int or source_lookahead < 1
                or type(slots) is not int or slots != source_lookahead + 1
                or layers != list(range(layers[-1] + 1))
                or any(max(0, target - source_lookahead) not in by_layer
                       for target in all_source_layers)):
            raise SystemExit(
                "joint prepare read-frontier requires contiguous source layers "
                "and max_cache_slots=prefetch_lookahead+1; otherwise source "
                "prefetch can outlive the phase that releases its bytes")
    phase_start_units = {}
    for layer in layers:
        names = sorted(by_layer[layer])
        part = 0
        source_layers = ([layer] if not (command == "prepare" and per_unit_window)
                         else [target for target in all_source_layers
                               if max(0, target - source_lookahead) == layer])
        if replay is not None and not source_layers and all(
                name in replayed for name in names):
            # Every unit of this layer is in the replay block, and the walk
            # declares no source extents of its own here, so this layer reads
            # nothing: beginning its phase would seal a transition over zero
            # bytes. A resuming walk has fewer phases than a fresh one, which
            # is the point -- the phases it does seal are the reads it makes.
            continue
        track.begin(_phase_module.phase_name(layer, part) if command == "prepare" and per_unit_window
                    else _run_progress_module.layer_phase_name(layer))
        for source_layer in source_layers:
            if source_schedule is not None and source_cache is None:
                for path, size in source_schedule["layers"][source_layer]:
                    track.add(path, 0, size, "source_authentication")
            for path, offset, length in _full_source_layer_extents(
                    campaign, source_layer,
                    LAYER_QNAME_RE.match(names[0]).group("prefix")):
                track.add(path, offset, length, "source_extents")
        if command == "run":
            # The streaming cost pass re-reads the layer's renders through the
            # production cache; it takes no captures and no wire bytes.
            for name in names:
                for fmt, _wire, _bytes in cells[name]:
                    _add_render(track, owners[name], render_sizes, name, fmt)
            continue
        captures = {name: campaign.capture_files_for([name]) for name in names}
        if not per_unit_window:
            for name in names:
                _add_capture(track, captures[name])
        for name in names:
            if name in replayed:
                # Already authenticated by ``_qualification_replay`` in the
                # replay phase that declared its bytes; the walk does not read
                # it again, so it draws no boundary here.
                continue
            if command == "prepare" and per_unit_window:
                # Only a complete unit is a durable progress boundary. Keep a
                # first unit whose source extents make the phase oversized;
                # the PB reader can still warm an entry-aligned prefix.
                if track._phase_bytes >= _phase_module.MAX_PHASE_BYTES:
                    part += 1
                    track.begin(_phase_module.phase_name(layer, part))
                phase_start_units[name] = track._phase
            if per_unit_window:
                _add_capture(track, captures[name])
            for fmt, wire, wire_bytes in cells[name]:
                _add_render(track, owners[name], render_sizes, name, fmt)
                track.add(wire, 0, wire_bytes, "wires")
    track.end()
    if not (command == "prepare" and per_unit_window):
        tail = [layer for layer in all_source_layers if layer not in by_layer]
        if tail:
            track.begin("source-tail")
            for layer in tail:
                if source_schedule is not None and source_cache is None:
                    for path, size in source_schedule["layers"][layer]:
                        track.add(path, 0, size, "source_authentication")
                for path, offset, length in _full_source_layer_extents(
                        campaign, layer, prefix):
                    track.add(path, offset, length, "source_extents")
            track.end()
    if source_schedule is not None and source_cache is None and source_schedule["completion"]:
        track.begin("source-complete")
        for path, size in source_schedule["completion"]:
            track.add(path, 0, size, "source_authentication")
        track.end()
    if not track.counts.get("renders"):
        # The row caches hold one render per measured rung and the pass opens
        # every one of them. A read set with none of them in it is the same
        # silent zero #524 refused for a row's seed wire: broken, not empty.
        raise SystemExit(
            "no render file of the merged roster was found under the campaign "
            f"row caches (first looked for {absent[0] if absent else '?'}); "
            "refusing to declare a read set that omits every render")

    if replay is not None:
        # One table for the resumed order: a replayed unit starts in its replay
        # phase, the rest start in their layer phase. The preparing action
        # looks every unit up in it, exactly as it does for a fresh run.
        phase_start_units.update(replay[_replay_module.PHASE_START_UNITS_KEY])

    annotations = {
        "entry_point": f"{JOINT_ENTRY_POINT}:{command}",
        "plan": plan_path,
        "plan_sha256": sha256_file(plan_path),
        "layers": layers,
        "units": len(roster),
        "measured_cells": sum(len(value) for value in cells.values()),
        "wire_dir": wire_dir,
        "capture_window": "per_unit" if per_unit_window else "per_layer",
        **({"source_prefetch_lookahead_layers": source_lookahead}
           if command == "prepare" and per_unit_window else {}),
        "renders_absent": len(absent),
        "renders_absent_first": absent[0] if absent else None,
        "synthesized_render_wire_bytes": synthesized_wire_bytes,
        "source_authentication_mode": ("not_applicable" if command == "run" else
            "verified_streamed_identity_cache" if source_cache is not None else
            "fresh_descriptor_sha256"),
        **({"source_identity_cache_host": socket.gethostname()}
           if source_cache is not None else {}),
        "sha256_present": False,
        "sha256_absent_reason": SHA256_ABSENT_REASON,
        "counts": track.counts,
        "bytes": track.bytes,
        "reread_bytes_by_phase": track.reread,
        "phases": track.phases,
        **({"phase_start_units": phase_start_units,
            "phase_target_bytes": _phase_module.MAX_PHASE_BYTES}
           if command == "prepare" and per_unit_window else {}),
        **({} if replay is None else replay),
        "argv": None if argv is None else [str(item) for item in argv],
    }
    return _finish(track, produced_by, annotations,
                   where=f"{JOINT_ENTRY_POINT}:{command}")


COST_COMPLETED_BINDING_SCHEMA = "prismaquant.joint_cost.validated_completed_units.v1"


def _cost_completed_units(binding, *, plan_sha256, prepared_sha256, roster):
    """Accept only an explicit caller-validated, identity-bound resume set."""
    if binding is None:
        return frozenset(), None
    if (not isinstance(binding, dict) or set(binding) != {
            "schema", "plan_sha256", "prepared_sha256", "units"}
            or binding.get("schema") != COST_COMPLETED_BINDING_SCHEMA
            or binding.get("plan_sha256") != plan_sha256
            or binding.get("prepared_sha256") != prepared_sha256):
        raise SystemExit("COST completed units need an exact validated plan/prepared binding")
    units = binding["units"]
    if (not isinstance(units, list) or any(not isinstance(name, str) for name in units)
            or units != sorted(set(units)) or not set(units) <= set(roster)):
        raise SystemExit("COST validated completed-unit roster is not a sorted campaign subset")
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    return frozenset(units), hashlib.sha256(encoded).hexdigest()


def _cost_layer_windows(names, *, formats_by_qname, census, owners,
                        render_sizes, budget, source_bytes):
    """Use real joint group geometry and serialized PWC file-size upper bounds.

    This import is COST-only. PREPARE V1 and module import remain torch-free;
    the COST producer runs in the pinned CPU environment that has the format
    registry. No render archive is opened or deserialized here.
    """
    import torch
    from prismaquant import format_registry as fr
    from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
    from prismaquant.joint_retained_window_plan import (
        plan_retained_targets, targets_from_statistics_plan)

    shapes, maxima = census.get("unit_shapes"), census.get("max_abs")
    if not isinstance(shapes, dict) or not isinstance(maxima, dict):
        raise SystemExit("COST census lacks unit_shapes or max_abs for joint groups")
    modules, specs, keys_by_name, costs, render_paths = {}, {}, {}, {}, {}
    for name in names:
        shape = shapes.get(name)
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                or any(type(axis) is not int or axis <= 0 for axis in shape)
                or name not in maxima):
            raise SystemExit(f"COST census lacks a complete joint target geometry: {name}")
        modules[name] = torch.nn.Linear(
            shape[1], shape[0], bias=False, device="meta", dtype=torch.bfloat16)
        fmts = formats_by_qname[name]
        specs[name] = {fmt: fr.get_format(fmt) for fmt in fmts}
        keys_by_name[name] = tuple((name, fmt) for fmt in fmts)
        for fmt in fmts:
            path, size = _render_file(owners[name], render_sizes, name, fmt)
            if size <= 0:
                raise SystemExit(f"prepared COST render is missing: {path}")
            costs[name, fmt] = {"incoming_storage_bytes": size,
                                "serialized_bytes": size}
            render_paths[name, fmt] = (path, size)
    statistics = plan_joint_statistics_target_windows(
        modules, specs, max_statistics_bytes=budget.statistics_cap_bytes,
        activation_max_abs={name: maxima[name] for name in names})
    targets = targets_from_statistics_plan(statistics, keys_by_name, costs)
    planned = plan_retained_targets(
        targets, budget=budget, source_bytes=source_bytes,
        footprint_scope="pwc_serialized_upper_bound")
    return planned, render_paths


def build_joint_cost_v2_manifest(plan_path, *, prepared, produced_by,
                                 retained_budget, source_bytes, n_probes=4,
                                 validated_completed_units=None, argv=None):
    """Declare COST's forward/reverse reads with sealed retained-window IDs.

    Full target partitions are derived before applying a validated resume
    subset. COST creates boundaries online from source and calibration IDs;
    it never reads PREPARE's per-unit activation capture payloads. Only
    completed members' renders are omitted. Original window phases and IDs
    remain in the read plan. Source rereads are explicit
    V2 references to one unique entry, never duplicate entry records.
    """
    from prismaquant.joint_retained_window_plan import RetainedWindowBudget

    if isinstance(retained_budget, dict):
        retained_budget = RetainedWindowBudget.from_dict(retained_budget)
    if not isinstance(retained_budget, RetainedWindowBudget):
        raise SystemExit("COST V2 requires a complete retained-window budget")
    if type(source_bytes) is not int or source_bytes <= 0:
        raise SystemExit("COST V2 requires the fixed declared source-owner byte cap")
    if type(n_probes) is not int or n_probes != 4:
        raise SystemExit("COST V2 currently declares exactly four joint probes")
    plan_path, prepared = os.path.abspath(plan_path), os.path.abspath(prepared)
    plan = _read_json(plan_path, "joint plan")
    if plan.get("schema") != JOINT_PLAN_SCHEMA:
        raise SystemExit(f"{plan_path}: not a {JOINT_PLAN_SCHEMA} plan")
    prefetch = plan.get("source_prefetch") or {}
    if (prefetch.get("prefetch_lookahead") != 1
            or prefetch.get("max_cache_slots") != 2):
        raise SystemExit("COST V2 requires source lookahead one and two cache slots")
    plan_sha256, prepared_sha256 = sha256_file(plan_path), sha256_file(prepared)
    inputs = plan["inputs"]
    campaign_plan_path = _bound(inputs["campaign_plan"], "plan inputs.campaign_plan")
    roster = _campaign_roster(campaign_plan_path)
    completed, completed_binding_sha256 = _cost_completed_units(
        validated_completed_units, plan_sha256=plan_sha256,
        prepared_sha256=prepared_sha256, roster=roster)
    completion = _read_json(prepared, "prepared completion")
    if completion.get("schema") != JOINT_PREPARED_SCHEMA or completion.get("status") != "complete":
        raise SystemExit("COST V2 requires a complete prepared completion")
    raw_formats = completion.get("formats_by_qname")
    if not isinstance(raw_formats, dict) or set(raw_formats) != set(roster):
        raise SystemExit("prepared COST candidate roster differs from the campaign")
    formats_by_qname = {}
    for name, raw in raw_formats.items():
        if (not isinstance(raw, list) or raw.count("BF16") != 1
                or raw[-1] != "BF16" or len(set(raw)) != len(raw)):
            raise SystemExit(f"prepared COST formats are incomplete or repeated: {name}")
        fmts = tuple(fmt for fmt in raw if fmt != "BF16")
        if not fmts or any(not isinstance(fmt, str) or not fmt for fmt in fmts):
            raise SystemExit(f"prepared COST has no measured candidate: {name}")
        formats_by_qname[name] = fmts

    workspace = os.path.dirname(campaign_plan_path)
    campaign = Campaign(workspace)
    if os.path.abspath(campaign.model_dir) != os.path.abspath(plan["model"]):
        raise SystemExit("COST plan source model differs from campaign source")
    owners = {name: row["dir"] for row in campaign.plan["rows"]
              for name in row["members"]}
    if set(owners) != set(roster):
        raise SystemExit("COST row-owner roster differs from campaign")
    render_sizes = {owner: _dir_sizes(os.path.join(owner, "cache"))
                    for owner in sorted(set(owners.values()))}
    census = _read_json(_bound(inputs["census"], "plan inputs.census"), "joint census")
    layer_of = _layer_index_of(roster)
    by_layer = {}
    for name in roster:
        by_layer.setdefault(layer_of[name], []).append(name)
    prefix = LAYER_QNAME_RE.match(roster[0]).group("prefix")
    source_layers = _source_layer_indices(
        campaign.weight_map, prefix, _backbone_layer_count(campaign.model_dir))
    if source_layers != list(range(source_layers[-1] + 1)):
        raise SystemExit("COST source layers are not a contiguous backbone")
    last_layer = source_layers[-1]
    if any(layer not in source_layers for layer in by_layer):
        raise SystemExit("COST target lies outside the source backbone")

    # Freeze every full per-layer target partition from file-size upper bounds
    # before applying completed units. Runtime PWC separately opens each
    # selected archive and proves storage <= the declared file size.
    windows_by_layer, render_paths = {}, {}
    for layer in source_layers:
        names = sorted(by_layer.get(layer, ()))
        if not names:
            windows_by_layer[layer] = ()
            continue
        planned, paths = _cost_layer_windows(
            names, formats_by_qname=formats_by_qname, census=census,
            owners=owners, render_sizes=render_sizes,
            budget=retained_budget, source_bytes=source_bytes)
        windows_by_layer[layer] = planned.windows
        render_paths.update(paths)

    track = _ReadPlanV2()
    track.begin("cost_setup")
    _parts, _states, _production_cache = _joint_head(
        track, plan_path, plan, roster=roster, prepared=prepared)
    for filename, size in sorted(_dir_sizes(plan["model"]).items()):
        if filename.endswith((".json", ".model", ".txt", ".jinja", ".py")):
            track.add(os.path.join(plan["model"], filename), 0, size,
                      "source_metadata")
    source_cache = plan.get("source_identity_cache")
    if source_cache is not None:
        path = _bound(source_cache, "source identity cache")
        if sha256_file(to_pool(path)) != source_cache.get("sha256"):
            raise SystemExit("bound COST source identity cache checksum changed")
        track.add(path, 0, _required_size(path, "source identity cache"),
                  "source_identity_cache")

    base = prefix.removesuffix("layers.")
    track.begin("cost_head")
    for path, offset, length in _source_tensor_extents(campaign, (
            base + "embed_tokens.", base + "norm.", base + "rotary_emb.",
            "lm_head."), required=False):
        track.add(path, offset, length, "source_head")

    source_extents = {layer: _full_source_layer_extents(campaign, layer, prefix)
                      for layer in source_layers}
    for layer in source_layers:
        track.begin(f"cost_capture_{layer:03d}")
        warmed = ((0, 1) if layer == 0 and last_layer >= 1 else (0,) if layer == 0
                  else (layer + 1,) if layer < last_layer else ())
        for source_layer in warmed:
            for path, offset, length in source_extents[source_layer]:
                track.add(path, offset, length, "source_forward")
    track.begin("cost_tail")  # retained last two source layers; no fresh read

    window_metadata = []
    for layer in reversed(source_layers):
        track.begin(f"cost_reverse_{layer:03d}_source")
        if 0 < layer < last_layer:
            for path, offset, length in source_extents[layer - 1]:
                track.add(path, offset, length, "source_reverse")
        for index, window in enumerate(windows_by_layer[layer]):
            phase_name = f"cost_reverse_{layer:03d}_window_{index:03d}"
            track.begin(phase_name)
            active = tuple(name for name in window.names if name not in completed)
            for name in active:
                for fmt in formats_by_qname[name]:
                    path, size = render_paths[name, fmt]
                    track.add(path, 0, size, "renders")
            window_metadata.append({
                "phase": phase_name, "layer": layer, "window_index": index,
                "original_full_target_names": list(window.names),
                "active_pending_names": list(active),
                "statistics_bytes": window.statistics_bytes,
                "render_file_upper_bound_bytes": window.render_bytes,
                "candidate_count": window.candidate_count,
            })

    partition = [(item["layer"], item["window_index"],
                  item["original_full_target_names"]) for item in window_metadata]
    partition_sha256 = hashlib.sha256(json.dumps(
        partition, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    annotations = {
        "entry_point": f"{JOINT_ENTRY_POINT}:run",
        "mode": "retained_cost_v2", "plan": plan_path,
        "plan_sha256": plan_sha256, "prepared": prepared,
        "prepared_sha256": prepared_sha256,
        "source_owner_cap_bytes": source_bytes,
        "retained_budget": retained_budget.as_dict(),
        "probes_per_window": n_probes,
        "source_prefetch_lookahead_layers": 1,
        "tail_retained_source_layers": source_layers[-2:],
        "window_partition_sha256": partition_sha256,
        "windows": window_metadata,
        "validated_completed_units_sha256": completed_binding_sha256,
        "validated_completed_units": len(completed),
        "sha256_present": False,
        "sha256_absent_reason": SHA256_ABSENT_REASON,
        "argv": None if argv is None else [str(item) for item in argv],
    }
    return track.manifest(produced_by, annotations,
                          where=f"{JOINT_ENTRY_POINT}:run:retained_v2")


def _finish(track: _Phases, produced_by: dict, annotations: dict, *, where: str) -> dict:
    for entry in track.entries:
        if not entry["path"].startswith(SHARED_MOUNT + "/"):
            raise SystemExit(f"{where}: entry outside the shared mount: {entry['path']}")
    manifest = {
        "schema": SCHEMA,
        "produced_by": produced_by,
        "mount_prefix": SHARED_MOUNT,
        "annotations": annotations,
        "entry_count": len(track.entries),
        "total_bytes": track.total_bytes,
        "entries": track.entries,
    }
    return check_manifest(manifest, where=where)


def build_allocation_manifest(joint_cost, plan, *, produced_by, argv=None):
    """The allocation handoff's read set.

    ``prismaquant.tessera_joint_allocation.handoff`` reads, in this order: the
    joint cost payload and the joint plan (both whole files, both hashed), the
    prepared completion the payload's own record names, the production cache
    that completion names, and then the head set of
    ``load_measured_anchor_input`` with ``verify_payloads=False`` -- the same
    head the joint pass reads, without any cell payload. It takes no capture,
    render, wire or source byte, so it has two phases and no layer phases.
    """
    joint_cost = os.path.abspath(joint_cost)
    plan_path = os.path.abspath(plan)
    payload = _read_pickle(joint_cost, "joint cost")
    evidence = (payload.get("provenance") or {}).get("tessera_joint_anchors")
    if not isinstance(evidence, dict) or "prepared" not in evidence:
        raise SystemExit(
            f"{joint_cost}: no tessera_joint_anchors record, so the prepared "
            "completion this handoff reads cannot be named")
    prepared = _bound(evidence["prepared"], "joint cost prepared binding")
    plan_payload = _read_json(plan_path, "joint plan")
    if plan_payload.get("schema") != JOINT_PLAN_SCHEMA:
        raise SystemExit(
            f"{plan_path}: schema is {plan_payload.get('schema')!r}, expected "
            f"{JOINT_PLAN_SCHEMA}")

    track = _Phases()
    track.begin("handoff")
    track.add(joint_cost, 0, _required_size(joint_cost, "joint cost"), "handoff")
    track.add(plan_path, 0, _required_size(plan_path, "joint plan"), "handoff")
    track.add(prepared, 0, _required_size(prepared, "prepared completion"), "handoff")
    completion = _read_json(prepared, "prepared completion")
    cache = _bound(completion["production_cache"], "prepared production_cache")
    track.add(cache, 0, _required_size(cache, "prepared production_cache"), "handoff")

    track.begin("head")
    _joint_head(track, plan_path, plan_payload,
                roster=_campaign_roster(_bound(
                    plan_payload["inputs"]["campaign_plan"],
                    "plan inputs.campaign_plan")),
                prepared=None)
    track.end()

    annotations = {
        "entry_point": ALLOCATION_ENTRY_POINT,
        "plan": plan_path,
        "plan_sha256": sha256_file(plan_path),
        "joint_cost": joint_cost,
        "prepared": prepared,
        "layers": [],
        "sha256_present": False,
        "sha256_absent_reason": SHA256_ABSENT_REASON,
        "counts": track.counts,
        "bytes": track.bytes,
        "reread_bytes_by_phase": track.reread,
        "phases": track.phases,
        "argv": None if argv is None else [str(item) for item in argv],
    }
    return _finish(track, produced_by, annotations, where=ALLOCATION_ENTRY_POINT)


def _read_pickle(path: str, label: str):
    try:
        with open(to_pool(path), "rb") as handle:
            return pickle.load(handle)
    except OSError as exc:
        raise SystemExit(f"{label}: unreadable {path}: {exc}") from exc


def read_assignment(path: str) -> dict:
    """``{qname: format}`` from a ``layer_config.json``.

    ``prismaquant.layer_config`` is the production parser and it is not
    importable here (it imports torch through ``schemas``), so this reads only
    the shape the export lane's own assignment file uses: a flat mapping of
    module qname to a format name, plus the reserved ``__prismaquant__``
    metadata key. A value that is not a plain format name is refused rather
    than interpreted, because interpreting it is the parser's job.
    """
    payload = _read_json(path, "assignment")
    out = {}
    for name, value in payload.items():
        if name == LAYER_CONFIG_META_KEY:
            continue
        if not isinstance(value, str):
            raise SystemExit(
                f"{path}: {name} carries a {type(value).__name__} recipe; this "
                "reader accepts only a format name, and the production parser "
                "in prismaquant.layer_config owns the richer shapes")
        out[str(name)] = value
    if not out:
        raise SystemExit(f"{path}: the assignment names no unit")
    return out


def build_export_manifest(plan, *, assignment, allocation_cost, produced_by,
                          argv=None):
    """The serving export's read set, for the units the assignment selects.

    Scope, stated because it is narrower than the export command: the export
    runs Tessera's own ``experiments/export_tessera_serving.py``, which is not
    in this repository, so its internal read *order* is not attested here.
    What is declared is the set of shared-mount bytes the export must read to
    write the artifact, in the order the artifact is written -- by ascending
    layer: for a unit the assignment gives a Tessera rung, that rung's wire
    blob; for a unit it leaves on the source precision, that unit's source byte
    extents. The allocation payload, the assignment and the joint plan's head
    records come first. Tensors outside the campaign roster -- embeddings,
    norms, the LM head -- are read by the exporter and are *not* declared here,
    because nothing in this repository names them; ``annotations`` says so.
    """
    plan_path = os.path.abspath(plan)
    assignment_path = os.path.abspath(assignment)
    allocation_cost = os.path.abspath(allocation_cost)
    plan_payload = _read_json(plan_path, "joint plan")
    if plan_payload.get("schema") != JOINT_PLAN_SCHEMA:
        raise SystemExit(
            f"{plan_path}: schema is {plan_payload.get('schema')!r}, expected "
            f"{JOINT_PLAN_SCHEMA}")
    selected = read_assignment(assignment_path)

    inputs = plan_payload["inputs"]
    checkpoint = _bound(inputs["merged_checkpoint"], "plan inputs.merged_checkpoint")
    _required_size(checkpoint, "plan inputs.merged_checkpoint")
    campaign_plan_path = _bound(inputs["campaign_plan"], "plan inputs.campaign_plan")
    roster = _campaign_roster(campaign_plan_path)
    unused = _Phases()
    unused.begin("roster")
    states = _checkpoint_unit_shards(unused, checkpoint, roster)
    merged_cost = _bound(inputs["merged_cost"], "plan inputs.merged_cost")
    payload = _read_pickle(merged_cost, "merged cost")
    wire_dir = str(payload["provenance"]["wire_dir"])
    del payload
    cells = _joint_cells(states, wire_dir)

    campaign = Campaign(os.path.dirname(campaign_plan_path))

    track = _Phases()
    track.begin("head")
    track.add(allocation_cost, 0, _required_size(allocation_cost, "allocation cost"), "head")
    track.add(assignment_path, 0, _required_size(assignment_path, "assignment"), "head")
    track.add(plan_path, 0, _required_size(plan_path, "joint plan"), "head")

    named = sorted(set(selected) & set(states))
    if not named:
        raise SystemExit(
            f"{assignment_path}: no assigned unit is in the campaign roster, "
            "so the export's selected read set would be empty")
    layer_of = _layer_index_of(named)
    by_layer = {}
    for name in named:
        by_layer.setdefault(layer_of[name], []).append(name)
    passthrough = []
    for layer in sorted(by_layer):
        track.begin(f"layer-{layer}")
        wired = []
        for name in sorted(by_layer[layer]):
            fmt = selected[name]
            match = [cell for cell in cells[name] if cell[0] == fmt]
            if match:
                wired.append(name)
                track.add(match[0][1], 0, match[0][2], "wires")
            else:
                passthrough.append(name)
        for path, offset, length in campaign.weight_extents_for(
                sorted(set(by_layer[layer]) - set(wired))):
            track.add(path, offset, length, "source_extents")
    track.end()

    annotations = {
        "entry_point": EXPORT_ENTRY_POINT,
        "plan": plan_path,
        "plan_sha256": sha256_file(plan_path),
        "assignment": assignment_path,
        "allocation_cost": allocation_cost,
        "layers": sorted(by_layer),
        "units": len(named),
        "passthrough_units": len(passthrough),
        "sha256_present": False,
        "sha256_absent_reason": SHA256_ABSENT_REASON,
        "read_order_attested": False,
        "read_order_reason": (
            "the exporter is Tessera's experiments/export_tessera_serving.py, "
            "outside this repository; the declared order is the artifact's "
            "layer order, and tensors outside the campaign roster -- "
            "embeddings, norms, the LM head -- are read by the exporter and "
            "are not declared here"),
        "counts": track.counts,
        "bytes": track.bytes,
        "reread_bytes_by_phase": track.reread,
        "phases": track.phases,
        "argv": None if argv is None else [str(item) for item in argv],
    }
    return _finish(track, produced_by, annotations, where=EXPORT_ENTRY_POINT)


def cached_activation_paths(act_dir: str, names) -> list:
    """The cached-activation files the AQUA stage opens for ``names``.

    The stage's own --act-dir layout is one file per unit,
    ``prismaquant.aqua_activation_cost.cached_act_path``. A unit whose file is
    not there is priced from the model's Gaussian fit instead, so a missing
    file is not a hole and declares no byte; what is declared is what the stage
    would open.
    """
    out = []
    for name in sorted(names):
        path = os.path.join(act_dir, str(name).replace(".", "__") + ".pt")
        if os.path.isfile(to_pool(path)):
            out.append(path)
    return out


def build_aqua_manifest(plan, *, card, cost_in, act_dir, produced_by, argv=None):
    """The AQUA stage's read set, for the campaign's requested (unit, format) roster.

    ``prismaquant.aqua_activation_cost`` walks the units the cost payload
    carries, resolves each one's weight through the model's own
    ``model.safetensors.index.json`` and streams that tensor to compute the
    activation-side term, then merges the result into the payload. So it reads
    the sensitivity card, the payload it merges into, the joint plan, the
    model's index and semantic config, and the source byte extents of every
    requested unit's weight -- the same extents the joint pass declares for its
    model source, keyed by unit rather than by layer/part. When measured
    pricing is asked for, the ``--act-dir`` cache is read too.

    What is *not* attested here is the read order. The stage iterates the
    artifact's units and streams per unit; the declared order is the model's
    layer order, which is the shape the prewarm loop windows on, not the order
    the stage opens the bytes in. Tensors outside the campaign roster --
    embeddings, norms, the LM head -- are never resolved by this stage and are
    not declared. A unit the index cannot resolve is priced as a hole and
    contributes no byte, so the extents are a lower-bound-or-equal claim on the
    model source and never a superset of it: the fallback path prices from the
    card, not from another tensor.
    """
    plan_path = os.path.abspath(plan)
    card = os.path.abspath(card)
    cost_in = os.path.abspath(cost_in)
    act_dir = os.path.abspath(act_dir) if act_dir else None
    plan_payload = _read_json(plan_path, "joint plan")
    if plan_payload.get("schema") != JOINT_PLAN_SCHEMA:
        raise SystemExit(
            f"{plan_path}: schema is {plan_payload.get('schema')!r}, expected "
            f"{JOINT_PLAN_SCHEMA}")
    inputs = plan_payload["inputs"]
    campaign_plan_path = _bound(inputs["campaign_plan"], "plan inputs.campaign_plan")
    roster = _campaign_roster(campaign_plan_path)
    model = os.path.abspath(str(plan_payload["model"]))

    track = _Phases()
    track.begin("head")
    track.add(card, 0, _required_size(card, "sensitivity card"), "head")
    track.add(cost_in, 0, _required_size(cost_in, "cost payload"), "head")
    track.add(plan_path, 0, _required_size(plan_path, "joint plan"), "head")
    for name in ("config.json", "model.safetensors.index.json"):
        path = os.path.join(model, name)
        track.add(path, 0, _required_size(path, name), "head")

    campaign = Campaign(os.path.dirname(campaign_plan_path))
    layer_of = _layer_index_of(roster)
    by_layer = {}
    for name in roster:
        by_layer.setdefault(layer_of[name], []).append(name)
    for layer in sorted(by_layer):
        track.begin(f"layer-{layer}")
        for path, offset, length in campaign.weight_extents_for(
                sorted(by_layer[layer])):
            track.add(path, offset, length, "source_extents")
    track.end()

    cached = []
    if act_dir is not None:
        cached = cached_activation_paths(act_dir, roster)
        track.begin("activations")
        for path in cached:
            track.add(path, 0, _required_size(path, "cached activation"),
                      "activations")
        track.end()

    annotations = {
        "entry_point": AQUA_ENTRY_POINT,
        "plan": plan_path,
        "plan_sha256": sha256_file(plan_path),
        "card": card,
        "cost_in": cost_in,
        "act_dir": act_dir,
        "model": model,
        "layers": sorted(by_layer),
        "units": len(roster),
        "cached_activations": len(cached),
        "sha256_present": False,
        "sha256_absent_reason": SHA256_ABSENT_REASON,
        "read_order_attested": False,
        "read_order_reason": (
            "the stage iterates the cost artifact's units and streams each "
            "unit's weight in one pass; the declared order is the model's "
            "layer order, and the merge writes the payload a byte at a time "
            "rather than in a read order this manifest could seal"),
        "counts": track.counts,
        "bytes": track.bytes,
        "reread_bytes_by_phase": track.reread,
        "phases": track.phases,
        "argv": None if argv is None else [str(item) for item in argv],
    }
    return _finish(track, produced_by, annotations, where=AQUA_ENTRY_POINT)


def deterministic_entry_provenance(entry_point: str, *, plan: str,
                                   plan_sha256: str, workspace: str) -> dict:
    """``produced_by`` for a post-campaign pass, with nothing run-specific in it.

    ``pbrun`` ingests the manifest as a content-addressed input and seals its
    digest into the action key, so a hostname or a clock reading here would
    give the same pass a new key on every submit and turn a finished action
    into a re-run. Every field is a property of the tree, the plan and the
    campaign.
    """
    return {
        "tool": "prismaquant/experiments/glm_data_manifests.py",
        "commit": git_commit(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "entry_point": entry_point,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "workspace": workspace,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = CAMPAIGN_BASE + "/transition-optimization-20260910"
    ap.add_argument("--workspace",
                    default=CAMPAIGN_BASE + "/first-proof-anchor-preparation-05/workspace")
    ap.add_argument("--campaign-manifest", default=base + "/resume-overlap-manifest.json")
    ap.add_argument("--out-dir", default=base + "/data-manifests")
    ap.add_argument("--out-manifest", default=base + "/resume-overlap-manifest.with-data.json")
    ap.add_argument("--sizes-cache",
                    default="/home/rob/tmp/glm-perf-20260910/w3/evidence/capture-file-sizes.txt")
    ap.add_argument("--stat", action="store_true",
                    help="stat every capture file instead of using the size cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and summarise; write nothing")
    ap.add_argument("--json", help="write the run summary here")
    args = ap.parse_args()

    src = args.campaign_manifest
    rows = json.load(open(src))
    if not isinstance(rows, list):
        raise SystemExit(f"{src}: expected a list of pbcampaign rows")
    src_sha = sha256_file(src)

    sizes = None if args.stat else load_sizes(args.sizes_cache)
    campaign = CachedSizeCampaign(args.workspace, sizes)
    produced_by = deterministic_provenance(
        args.workspace, campaign, "stat" if args.stat else args.sizes_cache)

    out_rows = []
    report = []
    for row in rows:
        rid = row_id_of(row)
        if rid is None:
            raise SystemExit(f"{src}: a row's argv names no single units/row-XXXX.json")
        if rid not in campaign.rows:
            raise SystemExit(f"{src}: {rid} is not a row of {args.workspace}/plan.json")
        man = build_manifest(campaign, rid, produced_by, row.get("argv"))
        path = os.path.join(args.out_dir, f"{rid}.data-manifest.json")
        blob = json.dumps(man, indent=1, sort_keys=False).encode() + b"\n"
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        new_row = dict(row)
        new_row["data_manifest"] = path
        out_rows.append(new_row)
        report.append({
            "row_id": rid, "path": path,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "manifest_bytes": len(blob),
            "entry_count": man["entry_count"],
            "total_bytes": man["total_bytes"],
            "captures": man["annotations"]["counts"]["captures"],
            "weight_extents": man["annotations"]["counts"]["weight_extents"],
            "seeds": man["annotations"]["counts"]["seeds"],
            "capture_bytes": man["annotations"]["bytes"]["captures"],
            "weight_bytes": man["annotations"]["bytes"]["weight_extents"],
            "seed_bytes": man["annotations"]["bytes"]["seeds"],
        })

    out_blob = json.dumps(out_rows, indent=1).encode() + b"\n"
    if not args.dry_run:
        tmp = args.out_manifest + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(out_blob)
        os.replace(tmp, args.out_manifest)

    summary = {
        "source_manifest": src,
        "source_manifest_sha256": src_sha,
        "out_manifest": args.out_manifest,
        "out_manifest_sha256": hashlib.sha256(out_blob).hexdigest(),
        "out_dir": args.out_dir,
        "rows": len(report),
        "total_bytes": sum(r["total_bytes"] for r in report),
        "dry_run": args.dry_run,
        "produced_by": produced_by,
        "manifests": report,
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    sys.stderr.write(
        f"\n{len(report)} rows, {summary['total_bytes'] / 1e9:.2f} GB planned\n"
        f"source   {src_sha}  {src}\n"
        f"with-data {summary['out_manifest_sha256']}  {args.out_manifest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
