"""Complete block-trial research CLI: encode and score, with --preflight.

eng-pq-fine-grained (#2329) driver for Rob's opt-in per-block trial. One entry
over the SAME actual arguments; encode and score share one persistent
``--output`` directory (score consumes the bank encode wrote there).

* ``--mode encode`` verifies the FIT capture receipt and encodes
  the five-rung parent bank (R768/R896/R960/R1024/R1152) from the FIT role
  only through the existing ``encode_parent_bank`` producer recipe. Canonical
  rung admission runs through the producer's public PURE metadata API
  (``--rung-reader-source``, loaded by importlib) BEFORE any encoding.
* ``--mode score`` prices every block with the existing conditional FIT
  prices, shrinks them toward group means, allocates one parent per block
  under the uniform A8S control's exact body budget, packs the selection with
  the existing reference wire at EXACTLY the control's serialized byte count,
  decodes, assembles and evaluates the complete FIT+HELDOUT quadratic, and
  reports concentration diagnostics per real existing body bit, including the
  actual conditional price per saved body bit of the R896 (and R768) single
  block replacements at both the primary 128x2 and the current 128x32
  geometry.
* ``--preflight`` turns either mode into a real-input census on ``--device
  cpu``: imports, true argument parsing, actual source/FIT slices, shapes,
  counts and admissions recorded. Score preflight also reads HELDOUT; encode
  preflight reads only its receipt metadata. Neither encodes or qualifies a
  GPU; missing prerequisites are named and nothing is manufactured.

Research only. FIT alone shapes encoding and prices; HELDOUT enters only the
final assembled error and never selects geometry or rungs. No serving
runtime, no G3, no capture, no speed or served claim. The control is a fresh
same-FIT uniform A8S R1024 whole unit and the candidate packs to the
control's exact byte count, so no improvement is credited to granularity or
size. The packed format carries ONE uniform block geometry per blob; the
optional extra geometries are whole-blob uniform arms, never a per-block
mixed-size choice.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import socket
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.digests import canonical_json_sha256  # noqa: E402
from prismaquant.qnames import DOTTED_LAYER_QNAME  # noqa: E402
from prismaquant.io_engine import load_file  # noqa: E402
from prismaquant.residency_map import bind_residency_manifest, residency_report  # noqa: E402
from prismaquant.staged_tier_policy import activate_staged_tier_policy  # noqa: E402
from tools.pq_block_concentration import (  # noqa: E402
    block_scores,
    concentration,
    file_sha256,
)
from tools.pq_block_price_solver import (  # noqa: E402
    allocate_body_budget,
    shrink_group_prices,
)
from tools.pq_block_reference_wire import (  # noqa: E402
    body_costs,
    decode_projection,
    fixed_bytes,
    pack_projection,
)
from tools.pq_block_trial_math import (  # noqa: E402
    assemble_weight,
    block_groups,
    conditional_fit_prices,
    encode_parent_bank,
    evaluate_packed_candidate,
    validate_moment,
    validate_sample_split,
)

FORMAT_NAME = "TESSERA_E4M3_K1"
BANK_RUNGS = (768, 896, 960, 1024, 1152)
CONTROL_RUNG = 1024
#: Actual price-per-saved-bit diagnostic arms (parent request): the primary
#: next-rung downgrade first, the deepest one second.
SAVED_BIT_DIAGNOSTIC_RUNGS = (896, 768)
PRIMARY_GEOMETRY = (128, 2)
SAVED_BIT_GEOMETRIES = ((128, 2), (128, 32))
SENSITIVITY_GEOMETRIES = ((128, 2), (128, 32))
SHRINKAGE = 0.25
INPUT_GROUP_COLS = 32

SPLIT_SCHEMA = "prismaquant.research_split_manifest.v1"
LAYER_SCHEMA = "prismaquant.research_split_layer_manifest.v1"
BANK_SCHEMA = "prismaquant.block_trial_parent_bank.v1"
ROLE_KEYS = ("fit", "heldout")
PRODUCER_READER_SHA256 = (
    "0f9a04ce84571c62cdf22e331258236437af27d414c25411253f314f1584396c"
)


class TrialRefused(ValueError):
    """The trial refuses by name; invalid or missing actual input."""


def _fail(message: str) -> None:
    raise TrialRefused(message)


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _geometry(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)x([0-9]+)", text)
    if match is None:
        _fail(f"--extra-geometry {text!r} is not ROWxCOLS like 256x2")
    rows, cols = int(match.group(1)), int(match.group(2))
    if rows < 8 or rows % 8:
        _fail(f"geometry {text}: block rows must be a positive multiple of 8")
    if cols < 1 or INPUT_GROUP_COLS % cols:
        _fail(f"geometry {text}: block cols must divide {INPUT_GROUP_COLS}")
    return rows, cols


# ---------------------------------------------------------------------------
# Producer rung admission (public metadata API, imported, never mirrored)
# ---------------------------------------------------------------------------

def load_rung_reader(reader_path: Path) -> tuple[object, dict]:
    """Importlib-load the producer-owned PURE metadata API and stamp it.

    The SHA stamp is a reference only: there is no identity gate, the
    module's own validation and admission are the contract.
    """
    if not reader_path.is_file():
        _fail(f"--rung-reader-source {reader_path} is not a readable file")
    spec = importlib.util.spec_from_file_location(
        "producer_rung_allowability", reader_path)
    if spec is None or spec.loader is None:
        _fail(f"{reader_path}: cannot load the producer rung reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attribute in ("validate_index", "validate_table", "admit_rung"):
        if not callable(getattr(module, attribute, None)):
            _fail(f"{reader_path}: producer reader lacks {attribute}()")
    stamp = {
        "path": str(reader_path),
        "sha256": file_sha256(reader_path),
        "producer_reference_sha256": PRODUCER_READER_SHA256,
        "identity_gate": False,
    }
    return module, stamp


def admit_bank_rungs(index_root: Path, kernel_build_id: str, reader) -> dict:
    """validate_index, resolve the CURRENT table version, validate, admit.

    Resolution follows the published index exactly:
    ``formats[FORMAT_NAME].kernel_builds[build].versions[str(current)]``.
    """
    index_path = index_root / "index.json"
    if not index_path.is_file():
        _fail(f"rung index {index_path} is missing")
    index = json.loads(index_path.read_bytes())
    reader.validate_index(index)
    formats = index["formats"]
    if FORMAT_NAME not in formats:
        _fail(f"rung index has no {FORMAT_NAME} format")
    builds = formats[FORMAT_NAME]["kernel_builds"]
    if kernel_build_id not in builds:
        _fail(f"rung index has no kernel build {kernel_build_id!r} "
              f"under {FORMAT_NAME}")
    entry = builds[kernel_build_id]
    current = entry["current_version"]
    record = entry["versions"][str(current)]
    table_path = index_root / record["path"]
    table = json.loads(table_path.read_bytes())
    reader.validate_table(table)
    decisions = {}
    for rung in BANK_RUNGS:
        decisions[str(rung)] = reader.admit_rung(
            table, format=FORMAT_NAME, kernel_build_id=kernel_build_id,
            rung=rung)
    return {
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "table_path": str(table_path),
        "table_sha256": file_sha256(table_path),
        "table_version": table["table_version"],
        "table_status": table["table_status"],
        "current_version": current,
        "format": FORMAT_NAME,
        "kernel_build_id": kernel_build_id,
        "bank_rungs": list(BANK_RUNGS),
        "decisions": decisions,
        "table": table,
    }


def require_all_admitted(admission: dict) -> None:
    refused = {
        rung: decision for rung, decision in admission["decisions"].items()
        if decision.get("status") != "allow"
    }
    if refused:
        _fail(f"canonical admission refuses bank rungs {refused}")


def admit_callable(reader, admission: dict):
    """The encode_parent_bank admission callback over the loaded API."""
    table = admission["table"]
    build = admission["kernel_build_id"]

    def admit(rung: int) -> dict:
        return reader.admit_rung(table, format=FORMAT_NAME,
                                 kernel_build_id=build, rung=rung)

    return admit


# ---------------------------------------------------------------------------
# Split capture intake: receipts verified against the actual bytes
# ---------------------------------------------------------------------------

def verify_split_manifest(capture_root: Path) -> dict:
    """Schema, fixed 384/128 split geometry and split-stamp rederivation."""
    path = capture_root / "split-manifest.json"
    if not path.is_file():
        _fail(f"split manifest {path} is missing")
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema") != SPLIT_SCHEMA:
        _fail(f"{path}: not a {SPLIT_SCHEMA} manifest")
    if manifest.get("sample_count") != 512 \
            or manifest.get("tokens_per_sample") != 512 \
            or manifest.get("same_forward_pass") is not True \
            or manifest.get("fit_stop") != 384:
        _fail(f"{path}: not the fixed 512x512 same-pass 384 split")
    roles = manifest["roles"]
    if roles["fit"]["sample_range"] != [0, 384] \
            or roles["heldout"]["sample_range"] != [384, 512]:
        _fail(f"{path}: role sample ranges are not the fixed 384/128 split")
    fit_provenance = roles["fit"]["provenance"]
    if fit_provenance.get("hessian_role") != "fit":
        _fail(f"{path}: the FIT role provenance is not hessian_role=fit")
    if fit_provenance.get("fit_tokens") != 384 * 512:
        _fail(f"{path}: FIT provenance does not stamp 196608 fit tokens")
    for field in ("text_sha256", "fit_ids_sha256"):
        if not isinstance(fit_provenance.get(field), str) \
                or not fit_provenance[field]:
            _fail(f"{path}: FIT provenance lacks {field}")
    if roles["heldout"]["provenance"].get("hessian_role") != "held-out":
        _fail(f"{path}: the HELDOUT role provenance is not held-out")
    stored = manifest.get("split_sha256")
    rederived = canonical_json_sha256(
        {key: value for key, value in manifest.items()
         if key != "split_sha256"}, where="block trial split stamp")
    if stored != rederived:
        _fail(f"{path}: split_sha256 {stored!r} does not rederive "
              f"({rederived})")
    return {
        "manifest": manifest,
        "path": str(path),
        "sha256": file_sha256(path),
        "split_sha256": stored,
        "fit_range": [0, 384],
        "heldout_range": [384, 512],
        "fit_identity": dict(fit_provenance),
    }


def layer_number(qname: str) -> int:
    match = DOTTED_LAYER_QNAME.search(qname)
    if match is None:
        _fail(f"{qname}: a trial unit must name a decoder layer")
    return int(match.group(1))


def _receipt_path(capture_root: Path, receipt: dict) -> Path:
    """Resolve a receipt's ROOT-RELATIVE file inside the capture root."""
    name = receipt.get("file")
    if not isinstance(name, str) or not name:
        _fail(f"receipt {receipt} carries no file name")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"receipt file {name!r} is not a safe capture-root-relative "
              "path")
    path = capture_root / relative
    if capture_root.resolve() not in path.resolve().parents:
        _fail(f"receipt file {name!r} escapes the capture root")
    return path


def _check_receipt(receipt: dict, role_key: str) -> None:
    for field in ("bytes", "sha256", "count", "hessian_shape"):
        if field not in receipt:
            _fail(f"{role_key} receipt lacks {field}: {receipt}")
    if type(receipt["bytes"]) is not int or receipt["bytes"] <= 0 \
            or type(receipt["count"]) is not int or receipt["count"] <= 0:
        _fail(f"{role_key} receipt bytes/count must be positive integers")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt["sha256"])):
        _fail(f"{role_key} receipt sha256 is not a hex digest")
    shape = receipt["hessian_shape"]
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != shape[1]:
        _fail(f"{role_key} receipt hessian_shape {shape} is not [K,K]")
    inputs_shape = receipt.get("inputs_shape")
    ids_shape = receipt.get("prefix_sample_ids_shape")
    # The publisher's finish always emits an int64 prefix id tensor; with
    # max_prefix_rows=0 (or nothing retained) that tensor is empty while the
    # inputs are None. An empty-prefix receipt is valid.
    if inputs_shape is None:
        if ids_shape is not None and ids_shape != [0]:
            _fail(f"{role_key} receipt has no inputs but prefix ids shape "
                  f"{ids_shape}")
    else:
        if not isinstance(inputs_shape, list) or len(inputs_shape) != 2 \
                or inputs_shape[1] != shape[1]:
            _fail(f"{role_key} receipt inputs_shape {inputs_shape} "
                  "disagrees with the [K,K] moment basis")
        if not isinstance(ids_shape, list) or len(ids_shape) != 1:
            _fail(f"{role_key} receipt prefix ids shape {ids_shape} "
                  "is not 1-D")


def load_role_payload(capture_root: Path, qname: str, layer_manifest: dict,
                      role_key: str, sample_range: list[int],
                      load_tensors: bool) -> dict:
    """Read verified tensors, or metadata only when the role is excluded.

    Encode mode never opens the HELDOUT payload, including during preflight.
    Its count/digest remain publisher receipt metadata until CPU scoring
    verifies the actual bytes and loads the complete HELDOUT moment.
    """
    units = layer_manifest["manifest"]["units"]
    if qname not in units or role_key not in units[qname]:
        _fail(f"layer manifest has no {qname} {role_key} receipt")
    receipt = units[qname][role_key]
    _check_receipt(receipt, role_key)
    dimension = int(receipt["hessian_shape"][1])
    path = _receipt_path(capture_root, receipt)
    record = {
        "role": role_key, "file": receipt["file"], "path": str(path),
        "bytes": int(receipt["bytes"]), "sha256": receipt["sha256"],
        "receipt_count": int(receipt["count"]),
        "hessian_shape": [int(v) for v in receipt["hessian_shape"]],
        "inputs_shape": receipt.get("inputs_shape"),
        "prefix_sample_ids_shape": receipt.get("prefix_sample_ids_shape"),
        "sample_range": list(sample_range),
        "tensors_loaded": False, "file_bytes_read": False,
    }
    if not load_tensors:
        return record
    if not path.is_file():
        _fail(f"{role_key} role file {path} is missing")
    actual_bytes = path.stat().st_size
    if actual_bytes != receipt["bytes"]:
        _fail(f"{path}: {actual_bytes} bytes, receipt stamps "
              f"{receipt['bytes']}")

    def decode_role(raw, observed, staged):
        return torch.load(raw.path, map_location="cpu", weights_only=True, mmap=True), None

    payload, load_observed = load_file(path, receipt["bytes"],
        binding=receipt["sha256"], decode=decode_role, sealed=True)
    actual_sha = load_observed[0]["sha256"]
    if actual_sha != receipt["sha256"]:
        _fail(f"{path}: own-byte digest does not match receipt")
    record.update({"bytes": actual_bytes, "sha256": actual_sha, "file_bytes_read": True})
    if payload["name"] != qname or payload["role"] != role_key:
        _fail(f"{path}: payload names {payload['name']}/{payload['role']}, "
              f"not {qname}/{role_key}")
    hessian = payload["hessian"]
    if not isinstance(hessian, torch.Tensor) or hessian.device.type != "cpu" \
            or hessian.dtype != torch.float32:
        _fail(f"{path}: the role hessian is not a CPU float32 tensor")
    if list(hessian.shape) != [dimension, dimension]:
        _fail(f"{path}: hessian {list(hessian.shape)} is not "
              f"[{dimension},{dimension}]")
    count = payload["count"]
    if type(count) is not int or count <= 0 or count != receipt["count"]:
        _fail(f"{path}: actual count {count!r} disagrees with receipt "
              f"{receipt['count']}")
    max_abs = payload["max_abs"]
    if not isinstance(max_abs, (int, float)) or not np.isfinite(max_abs) \
            or max_abs < 0:
        _fail(f"{path}: max_abs {max_abs!r} is not finite and nonnegative")
    inputs = payload["inputs"]
    ids = payload["prefix_sample_ids"]
    # The publisher's finish() always materializes an int64 id tensor; with
    # no retained prefix rows the inputs are None and the id tensor is
    # empty. An empty prefix is a valid role; counts stay the full actual
    # moment counts and are never inferred from prefix ids.
    empty_prefix = (isinstance(ids, torch.Tensor) and ids.dtype == torch.int64
                    and ids.ndim == 1 and ids.numel() == 0)
    if inputs is None:
        if ids is not None and not empty_prefix:
            _fail(f"{path}: prefix sample ids present without retained rows")
    else:
        if not isinstance(inputs, torch.Tensor) \
                or inputs.device.type != "cpu" \
                or inputs.dtype != torch.float32 or inputs.ndim != 2:
            _fail(f"{path}: role inputs are not a CPU float32 matrix")
        if list(inputs.shape) != [int(v) for v in receipt["inputs_shape"]]:
            _fail(f"{path}: inputs {list(inputs.shape)} disagree with "
                  f"receipt {receipt['inputs_shape']}")
        if inputs.shape[1] != dimension:
            _fail(f"{path}: inputs do not span the {dimension}-column basis")
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 \
                or ids.ndim != 1 or ids.numel() == 0:
            _fail(f"{path}: retained inputs need nonempty 1-D int64 prefix "
                  "sample ids")
        if list(ids.shape) != [int(v)
                               for v in receipt["prefix_sample_ids_shape"]]:
            _fail(f"{path}: prefix ids {list(ids.shape)} disagree with "
                  f"receipt {receipt['prefix_sample_ids_shape']}")
        # One sample id per RETAINED prefix row: never a full-moment roster.
        if ids.numel() != inputs.shape[0]:
            _fail(f"{path}: {ids.numel()} prefix ids for "
                  f"{inputs.shape[0]} retained rows")
        lo, hi = int(sample_range[0]), int(sample_range[1])
        if int(ids.min()) < lo or int(ids.max()) >= hi:
            _fail(f"{path}: prefix sample ids leave the {role_key} range "
                  f"[{lo},{hi})")
    validate_moment(hessian, count, dimension, f"{role_key} role")
    record.update({
        "tensors_loaded": True,
        "count": count,
        "count_source": "loaded PT count field, equal to its receipt",
        "max_abs": float(max_abs),
        "hessian": hessian,
        "inputs": inputs,
        "prefix_sample_ids": ids,
        "retained_prefix_rows": 0 if inputs is None else int(inputs.shape[0]),
    })
    return record


def load_layer_roles(capture_root: Path, qname: str, split: dict,
                     load_tensors: dict[str, bool]) -> dict:
    """Read the layer manifest and both role receipts for one unit."""
    layer = layer_number(qname)
    path = capture_root / "layers" / f"L{layer:03d}" / "manifest.json"
    if not path.is_file():
        _fail(f"layer manifest {path} is missing: the shared quantum for "
              f"layer {layer} is not published yet")
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema") != LAYER_SCHEMA:
        _fail(f"{path}: not a {LAYER_SCHEMA} manifest")
    if manifest.get("layer") != layer:
        _fail(f"{path}: manifest names layer {manifest.get('layer')}, "
              f"expected {layer}")
    if manifest.get("split_sha256") != split["split_sha256"]:
        _fail(f"{path}: split stamp {manifest.get('split_sha256')!r} is not "
              f"the verified split {split['split_sha256']!r}")
    if qname not in manifest.get("units", {}):
        _fail(f"{path}: no {qname} unit")
    if qname not in manifest.get("full_counts", {}):
        _fail(f"{path}: no {qname} full census count")
    full_count = int(manifest["full_counts"][qname])
    roles = {}
    for role_key in ROLE_KEYS:
        roles[role_key] = load_role_payload(
            capture_root, qname, {"manifest": manifest}, role_key,
            split["manifest"]["roles"][role_key]["sample_range"],
            load_tensors=bool(load_tensors.get(role_key)))

    def actual_count(role_key: str) -> int:
        record = roles[role_key]
        if record["tensors_loaded"]:
            return int(record["count"])
        return int(record["receipt_count"])

    fit_count, held_count = actual_count("fit"), actual_count("heldout")
    if fit_count + held_count != full_count:
        _fail(f"{qname}: fit {fit_count} + heldout {held_count} != census "
              f"{full_count}; the split dropped or duplicated rows")
    both_loaded = all(r["tensors_loaded"] for r in roles.values())
    census = {
        "fit_count": fit_count,
        "fit_count_source": "loaded PT" if roles["fit"]["tensors_loaded"]
        else "publisher-verified receipt",
        "heldout_count": held_count,
        "heldout_count_source": "loaded PT"
        if roles["heldout"]["tensors_loaded"]
        else "publisher-verified receipt",
        "full_count": full_count,
        "fit_retained_prefix_rows": roles["fit"]["retained_prefix_rows"]
        if roles["fit"]["tensors_loaded"] else None,
        "heldout_retained_prefix_rows":
            roles["heldout"]["retained_prefix_rows"]
            if roles["heldout"]["tensors_loaded"] else None,
        "prefix_ids_cover_prefix_rows_only": True,
        "counts_not_inferred_from_prefix_ids": True,
        "both_role_tensors_loaded_and_reverified": both_loaded,
    }
    return {
        "path": str(path), "sha256": file_sha256(path),
        "layer": layer, "split_sha256": manifest["split_sha256"],
        "full_count": full_count, "roles": roles, "census": census,
    }


INPUT_DIGESTS = {}  # explicit readset configuration, not a byte cache


def load_source_weight(source_root: Path, qname: str) -> dict:
    """Read the BF16 source weight named by the safetensors index."""
    index_path = source_root / "model.safetensors.index.json"
    if not index_path.is_file():
        _fail(f"source index {index_path} is missing")
    index = json.loads(index_path.read_bytes())
    key = qname + ".weight"
    shard = index.get("weight_map", {}).get(key)
    if shard is None:
        _fail(f"{index_path}: no {key} entry; the trial needs an existing "
              "source key")
    shard_path = source_root / shard
    def decode_source(raw, observed, staged):
        with safe_open(raw.path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key), None
    tensor, observed = load_file(shard_path, shard_path.stat().st_size,
        binding=INPUT_DIGESTS.get(str(shard_path)), decode=decode_source, sealed=True)
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        _fail(f"{key}: source weight is {tensor.dtype}/{tensor.ndim}D, "
              "the trial encodes a 2-D BF16 projection")
    return {
        "tensor": tensor, "key": key, "shard": shard,
        "source_root": str(source_root),
        "index_sha256": file_sha256(index_path),
        "shape": [int(v) for v in tensor.shape],
        "small_actual_values": tensor[:2, :8].float().tolist(),
        "source_file_bytes": observed[0]["bytes"],
        "source_file_sha256": observed[0]["sha256"],
        "dtype": str(tensor.dtype),
    }


#: The scientific FIT identity the score actually depends on; anything else
#: in the recorded provenance (model paths, prose sources) may drift and is
#: stamped, never a re-encode wall.
FIT_IDENTITY_FIELDS = ("hessian_role", "fit_ids_sha256", "text_sha256",
                       "fit_tokens", "nsamples", "seqlen", "seed")


def load_parent_bank(output: Path, intake: dict) -> dict:
    """Score-side bank verification against the caller's own output files."""
    bank_path = output / "parent-bank.json"
    if not bank_path.is_file():
        _fail(f"{bank_path} is missing; run --mode encode first")
    bank = json.loads(bank_path.read_bytes())
    if bank.get("schema") != BANK_SCHEMA:
        _fail(f"{bank_path}: not a {BANK_SCHEMA} bank")
    if bank.get("qname") != intake["qname"]:
        _fail(f"{bank_path}: bank encodes {bank.get('qname')!r}")
    if bank.get("shape") != intake["source"]["shape"]:
        _fail(f"{bank_path}: bank shape {bank.get('shape')} disagrees with "
              f"the source {intake['source']['shape']}")
    if bank.get("fit_count") != intake["census"]["fit_count"]:
        _fail(f"{bank_path}: bank fit_count {bank.get('fit_count')} is not "
              f"the actual {intake['census']['fit_count']}")
    recorded_identity = bank.get("fit_identity") or {}
    current_identity = intake["split"]["fit_identity"]
    mismatch = [field for field in FIT_IDENTITY_FIELDS
                if recorded_identity.get(field) != current_identity.get(field)]
    if mismatch:
        _fail(f"{bank_path}: bank FIT identity fields {mismatch} differ from "
              "the verified split; the bank was not encoded from this FIT")
    fit_identity_drift = {
        field: {"recorded": recorded_identity.get(field),
                "current": current_identity.get(field)}
        for field in sorted(set(recorded_identity) | set(current_identity))
        if recorded_identity.get(field) != current_identity.get(field)}
    if bank.get("heldout_consumed") is not False:
        _fail(f"{bank_path}: bank claims heldout consumption")
    if bank.get("uniform_control_rung") != CONTROL_RUNG:
        _fail(f"{bank_path}: bank control rung is not {CONTROL_RUNG}")
    bank_rungs = sorted(int(p["rung"]) for p in bank["parents"])
    if bank_rungs != sorted(BANK_RUNGS):
        _fail(f"{bank_path}: bank rungs {bank_rungs} are not "
              f"{sorted(BANK_RUNGS)}")
    parents = {}
    for entry in bank["parents"]:
        rung = int(entry["rung"])
        decision = entry.get("admission") or {}
        if decision.get("status") != "allow" or decision.get("rung") != rung:
            _fail(f"{bank_path}: parent R{rung} admission {decision}")
        path = output / f"parent-R{rung}.tessera"
        if not path.is_file():
            _fail(f"parent blob {path} is missing")
        blob = path.read_bytes()
        if len(blob) != entry["bytes"]:
            _fail(f"{path}: {len(blob)} bytes, bank stamps {entry['bytes']}")
        digest = _sha_bytes(blob)
        if digest != entry["sha256"]:
            _fail(f"{path}: digest {digest} does not match the bank")
        parents[rung] = {"entry": entry, "blob": blob, "path": str(path),
                         "bytes": len(blob), "sha256": digest}
    return {"bank": bank, "bank_path": str(bank_path),
            "bank_sha256": file_sha256(bank_path), "parents": parents,
            "fit_identity_drift": fit_identity_drift}


def parse_and_render(parents: dict, source_shape: list[int]) -> dict:
    """Parse every parent blob once; render its stock fp32 twin."""
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.stock import materialize_stock, stock_dequant
    parsed, rendered = {}, {}
    for rung, entry in parents.items():
        unit = parse_unit_artifact(entry["blob"], device="cpu")
        branch = getattr(unit.manifest, "branch", None)
        root_q = getattr(branch, "root_q256", None)
        if root_q != rung:
            _fail(f"parent-R{rung}.tessera stores root_q256 {root_q!r}")
        tensors = materialize_stock(unit.unit, unit.forests, unit.code)
        weights = stock_dequant(tensors)
        if [int(v) for v in weights.shape] != source_shape:
            _fail(f"parent R{rung} renders {list(weights.shape)}, "
                  f"expected {source_shape}")
        parsed[rung] = unit
        rendered[rung] = weights
        entry["stored_unit_id"] = str(getattr(branch, "unit_id", ""))
        del tensors
    return {"parsed": parsed, "rendered": rendered}


def candidate_order() -> list[int]:
    """Control first (the zero price column), remaining rungs ascending."""
    return [CONTROL_RUNG] + [r for r in sorted(BANK_RUNGS)
                             if r != CONTROL_RUNG]


# ---------------------------------------------------------------------------
# Preflight census (records instead of refusing; strict modes raise)
# ---------------------------------------------------------------------------

def preflight_census(args) -> tuple[dict, list[str]]:
    findings: dict = {}
    missing: list[str] = []

    def step(name, fn):
        try:
            findings[name] = fn() or {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 - recorded census entry
            findings[name] = {"status": "unavailable",
                              "error": f"{type(exc).__name__}: {exc}"}
            missing.append(name)

    reader, reader_stamp = load_rung_reader(Path(args.rung_reader_source))
    findings["rung_reader"] = reader_stamp

    def do_admission():
        record = admit_bank_rungs(
            Path(args.rung_index_root), args.kernel_build_id, reader)
        record.pop("table")
        record["all_allow"] = all(
            d.get("status") == "allow" for d in record["decisions"].values())
        return record

    step("rung_admission", do_admission)
    step("split_manifest",
         lambda: verify_split_manifest(Path(args.capture_root)))

    def do_source():
        record = load_source_weight(Path(args.source), args.qname)
        record.pop("tensor")
        return record

    step("source_weight", do_source)

    def do_roles():
        split = findings["split_manifest"]
        if split.get("status") == "unavailable":
            _fail("the layer census depends on the split manifest")
        return load_layer_roles(
            Path(args.capture_root), args.qname, split,
            load_tensors={"fit": True, "heldout": args.mode == "score"})

    step("layer_roles", do_roles)
    return findings, missing


def _json_safe_census(findings: dict) -> dict:
    """Census copy without torch tensors; every loaded tensor stays a stamp."""
    safe = {}
    for name, record in findings.items():
        if isinstance(record, dict):
            record = {key: value for key, value in record.items()
                      if not torch.is_tensor(value)}
        if name == "layer_roles" and isinstance(record, dict) \
                and "roles" in record:
            stripped = {key: value for key, value in record.items()
                        if key != "roles"}
            stripped["roles"] = {
                role: {key: value for key, value in role_record.items()
                       if not torch.is_tensor(value)}
                for role, role_record in record["roles"].items()}
            safe[name] = stripped
        else:
            safe[name] = record
    return safe


def run_preflight(args) -> int:
    """Real-input census; never encodes, never claims a qualification."""
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    findings, missing = preflight_census(args)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    slices = {}
    if "source_weight" not in missing:
        tensor = load_source_weight(Path(args.source), args.qname)["tensor"]
        slices["source_4x4_sum"] = float(tensor[:4, :4].float().sum())
        del tensor
    roles = (findings.get("layer_roles") or {}).get("roles") or {}
    for role_key in ROLE_KEYS:
        record = roles.get(role_key) or {}
        if record.get("tensors_loaded"):
            hessian = record["hessian"]
            slices[f"{role_key}_hessian_2x2_sum"] = \
                float(hessian[:2, :2].double().sum())
            if record["inputs"] is not None:
                slices[f"{role_key}_first_input_row_norm"] = float(
                    record["inputs"][0].double().norm())
                slices[f"{role_key}_prefix_ids_head"] = [
                    int(v) for v in record["prefix_sample_ids"][:8]]

    result = {
        "schema": "prismaquant.block_trial_preflight.v1",
        "mode": args.mode,
        "preflight": True,
        "device": args.device,
        "qname": args.qname,
        "structure": args.structure,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "imports_ok": True,
        "arguments_parsed": True,
        "census": _json_safe_census(findings),
        "missing_prerequisites": [
            {"step": name, "error": findings[name]["error"]}
            for name in missing],
        "actual_slices": slices,
        "shared_apis": [
            "tools.pq_block_trial_math", "tools.pq_block_price_solver",
            "tools.pq_block_reference_wire", "tools.pq_block_concentration",
            "prismaquant.qnames.DOTTED_LAYER_QNAME",
            "prismaquant.digests.canonical_json_sha256",
        ],
        "encoding_performed": False,
        "scoring_performed": False,
        "gpu_qualification_claimed": False,
        "d38_claimed": False,
        "not_ready_for_encode_or_score": bool(missing),
        "elapsed_seconds": time.monotonic() - started,
    }
    path = output / "preflight.json"
    _write_json(path, result)
    print(json.dumps({"preflight": True, "out": str(path),
                      "missing": sorted(missing),
                      "sha256": file_sha256(path)}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Strict intake shared by encode and score
# ---------------------------------------------------------------------------

def strict_intake(args, heldout_tensors: bool) -> dict:
    capture_root = Path(args.capture_root)
    reader, reader_stamp = load_rung_reader(Path(args.rung_reader_source))
    admission = admit_bank_rungs(
        Path(args.rung_index_root), args.kernel_build_id, reader)
    split = verify_split_manifest(capture_root)
    source = load_source_weight(Path(args.source), args.qname)
    layer = load_layer_roles(
        capture_root, args.qname, split,
        load_tensors={"fit": True, "heldout": heldout_tensors})
    fit_hessian = layer["roles"]["fit"]["hessian"]
    if list(fit_hessian.shape) != [source["shape"][1]] * 2:
        _fail("the FIT moment does not span the source projection columns")
    if heldout_tensors:
        held = layer["roles"]["heldout"]
        if list(held["hessian"].shape) != list(fit_hessian.shape):
            _fail("the HELDOUT moment basis differs from the FIT basis")
        fit_ids = layer["roles"]["fit"]["prefix_sample_ids"]
        held_ids = held["prefix_sample_ids"]
        if fit_ids is not None and held_ids is not None \
                and fit_ids.numel() > 0 and held_ids.numel() > 0:
            validate_sample_split(fit_ids.numpy(), held_ids.numpy())
    source_tensor = source.pop("tensor")
    return {
        "qname": args.qname, "admission": admission, "split": split,
        "source": source, "source_tensor": source_tensor, "layer": layer,
        "roles": layer["roles"], "census": layer["census"],
        "reader": reader, "reader_stamp": reader_stamp,
    }


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def run_encode(args) -> int:
    """Verify receipts, admit canonically, encode the FIT-only parent bank."""
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    intake = strict_intake(args, heldout_tensors=False)
    admission = intake["admission"]
    require_all_admitted(admission)

    output = Path(args.output)
    existing = output / "parent-bank.json"
    if existing.is_file():
        bank = json.loads(existing.read_bytes())
        if bank.get("qname") != args.qname \
                or bank.get("shape") != intake["source"]["shape"]:
            _fail(f"{existing} holds a different unit/shape; encode needs a "
                  "fresh --output")
    result = encode_parent_bank(
        intake["source_tensor"], intake["roles"]["fit"]["hessian"],
        intake["roles"]["fit"]["count"], intake["split"]["fit_identity"],
        args.qname, list(BANK_RUNGS), admit_callable(intake["reader"],
                                                     admission),
        args.structure, output, device=args.device)
    if result["heldout_consumed"] is not False:
        _fail("encoder claims heldout consumption")

    record = {
        "schema": "prismaquant.block_trial_encode.v1",
        "mode": "encode", "qname": args.qname,
        "structure": args.structure, "device": args.device,
        "shape": intake["source"]["shape"],
        "bank": {
            "path": str(output / "parent-bank.json"),
            "sha256": file_sha256(output / "parent-bank.json"),
            "uniform_control_rung": CONTROL_RUNG,
            "parents": [
                {"rung": p["rung"], "bytes": p["bytes"],
                 "sha256": p["sha256"], "admission": p["admission"]}
                for p in result["parents"]],
        },
        "census": intake["census"],
        "stamps": {
            "rung_reader": intake["reader_stamp"],
            "rung_index_sha256": admission["index_sha256"],
            "table_version": admission["table_version"],
            "table_status": admission["table_status"],
            "current_version": admission["current_version"],
            "decisions": admission["decisions"],
            "split_manifest_sha256": intake["split"]["sha256"],
            "split_sha256": intake["split"]["split_sha256"],
            "layer_manifest_sha256": intake["layer"]["sha256"],
            "source_index_sha256": intake["source"]["index_sha256"],
            "source_shard": intake["source"]["shard"],
            "source_key": intake["source"]["key"],
        },
        "heldout_handling": {
            "heldout_tensors_loaded_in_encode": False,
            "heldout_file_bytes_verified_against_receipt": False,
            "heldout_payload_opened_in_encode": False,
            "heldout_count_from_publisher_verified_receipt": True,
            "heldout_used_for_encoding": False,
        },
        "claims": {
            "gpu_qualification_claimed": False,
            "d38_claimed": False,
            "served_or_g3_claim": False,
            "identity_seals_added": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    path = output / "encode-result.json"
    _write_json(path, record)
    print(json.dumps({"encode": True, "out": str(path),
                      "bank": record["bank"]["path"],
                      "parents": len(result["parents"]),
                      "sha256": file_sha256(path)}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def price_per_saved_bit(prices_raw: np.ndarray, body: np.ndarray,
                        diagnostic_rung: int, order: list[int]) -> dict:
    """Actual conditional price per saved body bit for one downgrade.

    ``raw_delta[b, D] / (8 * (body[b, control] - body[b, D]))`` where the
    saved bytes are positive; zero-saving blocks are reported separately and
    never divided.  Signed values are preserved: a positive entry is the
    exact single-block output-error increase per saved bit when block b alone
    switches from the R1024 control to the diagnostic rung.
    """
    control_col = order.index(CONTROL_RUNG)
    diag_col = order.index(diagnostic_rung)
    saved = body[:, control_col] - body[:, diag_col]
    raw = prices_raw[:, diag_col]
    positive = saved > 0
    zero = saved == 0
    negative = saved < 0
    # Full-length per-bit vector: zeros where the alternative saves no bytes,
    # so the clamped CDF runs over ALL blocks, as requested.
    per_bit_full = np.zeros(saved.size, dtype=np.float64)
    per_bit_full[positive] = \
        raw[positive] / (8.0 * saved[positive].astype(np.float64))
    any_positive = bool((per_bit_full > 0).any())
    cdf = (concentration(torch.from_numpy(per_bit_full).clamp_min(0))
           if any_positive else None)
    per_bit = per_bit_full[positive]
    return {
        "diagnostic_rung": diagnostic_rung,
        "control_rung": CONTROL_RUNG,
        "definition": f"conditional_fit_prices[b, R{diagnostic_rung}] / "
                      f"(8 * (body_costs[b, R{CONTROL_RUNG}] - "
                      f"body_costs[b, R{diagnostic_rung}])), saved bytes > 0",
        "quantity": "exact single-block output-error increase per saved "
                    "body bit",
        "baseline": "complete R1024 control residual; FIT Hessian only",
        "not_additive_joint_gains": True,
        "blocks": int(saved.size),
        "blocks_with_saved_bytes": int(positive.sum()),
        "zero_saving_blocks": int(zero.sum()),
        "negative_saving_blocks": int(negative.sum()),
        "zero_saving_block_raw_prices":
            [float(v) for v in raw[zero]] if zero.any() else [],
        "signed_sum_per_saved_bit":
            float(per_bit.sum()) if per_bit.size else 0.0,
        "positive_mass_blocks": int((per_bit > 0).sum()),
        "negative_mass_blocks": int((per_bit < 0).sum()),
        "positive_mass_fraction_of_saved_byte_blocks":
            float((per_bit > 0).mean()) if per_bit.size else None,
        "positive_sensitivity_cdf_over_all_blocks": cdf,
        "cdf_population": "all blocks; clamped negative per-bit entries "
                          "contribute zero mass",
        "no_positive_mass_reason": None if any_positive
        else "no block raises its exact single-block output error per "
             "saved bit",
    }


def sensitivity_arm(error: torch.Tensor, h_fit: torch.Tensor,
                    fit_count: int, control_rates: np.ndarray,
                    rb: int, cb: int) -> dict:
    """Isolated block energy per REAL existing body bit, one geometry."""
    scores = block_scores(error, h_fit, rb, cb)
    nr, nc = scores.shape
    bits = torch.zeros(nr, nc, dtype=torch.float64)
    for j in range(nc):
        bits[:, j] = float(rb * int(control_rates[j * cb:(j + 1) * cb].sum()))
    per_bit = scores.double() / (fit_count * bits)
    return {
        "geometry": f"{rb}x{cb}",
        "weight_positions_per_block": rb * cb,
        "score": "isolated baseline-residual block energy on the FIT Hessian",
        "bits_denominator": "row_span * actual per-column rate sum of the "
                            "control",
        "retained_rows": concentration(per_bit.flatten()),
        "sum_isolated_energy": float(scores.double().sum()),
        "per_real_existing_body_bit_array": per_bit.flatten().numpy(),
    }


def run_score(args) -> int:
    """Allocate, pack at control bytes, decode, evaluate complete error."""
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    intake = strict_intake(args, heldout_tensors=True)
    admission = intake["admission"]
    require_all_admitted(admission)
    output = Path(args.output)
    bank_record = load_parent_bank(output, intake)
    bank = bank_record["bank"]
    # The actual D41 gate is the CURRENT canonical admission; a mere table
    # version move with every bank rung still allowed is stamped and
    # tolerated, while a rung that now refuses forces a re-encode.
    stamped = {str(int(p["rung"])): p["admission"] for p in bank["parents"]}
    admission_drift = {}
    for rung in BANK_RUNGS:
        key = str(rung)
        recorded, current = stamped[key], admission["decisions"][key]
        if recorded != current:
            if current.get("status") != "allow":
                _fail(f"rung {rung} is no longer admitted by the current "
                      f"table: {current} (bank recorded {recorded}); "
                      "re-encode")
            admission_drift[key] = {"recorded": recorded,
                                    "current": current}
    parsed_record = parse_and_render(bank_record["parents"],
                                     intake["source"]["shape"])
    parsed, rendered = parsed_record["parsed"], parsed_record["rendered"]

    source = intake["source_tensor"].float()
    fit = intake["roles"]["fit"]
    held = intake["roles"]["heldout"]
    h_fit, fit_count = fit["hessian"], fit["count"]
    h_held, held_count = held["hessian"], held["count"]
    control = rendered[CONTROL_RUNG]
    control_bytes = bank_record["parents"][CONTROL_RUNG]["bytes"]
    control_rates = np.asarray(parsed[CONTROL_RUNG].unit.rates,
                               dtype=np.int64)

    order = candidate_order()
    candidates = [rendered[rung] for rung in order]
    units = [parsed[rung] for rung in order]

    primary_rb, primary_cb = PRIMARY_GEOMETRY
    rows, cols = source.shape
    for rb, cb in set(SENSITIVITY_GEOMETRIES) | {PRIMARY_GEOMETRY}:
        if rows % rb or cols % cb:
            _fail(f"geometry {rb}x{cb} does not tile {rows}x{cols}")

    prices_raw = conditional_fit_prices(
        source, control, candidates, h_fit, fit_count,
        primary_rb, primary_cb)
    groups = block_groups(rows, cols, primary_rb, primary_cb,
                          INPUT_GROUP_COLS)
    prices_shrunk = shrink_group_prices(prices_raw, groups, SHRINKAGE)
    body = body_costs(units, primary_rb, primary_cb)
    fixed = fixed_bytes(units, primary_rb, primary_cb)
    cap = control_bytes - fixed
    if cap <= 0:
        _fail(f"control body budget {cap} is not positive")
    allocation = allocate_body_budget(prices_shrunk, body, cap,
                                      baseline_index=0)
    selection = allocation["selection"]
    selection_2d = selection.reshape(rows // primary_rb, cols // primary_cb)
    blob, breakdown = pack_projection(units, selection_2d, primary_rb,
                                      primary_cb,
                                      target_bytes=control_bytes)
    decoded = decode_projection(blob, device="cpu")
    assembled = assemble_weight(candidates, selection, primary_rb,
                                primary_cb)
    evaluation = evaluate_packed_candidate(
        source, control, assembled, decoded, h_fit, fit_count,
        h_held, held_count, len(blob), control_bytes)

    # Actual price-per-saved-bit diagnostics: primary geometry reuses the
    # unshrunk prices and actual body costs already computed; the current
    # 128x32 geometry recomputes both with the same parents and FIT H.
    saved_bit = {}
    arrays: dict[str, np.ndarray] = {}
    for rb, cb in SAVED_BIT_GEOMETRIES:
        if (rb, cb) == PRIMARY_GEOMETRY:
            g_raw, g_body = prices_raw, body
        else:
            g_raw = conditional_fit_prices(source, control, candidates,
                                           h_fit, fit_count, rb, cb)
            g_body = body_costs(units, rb, cb)
        for rung in SAVED_BIT_DIAGNOSTIC_RUNGS:
            arm = price_per_saved_bit(g_raw, g_body, rung, order)
            saved_bit[f"{rb}x{cb}_R{rung}"] = arm
            control_col, diag_col = order.index(CONTROL_RUNG), \
                order.index(rung)
            saved_bytes = g_body[:, control_col] - g_body[:, diag_col]
            positive = saved_bytes > 0
            arrays[f"savedbit_{rb}x{cb}_R{rung}_saved_bytes"] = saved_bytes
            arrays[f"savedbit_{rb}x{cb}_R{rung}_raw_price"] = \
                g_raw[:, diag_col].astype(np.float64)
            per_bit = np.zeros(saved_bytes.size, dtype=np.float64)
            per_bit[positive] = g_raw[positive, diag_col] / \
                (8.0 * saved_bytes[positive].astype(np.float64))
            arrays[f"savedbit_{rb}x{cb}_R{rung}_per_saved_bit"] = per_bit
        arrays[f"savedbit_{rb}x{cb}_raw_prices_unshrunk"] = g_raw.astype(
            np.float64)

    sweep = {}
    for text in args.extra_geometry or []:
        rb, cb = _geometry(text)
        if (rb, cb) == PRIMARY_GEOMETRY:
            _fail(f"--extra-geometry {text} duplicates the primary geometry")
        if rows % rb or cols % cb:
            _fail(f"{text} does not tile {rows}x{cols}")
        g_raw = conditional_fit_prices(source, control, candidates, h_fit,
                                       fit_count, rb, cb)
        g_groups = block_groups(rows, cols, rb, cb, INPUT_GROUP_COLS)
        g_shrunk = shrink_group_prices(g_raw, g_groups, SHRINKAGE)
        g_body = body_costs(units, rb, cb)
        g_fixed = fixed_bytes(units, rb, cb)
        g_alloc = allocate_body_budget(g_shrunk, g_body,
                                       control_bytes - g_fixed,
                                       baseline_index=0)
        sweep[f"{rb}x{cb}"] = {
            "uniform_geometry_arm": True,
            "never_a_per_block_mixed_size_choice": True,
            "fixed_bytes": g_fixed,
            "body_cap": control_bytes - g_fixed,
            "used_body_bytes": g_alloc["used_body_bytes"],
            "unused_body_bytes": g_alloc["unused_body_bytes"],
            "predicted_proxy_loss": g_alloc["predicted_loss"],
            "selection_distinct_parents": sorted(
                int(v) for v in np.unique(g_alloc["selection"])),
            "heldout_numbers_reported": False,
        }

    error = (control - source).double()
    sensitivity = {}
    for rb, cb in SENSITIVITY_GEOMETRIES:
        arm = sensitivity_arm(error, h_fit, fit_count, control_rates, rb, cb)
        arrays[f"sens_{rb}x{cb}_per_real_body_bit"] = \
            arm.pop("per_real_existing_body_bit_array")
        sensitivity[f"{rb}x{cb}"] = arm
    primary_sens = sensitivity[f"{primary_rb}x{primary_cb}"]
    complete_total = evaluation["fit_baseline_error"] * fit_count

    arrays_path = output / "sensitivity-arrays.npz"
    np.savez(arrays_path, **arrays)

    selection_path = output / "selection.npy"
    np.save(selection_path, selection.astype(np.int64, copy=False))
    blob_path = output / "packed-blocks.bin"
    blob_path.write_bytes(blob)
    breakdown_doc = {
        "schema": "prismaquant.block_trial_breakdown.v1",
        "qname": args.qname,
        "geometry": f"{primary_rb}x{primary_cb}",
        "uniform_geometry": True,
        "control": {
            "rung": CONTROL_RUNG,
            "label": "fresh same-FIT uniform A8S R1024 whole-unit control",
            "bytes": control_bytes,
            "sha256": bank_record["parents"][CONTROL_RUNG]["sha256"],
            "rate_values": sorted(int(v) for v in set(control_rates.tolist())),
            "no_granularity_or_size_credit": True,
        },
        "candidate_order": order,
        "budget": {
            "control_total_bytes": control_bytes,
            "fixed_bytes": fixed,
            "body_cap_bytes": cap,
            "used_body_bytes": allocation["used_body_bytes"],
            "unused_body_bytes": allocation["unused_body_bytes"],
            "pad_bytes": breakdown["pad_bytes"],
            "padding_charged_and_explicit": True,
        },
        "allocation": {
            "selection": [int(v) for v in selection],
            "predicted_loss_is_solver_proxy_only": True,
            "algorithm": allocation["algorithm"],
            "limits": allocation["limits"],
        },
        "wire": breakdown,
        "files": {
            "packed_blob": {"path": str(blob_path), "bytes": len(blob),
                            "sha256": _sha_bytes(blob)},
            "selection": {"path": str(selection_path),
                          "sha256": file_sha256(selection_path),
                          "order": "row-major block index b = i*ncb + j"},
            "sensitivity_arrays": {"path": str(arrays_path),
                                   "sha256": file_sha256(arrays_path)},
        },
    }
    _write_json(output / "breakdown.json", breakdown_doc)

    result = {
        "schema": "prismaquant.block_trial_score.v1",
        "qname": args.qname,
        "structure": args.structure,
        "device": args.device,
        "shape": [int(v) for v in source.shape],
        "geometry": {
            "primary": f"{primary_rb}x{primary_cb}",
            "uniform_geometry_only": True,
            "native_format_admits_no_mixed_block_sizes": True,
            "extra_uniform_arms": sweep,
            "no_cross_geometry_winner_claimed": True,
            "heldout_used_to_select_geometry": False,
        },
        "counts": intake["census"],
        "control": breakdown_doc["control"],
        "parent_bank": {
            "path": bank_record["bank_path"],
            "sha256": bank_record["bank_sha256"],
            "rungs": order,
            "admissions": {
                str(r): bank_record["parents"][r]["entry"]["admission"]
                for r in order},
        },
        "solver": {
            "shrinkage": SHRINKAGE,
            "group_definition": "output block x 32-column decode chunk",
            "predicted_loss_proxy": allocation["predicted_loss"],
            "used_body_bytes": allocation["used_body_bytes"],
            "unused_body_bytes": allocation["unused_body_bytes"],
            "algorithm": allocation["algorithm"],
            "limits": allocation["limits"],
        },
        "packed_candidate": {
            "bytes": len(blob),
            "sha256": _sha_bytes(blob),
            "equals_control_serialized_bytes": len(blob) == control_bytes,
            "pad_bytes": breakdown["pad_bytes"],
            "path": str(blob_path),
        },
        "evaluation": evaluation,
        "actual_price_per_saved_body_bit": saved_bit,
        "sensitivity": {
            "isolated_residual_energy_proxy": sensitivity,
            "proxy_note": "isolated block energies are NOT the complete "
                          "error; cross-input-block terms are excluded from "
                          "the proxy and retained in the complete quadratic",
            "complete_fit_error_times_count": complete_total,
            "sum_isolated_energy_128x2": primary_sens["sum_isolated_energy"],
            "cross_block_term_128x2": complete_total
            - primary_sens["sum_isolated_energy"],
            "sum_isolated_is_not_full_error": True,
        },
        "stamps": {
            "rung_reader": intake["reader_stamp"],
            "rung_index_sha256": admission["index_sha256"],
            "table_version": admission["table_version"],
            "admission_drift": admission_drift,
            "admission_drift_policy": "current canonical admission is the "
                                      "gate; allowed-rung differences are "
                                      "stamped, refusing rungs re-encode",
            "fit_identity_drift": bank_record["fit_identity_drift"],
            "fit_identity_drift_policy": "scientific FIT fields gate; other "
                                         "recorded provenance drift is "
                                         "stamped, never a re-encode wall",
            "split_manifest_sha256": intake["split"]["sha256"],
            "split_sha256": intake["split"]["split_sha256"],
            "layer_manifest_sha256": intake["layer"]["sha256"],
            "source_index_sha256": intake["source"]["index_sha256"],
            "source_shard": intake["source"]["shard"],
        },
        "claims": {
            "heldout_used_for_selection": False,
            "heldout_used_for_geometry": False,
            "rung_selected_after_heldout": False,
            "served_qualification_claimed": False,
            "g3_data_used": False,
            "speed_claimed": False,
            "gpu_qualification_claimed": False,
            "identity_seals_added": False,
            "not_a_served_G3_or_speed_qualification": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    print(json.dumps({
        "score": True, "out": str(result_path),
        "geometry": f"{primary_rb}x{primary_cb}",
        "heldout_reduction": evaluation["heldout_reduction"],
        "fit_reduction": evaluation["fit_reduction"],
        "gate_passes_10pct": evaluation["gate_passes_10pct"],
        "candidate_bytes": len(blob), "control_bytes": control_bytes,
        "sha256": file_sha256(result_path)}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("encode", "score"), required=True,
                        help="encode the parent bank, or score a packed trial")
    parser.add_argument("--preflight", action="store_true",
                        help="real-input CPU census instead of the full mode")
    parser.add_argument("--capture-root", type=Path, required=True,
                        help="shared split-capture root (publisher contract)")
    parser.add_argument("--qname", required=True,
                        help="unit qname; the source key is qname + .weight")
    parser.add_argument("--source", type=Path, required=True,
                        help="BF16 model root holding "
                             "model.safetensors.index.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="persistent trial directory; encode writes the "
                             "parent bank here, score consumes it")
    parser.add_argument("--structure", choices=("routed_moe", "dense"),
                        required=True,
                        help="actual served_recipe structure spelling")
    parser.add_argument("--rung-index-root", type=Path, required=True,
                        help="fleet-ceo rung-allowability index root")
    parser.add_argument("--kernel-build-id", required=True,
                        help="e.g. e4m3mma-sm_121-d12fba61b3467f3e")
    parser.add_argument("--rung-reader-source", type=Path, required=True,
                        help="unchanged public producer rung_allowability.py")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="cpu everywhere by default; cuda is encode-only "
                             "for the parent's Blackwell entry, while "
                             "preflight and score stay CPU research entries")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--data-manifest", type=Path, default=None,
                        help="actual PB readset; binds the existing staged I/O engine")
    parser.add_argument("--extra-geometry", action="append", default=None,
                        metavar="ROWSxCOLS",
                        help="optional extra UNIFORM geometry arms reported "
                             "with FIT-only accounting (e.g. 256x2, 512x2)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_manifest is not None:
        readset_bytes = args.data_manifest.read_bytes()
        bind_residency_manifest(hashlib.sha256(readset_bytes).hexdigest())
        readset = json.loads(readset_bytes)
        INPUT_DIGESTS.update({e["path"]: e.get("sha256") for e in readset["entries"] if e.get("offset", 0) == 0})
    if args.device == "cuda":
        if args.data_manifest is None:
            _fail("CUDA encode requires a declared PB data manifest")
        activate_staged_tier_policy("ram,ssd")
    if args.threads < 1:
        _fail("--threads must be positive")
    if args.device == "cuda" and (args.preflight or args.mode == "score"):
        _fail("--device cuda is encode-only; preflight and score are CPU "
              "research entries and never allocate a GPU")
    if args.preflight:
        return run_preflight(args)
    if args.mode == "encode":
        return run_encode(args)
    return run_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
