"""Closed, explicit source transition for a sealed joint PREPARE whose run never started.

The GLM-5.3-Flash run stage of 2026-09-18 refused its own prepared record at
the pre-install render proof: that proof compared each prepared render
identity's dtype and byte count with the meta skeleton's parameter, not with
the installed tensor prepare had verified against. The fix moves the
comparison to install time and changes no prepared byte, no arithmetic and no
gate. ``_preflight_run_prepared`` binds the complete producer package by hash,
so the fixed package cannot admit the sealed prepare on its own. This module
admits it, closed:

* ``_SOURCE_REWRITES`` carries the exact old->new source snippets, not
  patterns. ``source_proof`` reverses them, omits this module and its version
  dispatcher, and requires the reconstructed package to hash to the prepared
  ``implementation_sha256``. One byte anywhere else refuses.
* The receipt binds the plan, the prepared record, its production cache and
  the campaign identity by digest, and the actual new package by its complete
  content hash. It binds bytes, not a Git commit: PrismaBuild seals a distinct
  snapshot commit for every submission (the pbrun closure file is in the
  sealed tree), so the executing commit is recorded as an observation. In the
  campaign image there is no git binary, so that observation is the sealed
  checkout's HEAD read as a file, and the checkpoint identity's commit
  (``PRISMAQUANT_IDENTITY_GIT_COMMIT``, supplied by the container launcher)
  must equal it.
* Unlike the 2026-09-07 resume transition (``joint_aura_source_transition``)
  there is no checkpoint manifest, no preserved unit and no predecessor
  chain: the run starts fresh, and every unit it writes carries this receipt.
  An interrupted run resumes under the same receipt as long as the package
  bytes are the same.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import weakref

from .dev_mode import dev_mode_enabled, dev_stamp, dev_warning

VERSION = "meta_skeleton_render_proof_v1"
SCHEMA = "prismaquant.joint_aura.run_source_transition.v1"
PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v3"
_CONTRACT = {
    "source_sha256": "192e73f9d2388a80caa3a4b9da3a59fda74530bb1949fcf7b9aeadad86c52a8f",
    "git_commit": "208d340d27caa3f0e85b399893fca78a8cfebfa0",
    "plan_sha256": "0b2cc0066bb612e32af6d0c8c809912d325b2975583297eedeee97851ee545da",
    "prepared_sha256": "962207a3385e9531adaf951b823871a2fb7ff4684320e7a8e19a1d0aa85d8f16",
    "production_cache_sha256": "5bdd0f97849d1e9c6cd2ce126f582e551191b3684a48b7215471407ba13f0021",
    "campaign_identity_sha256": "beaa5ed9a00cfa1a8cf2cb8c0b4b1f838484b648581d064d6ea0ed8a1da7fd13",
    "measured_cells": 197990,
}
# The two files that did not exist in the sealed package. Reconstruction omits
# them; both must be present, and nothing else new may be.
_NEW_FILES = frozenset({"joint_aura_run_transition.py", "joint_aura_transitions.py"})
# Exact old/new snippets. Reverse these, omit the new files, and the entire
# old package must hash to the contract's source_sha256.
# BEGIN GENERATED REWRITES
_SOURCE_REWRITES = {'aura_cost.py': [('        # Hash actual decoded production outputs before checkpoint admission,\n'
                   '        # in layer-bounded prefetch windows. This is identity preparation,\n'
                   '        # outside the cotangent/projection hot path; no tensor copy is '
                   'retained.\n'
                   '        if prepared_render_identities is not None:\n'
                   '            expected_pairs = {(name, fmt) for name in names for fmt in '
                   'render_formats[name]}\n'
                   '            if (production_cache is None or not '
                   'isinstance(prepared_render_identities, dict)\n'
                   '                    or set(prepared_render_identities) != expected_pairs\n'
                   '                    or set(production_cache._expected_file_sha256 or {}) != '
                   'expected_pairs):\n'
                   "                raise RuntimeError('prepared render identity/file SHA coverage "
                   "differs from joint roster')\n"
                   "            verified = (production_cache.metadata or {}).get('verified_cells', "
                   '{})\n'
                   '            for name, fmt in sorted(expected_pairs):\n'
                   '                value = prepared_render_identities[name, fmt]\n'
                   '                source = linears[name].weight\n'
                   '                if (not isinstance(value, dict) or set(value) != {\n'
                   "                        'shape', 'dtype', 'logical_bytes', 'content_sha256'}\n"
                   "                        or value['shape'] != list(source.shape)\n"
                   "                        or value['dtype'] != str(source.dtype)\n"
                   "                        or value['logical_bytes'] != source.numel() * "
                   'source.element_size()\n'
                   "                        or not isinstance(value['content_sha256'], str)\n",
                   '        # Hash actual decoded production outputs before checkpoint admission,\n'
                   '        # in layer-bounded prefetch windows. This is identity preparation,\n'
                   '        # outside the cotangent/projection hot path; no tensor copy is '
                   'retained.\n'
                   "        # Before install, a streamed decoder Linear is the meta skeleton's\n"
                   "        # parameter: its shape is the checkpoint's, its dtype is torch's\n"
                   '        # default (``build_streaming_skeleton`` passes no dtype), so only the\n'
                   '        # shape is compared here. Prepare verified every render against the\n'
                   '        # INSTALLED tensor (``verify_anchor_render``); the run repeats that\n'
                   "        # comparison per layer, at install, before the layer's first render "
                   'is\n'
                   '        # consumed (``_require_installed_render_sources``).\n'
                   '        if prepared_render_identities is not None:\n'
                   '            expected_pairs = {(name, fmt) for name in names for fmt in '
                   'render_formats[name]}\n'
                   '            if (production_cache is None or not '
                   'isinstance(prepared_render_identities, dict)\n'
                   '                    or set(prepared_render_identities) != expected_pairs\n'
                   '                    or set(production_cache._expected_file_sha256 or {}) != '
                   'expected_pairs):\n'
                   "                raise RuntimeError('prepared render identity/file SHA coverage "
                   "differs from joint roster')\n"
                   "            verified = (production_cache.metadata or {}).get('verified_cells', "
                   '{})\n'
                   '            for name, fmt in sorted(expected_pairs):\n'
                   '                value = prepared_render_identities[name, fmt]\n'
                   '                source = linears[name].weight\n'
                   '                if (not isinstance(value, dict) or set(value) != {\n'
                   "                        'shape', 'dtype', 'logical_bytes', 'content_sha256'}\n"
                   "                        or value['shape'] != list(source.shape)\n"
                   "                        or not isinstance(value['dtype'], str)\n"
                   "                        or type(value['logical_bytes']) is not int\n"
                   "                        or not isinstance(value['content_sha256'], str)\n"),
                  ('    def _source_parameter(target):\n'
                   '        return target.parameter if isinstance(target, PackedExpertProjection) '
                   'else target.weight\n',
                   '    def _source_parameter(target):\n'
                   '        return target.parameter if isinstance(target, PackedExpertProjection) '
                   'else target.weight\n'
                   '\n'
                   '    def _require_installed_render_sources(layer):\n'
                   '        # Prepare compared each render with the installed source tensor. The\n'
                   "        # installed dtype is the loader's decision (``_read_layer_to_device``\n"
                   '        # casts to its dtype policy, ``_fast_install`` keeps the loaded '
                   'dtype),\n'
                   '        # not a fact the meta skeleton carries, so the dtype/byte half of the\n'
                   '        # prepared proof is checked here on the tensor the layer actually\n'
                   '        # holds, before any of its renders is consumed.\n'
                   '        if prepared_render_identities is None:\n'
                   '            return\n'
                   '        for name in names_by_layer.get(layer, ()):\n'
                   '            renders = joint_cache_renders.get(name)\n'
                   '            if not renders:\n'
                   '                continue\n'
                   '            source = linears[name].weight\n'
                   '            if source.is_meta:\n'
                   "                raise RuntimeError(f'installed source is still a meta "
                   "parameter for {name}')\n"
                   '            for fmt, value in renders.items():\n'
                   "                if (value['shape'] != list(source.shape)\n"
                   "                        or value['dtype'] != str(source.dtype)\n"
                   "                        or value['logical_bytes'] != source.numel() * "
                   'source.element_size()):\n'
                   '                    raise RuntimeError(\n'
                   "                        f'prepared render tensor proof differs from the "
                   "installed source for {name}@{fmt}')\n"),
                  ('        _refresh_packed_layer_views(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n',
                   '        _refresh_packed_layer_views(layer)\n'
                   '        _require_installed_render_sources(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n'),
                  ('        from prismaquant.joint_aura_source_transition import '
                   'require_verified_transition\n',
                   '        from prismaquant.joint_aura_transitions import '
                   'require_verified_transition\n')],
 'tessera_joint_aura.py': [('        from .joint_aura_source_transition import load_transition\n',
                            '        from .joint_aura_transitions import load_transition\n')]}
# END GENERATED REWRITES
_COMMIT = r"[0-9a-f]{40}|[0-9a-f]{64}"
_BYTES = ("producer_source_sha256", "reconstructed_source_sha256", "transition_module_sha256")


def _require(ok, message):
    if not ok:
        raise ValueError(f"joint source transition: {message}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label} requires independently bound path/SHA256")
    path = Path(record["path"])
    _require(path.is_file() and _sha(path) == record["sha256"], f"{label} bytes changed")
    return path


def source_proof(package_root=None):
    """Reconstruct the sealed package's bytes; any other source change fails closed.

    Under ``PRISMAQUANT_DEV_MODE=1`` (Rob, 2026-09-19: rapid iteration must not
    pay provenance tax) this check is a stamp, not a wall: any executing
    package is admitted, and the record carries the package's ACTUAL tree
    digest -- even a dev run records what ran, a record never a gate.
    """
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    current, original = hashlib.sha256(), hashlib.sha256()
    seen = set()

    def update(digest, name, data):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    if dev_mode_enabled():
        # The executing package is what it is; the seal returns at the
        # artifact gate, not here. Nothing is reconstructed and nothing is
        # compared against the contract -- only recorded: the ACTUAL tree
        # digest, a record never a gate.
        for path in sorted(root.rglob("*")):
            if (not path.is_file() or "__pycache__" in path.relative_to(root).parts
                    or path.suffix in {".pyc", ".pyo"}):
                continue
            update(current, path.relative_to(root).as_posix(), path.read_bytes())
        digest = current.hexdigest()
        dev_warning(
            f"source_proof admits any executing package under dev mode; "
            f"actual tree digest {digest} (contract {_CONTRACT['source_sha256']})")
        return {"producer_source_sha256": digest, "reconstructed_source_sha256": digest,
                "transition_module_sha256": _sha(root / "joint_aura_run_transition.py"),
                "dev_uncertified": True}
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.relative_to(root).parts
                or path.suffix in {".pyc", ".pyo"}):
            continue
        name = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        update(current, name, payload)
        if name in _NEW_FILES:
            seen.add(name)
            continue
        for old, new in reversed(_SOURCE_REWRITES.get(name, ())):
            _require(payload.count(new.encode()) == 1, f"unapproved or missing source hunk in {name}")
            payload = payload.replace(new.encode(), old.encode(), 1)
            seen.add(name)
        update(original, name, payload)
    _require(seen == set(_SOURCE_REWRITES) | _NEW_FILES, "incomplete source proof")
    _require(original.hexdigest() == _CONTRACT["source_sha256"], "unapproved producer package change")
    return {"producer_source_sha256": current.hexdigest(),
            "reconstructed_source_sha256": original.hexdigest(),
            "transition_module_sha256": _sha(root / "joint_aura_run_transition.py")}


def _bytes_identity(execution):
    _require(isinstance(execution, dict) and all(
        isinstance(execution.get(key), str) and len(execution[key]) == 64 for key in _BYTES),
        "execution record lacks the package byte identity")
    return {key: execution[key] for key in _BYTES}


def checkout_head_commit(repo_root):
    """The sealed checkout's HEAD commit, read as files.

    The campaign image carries no git binary. A PrismaBuild checkout is
    detached at its snapshot commit (``.git/HEAD`` holds the id); a developer
    worktree may hold ``ref: refs/heads/...`` resolved through the loose ref,
    the common directory of a linked worktree, or ``packed-refs``.
    """
    root = Path(repo_root)
    git = root / ".git"
    if git.is_file():
        pointer = git.read_text().strip()
        _require(pointer.startswith("gitdir: "), "unreadable .git pointer")
        git = Path(pointer[len("gitdir: "):])
        if not git.is_absolute():
            git = root / git
    _require(git.is_dir() and (git / "HEAD").is_file(), "no sealed Git checkout at the package root")
    head = (git / "HEAD").read_text().strip()
    if re.fullmatch(_COMMIT, head):
        return head
    _require(head.startswith("ref: "), "unreadable HEAD")
    ref = head[len("ref: "):]
    common = git
    if (git / "commondir").is_file():
        common = (git / (git / "commondir").read_text().strip()).resolve()
    for candidate in (git / ref, common / ref):
        if candidate.is_file():
            value = candidate.read_text().strip()
            _require(re.fullmatch(_COMMIT, value) is not None, f"unreadable ref {ref}")
            return value
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if not line or line[0] in "#^":
                continue
            value, _, name = line.partition(" ")
            if name == ref and re.fullmatch(_COMMIT, value):
                return value
    _require(False, f"HEAD ref {ref} is unresolved")


def _actual_execution():
    """The package that is executing: its bytes, and the commit it runs as.

    ``_checkpoint_git_commit`` is the identity the checkpoint manifest will
    carry: the override when the launcher supplied one, else git's answer with
    its exactness check, else a refusal. Whichever it is, it must be the
    sealed checkout's own HEAD; a caller-chosen label is refused here.
    """
    root = Path(__file__).resolve().parents[1]
    observed = checkout_head_commit(root)
    from .aura_cost import _checkpoint_git_commit
    commit = _checkpoint_git_commit()
    _require(commit == observed, "checkpoint Git identity contradicts the sealed checkout HEAD")
    return {"git_commit": commit, **source_proof()}


def _committed_package(repo_root):
    """Creating a receipt is a producer act: the package must be committed and clean."""
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "prismaquant"],
                                cwd=repo_root, check=True, capture_output=True, text=True, timeout=10).stdout
        parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=repo_root, check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        _require(False, f"creating a transition requires git and a committed checkout: {exc}")
    _require(not status.strip(), "producer package must be committed and clean")
    _require(re.fullmatch(_COMMIT, parent) is not None, "unreadable parent commit")
    # A PrismaBuild snapshot commit exists only in its bundle; its parent is
    # the branch commit a reader can find. Recorded, never compared.
    return {"git_parent_commit": parent}


def _load_inputs(bindings):
    _require(isinstance(bindings, dict) and set(bindings) == {"plan", "prepared", "campaign_identity"},
             "unexpected input bindings")
    for label in ("plan", "prepared", "campaign_identity"):
        _require(isinstance(bindings[label], dict) and
                 bindings[label].get("sha256") == _CONTRACT[f"{label}_sha256"], f"unapproved {label}")
    plan = json.loads(_bound(bindings["plan"], "plan").read_bytes())
    prepared = json.loads(_bound(bindings["prepared"], "prepared").read_bytes())
    _bound(bindings["campaign_identity"], "campaign identity")
    _require(prepared.get("schema") == PREPARED_SCHEMA and prepared.get("status") == "complete",
             "prepared completion")
    _require(prepared.get("implementation_sha256") == _CONTRACT["source_sha256"], "prepared source mismatch")
    _require(prepared.get("plan_sha256") == bindings["plan"]["sha256"], "prepared plan mismatch")
    _require(prepared.get("measured_cells") == _CONTRACT["measured_cells"], "prepared cell count mismatch")
    cache = prepared.get("production_cache")
    _require(isinstance(cache, dict) and cache.get("sha256") == _CONTRACT["production_cache_sha256"],
             "PWC binding mismatch")
    _bound(cache, "prepared production cache")
    return {"plan": plan, "prepared": prepared}


def _read_receipt(bound_receipt, *, execution):
    path = _bound(bound_receipt, "transition receipt")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == bound_receipt["sha256"], "receipt changed during read")
    receipt = json.loads(raw)
    _require(isinstance(receipt, dict) and set(receipt) == {"schema", "version", "execution", "original", "inputs"},
             "unexpected receipt fields")
    _require(receipt["schema"] == SCHEMA and receipt["version"] == VERSION and receipt["original"] == _CONTRACT,
             "unapproved transition contract")
    actual = receipt["execution"]
    if (_bytes_identity(actual) != _bytes_identity(execution) and dev_mode_enabled()):
        # Dev mode (Rob, 2026-09-19): any executing package is admitted under
        # the receipt; both identities are recorded, never gated.
        dev_warning(
            "receipt execution source differs from the executing package; "
            f"recorded, not gated: receipt={_bytes_identity(actual)} "
            f"current={_bytes_identity(execution)}")
    else:
        _require(isinstance(actual, dict) and re.fullmatch(_COMMIT, str(actual.get("git_commit", ""))) is not None
                 and _bytes_identity(actual) == _bytes_identity(execution),
                 "receipt execution source differs from current package")
    return receipt, raw


def create_transition(*, bindings, output):
    """Create once; the receipt is never rewritten."""
    root = Path(__file__).resolve().parents[1]
    execution = {**_actual_execution(), **_committed_package(root)}
    _load_inputs(bindings)
    receipt = {"schema": SCHEMA, "version": VERSION, "execution": execution,
               "original": dict(_CONTRACT), "inputs": bindings}
    raw = _canonical(receipt) + b"\n"
    with Path(output).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest()}


_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class VerifiedRunTransition:
    """Factory-issued immutable capability; arbitrary mappings are rejected."""
    _receipt_bytes: bytes
    _receipt_path: str
    _receipt_sha256: str
    _checkpoint_dir: str
    _observed_git_commit: str

    @property
    def measurement_source_sha256(self):
        return _CONTRACT["source_sha256"]

    @property
    def execution_provenance(self):
        receipt = json.loads(self._receipt_bytes)
        provenance = {"schema": SCHEMA, "version": VERSION, "receipt": {"sha256": self._receipt_sha256},
                      "execution": receipt["execution"], "measurement_source_sha256": _CONTRACT["source_sha256"]}
        if dev_mode_enabled():
            # Record what ran, not what the receipt sealed: the ACTUAL tree
            # digest beside the receipt's, under an unmistakable stamp. No
            # timestamp -- unit checkpoints compare this record across a
            # resume, so it must be equality-stable.
            actual = source_proof()
            provenance["execution"] = {**receipt["execution"], **actual}
            provenance.update(dev_stamp(actual["producer_source_sha256"], timestamped=False))
        return provenance

    def measurement_identity(self, actual):
        """Rewrite the checkpoint identity's source fields to the sealed prepare's.

        The package bytes must be the admitted ones and the commit must be the
        sealed checkout this run executes from; the receipt's own creation
        commit is not compared, because every PrismaBuild submission seals a
        different one over the same bytes.

        Under ``PRISMAQUANT_DEV_MODE=1`` the check is a stamp: the identity is
        returned UNREWRITTEN, carrying the executing package's actual fields,
        because even a dev run records what ran -- and the checkpoint lineage
        gate (``_prepare_aura_checkpoints``) archives, never silently reuses,
        a lineage whose recorded identity differs.
        """
        receipt = json.loads(self._receipt_bytes)
        if (actual.get("producer_source_sha256") != receipt["execution"]["producer_source_sha256"]
                or actual.get("git_commit") != self._observed_git_commit):
            if not dev_mode_enabled():
                _require(False, "actual checkpoint source does not match admitted execution")
            dev_warning(
                "actual checkpoint source differs from the admitted receipt; "
                f"recorded, not gated: checkpoint={actual.get('producer_source_sha256')} "
                f"receipt={receipt['execution']['producer_source_sha256']}")
            return dict(actual)
        identity = dict(actual)
        identity["git_commit"] = _CONTRACT["git_commit"]
        identity["producer_source_sha256"] = _CONTRACT["source_sha256"]
        return identity

    def final_provenance(self):
        from .aura_cost import _aura_unit_checkpoint_path, _load_aura_unit_checkpoint
        root = Path(self._checkpoint_dir)
        manifest = json.loads((root / "manifest.json").read_bytes())
        if (manifest["identity"]["git_commit"] != _CONTRACT["git_commit"]
                or manifest["identity"]["producer_source_sha256"] != _CONTRACT["source_sha256"]):
            if not dev_mode_enabled():
                _require(False, "checkpoint manifest carries another measurement source")
            dev_warning(
                "checkpoint manifest carries another measurement source; "
                f"recorded, not gated: manifest={manifest['identity'].get('producer_source_sha256')} "
                f"contract={_CONTRACT['source_sha256']}")
        provenance = self.execution_provenance
        count = 0
        for row in manifest["units"]:
            name = row["qname"]
            path = _aura_unit_checkpoint_path(root, name)
            _require(path.is_file(), f"missing unit checkpoint: {name}")
            state = _load_aura_unit_checkpoint(path, qname=name, identity_sha256=manifest["identity_sha256"])
            _require(state.get("execution_provenance") == provenance,
                     f"unit lacks bound execution provenance: {name}")
            count += 1
        return {**provenance, "units": count, "observed_git_commit": self._observed_git_commit}


def load_transition(bound_receipt, *, config, plan_sha256, prepared, checkpoint_dir):
    execution = _actual_execution()
    receipt, raw = _read_receipt(bound_receipt, execution=execution)
    loaded = _load_inputs(receipt["inputs"])
    _require(config == loaded["plan"] and plan_sha256 == receipt["inputs"]["plan"]["sha256"], "runtime plan changed")
    _require(prepared == receipt["inputs"]["prepared"], "runtime prepared binding changed")
    root = Path(checkpoint_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("identity", {}).get("git_commit") != _CONTRACT["git_commit"] or \
                manifest["identity"].get("producer_source_sha256") != _CONTRACT["source_sha256"]:
            if not dev_mode_enabled():
                _require(False, "existing checkpoint manifest carries another measurement source")
            dev_warning(
                "existing checkpoint manifest carries another measurement source; "
                f"recorded, not gated: manifest={manifest['identity'].get('producer_source_sha256')} "
                f"contract={_CONTRACT['source_sha256']}")
    else:
        _require(not any((root / "units").glob("*.pkl")) if (root / "units").is_dir() else True,
                 "checkpoint units exist without a manifest")
    verified = VerifiedRunTransition(raw, str(Path(bound_receipt["path"])), bound_receipt["sha256"],
                                     str(root.resolve()), execution["git_commit"])
    _ISSUED.add(verified)
    return verified


def require_verified_transition(value, *, checkpoint_dir, resume, joint_activation):
    _require(type(value) is VerifiedRunTransition and value in _ISSUED,
             "transition must be issued by the verified receipt loader")
    _require(resume is True and joint_activation is True and checkpoint_dir is not None and
             Path(checkpoint_dir).resolve() == Path(value._checkpoint_dir),
             "transition is restricted to bound joint resume")
    _require(_sha(value._receipt_path) == value._receipt_sha256, "admitted receipt bytes changed")
    execution = _actual_execution()
    receipt = json.loads(value._receipt_bytes)
    if (_bytes_identity(receipt["execution"]) != _bytes_identity(execution)
            or execution["git_commit"] != value._observed_git_commit):
        if not dev_mode_enabled():
            _require(False, "producer source changed after admission")
        dev_warning(
            "producer source changed after admission; recorded, not gated: "
            f"receipt={_bytes_identity(receipt['execution'])} "
            f"current={_bytes_identity(execution)}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "prepared", "campaign-identity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bindings = {name: {"path": str(getattr(args, name).resolve()), "sha256": getattr(args, name + "_sha256")}
                for name in ("plan", "prepared", "campaign_identity")}
    print(json.dumps(create_transition(bindings=bindings, output=args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
