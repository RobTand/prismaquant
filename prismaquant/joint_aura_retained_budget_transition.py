"""Closed run transition for a sealed PREPARE whose run plan corrects only its retained budget.

The GLM-5.3-Flash joint COST run of 2026-09-18 refused
``model.language_model.layers.44.mlp.experts.0.down_proj`` after 137.4 minutes
of boundary capture: the sealed plan's ``candidate_delta_bytes`` is 4,194,304
against a roster whose smallest target demands 33,554,432 and whose largest
demands 201,326,592, and ``max_windows_per_layer: 2`` is a second refusal
against a packing that needs 78 (#743). #745 derives those caps from the roster
they must admit, so a corrected plan exists -- and nothing could run it. The
pass refuses ``prepared.plan_sha256 != plan_sha256`` twice, and
``joint_aura_run_transition`` pins the sealed plan digest as a literal. A
re-prepare costs 7.41 hours (#725) and its own fixes move
``implementation_sha256``, which the same contract pins.

This module admits that one substitution, closed, and carries more than the
transition beside it rather than less:

* It binds **two** plans. ``prepared_plan`` is the plan the prepare was made
  against, pinned by the contract as a literal. ``run_plan`` is the plan the
  run executes, pinned by this receipt.
* Admission is a proof about their contents, not a name. Each key the contract
  enumerates is removed from both parsed plans, one whole key at a time, and
  the two residues must be identical. A difference anywhere else in the nested
  structure -- an added key, a removed key, a changed byte, at any depth --
  refuses. The enumerated paths are a literal in the contract: there is no
  pattern, no prefix rule and no walk.
* The receipt records both digests and the exact per-key difference, and the
  loader re-derives that difference from the bound plans and requires the
  recorded one to equal it. The record is the value a gate reads, not prose.

The residue proof is also *why* substituting ``plan_sha256`` is sound rather
than a second claim. On the run path ``tessera_joint_aura.execute`` re-derives
``source_model_identity``, ``source_execution``, ``calibration_input``,
``measured_cells``, ``reader_identity``, ``encoder_source_reuse``,
``render_origins``, ``render_comparisons``, ``projection_backend`` and
``formats_by_qname`` from the running plan and compares every one with the
prepared record. Each of those is derived from plan bytes the residue proof
holds identical -- ``model``, ``inputs``, ``calibration_input``, ``reader``,
``historical_encoder_reuse``, ``execution.projection_backend`` -- so
``plan_sha256`` is the only prepared field a retained-budget-only plan change
can reach. The retained budget itself is read by the run and by nothing the
prepare wrote: ``compute_aura_cost_streamed`` takes
``retained_operator_windows`` only on the cost leg, ``prepare_cache`` never
sees it, and the whole block reaches the prepare command as validation
(``normalize_retained_execution``) and an environment requirement
(``require_bounded_capture_environment``), neither of which produces a
measured byte.

Everything the sibling module binds is bound here unchanged: the prepared
record, its ProductionWeightCache, the campaign identity and the measured cell
count by digest; the executing package by its complete content hash with the
sealed package reconstructed from exact snippets; the executing checkout's
HEAD against the identity the checkpoint stamps. ``meta_skeleton_render_proof_v1``
is untouched and keeps refusing exactly what it refuses today.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import weakref

VERSION = "meta_skeleton_render_proof_retained_budget_v1"
SCHEMA = "prismaquant.joint_aura.run_source_transition.v2"
PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v3"
#: The budget object's own schema. It is not an admitted key, so a plan that
#: renames or reversions the budget refuses with every other residue change.
BUDGET_SCHEMA = "prismaquant.joint_retained_window_budget.v1"
_CONTRACT = {
    "source_sha256": "192e73f9d2388a80caa3a4b9da3a59fda74530bb1949fcf7b9aeadad86c52a8f",
    "git_commit": "208d340d27caa3f0e85b399893fca78a8cfebfa0",
    "prepared_plan_sha256": "0b2cc0066bb612e32af6d0c8c809912d325b2975583297eedeee97851ee545da",
    "prepared_sha256": "962207a3385e9531adaf951b823871a2fb7ff4684320e7a8e19a1d0aa85d8f16",
    "production_cache_sha256": "5bdd0f97849d1e9c6cd2ce126f582e551191b3684a48b7215471407ba13f0021",
    "campaign_identity_sha256": "beaa5ed9a00cfa1a8cf2cb8c0b4b1f838484b648581d064d6ea0ed8a1da7fd13",
    "measured_cells": 197990,
    # THE ENUMERATED SET. Every field of RetainedWindowBudget, named one at a
    # time. Each must be present in both plans and hold an exact integer; the
    # budget's own ``schema`` is deliberately absent from this list.
    "admitted_budget_keys": [
        ["execution", "retained_operator_windows", "budget", "auxiliary_reserve_bytes"],
        ["execution", "retained_operator_windows", "budget", "boundary_reserve_bytes"],
        ["execution", "retained_operator_windows", "budget", "candidate_delta_bytes"],
        ["execution", "retained_operator_windows", "budget", "load_buffer_bytes"],
        ["execution", "retained_operator_windows", "budget", "max_windows_per_layer"],
        ["execution", "retained_operator_windows", "budget", "metadata_reserve_bytes"],
        ["execution", "retained_operator_windows", "budget", "physical_limit_bytes"],
        ["execution", "retained_operator_windows", "budget", "read_page_reserve_bytes"],
        ["execution", "retained_operator_windows", "budget", "retained_render_cap_bytes"],
        ["execution", "retained_operator_windows", "budget", "runtime_reserve_bytes"],
        ["execution", "retained_operator_windows", "budget", "safety_margin_bytes"],
        ["execution", "retained_operator_windows", "budget", "statistics_cap_bytes"],
        ["execution", "retained_operator_windows", "budget", "workspace_reserve_bytes"],
    ],
    # The derivation record ``tools/derive_retained_window_budget.py`` stamps
    # beside the caps it composed. It may be absent from either plan and its
    # content is recorded verbatim in the receipt; nothing here reads it, so it
    # explains the caps and never stands for them.
    "admitted_record_keys": [
        ["retained_window_budget_derivation"],
    ],
}
_BUDGET_KEYS = tuple(tuple(path) for path in _CONTRACT["admitted_budget_keys"])
_RECORD_KEYS = tuple(tuple(path) for path in _CONTRACT["admitted_record_keys"])
# The three files that did not exist in the sealed package. Reconstruction
# omits them; all three must be present, and nothing else new may be.
_NEW_FILES = frozenset({"joint_aura_run_transition.py", "joint_aura_transitions.py",
                        "joint_aura_retained_budget_transition.py"})
# Exact old/new snippets, this module's own and not a reference to the sibling
# transition's: an edit there must never move this proof. Reverse these, omit
# the new files, and the entire old package must hash to the contract's
# source_sha256. Regenerate with tools/generate_transition_rewrites.py.
# BEGIN GENERATED REWRITES
_SOURCE_REWRITES = {'aura_cost.py': [('        from prismaquant.joint_aura_source_transition import '
                   'require_verified_transition\n',
                   '        from prismaquant.joint_aura_transitions import '
                   'require_verified_transition\n'),
                  ('\n    def _source_gradient(target, gradient):\n',
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
                   "installed source for {name}@{fmt}')\n"
                   '\n'
                   '    def _source_gradient(target, gradient):\n'),
                  ('        # outside the cotangent/projection hot path; no tensor copy is '
                   'retained.\n'
                   '        if prepared_render_identities is not None:\n',
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
                   '        if prepared_render_identities is not None:\n'),
                  ("                        or value['dtype'] != str(source.dtype)\n"
                   "                        or value['logical_bytes'] != source.numel() * "
                   'source.element_size()\n',
                   "                        or not isinstance(value['dtype'], str)\n"
                   "                        or type(value['logical_bytes']) is not int\n"),
                  ('        _refresh_packed_layer_views(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n',
                   '        _refresh_packed_layer_views(layer)\n'
                   '        _require_installed_render_sources(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n')],
 'tessera_joint_aura.py': [('        from .joint_aura_source_transition import load_transition\n',
                            '        from .joint_aura_transitions import load_transition\n'),
                           ('                          else '
                            'source_transition.measurement_source_sha256)\n'
                            '        # THE DEVICE ENVELOPE IS APPLIED HERE, after the refusals '
                            'that need no\n',
                            '                          else '
                            'source_transition.measurement_source_sha256)\n'
                            '        # THE PLAN DIGEST THE PREPARED RECORD MUST CARRY. Normally '
                            'the running\n'
                            "        # plan's: a prepared record made against another plan is a "
                            'stale record.\n'
                            '        # An admitted transition may state another one, and exactly '
                            'one kind\n'
                            '        # does -- the retained-budget transition, whose own proof '
                            'holds the two\n'
                            '        # plans byte-identical outside the budget keys its contract '
                            'enumerates,\n'
                            '        # which is what makes every other prepared field the checks '
                            'below\n'
                            '        # re-derive from the plan still the same field. The '
                            'dispatcher answers\n'
                            '        # per capability type from a literal table, so a transition '
                            'that was\n'
                            '        # never taught this refuses rather than silently reusing the '
                            "run's.\n"
                            '        prepared_plan_sha256 = plan_sha256\n'
                            '        if source_transition is not None:\n'
                            '            from .joint_aura_transitions import '
                            'transition_prepared_plan_sha256\n'
                            '            prepared_plan_sha256 = transition_prepared_plan_sha256(\n'
                            '                source_transition, plan_sha256=plan_sha256)\n'
                            '        # THE DEVICE ENVELOPE IS APPLIED HERE, after the refusals '
                            'that need no\n'),
                           ('            _preflight_run_prepared(prepared, '
                            'plan_sha256=plan_sha256,\n',
                            '            _preflight_run_prepared(prepared, '
                            'plan_sha256=prepared_plan_sha256,\n'),
                           ('            for key, value in (("plan_sha256", plan_sha256), '
                            '("implementation_sha256", implementation),\n',
                            '            for key, value in (("plan_sha256", prepared_plan_sha256), '
                            '("implementation_sha256", implementation),\n')]}
# END GENERATED REWRITES
_COMMIT = r"[0-9a-f]{40}|[0-9a-f]{64}"
_BYTES = ("producer_source_sha256", "reconstructed_source_sha256", "transition_module_sha256")
_RECEIPT_FIELDS = {"schema", "version", "execution", "original", "inputs", "plan_difference"}
_INPUT_LABELS = ("prepared_plan", "run_plan", "prepared", "campaign_identity")


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
    """Reconstruct the sealed package's bytes; any other source change fails closed."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    current, original = hashlib.sha256(), hashlib.sha256()
    seen = set()

    def update(digest, name, data):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

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
            "transition_module_sha256": _sha(root / "joint_aura_retained_budget_transition.py")}


def _bytes_identity(execution):
    _require(isinstance(execution, dict) and all(
        isinstance(execution.get(key), str) and len(execution[key]) == 64 for key in _BYTES),
        "execution record lacks the package byte identity")
    return {key: execution[key] for key in _BYTES}


def _take(plan, path):
    """Remove one enumerated key from a parsed plan; report whether it was there.

    The path names one key. Whatever that key holds -- an integer cap or a whole
    derivation record -- leaves with it, so a key that carries a subtree needs no
    prefix rule to describe. A path whose parent is not a mapping is a plan this
    transition cannot reason about, and refuses rather than silently matching
    nothing.
    """
    node = plan
    for step in path[:-1]:
        if not isinstance(node, dict):
            _require(False, f"plan path {'.'.join(path)} does not run through mappings")
        if step not in node:
            return {"present": False}
        node = node[step]
    _require(isinstance(node, dict), f"plan path {'.'.join(path)} does not run through mappings")
    if path[-1] not in node:
        return {"present": False}
    return {"present": True, "value": node.pop(path[-1])}


def _residue(plan):
    """The plan with every enumerated key removed, and what each key held."""
    residue = copy.deepcopy(plan)
    held = {}
    for path in _BUDGET_KEYS + _RECORD_KEYS:
        held[path] = _take(residue, path)
    return residue, held


def _reported(path, entry):
    """What the receipt records for one enumerated key.

    A budget cap is an integer and is recorded as itself. A record key holds a
    whole document, and the difference is stamped into every unit checkpoint
    through ``execution_provenance`` -- 36,423 of them on this campaign -- so a
    record is identified by the SHA256 of its canonical form instead. Both name
    the value exactly; only one of them is small enough to name by writing it
    down, and the plan that holds it is bound by its own digest besides.
    """
    if not entry["present"]:
        return {"present": False}
    if path in _BUDGET_KEYS:
        return {"present": True, "value": entry["value"]}
    return {"present": True,
            "canonical_sha256": hashlib.sha256(_canonical(entry["value"])).hexdigest()}


def plan_difference(prepared_plan, run_plan):
    """Admit a run plan that differs only inside the enumerated keys, and say how.

    The residues are compared as canonical JSON of the parsed plans. That is the
    object the pass consumes (``execute`` is handed the parsed ``config``), so a
    file that only reformats or reorders keys is the same plan here and both
    files are pinned by their own SHA256 besides.
    """
    _require(isinstance(prepared_plan, dict) and isinstance(run_plan, dict), "plans must be JSON objects")
    prepared_residue, prepared_held = _residue(prepared_plan)
    run_residue, run_held = _residue(run_plan)
    _require(_canonical(prepared_residue) == _canonical(run_residue),
             "run plan differs from the prepared plan outside the admitted retained-budget keys")
    for path in _BUDGET_KEYS:
        for label, held in (("prepared", prepared_held), ("run", run_held)):
            entry = held[path]
            _require(entry["present"] and type(entry["value"]) is int,
                     f"{label} plan lacks an exact integer at {'.'.join(path)}")
    for plan, label in ((prepared_plan, "prepared"), (run_plan, "run")):
        budget = plan.get("execution", {}).get("retained_operator_windows", {})
        _require(isinstance(budget, dict) and isinstance(budget.get("budget"), dict)
                 and budget["budget"].get("schema") == BUDGET_SCHEMA,
                 f"{label} plan carries no {BUDGET_SCHEMA} retained budget")
    return [{"path": list(path), "prepared_plan": _reported(path, prepared_held[path]),
             "run_plan": _reported(path, run_held[path])}
            for path in _BUDGET_KEYS + _RECORD_KEYS
            if prepared_held[path] != run_held[path]]


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
    """The package that is executing: its bytes, and the commit it runs as."""
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
    _require(isinstance(bindings, dict) and set(bindings) == set(_INPUT_LABELS),
             "unexpected input bindings")
    # The contract pins what the PREPARE was made against. The run plan is
    # pinned by this receipt and admitted by the difference proof below: its
    # digest cannot be a literal here, because the plan that corrects the caps
    # is composed from the roster after this module is written.
    for label in ("prepared_plan", "prepared", "campaign_identity"):
        _require(isinstance(bindings[label], dict) and
                 bindings[label].get("sha256") == _CONTRACT[f"{label}_sha256"], f"unapproved {label}")
    prepared_plan = json.loads(_bound(bindings["prepared_plan"], "prepared plan").read_bytes())
    run_plan = json.loads(_bound(bindings["run_plan"], "run plan").read_bytes())
    prepared = json.loads(_bound(bindings["prepared"], "prepared").read_bytes())
    _bound(bindings["campaign_identity"], "campaign identity")
    _require(prepared.get("schema") == PREPARED_SCHEMA and prepared.get("status") == "complete",
             "prepared completion")
    _require(prepared.get("implementation_sha256") == _CONTRACT["source_sha256"], "prepared source mismatch")
    _require(prepared.get("plan_sha256") == bindings["prepared_plan"]["sha256"], "prepared plan mismatch")
    _require(prepared.get("measured_cells") == _CONTRACT["measured_cells"], "prepared cell count mismatch")
    cache = prepared.get("production_cache")
    _require(isinstance(cache, dict) and cache.get("sha256") == _CONTRACT["production_cache_sha256"],
             "PWC binding mismatch")
    _bound(cache, "prepared production cache")
    difference = plan_difference(prepared_plan, run_plan)
    return {"prepared_plan": prepared_plan, "run_plan": run_plan, "prepared": prepared,
            "plan_difference": difference}


def _read_receipt(bound_receipt, *, execution):
    path = _bound(bound_receipt, "transition receipt")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == bound_receipt["sha256"], "receipt changed during read")
    receipt = json.loads(raw)
    _require(isinstance(receipt, dict) and set(receipt) == _RECEIPT_FIELDS, "unexpected receipt fields")
    _require(receipt["schema"] == SCHEMA and receipt["version"] == VERSION and receipt["original"] == _CONTRACT,
             "unapproved transition contract")
    actual = receipt["execution"]
    _require(isinstance(actual, dict) and re.fullmatch(_COMMIT, str(actual.get("git_commit", ""))) is not None
             and _bytes_identity(actual) == _bytes_identity(execution),
             "receipt execution source differs from current package")
    return receipt, raw


def create_transition(*, bindings, output):
    """Create once; the receipt is never rewritten."""
    root = Path(__file__).resolve().parents[1]
    execution = {**_actual_execution(), **_committed_package(root)}
    loaded = _load_inputs(bindings)
    receipt = {"schema": SCHEMA, "version": VERSION, "execution": execution,
               "original": dict(_CONTRACT), "inputs": bindings,
               "plan_difference": loaded["plan_difference"]}
    raw = _canonical(receipt) + b"\n"
    with Path(output).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest()}


_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class VerifiedRetainedBudgetTransition:
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
    def prepared_plan_sha256(self):
        """The plan digest the prepared record carries, which is not the running plan's."""
        return _CONTRACT["prepared_plan_sha256"]

    @property
    def execution_provenance(self):
        receipt = json.loads(self._receipt_bytes)
        return {"schema": SCHEMA, "version": VERSION, "receipt": {"sha256": self._receipt_sha256},
                "execution": receipt["execution"],
                "measurement_source_sha256": _CONTRACT["source_sha256"],
                "prepared_plan_sha256": receipt["inputs"]["prepared_plan"]["sha256"],
                "run_plan_sha256": receipt["inputs"]["run_plan"]["sha256"],
                "plan_difference": receipt["plan_difference"]}

    def measurement_identity(self, actual):
        """Rewrite the checkpoint identity's source fields to the sealed prepare's.

        Only the source fields. The plan digest the checkpoint stamps is the
        plan that ran, because that is the plan whose budget this run packed
        its windows under.
        """
        receipt = json.loads(self._receipt_bytes)
        _require(actual.get("producer_source_sha256") == receipt["execution"]["producer_source_sha256"] and
                 actual.get("git_commit") == self._observed_git_commit,
                 "actual checkpoint source does not match admitted execution")
        identity = dict(actual)
        identity["git_commit"] = _CONTRACT["git_commit"]
        identity["producer_source_sha256"] = _CONTRACT["source_sha256"]
        return identity

    def final_provenance(self):
        from .aura_cost import _aura_unit_checkpoint_path, _load_aura_unit_checkpoint
        root = Path(self._checkpoint_dir)
        manifest = json.loads((root / "manifest.json").read_bytes())
        _require(manifest["identity"]["git_commit"] == _CONTRACT["git_commit"] and
                 manifest["identity"]["producer_source_sha256"] == _CONTRACT["source_sha256"],
                 "checkpoint manifest carries another measurement source")
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
    _require(receipt["plan_difference"] == loaded["plan_difference"],
             "recorded plan difference is not the difference between the bound plans")
    _require(config == loaded["run_plan"] and plan_sha256 == receipt["inputs"]["run_plan"]["sha256"],
             "runtime plan changed")
    _require(prepared == receipt["inputs"]["prepared"], "runtime prepared binding changed")
    root = Path(checkpoint_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes())
        _require(manifest.get("identity", {}).get("git_commit") == _CONTRACT["git_commit"] and
                 manifest["identity"].get("producer_source_sha256") == _CONTRACT["source_sha256"],
                 "existing checkpoint manifest carries another measurement source")
    else:
        _require(not any((root / "units").glob("*.pkl")) if (root / "units").is_dir() else True,
                 "checkpoint units exist without a manifest")
    verified = VerifiedRetainedBudgetTransition(raw, str(Path(bound_receipt["path"])), bound_receipt["sha256"],
                                                str(root.resolve()), execution["git_commit"])
    _ISSUED.add(verified)
    return verified


def require_verified_transition(value, *, checkpoint_dir, resume, joint_activation):
    _require(type(value) is VerifiedRetainedBudgetTransition and value in _ISSUED,
             "transition must be issued by the verified receipt loader")
    _require(resume is True and joint_activation is True and checkpoint_dir is not None and
             Path(checkpoint_dir).resolve() == Path(value._checkpoint_dir),
             "transition is restricted to bound joint resume")
    _require(_sha(value._receipt_path) == value._receipt_sha256, "admitted receipt bytes changed")
    execution = _actual_execution()
    receipt = json.loads(value._receipt_bytes)
    _require(_bytes_identity(receipt["execution"]) == _bytes_identity(execution) and
             execution["git_commit"] == value._observed_git_commit, "producer source changed after admission")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-plan", "run-plan", "prepared", "campaign-identity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bindings = {name: {"path": str(getattr(args, name).resolve()), "sha256": getattr(args, name + "_sha256")}
                for name in _INPUT_LABELS}
    print(json.dumps(create_transition(bindings=bindings, output=args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
