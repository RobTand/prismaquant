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
                        "joint_aura_retained_budget_transition.py",
                        "joint_run_progress.py"})
# Exact old/new snippets, this module's own and not a reference to the sibling
# transition's: an edit there must never move this proof. Reverse these, omit
# the new files, and the entire old package must hash to the contract's
# source_sha256. Regenerate with tools/generate_transition_rewrites.py.
# BEGIN GENERATED REWRITES
_SOURCE_REWRITES = {'aura_cost.py': [('    cost_read_schedule=None,\n    profile=None,\n',
                   '    cost_read_schedule=None,\n'
                   '    progress_base: int = 0,\n'
                   '    profile=None,\n'),
                  ('        check_operator_allocation,\n',
                   '        check_operator_allocation, preflight_joint_operator_admission,\n'),
                  ('        from prismaquant.joint_aura_source_transition import '
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
                  ('    joint_prefetch_stats: list[dict] = []\n    if joint_activation:\n',
                   '    joint_prefetch_stats: list[dict] = []\n'
                   '    preflight_retained_windows = None\n'
                   '    if joint_activation:\n'),
                  ('        joint_projection_backend = '
                   'prewarm_projection_backend(joint_projection_backend, device=runner.device)\n'
                   '        from prismaquant.cost_streaming import '
                   'validate_streamed_model_identity\n',
                   '        joint_projection_backend = '
                   'prewarm_projection_backend(joint_projection_backend, device=runner.device)\n'
                   '        # Combined operator/loader/delta admission reads declared bytes only '
                   '--\n'
                   "        # the roster, each matrix's geometry, the PWC candidate file sizes "
                   'and\n'
                   '        # the sealed budget -- so it is answerable now, before the first\n'
                   '        # boundary capture, for every layer at once. It used to run inside '
                   'the\n'
                   '        # reverse loop, where an inadmissible plan cost a whole capture '
                   'before\n'
                   '        # it refused (#743). The per-layer call still re-derives its own plan\n'
                   '        # and is compared with what was admitted here through '
                   '``sealed_windows``.\n'
                   '        if operator_windows is not None:\n'
                   '            preflight_retained_windows = preflight_joint_operator_admission(\n'
                   '                {layer: [name for name in layer_names if '
                   'render_formats[name]]\n'
                   '                 for layer, layer_names in names_by_layer.items()},\n'
                   '                linears, render_formats, production_cache,\n'
                   '                policy=operator_windows, retained_budget=retained_budget,\n'
                   '                source_bytes=(None if retained_budget is None\n'
                   '                              else '
                   "retained_operator_windows['source_reserve_bytes']))\n"
                   '        from prismaquant.cost_streaming import '
                   'validate_streamed_model_identity\n'),
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
                  ('        }, n_probes=n_probes, check_memory=check_boundary_memory)\n'
                   '    def capture_source_phase(stage, layer, actual_auxiliary_bytes):\n',
                   '        }, n_probes=n_probes, check_memory=check_boundary_memory)\n'
                   '        # Every published entry file is a durable unit and the only thing the\n'
                   '        # window can advance on. The reporter owns the clock; the writer only\n'
                   '        # tells it a file landed. With no channel in the environment this is '
                   'a\n'
                   '        # log line and nothing else, which is what every run submitted '
                   'without\n'
                   '        # the transport keeps getting.\n'
                   '        from prismaquant.joint_run_progress import JointRunProgress\n'
                   '        run_progress = JointRunProgress(layers=runner.num_layers,\n'
                   '                                        partitions=len(row_offsets),\n'
                   '                                        base_units=progress_base, log=_log)\n'
                   '        run_progress.priced_units(len(completed_checkpoint_units))\n'
                   '        boundary_storage.watch_progress(run_progress)\n'
                   '    else:\n'
                   '        run_progress = None\n'
                   '\n'
                   '    def _report_units(phase_hint=None):\n'
                   '        """Fold the cost stage\'s journalled units into the run\'s one '
                   'counter."""\n'
                   '        if run_progress is None:\n'
                   '            return\n'
                   '        run_progress.priced_units(len(completed_checkpoint_units))\n'
                   '        if phase_hint is not None:\n'
                   '            run_progress.enter(phase_hint)\n'
                   '        run_progress.flush(force=True)\n'
                   '\n'
                   '    def capture_source_phase(stage, layer, actual_auxiliary_bytes):\n'),
                  ('                boundary_storage.check_auxiliary(batches)\n'
                   '    _log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} '
                   '"\n',
                   '                boundary_storage.check_auxiliary(batches)\n'
                   '    _report_units()\n'
                   '    _log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} '
                   '"\n'),
                  ("        cost_read_schedule.enter_phase('cost_tail', "
                   'len(completed_checkpoint_units))\n'
                   '    if retained_budget is not None:\n',
                   "        cost_read_schedule.enter_phase('cost_tail', "
                   'len(completed_checkpoint_units))\n'
                   '    _report_units()\n'
                   '    if retained_budget is not None:\n'),
                  ('\n    reverse_started = time.time()\n',
                   '\n    _report_units()\n    reverse_started = time.time()\n'),
                  ("            cost_read_schedule.enter_phase(f'cost_reverse_{layer:03d}_source', "
                   'len(completed_checkpoint_units))\n'
                   '        if retained_budget is not None:\n',
                   "            cost_read_schedule.enter_phase(f'cost_reverse_{layer:03d}_source', "
                   'len(completed_checkpoint_units))\n'
                   '        _report_units()\n'
                   '        if retained_budget is not None:\n'),
                  ('        _refresh_packed_layer_views(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n',
                   '        _refresh_packed_layer_views(layer)\n'
                   '        _require_installed_render_sources(layer)\n'
                   '        # Forward boundary capture leaves the final lookahead window hot.\n'),
                  ('                        sealed_windows=(None if cost_read_schedule is None '
                   'else\n'
                   '                                        '
                   'cost_read_schedule.windows_for_layer(layer)),\n',
                   '                        '
                   'sealed_windows=(cost_read_schedule.windows_for_layer(layer)\n'
                   '                                        if cost_read_schedule is not None '
                   'else\n'
                   '                                        (preflight_retained_windows or '
                   '{}).get(layer)),\n')],
 'cost_streaming.py': [('        self._check_memory = None\n        self._n_probes = 0\n',
                        '        self._check_memory = None\n'
                        '        self._progress = None\n'
                        '        self._n_probes = 0\n'),
                       ('\n    def _entry_identity(self, reference):\n',
                        '\n'
                        '    def watch_progress(self, reporter):\n'
                        '        """Report each published entry to ``joint_run_progress``, or to '
                        'nothing.\n'
                        '\n'
                        '        Publication is where a unit becomes durable, so it is the only '
                        'place\n'
                        '        the count may move: PB #480 buys a long run time on committed '
                        'work and\n'
                        '        a counter that ticked on intent would keep a wedged run alive. '
                        'The\n'
                        '        reporter is optional so a legacy or standalone caller keeps '
                        "today's\n"
                        '        behaviour byte for byte.\n'
                        '        """\n'
                        '        self._progress = reporter\n'
                        '\n'
                        '    def _entry_identity(self, reference):\n'),
                       ('            self._retire(previous)\n'
                        '        if self._check_memory is not None:\n',
                        '            self._retire(previous)\n'
                        '        if self._progress is not None:\n'
                        '            self._progress.entry(layer=boundary_index, '
                        'partition=batch_index, kind=kind)\n'
                        '        if self._check_memory is not None:\n')],
 'joint_retained_window_plan.py': [('from dataclasses import asdict, dataclass\n',
                                    'from dataclasses import asdict, dataclass, replace\n'),
                                   ("EXECUTION_SCHEMA = 'prismaquant.joint_retained_execution.v1'\n"
                                    '\n',
                                    "EXECUTION_SCHEMA = 'prismaquant.joint_retained_execution.v1'\n"
                                    'DERIVATION_SCHEMA = '
                                    "'prismaquant.joint_retained_window_budget_derivation.v1'\n"
                                    '\n'
                                    '#: The owners an operator declares: a physical bound and the '
                                    'reserves that are\n'
                                    '#: properties of the box, the runtime and the capture, not of '
                                    'the roster.\n'
                                    "DECLARED_BUDGET_FIELDS = ('physical_limit_bytes', "
                                    "'safety_margin_bytes',\n"
                                    "                          'metadata_reserve_bytes', "
                                    "'runtime_reserve_bytes',\n"
                                    "                          'workspace_reserve_bytes', "
                                    "'boundary_reserve_bytes',\n"
                                    "                          'auxiliary_reserve_bytes', "
                                    "'read_page_reserve_bytes')\n"
                                    '#: The caps that are a function of the roster the budget must '
                                    'admit. Every one\n'
                                    '#: of these is a maximum over declared bytes, so none of them '
                                    'is a judgement\n'
                                    '#: call and none of them belongs in a hand-written plan.\n'
                                    "DERIVED_BUDGET_FIELDS = ('load_buffer_bytes', "
                                    "'candidate_delta_bytes',\n"
                                    "                         'statistics_cap_bytes', "
                                    "'retained_render_cap_bytes',\n"
                                    "                         'max_windows_per_layer')\n"
                                    '#: The owners that land on the HOST side of a unified-memory '
                                    'box, which the\n'
                                    "#: kernel bounds with the container's own cgroup cap whatever "
                                    'the aggregate\n'
                                    '#: says (``CaptureMemoryGuard._check`` holds '
                                    '``memory.current`` against\n'
                                    '#: ``cap - margin`` in aggregate mode too). Retained renders '
                                    'belong here:\n'
                                    '#: ``ProductionWeightCache._load_file_tensor`` reads every '
                                    'candidate with\n'
                                    '#: ``map_location="cpu"`` and the retained window holds those '
                                    'CPU tensors for\n'
                                    '#: the whole window, while the fp32 delta and the statistics '
                                    'matrices are\n'
                                    '#: built on the device.\n'
                                    "HOST_RESIDENT_BUDGET_FIELDS = ('safety_margin_bytes', "
                                    "'metadata_reserve_bytes',\n"
                                    "                               'load_buffer_bytes', "
                                    "'read_page_reserve_bytes')\n"
                                    '\n'),
                                   ("            'source_reserve_bytes': source, "
                                    "'source_loading_reserve_bytes': load}\n",
                                    "            'source_reserve_bytes': source, "
                                    "'source_loading_reserve_bytes': load}\n"
                                    '\n'
                                    '\n'
                                    'def _roster_maximum(targets, field):\n'
                                    '    """The largest declared value of ``field``, and the '
                                    'target that sets it."""\n'
                                    '    winner = max(targets, key=lambda target: (getattr(target, '
                                    'field), target.name))\n'
                                    '    return getattr(winner, field), winner.name\n'
                                    '\n'
                                    '\n'
                                    'def derive_retained_window_budget(targets_by_layer, *, '
                                    'declared, source_bytes,\n'
                                    '                                  prefetch_workers, '
                                    'host_cap_bytes,\n'
                                    '                                  '
                                    "footprint_scope='pwc_serialized_upper_bound'):\n"
                                    '    """Derive every demand-driven cap from the roster the '
                                    'budget must admit.\n'
                                    '\n'
                                    '    An operator declares the physical bound and the reserves '
                                    'that belong to\n'
                                    '    the box, the runtime and the capture '
                                    '(``DECLARED_BUDGET_FIELDS``). The\n'
                                    '    five caps in ``DERIVED_BUDGET_FIELDS`` are not judgement '
                                    'calls: each is a\n'
                                    '    maximum over bytes the roster already states, so each is '
                                    'computed here and\n'
                                    '    the maximizing target is recorded beside it.\n'
                                    '\n'
                                    '    * ``candidate_delta_bytes`` is one FP32 delta over one '
                                    'whole matrix.\n'
                                    '      ``JointOperatorStatisticsLease.project`` charges the '
                                    'storages of a single\n'
                                    '      quantum and the replay hands it one ``(name, format)`` '
                                    'pair at a time, so\n'
                                    '      the demand is the largest single matrix in the roster, '
                                    'not a sum.\n'
                                    '    * ``load_buffer_bytes`` bounds the serialized bytes one '
                                    'load quantum reads\n'
                                    '      *at once*, and '
                                    '``ProductionWeightCache.plan_resident_windows`` closes a\n'
                                    '      quantum on it while ``prefetch_workers`` sets the '
                                    'loader concurrency. Its\n'
                                    '      floor is the single largest candidate file; at exactly '
                                    'that floor the\n'
                                    '      declared concurrency cannot be reached, because one '
                                    'file fills the\n'
                                    '      buffer. The derived value is therefore the declared '
                                    'concurrency times\n'
                                    "      that floor -- the smallest buffer at which the policy's "
                                    'own\n'
                                    '      ``prefetch_workers`` is honest.\n'
                                    '    * ``statistics_cap_bytes`` and '
                                    '``retained_render_cap_bytes`` bound one\n'
                                    '      window. The statistics matrices are device-side, so the '
                                    'aggregate\n'
                                    '      ``available_window_bytes`` is their only bound and the '
                                    "packer's\n"
                                    '      ``statistics + renders > available`` refusal already '
                                    'holds it. Retained\n'
                                    '      renders are host-side, so they are bounded twice: by '
                                    'that same aggregate\n'
                                    '      window and by ``host_cap_bytes`` less the host-resident '
                                    'owners\n'
                                    '      (``HOST_RESIDENT_BUDGET_FIELDS``), because the kernel '
                                    'holds the\n'
                                    "      container's cgroup cap whatever the aggregate says. The "
                                    'packing is\n'
                                    '      solved against both, and each cap is then set to the '
                                    'largest window the\n'
                                    '      packing actually produces. Tightening a cap to a '
                                    'maximum the packing\n'
                                    '      already satisfies cannot change a packing decision, and '
                                    'the fixed point\n'
                                    '      is asserted below rather than assumed.\n'
                                    "    * ``max_windows_per_layer`` is the worst layer's window "
                                    'count under that\n'
                                    '      packing, i.e. the fewest retained windows the physical '
                                    'budget admits. It\n'
                                    '      remains a refusal -- runtime geometry needing more '
                                    'windows than the\n'
                                    '      sealed packing still stops -- but it is no longer free '
                                    'headroom, and the\n'
                                    '      replay multiplier it implies is recorded so the cost is '
                                    'visible.\n'
                                    '\n'
                                    '    Returns ``(budget, derivation_record)``. The record is '
                                    "data for a plan's\n"
                                    '    top level; it is deliberately not a field of the budget, '
                                    'whose ``from_dict``\n'
                                    '    admits exactly its own keys.\n'
                                    '    """\n'
                                    '    if (not isinstance(declared, Mapping)\n'
                                    '            or set(declared) != '
                                    'set(DECLARED_BUDGET_FIELDS)):\n'
                                    "        raise ValueError('retained budget derivation requires "
                                    "exactly the declared owners')\n"
                                    '    for name in DECLARED_BUDGET_FIELDS:\n'
                                    '        _integer(declared[name], name,\n'
                                    '                 positive=name not in '
                                    "('boundary_reserve_bytes', 'auxiliary_reserve_bytes'))\n"
                                    "    _integer(source_bytes, 'source_bytes')\n"
                                    "    _integer(prefetch_workers, 'prefetch_workers', "
                                    'positive=True)\n'
                                    "    _integer(host_cap_bytes, 'host_cap_bytes', "
                                    'positive=True)\n'
                                    '    if not isinstance(targets_by_layer, Mapping) or not '
                                    'targets_by_layer:\n'
                                    "        raise ValueError('retained budget derivation requires "
                                    "a nonempty per-layer roster')\n"
                                    '    roster = []\n'
                                    '    for layer, targets in sorted(targets_by_layer.items()):\n'
                                    '        targets = tuple(targets)\n'
                                    '        if not targets or any(not isinstance(target, '
                                    'RetainedTarget) for target in targets):\n'
                                    "            raise ValueError(f'layer {layer} has no declared "
                                    "retained targets')\n"
                                    '        roster.extend(targets)\n'
                                    '    if len({target.name for target in roster}) != '
                                    'len(roster):\n'
                                    "        raise ValueError('retained budget derivation requires "
                                    "one owner per target name')\n"
                                    '\n'
                                    '    candidate_delta_bytes, candidate_delta_target = '
                                    "_roster_maximum(roster, 'candidate_delta_bytes')\n"
                                    '    largest_serialized_bytes, load_buffer_target = '
                                    "_roster_maximum(roster, 'largest_serialized_bytes')\n"
                                    '    load_buffer_bytes = prefetch_workers * '
                                    'largest_serialized_bytes\n'
                                    '    # Neither window cap enters ``fixed_bytes``, so the '
                                    'window space is settled\n'
                                    '    # once the two per-quantum owners above are.\n'
                                    '    probe = RetainedWindowBudget(\n'
                                    '        **declared, load_buffer_bytes=load_buffer_bytes,\n'
                                    '        candidate_delta_bytes=candidate_delta_bytes, '
                                    'statistics_cap_bytes=1,\n'
                                    '        retained_render_cap_bytes=1, '
                                    'max_windows_per_layer=1)\n'
                                    '    available = probe.available_window_bytes(source_bytes)\n'
                                    '    host_render_bound = host_cap_bytes - sum(getattr(probe, '
                                    'name)\n'
                                    '                                             for name in '
                                    'HOST_RESIDENT_BUDGET_FIELDS)\n'
                                    '    if host_render_bound <= 0:\n'
                                    "        raise RuntimeError('retained COST host owners exhaust "
                                    "the container cap before renders')\n"
                                    '    open_budget = replace(probe, '
                                    'statistics_cap_bytes=available,\n'
                                    '                          '
                                    'retained_render_cap_bytes=min(available, host_render_bound),\n'
                                    '                          '
                                    'max_windows_per_layer=max(len(tuple(targets))\n'
                                    '                                                    for '
                                    'targets in targets_by_layer.values()))\n'
                                    '    plans = {layer: plan_retained_targets(targets, '
                                    'budget=open_budget,\n'
                                    '                                          '
                                    'source_bytes=source_bytes,\n'
                                    '                                          '
                                    'footprint_scope=footprint_scope)\n'
                                    '             for layer, targets in '
                                    'sorted(targets_by_layer.items())}\n'
                                    '    windows = [window for plan in plans.values() for window '
                                    'in plan.windows]\n'
                                    '    budget = replace(open_budget,\n'
                                    '                     '
                                    'statistics_cap_bytes=max(window.statistics_bytes for window '
                                    'in windows),\n'
                                    '                     '
                                    'retained_render_cap_bytes=max(window.render_bytes for window '
                                    'in windows),\n'
                                    '                     '
                                    'max_windows_per_layer=max(len(plan.windows) for plan in '
                                    'plans.values()))\n'
                                    '    settled = {layer: plan_retained_targets(targets, '
                                    'budget=budget, source_bytes=source_bytes,\n'
                                    '                                            '
                                    'footprint_scope=footprint_scope)\n'
                                    '               for layer, targets in '
                                    'sorted(targets_by_layer.items())}\n'
                                    '    if any(settled[layer].windows != plan.windows for layer, '
                                    'plan in plans.items()):\n'
                                    "        raise RuntimeError('retained budget derivation did "
                                    "not reach a fixed point')\n"
                                    '\n'
                                    '    record = {\n'
                                    "        'schema': DERIVATION_SCHEMA,\n"
                                    "        'footprint_scope': footprint_scope,\n"
                                    "        'source_bytes': source_bytes,\n"
                                    "        'declared': {name: declared[name] for name in "
                                    'DECLARED_BUDGET_FIELDS},\n'
                                    "        'prefetch_workers': prefetch_workers,\n"
                                    "        'host_cap_bytes': host_cap_bytes,\n"
                                    "        'host_render_bound_bytes': host_render_bound,\n"
                                    "        'roster': {'targets': len(roster), 'layers': "
                                    'len(targets_by_layer),\n'
                                    "                   'targets_by_layer': {str(layer): "
                                    'len(tuple(targets))\n'
                                    '                                        for layer, targets in '
                                    'sorted(targets_by_layer.items())}},\n'
                                    "        'demand': {\n"
                                    "            'candidate_delta_bytes': {'bytes': "
                                    'candidate_delta_bytes,\n'
                                    "                                      'maximizing_target': "
                                    'candidate_delta_target,\n'
                                    "                                      'basis': 'one fp32 "
                                    "delta over one whole matrix'},\n"
                                    "            'load_buffer_bytes': {'bytes': "
                                    'load_buffer_bytes,\n'
                                    "                                  'maximizing_target': "
                                    'load_buffer_target,\n'
                                    "                                  'largest_serialized_bytes': "
                                    'largest_serialized_bytes,\n'
                                    "                                  'prefetch_workers': "
                                    'prefetch_workers,\n'
                                    "        'host_cap_bytes': host_cap_bytes,\n"
                                    "        'host_render_bound_bytes': host_render_bound,\n"
                                    "                                  'basis': 'declared loader "
                                    "concurrency times the largest '\n"
                                    "                                           'single serialized "
                                    "candidate file'},\n"
                                    "            'statistics_cap_bytes': {'bytes': "
                                    'budget.statistics_cap_bytes,\n'
                                    "                                     'basis': 'largest packed "
                                    "window statistics'},\n"
                                    "            'retained_render_cap_bytes': {'bytes': "
                                    'budget.retained_render_cap_bytes,\n'
                                    "                                          'host_cap_bytes': "
                                    'host_cap_bytes,\n'
                                    '                                          '
                                    "'host_render_bound_bytes': host_render_bound,\n"
                                    '                                          '
                                    "'aggregate_window_bytes': available,\n"
                                    "                                          'basis': 'largest "
                                    "packed window render files, under the '\n"
                                    "                                                   'smaller "
                                    "of the aggregate window and the '\n"
                                    '                                                   '
                                    '"container\'s host-side headroom"},\n'
                                    "            'max_windows_per_layer': {'windows': "
                                    'budget.max_windows_per_layer,\n'
                                    "                                      'basis': 'worst layer "
                                    "under the physical window bound'},\n"
                                    '        },\n'
                                    "        'fixed_bytes': budget.fixed_bytes(source_bytes),\n"
                                    "        'available_window_bytes': "
                                    'budget.available_window_bytes(source_bytes),\n'
                                    "        'windows_by_layer': {str(layer): len(plan.windows) "
                                    'for layer, plan in sorted(settled.items())},\n'
                                    "        'peak_planned_bytes': max(window.peak_planned_bytes\n"
                                    '                                  for plan in '
                                    'settled.values() for window in plan.windows),\n'
                                    "        'retained_window_replay_multiplier': "
                                    '(sum(len(plan.windows) for plan in settled.values())\n'
                                    '                                              / '
                                    'len(settled)),\n'
                                    "        'budget': budget.as_dict(),\n"
                                    '    }\n'
                                    '    return budget, record\n')],
 'joint_statistics_replay.py': [('from contextlib import contextmanager\nimport os\n',
                                 'from contextlib import contextmanager\n'
                                 'from dataclasses import dataclass\n'
                                 'import os\n'),
                                ('\ndef observe_and_project_retained_windows(\n',
                                 '\n'
                                 '@dataclass(frozen=True)\n'
                                 'class PreflightRetainedWindow:\n'
                                 '    """One admitted window, in the shape ``sealed_windows`` '
                                 'already compares.\n'
                                 '\n'
                                 '    A sealed PrismaBuild read schedule and this preflight answer '
                                 'the same\n'
                                 '    question from the same declared bytes, so they reach the '
                                 'per-layer replay\n'
                                 '    through one channel instead of two.\n'
                                 '    """\n'
                                 '    original_full_target_names: tuple[str, ...]\n'
                                 '    statistics_bytes: int\n'
                                 '    render_file_upper_bound_bytes: int\n'
                                 '    candidate_count: int\n'
                                 '\n'
                                 '\n'
                                 'def retained_admission_targets(statistics_plan, specs, cache):\n'
                                 '    """Join a statistics plan to the PWC\'s declared candidate '
                                 'file sizes.\n'
                                 '\n'
                                 '    Reads no tensor. ``resolve_key`` is an index lookup and '
                                 '``estimate_nbytes``\n'
                                 '    is one ``stat`` per candidate file, so every byte this '
                                 'returns is declared\n'
                                 '    before any capture, probe or projection runs.\n'
                                 '    """\n'
                                 '    keys_by_name, requested_by_name = {}, {}\n'
                                 '    for target in statistics_plan.targets:\n'
                                 '        requested = tuple((target.name, fmt) for fmt in '
                                 'specs[target.name])\n'
                                 '        keys = tuple(cache.resolve_key(name, fmt) for name, fmt '
                                 'in requested)\n'
                                 '        if any(key is None for key in keys):\n'
                                 '            missing = [pair for pair, key in zip(requested, '
                                 'keys) if key is None]\n'
                                 "            raise RuntimeError(f'retained joint PWC candidate "
                                 "entry missing: {missing}')\n"
                                 '        keys_by_name[target.name] = keys\n'
                                 '        requested_by_name[target.name] = requested\n'
                                 '    selected_keys = tuple(key for target in '
                                 'statistics_plan.targets\n'
                                 '                          for key in keys_by_name[target.name])\n'
                                 '    # File length is the sealed conservative storage bound. '
                                 'Archive validation\n'
                                 '    # belongs to the existing PWC window immediately before its '
                                 'first read,\n'
                                 '    # not an all-candidate header walk ahead of the PB read '
                                 'frontier.\n'
                                 '    key_costs = {}\n'
                                 '    for key in selected_keys:\n'
                                 '        size = cache.estimate_nbytes([key])\n'
                                 "        key_costs[key] = {'incoming_storage_bytes': size, "
                                 "'serialized_bytes': size}\n"
                                 '    targets = targets_from_statistics_plan(statistics_plan, '
                                 'keys_by_name, key_costs)\n'
                                 '    return keys_by_name, requested_by_name, targets\n'
                                 '\n'
                                 '\n'
                                 'def preflight_joint_operator_admission(names_by_layer, modules, '
                                 'formats_by_name, cache, *,\n'
                                 '                                       policy, '
                                 'retained_budget=None, source_bytes=None):\n'
                                 '    """Refuse an inadmissible operator-window plan before any '
                                 'capture work.\n'
                                 '\n'
                                 '    Every input is declared now: the target roster, each '
                                 "matrix's geometry, the\n"
                                 "    PWC's candidate file sizes and the sealed budget. Nothing "
                                 'here reads a\n'
                                 '    captured activation, a cotangent or a probe, which is '
                                 'exactly why the\n'
                                 '    refusals it raises do not belong after a boundary capture '
                                 '(#743).\n'
                                 '\n'
                                 "    The decoder's weights are still the streamed meta skeleton "
                                 'at this point,\n'
                                 '    so the statistics plan is built on meta twins of the real '
                                 'modules against\n'
                                 '    the torch reference backend. '
                                 '``_joint_projection_requirements`` groups on\n'
                                 '    the resolved ``FormatSpec`` and the calibrated activation '
                                 'maximum and sizes\n'
                                 '    statistics from ``numel``; it consults the backend only to '
                                 'refuse a device\n'
                                 '    it was not prewarmed for. The roster it returns here is '
                                 'therefore the one\n'
                                 '    the fused backend returns on the installed tensors, and the '
                                 'per-layer call\n'
                                 '    re-derives it and compares through ``sealed_windows``.\n'
                                 '\n'
                                 '    Returns the admitted windows per layer when a retained '
                                 'budget is in force,\n'
                                 '    and ``None`` otherwise.\n'
                                 '    """\n'
                                 '    from . import format_registry as fr\n'
                                 '\n'
                                 '    if policy is None:\n'
                                 "        raise ValueError('joint operator admission requires an "
                                 "operator-window policy')\n"
                                 '    policy = normalize_operator_windows(policy)\n'
                                 '    if retained_budget is not None:\n'
                                 '        if isinstance(retained_budget, dict):\n'
                                 '            retained_budget = '
                                 'RetainedWindowBudget.from_dict(retained_budget)\n'
                                 '        if not isinstance(retained_budget, '
                                 'RetainedWindowBudget):\n'
                                 "            raise TypeError('retained joint admission requires a "
                                 "versioned retained budget')\n"
                                 '        if type(source_bytes) is not int or source_bytes < 0:\n'
                                 "            raise ValueError('retained joint admission requires "
                                 "a declared source byte cap')\n"
                                 '    windows_by_layer = {}\n'
                                 '    for layer, names in sorted(names_by_layer.items()):\n'
                                 '        names = tuple(names)\n'
                                 '        if not names:\n'
                                 '            continue\n'
                                 '        twins, specs = {}, {}\n'
                                 '        for name in names:\n'
                                 '            rows, columns = tuple(modules[name].weight.shape)\n'
                                 '            twins[name] = torch.nn.Linear(columns, rows, '
                                 'bias=False,\n'
                                 "                                          device='meta', "
                                 'dtype=torch.bfloat16)\n'
                                 '            specs[name] = {fmt: fr.get_format(fmt) for fmt in '
                                 'formats_by_name[name]}\n'
                                 '        # The same geometry bound '
                                 '``observe_and_project_windows`` applies per\n'
                                 '        # layer, applied to every layer before the first of them '
                                 'is captured.\n'
                                 '        largest = max(4 * rows * columns for rows, columns in\n'
                                 '                      (tuple(module.weight.shape) for module in '
                                 'twins.values()))\n'
                                 "        if largest > min(policy['max_candidate_bytes'], "
                                 "policy['workspace_reserve_bytes']):\n"
                                 "            raise RuntimeError('joint single target exceeds "
                                 "candidate or matrix workspace budget')\n"
                                 '        # Both replay paths need every candidate to have a PWC '
                                 'entry, and both\n'
                                 '        # used to find out per layer: the retained one through\n'
                                 '        # ``retained_admission_targets`` and the windowed one '
                                 'when\n'
                                 '        # ``resident_candidates`` planned its first quantum.\n'
                                 '        missing = [(name, fmt) for name in names for fmt in '
                                 'formats_by_name[name]\n'
                                 '                   if cache.resolve_key(name, fmt) is None]\n'
                                 '        if missing:\n'
                                 "            raise RuntimeError('joint operator-window PWC "
                                 "candidate entry missing: '\n"
                                 "                               f'{len(missing)} of "
                                 "{sum(len(formats_by_name[n]) for n in names)}, '\n"
                                 "                               f'first {missing[:8]}')\n"
                                 '        if retained_budget is None:\n'
                                 '            continue\n'
                                 '        statistics_plan = plan_joint_statistics_target_windows(\n'
                                 '            twins, specs, '
                                 'max_statistics_bytes=retained_budget.statistics_cap_bytes,\n'
                                 '            activation_max_abs=cache.activation_max_abs, '
                                 'projection_backend=None)\n'
                                 '        _, _, targets = '
                                 'retained_admission_targets(statistics_plan, specs, cache)\n'
                                 '        plan = plan_retained_targets(targets, '
                                 'budget=retained_budget,\n'
                                 '                                     source_bytes=source_bytes,\n'
                                 '                                     '
                                 "footprint_scope='pwc_serialized_upper_bound')\n"
                                 '        windows_by_layer[layer] = tuple(\n'
                                 '            PreflightRetainedWindow(window.names, '
                                 'window.statistics_bytes,\n'
                                 '                                    window.render_bytes, '
                                 'window.candidate_count)\n'
                                 '            for window in plan.windows)\n'
                                 '    return None if retained_budget is None else '
                                 'windows_by_layer\n'
                                 '\n'
                                 '\n'
                                 'def observe_and_project_retained_windows(\n'),
                                ('    keys_by_name, requested_by_name = {}, {}\n'
                                 '    for target in statistics_plan.targets:\n'
                                 '        requested = tuple((target.name, fmt) for fmt in '
                                 'specs[target.name])\n'
                                 '        keys = tuple(cache.resolve_key(name, fmt) for name, fmt '
                                 'in requested)\n'
                                 '        if any(key is None for key in keys):\n'
                                 '            missing = [pair for pair, key in zip(requested, '
                                 'keys) if key is None]\n'
                                 "            raise RuntimeError(f'retained joint PWC candidate "
                                 "entry missing: {missing}')\n"
                                 '        keys_by_name[target.name] = keys\n'
                                 '        requested_by_name[target.name] = requested\n'
                                 '    selected_keys = tuple(key for target in '
                                 'statistics_plan.targets\n'
                                 '                          for key in keys_by_name[target.name])\n'
                                 '    # File length is the sealed conservative storage bound. '
                                 'Archive validation\n'
                                 '    # belongs to the existing PWC window immediately before its '
                                 'first read,\n'
                                 '    # not an all-candidate header walk ahead of the PB read '
                                 'frontier.\n'
                                 '    key_costs = {}\n'
                                 '    for key in selected_keys:\n'
                                 '        size = cache.estimate_nbytes([key])\n'
                                 "        key_costs[key] = {'incoming_storage_bytes': size, "
                                 "'serialized_bytes': size}\n"
                                 '    targets = targets_from_statistics_plan(statistics_plan, '
                                 'keys_by_name, key_costs)\n',
                                 '    keys_by_name, requested_by_name, targets = '
                                 'retained_admission_targets(\n'
                                 '        statistics_plan, specs, cache)\n')],
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
                           ('        # The head walk reports under the one phase every joint '
                            'prepare\n'
                            "        # manifest declares. A COST run's counter belongs to its own "
                            'read\n'
                            '        # schedule, so intake there reports nothing rather than under '
                            'a name\n'
                            '        # that schedule did not declare.\n',
                            '        # The head walk reports under the one phase every joint pass '
                            'manifest\n'
                            '        # declares -- prepare and run both open on ``head``. A run '
                            'whose read\n'
                            '        # schedule is sealed separately (the V2 cost read plan) '
                            'declares\n'
                            '        # ``cost_setup``/``cost_head`` instead and no ``head``, so '
                            'intake there\n'
                            '        # reports nothing rather than under a name that schedule did '
                            'not\n'
                            '        # declare and the worker would refuse.\n'
                            '        #\n'
                            "        # Why this is not cosmetic: on ``ad8803aa`` the run's head "
                            'resolved its\n'
                            '        # 512-entry anchor roster between the 12:10:14 claim and the '
                            '16:24:17\n'
                            '        # capture line -- 4 h 14 min in which the loop knew its own '
                            'count at\n'
                            '        # every step and committed none of it, so the residency '
                            'window had\n'
                            '        # nothing to advance on before the capture had even '
                            'started.\n'),
                           ('            progress_phase=(HEAD_PHASE if command == "prepare" else '
                            'None),\n',
                            '            progress_phase=(None if cost_read_manifest is not None '
                            'else HEAD_PHASE),\n'),
                           ('            for key, value in (("plan_sha256", plan_sha256), '
                            '("implementation_sha256", implementation),\n',
                            '            for key, value in (("plan_sha256", prepared_plan_sha256), '
                            '("implementation_sha256", implementation),\n'),
                           ('                cost_read_schedule=cost_schedule,\n'
                            '                prepared_render_identities={pair: '
                            'cache.metadata["verified_cells"][pair]["rendered_weight"]\n',
                            '                cost_read_schedule=cost_schedule,\n'
                            '                # The count PrismaBuild accepts is cumulative across '
                            'phases, so\n'
                            '                # the capture continues from what the head already '
                            'committed\n'
                            '                # rather than restarting at zero, which is a '
                            'regression and\n'
                            '                # buys no time.\n'
                            '                progress_base=data.progress_committed,\n'
                            '                prepared_render_identities={pair: '
                            'cache.metadata["verified_cells"][pair]["rendered_weight"]\n')]}
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
