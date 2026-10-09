"""The producer's packed-source -> serving-unit projection, carried by PrismaQuant.

PrismaQuant prices routed experts as per-expert 2-D units or packed decisions
estimated from sampled experts. The source units
(``model.layers.N.feed_forward.experts.<i>.w1``) follow the model profile's
declared packed split (``lfm2_moe.json`` ``projection_splits``). Tessera, the
producer, executes those units as ONE stack per MoE block (``<block>.experts``),
and publishes exactly which source tensor, which expert, which role and which
geometry each executed unit is through ``python -m tessera.producer_plan``
(schema ``tessera.expert_projection.v1``, ``tessera.serving_parts.source_identity``
for the checkpoint binding).

This module is the ONE place PrismaQuant reads that projection.  It does not
derive structure from tensor names outside the owning profile, and it never
slices a packed source tensor itself: a unit the producer projects as anything
but a whole unpacked per-expert source tensor is refused by name, because
executing ``first_half``/``second_half``/``transpose`` selectors here would be a
second home for ``export_tessera_serving.packed_expert_weight``.  Where the
producer cannot attest a unit (a layout it does not project, a stack it did not
plan) PrismaQuant refuses that unit by name rather than guessing a slice
(AGENTS.md: refuse where bytes are decided; one rule, one home).

What flows through here, in order:

* the campaign asks the producer for the projection of every in-scope stack
  (:func:`request_expert_projection`), binds it to the profile-declared units
  (:func:`bind_expert_projection`) and prices each executed unit under the
  producer's own ``unit_input_identity`` receipt;
* the allocator carries the projection block and the priced-wire receipts of
  the selected rungs into ``__prismaquant__`` unchanged;
* the export lane re-binds every selected routed unit to that projection
  (:func:`require_unit_assignment`, :func:`verify_expert_wire_record`)
  and writes the producer's ``tessera.cached_units.v1`` manifest
  (:func:`cached_units_manifest`) so the exporter packs the priced bytes
  unchanged (``--cached-expert-units``): priced == written.

RobTand/prismaquant#183.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# ``SOURCE_IDENTITY_KEYS`` is the key set ``tessera.serving_parts.source_identity``
# publishes; it, the error and the record check are lane-neutral (PQ #1555).
from .stage_inputs import (
    SOURCE_IDENTITY_KEYS, ExpertProjectionError, require_source_identity,
)
from .digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex

#: The producer's public projection schema (``tessera.export_serving.project_expert_plan``).
PROJECTION_SCHEMA = "tessera.expert_projection.v1"
#: The only source layout this bridge executes: one whole per-expert 2-D source
#: tensor per unit.  Pinned to ``tessera.serving.scheme.MOE_SOURCE_UNPACKED`` by
#: ``tests/test_tessera_expert_projection.py``; spelled here so the export gate
#: can refuse without importing the producer package.
SOURCE_LAYOUT_UNPACKED = "unpacked_per_expert"
#: The producer's whole-tensor selector for an unpacked unit.
WHOLE_SELECTOR = "whole"
#: The producer tool this bridge shells out to, as declared in
#: ``lane_specs/tessera.json`` ``campaign_tools`` (#1587: a campaign
#: dependency, not an export-arm call).
PRODUCER_PLAN_TOOL = "tessera.producer_plan"
#: Optional public producer interpreter; never the serving-runtime package pin.
PRODUCER_PYTHON_ENV = "TESSERA_PRODUCER_PYTHON"
#: The keys of a producer unit record that ``tessera.cached_unit.unit_input_identity``
#: seals into the priced-wire receipt.  Pinned against the producer by test.
UNIT_IDENTITY_KEYS = ("cols", "expert", "group", "projection", "rows",
                      "source_layout", "source_slice", "source_tensor", "tensor")
#: Where the campaign payload and the allocation carry the projection.
PROJECTION_KEY = "tessera_expert_projection"
#: Where the campaign payload and the allocation carry the priced-wire receipts
#: of projected expert units (payload: every priced rung; allocation: the
#: selected rung only).
EXPERT_WIRES_KEY = "tessera_expert_wires"
#: Where the campaign payload says which population it priced and omitted.
POPULATION_KEY = "population"
POPULATION_SCHEMA = "prismaquant.tessera_campaign_population.v2"
LEGACY_POPULATION_SCHEMA = "prismaquant.tessera_campaign_population.v1"
#: The projection block's own envelope schema inside PrismaQuant artifacts.
CARRIED_PROJECTION_SCHEMA = "prismaquant.tessera_expert_projection.v1"
#: The producer CLI option that hands the projection a stat-bound cache of
#: shard digests, so a repeated projection of unchanged checkpoint bytes does
#: not re-hash the whole source (tessera#790; PQ #2229).
SOURCE_DIGEST_CACHE_OPTION = "--source-digest-cache"
#: What the bridge writes into the returned projection at
#: ``source_digest_cache_use`` on EVERY call: the caller-side statement of
#: whether ``SOURCE_DIGEST_CACHE_OPTION`` was passed and why, so a consumer
#: never branches on the producer receipt's schema.  The producer's own
#: ``source_digest_cache`` receipt key stays the producer's (PQ #2229).
SOURCE_DIGEST_CACHE_USE_SCHEMA = "prismaquant.source_digest_cache_use.v1"


# ---------------------------------------------------------------------------
# Asking the producer
# ---------------------------------------------------------------------------
def _producer_python(env: Mapping[str, str] | None, python: str | None) -> str:
    supplied = os.environ if env is None else env
    return python or supplied.get(PRODUCER_PYTHON_ENV) or sys.executable


def _probe_producer_plan_tool(env: Mapping[str, str] | None,
                              python: str | None) -> tuple[str, str]:
    """Run the declared public CLI's ``--help`` once; return ``(module, help)``.

    CLI availability is checked in the producer interpreter, not by importing
    a serving runtime into PrismaQuant or locating a sibling checkout.  The
    one probe answers both callers: :func:`producer_plan_tool` needs the
    module, and :func:`request_expert_projection` also reads the help text to
    learn which options this installed producer carries (PQ #2229).
    """
    from .lane_spec import load_lane_spec

    for tool in load_lane_spec("tessera").campaign_tools:
        if tool.module == PRODUCER_PLAN_TOOL and tool.output_schema == PROJECTION_SCHEMA:
            completed = subprocess.run(
                [_producer_python(env, python), "-m", tool.module, "--help"],
                env=None if env is None else dict(env), capture_output=True, text=True)
            if completed.returncode:
                tail = "\n".join(completed.stderr.strip().splitlines()[-12:])
                raise ExpertProjectionError(
                    f"public producer {tool.module} unavailable (exit {completed.returncode}): {tail}")
            return tool.module, completed.stdout
    raise ExpertProjectionError(
        f"lane_specs/tessera.json campaign_tools does not declare {PRODUCER_PLAN_TOOL} "
        f"with {PROJECTION_SCHEMA}; the bridge needs the producer's explicit projection")


def producer_plan_tool(env: Mapping[str, str] | None = None, *, python: str | None = None) -> str:
    """Require the declared public CLI before an expensive packed capture.

    An explicit ``python=`` wins over ``TESSERA_PRODUCER_PYTHON``; absent both,
    the caller's interpreter remains the standalone-install default.
    """
    tool, _ = _probe_producer_plan_tool(env, python)
    return tool


def stack_plan_request(stacks: Mapping[str, tuple[str, int]]) -> dict:
    """The producer's stack-plan shape: ``{stack: {grid, q256, source_layout}}``.

    ``project_expert_plan`` requires exactly these keys with an int ``q256``.
    The projection's unit records do not depend on the rung -- it only checks
    the family route -- so the campaign asks once per stack at one legal rung
    and prices every rung against the same records.
    """
    plan = {}
    for stack, (grid, q256) in sorted(stacks.items()):
        if not isinstance(stack, str) or not stack:
            raise ExpertProjectionError("stack plan request needs non-empty stack names")
        if not isinstance(grid, str) or not grid:
            raise ExpertProjectionError(f"{stack}: stack plan request needs a grid name")
        if type(q256) is not int or q256 <= 0:
            raise ExpertProjectionError(f"{stack}: stack plan request needs a positive int q256")
        plan[stack] = {"grid": grid, "q256": q256, "source_layout": SOURCE_LAYOUT_UNPACKED}
    return plan


def _refuse_inside_source(path: Path, root: Path, *, what: str, remedy: str) -> None:
    """Refuse, by name, a path that would write inside the model source it seals."""
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        raise ExpertProjectionError(
            f"{resolved}: {what} lies inside the checkpoint {root} it would seal; {remedy}")


def _source_digest_cache_directory(override: str | Path | None, out: Path) -> Path:
    """The cache directory this projection hands the producer, not yet created.

    Defaults beside the projection output -- a campaign-owned directory,
    outside the model source tree -- and an explicit caller path wins.  The
    producer's ``SourceDigestCache`` refuses to live inside the source it
    seals; the bridge refuses that before any write, by name, so a bad
    directory is a named caller error instead of a producer traceback
    (PQ #2229, #2243).
    """
    return Path(override) if override is not None else out.parent / "source-digest-cache"


def _prepare_source_digest_cache(cache: Path) -> Path:
    """Create the cache directory, refusing an existing file, after the refusals."""
    if cache.exists() and not cache.is_dir():
        raise ExpertProjectionError(
            f"{cache.resolve()}: source digest cache is an existing file, not a directory; "
            "pass a directory the producer's SourceDigestCache can write entries into")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def request_expert_projection(model_path: str | Path, stacks: Mapping[str, tuple[str, int]],
                              *, out_path: str | Path, env: Mapping[str, str] | None = None,
                              python: str | None = None,
                              source_digest_cache: str | Path | None = None) -> dict:
    """Run the producer's projection tool ONCE for every requested stack.

    ``source_identity`` hashes every checkpoint file, so this is one subprocess
    per campaign (all stacks in scope), not one per stack.  The request is
    written beside the answer so a reader can see what was asked.  Both
    inside-source refusals -- the projection output's parent and the digest
    cache directory -- run before any write, so a refused call leaves the
    model source untouched (#2243).

    When the producer's ``--help`` advertises ``SOURCE_DIGEST_CACHE_OPTION``
    (tessera#790), the command carries it with a stat-bound shard-digest cache
    directory -- beside the projection output by default, or exactly the
    caller's ``source_digest_cache`` -- so unchanged checkpoint bytes are not
    re-hashed on the next call and the producer's ``source_digest_cache``
    receipt in the answer says how every shard digest was established.  The
    producer keeps invalidating on changed bytes; nothing here weakens that.

    Every returned projection also carries the caller's own statement at its
    own ``source_digest_cache_use`` key -- whether the option was passed, and
    why or why not -- so a consumer never branches on the producer receipt's
    schema.  That key is PrismaQuant's, not the producer's, and because
    ``carried_projection`` embeds this returned answer verbatim under
    ``producer``, the carried block's producer entry carries it too (#2243);
    nothing else in the answer is PrismaQuant's.  A producer without the
    option runs as before and is named, never silent.  A cache the caller
    asked for is never dropped silently either: an explicit
    ``source_digest_cache`` with a producer that lacks the option is refused
    by name, as is an override that is an existing file.
    """
    if not stacks:
        raise ExpertProjectionError("no stacks to project")
    child_env = dict(os.environ if env is None else env)
    producer_python = _producer_python(child_env, python)
    tool, help_text = _probe_producer_plan_tool(env=child_env, python=producer_python)
    carries_cache = SOURCE_DIGEST_CACHE_OPTION in help_text
    if not carries_cache and source_digest_cache is not None:
        raise ExpertProjectionError(
            f"producer tool {tool} does not advertise {SOURCE_DIGEST_CACHE_OPTION}; refusing "
            f"to drop the caller's source digest cache {source_digest_cache} silently (PQ #2229)")
    out = Path(out_path)
    # Both inside-source refusals run before any write (#2243): an out path
    # or cache directory inside the checkpoint must not leave the refused
    # call's request file behind in the tree every later source identity
    # hashes.
    root = Path(model_path).resolve()
    _refuse_inside_source(out.parent, root, what="projection output",
                          remedy="pass an --out path outside the model source")
    cache = None
    if carries_cache:
        cache = _source_digest_cache_directory(source_digest_cache, out)
        _refuse_inside_source(
            cache, root, what="source digest cache",
            remedy=f"pass a {SOURCE_DIGEST_CACHE_OPTION} directory outside the model source")
    out.parent.mkdir(parents=True, exist_ok=True)
    request = out.with_name(out.name + ".request.json")
    request.write_text(json.dumps(stack_plan_request(stacks), indent=1, sort_keys=True))
    command = [producer_python, "-m", tool, str(model_path),
               "--stack-plan", str(request), "--out", str(out)]
    if carries_cache:
        cache = _prepare_source_digest_cache(cache)
        command += [SOURCE_DIGEST_CACHE_OPTION, str(cache)]
    completed = subprocess.run(
        command, env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-12:])
        raise ExpertProjectionError(
            f"producer projection tool failed (exit {completed.returncode}): "
            f"{' '.join(command)}\n{tail}")
    try:
        projection = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        raise ExpertProjectionError(f"producer projection unreadable at {out}: {exc}") from exc
    # The producer's answer is kept verbatim: its ``source_digest_cache``
    # receipt key stays the producer's.  The caller's statement rides under
    # its own key on EVERY call, so a consumer never branches on the
    # receipt's schema to learn whether a cache was used.
    projection["source_digest_cache_use"] = {
        "schema": SOURCE_DIGEST_CACHE_USE_SCHEMA, "used": carries_cache,
        "reason": (f"producer tool {tool} advertises {SOURCE_DIGEST_CACHE_OPTION}; handed {cache}"
                   if carries_cache else
                   f"producer tool {tool} does not advertise {SOURCE_DIGEST_CACHE_OPTION}; "
                   "every checkpoint file was hashed"),
    }
    return projection


# ---------------------------------------------------------------------------
# Binding the projection to the profile-declared units
# ---------------------------------------------------------------------------
def unit_name_of(tensor: str) -> str:
    """``ActivationSource.unit_name``: the logical tensor name without ``.weight``."""
    if not isinstance(tensor, str) or not tensor.endswith(".weight") or len(tensor) <= 7:
        raise ExpertProjectionError(f"producer unit tensor must end with .weight: {tensor!r}")
    return tensor[:-len(".weight")]


def _validate_unit(stack: str, record: Any, declared_units: Mapping[str, tuple[int, int]],
                   tensors: Mapping[str, str]) -> tuple[str, dict]:
    if not isinstance(record, Mapping):
        raise ExpertProjectionError(f"{stack}: producer unit record is not an object")
    missing = set(UNIT_IDENTITY_KEYS) - set(record)
    if missing:
        raise ExpertProjectionError(
            f"{stack}: producer unit record lacks {sorted(missing)}")
    name = unit_name_of(record["tensor"])
    if record["source_layout"] != SOURCE_LAYOUT_UNPACKED:
        raise ExpertProjectionError(
            f"{name}: producer projects source layout {record['source_layout']!r}; this "
            f"bridge executes only {SOURCE_LAYOUT_UNPACKED!r} units, because slicing a "
            "packed source tensor here would be a second home for the producer's "
            "packed_expert_weight")
    if record["source_tensor"] != record["tensor"]:
        raise ExpertProjectionError(
            f"{name}: producer source tensor {record['source_tensor']!r} is not the unit "
            "tensor; an unpacked unit is its own whole source tensor")
    expert = record["expert"]
    if type(expert) is not int or expert < 0:
        raise ExpertProjectionError(f"{name}: producer expert id must be a non-negative int")
    expected_slice = {"expert": expert, "selector": WHOLE_SELECTOR, "transpose": False}
    if record["source_slice"] != expected_slice:
        raise ExpertProjectionError(
            f"{name}: producer source slice {record['source_slice']!r} is not the whole "
            f"unpacked tensor {expected_slice!r}")
    for key in ("projection", "group"):
        if not isinstance(record[key], str) or not record[key]:
            raise ExpertProjectionError(f"{name}: producer unit {key} must be a non-empty string")
    if name not in declared_units:
        raise ExpertProjectionError(
            f"{name}: producer projects a unit the profile does not declare in stack {stack}")
    rows, cols = record["rows"], record["cols"]
    if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0:
        raise ExpertProjectionError(f"{name}: producer rows/cols must be positive ints")
    if (rows, cols) != tuple(declared_units[name]):
        raise ExpertProjectionError(
            f"{name}: producer geometry [{rows}, {cols}] disagrees with the declared source "
            f"unit {list(declared_units[name])}")
    if record["source_tensor"] not in tensors:
        raise ExpertProjectionError(
            f"{name}: producer source tensor is not in the hashed checkpoint roster")
    return name, {key: record[key] for key in UNIT_IDENTITY_KEYS}


def bind_expert_projection(projection: Any, *,
                           declared: Mapping[str, Mapping[str, tuple[int, int]]],
                           allow_unrequested_stacks: bool = False,
                           ) -> dict[str, dict[str, dict]]:
    """Bind the producer's projection to PrismaQuant's declared units, exactly.

    ``declared`` is ``{stack: {unit_qname: (rows, cols)}}`` from the profile's
    packed split -- the units PrismaQuant prices or selected.  Every declared
    stack must be projected, and each projected stack must cover exactly its
    declared units with the same geometry: a stack executes whole, so a
    projection covering more or fewer experts than the profile declares is a
    projection of a different tensor.  Returns ``{stack: {unit_qname: unit
    record}}`` with the record trimmed to the keys the producer seals.
    """
    if not isinstance(projection, Mapping):
        raise ExpertProjectionError("producer projection is not an object")
    if projection.get("schema") != PROJECTION_SCHEMA:
        raise ExpertProjectionError(
            f"producer projection schema {projection.get('schema')!r} is not "
            f"{PROJECTION_SCHEMA!r}")
    if not {"schema", "stacks", "source"} <= set(projection):
        raise ExpertProjectionError("producer projection must carry schema, stacks and source")
    source = require_source_identity(projection["source"])
    stacks = projection["stacks"]
    if not isinstance(stacks, Mapping):
        raise ExpertProjectionError("producer projection stacks must be an object")
    if not declared:
        raise ExpertProjectionError("no declared stacks to bind")
    missing = sorted(set(declared) - set(stacks))
    if missing:
        raise ExpertProjectionError(
            f"producer projection does not plan stacks {missing}; their units are refused")
    extra = sorted(set(stacks) - set(declared))
    if extra and not allow_unrequested_stacks:
        raise ExpertProjectionError(
            f"producer projection plans stacks that were not requested: {extra}")
    bound: dict[str, dict[str, dict]] = {}
    for stack in sorted(declared):
        entry = stacks[stack]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("units"), list):
            raise ExpertProjectionError(f"{stack}: producer stack entry lacks a units list")
        if entry.get("source_layout") != SOURCE_LAYOUT_UNPACKED:
            raise ExpertProjectionError(
                f"{stack}: producer stack source layout {entry.get('source_layout')!r} is "
                f"not {SOURCE_LAYOUT_UNPACKED!r}; this bridge does not slice packed sources")
        units: dict[str, dict] = {}
        experts: set[int] = set()
        for record in entry["units"]:
            name, trimmed = _validate_unit(stack, record, declared[stack], source["tensors"])
            if name in units:
                raise ExpertProjectionError(f"{name}: producer projects the unit twice")
            units[name] = trimmed
            experts.add(int(trimmed["expert"]))
        undeclared = sorted(set(declared[stack]) - set(units))
        if undeclared:
            raise ExpertProjectionError(
                f"{stack}: producer projection does not cover declared units {undeclared}")
        # The producer states the stack's expert COUNT (``plan_expert_stack``
        # has already refused a gap or an undeclared index against
        # config.json); its units must then be exactly ``range(count)``.
        planned = entry.get("experts")
        if type(planned) is not int or sorted(experts) != list(range(planned)):
            raise ExpertProjectionError(
                f"{stack}: producer stack experts {planned!r} disagree with its units' "
                f"experts {sorted(experts)}")
        bound[stack] = units
    return bound


def carried_projection(projection: Mapping[str, Any], bound: Mapping[str, Mapping[str, dict]],
                       *, request: Mapping[str, Any], tool: str) -> dict:
    """The block the campaign payload and the allocation carry.

    The producer's answer is kept verbatim under ``producer`` -- verbatim
    except for the one caller-side key ``request_expert_projection`` adds to
    the returned answer at ``source_digest_cache_use``, which therefore rides
    inside this block (#2243).  The entry is otherwise the producer's
    statement, not PrismaQuant's, and the block is not restructured: beside
    the producer entry sit the exact binding that was priced and the request
    that produced it.
    """
    return {
        "schema": CARRIED_PROJECTION_SCHEMA,
        "tool": str(tool),
        "request": {stack: dict(entry) for stack, entry in sorted(request.items())},
        "producer": json.loads(DIRECT_ASCII_SPACED_LAX.text(projection)),
        "stacks": {stack: {name: dict(unit) for name, unit in sorted(units.items())}
                   for stack, units in sorted(bound.items())},
    }


def carried_units(carried: Any) -> tuple[dict, dict[str, dict], dict[str, str]]:
    """Read a carried projection block back: ``(source, units, stack_of)``.

    Every unit is re-validated against the producer's own answer inside the
    block, so a hand-edited allocation cannot carry a unit the producer never
    projected.
    """
    if not isinstance(carried, Mapping) or carried.get("schema") != CARRIED_PROJECTION_SCHEMA:
        raise ExpertProjectionError(
            "allocation carries no producer expert projection "
            f"({PROJECTION_KEY}: {CARRIED_PROJECTION_SCHEMA})")
    stacks = carried.get("stacks")
    if not isinstance(stacks, Mapping) or not stacks:
        raise ExpertProjectionError("carried expert projection names no stacks")
    declared = {}
    for stack, units in stacks.items():
        if not isinstance(units, Mapping):
            raise ExpertProjectionError(f"{stack}: carried stack is not an object")
        declared[stack] = {}
        for name, unit in units.items():
            if not isinstance(unit, Mapping) or set(unit) != set(UNIT_IDENTITY_KEYS):
                raise ExpertProjectionError(f"{name}: carried unit record is not sealed")
            declared[stack][name] = (unit["rows"], unit["cols"])
    bound = bind_expert_projection(carried.get("producer"), declared=declared,
                                   allow_unrequested_stacks=True)
    for stack, units in bound.items():
        for name, unit in units.items():
            if dict(stacks[stack][name]) != unit:
                raise ExpertProjectionError(
                    f"{name}: carried unit record disagrees with the producer's projection")
    source = require_source_identity(carried["producer"]["source"])
    units_flat = {name: unit for units in bound.values() for name, unit in units.items()}
    stack_of = {name: stack for stack, units in bound.items() for name in units}
    return source, units_flat, stack_of


# ---------------------------------------------------------------------------
# The source bytes the producer will read
# ---------------------------------------------------------------------------
def source_unit_weight(model_path: str | Path, source: Mapping[str, Any], unit: Mapping[str, Any],
                       *, source_authentication=None, pin_memory: bool = False):
    """Read the unit's whole source tensor from the shard the producer hashed.

    The exporter re-reads exactly this tensor (``packed_expert_weight`` on an
    unpacked unit) and re-derives the cached identity from its bytes, so the
    campaign must price these bytes and nothing else.

    The shard opens through ``layer_streaming._source_safe_open``, the seam
    every other source-shard read passes through, so under a PrismaBuild
    residency map the tensor's payload comes off the stage the same way the
    row's layer loads do, with the same declared-file fallback (PQ #1529).
    With no map that seam hands back ``safe_open`` itself, so the unmapped read
    is the call it always was.

    With ``pin_memory`` the returned tensor is page-locked for the device
    check (PQ #2039): under a staged map the payload lands in the pinned
    buffer itself with no second host copy, while the pool read and the
    qualified-original owner pin their tensor with the same single copy the
    device check always performed. Host consumers (stream head, export,
    materialization) keep the pageable default.
    """
    from .layer_streaming import _source_safe_open

    tensor = unit["source_tensor"]
    try:
        file = source["tensors"][tensor]
    except KeyError:
        raise ExpertProjectionError(f"{tensor}: not in the producer's hashed tensor roster")
    path = Path(model_path) / file
    # The qualified-original owner serves the sealed whole file and refuses
    # any opener argument beyond framework/device, so the pin request never
    # reaches it: its tensor is pinned below like the pool read (PQ #2039).
    pin_after_read = (source_authentication is not None
                      and getattr(source_authentication,
                                  'is_qualified_original_material', False))
    extra = {'pinned_host': True} if pin_memory and not pin_after_read else {}
    context = (_source_safe_open(str(path), framework="pt", device="cpu", **extra)
               if source_authentication is None else
               _source_safe_open(path, source_authentication=source_authentication,
                                 framework="pt", device="cpu", **extra))
    with context as handle:
        if tensor not in handle.keys():
            raise ExpertProjectionError(f"{tensor}: absent from {path}")
        weight = handle.get_tensor(tensor)
    if list(weight.shape) != [unit["rows"], unit["cols"]]:
        raise ExpertProjectionError(
            f"{tensor}: source shape {list(weight.shape)} disagrees with the projection "
            f"[{unit['rows']}, {unit['cols']}]")
    weight = weight.contiguous()
    if pin_memory and not weight.is_pinned():
        weight = weight.pin_memory()
    return weight


# ---------------------------------------------------------------------------
# The export side: selected units, per-unit rungs, priced-wire receipts
# ---------------------------------------------------------------------------
#: Where the allocation and the export lane carry the per-unit rungs of MIXED
#: routed stacks -- the sibling of :data:`STACK_FORMATS_KEY`.  A stack-uniform
#: world emits no such block: its artifacts and config spellings are the
#: stack-uniform ones, byte for byte (PrismaQuant #2319).
UNIT_RUNGS_KEY = "tessera_expert_unit_rungs"
UNIT_RUNGS_SCHEMA = "prismaquant.tessera_expert_unit_rungs.v1"

#: Sentinel default for ``require_unit_assignment``'s ``capability``: resolve
#: the installed Tessera runtime's per-unit capability only when a stack is
#: actually mixed, so a stack-uniform world never imports the producer
#: package and never changes behavior.
_RESOLVE_INSTALLED_CAPABILITY = object()


def _routed_unit_capability_refusal(stack: str, distinct: list[str],
                                    first_unit: str) -> str:
    return (
        f"{stack}: first mixed unit {first_unit}; selected rungs differ "
        f"across the stack {distinct}; planning per-unit rungs requires the Tessera "
        "runtime contract v57 with producer_interface.routed_units "
        "(tessera.routed-unit-assignment.v1)")


def require_unit_assignment(selected: Mapping[str, str], stack_of: Mapping[str, str],
                            units: Mapping[str, Mapping[str, Any]], *,
                            capability: Any = _RESOLVE_INSTALLED_CAPABILITY,
                            ) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """The complete per-unit assignment of the selected routed units, or refuse by name.

    The producer executes a stack whole: every projected unit of an executed
    stack must be selected (a partly selected stack refuses, exactly as
    before, and a selected unit outside the carried projection refuses).  What
    changes with the v57 per-unit contract (PrismaQuant #2319) is the rate
    axis: a stack whose selected units share one rung keeps the stack-uniform
    stamps -- ``{stack: format}``, byte for byte the spellings
    ``require_stack_uniform_assignment`` emitted -- while a stack whose units
    carry different rungs of the producer's served E4M3 grid is returned per
    unit, and only when the installed Tessera runtime publishes the per-unit
    capability (``producer_interface.routed_units``, contract v57).  A mixed
    stack on any other grid refuses by name: v57 serves per-unit rungs on
    E4M3 and BF16 only.  Without the capability the mixed stack refuses by
    unit, stack and required contract version.

    Returns ``(stack_formats, unit_rungs)``: ``stack_formats`` holds one
    format per stack-uniform executed stack; ``unit_rungs`` holds, for each
    MIXED stack, the complete ``{unit: format}`` member map.  A stack-uniform
    world returns an empty ``unit_rungs`` and never consults the capability.
    """
    by_stack: dict[str, dict[str, str]] = {}
    for name, fmt in selected.items():
        stack = stack_of.get(name)
        if stack is None:
            raise ExpertProjectionError(
                f"{name}: selected routed expert unit is not in the carried producer projection")
        by_stack.setdefault(stack, {})[name] = fmt
    stack_formats: dict[str, str] = {}
    unit_rungs: dict[str, dict[str, str]] = {}
    for stack, members in sorted(by_stack.items()):
        planned = sorted(n for n, s in stack_of.items() if s == stack)
        unselected = sorted(set(planned) - set(members))
        if unselected:
            raise ExpertProjectionError(
                f"{stack}: the producer executes the stack whole, but "
                f"{len(unselected)} of its {len(planned)} projected units are not "
                f"selected for Tessera (first: {unselected[0]})")
        distinct = sorted(set(members.values()))
        if len(distinct) == 1:
            stack_formats[stack] = distinct[0]
            continue
        ordered = sorted(members)
        reference = members[ordered[0]]
        first_unit = next((name for name in ordered if members[name] != reference), ordered[0])
        refusal = _routed_unit_capability_refusal(stack, distinct, first_unit)
        if capability is _RESOLVE_INSTALLED_CAPABILITY:
            from .tessera_runtime_contract import packaged_routed_unit_capability
            try:
                _sha, capability = packaged_routed_unit_capability()
            except Exception as exc:  # absent/stale/malformed: refuse by name
                raise ExpertProjectionError(
                    f"{refusal}; the installed Tessera runtime does not "
                    f"publish it: {exc}") from exc
        if not capability:
            raise ExpertProjectionError(refusal)
        from .tessera_formats import parse_tessera_format_name

        def _grid(fmt: str) -> str | None:
            parsed = parse_tessera_format_name(fmt)
            if parsed is None:
                return None
            spec, _rung = parsed
            return spec.base + ("" if spec.arity == 1 else f"x{spec.arity}")

        grids = sorted({grid for grid in (_grid(fmt) for fmt in distinct)
                        if grid is not None})
        spellings = sorted(fmt for fmt in distinct if _grid(fmt) is None)
        if spellings:
            raise ExpertProjectionError(
                f"{stack}: a mixed per-unit stack must carry Tessera wire "
                f"formats for every member; non-Tessera spellings "
                f"{spellings} cannot be planned beside them (first: {first_unit})")
        if len(grids) != 1:
            raise ExpertProjectionError(
                f"{stack}: per-unit rungs share one grid/family/body/plane and "
                f"differ only in rung; this stack mixes grids {grids} "
                f"(first: {first_unit})")
        if grids[0] not in {"E4M3", "BF16"}:
            raise ExpertProjectionError(
                f"{stack}: this stack carries {grids}; per-unit rungs are "
                "served on the producer's E4M3 and BF16 families only (v57)")
        unit_rungs[stack] = {name: members[name] for name in sorted(members)}
    return stack_formats, unit_rungs


def check_expert_wire_receipt(record: Any, *, name: str, unit: Mapping[str, Any],
                              q256: int, grid: str) -> dict:
    """Check a priced-wire receipt against the unit and rung it claims, without bytes.

    The allocator applies this to the receipt of every rung it selects, so an
    allocation never carries a receipt for another unit, another rung, another
    grid or another projection; the export lane adds the byte check
    (:func:`verify_expert_wire_record`) where the bytes are about to be handed
    over.  Returns the receipt's four producer fields and nothing else.
    """
    if not isinstance(record, Mapping) or set(record) != {"file", "blob_sha256",
                                                          "blob_bytes", "identity"}:
        raise ExpertProjectionError(f"{name}: priced-wire receipt is not a producer unit record")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise ExpertProjectionError(f"{name}: priced-wire receipt has no identity")
    if identity.get("unit") != name:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt is for unit {identity.get('unit')!r}")
    if identity.get("projection") != {key: unit[key] for key in UNIT_IDENTITY_KEYS}:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt was sealed under a different producer projection")
    recipe = identity.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("q256") != q256 or recipe.get("grid") != grid:
        raise ExpertProjectionError(
            f"{name}: priced-wire receipt recipe {recipe!r} is not the selected rung "
            f"(grid={grid!r}, q256={q256})")
    file = record["file"]
    if not isinstance(file, str) or Path(file).name != file or file in {".", ".."}:
        raise ExpertProjectionError(f"{name}: priced-wire receipt file is not a local leaf")
    return {key: record[key] for key in ("file", "blob_sha256", "blob_bytes", "identity")}


def verify_expert_wire_record(record: Any, *, name: str, unit: Mapping[str, Any],
                              q256: int, grid: str, wire_dir: Path) -> dict:
    """Check a carried priced-wire receipt against the unit it claims to price.

    The producer re-verifies the receipt against the identity it recomputes
    from the source bytes and the export's Hessian; this check refuses the
    cheaper contradictions first, by name: a receipt for another unit, another
    rung, another grid, another projection, or a blob that is not in the
    campaign's wire directory with the recorded bytes.
    """
    record = check_expert_wire_receipt(record, name=name, unit=unit, q256=q256, grid=grid)
    path = locate_expert_wire(record, name=name, wire_dir=wire_dir)
    blob = path.read_bytes()
    if len(blob) != record["blob_bytes"] or bytes_sha256hex(blob) != record["blob_sha256"]:
        raise ExpertProjectionError(f"{name}: priced wire {path} does not match its receipt")
    return record


def locate_expert_wire(record: Mapping[str, Any], *, name: str, wire_dir: Path) -> Path:
    """The receipt's blob path inside ``wire_dir`` with the recorded size, without reading it.

    For a stage that hands the receipt on to a reader that hashes the bytes it
    reads (the exporter's ``verify_cached_unit``); a stage that hands over the
    bytes themselves uses :func:`verify_expert_wire_record`.
    """
    path = wire_dir / record["file"]
    if path.is_symlink() or not path.is_file() or path.resolve().parent != wire_dir.resolve():
        raise ExpertProjectionError(f"{name}: priced wire {path} is not in the wire directory")
    if path.stat().st_size != record["blob_bytes"]:
        raise ExpertProjectionError(f"{name}: priced wire {path} does not match its receipt")
    return path


def cached_units_manifest(source: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]],
                          *, schema: str) -> dict:
    """The producer's ``tessera.cached_units.v1`` bundle for ``--cached-expert-units``.

    ``schema`` is the producer's own constant (``tessera.cached_unit.CACHE_SCHEMA``),
    passed in by the caller that imported it; this module does not restate it.
    """
    if not records:
        raise ExpertProjectionError("no priced expert wires to bundle")
    files = [record["file"] for record in records.values()]
    if len(set(files)) != len(files):
        raise ExpertProjectionError("priced expert wires share a filename")
    return {"schema": schema, "source": dict(source),
            "units": {name: dict(record) for name, record in sorted(records.items())}}


# ---------------------------------------------------------------------------
# The allocation side: what the layer config carries from the cost table
# ---------------------------------------------------------------------------
#: Layer-config metadata keys the allocator adds beside the three carried blocks.
STACK_FORMATS_KEY = "tessera_expert_stack_formats"
WIRE_DIR_KEY = "tessera_expert_wire_dir"


def _priced_row(row: Any, *, unit: str, fmt: str) -> tuple[float, int]:
    """One cost row's ``(predicted_dloss, wire_bytes)``, or refuse by name.

    Exact fields or nothing: a row that carries neither a float-able
    ``predicted_dloss`` nor an integer ``wire_bytes`` is not a price this
    allocator may spend, and guessing one would spend the byte budget on a
    number nobody measured.
    """
    if not isinstance(row, Mapping):
        raise ExpertProjectionError(
            f"{unit}@{fmt}: cost row is not an object; the per-unit upgrade "
            "allocator prices exact campaign rows only")
    try:
        dloss = row["predicted_dloss"]
        wire = row["wire_bytes"]
    except KeyError as exc:
        raise ExpertProjectionError(
            f"{unit}@{fmt}: priced row publishes no {exc.args[0]!r}; the "
            "per-unit upgrade allocator prices exact campaign rows only") from exc
    if dloss is None or isinstance(dloss, bool) or not isinstance(dloss, (int, float)):
        raise ExpertProjectionError(
            f"{unit}@{fmt}: priced row predicted_dloss {dloss!r} is not a number")
    if type(wire) is not int:
        raise ExpertProjectionError(
            f"{unit}@{fmt}: priced row wire_bytes {wire!r} is not an integer")
    return float(dloss), int(wire)


def select_priced_unit_upgrades(costs: Mapping[str, Mapping[str, Any]],
                                assignment: Mapping[str, str], *, byte_budget: int,
                                reserve_bytes: int = 0) -> tuple[dict[str, str], dict]:
    """Spend a hard serialized-byte budget at per-unit routed granularity.

    This is the allocator the D36 ruling asked for (PrismaQuant #2319): the
    price surface is per unit -- one expert projection's own campaign rows --
    and the export path used to express picks only whole per layer stack, so
    a 188,331,767 B headroom bought nothing.  The rule is the corrected
    derivation's, at unit granularity: repeatedly buy the best priced
    single-unit upgrade that fits, ordered by ascending ``predicted_dloss``
    delta per added wire byte (ties by unit name, then format spelling --
    a total order, so the picks never depend on row insertion order), and
    stop at the fixed point where no eligible single-unit upgrade fits the
    remaining headroom.  Nothing here invents an objective, a default rung or
    a stop-at-zero cutoff: the rows are the menu, the cap is hard.

    ``costs`` is the campaign table ``{unit: {format: row}}``; a unit counts
    as an upgrade candidate only when its CURRENT format's row and the
    candidate row both carry exact ``predicted_dloss`` and ``wire_bytes`` and
    both spellings parse to ONE Tessera grid (same family/body/plane; only
    the rung differs), with a strictly positive byte delta -- a downgrade or
    a same-bytes move is not an upgrade.  Units priced only at stack
    granularity, units whose rows carry no exact price fields, and BF16 or
    non-Tessera baselines stay grouped exactly where they were.
    ``byte_budget`` is that hard cap, in PRICE-ROW WIRE DELTA bytes --
    NOT a whole-artifact claim; ``reserve_bytes`` is an explicit fixed
    reserve taken off the cap before any pick (metadata/sidecar allowances
    the caller already owes), refused when it exceeds the budget.

    Returns ``(picks, record)``: ``picks`` maps unit -> new format in pick
    order, and ``record`` is the spend leg's own receipt -- budget, reserve,
    spend cap, spent and remaining wire-delta bytes, and the rule -- with
    ``whole_artifact_bytes_claimed`` false, because the whole-artifact
    accounting is the export's own exact owner and this record does not
    speak for it.
    """
    if type(byte_budget) is not int:
        raise ExpertProjectionError(
            f"byte_budget {byte_budget!r} is not an integer wire-delta byte cap")
    if byte_budget < 0:
        raise ExpertProjectionError(f"byte_budget {byte_budget} is negative")
    if type(reserve_bytes) is not int:
        raise ExpertProjectionError(
            f"reserve_bytes {reserve_bytes!r} is not an integer wire-delta reserve")
    if reserve_bytes < 0:
        raise ExpertProjectionError(f"reserve_bytes {reserve_bytes} is negative")
    if reserve_bytes > byte_budget:
        raise ExpertProjectionError(
            f"reserve_bytes {reserve_bytes} exceeds the byte_budget {byte_budget} "
            "it is reserved from")
    spend_cap = byte_budget - reserve_bytes

    from .tessera_formats import parse_tessera_format_name

    def _grid_rung(fmt: str):
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            return None
        spec, rung = parsed
        return (spec.base + ("" if spec.arity == 1 else f"x{spec.arity}"), int(rung))

    current: dict[str, tuple[float, int]] = {}
    for unit, fmt in assignment.items():
        rows = costs.get(unit)
        if not isinstance(rows, Mapping) or fmt not in rows:
            continue  # not priced at unit granularity: stays grouped
        baseline = rows.get(fmt)
        if not isinstance(baseline, Mapping) or "wire_bytes" not in baseline \
                or "predicted_dloss" not in baseline:
            continue  # stack-only / sampled rows: stay grouped, never guessed
        shape = _grid_rung(fmt)
        if shape is None:
            continue  # a non-Tessera baseline has no wire rows to climb
        current[unit] = _priced_row(baseline, unit=unit, fmt=fmt)

    current_fmt = {unit: assignment[unit] for unit in current}
    spent = 0
    picks: dict[str, str] = {}
    eligible_rows = 0
    first_scan = True
    while True:
        # Exactly one marginal per recompute: after every pick the margins
        # move (a unit's second step prices from its new rung, and the
        # remaining headroom shrinks), so a sorted snapshot from before the
        # pick cannot price the next one.  The single best fitting marginal
        # wins each round; the loop stops when no eligible priced upgrade
        # fits -- the fixed point the rule names.
        best_key = None
        best_pick = None
        for unit, (base_dloss, base_wire) in current.items():
            base_grid, base_rung = _grid_rung(current_fmt[unit])
            for fmt, row in costs[unit].items():
                shape = _grid_rung(fmt) if isinstance(fmt, str) else None
                if shape is None or shape[0] != base_grid or shape[1] == base_rung:
                    continue  # another grid, or the rung the unit already runs
                if not isinstance(row, Mapping):
                    raise ExpertProjectionError(
                        f"{unit}@{fmt}: cost row is not an object; the per-unit "
                        "upgrade allocator prices exact campaign rows only")
                if "predicted_dloss" not in row or "wire_bytes" not in row:
                    continue  # an aggregated/sampled cell is not an exact price
                dloss, wire = _priced_row(row, unit=unit, fmt=fmt)
                delta = wire - base_wire
                if delta <= 0:
                    continue  # only upgrades spend the headroom
                if first_scan:
                    eligible_rows += 1
                key = ((dloss - base_dloss) / delta, unit, fmt)
                if delta <= spend_cap - spent and (
                        best_key is None or key < best_key):
                    best_key = key
                    best_pick = (unit, fmt, delta, dloss, wire)
        first_scan = False
        if best_pick is None:
            break
        unit, fmt, delta, dloss, wire = best_pick
        picks[unit] = fmt
        spent += delta
        current[unit] = (dloss, wire)
        current_fmt[unit] = fmt

    record = {
        "schema": "prismaquant.priced_unit_upgrades.v1",
        "currency": "price_row_wire_delta_bytes",
        "byte_budget": byte_budget,
        "reserve_bytes": reserve_bytes,
        "spend_cap_bytes": spend_cap,
        "spent_wire_delta_bytes": spent,
        "remaining_wire_delta_bytes": spend_cap - spent,
        "upgrades": len(picks),
        "eligible_rows": eligible_rows,
        "whole_artifact_bytes_claimed": False,
        "rule": ("ascending predicted_dloss delta per wire-delta byte, ties by "
                 "unit then format; single best eligible upgrade at a time; "
                 "stops when no eligible priced upgrade fits the cap"),
    }
    return picks, record


def expand_stack_decision_assignment(assignment: Mapping[str, Any], population: Any,
                                     *, units: Mapping[str, Any], stack_of: Mapping[str, str],
                                     costs: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """Resolve explicit packed decisions to producer source units in memory.

    The population's complete member map owns the expansion. Return the source
    assignment and each expanded member's packed owner, without mutating either
    the serialized assignment or its metadata. Allocation and export share the
    ownership, coverage and contradictory-assignment checks here.
    """
    projected_assignment = dict(assignment)
    decisions = population.get("stack_decisions", {}) if isinstance(population, Mapping) else {}
    if not isinstance(decisions, Mapping):
        raise ExpertProjectionError("population stack_decisions must be an object")
    member_owner = {}
    for packed, decision in sorted(decisions.items()):
        packed_parameters = population.get("enumerated", {}).get("packed_parameters", {})
        if (packed not in packed_parameters or
                packed not in population.get("enumerated", {}).get("routed_experts", ()) or
                not isinstance(decision, Mapping)):
            raise ExpertProjectionError(f"{packed}: stack decision is not an enumerated packed parameter")
        members = decision.get("members")
        sampled = decision.get("sampled_members")
        if (not isinstance(members, list) or not members or
                any(not isinstance(n, str) or not n for n in members) or
                len(set(members)) != len(members) or
                not isinstance(sampled, list) or not sampled or
                any(not isinstance(n, str) for n in sampled) or
                len(set(sampled)) != len(sampled) or not set(sampled) <= set(members)):
            raise ExpertProjectionError(f"{packed}: stack decision has invalid source members")
        if packed not in assignment:
            raise ExpertProjectionError(f"{packed}: packed stack decision is not in the assignment")
        for name in members:
            if name in member_owner:
                raise ExpertProjectionError(f"{name}: source member belongs to multiple stack decisions")
            member_owner[name] = packed
            if name not in units or stack_of[name] != decision.get("stack"):
                raise ExpertProjectionError(f"{packed}: source member {name} is outside its producer stack")
            if (costs or {}).get(name):
                raise ExpertProjectionError(f"{name}: both source member and packed decision have cost rows")
            fmt = assignment[packed]
            if name in assignment and assignment[name] != fmt:
                raise ExpertProjectionError(f"{name}: source assignment disagrees with packed decision {packed}")
            projected_assignment[name] = fmt
        projected_assignment.pop(packed, None)
    missing = sorted(set(units) - set(projected_assignment))
    if missing:
        raise ExpertProjectionError(
            f"{len(missing)} of {len(units)} projected expert units are not in the "
            f"assignment (first: {missing[0]}); the producer executes every stack whole")
    return projected_assignment, member_owner


def allocation_expert_projection_block(payload: Mapping[str, Any],
                                       assignment: Mapping[str, Any]) -> dict:
    """Carry population/projection only after every selected receipt is present."""
    return _allocation_expert_projection_block(payload, assignment, require_wires=True)


def selection_expert_projection_block(payload: Mapping[str, Any],
                                      assignment: Mapping[str, Any]) -> dict:
    """Validate selection structure without claiming selected bytes exist.

    This is only for a non-exportable materialization request. The public
    allocation gate always requires every selected receipt.
    """
    return _allocation_expert_projection_block(payload, assignment, require_wires=False)


def _allocation_expert_projection_block(payload: Mapping[str, Any],
                                        assignment: Mapping[str, Any], *,
                                        require_wires: bool) -> dict:
    """What an allocation carries about the expert population it selected from.

    A stock table (no population statement, no projection, no wires) adds no
    keys.  A campaign table's ``population`` block travels verbatim, so the
    allocation says which units were priced and which were omitted without a
    reader inferring it from row keys.  When the table also carries the
    producer's projection, every projected unit must be placed by the
    assignment or an explicit population ``stack_decisions`` member map. A
    packed decision expands only for these receipt checks; members do not
    acquire separate prices. Each executed stack whose units share one rung is
    assigned that one format (``tessera_expert_stack_formats``, the
    stack-uniform spelling); a stack whose units carry different rungs of one
    grid is carried per unit under ``tessera_expert_unit_rungs`` and only
    when the installed Tessera runtime publishes the v57 per-unit capability
    (PrismaQuant #2319).  Every Tessera rung selected for a projected
    unit must have a receipt sealed under that unit's projection and that rung
    -- refused by name otherwise.  The receipts of exactly the selected rungs
    travel with the allocation (``tessera_expert_wires``), with the campaign's
    wire directory, so the export lane hands the exporter the priced bytes and
    nothing else.  A stack kept whole at a non-Tessera format needs no receipt;
    the block records the format so the export lane sees the same decision.
    """
    from .tessera_formats import parse_tessera_format_name

    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    if not isinstance(provenance, Mapping):
        return {}
    population = provenance.get(POPULATION_KEY)
    carried = provenance.get(PROJECTION_KEY)
    wires = payload.get(EXPERT_WIRES_KEY)
    if population is None and carried is None and wires is None:
        return {}
    block: dict[str, Any] = {}
    if population is not None:
        if not isinstance(population, Mapping) or population.get("schema") not in {
                POPULATION_SCHEMA, LEGACY_POPULATION_SCHEMA}:
            raise ExpertProjectionError(
                f"cost table population block is not {POPULATION_SCHEMA}; the allocation "
                "cannot say which population was priced")
        block[POPULATION_KEY] = json.loads(DIRECT_ASCII_SPACED_LAX.text(population))
        if population.get("schema") == POPULATION_SCHEMA:
            unpriced = population.get("unpriced")
            if not isinstance(unpriced, Mapping) or set(unpriced) != {"dense", "routed_experts"}:
                raise ExpertProjectionError("cost table population needs explicit unpriced units")
            priced = population.get("priced")
            if not isinstance(priced, Mapping):
                raise ExpertProjectionError("cost table population needs explicit priced units")
            enumerated = population.get("enumerated")
            if not isinstance(enumerated, Mapping):
                raise ExpertProjectionError("cost table population needs explicit enumerated units")
            priced_names = set()
            for kind in ("dense", "routed_experts"):
                names = priced.get(kind)
                if (not isinstance(names, list) or
                        any(not isinstance(name, str) or not name for name in names) or
                        len(set(names)) != len(names) or priced_names.intersection(names)):
                    raise ExpertProjectionError("population priced units must be unique names")
                priced_names.update(names)
            costs = payload.get("costs", {})
            if not isinstance(costs, Mapping):
                raise ExpertProjectionError("campaign cost rows must be an object")
            actual_priced = {name for name, rows in costs.items()
                             if isinstance(rows, Mapping) and rows}
            if priced_names != actual_priced:
                raise ExpertProjectionError(
                    "population priced units disagree with nonempty campaign cost rows")
            retained = []
            for kind in ("dense", "routed_experts"):
                if not isinstance(unpriced[kind], Mapping):
                    raise ExpertProjectionError(f"population unpriced.{kind} must name units")
                targets = enumerated.get(kind)
                if (not isinstance(targets, list) or
                        any(not isinstance(name, str) or not name for name in targets) or
                        len(set(targets)) != len(targets) or
                        set(targets) != set(priced[kind]) | set(unpriced[kind])):
                    raise ExpertProjectionError(
                        f"population enumerated.{kind} must equal its priced and unpriced units")
                for name, reason in sorted(unpriced[kind].items()):
                    if (not isinstance(name, str) or not name or reason not in
                            {"no_admitted_menu", "no_successful_anchor"}):
                        raise ExpertProjectionError("population has an invalid unpriced unit")
                    if name in priced_names or name in retained:
                        raise ExpertProjectionError(
                            f"population priced/unpriced units must be disjoint: {name}")
                    if assignment.get(name) != "BF16":
                        raise ExpertProjectionError(
                            f"unpriced campaign unit {name} ({reason}) must be explicitly "
                            "retained at BF16; absent or quantized assignments have no price")
                    retained.append(name)
            if retained:
                block[POPULATION_KEY]["retained_bf16"] = sorted(retained)
    if carried is None:
        if isinstance(population, Mapping) and population.get("stack_decisions"):
            raise ExpertProjectionError(
                "stack decisions have no producer projection to bind their source members")
        if wires:
            raise ExpertProjectionError(
                "cost table carries priced expert wires but no producer projection; "
                "the wires cannot be bound to any executed unit")
        return block
    _source, units, stack_of = carried_units(carried)
    projected_assignment, _owners = expand_stack_decision_assignment(
        assignment, population, units=units, stack_of=stack_of, costs=payload.get("costs", {}))
    selected = {name: str(projected_assignment[name]) for name in units}
    stack_formats, unit_rungs = require_unit_assignment(selected, stack_of, units)
    wire_dir = provenance.get("wire_dir")
    if not isinstance(wire_dir, str) or not wire_dir:
        raise ExpertProjectionError(
            "cost table names no wire_dir for its priced expert wires")
    if wires is None and not require_wires:
        wires = {}
    if not isinstance(wires, Mapping):
        raise ExpertProjectionError(
            "cost table carries a producer projection but no priced expert wires")
    receipts: dict[str, dict] = {}
    for name, fmt in sorted(selected.items()):
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            continue  # kept whole at a non-Tessera format: no wire to carry
        family, q256 = parsed
        per_unit = wires.get(name)
        record = per_unit.get(fmt) if isinstance(per_unit, Mapping) else None
        if record is None and not require_wires:
            continue
        if record is None:
            raise ExpertProjectionError(
                f"{name}: selected {fmt} has no priced wire receipt in the cost table; "
                "the exporter would encode bytes this allocation did not price")
        receipts[name] = check_expert_wire_receipt(
            record, name=name, unit=units[name], q256=int(q256),
            grid=family.payload_grid().name)
    block[PROJECTION_KEY] = json.loads(DIRECT_ASCII_SPACED_LAX.text(carried))
    block[EXPERT_WIRES_KEY] = receipts
    block[STACK_FORMATS_KEY] = dict(stack_formats)
    if unit_rungs:
        block[UNIT_RUNGS_KEY] = {"schema": UNIT_RUNGS_SCHEMA, "stacks": unit_rungs}
    block[WIRE_DIR_KEY] = wire_dir
    return block


def declared_stacks_from_members(members: Sequence[Any]) -> dict[str, dict[str, tuple[int, int]]]:
    """``{stack: {qname: (rows, cols)}}`` from ``PackedExpertProjection`` members.

    The stack is the packed module's own name (``<block>.experts``), which is
    also the producer's stack name for the unpacked checkpoint.
    """
    declared: dict[str, dict[str, tuple[int, int]]] = {}
    for member in members:
        shape = tuple(int(dim) for dim in member.weight.shape)
        if len(shape) != 2:
            raise ExpertProjectionError(
                f"{member.qname}: declared packed projection is not 2-D: {list(shape)}")
        declared.setdefault(member.module_qname, {})[member.qname] = shape
    return declared


__all__ = [
    "CARRIED_PROJECTION_SCHEMA",
    "EXPERT_WIRES_KEY",
    "ExpertProjectionError",
    "POPULATION_KEY",
    "POPULATION_SCHEMA",
    "PRODUCER_PLAN_TOOL",
    "PROJECTION_KEY",
    "PROJECTION_SCHEMA",
    "SOURCE_IDENTITY_KEYS",
    "SOURCE_LAYOUT_UNPACKED",
    "STACK_FORMATS_KEY",
    "UNIT_IDENTITY_KEYS",
    "WIRE_DIR_KEY",
    "allocation_expert_projection_block",
    "expand_stack_decision_assignment",
    "bind_expert_projection",
    "cached_units_manifest",
    "check_expert_wire_receipt",
    "carried_projection",
    "carried_units",
    "declared_stacks_from_members",
    "producer_plan_tool",
    "request_expert_projection",
    "require_unit_assignment",
    "select_priced_unit_upgrades",
    "source_unit_weight",
    "stack_plan_request",
    "unit_name_of",
    "verify_expert_wire_record",
]
