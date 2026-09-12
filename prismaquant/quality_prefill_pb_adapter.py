"""Turn a frozen quality--prefill phase plan into one PrismaBuild logical request.

DEPLOYMENT DEPENDENCY -- READ BEFORE USING THIS MODULE
=====================================================
This adapter targets PrismaBuild's pre-execution decomposition (PB issue #517).
That work lives on an **unmerged draft pull request, PB #518**, and is **not
deployed to the fleet**: the published runtime generation under
``/mnt/shared/prismabuild-fleet/repo`` carries ``tools/fleet/pbcampaign.py``
without a ``decompose`` entry point and carries no
``src/prismabuild/decomposition.py`` at all.  Two separate things must happen
before a phase emitted here can be submitted for real:

1. PB #518 is merged, and
2. a PrismaBuild runtime generation carrying it is **published** -- which needs
   Rob's explicit word and an idle queue, and is not something an agent does.

A source merge alone does not establish deployed support.  Until both hold, the
only place this adapter is exercised is **in process**, against a checkout of
the #518 branch, by ``tests/test_quality_prefill_pb_adapter_serve_once.py``.

PrismaBuild advertises no capability token for decomposition the way it does for
progress reporting (``prismabuild.core.PROGRESS_TAG == "progress-v1"``), so
:func:`decomposition_support` probes the runtime tree for the two artifacts the
capability is made of.  That absence is itself a finding worth reporting to PB;
it is recorded here rather than worked around.

**There is no fallback.**  :func:`prepare_logical_request` refuses by name when
decomposition support is absent.  It does not submit the phase as one
undecomposed action, and it does not shard the roster itself.  A missing
capability is reported, not routed around -- an app-owned dispatcher is exactly
what #517 exists to remove (specification §11).

What this module is
===================
The experiment declares *semantic* work: for one phase, a list of (serving
unit, numerical family, rate, context) items, each with a measured estimate of
what it costs once its residency is up, plus the measured cost of standing that
residency up.  PrismaBuild owns the cut: it derives one deterministic partition
of that roster, freezes it under the parent's key, and publishes each batch as
an ordinary sealed action.  PrismaQuant never names a host, a worker count or a
shard.

So the adapter is deliberately thin, and its thinness is the point: the phase
plan already carries every field the roster needs, and :func:`emit_logical_request`
is close to an identity over the work items plus the command, demand and batch
policy the phase declares.  Nothing here is defaulted.  A phase that has not
measured its setup cost, or has not resolved its wall limit, is refused before a
parent record can exist (specification §3.1: "phase resource/cost limits must be
resolved from measured work estimates before that phase is submitted").

Reuse and duplication, honestly recorded
========================================
Canonical JSON and its digest come from
:mod:`prismaquant.cost_stage_checkpoint` (``canonical_json``,
``canonical_json_sha256``), which is PrismaQuant's one canonicalizer/hasher.
The strict *reader* below -- closed key sets, duplicate-key refusal, non-finite
refusal -- is local because PrismaQuant has no single home for one: at least
fifteen modules carry a private copy, each raising its own module's error type
(``runtime_provenance``, ``measured_runtime_prices``, ``cluster_campaign_contract``,
``shipcard`` and others).  Consolidating them is a repo-wide change and is not
this work package's scope; this note is the debt line.

``prismaquant/schemas.py`` is the nearest existing home and is reused as far as
it goes: :class:`QualityPrefillAdapterError` subclasses its
:class:`~prismaquant.schemas.SchemaValidationError` (``schemas.py:38``).  Its
validators stop there -- ``_fail`` (``:46``), ``_as_non_negative_int`` (``:58``)
and ``_as_finite_cost_number`` (``:76``) check one field at a time against an
*open* mapping, by design ("older artifacts with extra fields still load",
``schemas.py:1-7``).  This adapter needs the opposite: closed key sets, refusal
of a repeated JSON key, and a canonical byte spelling, because its documents are
hashed into a sealed action key.  Extending ``schemas.py`` with that machinery
would collide head-on with the experiment-manifest schema being built in
parallel, so the strict reader stays here and the two meet at the exception
type.

The identifier grammar is **PrismaBuild's**, not PrismaQuant's: roster ids reach
a sealed action key, so they must satisfy ``prismabuild.core._ID_RE``
(``[a-z0-9][a-z0-9._/-]{0,255}``), which admits ``/`` and 256 characters where
``cluster_campaign_contract._ID_RE`` admits neither.  Validating against the
wrong one would refuse legal ids or, worse, accept ids PB refuses after a plan
was already frozen.

Like :mod:`prismaquant.prismabuild_progress`, this module is written against
PrismaBuild's **wire format** rather than by importing ``prismabuild``; the
fleet runtime is not on PrismaQuant's import path, and agreement with PB's own
validators is proved in the tests instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re

from prismaquant.cost_stage_checkpoint import canonical_json, canonical_json_sha256
from prismaquant.schemas import SchemaValidationError


# --------------------------------------------------------------------------
# Schema names
#
# The three ``prismabuild.*`` names are PrismaBuild's own and are quoted here
# because this module writes and reads those documents.  They are not
# redefined: a mismatch with the deployed runtime is a refusal at PB's
# validator, and the tests assert the two spellings agree.
# --------------------------------------------------------------------------

PHASE_PLAN_SCHEMA = "prismaquant.quality_prefill_phase_plan.v1"
TASK_PAYLOAD_SCHEMA = "prismaquant.quality_prefill_task.v1"

LOGICAL_REQUEST_SCHEMA = "prismabuild.logical_request.v1"
LOGICAL_TASK_ROSTER_SCHEMA = "prismabuild.logical_task_roster.v1"
ROSTER_BATCH_POLICY_SCHEMA = "prismabuild.roster_batch_policy.v1"
BATCH_ENVELOPE_SCHEMA = "prismabuild.task_batch.v1"
CHILD_RESULT_MANIFEST_SCHEMA = "prismabuild.child_result_manifest.v1"

#: PrismaBuild's reserved whole-argument placeholder.  It is substituted with
#: the CAS path of the child's own batch envelope before the action is sealed;
#: it is never shell-expanded and never interpolated inside a larger argument,
#: so a resolved path carrying a space, a quote or a ``$`` still reaches the
#: child as exactly one ``argv`` element.
TASK_BATCH_PLACEHOLDER = "{pb.task_batch}"

#: The two measurement currencies the specification names (§3.2).  Closed: a
#: scalar screen and a validated measured joint price are different claims, and
#: a third spelling here would let one be reported as the other.
TASK_CURRENCIES = frozenset({
    "output_mse_under_route_activation_contract",
    "joint_aura_predicted_dloss",
})

#: What a phase's children run, and where.  ``pbrun`` attests ``argv[0]`` off
#: the sealed action, so the interpreter is part of the plan rather than
#: ``sys.executable`` read at emit time -- an emitter that read the ambient
#: interpreter would not be deterministic across boxes.
_COMMAND_KEYS = frozenset({"interpreter", "module"})
_PHASE_PLAN_KEYS = frozenset({
    "schema", "phase_id", "experiment_plan_sha256", "command", "cwd", "env",
    "demand", "gpu_memory_gb", "data_manifest", "residencies", "batch_limits",
    "work_items",
})
_RESIDENCY_KEYS = frozenset({"key", "setup_seconds", "setup_evidence"})
_BATCH_LIMIT_KEYS = frozenset({"max_setup_fraction", "max_estimated_wall_seconds"})
_WORK_ITEM_KEYS = frozenset({
    "task_id", "output_id", "residency_key", "estimated_seconds",
    "estimate_evidence", "payload",
})
_TASK_PAYLOAD_KEYS = frozenset({
    "schema", "serving_unit", "members", "family", "rate", "context_id",
    "currency",
})

#: ``prismabuild.core._ID_RE`` verbatim.  See the module docstring.
_PB_ID_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,255}\Z")
#: ``prismabuild.core._ENV_RE`` verbatim.
_PB_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
#: A cost estimate's evidence names owned bytes, in the shape PrismaBuild's own
#: decomposition tests use (``cas:sha256:<digest>``).  Specification §3.2: path
#: strings are not identities; an evidence field with no digest in it is prose.
_EVIDENCE_RE = re.compile(r"[a-z][a-z0-9+.-]*:sha256:[0-9a-f]{64}\Z")
#: A dotted Python module path, for ``interpreter -m module``.
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


class QualityPrefillAdapterError(SchemaValidationError):
    """A phase plan, batch envelope or child result is structurally invalid.

    A subclass of :class:`prismaquant.schemas.SchemaValidationError` because
    that is PrismaQuant's existing name for "a handoff artifact is structurally
    invalid", and a caller that already catches it should catch these too.
    """


class DecompositionUnavailable(RuntimeError):
    """The pinned PrismaBuild runtime cannot decompose a logical request.

    Its own type because the caller's response differs in kind.  A schema error
    is a plan the author can fix; this one says the fleet does not yet carry the
    capability, and the only answers are to wait for PB #518 to merge and a
    runtime generation to be published, or to run the phase in process against a
    checkout of that branch.  Neither is something this module can do, and
    neither is "submit it undecomposed".
    """


# --------------------------------------------------------------------------
# Strict readers.  Every one refuses unknown fields: a field PrismaQuant does
# not understand is a field the plan's author believes is binding and that
# nothing downstream would read.
# --------------------------------------------------------------------------


def _fail(message: str) -> None:
    raise QualityPrefillAdapterError(message)


def _exact_mapping(
    value: object, *, keys: frozenset[str], where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        _fail(f"{where} keys must be strings")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        _fail(f"{where} fields differ: missing={missing}, extra={extra}")
    return value  # type: ignore[return-value]


def _text(
    value: object, *, where: str, pattern: re.Pattern[str] | None = None
) -> str:
    if type(value) is not str or not value:
        _fail(f"{where} must be a non-empty string")
    text = str(value)
    if text != text.strip() or any(ord(char) < 32 for char in text):
        _fail(f"{where} carries whitespace padding or control characters")
    if pattern is not None and pattern.fullmatch(text) is None:
        _fail(f"{where} is not a legal {pattern.pattern!r} value: {text!r}")
    return text


def _identifier(value: object, *, where: str) -> str:
    return _text(value, where=where, pattern=_PB_ID_RE)


def _sha256(value: object, *, where: str) -> str:
    return _text(value, where=where, pattern=_SHA256_RE)


def _evidence(value: object, *, where: str) -> str:
    return _text(value, where=where, pattern=_EVIDENCE_RE)


def _integer(value: object, *, where: str, minimum: int) -> int:
    # ``type(value) is int`` rather than ``isinstance``: a bool is an int in
    # Python and the specification refuses booleans in integer fields (§3.1).
    if type(value) is not int or value < minimum:
        _fail(f"{where} must be an integer >= {minimum}")
    return int(value)


def _finite(value: object, *, where: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        _fail(f"{where} must be a number")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        _fail(f"{where} must be finite")
    return number


def _positive(value: object, *, where: str) -> float:
    number = _finite(value, where=where)
    if not number > 0.0:
        _fail(f"{where} must be greater than zero")
    return number


def _nonnegative(value: object, *, where: str) -> float:
    number = _finite(value, where=where)
    if number < 0.0:
        _fail(f"{where} must not be negative")
    return number


def _sequence(value: object, *, where: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(f"{where} must be an array")
    if not value:  # type: ignore[arg-type]
        _fail(f"{where} must not be empty")
    return value  # type: ignore[return-value]


def load_strict_json(raw: bytes | str, *, where: str) -> object:
    """Parse JSON that refuses duplicate keys and non-finite numbers.

    Both refusals exist because the document is an identity.  A repeated key is
    two different statements under one name and ``json`` would silently keep the
    last; a ``NaN`` or ``Infinity`` is not canonical JSON and cannot be hashed
    into a key that another reader would reproduce.
    """

    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        seen: dict[str, object] = {}
        for key, value in items:
            if key in seen:
                _fail(f"{where} repeats the JSON key {key!r}")
            seen[key] = value
        return seen

    def constant(text: str) -> object:
        _fail(f"{where} carries the non-finite number {text}")
        raise AssertionError("unreachable")  # pragma: no cover

    try:
        return json.loads(
            raw.decode("utf-8") if isinstance(raw, bytes) else raw,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except QualityPrefillAdapterError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"{where} is not strict UTF-8 JSON: {exc}")
        raise AssertionError("unreachable")  # pragma: no cover


def document_bytes(value: object) -> bytes:
    """The one on-disk spelling of a document this adapter writes.

    Compact sorted canonical JSON with a trailing newline -- the same shape
    ``prismabuild.decomposition.document_bytes`` produces, so a document written
    here and a document written there are the same bytes and therefore the same
    CAS digest.  A trailing-newline difference reads as tampering, not as a
    formatting choice.
    """

    try:
        encoded = json.dumps(
            canonical_json(value, where="quality-prefill document"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"document is not canonical JSON data: {exc}")
        raise AssertionError("unreachable")  # pragma: no cover
    return encoded.encode("utf-8") + b"\n"


def canonical_sha256(value: object) -> str:
    """The digest of the value itself, without its file's trailing newline.

    The same digest ``prismabuild.core.canonical_sha256`` computes, so a value
    hashed here and the same value hashed by the fleet agree.  This is the one
    to hash an *identity* with; :func:`document_file_sha256` is the one a
    content-addressed store will give the file.  PrismaBuild keeps the same two
    and the distinction is the trailing newline, which is exactly the kind of
    difference that reads as a tampered blob rather than a formatting choice.
    """

    try:
        return canonical_json_sha256(value, where="quality-prefill document")
    except (TypeError, ValueError) as exc:
        _fail(f"document is not canonical JSON data: {exc}")
        raise AssertionError("unreachable")  # pragma: no cover


def document_file_sha256(value: object) -> str:
    """The digest the CAS will give this document's bytes, before it is written.

    The file-bytes digest, matching ``prismabuild.decomposition.document_sha256``:
    a blob's name in the store is the hash of what is on disk, newline included.
    """

    return hashlib.sha256(document_bytes(value)).hexdigest()


# --------------------------------------------------------------------------
# The frozen phase plan
# --------------------------------------------------------------------------


def validate_task_payload(value: object, *, where: str) -> dict[str, object]:
    """One work item's semantic content, carried opaquely by PrismaBuild.

    PrismaBuild stores the payload and hands it back to the child unchanged; it
    never looks inside.  So this is the only place the child's instructions are
    checked, and it is checked before the plan is frozen rather than on the
    worker, where a refusal has already cost a claim.

    ``members`` is ordered and unique because §2.2 requires a bidirectional
    member-to-serving-unit map in which every mutable member occurs exactly
    once; two spellings of one member in one unit would break that count.
    """

    payload = _exact_mapping(value, keys=_TASK_PAYLOAD_KEYS, where=where)
    if payload["schema"] != TASK_PAYLOAD_SCHEMA:
        _fail(f"{where}.schema must be {TASK_PAYLOAD_SCHEMA!r}")
    members: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(payload["members"],
                                          where=f"{where}.members")):
        member = _identifier(raw, where=f"{where}.members[{index}]")
        if member in seen:
            _fail(f"{where}.members[{index}] repeats {member!r}")
        seen.add(member)
        members.append(member)
    currency = _text(payload["currency"], where=f"{where}.currency")
    if currency not in TASK_CURRENCIES:
        _fail(
            f"{where}.currency must be one of {sorted(TASK_CURRENCIES)}; a "
            f"scalar screen and a measured joint price are different claims"
        )
    return {
        "schema": TASK_PAYLOAD_SCHEMA,
        "context_id": _identifier(payload["context_id"],
                                  where=f"{where}.context_id"),
        "currency": currency,
        "family": _identifier(payload["family"], where=f"{where}.family"),
        "members": members,
        # An integer: a rate is a legal code from the Tessera menu, and a float
        # rate is a rate nobody can serve.
        "rate": _integer(payload["rate"], where=f"{where}.rate", minimum=1),
        "serving_unit": _identifier(payload["serving_unit"],
                                    where=f"{where}.serving_unit"),
    }


def validate_phase_plan(value: object) -> dict[str, object]:
    """Canonicalize one frozen phase plan.

    This is the ``execution.phases[i]`` projection of the run manifest
    (``prismaquant.quality_prefill_experiment.v1``, specification §3.1), which a
    separate work package owns.  Every field a PrismaBuild roster entry needs is
    already here, one for one, so the emitter below is close to an identity --
    which is what keeps the two schemas from drifting apart.

    Nothing is defaulted.  Both batch limits, every residency's measured setup
    cost and every work item's measured estimate are required, because a default
    invented here would be PrismaQuant deciding how much of a GPU hour may go to
    setup -- the judgment the phase is supposed to have measured.
    """

    plan = _exact_mapping(value, keys=_PHASE_PLAN_KEYS, where="phase plan")
    if plan["schema"] != PHASE_PLAN_SCHEMA:
        _fail(f"phase plan schema must be {PHASE_PLAN_SCHEMA!r}")

    command = _exact_mapping(plan["command"], keys=_COMMAND_KEYS,
                             where="phase plan command")
    interpreter = _text(command["interpreter"], where="phase plan command interpreter")
    if not interpreter.startswith("/"):
        _fail("phase plan command interpreter must be an absolute path")
    module = _text(command["module"], where="phase plan command module",
                   pattern=_MODULE_RE)

    cwd = _text(plan["cwd"], where="phase plan cwd")
    if not cwd.startswith("/"):
        _fail("phase plan cwd must be an absolute path")

    raw_env = plan["env"]
    if not isinstance(raw_env, Mapping):
        _fail("phase plan env must be an object")
    env: dict[str, str] = {}
    for name in sorted(raw_env):  # type: ignore[union-attr]
        _text(name, where="phase plan env key", pattern=_PB_ENV_RE)
        env[name] = _text(raw_env[name], where=f"phase plan env[{name!r}]")

    raw_demand = plan["demand"]
    if not isinstance(raw_demand, Mapping) or not raw_demand:
        _fail("phase plan demand must be a non-empty object")
    demand: dict[str, int] = {}
    for name in sorted(raw_demand):  # type: ignore[union-attr]
        _text(name, where="phase plan demand key", pattern=_PB_ID_RE)
        demand[name] = _integer(raw_demand[name],
                                where=f"phase plan demand[{name!r}]", minimum=0)

    gpu_memory_gb = plan["gpu_memory_gb"]
    if gpu_memory_gb is not None:
        gpu_memory_gb = _positive(gpu_memory_gb, where="phase plan gpu_memory_gb")
    data_manifest = plan["data_manifest"]
    if data_manifest is not None:
        data_manifest = _text(data_manifest, where="phase plan data_manifest")

    residencies: list[dict[str, object]] = []
    priced: set[str] = set()
    for index, raw in enumerate(_sequence(plan["residencies"],
                                          where="phase plan residencies")):
        where = f"phase plan residencies[{index}]"
        entry = _exact_mapping(raw, keys=_RESIDENCY_KEYS, where=where)
        key = _identifier(entry["key"], where=f"{where}.key")
        if key in priced:
            _fail(f"{where}.key repeats an earlier residency: {key!r}")
        priced.add(key)
        residencies.append({
            "key": key,
            "setup_seconds": _nonnegative(entry["setup_seconds"],
                                          where=f"{where}.setup_seconds"),
            "setup_evidence": _evidence(entry["setup_evidence"],
                                        where=f"{where}.setup_evidence"),
        })

    limits = _exact_mapping(plan["batch_limits"], keys=_BATCH_LIMIT_KEYS,
                            where="phase plan batch_limits")
    fraction = _finite(limits["max_setup_fraction"],
                       where="phase plan batch_limits max_setup_fraction")
    if not 0.0 < fraction <= 1.0:
        _fail("phase plan batch_limits max_setup_fraction must be in (0, 1]")

    work_items: list[dict[str, object]] = []
    seen_tasks: set[str] = set()
    seen_outputs: set[str] = set()
    for index, raw in enumerate(_sequence(plan["work_items"],
                                          where="phase plan work_items")):
        where = f"phase plan work_items[{index}]"
        item = _exact_mapping(raw, keys=_WORK_ITEM_KEYS, where=where)
        task_id = _identifier(item["task_id"], where=f"{where}.task_id")
        output_id = _identifier(item["output_id"], where=f"{where}.output_id")
        if task_id in seen_tasks:
            _fail(f"{where}.task_id repeats an earlier task: {task_id!r}")
        if output_id in seen_outputs:
            _fail(f"{where}.output_id repeats an earlier output: {output_id!r}")
        seen_tasks.add(task_id)
        seen_outputs.add(output_id)
        residency_key = _identifier(item["residency_key"],
                                    where=f"{where}.residency_key")
        if residency_key not in priced:
            _fail(
                f"{where}.residency_key {residency_key!r} is not priced by this "
                f"phase; it prices {sorted(priced)}.  An unpriced residency has "
                f"no setup cost, so no batch of it has a reason to exist"
            )
        work_items.append({
            "task_id": task_id,
            "output_id": output_id,
            "residency_key": residency_key,
            "estimated_seconds": _positive(item["estimated_seconds"],
                                           where=f"{where}.estimated_seconds"),
            "estimate_evidence": _evidence(item["estimate_evidence"],
                                           where=f"{where}.estimate_evidence"),
            "payload": validate_task_payload(item["payload"],
                                             where=f"{where}.payload"),
        })

    return {
        "schema": PHASE_PLAN_SCHEMA,
        "batch_limits": {
            "max_estimated_wall_seconds": _positive(
                limits["max_estimated_wall_seconds"],
                where="phase plan batch_limits max_estimated_wall_seconds",
            ),
            "max_setup_fraction": fraction,
        },
        "command": {"interpreter": interpreter, "module": module},
        "cwd": cwd,
        "data_manifest": data_manifest,
        "demand": demand,
        "env": env,
        "experiment_plan_sha256": _sha256(plan["experiment_plan_sha256"],
                                          where="phase plan experiment_plan_sha256"),
        "gpu_memory_gb": gpu_memory_gb,
        "phase_id": _identifier(plan["phase_id"], where="phase plan phase_id"),
        "residencies": residencies,
        "work_items": work_items,
    }


def phase_plan_sha256(phase_plan: Mapping[str, object]) -> str:
    """The digest of the validated phase plan, for an evidence record to cite."""

    return canonical_sha256(validate_phase_plan(phase_plan))


# --------------------------------------------------------------------------
# The emitter
# --------------------------------------------------------------------------


def child_argv(interpreter: str, module: str) -> list[str]:
    """The command every child of this phase runs, with the batch slot reserved.

    ``-m`` rather than a script path: a child runs in its own private checkout
    of the sealed tree, and a module resolved off ``sys.path`` is the same code
    the closure attested, while a path would have to be rewritten per checkout.

    The placeholder is the last argument and appears exactly once.  PrismaBuild
    replaces that one whole element with the CAS path of this child's batch
    envelope; it is substitution, so a path containing a space, a quote or a
    ``$`` still arrives as one argument.
    """

    return [interpreter, "-m", module, TASK_BATCH_PLACEHOLDER]


def emit_logical_request(phase_plan: Mapping[str, object]) -> dict[str, object]:
    """One frozen phase plan as ``prismabuild.logical_request.v1``.

    Pure and deterministic: every value comes from the plan, and nothing is read
    from the ambient process -- not ``sys.executable``, not ``os.getcwd()``, not
    the hostname, not the clock.  The same frozen plan therefore emits the same
    bytes on any box, which is what lets a resumed campaign land on the same
    parent key and read back the same frozen cut instead of asking the batcher
    for a second opinion.

    The capability probe is deliberately **not** here.  Emitting is what a
    ``plan`` or ``freeze`` command does while the fleet may be anything at all;
    refusing on a missing runtime belongs at the submission boundary, which is
    :func:`prepare_logical_request`.
    """

    plan = validate_phase_plan(phase_plan)
    command = plan["command"]
    assert isinstance(command, Mapping)
    work_items = plan["work_items"]
    assert isinstance(work_items, Sequence)
    residencies = plan["residencies"]
    assert isinstance(residencies, Sequence)
    limits = plan["batch_limits"]
    assert isinstance(limits, Mapping)

    return {
        "schema": LOGICAL_REQUEST_SCHEMA,
        "common": {
            "argv": child_argv(str(command["interpreter"]), str(command["module"])),
            "cwd": plan["cwd"],
            "data_manifest": plan["data_manifest"],
            "demand": dict(plan["demand"]),        # type: ignore[arg-type]
            "env": dict(plan["env"]),              # type: ignore[arg-type]
            "gpu_memory_gb": plan["gpu_memory_gb"],
        },
        "roster": {
            "schema": LOGICAL_TASK_ROSTER_SCHEMA,
            # Roster order is meaningful: PrismaBuild only ever cuts a
            # contiguous same-residency run, so the order the phase declares is
            # its statement about which items may share one resident process.
            "tasks": [
                {
                    "id": item["task_id"],
                    "estimate_evidence": item["estimate_evidence"],
                    "estimated_seconds": item["estimated_seconds"],
                    "output_id": item["output_id"],
                    "payload": item["payload"],
                    "residency_key": item["residency_key"],
                }
                for item in work_items
            ],
        },
        "batch_policy": {
            "schema": ROSTER_BATCH_POLICY_SCHEMA,
            "max_estimated_wall_seconds": limits["max_estimated_wall_seconds"],
            "max_setup_fraction": limits["max_setup_fraction"],
            "residencies": [dict(entry) for entry in residencies],  # type: ignore[arg-type]
        },
    }


def logical_request_bytes(phase_plan: Mapping[str, object]) -> bytes:
    """The emitted request as the exact bytes a request file carries."""

    return document_bytes(emit_logical_request(phase_plan))


def write_logical_request(phase_plan: Mapping[str, object], path: str | Path) -> str:
    """Write the request ``pbcampaign`` reads, and return its digest."""

    request = emit_logical_request(phase_plan)
    Path(path).write_bytes(document_bytes(request))
    return document_file_sha256(request)


# --------------------------------------------------------------------------
# The capability probe, and the refusal
# --------------------------------------------------------------------------

#: The published PrismaBuild runtime the fleet actually executes.  Probed, never
#: assumed: this module records what it found rather than what a merged pull
#: request implies.
DEPLOYED_RUNTIME_ROOT = Path("/mnt/shared/prismabuild-fleet/repo")

_DECOMPOSITION_MODULE = Path("src/prismabuild/decomposition.py")
_CAMPAIGN_TOOL = Path("tools/fleet/pbcampaign.py")
_CAMPAIGN_ENTRY = "def decompose("


def decomposition_support(
    runtime_root: str | Path = DEPLOYED_RUNTIME_ROOT,
) -> dict[str, object]:
    """What one PrismaBuild runtime tree can be shown to carry.

    A filesystem probe rather than an import, for two reasons.  Importing a
    foreign runtime into the producer's process is what ``AGENTS.md`` forbids
    for a serving runtime and is no better here; and a runtime tree that is not
    on ``sys.path`` cannot be imported at all, which is the ordinary case.

    PrismaBuild publishes no capability token for decomposition -- ``core.py``
    carries ``PROGRESS_TAG = "progress-v1"`` and nothing equivalent for #517 --
    so support is the two artifacts it is made of: the module that owns the
    plan, and the campaign entry point that publishes children from one.  When
    #518 lands a token, this probe should read the token instead and this
    comment is the reason it does not yet.
    """

    root = Path(runtime_root)
    module = (root / _DECOMPOSITION_MODULE).is_file()
    tool = root / _CAMPAIGN_TOOL
    entry = False
    if tool.is_file():
        try:
            entry = _CAMPAIGN_ENTRY in tool.read_text(encoding="utf-8", errors="replace")
        except OSError:
            entry = False
    return {
        "runtime_root": str(root),
        "decomposition_module": module,
        "campaign_decompose_entry": entry,
        "supported": bool(module and entry),
    }


def require_decomposition_support(
    runtime_root: str | Path = DEPLOYED_RUNTIME_ROOT,
) -> dict[str, object]:
    """Refuse, by name, when the runtime cannot decompose a logical request.

    There is no fallback below this line and there must not be one.  Submitting
    the phase as a single undecomposed action would hand the fleet one long
    opaque job in place of independently retryable quanta, and cutting the
    roster here would make PrismaQuant a second scheduler -- both of which are
    what PB #517 exists to remove (specification §11: "Unsupported PB
    subdivision is a named capability gap; implementation can proceed, but no
    bespoke dispatcher fills it").
    """

    support = decomposition_support(runtime_root)
    if support["supported"]:
        return support
    missing = [
        name
        for name, present in (
            (str(_DECOMPOSITION_MODULE), support["decomposition_module"]),
            (f"{_CAMPAIGN_TOOL} :: {_CAMPAIGN_ENTRY}",
             support["campaign_decompose_entry"]),
        )
        if not present
    ]
    raise DecompositionUnavailable(
        f"PrismaBuild pre-execution decomposition (PB #517) is not available in "
        f"the runtime at {support['runtime_root']}: missing {missing}. "
        f"It needs PB pull request #518 merged AND a PrismaBuild runtime "
        f"generation carrying it published to the fleet; a merge alone is not "
        f"deployed support, and publishing a generation needs Rob's explicit "
        f"word and an idle queue. Until then this phase is exercised only in "
        f"process against a checkout of the #518 branch. This adapter does not "
        f"fall back to submitting the roster as one undecomposed action and "
        f"does not shard it itself."
    )


def prepare_logical_request(
    phase_plan: Mapping[str, object],
    *,
    runtime_root: str | Path = DEPLOYED_RUNTIME_ROOT,
) -> dict[str, object]:
    """The submission boundary: probe the runtime, then emit.

    The probe runs first so that a phase whose plan is fine but whose fleet is
    not hears the capability gap rather than a schema error, and so that no
    partially-published parent record can exist on a runtime that cannot read
    it back.
    """

    require_decomposition_support(runtime_root)
    return emit_logical_request(phase_plan)


# --------------------------------------------------------------------------
# The producer contract: what one child reads, and what it must write
# --------------------------------------------------------------------------

_ENVELOPE_KEYS = frozenset({
    "schema", "parent_key", "plan_key", "roster_sha256", "batch_policy_sha256",
    "child_ordinal", "result_manifest_path", "tasks",
})
_ENVELOPE_TASK_KEYS = frozenset({
    "id", "payload", "residency_key", "estimated_seconds", "estimate_evidence",
    "output_id",
})
_MANIFEST_KEYS = frozenset({
    "schema", "parent_key", "plan_key", "child_ordinal", "results",
})


def validate_batch_envelope(value: object) -> dict[str, object]:
    """The immutable file one child reads to learn exactly what it measures.

    PrismaBuild writes this document and seals its digest into the child's
    action key, so the child does not choose its own work: its batch is a fact
    about its key.  It is re-validated here rather than trusted because the
    child is where the payload is finally acted on, and a payload PrismaBuild
    stored opaquely has never been checked against PrismaQuant's own schema
    until now.

    ``result_manifest_path`` is told rather than derived: the action that
    declared it was sealed before this child ran, and a child that recomputed
    the naming rule could disagree with the seal.
    """

    envelope = _exact_mapping(value, keys=_ENVELOPE_KEYS, where="batch envelope")
    if envelope["schema"] != BATCH_ENVELOPE_SCHEMA:
        _fail(f"batch envelope schema must be {BATCH_ENVELOPE_SCHEMA!r}")
    path = _text(envelope["result_manifest_path"],
                 where="batch envelope result_manifest_path")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        _fail("batch envelope result_manifest_path must be a normalized "
              "relative path inside the child's own working tree")
    tasks: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(envelope["tasks"],
                                          where="batch envelope tasks")):
        where = f"batch envelope tasks[{index}]"
        task = _exact_mapping(raw, keys=_ENVELOPE_TASK_KEYS, where=where)
        task_id = _identifier(task["id"], where=f"{where}.id")
        if task_id in seen:
            _fail(f"{where}.id repeats an earlier task in this batch: {task_id!r}")
        seen.add(task_id)
        tasks.append({
            "id": task_id,
            "estimate_evidence": _evidence(task["estimate_evidence"],
                                           where=f"{where}.estimate_evidence"),
            "estimated_seconds": _positive(task["estimated_seconds"],
                                           where=f"{where}.estimated_seconds"),
            "output_id": _identifier(task["output_id"], where=f"{where}.output_id"),
            "payload": validate_task_payload(task["payload"],
                                             where=f"{where}.payload"),
            "residency_key": _identifier(task["residency_key"],
                                         where=f"{where}.residency_key"),
        })
    return {
        "schema": BATCH_ENVELOPE_SCHEMA,
        "batch_policy_sha256": _sha256(envelope["batch_policy_sha256"],
                                       where="batch envelope batch_policy_sha256"),
        "child_ordinal": _integer(envelope["child_ordinal"],
                                  where="batch envelope child_ordinal", minimum=0),
        "parent_key": _sha256(envelope["parent_key"],
                              where="batch envelope parent_key"),
        "plan_key": _sha256(envelope["plan_key"], where="batch envelope plan_key"),
        "result_manifest_path": path,
        "roster_sha256": _sha256(envelope["roster_sha256"],
                                 where="batch envelope roster_sha256"),
        "tasks": tasks,
    }


def read_batch_envelope(path: str | Path) -> dict[str, object]:
    """Read the envelope PrismaBuild substituted into ``{pb.task_batch}``."""

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        _fail(f"cannot read the batch envelope at {path}: {exc}")
        raise AssertionError("unreachable")  # pragma: no cover
    return validate_batch_envelope(
        load_strict_json(raw, where=f"batch envelope {path}")
    )


def build_child_result_manifest(
    envelope: Mapping[str, object],
    results: Mapping[str, object],
) -> dict[str, object]:
    """What this child says it measured, as ``prismabuild.child_result_manifest.v1``.

    Fail-closed in both directions, because the group receipt's whole claim is
    that the children between them answered the roster exactly once.  A result
    for a task outside this batch is another child's answer and would make one
    task answered twice; a task in the batch with no result is a batch that
    passed while answering nothing, which is the failure PrismaBuild's exact
    cover exists to catch.  Catching both here means the child fails where it
    ran, with its own diagnosis, instead of at a merge that can only say the
    cover is open.

    ``output_id`` is taken from the envelope, never from the caller: the roster
    named it, and a producer that answered under a name of its own choosing has
    produced something and nothing that was wanted.

    ``value_sha256`` is the canonical digest of the result record itself, so the
    merged digest a group receipt publishes is a statement about the values and
    not about where a worker happened to write them.
    """

    checked = validate_batch_envelope(envelope)
    tasks = checked["tasks"]
    assert isinstance(tasks, Sequence)
    if not isinstance(results, Mapping):
        _fail("child results must be a mapping of task id to result record")
    membership = {str(task["id"]): task for task in tasks}
    foreign = sorted(set(results) - set(membership))
    if foreign:
        _fail(
            f"child {checked['child_ordinal']} was asked to report "
            f"{len(foreign)} task(s) outside its own batch, beginning with "
            f"{foreign[0]!r}"
        )
    missing = [task_id for task_id in membership if task_id not in results]
    if missing:
        _fail(
            f"child {checked['child_ordinal']} has no result for "
            f"{len(missing)} of its {len(membership)} tasks, beginning with "
            f"{missing[0]!r}"
        )
    return {
        "schema": CHILD_RESULT_MANIFEST_SCHEMA,
        "child_ordinal": checked["child_ordinal"],
        "parent_key": checked["parent_key"],
        "plan_key": checked["plan_key"],
        # Batch order, so two runs of one child publish identical bytes.
        "results": [
            {
                "output_id": membership[str(task["id"])]["output_id"],
                "task_id": str(task["id"]),
                "value_sha256": canonical_sha256(results[str(task["id"])]),
            }
            for task in tasks
        ],
    }


_RESULT_KEYS = frozenset({"task_id", "output_id", "value_sha256"})


def validate_child_result_manifest(value: object) -> dict[str, object]:
    """Read back one child's declared result.

    The collector's half of the producer contract, and the same closed shape
    :func:`build_child_result_manifest` writes.  PrismaBuild validates this
    document too, on its way into the exact cover; this reader exists so a
    PrismaQuant-side collector can refuse a malformed manifest without importing
    the fleet runtime, and so a test can show the two readers agree.
    """

    manifest = _exact_mapping(value, keys=_MANIFEST_KEYS,
                              where="child result manifest")
    if manifest["schema"] != CHILD_RESULT_MANIFEST_SCHEMA:
        _fail(f"child result manifest schema must be "
              f"{CHILD_RESULT_MANIFEST_SCHEMA!r}")
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(manifest["results"],
                                          where="child result manifest results")):
        where = f"child result manifest results[{index}]"
        entry = _exact_mapping(raw, keys=_RESULT_KEYS, where=where)
        task_id = _identifier(entry["task_id"], where=f"{where}.task_id")
        if task_id in seen:
            _fail(f"{where}.task_id is answered twice in one manifest: {task_id!r}")
        seen.add(task_id)
        results.append({
            "output_id": _identifier(entry["output_id"], where=f"{where}.output_id"),
            "task_id": task_id,
            "value_sha256": _sha256(entry["value_sha256"],
                                    where=f"{where}.value_sha256"),
        })
    return {
        "schema": CHILD_RESULT_MANIFEST_SCHEMA,
        "child_ordinal": _integer(manifest["child_ordinal"],
                                  where="child result manifest child_ordinal",
                                  minimum=0),
        "parent_key": _sha256(manifest["parent_key"],
                              where="child result manifest parent_key"),
        "plan_key": _sha256(manifest["plan_key"],
                            where="child result manifest plan_key"),
        "results": results,
    }


def write_child_result_manifest(
    envelope: Mapping[str, object],
    results: Mapping[str, object],
    *,
    root: str | Path = ".",
) -> Path:
    """Publish this child's declared result where its sealed action says.

    The manifest is the child's *declared result*, not its log: a log says a
    process ran, and what the group receipt needs is which roster tasks now have
    an answer.  PrismaBuild reads it back through the receipt that fixed its
    digest, so the bytes written here are the bytes the cover is proved on.
    """

    checked = validate_batch_envelope(envelope)
    manifest = build_child_result_manifest(checked, results)
    path = Path(root) / str(checked["result_manifest_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document_bytes(manifest))
    return path
