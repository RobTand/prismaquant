"""Selected served-wire recipes and output-MSE cell measurements.

The caller owns resident weights, calibration inputs and publication lifetime.
This owner admits recipes before encoding and publishes decoded cache entries
and wires before returning synchronous measurements. Asynchronous callers must
wait for file jobs before committing receipts or checkpoint rows.
"""
from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from . import tessera_campaign_wire as campaign_wire
from . import tessera_hessian as th
from .nvfp4_activation_contract import ActivationScaleContractError as _ActivationScaleContractError
from .tessera_campaign_table import CampaignAnchor
from .tessera_publication import PublicationJob

if TYPE_CHECKING:
    from tessera.export import WireRecipe
    from .format_registry import FormatSpec
    from .tessera_formats import TesseraFamily

__all__ = [
    'CellRecipe', 'compatible_batches', 'encoder_kwargs_memo',
    'executes_static_activation_contract', 'measure_cell', 'measure_cells',
    'require_resumable_cell', 'resume_cell_activation', 'served_quantizer_record',
    'stale_wire_reason',
]

FILES_JOB = "files"


def encoder_kwargs_memo(source, weights, device, *, max_entries=None,
                        resource_check=None, factor_scratch_bytes=0):
    """The campaign's existing plane-keyed memo, with an explicit owner bound."""
    if max_entries is not None and (type(max_entries) is not int or max_entries < 1):
        raise ValueError('encoder memo capacity must be positive or unbounded')

    @functools.lru_cache(maxsize=max_entries)
    def for_unit(name, scale_plane):
        if resource_check is not None:
            resource_check('before_selected_encoder_factors:'+name,
                           reserve_bytes=factor_scratch_bytes)
        kwargs = th.encoder_kwargs(source, name, int(weights[name].shape[1]),
                                   device, scale_plane=scale_plane)
        if resource_check is not None:
            resource_check('after_selected_encoder_factors:'+name)
        return kwargs

    return for_unit


def measure_cell(
    *, qname: str, weight, activations, format_name: str, cache, wire_dir: Path,
    activation_kwargs_for=None, hessian_required: bool = True,
    static_input_scale: "float | None" = None, publisher=None,
    structure: "str | None" = None, profile=None,
) -> CampaignAnchor:
    """Render one rung, price it as served, and store the wire beside it.

    ``structure`` is the unit's serving structure (``dense``/``routed_moe``)
    when the run declares one; the wire is then the one that structure is
    served on (``tessera_served_wire_recipe``, #1502), not the research table.

    ``static_input_scale`` is the unit's calibrated NVFP4
    ``input_global_scale`` (fused-sibling unified), required whenever the
    rung's route executes the static UE4M3 activation contract and ignored on
    every other route.  The refusal for a missing one runs BEFORE the encode,
    because the encode is the expensive half and the refusal is about the
    whole run.

    ``activation_kwargs_for`` is a callable ``(qname, scale_plane) -> encoder
    kwargs`` -- the block-LDL of that unit's regularised ``XᵀX`` and the
    plane's refit metric -- which the caller memoises per ``(unit, plane)``.
    Per unit alone would be wrong and silently so: the refit objective is keyed
    by scale plane, so a memo that ignored the plane would price a unit's
    second family under the first family's objective.  Per rung would be
    wasteful: no rate reaches that call, and a twelve-anchor surface would
    otherwise factorise the same Hessian twelve times.  The plane is constant
    across every rung of a family and differs between families, so the bound is
    one factorisation per unit per family-plane while retained. Selected
    source campaigns bound the memo to the compatible anchor batch width.

    A **missing key is a hard failure**, never a silently H-free encode: this
    codebase has already been bitten once by a render whose activation lookup
    missed and quietly fell back to RTN, raising nothing, and an H-free encode
    of a rung whose shipping bytes are H-aware is exactly that bug with
    different bytes.
    """
    from . import tessera_render

    prepared = CellRecipe.prepare(
        qname=qname, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required, static_input_scale=static_input_scale,
        structure=structure, profile=profile)
    started = time.time()
    render, blob = tessera_render.encode_tessera_unit(
        weight, format_name, recipe=prepared.wire,
        activation_kwargs=prepared.activation_kwargs,
        hessian_required=prepared.hessian_required, verify=False)
    return _score_and_publish(qname=qname, weight=weight, activations=activations,
        format_name=format_name, cache=cache, wire_dir=wire_dir,
        prepared=prepared, render=render, blob=blob, elapsed=time.time() - started,
        publisher=publisher)


#: The context the campaign's pricing seam names when it binds the served
#: activation quantiser (RobTand/prismaquant#1481).
SERVED_QUANTIZER_CONTEXT = "Stage A Tessera campaign activation pricing"

#: The identity record this process bound through the owner, or ``None`` when
#: the campaign bound nothing itself. Stamped into the cost table's provenance.
_SERVED_QUANTIZER_RECORD = None


def _bind_served_quantizer(qname, format_name):
    """Bind the served activation arithmetic before this rung is scored.

    A rung whose route executes the static NVFP4 activation contract is priced
    by vLLM's registered ``scaled_fp4_quant``, and the contract refuses to price
    it unbound (RobTand/prismaquant#567). Until #1481 nothing on the campaign
    path bound it, so every A4 rung raised ``ServedQuantizerUnboundError``.

    The rule has one owner, ``joint_cost_quantum.bind_joint_served_quantizer``,
    which Stage B already calls; this seam calls it too. It requires the
    registered operator for a served contract, never falls back to the Torch
    model, and binds nothing for A8/A16 rungs, so those rosters never import the
    serving extension.

    A process that already holds a binding keeps it. Either an earlier rung of
    this run bound it, or a CPU screen declared the model on purpose
    (``tests/priced_model_screen.py``). The owner would refuse a second,
    different binding anyway, so re-asking could only turn a declared choice
    into a refusal.
    """
    global _SERVED_QUANTIZER_RECORD
    from .nvfp4_activation_contract import active_served_quantizer_identity
    from .joint_cost_quantum import bind_joint_served_quantizer

    if active_served_quantizer_identity() is not None:
        return
    record = bind_joint_served_quantizer(
        {qname: (format_name,)}, context=SERVED_QUANTIZER_CONTEXT)
    if record is not None:
        _SERVED_QUANTIZER_RECORD = dict(record)
        print(f"[campaign] served quantizer bound for {format_name}: backend "
              f"{record.get('backend')} op {record.get('op')} image "
              f"{record.get('image_content_sha256')} dequant "
              f"{record.get('dequant_kernel')}", flush=True)


def _is_glm_direct_consumer_profile(profile) -> bool:
    """Use the declared architecture before the undeclared profile identity."""
    if profile is None:
        return False
    architectures = profile.declared_architectures()
    if architectures:
        return architectures == ("Glm5NextForConditionalGeneration",)
    return profile.name == "glm5_next"


@dataclass(frozen=True)
class _DirectConsumer:
    module: str
    role: str
    activation_contract: str

    @staticmethod
    def _route_family(family):
        from tessera.metadata import route_family_for_grid
        grid = family.base + (f"x{family.arity}" if family.arity > 1 else "")
        return route_family_for_grid(grid)

    def resident_bytes(self, family, shape):
        from tessera.metadata import direct_consumer_resident_bytes
        rows, columns = map(int, shape)
        extra = direct_consumer_resident_bytes(
            self.module, self._route_family(family), rows, columns, [(self.role, rows)])
        return extra["resident_bytes_resident_mode"]

    def render(self, blob, family, device):
        from tessera.projection import direct_consumer_weight
        return direct_consumer_weight(blob, self.module, self.role,
                                      self._route_family(family), device=str(device))


def _direct_consumer(qname, *, profile):
    source = qname.removesuffix(".weight")
    if not source.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.weights_proj")):
        return None
    if profile is None:
        raise _ActivationScaleContractError(
            f"{qname}: a direct-consumer leaf needs the run profile to bind "
            "its activation contract; profile=None drops the GLM a16/a32 "
            "contract silently, so it refuses instead.")
    if not _is_glm_direct_consumer_profile(profile):
        return None
    from tessera.metadata import fused_module, role_name, direct_consumer_activation_contract
    tensor = source + ".weight"
    fused = fused_module(tensor, "Glm5NextForConditionalGeneration")
    module = source if fused is None else fused[0]
    role = role_name(tensor)
    contract = direct_consumer_activation_contract(module, role)
    return None if contract is None else _DirectConsumer(module, role, contract)


@dataclass(frozen=True)
class CellRecipe:
    """A cell's resolved wire, activation arithmetic and encoder inputs."""

    spec: FormatSpec
    family: TesseraFamily
    rung: int
    wire: WireRecipe
    structure: str | None
    activation_qdq: Callable
    input_scale: float | None
    activation_contract: str
    activation_kwargs: dict | None
    hessian_required: bool
    direct_consumer: _DirectConsumer | None

    @property
    def direct_contract(self) -> str | None:
        return None if self.direct_consumer is None else self.direct_consumer.activation_contract

    def memory_bytes_for_shape(self, shape) -> int:
        base = int(self.spec.memory_bytes_for_shape(tuple(shape)))
        extra = (0 if self.direct_consumer is None
                 else self.direct_consumer.resident_bytes(self.family, shape))
        return base + extra

    @classmethod
    def prepare(cls, *, qname, format_name, activation_kwargs_for,
                hessian_required, static_input_scale, structure=None, profile=None) -> CellRecipe:
        """Admit each unit's Hessian and served activation contract before encode."""
        direct_consumer = _direct_consumer(qname, profile=profile)
        direct_contract = None if direct_consumer is None else direct_consumer.activation_contract
        if direct_contract is None:
            _bind_served_quantizer(qname, format_name)
        from . import format_registry as fr
        from .tessera_formats import (
            parse_tessera_format_name, tessera_served_wire_recipe, tessera_serving_route,
        )
        from .tessera_render import HessianContractError

        from .tessera_render import rung_accepts_hessian

        spec = fr.get_format(format_name)
        family, rung = parse_tessera_format_name(format_name)
        # ONE resolve for this anchor. The plane the kwargs are built on, the plane
        # the predicate reads and the plane the encode writes are this object --
        # not three lookups that agree only while nothing clears the recipe memo.
        # It is the wire this unit's STRUCTURE is served on (#1502): a routed
        # E2M1x2 stack below the cap ships span-2 TCQ, not the research WINDOW
        # recipe, and a structure the pinned contract attests no wire for refuses
        # here rather than pricing bytes the serve cannot read.
        wire = tessera_served_wire_recipe(family, rung, structure=structure)
        # The A side, as served.  A route with a STATIC activation contract
        # executes vLLM's static-global-scale ``scaled_fp4_quant`` (UE4M3 block
        # scales) against the artifact's ``trellis_input_global_scale``; every
        # other route keeps the serving format's own dynamic quantiser, taken by
        # reference from the spec.
        #
        # WHICH of the two is the SPEC's answer, never a compare of the route's
        # source-format name against ``"NVFP4"`` (#205's rule, #221's fix): the
        # spec resolved above already carries the contract, because
        # ``synthesize_tessera_spec`` derived it from the registry row the route
        # names.  A Tessera rung routed through the same kernel has the same
        # contract and a different name, and the day a second registry row gets a
        # contract, a name compare here would price it under a quantiser the
        # runtime never runs while the cache scorer and the KL hooks -- which
        # already read the row -- refuse the same unit.
        route = tessera_serving_route(family, wire, rung)
        contract = spec.static_activation_contract
        if contract is not None:
            if static_input_scale is None or not float(static_input_scale) > 0:
                raise _ActivationScaleContractError(
                    f"{qname} {format_name}: this rung's route executes the "
                    f"static activation contract {contract.execution} "
                    f"({route.contract}) and no calibrated input_global_scale was "
                    "supplied. Scoring it with the registry's dynamic FP32-scale "
                    "quantiser would price an activation regime the serve does "
                    "not execute; a lookup that misses must refuse, not fall back."
                )
            input_scale = float(static_input_scale)
            # The contract's own oracle, not a second spelling of it.
            activation_qdq = (
                lambda x: contract.quantize_dequantize(x, input_scale))
        else:
            input_scale = None
            activation_qdq = spec.activation_quantize_dequantize
        if direct_contract is not None:
            activation_qdq = fr.get_format("BF16").activation_quantize_dequantize
        activation_kwargs = None
        # Whether an H can be applied is a property of the RUNG'S WIRE, not of the
        # run, and it is DERIVED from what the pinned ActivationSource emits for
        # that wire's plane rather than restated here (``rung_accepts_hessian``).
        # It reads True on every plane PrismaQuant resolves at this pin.
        hessian_required = bool(hessian_required) and rung_accepts_hessian(
            format_name, wire)
        if hessian_required:
            if activation_kwargs_for is None:
                raise HessianContractError(
                    f"{qname}: hessian_required=True but no activation-kwargs "
                    "source was passed to measure_cell")
            activation_kwargs = activation_kwargs_for(qname, wire.scale_plane)
            if not activation_kwargs:
                raise HessianContractError(
                    f"{qname}: no Hessian for this Linear. A lookup that misses "
                    "must not fall through to a weights-only encode.")
        return cls(spec=spec, family=family, rung=rung, wire=wire, structure=structure,
                    activation_qdq=activation_qdq, input_scale=input_scale,
                    activation_contract=direct_contract or str(spec.act_dtype_name or "a16"),
                    activation_kwargs=activation_kwargs,
                    hessian_required=hessian_required, direct_consumer=direct_consumer)


def _score_and_publish(*, qname, weight, activations, format_name, cache, wire_dir,
                       prepared, render, blob, elapsed, encoding_batch_size=1,
                       publisher=None):
    """Score decoded bytes and publish the existing cache/wire entries.

    ``publisher``, when given, is a
    :class:`~prismaquant.tessera_publication.BoundedPublisher` that performs
    the two writes on its own thread instead of here.  The bytes handed to it
    are the same bytes and the writes are the same calls; what moves is only
    which thread runs them, and the caller then owes the completion an
    ordering rule -- the anchor is not journalled until its files exist.  With
    no publisher both writes finish before this operation returns.
    """
    import torch
    from .production_weight_cache import (
        _canonical_rendered_weight_tensor, _local_forward_render_score,
        _store_rendered_weight_entry)

    spec, family, rung = prepared.spec, prepared.family, prepared.rung
    activation_qdq, input_scale = prepared.activation_qdq, prepared.input_scale
    direct_contract = prepared.direct_contract
    preserve_fp32 = direct_contract == "a32"
    cache_dtype = torch.float32 if preserve_fp32 else torch.bfloat16
    if direct_contract is not None:
        render = prepared.direct_consumer.render(blob, family, weight.device)
    publication_bytes = render.numel() * (4 if preserve_fp32 else 2) + len(blob)
    score, metric, quantized, _clipped = _local_forward_render_score(
        reference_weight=weight,
        rendered_weight=render,
        activations=activations,
        activation_quantize=activation_qdq,
        # No pre-clip on top: the static route's clamp lives inside the served
        # oracle (``nvfp4_activation_qdq_served`` clamps at the stored UE4M3
        # scale), and the dynamic routes have no calibrated clip at all.
        activation_max_abs=None,
    )
    if metric != "output_mse":
        raise RuntimeError(f"unexpected render-score metric {metric!r}")
    hessian_applied = bool(prepared.activation_kwargs)

    # Room first, then the bytes.  The copy below is a new host allocation the
    # writer will own, so the budget has to admit it BEFORE it exists: charging
    # it after the fact would leave this thread holding one artifact more than
    # the bound allows, every time the writer is behind.  The size is known
    # without a copy: the declared cache dtype sets each element's bytes.
    if publisher is not None:
        publisher.reserve(publication_bytes)
    try:
        # The device-to-host copy stays on the thread that owns the device
        # work, whichever way the bytes are written.
        # ``_store_rendered_weight_entry`` canonicalises again below and that
        # second call is an identity on a CPU tensor already in the target
        # dtype and contiguous, so the synchronous path stores the same object
        # it always stored.  Staging it here is what gives the writer thread
        # bytes nobody else owns.
        staged = _canonical_rendered_weight_tensor(
            render, weight_dtype=cache_dtype, preserve_fp32=preserve_fp32)
    except BaseException:
        if publisher is not None:
            publisher.release(publication_bytes)
        raise
    # The wire, beside the render.  A ``.tessera`` shard per (qname, rung),
    # named the way the cache names its weight shards, so the export leg can
    # find the exact bytes this row was priced on instead of re-encoding.
    wire_path = campaign_wire.wire_path(wire_dir, qname, format_name)

    def _publish():
        # Check on the writer thread as well as at batch admission, so queued
        # publications see every earlier durable coordinate before any write.
        campaign_wire.require_injective_anchor_filenames(
            cache, wire_dir, ((qname, format_name),))
        _store_rendered_weight_entry(
            weights=cache.weights,
            qname=qname,
            fmt=format_name,
            tensor=staged,
            cache_dir_path=Path(cache.cache_dir) if cache.cache_dir else None,
            weight_dtype=cache_dtype, preserve_fp32=preserve_fp32,
        )
        tmp = wire_path.with_suffix(".tessera.tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, wire_path)
        campaign_wire.wire_coordinates(cache, wire_dir).add((qname, format_name))
        if getattr(cache, 'metadata', {}).get('release_completed_anchor_file_pages'):
            # The existing PWC entry is already disk-backed. Completed anchor
            # files must not accumulate an unbounded page-cache owner across
            # rungs. The release helper checks the entry's identity, fsyncs
            # and advises it from its own descriptor. No verification consumes
            # a second hash of these completed files here.
            from .perturbed_x_cache import release_activation_cache_file_pages
            rendered_path = Path(cache.cache_dir)/cache.weights[(qname, format_name)]
            for path in (rendered_path, wire_path):
                release_activation_cache_file_pages(path, expected_stat=path.stat())

    if publisher is None:
        _publish()
    else:
        publisher.submit(PublicationJob(
            key=(FILES_JOB, qname, format_name),
            charged_bytes=publication_bytes,
            publish=_publish,
        ))

    bits = spec.bits_for_shape(tuple(weight.shape))
    return CampaignAnchor(
        qname=qname,
        family=family.name,
        format_name=format_name,
        body_rate_q256=int(rung),
        dloss=float(score),
        # One calibration draw: the stderr a single measurement supports is
        # zero, and inventing one would put a fabricated uncertainty into the
        # allocator's UCB hedge.  Reported as zero and named as such.
        dloss_stderr=0.0,
        memory_bytes=prepared.memory_bytes_for_shape(weight.shape),
        bits_per_param=float(bits) / max(1, int(weight.numel())),
        activation_contract=prepared.activation_contract,
        activation_quantized=bool(quantized),
        wire_bytes=len(blob),
        seconds=elapsed,
        hessian_applied=hessian_applied,
        input_global_scale=input_scale,
        encoding_batch_size=encoding_batch_size,
    )


def measure_cells(*, qnames, weights, activations, format_name,
                  cache, wire_dir, activation_kwargs_for=None,
                  hessian_required=True, static_input_scales=None,
                  publisher=None, structures=None, profile=None) -> list[CampaignAnchor]:
    """One producer batch, with the scalar path's per-unit gates and storage.

    ``structures`` maps a unit to its serving structure, as ``structure`` does
    for :func:`measure_cell`; a unit it does not name declares none.
    """
    if not qnames or len(set(qnames)) != len(qnames):
        raise ValueError("anchor batch needs distinct nonempty unit names")
    if len(weights) != len(qnames) or len(activations) != len(qnames):
        raise ValueError("anchor batch needs one weight and activation input per unit")
    if len(qnames) == 1:
        name = qnames[0]
        return [measure_cell(qname=name, weight=weights[0], activations=activations[0],
            format_name=format_name, cache=cache, wire_dir=wire_dir,
            activation_kwargs_for=activation_kwargs_for, hessian_required=hessian_required,
            static_input_scale=(static_input_scales or {}).get(name),
            structure=(structures or {}).get(name), publisher=publisher, profile=profile)]
    from .tessera_render import encode_tessera_units

    prepared = [CellRecipe.prepare(
        qname=name, format_name=format_name,
        activation_kwargs_for=activation_kwargs_for,
        hessian_required=hessian_required,
        static_input_scale=(static_input_scales or {}).get(name),
        structure=(structures or {}).get(name), profile=profile) for name in qnames]
    # The producer call takes ONE recipe and ONE Hessian requirement for the
    # whole batch, from its first entry.  Both derive from (family, rung,
    # structure) only, so a batch built by ``compatible_batches`` agrees by
    # construction; a batch
    # that does not would encode some unit under another unit's recipe, and is
    # refused rather than priced.
    for name, entry in zip(qnames[1:], prepared[1:]):
        if (entry.wire != prepared[0].wire
                or entry.hessian_required != prepared[0].hessian_required):
            raise ValueError(
                f"anchor batch {format_name}: {name} prepares a different wire "
                f"recipe or Hessian requirement than {qnames[0]}")
    campaign_wire.require_injective_anchor_filenames(
        cache, wire_dir, ((name, format_name) for name in qnames))
    started = time.time()
    encoded = encode_tessera_units(
        weights, format_name, recipe=prepared[0].wire,
        activation_kwargs=[entry.activation_kwargs for entry in prepared],
        hessian_required=prepared[0].hessian_required)
    # Batch wall time is apportioned, not attributed as a measured per-unit
    # duration. Summing anchor.seconds still reports the actual encoding time.
    elapsed = (time.time() - started) / len(qnames)
    return [_score_and_publish(qname=name, weight=weight, activations=acts,
        format_name=format_name, cache=cache, wire_dir=wire_dir, prepared=entry,
        render=render, blob=blob, elapsed=elapsed,
        encoding_batch_size=len(qnames), publisher=publisher)
        for name, weight, acts, entry, (render, blob) in zip(
            qnames, weights, activations, prepared, encoded)]


def compatible_batches(pending, *, weights, batch_size, structures=None):
    """Bound compatible encodes inside this action; never assign hosts.

    Membership is by ``(family, rung, structure, shape, dtype, device)`` for
    every unit, dense or projected expert.  A unit's wire recipe and Hessian
    requirement are functions of ``(family, rung, structure)`` alone
    (``CellRecipe.prepare``; structure since #1502, because a routed E2M1x2 stack
    below the cap is served on another wire than a dense unit), and the
    joined producer call is column-independent, so a dense unit joins
    same-shape units of other groups exactly as an expert joins its stack.
    Dense units used to be keyed by ``(unit, family, rung)``, which kept every
    dense encode at batch one: a long serial chain of narrow Viterbi calls at
    about 25 W of the 140 W envelope (RobTand/prismaquant#1479).  The ORDER of
    the batches is unit-major: the batches of one compatible key differ only
    by rung, and the chunk at one position holds the same members at every
    rung when the round pended every member at every rate (round one does,
    per member in rate order), so consecutive batches encode the same units
    at successive rungs.  The encoder memo is sized to the batch width
    (``encoder_memo_capacity``), so that order is what lets a unit's block-LDL
    factorization be reused across its rungs instead of refactorized once per
    (unit, rung) with 864 batches of other units in between
    (RobTand/prismaquant#389).  Rung-major emission -- every batch of rung A,
    then every batch of rung B -- is what the insertion-ordered grouping
    produced before, and with the memo at the batch width it hit nothing.
    """
    if batch_size < 1:
        raise ValueError("anchor batch size must be positive")
    if batch_size == 1:
        return [[item] for item in pending]
    import torch

    groups = {}
    for item in pending:
        name, family, rung = item
        weight = weights[name]
        # A ``meta`` placeholder is a host weight the row stream has not read
        # yet (PQ #1654); it batches exactly as the host copy it stands for.
        base = (family, (structures or {}).get(name), tuple(weight.shape),
                weight.dtype, torch.device("cpu") if weight.is_meta else weight.device)
        key = (family, rung, *base[1:])
        groups.setdefault(key, (base, []))[1].append(item)
    # Sort key: the base (the key minus its rung) in first-appearance order,
    # then the chunk position, then the rung in first-appearance order -- so
    # chunk 0 of every rung of one base precedes chunk 1 of any of them.
    base_rank, key_rank, chunks = {}, {}, []
    for key, (base, group) in groups.items():
        base_rank.setdefault(base, len(base_rank))
        key_rank[key] = len(key_rank)
        for position, start in enumerate(range(0, len(group), batch_size)):
            chunks.append(((base_rank[base], position, key_rank[key]),
                           group[start:start + batch_size]))
    chunks.sort(key=lambda chunk: chunk[0])
    return [batch for _rank, batch in chunks]


def stale_wire_reason(anchor, record, *, structure) -> "str | None":
    """Why a resumed row's receipt names a wire ``structure`` is not served on.

    ``None`` when the receipt's stamped recipe is the served recipe for the
    row's rung, or when the receipt carries no readable recipe (the full
    identity check that follows refuses those).  Only a receipt whose recipe
    is readable and differs is stale; every other disagreement stays a
    refusal of the whole resume.
    """
    if structure is None:
        return None
    from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe

    stamped = ((record or {}).get("identity") or {}).get("recipe")
    parsed = parse_tessera_format_name(anchor.format_name)
    if not isinstance(stamped, dict) or parsed is None:
        return None
    family, rung = parsed
    served = tessera_served_wire_recipe(family, rung, structure=structure,
                                        refuse_unattested=False).to_config()
    observed = {key: value for key, value in stamped.items() if key not in ("grid", "q256")}
    if observed == served:
        return None
    return (f"{anchor.format_name}: receipt stamped {observed}, but a {structure} unit is "
            f"served on {served} at this rung (#1502); re-priced, not adopted")


def served_quantizer_record():
    """The binding this owner made, independent of the returned provenance copy."""
    return None if _SERVED_QUANTIZER_RECORD is None else dict(_SERVED_QUANTIZER_RECORD)


def executes_static_activation_contract(format_name: str) -> bool:
    """Does this rung's route execute a STATIC activation contract?

    The one derivation (#205, #221): the route names the registry row whose
    contract it executes (``activation_source_format``) and the ROW owns the
    answer (``FormatSpec.static_activation_contract``) -- never a compare of
    that row's NAME against ``"NVFP4"``.  It is
    ``tessera_formats.route_static_activation_contract`` off the same route
    ``synthesize_tessera_spec`` stamps onto the rung's spec, so the rung this
    refuses to resume is exactly the rung ``measure_cell`` prices.

    The row is read live, every time: it is a dictionary lookup, and the
    registry is what a test (or a lane) replaces when a second row gains a
    contract.  What is memoised is the pure part -- canonical name to serving
    route -- sized by the format key space: the closed-roster bind asks it
    once per format per unit, 1,793 routes for each of an 864-unit row's
    holders, measured at ~1.1 s per unit on the CPU worker when each ask
    synthesized a whole spec, a GPU-idle startup phase of a quarter hour per
    row.
    """
    from . import format_registry as fr
    from .tessera_lane import is_tessera_format_name

    canonical = fr.canonical_format_name(format_name)
    row = fr.REGISTRY.get(canonical)
    if row is not None:
        return row.static_activation_contract is not None
    if not is_tessera_format_name(canonical):
        fr.get_format(canonical)  # raises the registry's KeyError
    from .tessera_formats import route_static_activation_contract

    return route_static_activation_contract(_tessera_route_memo()(canonical)) is not None


def _tessera_route(canonical: str):
    from .tessera_formats import (
        parse_tessera_format_name, tessera_serving_route, tessera_wire_recipe,
    )

    family, rung = parse_tessera_format_name(canonical)
    return tessera_serving_route(family, tessera_wire_recipe(family, rung), rung)


@functools.lru_cache(maxsize=1)
def _tessera_route_memo():
    # Size the memo from the producer format key space on first use.
    from .tessera_formats import lazily_sized_cache, recipe_cache_bound
    return lazily_sized_cache(recipe_cache_bound)(_tessera_route)


def require_resumable_cell(anchor: CampaignAnchor, static_scales, *,
                           check_scoring_metadata: bool = True, profile=None) -> None:
    """Refuse a resumed anchor priced under a different activation contract.

    This is the per-row half of the resume identity rule, shared by pre-link
    seed admission and the campaign checkpoint identity reader. The run-level
    campaign identity binds this run's static scales
    and policy into the journal. Resume merges checkpoint rows into
    this run's table, and the table's rows must be one currency.  A W4A4
    anchor with no ``input_global_scale`` was measured under the pre-#194
    dynamic FP32-scale quantiser; one with a *different* scale was measured
    on a different calibration; a dynamic-route row carrying a scale was
    stamped by no producer this campaign has.  None is a price of this run's
    served A side, and merging one silently is the exact mixed-table failure
    the Hessian identity guard exists to catch on its own axis.
    """
    from . import format_registry as fr

    spec = fr.get_format(anchor.format_name)
    if spec.static_activation_contract is None:
        if anchor.input_global_scale is not None:
            raise _ActivationScaleContractError(
                f"checkpoint anchor {anchor.qname} {anchor.format_name} "
                f"carries input_global_scale={anchor.input_global_scale!r} "
                "but its route keeps the serving format's dynamic activation "
                "quantiser: no producer of this campaign stamps a static scale "
                "on that route, so the row is not one of this run's prices.")
    elif anchor.input_global_scale is None:
        raise _ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} carries "
            "no input_global_scale: it was priced under the pre-served-"
            "contract dynamic activation quantiser and cannot be merged into "
            "a served-contract table. Delete the checkpoint (or pass a fresh "
            "--checkpoint) to re-measure these anchors."
        )
    elif (static_scales.get(anchor.qname) is None or
          float(anchor.input_global_scale) != float(static_scales[anchor.qname])):
        expected = static_scales.get(anchor.qname)
        raise _ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} was "
            f"priced at input_global_scale={anchor.input_global_scale!r} but "
            f"this run's calibration yields {expected!r}. A resumed anchor "
            "must have been scored under this run's own static scales, or "
            "the table mixes two activation calibrations under one identity."
        )
    # The closed-roster identity template is built before scoring from input
    # descriptors. It has no measured contract/observation to validate. Only
    # that caller opts out; actual published and reused rows use the default.
    if check_scoring_metadata is False:
        return
    # Producer byte identity does not authenticate measured scoring metadata.
    # Direct consumers use the same contract as the measured row.
    direct_consumer = _direct_consumer(anchor.qname, profile=profile)
    direct_contract = None if direct_consumer is None else direct_consumer.activation_contract
    expected_contract = direct_contract or str(spec.act_dtype_name or "a16")
    if anchor.activation_contract != expected_contract:
        raise _ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} activation contract "
            f"{anchor.activation_contract!r} differs from the expected {expected_contract!r}")
    # A direct consumer never quantizes its input rows.
    allows_quantized = direct_contract is None and spec.act_quant_changes_input
    if (type(anchor.activation_quantized) is not bool or
            (anchor.activation_quantized and not allows_quantized)):
        raise _ActivationScaleContractError(
            f"checkpoint anchor {anchor.qname} {anchor.format_name} activation observation "
            f"{anchor.activation_quantized!r} is incompatible with {expected_contract!r}")


def resume_cell_activation(anchor, calibration_source, static_scales, *, structure=None,
                           check_scoring_metadata=True, profile=None):
    """Admit the resumed wire's Hessian applicability before its scoring metadata."""
    from .tessera_formats import parse_tessera_format_name, tessera_served_wire_recipe
    from .tessera_render import rung_accepts_hessian

    family, rung = parse_tessera_format_name(anchor.format_name)
    wire = tessera_served_wire_recipe(family, rung, structure=structure,
                                      refuse_unattested=False)
    activation = (calibration_source if calibration_source is not None
                  and rung_accepts_hessian(anchor.format_name, wire) else None)
    if bool(anchor.hessian_applied) != (activation is not None):
        raise RuntimeError("checkpoint anchor Hessian applicability disagrees with the producer")
    require_resumable_cell(anchor, static_scales,
                          check_scoring_metadata=check_scoring_metadata, profile=profile)
    return activation
