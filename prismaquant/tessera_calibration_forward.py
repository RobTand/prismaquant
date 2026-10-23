"""Stateless calibration forward over caller-owned models and tokens.

Scoring rows are bounded; Hessians, maxima and row counts cover every observed
input. Model/source lifetime, calibration loading and cell recipes stay with
those existing owners. Capture hooks and partial outputs are released on failure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch import Tensor


class CalibrationStatistics(NamedTuple):
    rows: dict[str, Tensor | None]
    hessians: dict[str, Tensor | None]
    token_counts: dict[str, int]
    max_abs: dict[str, float]


def capture_activation_statistics(model, targets, tokens, max_rows: int, device,
                         *, want_hessian: bool = False, profile=None,
                         boundary_consumer=None, forward_batch=None, resource_check=None,
                         shared_packed_inputs: bool = False, on_forwards_complete=None,
                         expected_shared_input_groups=None, row_consumer=None) -> CalibrationStatistics:
    """One model forward per batch, for dense and declared packed projections.

    Returns ``(rows, hessians, token_counts, max_abs)``. Counts describe every
    observed input row, including when Hessians and retained scoring rows are
    disabled; they are calibration provenance, not a Hessian-computation flag.
    ``resource_check(label)`` optionally enforces the caller's existing guard
    before/after each output concat and Hessian CPU transfer. A refusal clears
    owned output tensors before propagating; integer row observations remain
    available for diagnostics. The default performs no additional checks.
    ``on_forwards_complete()`` is an optional source-lifecycle boundary after
    every original forward, hook removal and release of local packed weight
    views. It runs before CPU output materialization, within failure cleanup.
    The caller can release only exhausted source state while retaining the
    next resident layer and current hidden boundaries. The default is None.

    ``shared_packed_inputs`` is an experimental, default-off collector policy.
    Only packed siblings with the same module, expert and derived input kind
    share internal accumulation. The existing store retains private, bounded
    FP32 device prefixes until output materialization; returned CPU H/X tensors
    remain independent for every qname. This is not a new activation cache.

    ``row_consumer(unit_names, flat_rows)`` is an optional, default-off research
    seam. When given, it is called on the UNCAPPED flat rows of every unique
    input group -- dense pre-hook or routed derivation, both feed ``accumulate``
    -- once per group per batch, with the tuple of exact unit names sharing that
    input (``group_members``). The call happens before the scoring-prefix cap
    and independently of ``want_hessian``; zero-row calls never reach it. The
    consumer owns its own copies; the collector retains no consumer state and
    the default ``None`` keeps the historical behavior bit-identical.

    Three different things come out of the same hook, and they have different
    row budgets on purpose:

    * ``rows`` is capped at ``max_rows`` because it feeds
      ``_local_forward_render_score``, whose cost is linear in rows and whose
      job is to *rank* rungs.
    * ``hessians`` is ``XᵀX`` accumulated over **every** calibration row the
      Linear sees, because it feeds the encoder, whose job is to *build* the
      bytes.  Capping it at ``max_rows`` would hand a 3072-column
      ``down_proj`` a rank-256 Hessian -- rank-deficient by a factor of
      twelve, and wrong in a way no downstream check would catch.  So the
      accumulation runs before the keep's early return, not after it.
    * ``max_abs`` is ``max|x|`` over **every** calibration row, unconditional
      and for the same reason the Hessian is uncapped: it calibrates the
      static NVFP4 ``input_global_scale`` the W4A4 routes are priced and
      served under, and a maximum over a prefix of the rows is a different
      calibration than the one an exporter would take.

    Hessians accumulate in fp32 on the model's device and move to CPU once at
    the end. Legacy scoring chunks move to CPU as observed; the experimental
    policy instead transfers their bounded device prefix at the end. Packed units use the existing
    module-input collector and routing/SwiGLU derivation, and land in the SAME
    three accumulators as a dense Linear -- rows, Hessian and ``max_abs`` --
    so there is one notion of "what the campaign captured" for either
    population, and ``write_export_inputs`` has one input to write.  The
    score-row cap never caps routed rows before the Hessian or the maximum.
    The main-entry gate requires profile-declared projections and the
    producer's planning tool before calibration. The campaign later binds
    that producer projection to these live views before pricing the units
    and carrying their wire receipts into export (PrismaQuant #183).
    """
    import torch

    store: dict[str, list] = {name: [] for name in targets}
    kept: dict[str, int] = {name: 0 for name in targets}
    hess: dict[str, object] = {name: None for name in targets}
    seen: dict[str, int] = {name: 0 for name in targets}
    amax: dict[str, float] = {name: 0.0 for name in targets}
    # Singleton dense owners stay distinct. Packed groups below may replace
    # only proven identical gate/up owners; seen remains a per-qname diagnostic.
    group_members = {name: [name] for name in targets}
    routed_seen: dict[str, int] = {}
    handles = []
    packed_collector = None
    inventory = by_name = selected_by_module = parents = None
    member = selected = unique = consume_packed = None
    calibration_coordinates = None

    def consume_boundary(qname, module, args, kwargs):
        boundary_consumer(qname, module, args, kwargs, calibration_coordinates)

    def accumulate(name, x):
        # ONE accumulator for both populations: a dense Linear's pre-hook and
        # a packed projection's routed rows land here, so the three outputs
        # (score rows, Hessian, max|x|) are the same three for either, and a
        # packed unit can feed ``_static_input_scales``/``write_export_inputs``
        # by the same path a dense one does once the main-entry packed gate
        # opens.  A zero-row call (an expert no token was routed to on this
        # batch) contributes nothing to any of them.
        flat = x.detach().reshape(-1, x.shape[-1])
        if not flat.shape[0]:
            return
        for member_name in group_members[name]:
            if member_name in routed_seen:
                routed_seen[member_name] += int(flat.shape[0])
            seen[member_name] += int(flat.shape[0])
        batch_max = flat.abs().amax().float()
        previous_max = amax[name]
        if not isinstance(previous_max, torch.Tensor):
            previous_max = torch.zeros((), dtype=torch.float32, device=batch_max.device)
        # Python max(previous, NaN) preserves previous. fmax keeps that exact
        # policy while avoiding a device-to-host scalar read on every batch.
        amax[name] = torch.fmax(previous_max, batch_max)
        if row_consumer is not None:
            # Research seam: the full uncapped rows, once per unique input
            # group, before any scoring-prefix cap and regardless of
            # ``want_hessian``.  The consumer must own its bytes; the flat
            # tensor may still alias a source/derived activation plane.
            row_consumer(tuple(group_members[name]), flat)
        if want_hessian:
            # Every row, before any cap: see the docstring.
            f32 = flat.to(dtype=torch.float32)
            gram = f32.t() @ f32
            if hess[name] is None:
                hess[name] = gram
            else:
                hess[name] += gram
        room = max_rows - kept[name]
        if room <= 0:
            return
        if shared_packed_inputs:
            # Never retain a view of a source/derived activation plane. One
            # private prefix buffer is owned by the existing store, and copy_
            # preserves the original row order and BF16-to-FP32 conversion.
            take_rows = min(room, int(flat.shape[0]))
            if not store[name]:
                store[name].append(torch.empty(
                    (max_rows, flat.shape[1]), dtype=torch.float32, device=flat.device))
            store[name][0][kept[name]:kept[name] + take_rows].copy_(flat[:take_rows])
            kept[name] += take_rows
        else:
            take = flat[:room].to(dtype=torch.float32, device="cpu")
            store[name].append(take)
            kept[name] += int(take.shape[0])

    def make_hook(name):
        def hook(_module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            accumulate(name, x)
        return hook

    modules = dict(model.named_modules())
    missing_modules = set(targets) - modules.keys()
    if missing_modules:
        from .routed_experts import (
            profile_declared_packed_expert_projections,
            resolve_routed_expert_profile,
            packed_experts_parent_module,
            derive_per_expert_activations,
        )
        from .production_weight_cache import _PackedExpertActivationCollector

        profile = resolve_routed_expert_profile(model, profile)
        inventory = profile_declared_packed_expert_projections(model, profile)
        by_name = {member.qname: member for member in inventory}
        unknown = missing_modules - by_name.keys()
        if unknown:
            raise RuntimeError(f"campaign activation targets are not declared units: {sorted(unknown)}")
        selected_by_module = {}
        # These are the input kinds published by the existing derivation,
        # not a mapping to the producer's served role/group vocabulary.
        from .routed_experts import packed_activation_input_kind
        input_kind = {}
        for name in sorted(missing_modules):
            member = by_name[name]
            if member.param_name not in input_kind:
                input_kind[member.param_name] = packed_activation_input_kind(member.param_name)
            selected_by_module.setdefault(member.module_qname, []).append(member)
            routed_seen[name] = 0
        parents = {name: packed_experts_parent_module(model, name)
                   for name in selected_by_module}
        if shared_packed_inputs:
            for module_qname, selected in selected_by_module.items():
                owners = {}
                unique = []
                for member in selected:
                    key = (module_qname, member.expert_id, input_kind[member.param_name])
                    owner = owners.get(key)
                    if owner is None:
                        owners[key] = member.qname
                        unique.append(member)
                    else:
                        group_members[owner].append(member.qname)
                        del group_members[member.qname]
                        for state in (store, kept, hess, amax):
                            del state[member.qname]
                        state = None
                selected_by_module[module_qname] = unique

        def consume_packed(module_qname, x):
            selected = selected_by_module.get(module_qname)
            if not selected:
                return
            derived = derive_per_expert_activations(
                selected[0].module, x, parents[module_qname],
                capture_down=any(input_kind[member.param_name] == "down"
                                 for member in selected),
                max_rows_per_expert=None,
            )
            for member in selected:
                accumulate(member.qname,
                           derived[input_kind[member.param_name]][member.expert_id])

        packed_collector = _PackedExpertActivationCollector(
            model, {member.module_qname for member in inventory},
            module_token_budget=0, store_device=device, store_qnames=set(),
            profile=profile, row_consumer=consume_packed,
            boundary_consumer=(consume_boundary if boundary_consumer is not None else None),
        )
    try:
        if expected_shared_input_groups is not None:
            actual = sorted(sorted(names) for names in group_members.values())
            expected = sorted(sorted(names) for names in expected_shared_input_groups.values())
            if not shared_packed_inputs or actual != expected:
                raise RuntimeError('actual capture input groups differ from memory admission')
        for name in targets:
            if name not in missing_modules:
                handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
        if packed_collector is not None:
            packed_collector.install()
        with torch.no_grad():
            sample_offset = 0
            for batch in tokens:
                if boundary_consumer is not None:
                    if batch.ndim != 2:
                        raise RuntimeError("boundary capture requires [sample, token] calibration IDs")
                    sample_ids = torch.arange(sample_offset, sample_offset + batch.shape[0], dtype=torch.int64)
                    positions = torch.arange(batch.shape[1], dtype=torch.int64)
                    calibration_coordinates = torch.cartesian_prod(sample_ids, positions)
                    sample_offset += batch.shape[0]
                (model if forward_batch is None else forward_batch)(batch.to(device))
    except BaseException as error:
        if shared_packed_inputs or on_forwards_complete is not None:
            store.clear()
            hess.clear()
            amax.clear()
            if torch.device(device).type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_error:
                    error.add_note(f'capture CUDA cache release failed: {cleanup_error!r}')
        raise
    finally:
        for handle in handles:
            handle.remove()
        if packed_collector is not None:
            packed_collector.remove()
        if on_forwards_complete is not None:
            # Hook removal alone leaves row_consumer's closure and these
            # PackedExpertProjection.weight views owning the old packed slabs.
            # Loop variables own views too, even after their dictionaries die.
            packed_collector = None
            inventory = by_name = selected_by_module = parents = None
            member = selected = unique = consume_packed = None
    unobserved = [name for name, count in routed_seen.items() if not count]
    if unobserved:
        if shared_packed_inputs or on_forwards_complete is not None:
            store.clear()
            hess.clear()
            amax.clear()
        raise RuntimeError(
            "packed campaign units have no routed calibration rows: "
            f"{sorted(unobserved)}; refusing a shared-Hessian or weight-only fallback")
    # All forwards are complete. Keep integer `seen` available to failure
    # diagnostics, but never pin a partial capture through this traceback.
    rows, hessians = {}, {}
    h = chunks = cpu_rows = cpu_h = None
    try:
        if on_forwards_complete is not None:
            on_forwards_complete()
        if resource_check is not None:
            resource_check('before_output_materialization')
        # Materialize each unit's maximum only once.
        amax = {name: float(value.item()) if isinstance(value, torch.Tensor) else value
                for name, value in amax.items()}
        if shared_packed_inputs:
            # Drain one unique group before making independent CPU siblings.
            # The final CPU footprint is unchanged. GPU storage (including
            # cached allocator blocks) must not accumulate behind those clones.
            maxima = {}
            for name, members in group_members.items():
                if resource_check is not None:
                    resource_check(f'before_output_group:{name}')
                if resource_check is not None:
                    resource_check(f'before_rows_concat:{name}')
                chunks = store.pop(name)
                # copy=True also compacts CPU qualification outputs: a short
                # prefix must not serialize the uninitialized buffer tail.
                cpu_rows = chunks[0][:kept[name]].to(device='cpu', copy=True) if chunks else None
                released_cuda = bool(chunks and chunks[0].is_cuda)
                chunks = None
                if resource_check is not None:
                    resource_check(f'after_rows_concat:{name}')
                if want_hessian:
                    if resource_check is not None:
                        resource_check(f'before_hessian_cpu_transfer:{name}')
                    h = hess.pop(name)
                    cpu_h = None if h is None else h.to(device='cpu')
                    released_cuda = released_cuda or (h is not None and h.is_cuda)
                    h = None
                    if resource_check is not None:
                        resource_check(f'after_hessian_cpu_transfer:{name}')
                if released_cuda:
                    torch.cuda.empty_cache()
                for index, member_name in enumerate(members):
                    if resource_check is not None:
                        resource_check(f'before_output_clone:{member_name}')
                    rows[member_name] = (cpu_rows if index == 0 or cpu_rows is None
                                         else cpu_rows.clone())
                    if want_hessian:
                        hessians[member_name] = (cpu_h if index == 0 or cpu_h is None
                                                 else cpu_h.clone())
                    maxima[member_name] = amax[name]
                    if resource_check is not None:
                        resource_check(f'after_output_clone:{member_name}')
                cpu_rows = cpu_h = None
                if resource_check is not None:
                    resource_check(f'after_output_group:{name}')
            amax = maxima
        # Do not retain both original chunks and concatenated outputs for the
        # whole scope. Full-model captures can hold GiB of scoring rows.
        for name in list(store):
            if resource_check is not None:
                resource_check(f'before_rows_concat:{name}')
            chunks = store.pop(name)
            rows[name] = torch.cat(chunks, dim=0) if chunks else None
            chunks = None
            if resource_check is not None:
                resource_check(f'after_rows_concat:{name}')
        if want_hessian and not shared_packed_inputs:
            # GB10 shares physical DRAM with CPU tensors. Periodically return
            # cached CUDA blocks while transferring the full H, but also let
            # the caller refuse before the next unit if reserved blocks stay
            # resident. Python-reference release alone cannot prove fit.
            released_cuda_bytes = 0
            for name in list(hess):
                if resource_check is not None:
                    resource_check(f'before_hessian_cpu_transfer:{name}')
                h = hess.pop(name)
                hessians[name] = None if h is None else h.to(device='cpu')
                if h is not None and h.is_cuda:
                    released_cuda_bytes += h.numel() * h.element_size()
                h = None
                if released_cuda_bytes >= 256 * 1024**2:
                    torch.cuda.empty_cache()
                    released_cuda_bytes = 0
                if resource_check is not None:
                    resource_check(f'after_hessian_cpu_transfer:{name}')
            if released_cuda_bytes:
                torch.cuda.empty_cache()
        if resource_check is not None:
            resource_check('after_output_materialization')
    except BaseException as error:
        # Exception tracebacks keep this frame alive. Explicitly drain every
        # output owner and loop temporary while retaining only row counters.
        h = chunks = cpu_rows = cpu_h = None
        store.clear()
        hess.clear()
        amax.clear()
        rows.clear()
        hessians.clear()
        if (resource_check is not None or on_forwards_complete is not None) and torch.device(device).type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                error.add_note(f'capture CUDA cache release failed: {cleanup_error!r}')
        raise
    return CalibrationStatistics(rows, hessians, dict(seen), dict(amax))
