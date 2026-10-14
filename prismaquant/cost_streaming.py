"""Streamed causal-LM forwards over the existing model residency owner."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import re
from types import SimpleNamespace
from typing import Any, Iterator, Protocol, runtime_checkable

import torch

from prismaquant.memory_management import reserve_allocation
from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)


from .streamed_boundary_artifacts import StreamedForwardBoundaries, prefetched_boundary_batches

@runtime_checkable
class SelectedSource(Protocol):
    """A source a selected-source consumer snapshots weights from.

    The body's ``StreamedCausalLM`` satisfies it, and so does one built over a
    profile-declared source scope (``build_streamed_causal_lm(source_scope=)``),
    which has no body forward. ``SELECTED_SOURCE_MEMBERS`` lists these members; a
    test pins it and keeps the campaign's ``runner.`` uses inside it.
    """

    num_layers: int
    context: Any

    @property
    def source_layers(self) -> tuple[int, ...]: ...

    def layer_index_for_qname(self, qname: str) -> int: ...

    def selected_weight_specs(self, names) -> dict: ...

    def snapshot_selected_weights(self, names, *, max_resident_bytes: int, **kwargs): ...

    def shutdown(self) -> None: ...


class StreamedCausalLM:
    """Causal-LM forward adapter over an existing ``StreamingContext``.

    ``pin_layer_for_qname`` keeps exactly one decoder layer installed while a
    caller temporarily mutates and restores a serving unit in it.  All other
    layers continue to stream through the context's cache.  This is the seam
    used by empirical expert KL; AURA additionally consumes the explicit
    boundary/isolated-layer methods for its streamed adjoint.
    """

    def __init__(
        self,
        context,
        profile,
        *,
        prefetch_lookahead: int = 2,
        require_prefetched_residency: bool = False,
    ):
        if type(require_prefetched_residency) is not bool:
            raise TypeError(
                "require_prefetched_residency must be a bool"
            )
        self.context = context
        self.model = context.model
        self.base_model = context.base_model
        self.layers = context.layers
        self.layers_prefix = str(context.layers_prefix)
        self.num_layers = int(context.num_layers)
        self.device = torch.device(context.device)
        self.dtype = context.dtype
        self.profile = profile
        self.prefetch_lookahead = max(0, int(prefetch_lookahead))
        self.require_prefetched_residency = require_prefetched_residency
        self._pinned_layer: int | None = None
        # Layers whose pre-install re-assert had to issue a fresh source read
        # during the last exact layer-major traversal (#403). Empty when every
        # speculative prefetch was still held by the runner at install time.
        self.layer_major_prefetch_retries: tuple[int, ...] = ()

    @property
    def source_layers(self) -> tuple[int, ...]:
        """The checkpoint layers this runner's context can read."""
        return tuple(getattr(self.context, "source_layers", range(self.num_layers)))

    def layer_index_for_qname(self, qname: str) -> int:
        match = re.match(
            rf"^{re.escape(self.layers_prefix)}([0-9]+)(?:\.|$)",
            str(qname),
        )
        if match is None:
            raise RuntimeError(
                f"streamed cost unit {qname!r} is not under decoder prefix "
                f"{self.layers_prefix!r}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.num_layers or layer not in self.source_layers:
            raise RuntimeError(
                f"streamed cost unit {qname!r} resolved invalid layer {layer}"
            )
        return layer

    def selected_weight_specs(self, names):
        """``{unit: (shape, dtype, nbytes)}`` of each selected source Linear, with no I/O.

        The shape comes from the skeleton's (meta) parameter view and the dtype
        from the one the layer install would materialize it in, so these are the
        values a snapshot copy carries. The streaming row head reads a projected
        expert's source tensor on its reader threads instead of snapshotting its
        layer, and checks what it read against this (PQ #1654).
        """
        from .routed_experts import profile_declared_packed_expert_projections

        modules = dict(self.model.named_modules())
        projected = {member.qname: member for member in
                     profile_declared_packed_expert_projections(self.model, self.profile)}
        specs = {}
        for name in sorted(names):
            if name in projected:
                weight = projected[name].weight
                parameter_name = projected[name].module_qname+'.'+projected[name].param_name
            elif self.profile.is_dense_parameter_owner(name, modules.get(name)):
                weight = modules[name].weight
                parameter_name = name+'.weight'
            else:
                raise RuntimeError(f"selected source unit is not a declared Linear: {name}")
            dtype = getattr(self.context, 'buffer_dtypes', {}).get(parameter_name, self.dtype)
            specs[name] = (tuple(weight.shape), dtype,
                           weight.numel()*torch.empty((), dtype=dtype).element_size())
            del weight
        return specs

    def snapshot_selected_weights(self, names, *, max_resident_bytes: int,
                                  resource_check=None, expected_source_keys=None,
                                  host=False):
        """Copy selected source Linears from the existing resident layer cache.

        This is preparation for a consumer that already owns its ``weights``
        mapping and needs no source forward. The finite layer sequence is
        prefetched through StreamingContext; ordinary adjacent-layer top-up
        is disabled so a sparse selection never reads unrelated layers.
        Independent copies prevent an expert view from pinning its complete
        packed parent after the source layer has been released.

        ``host=True`` makes each copy a contiguous CPU tensor. The streaming
        row head hashes weights on reader threads while the encode thread owns
        the device, so its weights must not be device tensors.
        """
        from .routed_experts import (
            profile_declared_packed_expert_projections,
            refresh_packed_expert_projections,
        )

        names = tuple(names)
        if not names or len(set(names)) != len(names):
            raise ValueError("selected source requires unique nonempty unit names")
        if type(max_resident_bytes) is not int or max_resident_bytes <= 0:
            raise ValueError("selected source requires a positive resident byte budget")
        if self._pinned_layer is not None:
            raise RuntimeError("selected source cannot start with a pinned layer")
        modules = dict(self.model.named_modules())
        projected = {member.qname: member for member in
                     profile_declared_packed_expert_projections(self.model, self.profile)}
        shapes = self.selected_weight_specs(names)
        layers = {}
        for name in sorted(names):
            layers.setdefault(self.layer_index_for_qname(name), []).append(name)
        required = sum(shape[2] for shape in shapes.values())
        if required > max_resident_bytes:
            raise RuntimeError("selected source weights exceed their resident byte budget")

        snapshot_only = getattr(self.context, 'source_snapshot_only', False)
        if snapshot_only:
            if expected_source_keys is None:
                raise RuntimeError('snapshot requires admitted source keys before source I/O')
            from .layer_streaming import selected_weight_source_keys
            planned_keys = tuple(expected_source_keys)
            actual_keys = selected_weight_source_keys(names, self.profile, self.context.weight_ckpt)
            if planned_keys != actual_keys:
                raise RuntimeError('snapshot source dependencies differ from the admitted plan')
            self.context.configure_selected_snapshot(names, self.profile)

        ordered = sorted(layers)
        # `max_cache_slots` is None when the layer cache is bounded by bytes and
        # carries no slot cap -- the ordinary autoscaled case, and what every
        # other reader spells out (`StreamingContext.suggest_prefetch_lookahead`).
        # This one subtracted from it, so a sparse selected-source walk on an
        # autoscaled cache died on None - 1 instead of walking. With no cap the
        # window is the caller's lookahead; with one it keeps a slot for the
        # layer `install()` still owns.
        slots = self.context.max_cache_slots
        window = max(1, self.prefetch_lookahead if slots is None
                     else min(self.prefetch_lookahead, slots - 1))
        weights, records = {}, []
        for layer in ordered[:window]:
            self.context.schedule_prefetch(layer)
        for index, layer in enumerate(ordered):
            source = self.context.install(layer, require_prefetched=True,
                                          prefetch_following=False,
                                          **({'snapshot': True} if snapshot_only else {}))
            if index + window < len(ordered):
                self.context.schedule_prefetch(ordered[index + window])
            live = {}
            try:
                live = {member.qname: member for member in refresh_packed_expert_projections(
                    [projected[name] for name in layers[layer] if name in projected], self.profile)}
                with torch.no_grad():
                    for name in layers[layer]:
                        value = live[name].weight if name in live else modules[name].weight
                        shape, dtype, nbytes = shapes[name]
                        if value.is_meta or tuple(value.shape) != shape or value.dtype != dtype:
                            raise RuntimeError(f"selected source has wrong resident tensor: {name}")
                        if resource_check is not None:
                            # Where this copy lands decides whose budget it is:
                            # the ``host`` arm materializes on the CPU, and the
                            # other arm clones the resident tensor on whatever
                            # device it already lives on. Charging a device
                            # clone to the cgroup cap is the same conflation as
                            # the capture path's, one call site over.
                            reserve_allocation(
                                resource_check,
                                f"before_selected_source_copy:{name}",
                                cpu_bytes=nbytes if host else 0,
                                device_bytes=0 if host else nbytes)
                        if host:
                            copy = torch.empty(shape, dtype=dtype, device="cpu")
                            copy.copy_(value.detach())
                            weights[name] = copy
                            del copy
                        else:
                            weights[name] = value.detach().clone(
                                memory_format=torch.contiguous_format)
                        del value
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            finally:
                live.clear()
                self.context.release_completed_layer(layer)
            records.append(dict(layer=layer, units=layers[layer], source=source))
            if resource_check is not None:
                resource_check(f"after_selected_source_release:{layer}")
        return weights, dict(schema="prismaquant.selected_source_weights.v1",
            units=sorted(weights), layers=records, resident_bytes=required,
            source_forward_count=0, packed_parent_storage_retained=False,
            **({'source_snapshot_policy': 'selected-tensors-v1',
                'source_tensor_keys': list(self.context._snapshot_source_keys),
                'nonbody_materialized': False} if snapshot_only else {}))

    def observe_source_waits(self, sink):
        """Scope source-delivery observation through the existing residency owner."""
        return self.context.observe_source_waits(sink)

    @contextmanager
    def pin_layer(self, layer: int) -> Iterator[None]:
        layer = int(layer)
        if self._pinned_layer is not None:
            raise RuntimeError(
                f"streamed cost already pins layer {self._pinned_layer}; "
                f"cannot also pin layer {layer}"
            )
        self.context.install(layer)
        self._pinned_layer = layer
        try:
            yield
        finally:
            self._pinned_layer = None
            self.context.unload(layer)

    @contextmanager
    def pin_layer_for_qname(self, qname: str) -> Iterator[None]:
        with self.pin_layer(self.layer_index_for_qname(qname)):
            yield

    def _head(self):
        name = str(self.profile.lm_head_name())
        try:
            return self.model.get_submodule(name)
        except (AttributeError, KeyError):
            head = getattr(self.model, "lm_head", None)
            if head is None:
                raise RuntimeError(
                    f"streamed cost could not resolve profile lm_head {name!r}"
                )
            return head

    def _release_source_pages_before(self, next_layer: int, stop: int):
        """A recording source owner's bounded page policy (PQ #1896); a no-op otherwise.

        Only a streamed capture's runner reads through a recording owner, and
        only its context drops pages here (``release_source_pages_before``).
        """
        owner = getattr(self.context, "source_authentication", None)
        if getattr(owner, "is_recording", False):
            self.context.release_source_pages_before(next_layer, stop)

    def _prepare(self, input_ids: torch.Tensor):
        if getattr(self.context, 'source_snapshot_only', False):
            raise RuntimeError('snapshot-only source cannot execute a forward')
        if getattr(self.context, 'source_scope', None) is not None:
            raise RuntimeError(f'source scope {self.context.source_scope!r} cannot execute a forward')
        ids = input_ids.to(self.device)
        position_ids = torch.arange(
            ids.size(-1), device=self.device
        ).unsqueeze(0)
        hidden = self.base_model.embed_tokens(ids).to(self.dtype)
        position_embeddings = _compute_position_embeddings(
            self.base_model, hidden, position_ids, self.profile
        )
        attention_mask = _compute_attention_mask(
            self.base_model, hidden, position_ids
        )
        hidden = self.profile.expand_hidden_for_layers(
            hidden, self.base_model
        )
        return ids, position_ids, hidden, position_embeddings, attention_mask

    def _call(self, layer: int, hidden: torch.Tensor, *, batch, pass_state):
        return _call_layer(
            self.layers[layer],
            hidden,
            position_embeddings=batch.position_embeddings,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            **self.profile.extra_layer_kwargs(input_ids=batch.input_ids),
            pass_state=pass_state,
        )

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.profile.collapse_hidden_after_layers(
            hidden, self.base_model
        )
        norm = _get_final_norm(self.base_model)
        if norm is not None:
            hidden = norm(hidden)
        return self._head()(hidden)

    def capture_boundaries(
        self, input_ids: torch.Tensor, *, boundary_writer=None, resource_check=None,
    ) -> StreamedForwardBoundaries:
        """Stream a no-grad source forward and retain only boundary acts."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        if resource_check is not None:
            resource_check(batch, pass_state)
        batch.activations_cpu.append(hidden.detach().to("cpu") if boundary_writer is None
                                     else boundary_writer(0, hidden))
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                with torch.no_grad():
                    hidden = self._call(
                        layer, hidden, batch=batch, pass_state=pass_state
                    )
                if resource_check is not None:
                    resource_check(batch, pass_state)
                batch.activations_cpu.append(hidden.detach().to("cpu") if boundary_writer is None
                                             else boundary_writer(layer + 1, hidden))
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        batch.shared_pass_state = self.profile.capture_forward_pass_state(
            pass_state
        )
        if resource_check is not None:
            resource_check(batch, pass_state)
        return batch

    def isolated_layer(
        self,
        batch: StreamedForwardBoundaries,
        layer: int,
        hidden: torch.Tensor,
        *,
        pass_state: dict | None,
    ) -> torch.Tensor:
        return self._call(layer, hidden, batch=batch, pass_state=pass_state)

    def tail_logits(
        self, batch: StreamedForwardBoundaries, hidden: torch.Tensor
    ) -> torch.Tensor:
        return self._finish(hidden)

    def schedule_reverse_prefetch(self, layer: int):
        """Prefetch the next layer in this runner's reverse traversal.

        This is the reverse-direction twin of the forward loops' explicit
        ``schedule_prefetch(layer + lookahead)`` call.  Residency remains
        owned by the existing :class:`StreamingContext` / ``LayerCache``;
        this method only supplies the traversal direction so reverse AURA
        can overlap the next source-layer read with the current layer's
        render and backward work.
        """
        target = int(layer) - self.prefetch_lookahead
        if self.prefetch_lookahead <= 0 or target < 0:
            return None
        return self.context.schedule_prefetch(target)

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any):
        """Run an end-to-end no-cache forward while streaming body layers."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(
                    layer,
                    require_prefetched=self.require_prefetched_residency,
                )
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                hidden = self._call(
                    layer, hidden, batch=batch, pass_state=pass_state
                )
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        return SimpleNamespace(logits=self.tail_logits(batch, hidden))

    def shutdown(self) -> None:
        self.context.shutdown()

    def capture_layer_major_boundaries(self, input_batches, *, storage, source_phase=None,
                                       forward_recovery=None, batch_offset=0):
        """Capture exact baseline boundaries through the existing layer visitor.

        ``batch_offset`` is the global index of ``input_batches[0]``: a
        forward split quantum captures one contiguous range of the run's
        calibration partitions and names every entry by its global batch, so
        its entries are the single owner's (PQ #738).
        """
        if storage.config.get("capture_order") != "layer_major":
            raise ValueError("layer-major boundary capture requires the explicit v2 policy")
        input_batches = tuple(input_batches)
        def visit(_layer, forward_batch):
            for input_ids in input_batches:
                forward_batch(input_ids)
        return self.visit_layer_batches(input_batches, visit, boundary_storage=storage,
                                        source_phase=source_phase, forward_recovery=forward_recovery,
                                        batch_offset=batch_offset)

    def visit_layer_batches(self, input_batches, visitor, *, output_consumer=None,
                            boundary_storage=None, source_phase=None, forward_recovery=None,
                            batch_offset=0, start=None, stop_layer=None,
                            boundary_consumer=None):
        """Visit one resident source layer over the original ordered batches.

        The original visitor retains one current hidden tensor per batch. With
        explicit v2 boundary storage, it instead leases exact input windows and
        writes each original output through the existing activation owner. All
        per-batch source kwargs/pass-state remain independent and unchanged.
        The new path requires evaluation mode and observes Torch CPU plus this
        runner's CUDA RNG state around preparation/source calls. RNG-consuming
        sources refuse; this is not equivalence for arbitrary stateful models.

        A capture chain quantum (PQ #1885) visits ``[start.layer, stop_layer)``
        on the in-memory path: ``start`` supplies each batch's hidden at its
        boundary, in the dtype it was stored in, and ``boundary_consumer(index,
        hidden)`` receives each batch's hidden after the last visited layer.
        Nothing at or past ``stop_layer`` is read.
        """
        if self._pinned_layer is not None:
            raise RuntimeError("layer-batch traversal cannot start with a pinned layer")
        exact = boundary_storage is not None
        if exact and (start is not None or stop_layer is not None
                      or boundary_consumer is not None):
            raise ValueError("a layer frontier, stop layer or boundary consumer is the "
                             "in-memory path's; exact storage writes every boundary")
        stop = self.num_layers if stop_layer is None else stop_layer
        start_layer = 0 if start is None else start.layer
        if (type(start_layer) is not int or type(stop) is not int
                or not 0 <= start_layer < stop <= self.num_layers
                or (start is not None and start_layer == 0)):
            raise ValueError("a layer chain visits 0 <= start < stop <= num_layers, "
                             "and a frontier starts past layer 0")
        if output_consumer is not None and stop != self.num_layers:
            raise ValueError("tail logits need the last layer; a stopped traversal has none")
        bounded = stop != self.num_layers or start is not None
        if start is not None:
            from .joint_forward_resume import require_stateless_profile
            require_stateless_profile(self)
        frontier = None if start is None else iter(start.hidden_batches())
        if type(batch_offset) is not int or batch_offset < 0:
            raise ValueError("a layer visitor's batch offset is a nonnegative integer")
        if batch_offset and (not exact or forward_recovery is not None):
            # Only exact entries carry a global batch coordinate, and a
            # forward-recovery capsule lends the whole run's partitions.
            raise ValueError("a batch offset needs exact boundaries and no forward recovery")
        if forward_recovery is not None:
            from .joint_forward_resume import require_stateless_profile
            require_stateless_profile(self)
            if not exact or len(input_batches) != forward_recovery.n_batches:
                raise RuntimeError("forward recovery changed the calibration partition count")
            start_layer = forward_recovery.frontier
            if not 0 < start_layer <= self.num_layers:
                raise RuntimeError("forward recovery frontier is outside this model")
        if source_phase is not None and (not exact or not callable(source_phase)):
            raise ValueError('source phase admission requires exact boundaries and a callable observer')
        if exact:
            if boundary_storage.config.get("capture_order") != "layer_major":
                raise ValueError("layer visitor exact storage requires layer_major v2 policy")
            if not self.require_prefetched_residency:
                raise RuntimeError("layer-major capture requires prefetched source residency")
            if any(module.training for module in self.model.modules()):
                raise RuntimeError("layer-major capture requires evaluation mode")
        states, batches = [], []
        if exact:
            boundary_storage.watch_auxiliary(batches, [])

        def checked(function, *args, **kwargs):
            if not exact:
                return function(*args, **kwargs)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            result = function(*args, **kwargs)
            if (not torch.equal(cpu_rng, torch.get_rng_state()) or
                    (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.device)))):
                raise RuntimeError("layer-major capture observed source Torch RNG consumption")
            return result

        def check_state():
            if exact:
                boundary_storage.check_auxiliary(batches,
                    extra=[state[2] for state in states],
                    shared_extra=[state[2] for batch, state in zip(batches, states)
                                  if batch.shared_pass_state is None])

        def report_source_phase(stage, layer):
            if source_phase is not None:
                source_phase(stage, layer, boundary_storage.actual_auxiliary_bytes(
                    batches, extra=[state[2] for state in states]))

        # The runner owns residency; this visitor keeps no residency state of
        # its own. `schedule_prefetch` is idempotent: it returns the same future
        # for a read it already holds (in flight or delivered and unclaimed),
        # owns a hot layer through a completed future until its install
        # (#1124), and submits a fresh read only when nothing is held. Each
        # layer is speculated once ahead of its turn and re-asserted once
        # immediately before its install, so a speculation the runner no
        # longer holds (the pressure floor refused the read, or a hot layer
        # left the cache before the schedule owned it) gets one bounded retry and the
        # `require_prefetched` refusal stays fail-closed for anything else (#403).
        # The speculation record is the runner's future, whose result is the
        # layer's tensors; it is released at re-assert, the same moment the
        # runner drops its own reference at install, so this visitor is never
        # a second owner of a claimed layer's source bytes.
        speculated: dict[int, object] = {}
        retried: list[int] = []
        unspeculated = object()

        def speculate(layer):
            if 0 <= layer < self.num_layers and layer not in speculated:
                speculated[layer] = self.context.schedule_prefetch(layer)

        def reassert(layer):
            previous = speculated.pop(layer, unspeculated)
            held = self.context.schedule_prefetch(layer)
            if previous is not unspeculated and held is not None and held is not previous:
                retried.append(layer)

        if exact:
            self.layer_major_prefetch_retries = ()
        try:
            with torch.no_grad():
                for batch_index, input_ids in enumerate(input_batches):
                    check_state()
                    ids, positions, hidden, embeddings, mask = checked(self._prepare, input_ids)
                    pass_state = checked(self.profile.new_forward_pass_state)
                    if exact:
                        batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], None)
                        batches.append(batch)
                        states.append([ids, None, pass_state])
                        check_state()
                        if forward_recovery is None:
                            batch.activations_cpu.append(boundary_storage.write(hidden,
                                batch_index=batch_offset + batch_index, boundary_index=0))
                        else:
                            refs = forward_recovery.batch_references(batch_index)
                            if any(ref.shape != tuple(hidden.shape) or ref.dtype != str(hidden.dtype)
                                   for ref in refs):
                                raise RuntimeError("forward recovery boundary tensor geometry differs")
                            batch.activations_cpu.extend(refs)
                        del hidden, pass_state
                    else:
                        if frontier is not None:
                            stored = next(frontier, None)
                            if stored is None:
                                raise RuntimeError("the layer frontier has fewer batches than the draw")
                            if stored.dtype != self.dtype:
                                raise RuntimeError(
                                    f"the layer frontier stores {stored.dtype}, "
                                    f"not this runner's {self.dtype}")
                            hidden = stored.to(device=self.device)
                            del stored
                        states.append([ids, hidden, pass_state])
                    del positions, embeddings, mask
                if frontier is not None and next(frontier, None) is not None:
                    raise RuntimeError("the layer frontier has more batches than the draw")
                if not states:
                    raise ValueError("layer-batch traversal requires calibration batches")
                if exact:
                    if start_layer < self.num_layers:
                        report_source_phase('source_loading', start_layer)
                    elif forward_recovery is not None:
                        report_source_phase('capture_forward', self.num_layers - 1)
                    for depth in range(start_layer, min(self.num_layers,
                            start_layer + max(1, self.prefetch_lookahead))):
                        speculate(depth)
                else:
                    for depth in range(start_layer, min(stop, start_layer + self.prefetch_lookahead + 1)):
                        self.context.schedule_prefetch(depth)
                # A recording source owner (PQ #1896) retains each file's pages
                # from its hash for the reads that follow: the head's files go
                # now, each body file after the last layer that reads it.
                self._release_source_pages_before(start_layer, stop)
                for layer in range(start_layer, stop):
                    if layer > start_layer:
                        report_source_phase('source_loading', layer)
                    if exact:
                        check_state()
                        reassert(layer)
                        self.context.install(layer, require_prefetched=True, prefetch_following=False)
                    elif bounded:
                        # The runner's own top-up would read past ``stop``. The
                        # explicit window below is bounded instead, and this
                        # idempotent re-assert retries a refused speculation
                        # once, as the exact path's ``reassert`` does.
                        self.context.schedule_prefetch(layer)
                        self.context.install(layer, require_prefetched=self.require_prefetched_residency,
                                             prefetch_following=False)
                    else:
                        self.context.install(layer, require_prefetched=self.require_prefetched_residency)
                    try:
                        if exact:
                            speculate(layer + self.prefetch_lookahead)
                        elif not bounded:
                            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
                        elif layer + self.prefetch_lookahead < stop:
                            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
                        if source_phase is not None:
                            # Loading may retain old source and packing buffers;
                            # do not overlap that admitted phase with the graph
                            # workspace. The existing source owner still owns
                            # lookahead and its storage.
                            self.context.settle_prefetched_layers(range(
                                layer + 1, min(self.num_layers, layer + self.prefetch_lookahead + 1)),
                                retry_availability=True)
                            report_source_phase('capture_forward', layer)
                        next_batch = 0
                        with prefetched_boundary_batches(boundary_storage, batches, layer) if exact else nullcontext() as resident:
                            def forward_batch(input_ids):
                                nonlocal next_batch
                                if next_batch >= len(states):
                                    raise RuntimeError("layer visitor repeated a calibration batch")
                                ids, hidden, pass_state = states[next_batch]
                                if not torch.equal(input_ids.to(device=ids.device), ids):
                                    raise RuntimeError("layer visitor changed calibration batch order or tokens")
                                if exact:
                                    index, batch, cpu_hidden, _unused = next(resident)
                                    if index != next_batch:
                                        raise RuntimeError("exact boundary window changed batch order")
                                    hidden = cpu_hidden.to(device=self.device, dtype=self.dtype)
                                else:
                                    _ids, positions, initial, embeddings, mask = self._prepare(ids)
                                    del initial
                                    batch = StreamedForwardBoundaries(_ids, positions, embeddings, mask, [], None)
                                try:
                                    output = checked(self._call, layer, hidden, batch=batch, pass_state=pass_state)
                                    if exact:
                                        check_state()
                                        batch.activations_cpu.append(boundary_storage.write(output,
                                            batch_index=batch_offset + next_batch,
                                            boundary_index=layer + 1))
                                    else:
                                        states[next_batch][1] = output
                                    next_batch += 1
                                finally:
                                    hidden = output = None
                                    if exact:
                                        cpu_hidden = None
                            visitor(layer, forward_batch)
                            if next_batch != len(states):
                                raise RuntimeError("layer visitor omitted calibration batches")
                    finally:
                        self.context.unload(layer)
                    self._release_source_pages_before(layer + 1, stop)
                if exact:
                    for batch, state in zip(batches, states):
                        batch.shared_pass_state = checked(self.profile.capture_forward_pass_state, state[2])
                        check_state()  # Charge the CPU capture and still-live original state together.
                        state[2] = None
                    check_state()
                    if output_consumer is not None:
                        with prefetched_boundary_batches(boundary_storage, batches, self.num_layers) as resident:
                            for index, batch, cpu_hidden, _unused in resident:
                                hidden = cpu_hidden.to(device=self.device, dtype=self.dtype)
                                output_consumer(index, checked(self.tail_logits, batch, hidden))
                                del hidden, cpu_hidden
                    return batches
                if output_consumer is not None:
                    for index, (ids, hidden, _pass_state) in enumerate(states):
                        _ids, positions, initial, embeddings, mask = self._prepare(ids)
                        del initial
                        batch = StreamedForwardBoundaries(_ids, positions, embeddings, mask, [], None)
                        output_consumer(index, self.tail_logits(batch, hidden))
                if boundary_consumer is not None:
                    for index, state in enumerate(states):
                        boundary_consumer(index, state[1])
                        state[1] = None
        finally:
            states.clear()
            if frontier is not None and hasattr(frontier, "close"):
                frontier.close()
            if exact:
                self.layer_major_prefetch_retries = tuple(retried)


def build_streamed_causal_lm(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offload_folder: str,
    profile,
    cache_headroom_gb: float | None = None,
    max_cache_slots: int | None = None,
    prefetch_workers: int | str | None = None,
    prefetch_min_available_gb: float | str | None = None,
    prefetch_lookahead: int = 2,
    require_prefetched_residency: bool = False,
    attn_implementation: str | None = None,
    source_authentication=None,
    source_derivative=None,
    source_snapshot_only=False,
    sealed_head_tensors=None,
    planned_source_window_bytes: int | None = None,
    source_scope: str | None = None,
) -> StreamedCausalLM:
    """Build the repository's existing streaming context and wrap it.

    ``sealed_head_tensors`` is the resident head a read manifest declared;
    the context refuses before its first head read when its own selection
    differs (PQ #1095). None checks nothing, as before.

    ``planned_source_window_bytes`` bounds the prefetch note by the sealed
    plan's source window (PQ #1134); None leaves the note as before.

    ``source_scope`` names a profile-declared out-of-body source (PQ #1316).
    The runner reads and installs only the scope's layers and never runs the
    body forward (PQ #1338). None builds the body, as before.
    """
    from prismaquant.streaming_model import _build_streaming_context

    context = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=offload_folder,
        cache_headroom_gb=cache_headroom_gb,
        max_cache_slots=max_cache_slots,
        prefetch_workers=prefetch_workers,
        prefetch_min_available_gb=prefetch_min_available_gb,
        log_prefix="[cost-streaming]",
        attn_implementation=attn_implementation,
        **({'source_authentication': source_authentication} if source_authentication is not None else {}),
        **({'source_snapshot_only': True} if source_snapshot_only else {}),
        **({'sealed_head_tensors': sealed_head_tensors}
           if sealed_head_tensors is not None else {}),
        **({'planned_source_window_bytes': planned_source_window_bytes}
           if planned_source_window_bytes is not None else {}),
        **({'source_scope': source_scope} if source_scope is not None else {}),
    )
    runner = None
    try:
        effective_lookahead = max(0, int(prefetch_lookahead))
        if context.max_cache_slots is not None:
            effective_lookahead = min(
                effective_lookahead,
                max(0, context.max_cache_slots - 1),
            )
        runner = StreamedCausalLM(
            context,
            profile,
            prefetch_lookahead=effective_lookahead,
            require_prefetched_residency=require_prefetched_residency,
        )
        from .glm_source_derivative import bind_source_derivative
        bind_source_derivative(runner.model, profile, source_derivative)
    except BaseException:
        (context if runner is None else runner).shutdown()
        raise
    return runner
