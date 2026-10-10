"""Bounded body-layer row replay using the existing sealed source transport.

This is a local mechanism discriminator, never a whole-draw price or source
initialization attestation. Original inputs require Astra's accepted action
protocol; --prepare-synthetic creates only seeded, small fixture inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from safetensors.torch import save_file
from transformers.models.glm5_next import Glm5NextConfig

from experiments.alloc_lead_asd.glm_row_binding import RoutingTap
from experiments.alloc_lead_asd.qualify_tiny_glm import DERIVATIVE, fixture_module
from prismaquant import format_registry
from prismaquant.cost_streaming import StreamedCausalLM, StreamedForwardBoundaries
from prismaquant.glm_source_derivative import bind_source_derivative
from prismaquant.joint_aura import SignedJointProjectionLease, activation_identity, select_invocation_gradient
from prismaquant.joint_projection_backend import normalize_projection_backend, prewarm_projection_backend
from prismaquant.memory_management import CaptureMemoryGuard, allocator_device, enforce_device_envelope, reserve_allocation
from prismaquant.layer_streaming import (
    LayerCache, _build_concat_merger, _build_expert_packer, _build_install_resolver,
    _compute_attention_mask, _compute_position_embeddings, live_weight_map,
)
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.perturbed_x_cache import _activation_qdq, write_exact_activation_cache_entry
from prismaquant.residency_map import bind_residency_manifest, residency_report, residency_resolver
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_sealed_file, read_staged_whole_file
from prismaquant.streaming_model import StreamingContext, build_streaming_skeleton


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open('w') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def progress(units, phase):
    from prismaquant.prismabuild_progress import commit
    commit(units, phase, 'durable_body_layer_local_control')


def save_tensors(path, value):
    torch.save(value, path)
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


LEGACY_L05_SCOPE = 'original L05 local mechanism control'
REPLAY_SCOPE_PREFIX = 'original L'
REPLAY_SCOPE_SUFFIX = ' full-draw replay slice'
PUBLISHER_REVISION = 'a6c167b62691b2bac901344b65cb651a70f53e43'
AUXILIARY_DIGESTS = {'config.json': '33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943',
                     'model.safetensors.index.json': 'e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f'}
REPLAY_LAYERS = (5, 9)
REPLAY_SLICE_LENGTHS = (4, 8, 16, 32)


def replay_slice_scope(layer, sequences):
    return f'{REPLAY_SCOPE_PREFIX}{layer:02d}{REPLAY_SCOPE_SUFFIX} {sequences[0]:03d}-{sequences[-1]:03d}'


def validate_original_binding(binding, args):
    if binding['scope'] == 'synthetic only':
        return False
    layer = binding.get('layer')
    sequences = binding.get('sequences')
    if binding['scope'] == LEGACY_L05_SCOPE:
        if layer != 5 or sequences != [0, 1, 2, 3]:
            raise RuntimeError('legacy local control scope differs from its accepted contract')
    elif isinstance(layer, int) and layer in REPLAY_LAYERS and isinstance(sequences, list) and sequences:
        if (sequences != list(range(sequences[0], sequences[-1] + 1))
                or sequences[0] < 0 or sequences[-1] > 511
                or len(sequences) not in REPLAY_SLICE_LENGTHS
                or binding['scope'] != replay_slice_scope(layer, sequences)):
            raise RuntimeError('replay slice scope differs from its layer and sequence range')
    else:
        raise RuntimeError('original local control lacks a known replay scope')
    expected = dict(publisher_revision=PUBLISHER_REVISION,
        sequence_length=512, n_probes=4, global_token_count=262144,
        probe_seed_base=7000, temperature=1.0, loss_positions='all',
        cotangents_are_banked_fixed_inputs=True, dtype='torch.bfloat16',
        activation_format='TESSERA_E4M3_K1_R1024')
    keys = binding.get('layer_keys', [])
    prefix = f'model.language_model.layers.{layer}.'
    if (not getattr(args, 'original_local_control', False) or args.device != 'cuda'
            or any(binding.get(name) != value for name, value in expected.items())
            or not keys or len(set(keys)) != len(keys)
            or any(not isinstance(name, str) or not name.startswith(prefix) for name in keys)
            or not args.projection_contract
            or binding.get('projection_contract') != dict(path=str(args.projection_contract),
                                                        sha256=args.projection_contract_sha256)):
        raise RuntimeError('original local control lacks the exact accepted contract and explicit opt-in')
    fact_units = binding.get('fact_units', [])
    units = {name.removesuffix('.weight') for name in keys if name.endswith('.weight')}
    if (not isinstance(fact_units, list) or any(unit not in units for unit in fact_units)):
        raise RuntimeError('fact units differ from the admitted layer roster')
    auxiliary = {row['name']: row['sha256'] for row in binding.get('metadata', [])}
    if auxiliary != AUXILIARY_DIGESTS:
        raise RuntimeError('original publisher auxiliary bytes differ from the accepted source')
    limits = binding.get('allocator_limits', {})
    for name, value in dict(cpu_cgroup_bytes=52 << 30, device_envelope_bytes=44 << 30,
            torch_allocator_bytes=40 << 30, native_device_allowance_bytes=4 << 30,
            aggregate_bytes=96 << 30).items():
        if type(limits.get(name)) is not int or limits[name] != value:
            raise RuntimeError('original allocator envelope differs from the reviewed protocol')
    return True


@contextmanager
def bounded_cuda_allocator(binding, device):
    limits = binding.get('allocator_limits')
    if limits is None:
        yield None
        return
    index = allocator_device(device)
    prior = torch.cuda.get_per_process_memory_fraction(index)
    try:
        record = enforce_device_envelope(device, limits['torch_allocator_bytes'], where='L05 local control')
        yield record
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.set_per_process_memory_fraction(prior, index)


def qualify_historical_activation_policy(contract):
    identities = contract.get('activation_identities', [])
    if len(identities) != 1:
        raise RuntimeError('historical diagnostic requires one exact resolved activation policy')
    expected = identities[0]
    spec = format_registry.get_format(contract['format'])
    maximum = expected['activation_max_abs']
    actual = activation_identity(spec, {'diagnostic': maximum}, 'diagnostic')
    if actual != expected:
        raise RuntimeError('historical activation callable/scale/clip policy differs')
    return expected


def record(path):
    raw = path.read_bytes()
    return dict(path=str(path), name=path.name, bytes=len(raw), sha256=sha(raw))


@contextmanager
def profiled(path, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile = torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True)
    try:
        with profile:
            yield
    finally:
        trace = Path(str(path) + '.trace.json')
        profile.export_chrome_trace(str(trace))
        with trace.open('rb') as source, gzip.open(str(trace) + '.gz', 'wb') as target:
            target.write(source.read())
        trace.unlink()
        write_json(Path(str(path) + '.profile.json'), dict(
            scope='operator allocation accounting; inclusive/cumulative bytes are not a physical peak',
            operators=[dict(name=event.key, calls=event.count,
                            self_cpu_memory_bytes=event.self_cpu_memory_usage,
                            inclusive_cpu_memory_bytes=event.cpu_memory_usage,
                            self_device_memory_bytes=event.self_device_memory_usage)
                       for event in profile.key_averages()]))


class _SealedDecoders:
    """Exact held decoder adapter adopted from #1934 source bf579d382dd3.

    Its safe_open body is unchanged from profile_expert_routing_1931.py.
    Authority remains the independently bound whole-file bytes; this class
    contributes no production admission or persistent cache.
    """
    def __init__(self, root, buffers):
        self.root, self.buffers = root, buffers

    @contextmanager
    def safe_open(self, factory, path, *args, **kwargs):
        from safetensors import safe_open
        path = Path(path)
        if path.parent != self.root or path.name not in self.buffers:
            raise RuntimeError('measurement decoder outside exact sealed source roster')
        if kwargs.get('device', 'cpu') != 'cpu':
            raise RuntimeError('source decoder must remain on the CPU')
        buffer = self.buffers[path.name]
        buffer.require_sealed()
        with safe_open(buffer.path, *args, **kwargs) as handle:
            yield handle
        buffer.require_sealed()


class _CopyReader:
    """Keep one native source alias and private staging through stream completion.

    A getter fences the preceding transfer before replacing that staging.
    This bounds transient pinning to one tensor per reader, rather than a
    complete shard. CPU results own copies and cannot retain a memfd alias.
    """
    def __init__(self, reader, transport):
        self.reader, self.transport = reader, transport
        self.held = []

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def finish(self):
        if self.held and self.transport.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.transport.device))
            event.synchronize()
            self.transport.copy_fences += 1
        self.held.clear()
        self.transport.guard.check('after_source_copy_fence')

    def get_tensor(self, name):
        self.finish()
        value = self.reader.get_tensor(name)
        dtype = self.transport.dtypes.get(name, self.transport.dtype)
        if self.transport.device.type == 'cuda':
            nbytes = value.numel() * torch.empty((), dtype=dtype).element_size()
            reserve_allocation(self.transport.guard.check, 'before_source_tensor_pin_and_copy',
                               cpu_bytes=nbytes * 2, device_bytes=nbytes)
            converted = value.to(dtype=dtype)
            result = converted.pin_memory()
            self.held = [value, converted, result]
            extra = result.numel() * result.element_size()
            if converted.untyped_storage().data_ptr() != value.untyped_storage().data_ptr():
                extra += converted.numel() * converted.element_size()
            self.transport.transient_staging_peak = max(self.transport.transient_staging_peak, extra)
        else:
            result = value.to(dtype=dtype, copy=True)
        return result


class ReplayDecoders(_SealedDecoders):
    """Bounded metadata/copy seam on the same #1934 held decoder transport."""
    def __init__(self, binding, device, dtype):
        super().__init__(Path(binding['model']), {})
        self.binding, self.device, self.dtype = binding, device, dtype
        self.metadata, self.all_buffers, self.readers, self.dtypes = {}, [], [], {}
        self.copy_fences = self.transient_staging_peak = 0
        limits = binding.get('allocator_limits')
        self.guard = CaptureMemoryGuard(device, **(dict(device_bytes=limits['device_envelope_bytes'],
            aggregate_envelope=True) if limits else {}))
        if limits and self.guard.cpu_cap_bytes != limits['cpu_cgroup_bytes']:
            raise RuntimeError('inner CPU cgroup differs from the reviewed physical envelope')

    def open(self):
        # Low-level27b owns staged leases, fill, actual digest and kernel seals.
        # Do not reread a pool path or derive authority from a descriptor/stat.
        for kind, rows in (('metadata', self.binding['metadata']), ('weights', self.binding['shards'])):
            for row in rows:
                self.guard.check('before_sealed_source:' + row['name'], reserve_bytes=row['bytes'] * 2)
                buffer = read_staged_sealed_file(Path(row['path']), row['sha256'], row['bytes'], label=row['name'])
                self.all_buffers.append(buffer)
                buffer.require_sealed()
                if kind == 'metadata':
                    self.metadata[Path(row['logical_path'])] = buffer
                    git_blob = row.get('publisher_git_blob')
                    if git_blob is not None:
                        with buffer.readonly() as handle:
                            raw = handle.read()
                        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                        if actual != git_blob:
                            raise RuntimeError('publisher auxiliary native Git object differs')
                else:
                    self.buffers[row['name']] = buffer
        return self

    def require_unchanged(self):
        for buffer in self.all_buffers:
            buffer.require_sealed()

    def read_json(self, path):
        buffer = self.metadata.get(Path(path))
        if buffer is None:
            raise RuntimeError('metadata read outside independently bound auxiliary roster')
        buffer.require_sealed()
        with buffer.readonly() as handle:
            raw = handle.read()
        from prismaquant.schemas import strict_json_loads
        return strict_json_loads(raw,
            duplicate=lambda key: RuntimeError('duplicate metadata key: ' + key),
            constant=lambda value: RuntimeError('invalid metadata constant: ' + value))

    @contextmanager
    def safe_open(self, factory, path, *args, **kwargs):
        with super().safe_open(factory, path, *args, **kwargs) as source:
            reader = _CopyReader(source, self)
            self.readers.append(reader)
            try:
                yield reader
            finally:
                # Failed completion leaves the owner/aliases held. PB containment
                # reaps the process; it must not falsely close decoder material.
                reader.finish()
                self.readers.remove(reader)

    def close(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if self.readers:
            raise RuntimeError('sealed transport still has live decoder readers')
        self.metadata.clear()
        self.buffers.clear()
        for buffer in reversed(self.all_buffers):
            buffer.close()
        self.all_buffers.clear()


class RowContractions(SignedJointProjectionLease):
    """Production signed terms plus fixed-cotangent rows with explicit coordinates."""
    def __init__(self, modules, moe, sequences, length, format_name, projection_backend=None, resource_check=None,
                 activation_policy=None, fact_units=()):
        self.routing = RoutingTap(moe.experts)
        self.sequences, self.length = tuple(sequences), length
        self.rows, self.pending, self.probe = [], {}, 0
        self.fact_units = frozenset(fact_units)
        self.facts = []
        self.resource_check = resource_check
        spec = format_registry.get_format(format_name)
        maxima = ({name: activation_policy['activation_max_abs'] for name in modules}
                  if activation_policy else {})
        deltas = {(name, spec.name): torch.zeros((), dtype=torch.float32, device=module.weight.device).expand(module.weight.shape)
                  for name, module in modules.items()}
        super().__init__(modules, {name: {spec.name: spec} for name in modules}, deltas,
                         projection_backend=projection_backend, activation_max_abs=maxima)
        self.spec = spec

    def _observe(self, name, weight, x, output, output_slice=None, row_slice=None):
        # Bound the imminent FP32 QDQ/cast/contraction temporaries. The CUDA
        # allocator envelope also holds unknown library/graph allocations.
        reserve_allocation(self.resource_check, 'before_unit_qdq_and_projection:' + name,
            cpu_bytes=x.numel() // x.shape[-1] * 40,
            device_bytes=12 * (x.numel() + output.numel()) * 4 + 3 * weight.numel() * 4)
        super()._observe(name, weight, x, output, output_slice, row_slice)
        if not output.requires_grad:
            raise RuntimeError('observed unit lacks its local backward graph')
        x2 = x.detach().reshape(-1, x.shape[-1])
        if '.experts.' in name:
            expert = int(name.split('.experts.')[1].split('.')[0])
            token, slot = self.routing.rows(expert, row_slice, len(x2))
            if name.endswith(('gate_proj', 'up_proj')) and not torch.equal(x2, self.routing.hidden[token]):
                raise RuntimeError('routed source activation differs from token identities')
        else:
            token = torch.arange(len(x2), device=x2.device)
            slot = torch.full_like(token, -1)
        sequence = torch.tensor(self.sequences, device=token.device)[token // self.length]
        coordinates = torch.stack((sequence, token % self.length, slot), -1).cpu()
        dx = _activation_qdq(x2, self.spec, self.activation_max_abs, name).float() - x2.float()
        dy = dx @ weight.detach().float().T
        if name in self.pending:
            raise RuntimeError('unit was observed twice in one local forward')
        self.pending[name] = (coordinates, dy)
        take_facts = name in self.fact_units
        fact_route = self.routing.weights[token, slot].detach().float().cpu() if take_facts and '.experts.' in name else None
        fact_amax = x2.float().abs().amax(-1).detach().cpu() if take_facts else None
        fact_token = token.detach().cpu() if take_facts else None
        def observed(gradient):
            selected = select_invocation_gradient(name, weight, x, gradient,
                output_slice=output_slice, row_slice=row_slice).detach().reshape(-1, dy.shape[-1])
            rows = (selected.double() * dy.double()).sum(-1)
            if not torch.isfinite(rows).all():
                raise RuntimeError('nonfinite fixed-cotangent row contraction')
            self.rows.append(dict(unit=name, probe=self.probe, coordinates=coordinates,
                                  signed_rows=rows.cpu(), fixed_g_dot=float(rows.sum())))
            if take_facts:
                if not torch.isfinite(selected).all() or not torch.isfinite(dy).all():
                    raise RuntimeError('nonfinite fact operands for recorded unit')
                selected64, dy64 = selected.double(), dy.double()
                g_norm = selected64.norm(dim=-1)
                along = rows / dy64.norm(dim=-1).clamp_min(1e-30)
                route = torch.ones(len(rows)) if fact_route is None else fact_route
                if len(route) != len(rows):
                    raise RuntimeError('route weights differ from recorded rows')
                for index in range(len(rows)):
                    coordinate = coordinates[index].tolist()
                    self.facts.append(dict(unit=name, probe=self.probe,
                        sequence=int(coordinate[0]), position=int(coordinate[1]), slot=int(coordinate[2]),
                        token=int(fact_token[index]),
                        route_weight=float(route[index]), input_amax=float(fact_amax[index]),
                        cotangent_norm=float(g_norm[index]), along_output=float(along[index]),
                        signed_row=float(rows[index])))
            return gradient
        output.register_hook(observed)

    def close(self):
        self.routing.remove()
        self.pending.clear()


def make_context(transport, layer, device, dtype, output):
    profile = Glm5NextProfile()
    config = Glm5NextConfig.from_dict(transport.read_json(transport.root / 'config.json'))
    config._experts_implementation = config.text_config._experts_implementation = 'grouped_mm'
    model = build_streaming_skeleton(config, multimodal=True, attn_implementation='eager')
    derivative = bind_source_derivative(model, profile, DERIVATIVE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    index = transport.read_json(transport.root / 'model.safetensors.index.json')
    shards, keys = live_weight_map(index['weight_map'], str(transport.root),
        lambda key: profile.checkpoint_to_live_name(key, multimodal=True))
    prefix = 'model.language_model.layers.'
    selected = tuple(sorted(key for key in keys if key.startswith(f'{prefix}{layer}.')))
    if tuple(selected) != tuple(sorted(transport.binding['layer_keys'])):
        raise RuntimeError('delivered index selection differs from admitted layer keys')
    base = model.model.language_model
    merger = _build_concat_merger(model, keys)
    context = StreamingContext(model=model, base_model=base, layers=base.layers,
        layers_prefix=prefix, num_layers=len(base.layers),
        install_resolvers=[_build_install_resolver(model, f'{prefix}{i}') for i in range(len(base.layers))],
        weight_shard=shards, weight_ckpt=keys,
        layer_cache=LayerCache(max_bytes=transport.binding['layer_cache_bytes'], max_entries=1),
        prefetch_pool=ThreadPoolExecutor(max_workers=1), device=device, dtype=dtype,
        offload_folder=str(output), prefetch_workers=1, source_authentication=transport,
        source_snapshot_only=True, source_layers=(layer,),
        expert_packer=_build_expert_packer(model, keys), concat_merger=merger)
    transport.dtypes = {keys[name]: dtype for name, dtype in context.buffer_dtypes.items() if name in keys}
    context.configure_selected_source_tensors(selected, layers=(layer,))
    # One deliberate source-cache fill precedes every graph. No other layer is
    # eligible, and install must find this layer already resident afterwards.
    context.ensure_loaded(layer)
    context.install(layer, require_prefetched=True, prefetch_following=False, snapshot=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()  # Only free blocks in this admitted process.
    if any(parameter.is_meta for parameter in base.layers[layer].parameters()):
        raise RuntimeError('selected isolated layer still has meta source slots')
    if not model.get_input_embeddings().weight.is_meta or not model.get_output_embeddings().weight.is_meta:
        raise RuntimeError('isolated replay materialized nonbody weights')
    runner = StreamedCausalLM(context, profile, prefetch_lookahead=0, require_prefetched_residency=True)
    return runner, derivative


def read_capture(entry):
    # Same staged whole-file primitive and same retained decoder lifetime;
    # metadata checks reproduce the existing exact-entry owner's payload gate.
    buffer = read_staged_sealed_file(Path(entry['path']), entry['sha256'], entry['file_bytes'], label=entry['name'])
    try:
        value = torch.load(buffer.path, mmap=True, map_location='cpu', weights_only=True)
        tensor = value.get('inputs') if isinstance(value, dict) else None
        if (not isinstance(tensor, torch.Tensor) or set(value) != {'inputs', 'name', 'source', 'exact'}
                or value['name'] != entry['name'] or value['source'] != 'exact_activation'
                or value['exact'] != entry['metadata'] or list(tensor.shape) != entry['shape']
                or str(tensor.dtype) != entry['dtype'] or not tensor.is_contiguous()
                or tensor.layout != torch.strided or tensor.requires_grad
                or tensor.untyped_storage().nbytes() != entry['tensor_bytes']
                or tensor.numel() * tensor.element_size() != entry['tensor_bytes']):
            raise RuntimeError('retained capture payload differs from exact-entry coordinates')
        result = tensor.clone()
        del tensor, value
        return result
    finally:
        tensor = value = None
        buffer.close()


def verify_installed_source_tensors(runner, transport):
    """Compare installed body values to the already-held source aliases.

    Virtual projections come from the existing profile owner, so this does
    not implement another packed-weight decoder or reread source files.
    This diagnostic is limited to the admitted synthetic body selection.
    """
    context = runner.context
    values = dict(runner.model.named_parameters())
    values.update(runner.model.named_buffers())
    values.update({member.qname + '.weight': member.weight for member in
                   profile_declared_packed_expert_projections(runner.model, runner.profile)})
    if set(context.weight_ckpt) != set(context.weight_shard):
        raise RuntimeError('selected source maps differ before installed-value check')
    concat_views = {}
    targets = getattr(getattr(context, 'concat_merger', None), 'source_targets', {})
    targets = {name: target for name, target in targets.items() if name in context.weight_ckpt}
    for target in sorted(set(targets.values())):
        if target not in values:
            raise RuntimeError('concat source lacks its installed target: ' + target)
        groups = [(suffix, sources, dim) for suffix, sources, dim in runner.profile.concat_merge_groups()
                  if target.endswith(suffix)]
        if len(groups) != 1:
            raise RuntimeError('concat source target lacks one declared profile group: ' + target)
        suffix, sources, dim = groups[0]
        prefix = target[:-len(suffix)]
        names = tuple(prefix + source for source in sources)
        if set(names) != {name for name, mapped in targets.items() if mapped == target}:
            raise RuntimeError('concat source selection differs from its complete profile group')
        installed = values[target]
        if not -installed.ndim <= dim < installed.ndim:
            raise RuntimeError('concat source has an invalid declared dimension')
        dim %= installed.ndim
        shapes = []
        for name in names:
            with _SealedDecoders.safe_open(transport, None, context.weight_shard[name],
                                          framework='pt', device='cpu') as reader:
                shape = tuple(reader.get_slice(context.weight_ckpt[name]).get_shape())
            if (len(shape) != installed.ndim or any(type(size) is not int or size <= 0 for size in shape)
                    or any(size != installed.shape[index] for index, size in enumerate(shape) if index != dim)):
                raise RuntimeError('concat source member shape differs: ' + name)
            shapes.append(shape)
        if sum(shape[dim] for shape in shapes) != installed.shape[dim]:
            raise RuntimeError('concat source joined shape differs from installed target')
        offset = 0
        for name, shape in zip(names, shapes):
            values[name] = installed.narrow(dim, offset, shape[dim])
            concat_views[name] = dict(target=target, dimension=dim, offset=offset, length=shape[dim])
            offset += shape[dim]
    by_shard = {}
    for live_name, source_name in sorted(context.weight_ckpt.items()):
        if live_name not in values:
            raise RuntimeError('selected source lacks a direct or profile projection view: ' + live_name)
        by_shard.setdefault(context.weight_shard[live_name], []).append((live_name, source_name))
    observations = []
    for path, names in by_shard.items():
        # Exact same held decoder and kernel seals; skip only the private-pin
        # intake because this read is a bounded CPU comparison after load.
        with _SealedDecoders.safe_open(transport, None, path, framework='pt', device='cpu') as reader:
            for live_name, source_name in names:
                source = converted = installed_cpu = None
                try:
                    source = reader.get_tensor(source_name)
                    installed = values[live_name].detach()
                    declared_dtype = transport.dtypes.get(source_name, transport.dtype)
                    if installed.dtype != declared_dtype or installed.shape != source.shape:
                        raise RuntimeError('installed source shape/dtype differs: ' + live_name)
                    nbytes = installed.numel() * installed.element_size()
                    reserve_allocation(transport.guard.check, 'before_installed_source_value_check',
                                       cpu_bytes=nbytes * 2, device_bytes=0)
                    converted = source.to(dtype=declared_dtype)
                    installed_cpu = installed.to('cpu', copy=True)
                    if not torch.equal(converted, installed_cpu):
                        raise RuntimeError('installed source value differs: ' + live_name)
                    observations.append(dict(live_name=live_name, source_name=source_name,
                        shape=list(source.shape), source_dtype=str(source.dtype),
                        installed_dtype=str(declared_dtype), exact=True, concat_view=concat_views.get(live_name)))
                finally:
                    source = converted = installed_cpu = None
    if len(observations) != len(transport.binding['layer_keys']):
        raise RuntimeError('installed-value check does not cover the exact selected source roster')
    return dict(scope='post-load installed body values versus held sealed source',
                exact=True, selected_keys=len(observations), observations=observations)


def reference_comparison(result, expected):
    """Retain CPU/GPU context without equating different primal backends."""
    actual_rows = {(row['unit'], row['probe']): row for row in result['rows']}
    expected_rows = {(row['unit'], row['probe']): row for row in expected['rows']}
    if len(actual_rows) != len(result['rows']) or len(expected_rows) != len(expected['rows']):
        raise RuntimeError('duplicate resident/staged unit-probe observations')
    observations = []
    for key in sorted(set(actual_rows) & set(expected_rows)):
        row, reference = actual_rows[key], expected_rows[key]
        if not torch.isfinite(row['signed_rows']).all() or not torch.isfinite(reference['signed_rows']).all():
            raise RuntimeError('nonfinite resident/staged contextual signed rows')
        coordinates_equal = torch.equal(row['coordinates'], reference['coordinates'])
        observation = dict(unit=key[0], probe=key[1], coordinates_equal=coordinates_equal)
        if coordinates_equal:
            if row['signed_rows'].shape != reference['signed_rows'].shape:
                raise RuntimeError('matching coordinates have different contextual row shapes')
            difference = (row['signed_rows'] - reference['signed_rows']).abs()
            if not torch.isfinite(difference).all():
                raise RuntimeError('nonfinite resident/staged contextual difference')
            observation.update(max_abs_difference=float(difference.max()) if difference.numel() else 0.0,
                unchanged_cpu_tolerance_passed=bool(torch.isclose(row['signed_rows'], reference['signed_rows'],
                    rtol=1e-5, atol=1e-9).all()))
        observations.append(observation)
    return dict(scope='banked CPU resident primal versus executing staged primal',
                unit_probe_sets_equal=set(actual_rows) == set(expected_rows), observations=observations)


@contextmanager
def historical_projection(contract, device):
    """Use the existing prewarm owner on one held SDK staged code artifact."""
    import prismaquant.joint_projection_backend as owner
    from prismaquant.staged_lease import acquire_entry_window
    if device.type != 'cuda':
        raise RuntimeError('historical projection qualification requires its GPU runtime')
    expected = contract['projection_identity']
    qualify_historical_activation_policy(contract)
    config = normalize_projection_backend(contract['projection_config'])
    qualification, digest = owner._qualification()
    # DEV mode cannot weaken these checks. Check before loading/executing code.
    if (digest != expected['qualification_sha256'] or qualification['build'] != expected['build']
            or owner._runtime_identity(device) != expected['runtime']):
        raise RuntimeError('historical projector runtime/build/qualification mismatch')
    path = Path(config['binary']['path'])
    resolver = residency_resolver()
    entry = resolver.staged_read(path, expected_sha256=config['binary']['sha256'])
    if entry is None:
        raise RuntimeError('projector code artifact is not staged')
    window, key = acquire_entry_window(resolver, path, entry)
    with window:
        fd, serving = window.open(key)
        staged_path = window.stage_path(key)
        if not staged_path:
            raise RuntimeError('projector lease supplied no staged code path')
        if os.fstat(fd).st_size != 1523600 or not os.path.samefile(staged_path, f'/proc/self/fd/{fd}'):
            raise RuntimeError('projector path differs from the held staged descriptor')
        routed = {'name': config['name'], 'binary': {**config['binary'], 'path': staged_path}}
        backend = prewarm_projection_backend(routed, device=device)
        if backend.identity != expected:
            raise RuntimeError('loaded projector identity differs from historical contract')
        try:
            yield backend, dict(serving_tier=window.serving_tier, serving=serving,
                                staged_path=staged_path, held_through_final_gpu_fence=True)
        finally:
            # Fence every queued user before the executable lease is released,
            # including cancellation/failure. An uncertain fence fails closed.
            torch.cuda.synchronize(device)
            if owner._sha(Path(staged_path)) != config['binary']['sha256']:
                raise RuntimeError('held projector code bytes changed after control')
            window.close_fd(fd)


def cpu_kernel_control(function):
    """Bind the real derivative first; scope library CPU substitutes to execution."""
    @wraps(function)
    def controlled(runner, *args, **kwargs):
        from transformers.models.glm5_next import modeling_glm5_next as glm
        originals = {}
        if runner.device.type == 'cpu':
            for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_kimi_delta_attention', 'recurrent_kimi_delta_attention'):
                value = getattr(glm, name)
                originals[name] = value
                setattr(glm, name, getattr(value, '__wrapped__', value))
        try:
            return function(runner, *args, **kwargs)
        finally:
            for name, value in originals.items():
                setattr(glm, name, value)
    return controlled


@cpu_kernel_control
def replay(runner, binding, *, captured=None, projection_backend=None, resource_check=None, checkpoint=None):
    layer, rows, length, probes = binding['layer'], binding['sequences'], binding['sequence_length'], binding['n_probes']
    if [row['metadata']['identity']['coordinates']['batch'] for row in binding['boundaries']] != rows:
        raise RuntimeError('boundary entries differ from the exact ordered sequence coordinates')
    ordered = []
    for probe in range(probes):
        entries = [row for row in binding['cotangents'] if row['metadata']['identity']['coordinates']['probe'] == probe]
        if [row['metadata']['identity']['coordinates']['batch'] for row in entries] != rows:
            raise RuntimeError('cotangent entries differ from the exact probe/sequence coordinates')
        ordered.append(entries)
    if sum(map(len, ordered)) != len(binding['cotangents']):
        raise RuntimeError('cotangent entries contain a foreign probe')
    if captured is None:
        decoded = sum(row['tensor_bytes'] for row in binding['boundaries'] + binding['cotangents'])
        largest = max(row['tensor_bytes'] for row in binding['boundaries'] + binding['cotangents'])
        reserve_allocation(resource_check, 'before_retained_capture_copies',
                           cpu_bytes=decoded + largest * max(2, len(rows)), device_bytes=decoded)
    xs, cs = (captured if captured is not None else
              ([read_capture(row) for row in binding['boundaries']],
               [[read_capture(row) for row in entries] for entries in ordered]))
    x = torch.cat(xs, 0).to(runner.device, runner.dtype)
    cotangents = [torch.cat(values, 0).to(runner.device, runner.dtype) for values in cs]
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=runner.device)
    positions = torch.arange(length, device=runner.device).unsqueeze(0)
    embeddings = _compute_position_embeddings(runner.base_model, x, positions, runner.profile)
    mask = _compute_attention_mask(runner.base_model, x, positions)
    batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], None)
    from prismaquant import aura_cost
    modules = aura_cost._target_linears(runner.model, include_lm_head=False, include_routed_experts=True, profile=runner.profile)
    modules.update({member.qname: member for member in profile_declared_packed_expert_projections(runner.model, runner.profile)})
    modules = {name: module for name, module in modules.items() if name.startswith(f'{runner.layers_prefix}{layer}.mlp.')
               and ('.experts.' in name or '.shared_experts.' in name)}
    moe = runner.layers[layer].mlp
    leaf = []
    def detached(module, args):
        value = args[0].detach().requires_grad_(True)
        leaf.append(value)
        return (value, *args[1:])
    hook = moe.register_forward_pre_hook(detached)
    observer = RowContractions(modules, moe, rows, length, binding['activation_format'], projection_backend, resource_check,
                               binding.get('activation_policy'), binding.get('fact_units', []))
    terms = []
    try:
        with observer:
            observer.begin_probe()
            out = runner.isolated_layer(batch, layer, x, pass_state={})
            if len(leaf) != 1 or not out.requires_grad:
                raise RuntimeError('local MoE leaf did not own the expected graph')
            for probe, cotangent in enumerate(cotangents):
                if probe:
                    observer.begin_probe()
                observer.probe = probe
                if resource_check is not None:
                    resource_check('before_local_backward_probe:' + str(probe))
                torch.autograd.grad(out, leaf[0], cotangent, retain_graph=probe < probes - 1)
                terms.append(observer.finish_probe())
                if resource_check is not None:
                    resource_check('after_local_backward_probe:' + str(probe))
                if checkpoint is not None:
                    checkpoint(probe, [row for row in observer.rows if row['probe'] == probe], terms[-1])
        errors, references = [], []
        for row in observer.rows:
            reference = terms[row['probe']][row['unit'], observer.spec.name]['activation']
            errors.append(abs(reference - row['fixed_g_dot']))
            references.append(reference)
        reference = torch.tensor(references, dtype=torch.float64)
        rms = float(reference.square().mean().sqrt())
        relative = max(errors, default=0.0) / rms if rms > 0 else float('inf')
        if not torch.isfinite(reference).all() or not torch.isfinite(torch.tensor(relative)) or relative > 1e-4:
            raise RuntimeError('fixed-cotangent rows differ from the production signed projection')
        expected_dense = {(sequence, pos, -1) for sequence in rows for pos in range(length)}
        expected_routed = {(sequence, pos, slot) for sequence in rows for pos in range(length)
                           for slot in range(moe.experts.config.num_experts_per_tok)}
        for probe in range(probes):
            for role in ('gate_proj', 'up_proj', 'down_proj'):
                for routed in (False, True):
                    coordinates = [tuple(coordinate) for row in observer.rows if row['probe'] == probe
                        and row['unit'].endswith(role) and ('.experts.' in row['unit']) == routed
                        for coordinate in row['coordinates'].tolist()]
                    expected = expected_routed if routed else expected_dense
                    if len(coordinates) != len(expected) or set(coordinates) != expected:
                        raise RuntimeError('routed/shared sequence-position-slot coverage differs')
        return dict(rows=observer.rows, facts=observer.facts, fact_units=sorted(observer.fact_units),
                    terms=terms, selected_units=sorted(modules),
                    max_abs_fixed_g_operator_difference=max(errors, default=0.0),
                    max_relative_to_operator_rms=relative, coordinate_coverage_exact=True,
                    projection_backend=observer.projection_backend.identity,
                    activation_format=observer.spec.name,
                    activation_callable=observer.spec.activation_quantize_dequantize.__module__ + '.' +
                        observer.spec.activation_quantize_dequantize.__qualname__)
    finally:
        observer.close()
        hook.remove()


def prepare_synthetic(output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = fixture_module()
    config = fixture._tiny_config()
    config.text_config.first_k_dense_replace = 0
    config.text_config.mlp_layer_types = ['sparse', 'sparse']
    torch.manual_seed(1962)
    model = fixture._build_model(config)
    model.config.architectures = [type(model).__name__]
    root = output / 'model'
    root.mkdir()
    state = {}
    for name, value in model.state_dict().items():
        if '.mlp.experts.gate_up_proj' in name:
            parent = name.removesuffix('gate_up_proj')
            for expert, weight in enumerate(value):
                gate, up = weight.chunk(2, 0)
                state[f'{parent}{expert}.gate_proj.weight'] = gate.contiguous()
                state[f'{parent}{expert}.up_proj.weight'] = up.contiguous()
        elif '.mlp.experts.down_proj' in name:
            parent = name.removesuffix('down_proj')
            for expert, weight in enumerate(value):
                state[f'{parent}{expert}.down_proj.weight'] = weight.contiguous()
        else:
            state[name] = value.contiguous()
    weights = root / 'model.safetensors'
    save_file(state, str(weights))
    write_json(root / 'config.json', model.config.to_dict())
    write_json(root / 'model.safetensors.index.json', {'weight_map': {name: weights.name for name in state}})
    captures = output / 'captures'
    captures.mkdir()
    boundaries, cotangents, xs, cs = [], [], [], [[], []]
    for sequence in range(4):
        for probe in (None, 0, 1):
            tensor = torch.randn((1, 32, 4, 64), generator=torch.Generator().manual_seed(1962 + sequence * 10 + (probe or 0)))
            name = f'{"boundary" if probe is None else "cotangent"}-{sequence}-{probe}'
            ident = {'session': {'generation': 'synthetic-1962', 'run_identity_sha256': sha(b'synthetic-only')},
                     'kind': 'boundary' if probe is None else 'cotangent', 'slot': name,
                     'coordinates': {'batch': sequence, 'boundary': 0 if probe is None else 1, 'probe': probe}}
            ref = write_exact_activation_cache_entry(captures, name, tensor, identity=ident,
                max_tensor_bytes=1 << 20, max_file_bytes=2 << 20)
            row = dict(path=ref.path, name=ref.name, metadata=json.loads(ref.metadata_json),
                       shape=list(ref.shape), dtype=ref.dtype, tensor_bytes=ref.tensor_bytes,
                       file_bytes=ref.file_bytes, sha256=ref.sha256)
            (boundaries if probe is None else cotangents).append(row)
            (xs if probe is None else cs[probe]).append(tensor)
    binding = dict(scope='synthetic only', model=str(root), layer=0,
        sequences=list(range(4)), sequence_length=32, n_probes=2, global_token_count=128,
        activation_format='TESSERA_E4M3_K1_R1024',
        layer_keys=sorted(name for name in state if name.startswith('model.language_model.layers.0.')),
        layer_cache_bytes=32 << 20, metadata=[{**record(path), 'logical_path': str(path)} for path in
            (root / 'config.json', root / 'model.safetensors.index.json')],
        shards=[record(weights)], boundaries=boundaries, cotangents=cotangents)
    model.config._experts_implementation = model.config.text_config._experts_implementation = 'grouped_mm'
    bind_source_derivative(model, Glm5NextProfile(), DERIVATIVE)
    runner = fixture._streamed_runner(model)
    with profiled(output / 'resident-reference', torch.device('cpu')):
        baseline = replay(runner, binding, captured=(xs, cs))
    torch.save(baseline, output / 'resident-reference.pt')
    binding['reference'] = record(output / 'resident-reference.pt')
    write_json(output / 'binding.json', binding)
    entries = [{'path': str(output / 'binding.json'), 'offset': 0, 'bytes': (output / 'binding.json').stat().st_size,
                'sha256': sha((output / 'binding.json').read_bytes())}]
    for row in binding['metadata'] + binding['shards']:
        entries.append(dict(path=row['path'], offset=0, bytes=row['bytes'], sha256=row['sha256']))
    entries.append(dict(path=binding['reference']['path'], offset=0,
                        bytes=binding['reference']['bytes'], sha256=binding['reference']['sha256']))
    for row in boundaries + cotangents:
        entries.append(dict(path=row['path'], offset=0, bytes=row['file_bytes'], sha256=row['sha256']))
    source_commit = os.environ.get('PRISMAQUANT_IDENTITY_GIT_COMMIT') or subprocess.check_output(
        ['git', '-c', 'safe.directory=*', 'rev-parse', 'HEAD'], text=True).strip()
    total = sum(row['bytes'] for row in entries)
    write_json(output / 'data-manifest.json', {
        'schema': 'prismaquant.prismabuild.data_manifest.v1', 'mount_prefix': '/mnt/shared',
        'produced_by': {'tool': 'issue2032-synthetic-source-replay', 'source_commit': source_commit,
                        'size_source': 'qualified-output-stat'},
        'entry_count': len(entries), 'total_bytes': total,
        'annotations': {'scope': 'issue2032-synthetic-source-replay', 'allowed_tiers': 'ram,ssd',
                        'phases': [{'name': 'startup', 'bytes': total, 'cumulative_bytes': total,
                                    'note': 'whole immutable synthetic inputs before replay'}]},
        'entries': entries})
    print(json.dumps(dict(synthetic=True, binding=record(output / 'binding.json'), manifest=record(output / 'data-manifest.json'))), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-synthetic', type=Path)
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha256')
    p.add_argument('--data-manifest-sha256')
    p.add_argument('--projection-contract', type=Path)
    p.add_argument('--projection-contract-sha256')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--original-local-control', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not args.prepare_synthetic and not all((args.binding, args.binding_sha256, args.data_manifest_sha256, args.output)):
        p.error('replay requires immutable binding, bound data manifest and unused output directory')
    return args


def execute(args):
    torch.set_num_threads(1)
    if args.prepare_synthetic:
        prepare_synthetic(args.prepare_synthetic)
        return
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    binding = json.loads(read_staged_whole_file(args.binding, args.binding_sha256, label='replay binding'))
    original = validate_original_binding(binding, args)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if original else torch.float32
    os.environ['PRISMAQUANT_DIRECT_CUDA_LOAD'] = '0'
    os.environ['PRISMAQUANT_LAYER_READ_THREADS'] = '1'
    transport = ReplayDecoders(binding, device, dtype)
    runner = None
    try:
        from contextlib import nullcontext
        projector = nullcontext((None, {'name': 'CPU torch reference'}))
        if args.projection_contract:
            if not args.projection_contract_sha256:
                raise RuntimeError('projection contract requires an independently bound digest')
            contract = json.loads(read_staged_whole_file(args.projection_contract,
                args.projection_contract_sha256, label='historical projector contract'))
            if original and binding.get('activation_policy') != qualify_historical_activation_policy(contract):
                raise RuntimeError('bound original activation policy differs from historical contract')
            projector = historical_projection(contract, device)
        with profiled(args.output / 'entrypoint', device), bounded_cuda_allocator(binding, device) as envelope, projector as (backend, code_route):
            transport.open()
            runner, derivative = make_context(transport, binding['layer'], device, dtype, args.output)
            source_values = verify_installed_source_tensors(runner, transport)
            write_json(args.output / 'startup.json', dict(scope=binding['scope'],
                selected_source_keys=len(binding['layer_keys']), derivative=derivative,
                installed_source_values=source_values, code_artifact_route=code_route,
                allocator_envelope=envelope, memory_guard=transport.guard.snapshot()))
            progress(1, 'startup')
            checkpoint = None
            if original:
                remaining = max(0, binding['allocator_limits']['torch_allocator_bytes'] - torch.cuda.memory_reserved(device))
                transport.guard.check('before_original_graph_envelope',
                    reserve_bytes=binding['trace_host_reserve_bytes'], reserve_device_bytes=remaining)
                def checkpoint(probe, rows, terms):
                    save_tensors(args.output / f'raw-probe-{probe:03d}.pt', dict(
                        scope=binding['scope'], probe=probe, rows=rows, terms=terms, qualification_pending=True))
                    progress(probe + 2, 'control')
            else:
                transport.guard.check('before_local_graph', reserve_bytes=binding.get('graph_reserve_bytes', 64 << 20))
            result = replay(runner, binding, projection_backend=backend,
                            resource_check=transport.guard.check, checkpoint=checkpoint)
            transport.guard.check('after_local_graph')
        # Retain the completed same-primal operator/coordinate control before
        # any independent reference gate. A later refusal is still a refusal,
        # but must not erase the measured local tensors or their scope.
        save_tensors(args.output / 'rows.pt', result)
        write_json(args.output / 'local-control.json', dict(
            scope=binding['scope'], same_primal_operator_gate_passed=True,
            resident_reference_gate_passed=False, rows=record(args.output / 'rows.pt'),
            selected_source_keys=len(binding['layer_keys']), selected_units=len(result['selected_units']),
            max_abs_fixed_g_operator_difference=result['max_abs_fixed_g_operator_difference'],
            max_relative_to_operator_rms=result['max_relative_to_operator_rms'],
            coordinate_coverage_exact=result['coordinate_coverage_exact'],
            copy_fences=transport.copy_fences, transient_staging_peak=transport.transient_staging_peak,
            installed_source_values=source_values,
            projection_backend=result['projection_backend'], activation_format=result['activation_format'],
            activation_callable=result['activation_callable'], code_artifact_route=code_route,
            allocator_envelope=envelope, memory_guard=transport.guard.snapshot(),
            derivative=derivative, residency=residency_report()))
        completed = binding['n_probes'] + 2 if original else 2
        progress(completed, 'control')
        comparison = dict(scope='no matched historical four-row resident price exists',
                          used_for_acceptance=False, executing_device=str(device))
        if not original:
            ref = binding['reference']
            buffer = read_staged_sealed_file(Path(ref['path']), ref['sha256'], ref['bytes'], label='resident reference')
            try:
                expected = torch.load(buffer.path, map_location='cpu', weights_only=True)
            finally:
                buffer.close()
            comparison = reference_comparison(result, expected)
        comparison['used_for_acceptance'] = device.type == 'cpu'
        comparison['executing_device'] = str(device)
        write_json(args.output / 'reference-comparison.json', comparison)
        if comparison['used_for_acceptance']:
            if not comparison['unit_probe_sets_equal']:
                raise RuntimeError('resident/staged unit-probe observations differ')
            for observation in comparison['observations']:
                if not observation['coordinates_equal']:
                    raise RuntimeError('resident/staged routed coordinates differ')
                if not observation['unchanged_cpu_tolerance_passed']:
                    raise AssertionError('resident/staged CPU signed rows differ at unchanged tolerance')
        write_json(args.output / 'result.json', dict(scope=binding['scope'], passed=True,
            source_transport='shared lowlevel27b/#1934 held decoder plus bounded metadata/copy seam',
            selected_source_keys=len(binding['layer_keys']), selected_units=len(result['selected_units']),
            fact_units=result.get('fact_units', []), fact_rows=len(result.get('facts', [])),
            max_abs_fixed_g_operator_difference=result['max_abs_fixed_g_operator_difference'],
            max_relative_to_operator_rms=result['max_relative_to_operator_rms'], coordinate_coverage_exact=True,
            copy_fences=transport.copy_fences, transient_staging_peak=transport.transient_staging_peak,
            installed_source_values=source_values,
            reference_comparison=comparison,
            projection_backend=result['projection_backend'], activation_format=result['activation_format'],
            activation_callable=result['activation_callable'],
            code_artifact_route=code_route, memory_guard=transport.guard.snapshot(),
            derivative=derivative, residency=residency_report(),
            allocator_envelope=envelope,
            limitations=(binding['limitations'] if original else
                ['synthetic dimensions', 'CPU torch-only convolution when CPU', 'no original source or full-model gate'])))
        print(json.dumps(dict(fact_units=result.get('fact_units', []),
            fact_rows=len(result.get('facts', [])),
            fact_row_keys=sorted({(fact['unit'], fact['probe']) for fact in result.get('facts', [])}))), flush=True)
        print('PASS: local source/cache/isolation/coordinate/contraction entrypoint; no whole-model price', flush=True)
        progress(completed + 1, 'publish')
    finally:
        if runner is not None:
            runner.shutdown()
            runner.context.layer_cache.clear()
        gc.collect()
        transport.close()


def main():
    args = parse_args()
    prior = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic
    try:
        pin_matmul_arithmetic()
        execute(args)
    finally:
        torch.set_float32_matmul_precision(prior[0])
        torch.backends.cuda.matmul.allow_tf32 = prior[1]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prior[2]


if __name__ == '__main__':
    main()
