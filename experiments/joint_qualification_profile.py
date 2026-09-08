"""Matched qualification of actual tiny GLM campaign anchors and PWC donors.

The source is the original-layout two-layer GLM fixture, not a trained model.
Its full census/capture is real; the priced scope is one dense group and one
complete packed-expert group with two real H-aware formats. Every qualification
uses prepare_cache and the producer's unchanged source/H/wire/render verifiers.
This measures preparation ownership, not a calibration probe, KL, serving,
full-GLM fit or production throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import threading
import time
import traceback
from unittest.mock import patch

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.workspace_netdata import NetdataWriter, sample_netdata


FORMATS = ('TESSERA_E4M3_K1_R768', 'TESSERA_E4M3_K1_R1024')
SEED = 1917
TOKENS = 257
PREFIX_ROWS = 64


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes
    atomic_write_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True,
                                               allow_nan=False) + '\n').encode())


def select_groups(census):
    """Choose two whole existing groups; never invent an expert projection."""
    groups = census['anchor_groups']
    shapes = census['unit_shapes']
    eligible = {key: members for key, members in groups.items()
                if members and all(len(shapes[name]) == 2 and
                    all(dim >= 256 and dim % 256 == 0 for dim in shapes[name])
                    for name in members)}
    dense = sorted(key for key, members in eligible.items()
                   if all('.experts.' not in name for name in members))
    packed = sorted(key for key, members in eligible.items()
                    if all('.experts.' in name for name in members))
    if not dense or not packed:
        raise RuntimeError('tiny census lacks complete dense and packed groups for both H-aware rungs')
    # Prefer the largest dense input dimension so H ownership is observable.
    dense_key = max(dense, key=lambda key: (max(shapes[name][1] for name in groups[key]), key))
    return [dict(key=key, members=list(groups[key])) for key in (dense_key, packed[0])]


def build_fixture(root):
    """Publish original-layout source, complete capture and measured anchors."""
    from prismaquant import tessera_campaign as campaign, tessera_calibration_cache as cc
    from transformers.models.glm5_next import modeling_glm5_next as upstream

    # Reuse the repository's genuine initialization and original-name writer.
    tests = Path(__file__).resolve().parents[1] / 'tests'
    sys.path.insert(0, str(tests))
    try:
        from test_glm5_next_streamed_forward_parity import _tiny_config, _build_model
        from test_glm_campaign_streaming import write_original_layout_checkpoint
    finally:
        sys.path.remove(str(tests))
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    torch.manual_seed(SEED)
    config = _tiny_config()
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.moe_intermediate_size = 256
    config.vision_config.out_hidden_size = 256
    config = type(config).from_dict(config.to_dict())
    model = _build_model(config).to(torch.bfloat16)
    write_original_layout_checkpoint(model, source)
    del model
    tokens = [torch.arange(TOKENS).remainder(126).add(2).reshape(1, -1)]
    text = 'tiny GLM frozen draw for bounded anchor qualification; not a dataset sample'
    census_path = root / 'census.json'
    capture_root = root / 'capture'
    common = ['--model', str(source), '--out', str(root / 'unused.pkl'),
        '--menu-mode', 'research', '--nsamples', '1', '--seed', str(SEED),
        '--seqlen', str(TOKENS), '--max-act-rows', str(PREFIX_ROWS),
        '--attention-implementation', 'eager', '--streaming',
        '--streaming-cache-headroom-gb', '0', '--streaming-cache-slots', '2',
        '--streaming-prefetch-workers', '1']
    with ExitStack() as stack:
        stack.enter_context(patch.object(campaign, '_calibration_tokens', lambda *_: (tokens, text)))
        # The existing tiny campaign test uses these same upstream reference
        # kernels, including on CUDA. Record this fixture-only choice explicitly.
        forward_kernels = {}
        for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                     'chunk_kimi_delta_attention', 'recurrent_kimi_delta_attention'):
            current = getattr(upstream, name)
            reference = getattr(current, '__wrapped__', None)
            if reference is None:
                raise RuntimeError(f'cannot recover the existing tiny fixture reference kernel: {name}')
            stack.enter_context(patch.object(upstream, name, reference))
            forward_kernels[name] = f'{reference.__module__}.{reference.__qualname__}'
        if campaign.main([*common, '--cache-dir', str(root / 'census-cache'),
                          '--census-out', str(census_path)]) != 0:
            raise RuntimeError('actual tiny source census failed')
        census = json.loads(census_path.read_text())
        if campaign.main([*common, '--cache-dir', str(root / 'capture-cache'),
                '--calibration-census', str(census_path), '--capture-calibration-out', str(capture_root),
                '--streaming-capture-policy', 'shared-inputs-bounded-v1']) != 0:
            raise RuntimeError('actual canonical tiny capture failed')
        capture_path = capture_root / 'capture_manifest.json'
        capture = dict(path=str(capture_path), sha256=cc.sha256(capture_path))
        cc.require_capture_contract(capture_path, expected_sha256=capture['sha256'])
        groups = select_groups(census)
        selected = sorted(name for group in groups for name in group['members'])
        selection = root / 'selected-units.json'
        write_json(selection, dict(schema=campaign.UNITS_SCHEMA, model=str(source),
                                   layer_stride=1, groups=groups))
        original_menus = campaign.expand_menus_for_targets
        def exact_menu(weights, targets, **kwargs):
            menus = original_menus(weights, targets, **kwargs)
            if set(weights) != set(selected):
                raise RuntimeError('selected campaign changed its complete group roster')
            narrowed = {name: [row for row in rows if row.format_name in FORMATS]
                        for name, rows in menus.items()}
            if any({row.format_name for row in rows} != set(FORMATS) for rows in narrowed.values()):
                raise RuntimeError('both real H-aware rungs must be admitted for every selected unit')
            return narrowed
        stack.enter_context(patch.object(campaign, 'expand_menus_for_targets', exact_menu))
        def no_capture(*_args, **_kwargs):
            raise RuntimeError('selected anchor reuse attempted a new calibration forward')
        stack.enter_context(patch.object(campaign, '_collect_activations', no_capture))
        output = root / 'selected-cost.pkl'
        if campaign.main([*common, '--out', str(output), '--cache-dir', str(root / 'selected-cache'),
                '--units', str(selection), '--calibration-census', str(census_path),
                '--calibration-cache', str(capture_path), '--calibration-cache-sha256', capture['sha256'],
                '--anchors', '2', '--anchor-batch-size', '1', '--max-rounds', '1']) != 0:
            raise RuntimeError('actual selected anchor campaign failed')
    payload = pickle.loads(output.read_bytes())
    preparation = payload['provenance']['selected_source_preparation']
    if preparation['source_forward_count'] != 0 or preparation['full_source_initialization_repeated']:
        raise RuntimeError('selected anchors repeated a source forward or initialization witness')
    result = dict(schema='prismaquant.joint_qualification_fixture.v1', source=str(source),
        census=dict(path=str(census_path), sha256=sha(census_path)), capture=capture,
        selected_cost=dict(path=str(output), sha256=sha(output)),
        selected_checkpoint=dict(path=str(output.with_suffix('.anchors.json')),
                                 sha256=sha(output.with_suffix('.anchors.json'))),
        selected_groups=groups, selected_units=selected, formats=list(FORMATS),
        source_kind='genuinely initialized tiny random GLM; original checkpoint names/layout',
        scope='selected dense group and complete packed group from full tiny census',
        seed=SEED, token_rule='arange(257).remainder(126).add(2)',
        token_bytes_sha256=hashlib.sha256(tokens[0].numpy().tobytes()).hexdigest(),
        calibration_text=text, fixture_forward_kernels=forward_kernels,
        selected_source_preparation=preparation,
        producer_checkpoint=str(Path(os.environ['TESSERA_REPO']).resolve()))
    write_json(root / 'fixture.json', result)
    return result


def measured_input(fixture):
    """Read actual selected journals; make no whole-campaign completion claim."""
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path, canonical_json_sha256
    from prismaquant.production_weight_cache import _cache_weight_filename
    from prismaquant.tessera_joint_aura import MeasuredAnchorInput

    for key in ('census', 'capture', 'selected_cost', 'selected_checkpoint'):
        if sha(fixture[key]['path']) != fixture[key]['sha256']:
            raise RuntimeError(f'fixture input changed: {key}')
    census = json.loads(Path(fixture['census']['path']).read_text())
    payload = pickle.loads(Path(fixture['selected_cost']['path']).read_bytes())
    journal = Path(fixture['selected_checkpoint']['path'])
    manifest = json.loads(journal.read_text())
    identity = manifest['identity']
    seal = canonical_json_sha256(identity, where='bounded native qualification')
    if seal != manifest['identity_sha256']:
        raise RuntimeError('actual selected journal identity changed')
    names = fixture['selected_units']
    if set(payload['costs']) != set(names) or set(identity['units']) != set(names):
        raise RuntimeError('actual selected payload/journal roster differs')
    cells, formats = {}, {}
    for name in names:
        state = _load_unit(unit_path(journal.with_name(journal.name + '.parts'), name),
                           stage='Tessera campaign', qname=name, identity_sha256=seal)
        anchors = {anchor['format_name']: anchor for anchor in state['anchors']}
        if set(anchors) != set(FORMATS) or set(state['wire_records']) != set(FORMATS):
            raise RuntimeError(f'{name}: two actual measured anchors and wire receipts required')
        if any(not anchor['hessian_applied'] for anchor in anchors.values()):
            raise RuntimeError(f'{name}: qualification requires real H-aware anchors')
        if not any(anchor['dloss'] > 0 for anchor in anchors.values()):
            raise RuntimeError(f'{name}: all-zero anchor costs do not exercise render qualification')
        for fmt, anchor in anchors.items():
            row = payload['costs'][name][fmt]
            if row.get('output_mse_measured') is not True or row['output_mse'] != anchor['dloss']:
                raise RuntimeError(f'{name}@{fmt}: actual measured cost/journal mismatch')
            record = state['wire_records'][fmt]
            wire = Path(payload['provenance']['wire_dir']) / record['file']
            render = Path(payload['provenance']['cache_dir']) / _cache_weight_filename(name, fmt)
            if sha(wire) != record['blob_sha256'] or wire.stat().st_size != record['blob_bytes']:
                raise RuntimeError(f'{name}@{fmt}: actual wire differs from its journal')
            cells[name, fmt] = dict(anchor=anchor, record=record, wire=str(wire),
                                    render=str(render), render_file_sha256=sha(render))
        formats[name] = (*sorted(anchors), 'BF16')
    # The qualifier consumes these real selected records with the full census
    # and original canonical capture. Its own gates are never patched out.
    return MeasuredAnchorInput(inputs={'census': fixture['census'],
        'qualification_fixture': 'explicit selected native scope; no complete campaign receipt'},
        payload=payload, manifest=manifest, census=census, campaign_plan={},
        cells=cells, formats_by_qname=formats)


def input_inventory(fixture, data):
    paths = {Path(fixture[key]['path']) for key in
             ('census', 'capture', 'selected_cost', 'selected_checkpoint')}
    paths.update(path for path in Path(fixture['source']).iterdir() if path.is_file())
    capture_path = Path(fixture['capture']['path'])
    manifest = json.loads(capture_path.read_text())
    paths.update(capture_path.parent / entry['path'] for entry in manifest['entries'].values())
    paths.update(Path(cell[kind]) for cell in data.cells.values() for kind in ('wire', 'render'))
    journal = Path(fixture['selected_checkpoint']['path'])
    paths.update(path for path in journal.with_name(journal.name + '.parts').rglob('*') if path.is_file())
    return {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(paths)}


def physical_snapshot(device):
    fields = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB':
            fields[parts[0].removesuffix(':')] = int(parts[1]) * 1024
    result = dict(time_unix=time.time(), smaps_bytes=fields,
                  process_io=Path('/proc/self/io').read_text())
    scope = Path('/sys/fs/cgroup')
    for name in ('memory.current', 'memory.peak', 'memory.max'):
        path = scope / name
        if path.is_file():
            value = path.read_text().strip()
            result[name] = value if value == 'max' else int(value)
    path = scope / 'memory.stat'
    if path.is_file():
        result['memory.stat'] = {key: int(value) for key, value in
                                 (line.split() for line in path.read_text().splitlines())}
    if device.type == 'cuda':
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


class OwnershipObserver:
    """Weak storage observations around existing methods, retaining no tensors."""
    def __init__(self, kind, device):
        self.kind, self.device = kind, device
        self.storages = {}
        self.events = []
        self.source_reads = Counter()
        self.peak_bytes = Counter()

    def register(self, kind, name, tensor):
        storage = tensor.untyped_storage()
        self.storages[kind, storage._cdata] = (StorageWeakRef(storage), storage.nbytes(), name)

    def live(self):
        counts, sizes = Counter(), Counter()
        for (kind, _), (reference, nbytes, _) in self.storages.items():
            if not reference.expired():
                counts[kind] += 1
                sizes[kind] += nbytes
        for kind, size in sizes.items():
            self.peak_bytes[kind] = max(self.peak_bytes[kind], size)
        return dict(counts=dict(counts), bytes=dict(sizes))

    def record(self, label, **fields):
        row = dict(label=label, live=self.live(), **fields)
        self.events.append(row)
        return row

    def __enter__(self):
        from prismaquant import tessera_calibration_cache as cc, streaming_model as sm
        from prismaquant.production_weight_cache import ProductionWeightCache
        self.stack = ExitStack()
        capture = cc.prefetch_capture
        def observed_capture(*args, names, **kwargs):
            row = self.record('before_capture', names=list(names), physical=physical_snapshot(self.device))
            if self.kind == 'window' and row['live']['counts']:
                raise RuntimeError('previous capture/PWC backing storage is still live at the next unit')
            with torch.profiler.record_function('qualification_capture_prefetch'):
                result = capture(*args, names=names, **kwargs)
            acts, hessians, _, _ = result[0]
            for label, values in (('X', acts), ('H', hessians)):
                for name, tensor in values.items():
                    self.register(label, name, tensor)
            self.record('after_capture', names=list(names), physical=physical_snapshot(self.device))
            return result
        self.stack.enter_context(patch.object(cc, 'prefetch_capture', observed_capture))
        loader = sm._read_layer_to_device
        def observed_source(prefix, *args, **kwargs):
            self.source_reads[str(prefix)] += 1
            return loader(prefix, *args, **kwargs)
        self.stack.enter_context(patch.object(sm, '_read_layer_to_device', observed_source))
        record_load = ProductionWeightCache._record_file_load
        def observed_pwc(cache, key, tensor, receipt):
            result = record_load(cache, key, tensor, receipt)
            self.register('PWC', '|'.join(key), tensor)
            self.record('pwc_loaded', key=list(key))
            return result
        self.stack.enter_context(patch.object(ProductionWeightCache, '_record_file_load', observed_pwc))
        release = ProductionWeightCache.release_resident_tensors
        def observed_release(cache, keys=None):
            result = release(cache, keys)
            row = self.record('pwc_released', keys=None if keys is None else list(keys))
            if self.kind == 'window' and row['live']['counts'].get('PWC', 0):
                raise RuntimeError('PWC window retained backing storage after selected release')
            return result
        self.stack.enter_context(patch.object(ProductionWeightCache, 'release_resident_tensors', observed_release))
        return self

    def __exit__(self, *_args):
        self.stack.close()


def verified_rows(cache):
    return [dict(unit=name, format=fmt, record=record)
            for (name, fmt), record in sorted(cache.metadata['verified_cells'].items())]


def qualify_once(kind, fixture, data, device, *, trace=None):
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.tessera_joint_aura import prepare_cache, QUALIFICATION_WINDOW_SCHEMA
    max_file = max(Path(cell['render']).stat().st_size for cell in data.cells.values())
    max_capture = max(4 * (shape[1] ** 2 + min(data.census['counts'][name], PREFIX_ROWS) * shape[1])
                      for name, shape in data.census['unit_shapes'].items()
                      if name in data.formats_by_qname)
    # A common PWC cap admits the legacy whole selected layer. The new load
    # buffer cap deliberately admits one donor per quantum, independent of menu.
    render_cap = sum(Path(cell['render']).stat().st_size for cell in data.cells.values())
    policy = dict(schema=QUALIFICATION_WINDOW_SCHEMA, max_capture_resident_bytes=max_capture,
                  max_load_buffer_bytes=max_file, workspace_reserve_bytes=64 * 1024**2)
    with OwnershipObserver(kind, device) as observer:
        runner = build_streamed_causal_lm(fixture['source'], device=device, dtype=torch.bfloat16,
            offload_folder=str(Path(fixture['source']).parent / 'qualifier-offload'),
            profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
            prefetch_lookahead=1, require_prefetched_residency=True,
            cache_headroom_gb=0, prefetch_min_available_gb=0, attn_implementation='eager')
        initialization_reads = dict(observer.source_reads)
        observer.source_reads.clear()
        def forbid_forward(*_args, **_kwargs):
            raise RuntimeError('anchor qualifier attempted a source forward')
        handles = [module.register_forward_pre_hook(forbid_forward)
                   for module in (runner.model, *runner.context.layers)]
        profiler = None
        try:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            before = physical_snapshot(device)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with ExitStack() as stack:
                if trace is not None:
                    profiler = stack.enter_context(torch.profiler.profile(activities=activities,
                        profile_memory=True, record_shapes=True))
                start = time.perf_counter()
                started_unix = time.time()
                with torch.profiler.record_function('joint_anchor_qualification_' + kind):
                    cache = prepare_cache(runner, copy.deepcopy(data), capture=fixture['capture'],
                        max_render_bytes=render_cap, file_load_workers=2,
                        qualification_window=policy if kind == 'window' else None)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
            after = physical_snapshot(device)
            observed = observer.record('prepare_returned')
            if observed['live']['counts'] or any(isinstance(value, torch.Tensor)
                                                for value in cache.weights.values()):
                raise RuntimeError('qualifier returned with live capture/PWC backing owners')
            records = verified_rows(cache)
            roster = [list(key) for key in sorted(cache.weights)]
            result = dict(kind=kind, elapsed_seconds=elapsed, started_unix=started_unix,
                finished_unix=time.time(), verified_cells=records,
                verified_cells_sha256=canonical_sha(records), candidate_roster=roster,
                formats_by_qname=data.formats_by_qname,
                initialization_source_read_counts=initialization_reads,
                source_layer_load_counts=dict(observer.source_reads), source_forward_count=0,
                source_layer_count=runner.num_layers, physical_before=before, physical_after=after,
                live_storage_peak_bytes=dict(observer.peak_bytes), ownership_events=observer.events,
                cache_prefetch=cache.metadata['prefetch'], max_render_bytes=render_cap,
                qualification_window=policy if kind == 'window' else None,
                physical_guard=cache.metadata.get('qualification_memory_guard'))
            if profiler is not None:
                profiler.export_chrome_trace(str(trace))
                result['profile'] = dict(path=str(trace), sha256=sha(trace), events=[
                    dict(key=item.key, calls=item.count, self_cpu_us=item.self_cpu_time_total,
                         self_device_us=item.self_device_time_total)
                    for item in profiler.key_averages()])
            expected_reads = {f'{runner.context.layers_prefix}{index}.': 1
                              for index in range(runner.num_layers)}
            if dict(observer.source_reads) != expected_reads:
                raise RuntimeError('qualifier source-layer reads differ: '
                                   f'{dict(observer.source_reads)!r} != {expected_reads!r}')
            del cache
        finally:
            for handle in handles:
                handle.remove()
            runner.shutdown()
            del runner, handles, profiler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    result['physical_after_source_release'] = physical_snapshot(device)
    return result


def require_parity(reference, observed):
    for key in ('verified_cells', 'verified_cells_sha256', 'candidate_roster',
                'formats_by_qname', 'initialization_source_read_counts',
                'source_layer_load_counts', 'source_forward_count'):
        if observed[key] != reference[key]:
            raise RuntimeError(f'legacy/window qualification mismatch: {key}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be between 1 and 30')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA qualification requires an admitted native GPU')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    record = dict(schema='prismaquant.joint_qualification_profile.v1', status='running',
        started_unix=time.time(), torch=str(torch.__version__), cuda=torch.version.cuda,
        device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        cpu_affinity=sorted(os.sched_getaffinity(0)), seed=SEED, seconds_per_arm=args.seconds,
        scope=__doc__, arms=[], telemetry_errors=[])
    stopped = threading.Event()
    def monitor():
        try:
            with (args.out / 'netdata.jsonl').open('x') as stream:
                writer = NetdataWriter(stream)
                while not stopped.is_set():
                    for host in ('sparky', 'sparklina'):
                        writer.write(sample_netdata(host))
                    stopped.wait(1.)
        except BaseException as exc:
            record['telemetry_errors'].append(repr(exc))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        fixture = build_fixture(args.out / 'fixture')
        data = measured_input(fixture)
        inventory = input_inventory(fixture, data)
        record.update(fixture=fixture, input_inventory=inventory,
                      input_inventory_sha256=canonical_sha(inventory))
        write_json(args.out / 'progress.json', record)
        reference = None
        for index, kind in enumerate(('legacy', 'window', 'window', 'legacy')):
            if input_inventory(fixture, data) != inventory:
                raise RuntimeError('source, capture, journal, PWC or wire input bytes changed before an arm')
            arm = dict(kind=kind, index=index, started_unix=time.time(), invocations=[])
            record['arms'].append(arm)
            profiled = qualify_once(kind, fixture, data, device,
                                    trace=args.out / f'arm-{index}-{kind}.trace.json')
            if reference is None:
                reference = profiled
            require_parity(reference, profiled)
            arm['profiled'] = profiled
            write_json(args.out / 'progress.json', record)  # Retain the first baseline immediately.
            started = time.perf_counter()
            arm['measurement_started_unix'] = time.time()
            while time.perf_counter() - started < args.seconds:
                observed = qualify_once(kind, fixture, data, device)
                require_parity(reference, observed)
                arm['invocations'].append(observed)
            arm['measurement_finished_unix'] = time.time()
            arm['finished_unix'] = time.time()
            arm['median_qualifier_seconds'] = statistics.median(
                row['elapsed_seconds'] for row in arm['invocations'])
            if input_inventory(fixture, data) != inventory:
                raise RuntimeError('source, capture, journal, PWC or wire input bytes changed during an arm')
            write_json(args.out / 'progress.json', record)
        record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        stopped.set()
        thread.join(timeout=15)
        if thread.is_alive():
            record['telemetry_errors'].append('both-host Netdata sampler did not stop')
        if record['telemetry_errors']:
            record['status'] = 'failed'
        record['finished_unix'] = time.time()
        write_json(args.out / 'result.json', record)
        print(json.dumps(dict(path=str(args.out / 'result.json'), status=record['status'],
                              sha256=sha(args.out / 'result.json'))), flush=True)
    if record['status'] != 'complete':
        raise RuntimeError('native qualification evidence is incomplete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
