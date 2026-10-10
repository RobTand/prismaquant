"""G3's requested-arm source reads and PB read-order manifest (PQ #2301).

An omitted tensor has writable storage with its original shape and dtype.
Each requested arm replaces the whole tensor before its forward.
Null and SOURCE rows retain their bytes and their source digest checks.
Partial or unknown replacement shapes retain their source tensor.

CPU manifest construction runs through pbrun with G3_PQ_ROOT naming the
unchanged pinned PQ host checkout; /pq is only the container's mount path.
"""
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import struct
import threading

import torch
from safetensors import safe_open
from safetensors.torch import _TYPES


def shard_header(path, *, with_size=False):
    from g3_lib import read_range
    size = os.stat(path).st_size
    prefix = read_range(str(path), 0, 8)
    n = struct.unpack("<Q", prefix)[0]
    if n > 100_000_000 or n > size - 8:
        raise ValueError(f"{path}: invalid safetensors header length")
    raw = read_range(str(path), 8, n)
    entries = {}
    for name, row in json.loads(raw).items():
        if name == '__metadata__':
            continue
        start, end = row['data_offsets']
        dtype, shape = _TYPES[row['dtype']], row['shape']
        length = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if start < 0 or end - start != length or end + 8 + n > size:
            raise ValueError(f'{path}: corrupt tensor extent {name}')
        entries[name] = {'offset': 8 + n + start, 'bytes': length, 'dtype': dtype, 'shape': shape}
    return (entries, n) if with_size else entries


def source_indices(arms, rows):
    from g3_offline_decoded_kl import arm_wants
    wants = [arm_wants(a, rows) for a in arms]
    return [i for i in range(len(rows)) if any(w[i] is None for w in wants)]


class SourceReads:
    def __init__(self, model, arms, by_layer):
        from g3_offline_decoded_kl import plan_arms, ARMS
        plan, _ = plan_arms(arms, by_layer)
        self.model = Path(model)
        from g3_residency import read_g3_input
        self.weight_map = json.loads(read_g3_input(self.model / "model.safetensors.index.json"))["weight_map"]
        headers = {s: shard_header(self.model / s, with_size=True) for s in dict.fromkeys(self.weight_map.values())}
        self.entries = {s: h[0] for s, h in headers.items()}
        self.header_sizes = {s: h[1] for s, h in headers.items()}
        candidates, retained = set(), set()
        for layer, rows in by_layer.items():
            needed = set(source_indices(arms, rows))
            decoded = set(plan[layer][0]["decode"])
            for i, row in enumerate(rows):
                name = row['qname'] + '.weight'
                if name not in self.weight_map:
                    continue
                shape = self.entries[self.weight_map[name]][name]['shape']
                complete = all(row.get(ARMS[arm][0] + '_rendered_shape') == shape
                               for arm in arms if ARMS[arm][0] is not None)
                if i not in needed and i in decoded and complete:
                    candidates.add(name)
                else:
                    retained.add(name)
        self.omitted = candidates - retained
        self.stats = {'omitted_tensors': 0, 'omitted_bytes': 0, 'source_tensors': 0, 'source_bytes': 0}
        self.lock = threading.Lock()

    def open(self, path, *, source_authentication=None, **kwargs):
        if source_authentication is not None:
            return source_authentication.safe_open(self.open, path, **kwargs)
        return _SourceOpen(self, path, kwargs)

    def install(self):
        from prismaquant import layer_streaming
        layer_streaming._source_safe_open = self.open
        layer_streaming._source_json = self.read_json

    def read_json(self, path, source_authentication=None):
        if source_authentication is not None:
            return source_authentication.read_json(path)
        from g3_residency import read_g3_input
        return json.loads(read_g3_input(path))


class _SourceOpen:
    def __init__(self, reads, path, kwargs):
        self.reads, self.path, self.kwargs = reads, path, kwargs
        self.header = reads.entries.get(os.path.relpath(path, reads.model))
        if self.header is None:
            self.header = shard_header(path)
        self.original = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.original.__exit__(*exc)

    def keys(self):
        return self.header.keys()

    def get_tensor(self, name):
        entry = self.header[name]
        omitted = name in self.reads.omitted
        with self.reads.lock:
            key = 'omitted' if omitted else 'source'
            self.reads.stats[key + '_tensors'] += 1
            self.reads.stats[key + '_bytes'] += entry['bytes']
        if omitted:
            return torch.empty(entry['shape'], dtype=entry['dtype'], device=self.kwargs.get('device', 'cpu'))
        from g3_residency import staged_range
        raw = staged_range(self.path, entry['offset'], entry['bytes'])
        if raw is not None:
            return torch.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"]).to(self.kwargs.get("device", "cpu"))
        if self.original is None:
            self.original = safe_open(self.path, **self.kwargs)
            self.original.__enter__()
        return self.original.get_tensor(name)


def build_manifest(reads, arms, by_layer, roots, teachers, window_ids, *, mount_prefix='/mnt/shared', setup_files=()):
    from g3_offline_decoded_kl import plan_arms, ARMS
    from g3_lib import member_location
    plan, _ = plan_arms(arms, by_layer)
    entries, registry, phases = [], {}, []
    def add(path, offset, size, sha=None):
        path = os.path.normpath(str(path))
        key = (path, offset)
        if key in registry:
            old = entries[registry[key]]
            if old['bytes'] != size or (sha and old['sha256'] and sha != old['sha256']):
                raise ValueError(f'{path}:{offset}: contradictory read ranges')
            if sha:
                old['sha256'] = sha
            return registry[key]
        registry[key] = len(entries)
        entries.append({'path': path, 'offset': offset, 'bytes': size, 'sha256': sha})
        return len(entries) - 1
    cumulative = 0
    def phase(name, refs):
        nonlocal cumulative
        refs = list(dict.fromkeys(refs))
        size = sum(entries[i]['bytes'] for i in refs)
        cumulative += size
        phases.append({'name': name, 'entry_indices': refs, 'bytes': size, 'cumulative_bytes': cumulative})
    setup = [add(p, 0, Path(p).stat().st_size) for p in setup_files if Path(p).stat().st_size]
    for shard, n in reads.header_sizes.items():
        setup.extend([add(reads.model / shard, 0, 8), add(reads.model / shard, 8, n)])
    for name in ("config.json", "model.safetensors.index.json"):
        p = reads.model / name
        if p.exists():
            setup.append(add(p, 0, p.stat().st_size))
    source_by_phase = defaultdict(list)
    # Use the pinned profile's checkpoint-name mapping and decoder-layer
    # ownership, not a roster or a source-name regex.
    from prismaquant.model_profiles import detect_profile
    profile = detect_profile(str(reads.model))
    layer_prefixes = {
        layer: profile.checkpoint_to_live_name(
            f'{profile.body_layer_prefix()}.{layer}.', multimodal=False)
        for layer in by_layer
    }
    for ckpt, shard in reads.weight_map.items():
        if ckpt in reads.omitted:
            continue
        live = profile.checkpoint_to_live_name(ckpt, multimodal=False)
        if live is None:
            continue
        owner = next((layer for layer, prefix in layer_prefixes.items()
                      if prefix is not None and live.startswith(prefix)), None)
        ent = reads.entries[shard][ckpt]
        source_by_phase['setup' if owner is None else f'layer-{owner:02d}'].append(add(reads.model / shard, ent['offset'], ent['bytes']))
    for root, _ in teachers:
        p = Path(root) / 'teacher.json'
        setup.append(add(p, 0, p.stat().st_size))
    phase('setup', setup + source_by_phase['setup'])
    for layer, rows in sorted(by_layer.items()):
        refs = list(source_by_phase[f'layer-{layer:02d}'])
        for k, arm in enumerate(arms):
            key = ARMS[arm][0]
            for i in plan[layer][k]['decode']:
                loc = rows[i][key + '_wire']
                root = roots[loc.get('root', key)]
                if 'ranges' in loc:
                    refs.extend(add(Path(root) / shard, off, n) for shard, off, n in loc['ranges'])
                else:
                    off, n = member_location(loc)
                    refs.append(add(Path(root) / loc["shard"], off, n))
        phase(f'layer-{layer:02d}', refs)
    refs = []
    for wid in window_ids:
        for root, spec in teachers:
            win = next(w for w in spec['windows'] if w['window_id'] == wid)
            refs.append(add(Path(root) / win['path'], 0, win['bytes'], win['sha256']))
    phase('teachers', refs)
    order = list(dict.fromkeys(i for p in phases for i in p['entry_indices']))
    positions = {old: new for new, old in enumerate(order)}
    entries = [entries[i] for i in order]
    for p in phases:
        p['entry_indices'] = [positions[i] for i in p['entry_indices']]
    return {'schema': 'prismaquant.prismabuild.data_manifest.v2', 'mount_prefix': mount_prefix,
            'produced_by': {'tool': 'g3_readset', 'issue': 'RobTand/prismaquant#2301'},
            'annotations': {'teacher_repeats': len(arms), 'omitted_source_tensors': len(reads.omitted)},
            'entries': entries, 'entry_count': len(entries), 'total_bytes': sum(e['bytes'] for e in entries),
            'read_plan': {'phases': phases, 'read_bytes': cumulative}}


def main():
    import argparse
    import sys
    import g3_offline_decoded_kl as g
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--root', action='append', default=[], help='wire root KEY=HOST_PATH')
    p.add_argument('--teacher', action='append', required=True)
    p.add_argument('--panel', required=True)
    p.add_argument('--arrays-root', help='host path for the panel arrays; the container uses /panel/arrays')
    p.add_argument('--setup-file', action='append', default=[])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    sys.path.insert(0, str(g.PQ))
    m, by_layer, _ = g.read_g3_manifest(args.manifest, args.manifest_sha256, None)
    g.register_picks(m)
    arms = args.arms.split(',')
    reads = SourceReads(args.model, arms, by_layer)
    specs = [(root, json.loads((Path(root) / 'teacher.json').read_bytes())) for root in args.teacher]
    panel = json.loads(Path(args.panel).read_bytes())
    window_ids = [w['window_id'] for w in panel['windows']]
    arrays = Path(args.arrays_root) if args.arrays_root else Path(args.panel).parent / 'arrays'
    setup = [args.manifest, args.panel, arrays / Path(panel['causal_mask_array']).name]
    setup.extend(arrays / Path(w['tokens_path']).name for w in panel['windows'])
    setup.extend(args.setup_file)
    setup.extend(reads.model / row['name'] for row in m.get('source_metadata', []))
    result = build_manifest(reads, arms, by_layer, dict(x.split('=', 1) for x in args.root), specs, window_ids, setup_files=setup)
    with open(args.out, 'xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0, filename='') as zipped:
            zipped.write(json.dumps(result, separators=(',', ':')).encode())
    print(json.dumps({'out': args.out, 'entries': result['entry_count'], 'bytes': result['total_bytes'], 'omitted_source_tensors': len(reads.omitted)}))


if __name__ == '__main__':
    main()
