"""Seal a recovery launcher and ordinary windowed input manifest from recovery proof.

``--label`` names the attempt (r10, r11, ...). The launcher is always derived
from the reviewed R9 launcher; only its source, manifest and recovery bind move.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_forward_resume import _read, chain_documents

ORIGINAL_MANIFEST_SHA = 'ea0b9f4b7e8ca1d4224cabff6a7e8d1dad5cf7f7173cef0f5a900f50ea23a1a8'


def recovery_manifest(original, capsule, bound):
    """Keep source reads, add imported inputs at their actual read phases."""
    manifest = json.loads(json.dumps(original))
    frontier = capsule['frontier']
    layers = 1 + max(int(p['name'].split('-')[1]) for p in original['read_plan']['phases']
                     if p['name'].startswith('forward-'))
    if not 0 < frontier <= layers:
        raise ValueError('recovery frontier outside original source phase plan')
    entries = manifest['entries']
    boundaries = {}
    # A chained capsule's older segments are read at their own phases, and
    # every imported capsule is read at load, so each is a head input.
    chain = chain_documents(capsule)
    for group in (group for segment in chain for group in segment['groups']):
        for item in group['manifest']['entries']:
            path = item['destination_path']
            match = re.fullmatch(r'boundary-(\d+)-(\d+)-at-(\d+)\.pt', Path(path).name)
            if match is None or match[2] != match[3]:
                raise ValueError('non-boundary in recovery capsule')
            index = len(entries)
            entries.append({'path': path, 'offset': 0, 'bytes': item['bytes'], 'sha256': item['sha256']})
            boundaries.setdefault(int(match[2]), []).append((int(match[1]), index))
    capsule_indices = []
    for proof in [bound, *(segment['imported'] for segment in chain[:-1])]:
        capsule_indices.append(len(entries))
        entries.append({'path': str(proof['path']), 'offset': 0,
                        'bytes': Path(proof['path']).stat().st_size, 'sha256': proof['sha256']})
    phases = []
    for phase in manifest['read_plan']['phases']:
        name = phase['name']
        if name.startswith('forward-') and int(name.split('-')[1]) < frontier:
            continue
        indices = phase['entry_indices']
        if name == 'head':
            indices.extend(capsule_indices)
        boundary = None
        if name == f'forward-{frontier:03d}':
            boundary = frontier
        elif name.startswith('chain-') and int(name.split('-')[1]) <= frontier:
            boundary = int(name.split('-')[1])
        if boundary is not None:
            rows = sorted(boundaries.get(boundary, ()))
            if [batch for batch, _ in rows] != list(range(capsule['n_batches'])):
                raise ValueError('incomplete recovery input phase')
            indices.extend(index for _, index in rows)
        phases.append(phase)
    # A fully captured tail frontier is read while the last forward phase is
    # held. Preserve that declared phase even though no source layer replays.
    if frontier == layers:
        tail = next(p for p in original['read_plan']['phases'] if p['name'] == f'forward-{layers-1:03d}')
        tail = json.loads(json.dumps(tail))
        tail['entry_indices'].extend(index for _, index in sorted(boundaries[layers]))
        phases.insert(1, tail)
    used = sorted({i for phase in phases for i in phase['entry_indices']})
    remap = {old: new for new, old in enumerate(used)}
    entries = [entries[i] for i in used]
    cumulative = 0
    for phase in phases:
        phase['entry_indices'] = [remap[i] for i in phase['entry_indices']]
        phase['bytes'] = sum(entries[i]['bytes'] for i in phase['entry_indices'])
        cumulative += phase['bytes']
        phase['cumulative_bytes'] = cumulative
    manifest.update(entries=entries, entry_count=len(entries), total_bytes=sum(e['bytes'] for e in entries))
    manifest['read_plan'] = {'phases': phases, 'read_bytes': cumulative}
    manifest['annotations']['forward_recovery'] = bound
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', required=True)
    parser.add_argument('--capsule-sha256', required=True)
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--r9-package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-head', required=True)
    parser.add_argument('--label', default='r10',
                        help='attempt label: launch-<label>.py, template suffix, schema')
    args = parser.parse_args()
    if not re.fullmatch(r'r(1\d|[2-9]\d)', args.label):
        parser.error('--label names a post-R9 attempt, for example r11')
    label, upper = args.label, args.label.upper()
    bound = {'path': args.capsule, 'sha256': args.capsule_sha256}
    capsule, _ = _read(args.capsule, args.capsule_sha256)
    wire = Path(args.original_manifest).read_bytes()
    if hashlib.sha256(wire).hexdigest() != ORIGINAL_MANIFEST_SHA:
        raise ValueError('original scientific read manifest changed')
    original = json.loads(gzip.decompress(wire))
    manifest = recovery_manifest(original, capsule, bound)
    from prismaquant.staged_lease import sdk_submodule
    sdk_submodule('core').validate_data_manifest(manifest)
    root, prior = Path(args.output), Path(args.r9_package)
    root.mkdir(parents=True, exist_ok=False)
    wire = gzip.compress((json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode(), mtime=0)
    (root / 'adjoint-recovery-manifest.json.gz').write_bytes(wire)
    digest = hashlib.sha256(wire).hexdigest()
    names = ['prefetch-override-r7.json', 'stage-a-spec-r7.json', 'stage-a-dispatch-config.json']
    for name in names:
        shutil.copyfile(prior / name, root / name)
    template = json.loads((prior / 'stagea-512-template-r7.json').read_text())
    template['template_id'] = template['template_id'].removesuffix('-r9') + '-' + label
    (root / 'stagea-512-template-r7.json').write_text(json.dumps(template, sort_keys=True, indent=2) + '\n')
    names.append('stagea-512-template-r7.json')
    (root / 'package-pins.json').write_text(json.dumps({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                     for name in names}, sort_keys=True, indent=2) + '\n')
    launcher = (prior / 'launch-r9.py').read_text()
    launcher = launcher.replace('5fff97a621c398cbd65810cec3f535dcf121b2b5', args.source_head)
    launcher = launcher.replace("MANIFEST = PANEL / 'stage-a-recovery-20260921/adjoint-manifest.json.gz'",
                                "MANIFEST = ROOT / 'adjoint-recovery-manifest.json.gz'")
    launcher = launcher.replace(ORIGINAL_MANIFEST_SHA, digest)
    needle = "    original = result[inner_separator + 1:]\n"
    if launcher.count(needle) != 1:
        raise ValueError('R9 launcher seam changed')
    launcher = launcher.replace(needle, needle + '    original += ' + repr([
        '--forward-recovery', args.capsule, '--forward-recovery-sha256', args.capsule_sha256]) + '\n')
    # Adjoint checkpoints have fixed paths and are never overwritten; a
    # contained attempt's checkpoints would fail this one after its tail.
    needle = "            raise RuntimeError('parent must accept and pin the deployed runtime')\n"
    if launcher.count(needle) != 1:
        raise ValueError('R9 launcher runtime seam changed')
    launcher = launcher.replace(needle, needle + (
        "        stale = sorted((RUN / 'layer-quanta/adjoint/checkpoints').glob('boundary-[0-9][0-9][0-9]'))\n"
        "        if stale:\n"
        "            raise RuntimeError('adjoint checkpoint paths are occupied: ' + ', '.join(map(str, stale)))\n"))
    launcher = launcher.replace('R9', upper).replace(
        'r9.local_output.v1', label + '.forward_recovery.v1').replace('attempt-r9-', f'attempt-{label}-')
    (root / f'launch-{label}.py').write_text(launcher)
    print(json.dumps({'package': str(root), 'source_head': args.source_head,
        'data_manifest_sha256': digest, 'capsule': bound, 'frontier': capsule['frontier'],
        'phases': len(manifest['read_plan']['phases']),
        'max_phase_bytes': max(p['bytes'] for p in manifest['read_plan']['phases']),
        'runtime_pin_pending': True}, sort_keys=True))


if __name__ == '__main__':
    main()
