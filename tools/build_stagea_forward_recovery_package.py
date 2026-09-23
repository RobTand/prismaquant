"""Seal a recovery launcher and ordinary windowed input manifest from recovery proof.

``--label`` names the attempt (r10, r11, ...). The launcher is rendered from
``tools/templates/stagea_forward_recovery_launch.py.template``: the campaign
fields come from a reviewed declaration (``--campaign-fields``), and the
attempt fields (label, source head, recovery manifest digest, capsule) from
this build. Every template field must be declared, and no other.

The recovery manifest keeps the source run's reads, less the head walk's:
Stage A takes its head from the prepared completion (PQ #1051), so a source
manifest built before #1051 declares reads the relaunch never makes
(``stage_a_head.drop_source_head_walk_reads``). ``--plan`` is the plan the
campaign binds and the source manifest names; its ``inputs`` say which head
entries are the walk's. ``launch-fields.json`` records the entries and bytes
left out.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import string
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_forward_resume import _read, chain_documents
from prismaquant.stage_a_head import drop_source_head_walk_reads

TEMPLATE = Path(__file__).resolve().parent / 'templates' / 'stagea_forward_recovery_launch.py.template'
MANIFEST_NAME = 'adjoint-recovery-manifest.json.gz'
#: Campaign fields a reviewed declaration supplies. ``original_manifest_sha256``
#: binds the scientific read manifest the recovery manifest is derived from.
CAMPAIGN_FIELDS = frozenset((
    'artifact_budget_bytes', 'bound_cpus', 'bound_demand', 'campaign_bindings',
    'campaign_name', 'campaign_scope', 'original_manifest_sha256', 'panel', 'records',
    'reviewed_cpus', 'reviewed_demand', 'run', 'scope_refusal'))


def render_launcher(campaign, *, label, source_head, manifest_sha256, capsule):
    """Render the attempt launcher; Python literals are substituted by repr."""
    if set(campaign) != CAMPAIGN_FIELDS:
        raise ValueError('launcher fields differ from the declared set: missing '
                         f'{sorted(CAMPAIGN_FIELDS - set(campaign))}, '
                         f'unexpected {sorted(set(campaign) - CAMPAIGN_FIELDS)}')
    if not re.fullmatch(r'r(1\d|[2-9]\d)', label):
        raise ValueError('launcher fields: label names a post-R9 attempt, for example r11')
    literal = {name: repr(value) for name, value in campaign.items()
               if name not in ('campaign_name', 'original_manifest_sha256')}
    fields = {**literal, 'campaign_name': str(campaign['campaign_name']),
              'label': label, 'LABEL': label.upper(),
              'source_head': repr(str(source_head)),
              'manifest_name': repr(MANIFEST_NAME),
              'manifest_sha256': repr(str(manifest_sha256)),
              'capsule_path': repr(str(capsule['path'])),
              'capsule_sha256': repr(str(capsule['sha256'])),
              'first_record': repr(str(campaign['records']).rstrip('/') + '/layer-000.json')}
    template = string.Template(TEMPLATE.read_text())
    used = {m.group('named') or m.group('braced') for m in template.pattern.finditer(template.template)
            if m.group('named') or m.group('braced')}
    if used != set(fields):
        raise ValueError(f'launcher fields and template placeholders differ: {sorted(used ^ set(fields))}')
    return template.substitute(fields)


def load_original(path, campaign, plan):
    """``(manifest, dropped)``: the source run's manifest, less the head walk's reads.

    ``plan`` is the pinned ``{path, sha256}`` of the campaign's plan. Refuses a
    manifest whose digest is not the campaign's, and a plan the campaign does
    not bind.
    """
    wire = Path(path).read_bytes()
    if hashlib.sha256(wire).hexdigest() != campaign.get('original_manifest_sha256'):
        raise ValueError('original scientific read manifest changed')
    if plan['sha256'] != campaign.get('campaign_bindings', {}).get('plan_sha256'):
        raise ValueError('the plan is not the one the campaign binds')
    return drop_source_head_walk_reads(json.loads(gzip.decompress(wire)), plan)


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
    parser.add_argument('--plan', required=True,
                        help='the plan the campaign binds; its inputs say which '
                             'head entries are the head walk\'s (PQ #1051)')
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--campaign-fields', required=True,
                        help='reviewed campaign launcher declaration (JSON), for example '
                             'tools/templates/glm_full512_stagea_campaign_fields.json')
    parser.add_argument('--r9-package', required=True,
                        help='reviewed package whose spec, prefetch, dispatch and template files are reused')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-head', required=True)
    parser.add_argument('--label', default='r10',
                        help='attempt label: launch-<label>.py, template suffix, schema')
    args = parser.parse_args()
    label = args.label
    campaign = json.loads(Path(args.campaign_fields).read_text())
    bound = {'path': args.capsule, 'sha256': args.capsule_sha256}
    capsule, _ = _read(args.capsule, args.capsule_sha256)
    original, dropped = load_original(args.original_manifest, campaign,
                                      {'path': args.plan, 'sha256': args.plan_sha256})
    manifest = recovery_manifest(original, capsule, bound)
    from prismaquant.staged_lease import sdk_submodule
    sdk_submodule('core').validate_data_manifest(manifest)
    wire = gzip.compress((json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode(), mtime=0)
    digest = hashlib.sha256(wire).hexdigest()
    # Render before writing anything, so a field error leaves no package.
    launcher = render_launcher(campaign, label=label, source_head=args.source_head,
                               manifest_sha256=digest, capsule=bound)
    root, prior = Path(args.output), Path(args.r9_package)
    root.mkdir(parents=True, exist_ok=False)
    (root / MANIFEST_NAME).write_bytes(wire)
    names = ['prefetch-override-r7.json', 'stage-a-spec-r7.json', 'stage-a-dispatch-config.json']
    for name in names:
        shutil.copyfile(prior / name, root / name)
    template = json.loads((prior / 'stagea-512-template-r7.json').read_text())
    template['template_id'] = template['template_id'].removesuffix('-r9') + '-' + label
    (root / 'stagea-512-template-r7.json').write_text(json.dumps(template, sort_keys=True, indent=2) + '\n')
    names.append('stagea-512-template-r7.json')
    (root / 'launch-fields.json').write_text(json.dumps({
        'campaign': campaign, 'label': label, 'source_head': args.source_head,
        'manifest_sha256': digest, 'capsule': bound, 'head_walk_reads_dropped': dropped,
        'template_sha256': hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()},
        indent=2) + '\n')
    names.append('launch-fields.json')
    (root / 'package-pins.json').write_text(json.dumps({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                     for name in names}, sort_keys=True, indent=2) + '\n')
    (root / f'launch-{label}.py').write_text(launcher)
    print(json.dumps({'package': str(root), 'source_head': args.source_head,
        'data_manifest_sha256': digest, 'capsule': bound, 'frontier': capsule['frontier'],
        'head_walk_reads_dropped': dropped,
        'phases': len(manifest['read_plan']['phases']),
        'max_phase_bytes': max(p['bytes'] for p in manifest['read_plan']['phases']),
        'runtime_pin_pending': True}, sort_keys=True))


if __name__ == '__main__':
    main()
