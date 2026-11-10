"""Rerunnable CPU-only native staged input and canonical run-set proof.

Run controller in an admitted Flux job from a clean detached shared checkout.
All children use flux_jobs' single baseline resource/environment profile.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prismaquant.digests import bytes_sha256hex as sha

HELPER = '/mnt/shared/flux-migration/caching/fluxcache.py'
BLOBS = (b'FIRST independently known staged bytes\x00\xff', b'SECOND independent metadata\n')


def helper(root, *args, ok=True):
    result = subprocess.run(['/usr/bin/python3', HELPER, '--root', str(root), *args], capture_output=True, text=True)
    if (result.returncode == 0) != ok:
        raise RuntimeError(f'canonical helper verdict unexpected: {result.returncode}: {result.stdout} {result.stderr}')
    return result


def warm(plan):
    root = Path(plan['root'])
    root.parent.mkdir(mode=0o700)
    helper(root, 'init', '--cap-bytes', '4096')
    for obj in plan['objects']:
        helper(root, 'warm', obj['sha256'], '--source', obj['path'], '--bytes', str(obj['bytes']))
    print('warm-complete', flush=True)


def consume(plan):
    from prismaquant import residency_map, io_engine
    from prismaquant.staged_whole_file import read_staged_whole_file
    resolver = residency_map.residency_resolver()
    assert resolver is not None and len(resolver._pins) == 2
    original_map = os.environ['FLUXCACHE_MAP_FILE']
    partial_map = Path(os.environ['TMPDIR']) / 'incomplete-map.json'
    mapping = json.loads(Path(original_map).read_text())
    mapping.pop(next(iter(mapping)))
    partial_map.write_text(json.dumps(mapping))
    os.environ['FLUXCACHE_MAP_FILE'] = str(partial_map)
    try:
        from prismaquant.staged_manifest import ManifestRefused
        try:
            residency_map.ResidencyResolver(resolver.manifest.filename, manifest_sha256=resolver.manifest.sha256)
        except ManifestRefused as exc:
            assert 'exact bound object set' in str(exc)
        else:
            raise AssertionError('incomplete inherited map was admitted')
    finally:
        os.environ['FLUXCACHE_MAP_FILE'] = original_map
    no_pins = subprocess.run(['/usr/bin/python3', '-c',
        'from prismaquant.residency_map import residency_resolver; residency_resolver()'],
        capture_output=True, text=True)
    assert no_pins.returncode != 0 and 'inherited run-set pin missing' in no_pins.stderr
    for filename, expected in zip(plan['sources'], BLOBS):
        assert read_staged_whole_file(Path(filename), sha(expected), label='native-proof') == expected
    entries = [io_engine.ReadEntry(str(i), filename, len(raw), len(raw), len(raw), sha(raw),
        lambda buffer, receipt, staged: (bytes(buffer), None), 0)
        for i, (filename, raw) in enumerate(zip(plan['sources'], BLOBS))]
    with io_engine.read_stream(entries, budget=io_engine.FixedBudget(buffer_bytes=1024, headroom=4096)) as stream:
        assert [item.value for item in stream.take(0)] == list(BLOBS)
    # Cleaner must fail to evict either object while the real foreground
    # reader owns inherited pins. No mock or second locking protocol.
    result = helper(plan['root'], 'clean', '--target-bytes', '0', ok=False)
    pinned = {json.loads(line).get('key') for line in result.stdout.splitlines()
              if json.loads(line).get('event') == 'pinned'}
    assert pinned == {obj['sha256'] for obj in plan['objects']}, result.stdout
    assert all((Path(plan['root']) / 'objects' / key).exists() for key in pinned)
    assert resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == 136
    assert not any(name == 'prismabuild' or name.startswith('prismabuild.') for name in sys.modules)
    report = {'known_bytes': [sha(raw) for raw in BLOBS], 'object_offsets': [13, 13],
              'pins': sorted(pinned), 'cleaner_exit': result.returncode, 'pb_imports': [],
              'incomplete_inherited_map_refused': True,
              'missing_inherited_pins_refused': True,
              'host': __import__('socket').gethostname(), 'report': resolver.report()}
    resolver.close()
    Path(os.environ['OUT'], 'reader-proof.json').write_text(json.dumps(report, indent=2))


def cleanup(plan):
    root = Path(plan['root'])
    helper(root, 'clean', '--target-bytes', '0')
    assert not list((root / 'objects').iterdir())
    shutil.rmtree(root.parent)
    assert not root.parent.exists()
    Path(os.environ['OUT'], 'cache-cleanup.json').write_text(json.dumps({'path': str(root.parent), 'absent': True}))


def controller():
    from prismaquant import tessera_campaign_rows as rows, flux_jobs
    from prismaquant.staged_manifest import publish_staged_json, SCHEMA
    out = Path(os.environ['OUT'])
    phase = out / 'phase'; phase.mkdir()
    root = str(Path(os.environ['TMPDIR']) / 'cache')
    source_paths, objects, extents = [], [], []
    for i, raw in enumerate(BLOBS):
        source = str(out / f'canonical-does-not-exist-{i}')
        blob = b'NOT-A-HEADER:' + raw + b':TRAILER'
        obj = {'path': str(phase / f'published-{i}'), 'bytes': len(blob), 'sha256': sha(blob)}
        Path(obj['path']).write_bytes(blob)
        objects.append(obj); source_paths.append(source)
        extents.append({'source': {'path': source, 'bytes': len(raw), 'sha256': sha(raw)},
            'object': obj, 'source_offset': 0, 'object_offset': 13, 'length': len(raw), 'sha256': sha(raw)})
    contract = out / 'read-contract.json'
    read_sha = publish_staged_json(contract, {'schema': 'prismaquant.read_contract.v1', 'entries': [
        {'path': e['source']['path'], 'offset': 0, 'bytes': e['length'], 'sha256': e['sha256']} for e in extents]})
    manifest = out / 'staged.json'
    manifest_sha = publish_staged_json(manifest, {'schema': SCHEMA, 'phase_id': 'native-unit2-cpu-fixture',
        'read_contract_sha256': read_sha, 'producer': {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'receipt_sha256': sha(b'CPU fixture publisher, no scientific authority')},
        'lifetime': {'mode': 'host_local_cache', 'root': root, 'owner_host': 'sparky'}, 'entries': extents})
    plan = out / 'plan.json'; publish_staged_json(plan, {'root': root, 'sources': source_paths, 'objects': objects})
    binding = {'schema': 'prismaquant.staged_launch.v1', 'path': str(manifest), 'sha256': manifest_sha,
               'read_contract_path': str(contract), 'read_contract_sha256': read_sha, 'allowed_tiers': 'ssd'}
    checkout = Path.cwd()
    def row(mode, staged=False):
        value = {'argv': ['/usr/bin/python3','-m','tools.unit2_staged_proof',mode,str(plan)],
            'cwd': str(checkout), 'demand': {'cpu': 2, 'mem_gb': 4, 'gpu': 0},
            'tags': ['host:sparky'], 'timeout_s': 180, 'env': {}}
        if staged:
            value.update(staged_input=binding, data_manifest=str(contract), residency='stage')
        return value
    submitted = []
    try:
        warm_rows = [row('warm')]
        producer = rows.submit_rows(warm_rows, out/'warm'); submitted.extend(producer)
        consumer_rows = [row('consume', True)]
        consumers = rows.submit_rows(consumer_rows, out/'consumer', afterok=producer); submitted.extend(consumers)
        # Exercise exact-intent saved-ID resume before any output is trusted.
        assert rows.submit_rows(consumer_rows, out/'consumer', afterok=producer) == consumers
        for submission in submitted:
            result = flux_jobs.collect_job(submission)
            assert result.status == 'succeeded', result
        rows.require_native_success(out/'consumer', 1, rows=consumer_rows)
        launch = json.loads((consumers[0].record_dir/'launch.json').read_text())
        assert launch['staged_input'] == binding and launch['afterok'] == [producer[0].job_id]
        proof = json.loads((consumers[0].record_dir/'reader-proof.json').read_text())
        assert proof['known_bytes'] == [sha(raw) for raw in BLOBS]
        assert proof['cleaner_exit'] != 0 and len(proof['pins']) == 2
        finished = rows.submit_rows([row('cleanup')], out/'cache-cleanup', afterok=consumers)
        submitted.extend(finished)
        assert flux_jobs.collect_job(finished[0]).status == 'succeeded'
        assert json.loads((finished[0].record_dir/'cache-cleanup.json').read_text())['absent'] is True
        (out/'canonical-helper.sha256').write_text(sha(Path(HELPER).read_bytes())+'\n')
        (out/'native-proof.json').write_text(json.dumps({'jobs': [s.job_id for s in submitted], 'passed': True}, indent=2))
        print('NATIVE_STAGED_PROOF_OK', [s.job_id for s in submitted], flush=True)
    finally:
        for submission in submitted:
            result = flux_jobs.collect_job(submission, wait=True)
            print('collected', submission.job_id, result.status, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controller()
    else:
        {'warm': warm, 'consume': consume, 'cleanup': cleanup}[sys.argv[1]](json.loads(Path(sys.argv[2]).read_text()))
