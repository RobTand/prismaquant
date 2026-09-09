"""Verify the resident-H native comparison against PB output and artifacts."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--action', required=True)
    a = p.parse_args()
    root = Path(a.root)
    fleet = Path('/mnt/shared/prismabuild-fleet')
    terminal = json.loads((fleet/'pb-queue/done'/f'{a.action}.json').read_text())
    assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
    receipt = json.loads((fleet/'cas/actions/v3'/a.action[:2]/f'{a.action}.json').read_text())
    body = {k:v for k,v in receipt.items() if k != 'receipt_sha256'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest() == receipt['receipt_sha256']
    meta = receipt['result']
    payload = fleet/'cas/blobs'/meta['sha256'][:2]/meta['sha256']
    assert digest(payload) == meta['sha256'] and payload.stat().st_size == meta['bytes']
    r = json.loads((root/'result.json').read_text())
    printed = [json.loads(line) for line in payload.read_text().splitlines()
               if line.startswith('{"schema": "prismaquant.glm_hessian_seal_profile.v1"')]
    assert printed == [r] and r['status'] == 'complete' and r['returncode'] == 0
    assert not r['full_campaign_complete'] and r['exact_wire_and_score_parity']
    assert r['resident_units'] == 864 and r['resident_hessian_bytes'] == 43486543872
    assert r['producer_source_sha256'] == '959a1a43b26865e5e04dfacbabd627634ade4537979568af9c00c76d9604f9ea'
    capture = r['reference_capture']
    assert digest(capture['path']) == capture['sha256']
    assert json.loads(Path(capture['path']).read_text())['capture_sha256'] == capture['capture_sha256']
    assert len(r['arms']) == 6
    profiles = []
    for i, arm in enumerate(r['arms']):
        enabled = i in (1, 3, 4)
        assert arm['resident'] == enabled and arm['profiled'] == (i < 2)
        assert arm['capture_sha256'] == capture['capture_sha256']
        assert arm['tensor_identity'] == dict(calls=0 if enabled else 864, bytes=0 if enabled else r['resident_hessian_bytes'])
        assert abs(arm['finished_unix'] - arm['started_unix'] - arm['seconds']) < .01
        if enabled:
            reader = arm['reader']
            assert reader['resident_bound'] and reader['committed_units'] == 864
            assert reader['loaded_entries'] == reader['source_read_bytes'] == 0
        if i < 2:
            meta = arm['profile']; path = Path(meta['path'])
            assert digest(path) == meta['sha256'] and path.stat().st_size == meta['bytes']
            events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
            for name in ('construct_and_seal_resident_capture', 'consume_one_actual_hessian'):
                assert len([e for e in events if e.get('name') == name and e.get('ph') == 'X']) == 1
            kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
            assert kernels
            profiles.append(dict(sha256=meta['sha256'], kernels=len(kernels),
                kernel_us=sum(e['dur'] for e in kernels)))
    assert r['arms'][1]['factor_and_metric_exact_parity']
    assert terminal['detail']['profile']['blob_sha256'] == r['arms'][1]['profile']['sha256']
    assert digest(r['reference']['path']) == r['reference']['sha256']
    old = {s['qname']:s for s in json.loads(Path(r['reference']['path']).read_text())['arms'][0]['signatures']}
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    m = json.loads((root/'cost.anchors.json').read_text())
    assert len(r['signatures']) == len(r['scheduled']) == len(r['prepare_calls']) == 16
    for sig in r['signatures']:
        name = sig['qname']
        s = _load_unit(unit_path(root/'cost.anchors.json.parts', name), stage=m['stage'], qname=name, identity_sha256=m['identity_sha256'])
        assert len(s['anchors']) == len(s['wire_records']) == 1
        anchor = s['anchors'][0]; wire = s['wire_records'][anchor['format_name']]
        path = root/'cache/wire'/wire['file']
        assert digest(path) == wire['blob_sha256'] == sig['wire_sha256'] == old[name]['wire_sha256']
        assert path.stat().st_size == wire['blob_bytes']
        assert anchor['dloss'] == sig['dloss'] == old[name]['dloss']
    observer = json.loads((root/'observer/result.json').read_text())
    assert observer['status'] == 'complete' and not observer['errors']
    nd = [json.loads(line) for line in (root/'observer/netdata.jsonl').read_text().splitlines()]
    hosts = {host:len([v for v in nd if v['host'] == host]) for host in ('sparky','sparklina')}
    assert all(hosts.values())
    power = []
    for source in json.loads((root/'root-pqteld-sources.json').read_text()):
        path = root/(source['host']+'-pqteld.csv')
        assert digest(path) == source['sha256']
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == source['rows']
        assert float(rows[0]['epoch_ms'])/1000 <= r['started_unix'] < r['finished_unix'] <= float(rows[-1]['epoch_ms'])/1000
        power.append(dict(host=source['host'], samples=len(rows)))
    assert {v['host'] for v in power} == set(hosts)
    guard = r['completed_guard']
    assert guard['peak_conservative_bytes'] <= guard['budget_bytes'] - guard['margin_bytes']
    plain = sum(v['seconds'] for v in r['arms'][2:] if not v['resident']) / 2
    resident = sum(v['seconds'] for v in r['arms'][2:] if v['resident']) / 2
    print(json.dumps(dict(status='PASS', action=a.action, host=terminal['claimed_host'],
        plain_mean_seconds=plain, resident_mean_seconds=resident, saved_seconds=plain-resident,
        profiles=profiles, netdata_samples=hosts, power=power, prepare_calls=r['prepare_calls'],
        wire_and_score_matches=16, full_campaign_complete=False,
        limitations=['Capture sealing only; whole-model performance is unmeasured.',
                    'Resident seal intervals are too short for a precise 2 Hz energy comparison.']), sort_keys=True))


if __name__ == '__main__':
    main()
