import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from prismaquant.cost_stage_checkpoint import _load_unit, canonical_json_sha256, unit_path

base = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02')
capture_sha = 'f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c'
pricing_sha = '5139995a8c48a3fc3a2ffc91c5e7f5a54d45fc4b34cccf9b8a8fd891c334b2e7'
rows = [('row-0003', '74b588f7464a431c8eff2420589319ead099d2add5f5a4901a0b1f32f3b32481'), ('row-0004', '83db680aad9daf6ea7aadc3564bd3c6e7b79769b6cfa065dcea1313b181b5714'), ('row-0118', '90d0fdfa4e7ddb28494c2e62ce69bde4f741f617437f85a91c844d52434d7668')]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def audit(item):
    row_id, action = item
    terminal = json.loads((Path('/mnt/shared/prismabuild-fleet/pb-queue/done') / (action + '.json')).read_text())
    assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
    row = base / 'workspace/rows' / row_id
    units = json.loads((base / 'workspace/units' / (row_id + '.json')).read_text())
    names = {name for group in units['groups'] for name in group['members']}
    cost = pickle.loads((row / 'cost.pkl').read_bytes())
    assert cost['schema'] == 'prismaquant.tessera_campaign_cost.v1'
    assert cost['currency'] == 'output_mse_under_route_activation_contract'
    assert set(cost['costs']) == names
    provenance = cost['provenance']
    assert provenance['stopped_early'] is False
    assert provenance['calibration_cache']['sha256'] == capture_sha
    selected = provenance['selected_source_preparation']
    assert selected['source_forward_count'] == 0 and set(selected['units']) == names
    execution = selected['capture_load_execution']
    assert digest(Path(execution['path'])) == execution['sha256']
    checkpoint_path = row / 'cost.anchors.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    seal = canonical_json_sha256(checkpoint['identity'], where='root artifact audit')
    assert seal == checkpoint['identity_sha256']
    assert checkpoint['identity']['prismaquant_source_sha256'] == pricing_sha
    assert set(checkpoint['identity']['units']) == names
    artifacts = {}
    wire_count = measured = total = 0
    for name in sorted(names):
        unit_file = unit_path(checkpoint_path.with_name(checkpoint_path.name + '.parts'), name)
        state = _load_unit(unit_file, stage=checkpoint['stage'], qname=name, identity_sha256=seal)
        anchors = {entry['format_name']: entry for entry in state['anchors']}
        assert len(anchors) == len(state['anchors'])
        measured_formats = {fmt for fmt, value in cost['costs'][name].items() if value['output_mse_measured']}
        assert set(anchors) == set(state['wire_records']) == measured_formats
        for fmt, value in cost['costs'][name].items():
            total += 1
            assert math.isfinite(value['output_mse']) and value['output_mse'] >= 0
            assert value['hessian_identity']['reference_binding']['canonical_capture_sha256'] == capture_sha
            if fmt not in measured_formats:
                continue
            measured += 1
            anchor = anchors[fmt]
            assert value['output_mse'] == anchor['dloss']
            record = state['wire_records'][fmt]
            wire = row / 'cache/wire' / record['file']
            assert not wire.is_symlink() and wire.resolve().parent == (row / 'cache/wire').resolve()
            assert wire.stat().st_size == record['blob_bytes'] == value['wire_bytes']
            assert digest(wire) == record['blob_sha256']
            wire_count += 1
            artifacts[str(wire)] = {'bytes': wire.stat().st_size, 'sha256': record['blob_sha256']}
        artifacts[str(unit_file)] = {'bytes': unit_file.stat().st_size, 'sha256': digest(unit_file)}
    for file in [row / 'cost.pkl', checkpoint_path, *(row / 'cache').glob('*.pt')]:
        artifacts[str(file)] = {'bytes': file.stat().st_size, 'sha256': digest(file)}
    return {'row': row_id, 'action': action, 'units': sorted(names), 'cost_rows': total,
            'measured': measured, 'interpolated': total-measured, 'actual_wire_hashes_checked': wire_count,
            'rounds': provenance['rounds_run'], 'surfaces': provenance['surfaces'],
            'capture_load_execution': execution, 'memory_guard': selected.get('memory_guard'),
            'source_forward_count': 0, 'artifacts': artifacts}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    reports = list(executor.map(audit, rows))
print(json.dumps({'rows': reports, 'status': 'PASS'}, sort_keys=True))
