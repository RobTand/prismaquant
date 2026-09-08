"""Generate review-only rows with the existing dispatcher; never submit GPU work.

This is deliberately not cmd_plan: the original completed capture is not frozen
here. Root must derive the actual 132-row plan from its accepted capture and then
instrument the two actual master rows, preserving those same row objects for
qualification and full submission.
"""
import json
from pathlib import Path
from tools.dispatch_tessera_campaign import load_spec, _row

HERE = Path(__file__).parent
ROOT = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
OUT = ROOT/'full-anchor-preparation-01'
WORKSPACE = OUT/'workspace'
spec = load_spec(HERE/'anchor-spec.draft.json')
spec['cpus'] = 8
rows = []
for row_id, memory in [('row-0076', 103), ('row-0087', 35)]:
    row_dir = WORKSPACE/'rows'/row_id
    original = ['--model', spec['model'], '--out', str(row_dir/'cost.pkl'),
        '--cache-dir', str(row_dir/'cache'), '--checkpoint', str(row_dir/'cost.anchors.json'),
        '--units', str(WORKSPACE/'units'/f'{row_id}.json'),
        '--calibration-census', str(WORKSPACE/'census.json'), *spec['campaign_argv'],
        '--calibration-cache', str(ROOT/'workspace/calibration-cache/capture_manifest.json'),
        '--calibration-cache-sha256', 'UNFROZEN_CAPTURE_SHA256']
    row = _row(spec, ['--selected-anchors', '--anchor-profile-calls', '0,31',
        '--anchor-trace-max-bytes', '536870912', '--evidence-out', str(row_dir/'profile'),
        '--', *original], mem_gb=memory, timeout_s=14400,
        module='experiments.glm_full_capture_profile')
    row.update(measurement=True, host_class='gb10')
    rows.append(row)
result = OUT/'qualification-manifest.draft.json'
result.write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
print(f'Wrote two review-only rows to {result}; UNFROZEN_CAPTURE_SHA256 refuses execution.')
