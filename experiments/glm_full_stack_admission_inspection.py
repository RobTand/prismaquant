"""Recompute current dispatcher admission from metadata only; execute through PB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.dispatch_tessera_campaign import _streamed_resource_plan, partition_rows_by_fit


def bound(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'changed metadata: {path}')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['census', 'census-sha256', 'spec', 'spec-sha256', 'output']:
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    census = bound(args.census, args.census_sha256)
    spec = bound(args.spec, args.spec_sha256)
    rows, selected = {}, []
    for index, (group, members) in enumerate(sorted(census['anchor_groups'].items())):
        plan = _streamed_resource_plan(spec, census, members, selected_source=True)
        row = f'row-{index:04d}'
        rows[row] = dict(group=group, members=sorted(members), resources=plan,
            mem_gib_ceil=(plan['memory_bytes']+1024**3-1)//1024**3,
            phase_bytes={name:sum(phase.values()) for name,phase in plan['phases'].items()})
        selected.extend(members)
    if len(selected) != len(set(selected)) or set(selected) != set(census['unit_shapes']):
        raise ValueError('group partition is not the exact complete unit roster')
    admitted, declined = partition_rows_by_fit(
        {key: row['mem_gib_ceil'] for key,row in rows.items()}, 1, spec['box_memory_gb'])
    result = dict(schema='prismaquant.glm_full_stack_admission_inspection.v1',
        status='DERIVED_METADATA_ONLY_NOT_NATIVE_QUALIFICATION',
        census=dict(path=args.census,sha256=args.census_sha256),
        source_spec=dict(path=args.spec,sha256=args.spec_sha256,
            scope='Existing draft consulted only for resource fields; no menu/budget or execution approval.'),
        source_files={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['prismaquant/autoscale.py','tools/dispatch_tessera_campaign.py']},
        groups_per_row=1, rows_per_box_fit_check=1, box_memory_gib=spec['box_memory_gb'],
        units=len(selected),groups=len(rows),stack_sample=None,full_roster=True,
        admissible=admitted,inadmissible=declined,rows=rows,
        scope='No model/capture tensor payload reads, capture issuance, plan publication, probe or native execution; source headers and census only.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();out.write_bytes(raw)
    print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(raw).hexdigest(),
        groups=len(rows),units=len(selected),admissible=len(admitted),declined=len(declined),
        maximum_bytes=max(row['resources']['memory_bytes'] for row in rows.values()))))


if __name__=='__main__':
    main()
