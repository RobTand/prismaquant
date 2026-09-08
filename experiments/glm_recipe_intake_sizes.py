"""Metadata-only sizing of existing whole-census recipe intake owners; use PB."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census',required=True)
    parser.add_argument('--census-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raw=Path(args.census).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.census_sha256:
        raise ValueError('census bytes changed')
    census=json.loads(raw)
    shapes=census['unit_shapes']
    columns=Counter(shape[1] for shape in shapes.values())
    by_group={group:sum(4*shapes[name][1]**2 for name in names)
        for group,names in census['anchor_groups'].items()}
    result=dict(schema='prismaquant.glm_recipe_intake_sizes.v1',status='DERIVED_METADATA_ONLY',
        census=dict(path=args.census,sha256=args.census_sha256),units=len(shapes),
        groups=len(by_group),column_histogram=dict(columns),
        full_hessian_logical_bytes=sum(4*shape[1]**2 for shape in shapes.values()),
        largest_group_hessian_logical_bytes=max(by_group.values()),
        hessian_logical_bytes_by_group=by_group,
        tensor_contract='Per-unit K by K FP32 Hessian; excludes serialization, Python, copies and allocator overhead.',
        scope='Logical original full-census H ownership; no payload reads, no deduplication or measured process-memory claim.',
        consumer='tools/dispatch_tessera_campaign.py:885-934 retains all row H tensors before write_export_inputs.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();output.write_bytes(encoded)
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(encoded).hexdigest(),
        full_hessian_logical_bytes=result['full_hessian_logical_bytes'],units=result['units'])))

if __name__=='__main__':
    main()
