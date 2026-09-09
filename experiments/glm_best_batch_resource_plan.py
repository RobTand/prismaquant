"""Derive wider actual-encode batch reservations with the existing planner."""
import copy
import json
import math
from pathlib import Path
from tools.dispatch_tessera_campaign import _streamed_resource_plan

base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02')
spec=json.loads((base/'anchor-spec.frozen.json').read_text())
census=json.loads((base/'workspace/census.json').read_text())
units=json.loads((base/'workspace/units/row-0076.json').read_text())
members=sorted(n for g in units['groups'] for n in g['members'])
reports=[]
for width in (8,16,32):
    current=copy.deepcopy(spec)
    current['campaign_argv'][current['campaign_argv'].index('--anchor-batch-size')+1]=str(width)
    resource=_streamed_resource_plan(current,census,members,selected_source=True)
    assert resource['encoder_memo_capacity']==width and resource['source_forward_count']==0
    reports.append(dict(batch=width,base_gib=math.ceil(resource['memory_bytes']/2**30),
        observer_gib=4,final_gib=math.ceil(resource['memory_bytes']/2**30)+4,resource=resource))
print(json.dumps(dict(status='DERIVED_NOT_MEASURED',row='row-0076',units=len(members),reports=reports),sort_keys=True))
