"""Summarize retained remote pytest phases and sampled call stacks."""
from collections import Counter, defaultdict
import json,sys
from pathlib import Path
folder=Path(sys.argv[1]); data=json.loads((folder/'timing.json').read_text())
phases=Counter();nodes=defaultdict(Counter)
for r in data['reports']:
    phases[r['phase']]+=r['seconds'];nodes[r['nodeid']][r['phase']]+=r['seconds']
leaf=Counter();inclusive=Counter()
for s in data['samples']:
    stack=s['stack'].splitlines()
    if stack:leaf[stack[0]]+=s['count']
    for frame in set(stack):inclusive[frame]+=s['count']
result={'exitstatus':data['exitstatus'],'phase_seconds':dict(phases),'total_phase_seconds':sum(phases.values()),'slowest_tests':[dict(node=n,phases=dict(v),seconds=sum(v.values())) for n,v in sorted(nodes.items(),key=lambda p:-sum(p[1].values()))[:15]],'samples':sum(s['count'] for s in data['samples']),'top_leaf':leaf.most_common(20),'top_inclusive':inclusive.most_common(30)}
(folder/'profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
