"""Audit full canonical containers beside the asserted dict-key projection."""
import gzip,json,sys
from pathlib import Path
def load(path):
    p=Path(path);raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
before,after=(load(p) for p in sys.argv[1:3])
def decode(value):
    if not isinstance(value,list):return value
    if len(value)==2 and value[0]=='dict':return {decode(k):decode(v) for k,v in value[1]}
    if len(value)==2 and value[0] in ('list','tuple','set','dict_values'):return [decode(v) for v in value[1]]
    if len(value)==2 and value[0]=='float':return float.fromhex(value[1])
    return [decode(v) for v in value]
def differences(a,b,path=''):
    if a==b:return []
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys(),('dictionary keys changed',path)
        return [p for k in a for p in differences(a[k],b[k],path+'.'+str(k))]
    if isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b),('list length changed',path)
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,f'{path}[{i}]')]
    return [path.lstrip('.')]
rows=[]
a=before['membership_containers'];b=after['membership_containers']
assert len(a)==len(b)
for x,y in zip(a,b):
    assert all(x[k]==y[k] for k in ('node','expression','operand'))
    changed=differences(decode(x['value']),decode(y['value']))
    if changed:rows.append({k:x[k] for k in ('node','expression','operand')}|{'changed_unasserted_fields':changed})
assert all(field.endswith('.solver_seconds') for row in rows for field in row['changed_unasserted_fields']),rows
original=json.loads(Path(sys.argv[1]+'.source.json').read_text())
seen={r['expression'] for r in before['assertions']}
assert set(original)<=seen
assert all(before['outcomes'][node+'::call']==after['outcomes'][node+'::call']=='passed' for node in before['nodes'])
report={'original_cases':len(before['nodes']),'original_source_assertions':len(original),'all_original_assertions_evaluated':True,'all_original_cases_passed':True,'membership_containers':len(a),'changed_unasserted_container_values':rows,'only_unasserted_solver_seconds_differ':True}
Path(sys.argv[3]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
