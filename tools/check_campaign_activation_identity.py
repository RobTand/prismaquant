"""Read-only identity audit of bound prepared activation records (no tensor loads)."""
import argparse, collections, json, os, pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.stage_inputs import read_bound as _read_bound
from prismaquant.joint_aura import activation_identity
from prismaquant import format_registry
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--prepared',required=True);p.add_argument('--prepared-sha256',required=True);a=p.parse_args()
plan=json.loads(_read_bound({'path':a.plan,'sha256':a.plan_sha256},'plan')); prepared=json.loads(_read_bound({'path':a.prepared,'sha256':a.prepared_sha256},'prepared')); cache=pickle.loads(_read_bound(prepared['production_cache'],'production cache'))
os.environ['PRISMAQUANT_PROD_ACT_SCALES']=plan['execution']['production_act_scales']
counts=collections.Counter();mismatch=[];specs={}
for (name,fmt),record in cache.metadata['verified_cells'].items():
 spec=specs.setdefault(fmt,format_registry.get_format(fmt)); expected=activation_identity(spec,cache.activation_max_abs,name); observed=record['activation'];counts[(fmt,observed['clip_enabled'])]+=1
 if observed!=expected:mismatch.append({'qname':name,'format':fmt,'observed':observed,'expected':expected})
print(DIRECT_ASCII_SPACED_LAX.text({'plan_sha256':a.plan_sha256,'prepared_sha256':a.prepared_sha256,'production_act_scales':os.environ['PRISMAQUANT_PROD_ACT_SCALES'],'cells':sum(counts.values()),'counts':[{'format':f,'clip_enabled':c,'cells':v} for (f,c),v in sorted(counts.items())],'mismatches':len(mismatch),'examples':mismatch[:3]}))
raise SystemExit(bool(mismatch))
