import json,pickle,collections,hashlib
from pathlib import Path
r=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-e2m1-01/workspace/merged-c92826fa4')
prep=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json')
cost=pickle.loads((r/'cost.pkl').read_bytes());p=json.loads(prep.read_text()); fmt='TESSERA_E2M1_K2_R896'
experts={q for q in p['formats_by_qname'] if '.experts.' in q}; actual={q for q,v in cost['costs'].items() if fmt in v};q=sorted(actual)[0]
print('example_cost',repr(cost['costs'][q])[:2500],flush=True)
print('provenance',json.dumps(cost['provenance'],default=str)[:6500],flush=True)
print('wire_entry',repr(cost['tessera_expert_wires'][q])[:6000],flush=True)
summary={'expected_expert_qnames':len(experts),'priced_a4_qnames':len(actual),'missing_qnames':sorted(experts-actual)[:20],'extra_qnames':sorted(actual-experts)[:20],'formats':cost['formats'],'example_qname':q,'prepared_missing_a4_experts':sum(fmt not in p['formats_by_qname'][q] for q in experts)}
j=json.loads((r/'cost.anchors.json').read_text());summary['calibration_equal']=j['identity']['calibration']==p['calibration_input']['provenance'];summary['encoder_source_sha256']=j['identity']['encoder_source_sha256'];summary['journal_units']=len(j['units']);summary['identity_migration']=j.get('identity_migration')
entry=next(v for v in j['units'] if v['qname']==q);outer=pickle.loads((r/'cost.anchors.json.parts'/entry['file']).read_bytes());assert hashlib.sha256(outer['payload']).hexdigest()==outer['payload_sha256']; payload=pickle.loads(outer['payload']);record=payload['wire_records'][fmt];summary['example_wire_record']=record
print('summary',json.dumps(summary,default=str),flush=True)
out=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922/coverage.json');out.write_text(json.dumps(summary,indent=2,default=str)+'\n')
