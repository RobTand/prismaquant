"""Header/logical-body forecast only; runtime admission uses actual PWC archives."""
import collections,json,math,pathlib,pickle,re
import torch
from prismaquant import format_registry as fr
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
from prismaquant.joint_retained_window_plan import RetainedTarget,RetainedWindowBudget,plan_retained_targets
G=1024**3
base=pathlib.Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace')
source=json.load(open('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/source-tensors.json'))
body=collections.Counter()
for row in source:
 m=re.search(r'\.layers\.(\d+)\.',row['name'])
 if m and int(m[1])<45:body[int(m[1])]+=row['bytes']
head=sum(row['bytes'] for row in source if row['name'] in ('model.language_model.embed_tokens.weight','lm_head.weight','model.language_model.norm.weight'))
budget=RetainedWindowBudget(physical_limit_bytes=104*G,safety_margin_bytes=2*G,metadata_reserve_bytes=20*G,runtime_reserve_bytes=4*G,workspace_reserve_bytes=16*G,boundary_reserve_bytes=17*G//8,auxiliary_reserve_bytes=2*G,load_buffer_bytes=G//2,read_page_reserve_bytes=G//2,candidate_delta_bytes=G//4,statistics_cap_bytes=32*G,retained_render_cap_bytes=32*G,max_windows_per_layer=16)
c=json.load(open(base/'census.json'));p=pickle.load(open(base/'merged/cost.pkl','rb'));layers=collections.defaultdict(dict)
for name,rows in p['costs'].items():layers[int(re.search(r'\.layers\.(\d+)\.',name)[1])][name]=[f for f,r in rows.items() if r.get('cost_source')=='tessera_campaign_measured']
results=[]
for layer,units in sorted(layers.items()):
 mods={n:torch.nn.Linear(c['unit_shapes'][n][1],c['unit_shapes'][n][0],bias=False,device='meta',dtype=torch.bfloat16) for n in units}
 specs={n:{f:fr.get_format(f) for f in fmts} for n,fmts in units.items()}
 statsplan=plan_joint_statistics_target_windows(mods,specs,max_statistics_bytes=32*G,activation_max_abs={n:c['max_abs'][n] for n in units})
 targets=[RetainedTarget(t.name,t.statistics_bytes,2*math.prod(t.shape)*len(units[t.name]),2*math.prod(t.shape),4*math.prod(t.shape),len(units[t.name])) for t in statsplan.targets]
 source_bytes=head+body[layer]+body[layer-1 if layer>0 else 1]
 plan=plan_retained_targets(targets,budget=budget,source_bytes=source_bytes,footprint_scope='logical_bf16_geometry_forecast')
 record=dict(layer=layer,source_bytes=source_bytes,baseline_windows=len(statsplan.windows),windows=len(plan.windows),window_units=[len(w.names) for w in plan.windows],window_statistics_bytes=[w.statistics_bytes for w in plan.windows],window_render_bytes=[w.render_bytes for w in plan.windows],peak_planned_bytes=max(w.peak_planned_bytes for w in plan.windows),decoded_bytes=sum(t.render_bytes for t in targets));results.append(record)
 print(json.dumps(record),flush=True)
windows=sum(r['windows'] for r in results);decoded=sum(r['decoded_bytes'] for r in results)
result=dict(schema='glm.retained_window_forecast.v1',status='conditional_geometry_forecast_not_archive_admission',measurement_claim=False,budget=budget.as_dict(),metadata_scope='20GiB declared conservative allowance over prior19.3GiB observedRSS; actualguardbaseline must fit beforeforward',runtime_reserve_scope='4GiB explicit allowance, not a measured quantity; actualowner/guardchecks refuse overflow',evaluation_windows=16,probes=4,statistics_windows=windows,reverse_forward_backward_batches=windows*4*16,baseline_windows=87,baseline_reverse_batches=87*4*16,replay_multiplier=windows/87,decoded_tensor_bytes_one_pass=decoded,baseline_decoded_tensor_bytes_four_passes=4*decoded,source_head_bytes=head,source_core_layer_range=[0,44],tail_retains_last_two_source_layers=True,archive_overhead_not_measured=True,exact_archive_sizes_required_before_each_layer_projection=True,layers=results)
out=pathlib.Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/cost-window-plan');out.mkdir(exist_ok=True);(out/'logical-forecast-read-pages-v2.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='layers'}),flush=True)
