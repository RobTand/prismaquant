"""Print one expert's verified-cell and old-journal identity.

Runs only as a script; nothing executes at import.
"""
import pickle,json,hashlib
from pathlib import Path


def main():
 r=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare'); c=pickle.loads((r/'production.pkl').read_bytes());q='model.language_model.layers.10.mlp.experts.0.down_proj';k=next(k for k in c.metadata['verified_cells'] if k[0]==q)
 print('metadata_keys',list(c.metadata)); print('verified',k,json.dumps(c.metadata['verified_cells'][k],default=str)); print('scale',c.activation_max_abs[q],c.activation_scales[q]);
 base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace/rows/row-0045/cost.anchors.json.parts/units');p=pickle.loads((base/(hashlib.sha256(q.encode()).hexdigest()+'.pkl')).read_bytes());u=pickle.loads(p['payload']);print('oldidentity',k[1],json.dumps(u['wire_records'][k[1]]['identity']))


if __name__ == '__main__':
    main()
