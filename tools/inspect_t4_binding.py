"""Print one expert's verified-cell and old-journal identity.

Runs only as a script; nothing executes at import.
"""
import pickle,json
import importlib.util
import os
import sys
from pathlib import Path
if __package__:
    from prismaquant.digests import bytes_sha256hex
else:
    # Stdlib-only script use binds the same tree's owner by file path, so the
    # host needs no installed package. digests.py is stdlib-only: register the
    # file module under a name carrying that path's exact bytes, so two
    # checkouts in one process never share a cache entry.
    _digest_owner_path = Path(__file__).resolve().parents[1] / "prismaquant" / "digests.py"
    _digest_owner_name = "_prismaquant_standalone_digest_owner_" + os.fsencode(_digest_owner_path).hex()
    if _digest_owner_name not in sys.modules:
        _digest_owner_spec = importlib.util.spec_from_file_location(_digest_owner_name, _digest_owner_path)
        _digest_owner_module = importlib.util.module_from_spec(_digest_owner_spec)
        sys.modules[_digest_owner_name] = _digest_owner_module
        try:
            _digest_owner_spec.loader.exec_module(_digest_owner_module)
        except BaseException:
            del sys.modules[_digest_owner_name]
            raise
    bytes_sha256hex = sys.modules[_digest_owner_name].bytes_sha256hex


def main():
 r=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare'); c=pickle.loads((r/'production.pkl').read_bytes());q='model.language_model.layers.10.mlp.experts.0.down_proj';k=next(k for k in c.metadata['verified_cells'] if k[0]==q)
 print('metadata_keys',list(c.metadata)); print('verified',k,json.dumps(c.metadata['verified_cells'][k],default=str)); print('scale',c.activation_max_abs[q],c.activation_scales[q]);
 base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace/rows/row-0045/cost.anchors.json.parts/units');p=pickle.loads((base/(bytes_sha256hex(q.encode())+'.pkl')).read_bytes());u=pickle.loads(p['payload']);print('oldidentity',k[1],json.dumps(u['wire_records'][k[1]]['identity']))


if __name__ == '__main__':
    main()
