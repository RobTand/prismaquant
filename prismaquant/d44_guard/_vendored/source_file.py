"""Read one tensor from a PB-admitted whole source file using the existing lease SDK.

The legacy G3 range adapter keys by (path, offset), not length. Whole-file source
admission therefore cannot also bind an eight-byte range at offset zero. Keep
one whole-file entry, verify its own bytes, then copy only the requested tensor
through the already-pinned descriptor. This is not a provider qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid


def tensor(path, name):
    import torch
    import stage1 as S
    if not os.environ.get('PRISMABUILD_RESIDENCY_MAP'):
        # Original CD2 source-reader seam when no staged mapping is injected.
        from safetensors import safe_open
        with safe_open(str(path),framework='pt',device='cpu') as f:
            return f.get_tensor(name)
    from g3_residency import StagedReader, host_path, read_fd
    reader = StagedReader()
    mapping = reader.maps.read_map(reader.ctx['map_path'])
    key = reader.maps.residency_map_key(host_path(str(path)),0)
    entry = mapping['entries'].get(key)
    S.require(entry is not None and entry['bytes']==Path(path).stat().st_size,'Actual whole source file absent from admitted read set')
    tier, epoch = mapping['tier_id'], ''
    covers = reader.lease.covers_for_keys(reader.root,reader.ctx['action_key'],[key],tier_id=tier,manifest_sha256=mapping['manifest_sha256'],epoch=epoch)
    S.require(covers['ok'],'PB source covering material: '+str(covers))
    acq = reader.lease.acquire_for(reader.ctx,tier_id=tier,epoch=epoch,covers=covers['covers'],
               expected={key:{'bytes':entry['bytes'],'sha256':entry['sha256']}},
               span={'start_bytes':0,'end_bytes':entry['bytes']},acquire_token=uuid.uuid4().hex,residency_root=reader.root)
    S.require(acq['ok'],'PB source file lease: '+str(acq))
    fd = None
    try:
        fd,_ = reader.lease.open_pinned(reader.queue,acq['pin'],acq['ref_id'],key,residency_root=reader.root)
        h = hashlib.sha256()
        for offset in range(0,entry['bytes'],8<<20):
            h.update(read_fd(fd,offset,min(8<<20,entry['bytes']-offset)))
        S.require(h.hexdigest()==entry['sha256'],'Actual consumed whole source file byte corruption')
        prefix = read_fd(fd,0,8)
        n = int.from_bytes(prefix,'little')
        S.require(0<n<=entry['bytes']-8,'Actual source header length invalid')
        e = json.loads(read_fd(fd,8,n))[name]
        S.require(e['dtype']=='BF16','Actual source tensor is not bfloat16')
        lo,hi = e['data_offsets']
        S.require(0<=lo<=hi<=entry['bytes']-8-n,'Actual source tensor offsets invalid')
        raw = read_fd(fd,8+n+lo,hi-lo)
        return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(e['shape'])
    finally:
        if fd is not None:
            os.close(fd)
        released = reader.lease.release(reader.queue,acq['pin_id'],acq['ref_id'],consumer_action_key=reader.ctx['action_key'],residency_root=reader.root)
        S.require(bool(released),'PB source file lease release failed')
