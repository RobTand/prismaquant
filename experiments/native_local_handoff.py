"""Bounded local scientific inputs owned by one admitted native measurement.

This serializes already-leased inputs; it never reads canonical payload paths
or distributes work. The host launcher owns the private local SSD directory.
"""
import hashlib,io,json,os
from pathlib import Path

class _BoundedWriter:
    def __init__(self,handle,limit):self.handle,self.limit=handle,limit
    def write(self,data):
        if self.handle.tell()+len(data)>self.limit:raise RuntimeError('local handoff serializer exceeded its reservation')
        return self.handle.write(data)
    def tell(self):return self.handle.tell()
    def seek(self,*args):return self.handle.seek(*args)
    def flush(self):return self.handle.flush()

class LocalHandoff:
    def __init__(self,root,*,max_bytes):
        self.root=Path(root);self.max_bytes=max_bytes;self.reserved=0;self.files={}
        self.root.mkdir(mode=0o700,parents=False,exist_ok=False)
        self.fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
    def write(self,name,*,tensor=None,raw=None):
        if (not name or Path(name).name!=name or name in self.files or (tensor is None)==(raw is None)):
            raise ValueError('invalid local handoff entry')
        ceiling=len(raw) if raw is not None else tensor.numel()*tensor.element_size()+65536
        if ceiling<=0 or self.reserved+ceiling>self.max_bytes:raise RuntimeError('local handoff disk ceiling exceeded')
        fd=os.open(name,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        try:
            os.posix_fallocate(fd,0,ceiling);self.reserved+=ceiling
            with os.fdopen(fd,'w+b',closefd=False) as handle:
                writer=_BoundedWriter(handle,ceiling)
                if raw is not None:writer.write(raw)
                else:
                    import torch
                    torch.save(tensor.detach().cpu().contiguous(),writer)
                size=handle.tell();handle.flush();os.ftruncate(fd,size);os.fsync(fd)
                handle.seek(0);digest=hashlib.file_digest(handle,'sha256').hexdigest()
            os.fchmod(fd,0o400)
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            result={'path':str(self.root/name),'sha256':digest,'bytes':size}
            self.files[name]=result
            return result
        finally:os.close(fd)
    def seal(self,action_key):
        value={'schema':'prismaquant.native_local_handoff.v1','action_key':action_key,
               'scope':'private local SSD transport within the same admitted action; not reusable checkpoint',
               'max_bytes':self.max_bytes,'reserved_bytes':self.reserved,'files':self.files}
        data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
        result=self.write('bundle.json',raw=data);os.fsync(self.fd);self.close()
        return result


def strict_read(binding):
    """Reuse the existing pinned whole-file byte reader and its digest gate."""
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    path=Path(binding['path']);size=binding.get('bytes',path.stat().st_size)
    if size>512<<20:raise ValueError('native metadata/file exceeds bounded read')
    return _read_verified_wire_blob({'wire':str(path),'record':{
        'blob_bytes':size,'blob_sha256':binding['sha256']}})[0]


def bind_strict_inputs():
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.staged_lease import resolve_sealed_readset,load_sealed_readset
    from prismaquant.residency_map import bind_residency_manifest
    activate_staged_tier_policy('ram,ssd')
    _cas,digest,_size=resolve_sealed_readset();load_sealed_readset(digest);bind_residency_manifest(digest)


class LocalSourceMapping:
    """Read at most one bounded, independently hashed source tensor per borrow."""
    def __init__(self,bindings):self.bindings=bindings
    def __iter__(self):return iter(self.bindings)
    def __getitem__(self,name):
        import torch
        record=self.bindings[name]
        if record['bytes']>64<<20:raise ValueError('source member exceeds local read ceiling')
        raw=Path(record['path']).read_bytes()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('owned local source tensor changed')
        tensor=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if not isinstance(tensor,torch.Tensor):raise TypeError('source handoff is not a tensor')
        return tensor
