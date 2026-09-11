"""Read one explicit existing file set into the server's ordinary ZFS cache."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def arc():
    return {v[0]:int(v[2]) for line in Path('/proc/spl/kstat/zfs/arcstats').read_text().splitlines()[2:]
            if len(v:=line.split())==3}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    raw=Path(args.plan).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.plan_sha256:
        raise ValueError('prime plan changed')
    plan=json.loads(raw)
    if plan['schema']!='prismaquant.server_cache_prime.v1' or socket.gethostname()!='dl380g10':
        raise ValueError('priming requires the declared storage server')
    files=plan['files']
    if len({r['local_path'] for r in files})!=len(files) or sum(r['bytes'] for r in files)!=plan['total_bytes']:
        raise ValueError('prime plan is not a unique bounded file set')
    if not 0<plan['total_bytes']<=67*1024**3:
        raise ValueError('prime file set exceeds the admitted cache allowance')
    for row in files:
        path=Path(row['local_path'])
        if not path.is_relative_to('/storage_pool/shared') or path.resolve()!=path:
            raise ValueError('prime path escapes the local shared dataset')
        st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):
            raise ValueError('prime input metadata changed')
    started=time.time()
    output=Path(plan['output'])
    before=dict(unix=started,arc=arc(),diskstats=Path('/proc/diskstats').read_text(),
        meminfo=Path('/proc/meminfo').read_text(),process_io=Path('/proc/self/io').read_text())
    output.with_suffix('.started.json').write_text(json.dumps(before,indent=2)+'\n')
    def read(row):
        path=Path(row['local_path']);t=time.time();count=0;buffer=bytearray(8*1024**2)
        with path.open('rb',buffering=0) as stream:
            st=os.fstat(stream.fileno())
            if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):
                raise ValueError('prime input changed before reading')
            while size:=stream.readinto(buffer):count+=size
            end=os.fstat(stream.fileno())
        if (end.st_size,end.st_mtime_ns)!=(st.st_size,st.st_mtime_ns) or count!=row['bytes']:
            raise ValueError('prime input changed while reading')
        return dict(path=str(path),bytes=count,started_unix=t,finished_unix=time.time())
    # Four independent read streams feed the existing four-disk/NVMe ZFS pool.
    # No bytes are retained by this process beyond its four 8 MiB buffers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(read,files))
    after=dict(unix=time.time(),arc=arc(),diskstats=Path('/proc/diskstats').read_text(),
        meminfo=Path('/proc/meminfo').read_text(),process_io=Path('/proc/self/io').read_text())
    result=dict(schema=plan['schema'],plan_sha256=args.plan_sha256,scope=plan['scope'],
        files=rows,before=before,after=after,bytes_read=sum(r['bytes'] for r in rows),
        seconds=after['unix']-started,cache_retention_guaranteed=False)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('bytes_read','seconds','cache_retention_guaranteed')}))


if __name__=='__main__':main()
