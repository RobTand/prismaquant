"""Read-only CPU prerequisite for the exact raw Netdata control collector."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

from experiments.original_cuda_control import collect, netdata, CONTEXTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    before=int(time.time())
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for host in ('sparky','sparklina'):
            selected={key:value for key,value in netdata(host,'charts')['charts'].items()
                      if value.get('context') in CONTEXTS}
            if {meta.get('context') for meta in selected.values()}!=CONTEXTS:
                raise RuntimeError('Netdata context missing: '+host)
            jobs.append(pool.submit(collect,host,selected,before-60,before,a.out))
        rows=[job.result() for job in jobs]
    record=dict(schema='prismaquant.original_cuda_netdata_prerequisite.v1',
                cpu_only=True,actual_cuda_control=False,actual_glm=False,hosts=rows)
    (a.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
