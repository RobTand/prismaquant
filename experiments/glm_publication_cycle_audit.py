"""Audit bounded publication cycles against actual wires, receipts and telemetry."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def energy(rows, start, end):
    assert rows[0][0] <= start < end <= rows[-1][0]
    joules=covered=0.; gaps=[]
    for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
        lo,hi=max(start,t0),min(end,t1)
        if lo >= hi: continue
        gaps.append(t1-t0)
        a=p0+(p1-p0)*(lo-t0)/(t1-t0)
        b=p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules+=(a+b)*.5*(hi-lo); covered+=hi-lo
    assert abs(covered-(end-start)) < 1e-5 and max(gaps) < 1
    return dict(joules=joules,mean_w=joules/(end-start),max_gap_s=max(gaps))


def union_seconds(intervals):
    end=None; total=0.
    for lo,hi in sorted(intervals):
        assert hi >= lo
        if end is None or lo > end: total+=hi-lo
        elif hi > end: total+=hi-end
        end=max(hi,end) if end is not None else hi
    return total


def profile(meta):
    path=Path(meta['path'])
    assert digest(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
    events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    kernels=sorted([e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X'],key=lambda e:e['ts'])
    assert kernels and all(e['dur'] >= 0 for e in kernels)
    steps=[e for e in kernels if e['name']=='_step_best']
    assert len(steps)==4095
    overlaps=[max(0.,a['ts']+a['dur']-b['ts']) for a,b in zip(steps,steps[1:])]
    assert max(overlaps,default=0)<1 and sum(overlaps)/sum(e['dur'] for e in steps)<.0001
    assert len([e for e in kernels if e['name']=='_traceback'])==1
    assert len([e for e in events if e.get('name')=='cudaGraphLaunch' and e.get('ph')=='X'])==1
    assert not any('StreamBeginCapture' in e.get('name','') for e in events)
    span=max(e['ts']+e['dur'] for e in kernels)-kernels[0]['ts']
    assert sum(e['dur'] for e in kernels)<=span*1.01
    return dict(sha256=meta['sha256'],steps=len(steps),step_us=sum(e['dur'] for e in steps),
        kernel_span_us=span,max_overlap_us=max(overlaps,default=0),shape=meta['targets_shape'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--action',required=True)
    parser.add_argument('--reference',required=True)
    args=parser.parse_args()
    root=Path(args.root); fleet=Path('/mnt/shared/prismabuild-fleet'); key=args.action
    terminal=json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
    assert terminal['detail']['returncode']==0 and terminal['resource_scope_cleanup']['complete']
    receipt=json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
    body={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==receipt['receipt_sha256']
    meta=receipt['result']; payload=fleet/'cas/blobs'/meta['sha256'][:2]/meta['sha256']
    assert digest(payload)==meta['sha256'] and payload.stat().st_size==meta['bytes']
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='complete' and result['full_campaign_complete'] is False
    printed=[json.loads(line) for line in payload.read_text().splitlines() if line.startswith('{"label":')]
    assert printed==result['arms'] and len(printed)==6
    assert result['producer_source_sha256']=='bcad2ef2a7fdec2aab51b30d59f1f5e10b4933637ca4ffc74ee05e20816c1822'
    reference=json.loads(Path(args.reference).read_text())
    old=reference['arms'][0]['signatures']
    old={s['qname']:(s['wire_sha256'],s['dloss']) for s in old}
    signatures=result['arms'][0]['signatures']
    compared=0
    for s in signatures:
        if s['qname'] in old:
            assert (s['wire_sha256'],s['dloss'])==old[s['qname']]
            compared+=1
    assert compared==len(old)>0
    observer=json.loads((root/'observer/result.json').read_text())
    assert observer['status']=='complete' and not observer['errors']
    nd=[json.loads(line) for line in (root/'observer/netdata.jsonl').read_text().splitlines()]
    power={}
    for source in json.loads((root/'root-pqteld-sources.json').read_text()):
        path=root/(source['host']+'-pqteld.csv'); assert digest(path)==source['sha256']
        rows=list(csv.DictReader(path.open())); assert len(rows)==source['rows']
        power[source['host']]=[(float(r['epoch_ms'])/1000,float(r['power_draw_w'])) for r in rows if r['power_draw_w']]
    assert set(power)=={'sparky','sparklina'}
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    reports=[]; profiles=[]
    for index,arm in enumerate(result['arms']):
        enabled=bool(arm['overlap_bytes']); assert enabled==(index in (1,3,4))
        assert arm['signatures']==signatures and arm['exact_parity'] and arm['returncode']==0
        runroot=root/arm['label']; manifest=json.loads((runroot/'cost.anchors.json').read_text())
        assert len(arm['scheduled'])==result['units']==len(signatures)
        for signature in signatures:
            name=signature['qname']; state=_load_unit(unit_path(runroot/'cost.anchors.json.parts',name),
                stage=manifest['stage'],qname=name,identity_sha256=manifest['identity_sha256'])
            assert len(state['anchors'])==len(state['wire_records'])==1
            anchor=state['anchors'][0]; wire=state['wire_records'][anchor['format_name']]
            path=runroot/'cache/wire'/wire['file']
            assert digest(path)==wire['blob_sha256']==signature['wire_sha256']
            assert path.stat().st_size==wire['blob_bytes'] and anchor['dloss']==signature['dloss']
        if index<2:
            assert len(arm['profiles'])==1; profiles.append(profile(arm['profiles'][0]))
        else: assert not arm['profiles'] and arm['plans_built']==0
        stats=arm['publication_stats']
        if enabled:
            assert not stats['failed'] and stats['budget_bytes']==arm['overlap_bytes']
            # File and receipt jobs are per unit. Checkpoint queue jobs hold
            # snapshots of multiple units and invoke write_unit for each.
            assert 0<stats['peak_charged_bytes']<=stats['budget_bytes'] and stats['published']>2*result['units']
        else: assert stats is None
        io={}
        for call in arm['io_calls']:
            k=call['kind']; entry=io.setdefault(k,dict(count=0,inclusive_seconds=0.,max_seconds=0.,threads=set()))
            entry['count']+=1; entry['inclusive_seconds']+=call['seconds']
            entry['max_seconds']=max(entry['max_seconds'],call['seconds']);entry['threads'].add(call['thread'])
            if k in ('save','write_bytes','_checkpoint_wire_record','write_unit'):
                assert call['thread'].startswith('tessera-publication') if enabled else call['thread']=='MainThread'
        for entry in io.values():entry['threads']=sorted(entry['threads'])
        for k in ('save','write_bytes','_checkpoint_wire_record','write_unit','_finish_anchor'):
            assert io[k]['count']==result['units']
        main_io=union_seconds([(c['started_unix'],c['finished_unix']) for c in arm['io_calls']
            if c['thread']=='MainThread' and c['kind'] in ('save','write_bytes','fsync','_checkpoint_wire_record','atomic_write_bytes','dump')])
        j=energy(power[terminal['claimed_host']],arm['cycle_started_unix'],arm['cycle_finished_unix'])
        for host in power:energy(power[host],arm['cycle_started_unix'],arm['cycle_finished_unix'])
        guard=arm['completed_guard']; assert guard['peak_conservative_bytes']<=guard['budget_bytes']-guard['margin_bytes']
        reports.append(dict(label=arm['label'],enabled=enabled,seconds=arm['cycle_seconds'],**j,
            main_io_union_seconds=main_io,io=io,publication=stats,guard=guard,
            before_collection=arm['before_collection'],after_collection=arm['after_collection']))
    assert terminal['detail']['profile']['blob_sha256']==result['arms'][1]['profiles'][0]['sha256']
    totals={}
    for enabled in (False,True):
        arms=[a for a in reports[2:] if a['enabled']==enabled];assert len(arms)==2
        seconds=sum(a['seconds'] for a in arms);joules=sum(a['joules'] for a in arms)
        totals['async' if enabled else 'sync']=dict(seconds=seconds,joules=joules,mean_w=joules/seconds,
            units_per_second=2*result['units']/seconds,units_per_joule=2*result['units']/joules)
    hosts={h:len([v for v in nd if v['host']==h]) for h in power};assert all(hosts.values())
    print(json.dumps(dict(status='PASS',action=key,host=terminal['claimed_host'],units=result['units'],
        full_campaign_complete=False,arms=reports,profiles=profiles,totals=totals,
        throughput_ratio=totals['sync']['seconds']/totals['async']['seconds'],
        work_per_joule_ratio=totals['sync']['joules']/totals['async']['joules'],netdata_samples=hosts,
        prior_reference=dict(path=args.reference,sha256=digest(args.reference),compared_units=compared),
        limitations=['Bounded first-round prefix with full selected-row residency, not a complete pricing row.',
            'Profiles cover the actual first Viterbi call; CPU sampler and timed call intervals cover publication.',
            'Timing begins after capture/source residency and includes final checkpoint and cost output.',
            'Inclusive call categories overlap; main I/O union avoids summing nested calls.']),sort_keys=True))


if __name__=='__main__':main()
