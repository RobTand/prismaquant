"""Summarize the G2 identity-proof prefix arms: identity verdict, per-shape timing, power, profile.

Read-only over the arm outputs, the watcher profiles and the stored rows.
v2: reads arms-v2/, profiles-v3/ and the arms-v2 keys and env records, and
reports whether each arm compared exactly EXPECTED_CELLS cells (a comparator
that compares zero cells passes vacuously).
usage: analyze_arms_v2.py OUT_JSON
"""
import collections
import json
import pickle
import statistics
import sys
from pathlib import Path

PROOF = Path('/mnt/shared/tessera-measurements/claude-ldlq-perf-20260927/g2-identity-proof')
UNITS_PER_ARM = 128
EXPECTED_CELLS = 128


def shape_class(qnames):
    kinds = {q.rsplit('.', 1)[-1] for q in qnames}
    return 'down' if kinds <= {'down_proj'} else ('gate_up' if kinds <= {'gate_proj', 'up_proj'} else 'mixed')


def watched_pids():
    found = {}
    for log in PROOF.glob('profiles-v3/*/watch.log'):
        host = log.parent.name
        for line in log.read_text().splitlines():
            if ' NEW pid ' not in line or 'out=' not in line:
                continue
            pid = line.split(' NEW pid ')[1].split()[0]
            out = line.split('out=')[1].strip()
            found.setdefault(Path(out).name, []).append((host, pid))
    return found


def power_window(host, pid, start, end):
    path = PROOF/'profiles-v3'/host/f'power.{pid}.csv'
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(',')
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except (ValueError, IndexError):
            continue
    inside = [r for r in rows if start <= r[0] <= end]
    if len(inside) < 2:
        return None
    energy = sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(inside, inside[1:]))
    return dict(samples=len(inside), mean_w=statistics.fmean(r[1] for r in inside),
                envelope_fraction=statistics.fmean(r[1] for r in inside)/140.0,
                mean_sm_mhz=statistics.fmean(r[2] for r in inside), energy_j=energy,
                units_per_kj=UNITS_PER_ARM/(energy/1000.0) if energy > 0 else None)


def profile_summary(host, pid, hz=20.0):
    path = PROOF/'profiles-v3'/host/f'record.{pid}.raw'
    if not path.is_file():
        return None
    main_leaf, main_bucket, threads = collections.Counter(), collections.Counter(), collections.Counter()
    for line in path.read_text().splitlines():
        stack, _, n = line.rpartition(' ')
        try:
            n = int(n)
        except ValueError:
            continue
        frames = stack.split(';')
        thread = frames[0]
        names = [f.split(' (')[0] for f in frames[1:]]
        tname = thread.split(': ', 1)[1] if ': ' in thread else thread
        threads[tname] += n
        if 'MainThread' not in tname:
            continue
        s = set(names)
        if '_measure_anchor_batch' in s or '_measure_anchor' in s:
            bucket = 'encode (_measure_anchor_batch)'
        elif 'admit' in s:
            bucket = 'stream admit (RowStream.admit)'
        elif '_prefetch_selected_capture' in s:
            bucket = 'prefetch (_prefetch_selected_capture)'
        elif 'compare_rows' in s:
            bucket = 'compare_rows'
        elif '_find_and_load' in s:
            bucket = 'imports'
        else:
            bucket = 'other'
        main_bucket[bucket] += n
        if names:
            main_leaf[names[-1]] += n
    total = sum(main_bucket.values())
    return dict(main_thread_seconds=total/hz,
                main_buckets={k: round(v/hz, 1) for k, v in main_bucket.most_common()},
                main_top_leaves=[(k, round(v/hz, 1)) for k, v in main_leaf.most_common(8)],
                threads={k: round(v/hz, 1) for k, v in threads.most_common(6)})


def stored_anchor_seconds(stored_row, keys):
    """Mean per-anchor 'seconds' in the stored (unprofiled) row, per (class, rung)."""
    manifest = json.loads((Path(stored_row)/'cost.anchors.json.stream'/'manifest.json').read_text())
    wanted = {q for q, _ in keys}
    got = collections.defaultdict(list)
    for entry in manifest['units']:
        if entry['qname'] not in wanted:
            continue
        unit = pickle.loads(pickle.loads((Path(stored_row)/'cost.anchors.json.stream'/entry['file']).read_bytes())['payload'])
        for anchor in unit['anchors']:
            if (anchor['qname'], anchor['format_name']) in keys:
                got[(shape_class([anchor['qname']]), anchor['body_rate_q256'])].append(
                    (anchor['seconds'], anchor.get('encoding_batch_size')))
    return {f'{c}@{r}': dict(anchors=len(v), mean_anchor_s=statistics.fmean(s for s, _ in v),
                              batch=sorted({b for _, b in v}),
                              batch_equivalent_s=statistics.fmean(s for s, _ in v)*(v[0][1] or 1))
            for (c, r), v in sorted(got.items())}


def main():
    keys = json.loads((PROOF/'harness'/'arms-v2.keys.json').read_text())
    envs = json.loads((PROOF/'harness'/'arms-v2.env-records.json').read_text())
    pids = watched_pids()
    report = {}
    for name, key in keys.items():
        out = PROOF/'arms-v2'/name
        item = dict(action_key=key, out=str(out), rate_streams=envs[name]['rate_streams'],
                    tessera=envs[name]['tessera_commit'][:8])
        path = out/'result.json'
        if not path.is_file():
            item['status'] = 'no result'
            report[name] = item
            continue
        r = json.loads(path.read_text())
        env = r.get('environment', {})
        item.update(status=r.get('status'), ok=r.get('ok'), error=r.get('error'),
                    host=env.get('hostname'), device=env.get('device'), torch=env.get('torch'),
                    running_pins=dict(prismaquant=env.get('prismaquant_source_sha256'),
                                      encoder=env.get('encoder_source_sha256')),
                    encoder_fixture_id=env.get('encoder_fixture_id'))
        comparison = r.get('comparison') or {}
        cells = len(comparison.get('cells') or [])
        item['cells_equal_expected'] = cells == EXPECTED_CELLS
        item['comparison'] = dict(ok=comparison.get('ok'), cells=cells, journal=comparison.get('journal'),
                                  byte_identical=sum(1 for c in comparison.get('cells') or [] if c.get('byte_identical')),
                                  strata=comparison.get('strata'),
                                  failures=(comparison.get('failures') or [])[:6])
        item['fused'] = (r.get('fused_engagement') or {}).get('summary')
        item['row_head'] = (r.get('prefix') or {}).get('row_head')
        calls = (r.get('prefix') or {}).get('prefix_calls') or []
        if calls:
            per = collections.defaultdict(list)
            for call in calls:
                rung = int(call['format_name'].rsplit('_R', 1)[-1])
                per[(shape_class(call['qnames']), rung)].append(call['finished_unix']-call['started_unix'])
            item['batches'] = {f'{c}@{q}': dict(n=len(v), mean_s=statistics.fmean(v), min_s=min(v), max_s=max(v))
                               for (c, q), v in sorted(per.items())}
            first, last = calls[0]['started_unix'], calls[-1]['finished_unix']
            item['startup_s'] = first - r['started_unix']
            item['encode_span_s'] = last - first
            item['encode_sum_s'] = sum(c['finished_unix']-c['started_unix'] for c in calls)
            watched = [(h, p) for h, p in pids.get(name, []) if (PROOF/'profiles-v3'/h/f'record.{p}.raw').exists()
                       or (PROOF/'profiles-v3'/h/f'power.{p}.csv').exists()]
            arm_pid = [(h, p) for h, p in watched
                       if 'reseal_identity_proof.py' in (PROOF/'profiles-v3'/h/f'cmdline.{p}.txt').read_text()
                       and 'docker' not in (PROOF/'profiles-v3'/h/f'cmdline.{p}.txt').read_text().split()[0]]
            if arm_pid:
                host, pid = arm_pid[0]
                item['watched'] = dict(host=host, pid=pid)
                item['power_encode_window'] = power_window(host, pid, first, last)
                item['profile'] = profile_summary(host, pid)
            item['stored_row'] = stored_anchor_seconds(
                r['stored_row'], {(q, c['format_name']) for c in calls for q in c['qnames']})
        report[name] = item
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + '\n')
    for name, item in report.items():
        line = (f"{name:32s} {item.get('status')!s:9s} ok={item.get('ok')!s:5s} "
                f"cells={(item.get('comparison') or {}).get('cells')} head={item.get('row_head')}")
        if 'batches' in item:
            line += ' ' + ' '.join(f"{k}={v['mean_s']:.2f}s" for k, v in item['batches'].items())
            pw = item.get('power_encode_window') or {}
            if pw:
                line += f" P={pw['mean_w']:.1f}W u/kJ={pw['units_per_kj']:.3f}"
        print(line)


if __name__ == '__main__':
    main()
