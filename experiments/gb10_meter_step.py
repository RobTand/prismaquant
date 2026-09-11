"""Step response of the GB10 NVML power/utilization meters against known kernel activity.

Runs idle -> busy (large matmuls) -> idle -> busy (tiny kernels) -> idle on the
GPU while sampling NVML (pynvml when present, nvidia-smi otherwise) at ~10 Hz,
and reports how long after kernels start/stop the meters reflect it.  This is
an instrument calibration for the pqteld 0.5 s recorder's ``power_draw_w`` and
``gpu_util`` columns, which the batch-boundary analyses integrate.
"""
import argparse, json, subprocess, threading, time
from pathlib import Path

import torch


def sampler(stop, rows):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def read():
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            try:
                inst = pynvml.nvmlDeviceGetFieldValues(h, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0].value.uiVal / 1000.0
            except Exception:
                inst = None
            return dict(power=power, instant=inst, util=util, source='pynvml')
    except Exception:
        def read():
            out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,power.draw.instant,utilization.gpu',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
            p, i, u = [x.strip() for x in out.split(',')]
            def f(x):
                try: return float(x)
                except ValueError: return None
            return dict(power=f(p), instant=f(i), util=f(u), source='nvidia-smi')
    while not stop.is_set():
        t = time.time()
        try:
            r = read()
        except Exception as e:
            r = dict(error=str(e))
        r['time'] = t
        rows.append(r)
        time.sleep(max(0.0, 0.1 - (time.time() - t)))


def busy_matmul(seconds):
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            a = a @ b
            a = a / a.abs().amax()
        torch.cuda.synchronize()
    torch.cuda.synchronize()


def busy_tiny(seconds):
    x = torch.zeros(1024, device='cuda')
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(500):
            x.add_(1.0)
        torch.cuda.synchronize()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    rows, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(stop, rows), daemon=True); th.start()
    phases = []
    def phase(name, fn, seconds):
        t0 = time.time(); fn(seconds); t1 = time.time(); phases.append(dict(name=name, start=t0, end=t1))
    phase('idle0', lambda s: time.sleep(s), 6)
    phase('matmul', busy_matmul, 12)
    phase('idle1', lambda s: time.sleep(s), 8)
    phase('tiny', busy_tiny, 10)
    phase('idle2', lambda s: time.sleep(s), 8)
    stop.set(); th.join()
    (out/'samples.json').write_text(json.dumps(dict(phases=phases, samples=rows, device=torch.cuda.get_device_name(0)), indent=0))
    # step delays
    def first(cond, t_from):
        for r in rows:
            if r['time'] >= t_from and 'error' not in r and cond(r):
                return r['time'] - t_from
        return None
    report = {}
    for ph in phases:
        if ph['name'] in ('matmul', 'tiny'):
            plateau_p = max(r['power'] for r in rows if ph['start'] + 4 < r['time'] < ph['end'] and r.get('power') is not None)
            idle_p = min(r['power'] for r in rows if r['time'] < phases[0]['end'] and r.get('power') is not None)
            report[ph['name']] = dict(
                plateau_power=plateau_p, idle_power=idle_p,
                rise_util_ge50=first(lambda r: (r.get('util') or 0) >= 50, ph['start']),
                rise_util_ge90=first(lambda r: (r.get('util') or 0) >= 90, ph['start']),
                rise_power_half=first(lambda r: (r.get('power') or 0) >= idle_p + 0.5 * (plateau_p - idle_p), ph['start']),
                rise_power_90=first(lambda r: (r.get('power') or 0) >= idle_p + 0.9 * (plateau_p - idle_p), ph['start']),
                rise_instant_half=first(lambda r: (r.get('instant') or 0) >= idle_p + 0.5 * (plateau_p - idle_p), ph['start']),
                fall_util_le10=first(lambda r: (r.get('util') or 100) <= 10, ph['end']),
                fall_power_half=first(lambda r: (r.get('power') or 1e9) <= idle_p + 0.5 * (plateau_p - idle_p), ph['end']),
                fall_instant_half=first(lambda r: (r.get('instant') or 1e9) <= idle_p + 0.5 * (plateau_p - idle_p), ph['end']),
            )
    report['sampler_source'] = rows[0].get('source'); report['samples'] = len(rows)
    (out/'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
