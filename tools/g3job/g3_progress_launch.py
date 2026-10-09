"""Forward completed G3 work to PB's matching linear read phases (PQ #2301).

Run pbrun with setup, layer-00..44, and teachers phases in that order.
Use the g3_readset v2 manifest with stage/share auto and qualified RAM auto.
No timer/scheduler, phase restarts, proof barrier or quality replay lives here.
"""
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys


class ReadProgress:
    def __init__(self, arms, num_layers=45):
        self.arms, self.num_layers = arms, num_layers
        self.units = 0
        self.layers, self.windows = set(), set()
        self.started = False
        self.setup = False

    def observe(self, line):
        if not self.arms and line.startswith("{"):
            payload = json.loads(line)
            if payload.get("schema") == "campaign.g3.staged_consumer_smoke.v1":
                if not payload.get("own_digest_verified") or payload.get("gpu_roundtrip_verified") != 1 or self.units:
                    raise ValueError("incomplete or repeated staged smoke completion")
                self.units = 1
                return {"phase": "smoke", "units_completed": 1,
                        "unit": "verified_staged_gpu_range", "evidence": line.strip()}
        kind = None
        if re.search(r'\[g3\s+[0-9.]+s\] manifest ', line) and not self.setup:
            self.setup = True
            phase, kind = 'setup', 'loaded_manifest'
        elif re.search(r'\[g3\s+[0-9.]+s\] runner ready ', line) and not self.started:
            self.started = True
            phase, kind = 'layer-00', 'completed_setup'
        else:
            match = re.search(r'\[g3\s+[0-9.]+s\] layer ([0-9]+):', line)
            if match:
                layer = int(match.group(1))
                if layer != len(self.layers) or layer >= self.num_layers:
                    raise ValueError('completed layer sequence changed')
                self.layers.add(layer)
                # PB's current phase frees only preceding phases. Announce the
                # next layer after this layer's entire arm population completed.
                phase = f'layer-{layer + 1:02d}' if layer + 1 < self.num_layers else 'teachers'
                kind = 'completed_used_byte_checked_layer'
            else:
                # Multi-arm logs name the arm; single-arm logs name only the window.
                match = re.search(r'\[g3\s+[0-9.]+s\] (?:(\S+) )?window (\S+): KL ', line)
                if match:
                    logged = match.group(1) or (self.arms[0] if len(self.arms) == 1 else None)
                    if logged is None or logged not in self.arms:
                        raise ValueError(f"completed metric names an arm not requested: {logged!r}")
                    window = (logged, match.group(2))
                    if window in self.windows:
                        raise ValueError('completed metric sequence changed')
                    self.windows.add(window)
                    phase, kind = 'teachers', 'scored_window'
        if kind is None:
            return None
        self.units += 1
        return {'phase': phase, 'units_completed': self.units, 'unit': kind, 'evidence': line.strip()}


def select_arms(args):
    """The phased arms this launch scores; the pilot has no arm completion plan."""
    if "--pilot" in args:
        raise SystemExit("--pilot has no phased read plan; G3 progress phases are arm completion events")
    if "--reader-smoke" in args:
        return []
    if "--arms" in args:
        return args[args.index("--arms") + 1].split(",")
    return [args[args.index("--arm") + 1]]


def main():
    args = sys.argv[1:]
    arms = select_arms(args)
    commit = runpy.run_path(os.environ['PRISMABUILD_ACTION_PROGRESS_HELPER'])['commit']
    out = Path(args[args.index('--output-root') + 1])
    out.mkdir(parents=True, exist_ok=True)
    smoke = "--reader-smoke" in args
    progress = ReadProgress(arms, num_layers=0 if smoke else 45)
    with (out / 'semantic-progress.jsonl').open('x') as ledger:
        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('g3_launch.py')), *args],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                event = progress.observe(line)
                if event is None:
                    continue
                ledger.write(json.dumps(event) + '\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                if not commit(event['units_completed'], phase=event['phase'], unit=event['unit']):
                    raise RuntimeError('semantic progress did not reach the admitted PB channel')
        except BaseException:
            child.terminate()
            child.wait()
            raise
        status = child.wait()
    if status == 0 and ((smoke and progress.units != 1) or (not smoke and
            (len(progress.layers) != 45 or len(progress.windows) != 25 * len(arms)))):
        raise ValueError("child exited without its complete requested population")
    return status


if __name__ == '__main__':
    raise SystemExit(main())
