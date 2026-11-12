"""Scientific barrier fixtures only. Real native admission is proven separately."""
import json
from pathlib import Path
from prismaquant import flux_jobs, tessera_campaign_rows


class FakeNative:
    def __init__(self, monkeypatch):
        self.submitted = []
        self.states = {}
        self.results = {}
        monkeypatch.setattr(tessera_campaign_rows, 'validate_row', lambda row: None)
        monkeypatch.setattr(tessera_campaign_rows, 'submit_rows', self.submit)
        monkeypatch.setattr(tessera_campaign_rows, 'collect_rows', self.collect)
        monkeypatch.setattr(flux_jobs, 'load_submission', lambda record: self.states[str(Path(record).parents[1])])

    def submit(self, rows, state_dir, *, afterok=()):
        state = Path(state_dir)
        key = str(state)
        if key not in self.states:
            state.mkdir(parents=True)
            (state / 'manifest.json').write_text(json.dumps({'rows': rows}))
            job_id = f'fixture-flux-{len(self.submitted) + 1}'
            sub = flux_jobs.FluxSubmission(job_id, state / 'rows/00000', 'a' * 64)
            self.states[key] = sub
            self.submitted.append(job_id)
            self.results[key] = {'job_id': job_id, 'record_dir': str(sub.record_dir),
                                 'status': 'running', 'payload_exit': None, 'query_error': None}
        else:
            assert json.loads((state / 'manifest.json').read_text())['rows'] == rows
        return [self.states[key]]

    def collect(self, state_dir, *, wait=False):
        return [self.results[str(state_dir)]]

    def finish(self, job_id, returncode=0):
        result, = [r for r in self.results.values() if r['job_id'] == job_id]
        result.update(status='succeeded' if returncode == 0 else 'failed', payload_exit=returncode)

    def end(self, round_dir, name, *, returncode=0):
        saved = json.loads((Path(round_dir) / 'submissions' / f'{name}.json').read_text())
        self.finish(self.states[saved['state_dir']].job_id, returncode)
