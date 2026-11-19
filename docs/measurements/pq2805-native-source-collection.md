# #2805: Native-source fixture collection

This record does not change the shared fixture. `_source` keeps both GLM imports
inside its body. `producer_source` still calls `initialize_model_runtime()`
before it imports and constructs `AutoModelForCausalLM`. The fixture API does
not change.

## Paired results

These runs preceded the new `test_expert_collection_does_not_enter_glm_import_path` test.
That test checks the import boundary in a separate pytest process.
It does not collect or run any GLM test nodes.

All runs used Python 3.12.3 on AArch64.
The final run used the same interpreter on both sides.
The fixture source hashes (SHA-256) were:

- Eager control: `5814d7cffce71556000ea379b66cd6d859f021cf8ae598e28a1bbc9f7848c39c`
- Lazy checkout: `c1a037e52c21703b3fd32eeca8a3566b5aa028e45e4a97bad198edfdeea081f7`

| Sample | Collection profile total, eager / lazy (s) | Fixture cumulative import, eager / lazy (µs) | Executed lines, eager / lazy | Passed nodes, eager / lazy |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 8.1677 / 8.0593 | 207284 / 162124 | 13077 / 12973 | 2 / 2 |
| GLM | 8.1773 / 8.0919 | 133043 / 98002 | 15054 / 15054 | 7 / 7 |

In both eager profiles, collection imported the parity and checkpoint writer
modules and entered the `glm5_next` skip gate. Neither lazy profile recorded
these events. The lazy Qwen outcome run also recorded none of these events.
The lazy GLM outcome run recorded them during execution.

| Sample and source | Import phase | Parity cumulative (µs) | Checkpoint writer cumulative (µs) |
| --- | --- | ---: | ---: |
| Qwen eager | Collection | 19225 | 26504 |
| Qwen lazy | No import in collection | — | — |
| GLM eager | Collection | 7447 | 26571 |
| GLM lazy | Execution | 8020 | 26606 |

Each node below passed on both sides. Setup, call and teardown outcomes
matched. GLM executed the same lines after the line number conversion. Qwen
had no lazy-only lines. Its 104 eager-only lines were:

| File | Eager-only lines |
| --- | ---: |
| `prismaquant/__init__.py` | 1 |
| `prismaquant/tessera_calibration_forward.py` | 14 |
| `prismaquant/tessera_campaign_selection.py` | 11 |
| `tests/legacy_capture_fixtures.py` | 5 |
| `tests/native_source_contract_fixtures.py` | 2 (112 and 113) |
| `tests/test_glm5_next_streamed_forward_parity.py` | 40 |
| `tests/test_glm_campaign_streaming.py` | 31 |

## Interpretation

The lazy fixture avoids the GLM import chain in the Qwen sample. The final
Qwen profile difference was 0.1084 s in approximately 8 s of collection. The
initial Qwen profile was 8.9887 s eager and 8.0970 s lazy. The difference
varied between runs. The profiles do not establish a stable timing gain.

For GLM, the lazy fixture moves comparable import work from collection to
execution. The fixture's cumulative import time includes nested imports.
Do not add it to the builder-module times. These measurements use fresh
processes. They do not show a saving per test or a whole-suite wall-time change.

W2's ranking and its owner remain unverified. `docs/own_process_measurement.md`
names #2804 for full-suite coverage but gives no W2 ranking for this path.
The focused comparison does not establish a rank.

## Method

No identifiable local pre-move source was available. The eager control moves
the two imports to module scope in memory. It is not a historical revision.
Both sides use the same source loader. The script records both source hashes.
It maps the control's executed line numbers back to the checkout, including
the two moved imports. It compares sets of executed repository lines across
collection and test execution. It does not compare coverage percentages.

The expert audit is the non-GLM sample. The stock audit exercises real GLM CPU
streamed delivery. The seed audit exercises native FP8 and BF16 with both
fresh and old journals. The two samples run in separate processes. Each pair
uses the same selectors, interpreter and settings. Pytest uses one process
without xdist workers. OpenMP and MKL each use one thread. CUDA is disabled.

Run from the checkout root. The command extracts the Python block below.
Find these outputs under `$TL3B_OUT/pq2805-confirm/`:

- `commands.jsonl` and `probe.py`: child invocations and the text for `-c`.
- `{qwen,glm}-{eager,lazy}-profile.{prof,txt,json}`: in-process pytest
  collection profiles and import events.
- `{qwen,glm}-{eager,lazy}-imports.stderr`: `-X importtime` traces.
  The `END PYTEST COLLECTION` marker separates collection from execution.
- `{qwen,glm}-{eager,lazy}-coverage.{log,json}`: ordinary pytest outcomes
  and executed line sets. The import observer does not block any dependency.
- `{qwen,glm}-line-diff.json`: executed lines unique to either side.

The full node IDs matched across the profile, import and outcome runs:

```text
tests/test_native_expert_source_binding_audit.py::test_actual_expert_main_binds_only_fresh_rows[native-forked]
tests/test_native_expert_source_binding_audit.py::test_actual_expert_main_binds_only_fresh_rows[bf16-forked]
tests/test_native_glm_stock_source_binding_audit.py::test_stock_preserves_independent_empirical_population[absent]
tests/test_native_glm_stock_source_binding_audit.py::test_stock_preserves_independent_empirical_population[different-draw]
tests/test_native_glm_stock_source_binding_audit.py::test_stock_refuses_foreign_per_cell_source_before_merge
tests/test_native_source_binding_seed_audit.py::test_real_harvest_binds_native_before_numerical_measurement[fresh-native-binding]
tests/test_native_source_binding_seed_audit.py::test_real_harvest_binds_native_before_numerical_measurement[fresh-bf16-unchanged]
tests/test_native_source_binding_seed_audit.py::test_real_harvest_binds_native_before_numerical_measurement[old-journal-native-binding]
tests/test_native_source_binding_seed_audit.py::test_real_harvest_binds_native_before_numerical_measurement[old-journal-bf16-unchanged]
```

```sh
python -c 'from pathlib import Path; p = Path("docs/measurements/pq2805-native-source-collection.md"); exec(p.read_text().split("```python\n")[1].split("\n```")[0])'
```

```python
import os
import subprocess
import sys
from pathlib import Path

child = r'''
import builtins
import cProfile
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

import pytest

side, mode, dest, *nodes = sys.argv[1:]
root = Path.cwd()
base = Path(dest)
fixture = 'tests/native_source_contract_fixtures.py'
lines = (root / fixture).read_text().splitlines(keepends=True)
insert = next(i for i, text in enumerate(lines, 1) if text.startswith('CALIB_HASH ='))
move = [i for i, text in enumerate(lines, 1) if text.startswith(
    ('    from test_glm5_next_streamed_forward_parity ', '    from tests.test_glm_campaign_streaming '))]
assert len(move) == 2 and lines[move[0] - 2].startswith('def _source(')
if side == 'eager':
    imports = [lines[i - 1].lstrip() for i in move]
    for i in move:
        lines[i - 1] = '\n'
    lines[insert - 1:insert - 1] = imports
source = ''.join(lines)
watched = {'test_glm5_next_streamed_forward_parity', 'test_glm_campaign_streaming',
           'transformers.models.glm5_next'}
events = []
phase = 'collection'

class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(source, self.path, 'exec')

class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.removeprefix('tests.') in watched:
            events.append((phase, 'lookup', fullname))
        if fullname == 'tests.native_source_contract_fixtures':
            path = str(root / fixture)
            return importlib.util.spec_from_file_location(fullname, path, loader=Loader(fullname, path))

sys.meta_path.insert(0, Finder())
original_import = builtins.__import__
original = pytest.importorskip

def observe_import(name, *args, **kwargs):
    if name.removeprefix('tests.') in watched:
        events.append((phase, 'import', name))
    return original_import(name, *args, **kwargs)

builtins.__import__ = observe_import

def observe(name, *args, **kwargs):
    if name == 'transformers.models.glm5_next':
        events.append((phase, 'skip gate', name))
    return original(name, *args, **kwargs)

pytest.importorskip = observe
executed = {}

@cache
def local_file(filename):
    path = Path(filename)
    if path.is_absolute() and path.is_relative_to(root):
        return str(path.relative_to(root))


def cover(code, lineno):
    path = local_file(code.co_filename)
    if path is not None:
        if side == 'eager' and path == fixture and lineno >= insert:
            lineno = move[lineno - insert] if lineno < insert + len(move) else lineno - len(move)
        executed.setdefault(path, set()).add(lineno)


class Observe:
    reports = {}
    selected = []

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection(self):
        global phase
        if mode == 'profile':
            with cProfile.Profile() as profile:
                result = yield
            profile.dump_stats(str(base) + '.prof')
        else:
            result = yield
        phase = 'execution'
        if mode == 'imports':
            print('END PYTEST COLLECTION', file=sys.stderr, flush=True)
        return result

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, {})[report.when] = report.outcome


probe = Observe()
if mode == 'coverage':
    sys.monitoring.use_tool_id(sys.monitoring.COVERAGE_ID, 'pq2805')
    sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.LINE, cover)
    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, sys.monitoring.events.LINE)
args = ['-vv', '-s', '-o', 'addopts=']
if mode == 'profile' or (mode == 'imports' and 'expert' in nodes[0]):
    args += ['--collect-only']
status = pytest.main(args + nodes, plugins=[probe])
if mode == 'coverage':
    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, 0)
base.with_suffix('.json').write_text(json.dumps(dict(
    status=int(status), nodes=probe.selected, outcomes=probe.reports, events=events,
    sha256=hashlib.sha256(source.encode()).hexdigest(),
    lines={path: sorted(values) for path, values in sorted(executed.items())}), indent=2))
'''

import json
import pstats

out = Path(os.environ['TL3B_OUT']) / 'pq2805-confirm'
out.mkdir(exist_ok=True)
runner = out / 'probe.py'
runner.write_text(child)
samples = {
    'qwen': ['tests/test_native_expert_source_binding_audit.py::test_actual_expert_main_binds_only_fresh_rows'],
    'glm': ['tests/test_native_glm_stock_source_binding_audit.py',
            'tests/test_native_source_binding_seed_audit.py::test_real_harvest_binds_native_before_numerical_measurement'],
}
env = dict(PYTEST_ADDOPTS='', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
commands = []
print(sys.executable, sys.version.splitlines()[0], flush=True)

def event_labels(data):
    return list(dict.fromkeys((phase, kind, name.rsplit('.', 1)[-1]) for phase, kind, name in data['events']))

for sample, nodes in samples.items():
    coverage = {}
    selected = []
    for side in ('eager', 'lazy'):
        for mode in ('profile', 'imports', 'coverage'):
            base = out / f'{sample}-{side}-{mode}'
            flags = ['-X', 'importtime'] if mode == 'imports' else []
            args = [sys.executable, *flags, '-c', child, side, mode, str(base), *nodes]
            commands.append(dict(env=env, args=args[:args.index('-c')] + ['-c', '<probe.py>'] + args[-(3 + len(nodes)):]))
            (out / 'commands.jsonl').write_text(''.join(json.dumps(cmd) + '\n' for cmd in commands))
            with base.with_suffix('.log').open('w') as stdout, base.with_suffix('.stderr').open('w') as stderr:
                subprocess.run(args, env=os.environ | env, stdout=stdout, stderr=stderr, check=True)
            data = json.loads(base.with_suffix('.json').read_text())
            selected.append(data['nodes'])
            print(base.name, data['status'], len(data['nodes']), flush=True)
            if data['status']:
                print(base.with_suffix('.log').read_text()[-1000:], flush=True)
                raise SystemExit(data['status'])
            if mode == 'profile':
                with base.with_suffix('.txt').open('w') as report:
                    stats = pstats.Stats(str(base) + '.prof', stream=report).sort_stats('cumulative')
                    stats.print_stats(40)
                print(round(stats.total_tt, 4), event_labels(data), flush=True)
                if sample == 'qwen':
                    print(data['sha256'], flush=True)
            if mode == 'imports':
                for text in base.with_suffix('.stderr').read_text().splitlines():
                    if text == 'END PYTEST COLLECTION' or text.rsplit('|', 1)[-1].strip() in (
                        'tests.native_source_contract_fixtures', 'test_glm5_next_streamed_forward_parity',
                        'tests.test_glm_campaign_streaming', 'transformers.models.glm5_next'):
                        print(text, flush=True)
            if mode == 'coverage':
                coverage[side] = data
                print([phases.get('call', phases) for phases in data['outcomes'].values()], flush=True)
    before, after = (coverage[side]['lines'] for side in ('eager', 'lazy'))
    diff = {}
    for label, a, b in [('eager_only', before, after), ('lazy_only', after, before)]:
        diff[label] = {path: sorted(set(a[path]) - set(b.get(path, []))) for path in a if set(a[path]) - set(b.get(path, []))}
    (out / f'{sample}-line-diff.json').write_text(json.dumps(diff, indent=2))
    print('lines', *(sum(map(len, cover.values())) for cover in (before, after)),
          'only', *(sum(map(len, cover.values())) for cover in diff.values()), flush=True)
    print('fixture lines only', [cover.get('tests/native_source_contract_fixtures.py', []) for cover in diff.values()], flush=True)
    print('lazy events', event_labels(coverage['lazy']), flush=True)
    print('same nodes in all runs', all(nodes == selected[0] for nodes in selected), flush=True)
    print('same phase outcomes', coverage['eager']['outcomes'] == coverage['lazy']['outcomes'], flush=True)
    print('eager-only lines by file', {path: len(values) for path, values in diff['eager_only'].items()}, flush=True)
```
