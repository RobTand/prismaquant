#!/usr/bin/env python3
"""Explain the merge gate selection without importing or executing test code."""
from __future__ import annotations

import argparse
from collections import Counter
import ast
import importlib.util
import heapq
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.error
import urllib.request

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prismaquant.digests import DIRECT_ASCII_LAX


PATH_SPEC = importlib.util.spec_from_file_location('gate_paths', Path(__file__).with_name('gate_paths.py'))
gate_paths = importlib.util.module_from_spec(PATH_SPEC)
PATH_SPEC.loader.exec_module(gate_paths)

POLICY = 'specific-references-runtime-data-full-v5'
DEDICATED_NODEIDS = frozenset({'tests/test_graph_impact.py::test_graph_impact_gate'})


def load_gate_json(path: Path) -> dict:
    return json.loads(path.read_text())


def changed_files(root: Path, base: str, head: str) -> list[str]:
    """Include deleted and renamed module names as well as new names."""
    if not all(re.fullmatch(r'[0-9a-fA-F]{40,64}', sha) for sha in (base, head)):
        raise ValueError('base and head must be full Git object IDs')
    # Compare the event's trees, including queued-ahead changes in its head.
    # This needs only those two objects, even in a shallow checkout.
    # --no-renames retains both sides of every rename.
    result = subprocess.run(['git', 'diff', '--name-only', '--no-renames', '-z',
                             base, head, '--'], cwd=root,
                            capture_output=True, check=True)
    return sorted(set(result.stdout.decode().rstrip('\0').split('\0')) - {''})


def module_names(path: str) -> set[str]:
    if not path.endswith('.py') or not path.startswith(('prismaquant/', 'tools/', 'tests/')):
        return set()
    module = path.removesuffix('.py').removeprefix('src/').replace('/', '.')
    module = module.removesuffix('.__init__')
    names = {module}
    if path.startswith(('tools/', 'tests/')):
        names.add(module.split('.', 1)[1] if '.' in module else module)
        names.add(module.rsplit('.', 1)[-1])
    return names


def production_loads(tree: ast.AST, path: str) -> tuple[set[str], set[str]]:
    """Recognize literal dynamic imports and resolvable Python path loaders."""
    aliases = {'__import__': '__import__'}
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = name.name if name.asname else name.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for name in node.names:
                aliases[name.asname or name.name] = f'{node.module}.{name.name}'
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node)
    modules, paths = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = gate_paths.dotted(node.func)
        first, _, tail = name.partition('.')
        name = aliases.get(first, '') + ('.' + tail if tail else '')
        if name in {'importlib.import_module', '__import__'} and node.args:
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                module = target.value
                if module.startswith('.') and name == 'importlib.import_module':
                    package = next((item.value for item in node.keywords if item.arg == 'package'),
                                   node.args[1] if len(node.args) > 1 else None)
                    if not isinstance(package, ast.Constant) or not isinstance(package.value, str):
                        continue
                    parts = package.value.split('.')
                    level = len(module) - len(module.lstrip('.'))
                    module = '.'.join(parts[:len(parts) - level + 1] + [module.lstrip('.')]).rstrip('.')
                modules.add(module)
        slots = {'importlib.util.spec_from_file_location': (1, 'location'),
                 'importlib.machinery.SourceFileLoader': (1, 'path'),
                 'importlib.machinery.SourcelessFileLoader': (1, 'path'),
                 'runpy.run_path': (0, 'path_name')}
        if name not in slots:
            continue
        position, keyword = slots[name]
        target = next((item.value for item in node.keywords if item.arg == keyword),
                      node.args[position] if len(node.args) > position else None)
        if target is None:
            continue
        # Keep intermediate loader path literals conservatively: even a basename
        # joined to /tmp may select a same-named repository file. Accuracy wins.
        pending = [child.id for child in ast.walk(target) if isinstance(child, ast.Name)]
        seen, bindings = set(), set()
        while pending:
            variable = pending.pop()
            if variable in seen:
                continue
            seen.add(variable)
            for assignment in assignments.get(variable, ()):
                if assignment.value is not None:
                    bindings.add(assignment)
                    pending.extend(child.id for child in ast.walk(assignment.value) if isinstance(child, ast.Name))
        evidence = ast.Module(body=[*sorted(bindings, key=lambda item: item.lineno), ast.Expr(value=target)],
                              type_ignores=[])
        paths.update(gate_paths.references(evidence, path))
    return modules, paths


def dependencies(source: str, path: str = '', *, production: bool = False) -> tuple[set[str], set[str]]:
    parsed = ast.parse(source)
    trees = [parsed]
    # Subprocess Python snippets are direct evidence too.
    for node in (() if production else ast.walk(parsed)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'import ' in node.value:
            try:
                trees.append(ast.parse(node.value))
            except SyntaxError:
                pass
    modules, paths = production_loads(parsed, path) if production else (set(), set())
    for tree in trees:
        if not production:
            paths.update(gate_paths.references(tree, path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level and path:
                    package = list(PurePosixPath(path.removeprefix('src/')).parent.parts)
                    package = package[:len(package) - node.level + 1]
                    prefix = '.'.join([*package, *([prefix] if prefix else [])])
                if prefix:
                    modules.add(prefix)
                modules.update(f'{prefix}.{alias.name}' if prefix else alias.name for alias in node.names)
            if not production and isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if not value.endswith('.py') and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', value):
                    modules.add(value)
    return modules, paths


def dependency_index(root: Path) -> dict[str, tuple[set[str], set[str]]]:
    return {path.relative_to(root).as_posix(): dependencies(path.read_text(), path.relative_to(root).as_posix(),
                                                         production=directory != 'tests')
            for directory in ('tests', 'prismaquant', 'tools')
            for path in sorted((root / directory).rglob('*.py'))}


def production_import_closure(index: dict, changed: list[str]) -> set[str]:
    """Follow reverse production imports before selecting tests and their helpers."""
    production = {path for path in index if not path.startswith('tests/')}
    seeds = {path for path in changed if path.endswith('.py') and
             path.startswith(('prismaquant/', 'tools/'))}
    aliases: dict[str, set[str]] = {}
    for path in production | seeds:
        for alias in module_names(path):
            aliases.setdefault(alias, set()).add(path)
    reverse: dict[str, set[str]] = {}
    for path in production:
        for target in index[path][1] & (production | seeds):
            reverse.setdefault(target, set()).add(path)
        for dependency in index[path][0]:
            parts = dependency.split('.')
            for end in range(1, len(parts) + 1):
                for target in aliases.get('.'.join(parts[:end]), ()):
                    reverse.setdefault(target, set()).add(path)
    affected = set(seeds)
    pending = list(seeds)
    while pending:
        for importer in reverse.get(pending.pop(), ()):
            if importer not in affected:
                affected.add(importer)
                pending.append(importer)
    return affected


def helper_importers(index: dict, changed: list[str], existing: dict | None = None) -> dict[str, set[str]]:
    index = {path: value for path, value in index.items() if path.startswith('tests/')}
    targets_to_index = {name for name in changed if name.startswith('tests/') and name.endswith('.py')}
    targets_to_index = (set(index) | targets_to_index) if existing is None else targets_to_index - index.keys()
    if not targets_to_index:
        return existing if existing is not None else {}
    aliases: dict[str, set[str]] = {}
    for path in targets_to_index:
        for alias in module_names(path):
            aliases.setdefault(alias, set()).add(path)
    reverse: dict[str, set[str]] = {} if existing is None else dict(existing)
    for path, (imports, references) in index.items():
        targets = {target for module in imports for target in aliases.get(module, ())}
        targets.update(references & targets_to_index)
        patterns = {reference for reference in references if '**' in reference}
        if patterns:
            targets.update(target for target in targets_to_index if gate_paths.gate_path_matches(target, patterns))
        for target in targets - {path}:
            reverse.setdefault(target, set()).add(path)
    return reverse


def estimate(files: list[str], durations: dict, workers: int = 4) -> dict:
    values = durations['files']
    missing = sorted(set(files) - values.keys())
    if missing:
        return {'serial_seconds': None, 'wall_seconds': None, 'workers': workers,
                'missing_files': missing, 'source': durations.get('source')}
    costs = [values[name] for name in files]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError('file durations must be finite nonnegative seconds')
    # Conservative file-level estimate; work stealing may split long files.
    slots = [0.0] * workers
    for cost in sorted(costs, reverse=True):
        heapq.heappush(slots, heapq.heappop(slots) + cost)
    return {'serial_seconds': round(sum(costs), 3), 'wall_seconds': round(max(slots), 3),
            'workers': workers, 'missing_files': [], 'source': durations.get('source')}


def exclusion_entries(quarantine: dict) -> list[dict]:
    entries = [dict(row, kind=kind) for kind in ('entries', 'cost_exclusions', 'environment_lanes')
               for row in quarantine.get(kind, [])]
    for row in entries:
        target = row.get('target', '')
        path = PurePosixPath(target.split('::', 1)[0])
        issue = row.get('issue')
        if (not target or not str(path).startswith('tests/') or path.is_absolute() or
            '..' in path.parts or not str(path).endswith('.py') or
            not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0 or issue == 3180 or
            not isinstance(row.get('reason'), str) or not row['reason'].strip()):
            raise ValueError(f'invalid gate exclusion or missing owner: {target!r}')
        if row['kind'] == 'environment_lanes' and (not isinstance(row.get('lane'), str) or not row['lane'].strip()):
            raise ValueError('environment lane needs a named capability')
        if row['kind'] == 'cost_exclusions' and '::' in target:
            raise ValueError('cost exclusions must name complete files')
    if len({row['target'] for row in entries}) != len(entries):
        raise ValueError('duplicate gate exclusions')
    return entries


QUARANTINE_PATH = 'tools/ci/gate_quarantine.json'


def quarantine_policy(current: dict, baseline: dict | None, changed: list[str]) -> None:
    entries = exclusion_entries(current)
    if baseline is None:
        return  # Initial policy, like the budget bootstrap, is reviewed with the gate.
    previous = {row['target']: row for row in exclusion_entries(baseline)}
    def covered(row):
        target = row['target']
        old = previous.get(target, previous.get(target.split('::', 1)[0]))
        return old is not None and row['issue'] == old['issue']
    growth = [row['target'] for row in entries if not covered(row)]
    isolated = QUARANTINE_PATH in changed and all(
        path == QUARANTINE_PATH or path.startswith('docs/') and path.endswith(('.md', '.rst')) or
        path in {'README.md', 'CHANGELOG.md'} for path in changed)
    if growth and not isolated:
        raise ValueError('Quarantine growth requires a quarantine-only PR with optional docs: ' + ', '.join(growth))


def fetch_quarantine_owner(issue: int) -> dict:
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'prismaquant-gate'}
    # Anonymous REST shares a small per-IP limit across hosted runners; the
    # workflow's read-only job token has its own allowance.
    if token := os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(
        f'https://api.github.com/repos/RobTand/prismaquant/issues/{issue}', headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            raise ValueError(f'Quarantine owner #{issue} must be an open issue') from error
        raise


def check_quarantine_owners(entries: list[dict], *, fetch=fetch_quarantine_owner) -> None:
    for issue in sorted({row['issue'] for row in entries}):
        try:
            owner = fetch(issue)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, GitHub HTTP {error.code}') from error
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, GitHub unavailable: {error}') from error
        if not isinstance(owner, dict) or owner.get('state') not in {'open', 'closed'}:
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, invalid GitHub response')
        if owner['state'] != 'open' or 'pull_request' in owner:
            raise ValueError(f'Quarantine owner #{issue} must be an open issue')


def base_quarantine(root: Path, base: str) -> dict | None:
    subprocess.run(['git', 'cat-file', '-e', f'{base}^{{commit}}'], cwd=root,
                   capture_output=True, check=True)
    tree = subprocess.run(['git', 'ls-tree', base, '--', QUARANTINE_PATH], cwd=root,
                          capture_output=True, text=True, check=True)
    if not tree.stdout.strip():
        return None
    return json.loads(subprocess.check_output(['git', 'show', f'{base}:{QUARANTINE_PATH}'],
                                             cwd=root, text=True))


def create_plan(root: Path, changed: list[str], *, durations: dict, quarantine: dict,
                index: dict | None = None, mode: str = 'selection', helper_graph: dict | None = None) -> dict:
    root = Path(root)
    triggers = sorted(path for path in set(changed) if
        Path(path).name == 'conftest.py' or
        path in {'pytest.ini', 'pyproject.toml', 'setup.cfg', 'setup.py', 'uv.lock',
                 'tools/resolve_tessera_dev_pin.py', 'prismaquant/tessera_runtime_contract.py',
                 'prismaquant/tessera_runtime/tessera_serving_runtime_pin.json'} or
        path.startswith(('.github/', 'tools/ci/gate_', 'tests/test_gate_')) or
        'constraints' in path or 'requirements' in path or
        path == 'run-pipeline.sh' or
        not (root / path).is_file() and not (path.startswith('docs/') and path.endswith(('.md', '.rst'))) and
        path not in {'README.md', 'CHANGELOG.md', 'LICENSE'} or
        path.startswith(('prismaquant/', 'tools/')) and not path.endswith(('.py', '.md', '.rst')))
    if mode == 'full' or triggers:
        names = index if index is not None else (
            path.relative_to(root).as_posix() for path in sorted((root / 'tests').rglob('*.py')))
        index = {name: (set(), set()) for name in names}
    elif index is None:
        try:
            index = dependency_index(root)
        except (OSError, SyntaxError, UnicodeError) as error:
            triggers.append('dependency scan unavailable: ' + type(error).__name__)
            index = {path.relative_to(root).as_posix(): (set(), set())
                     for path in sorted((root / 'tests').rglob('*.py'))}
    if mode not in {'selection', 'full'}:
        raise ValueError('mode must be selection or full')
    all_tests = {path for path in index if path.startswith('tests/') and Path(path).name.startswith('test_')}
    exclusions = exclusion_entries(quarantine)
    by_file = {row['target']: row for row in exclusions if '::' not in row['target']}
    population = all_tests - by_file.keys()
    version_body = {'policy': POLICY, 'files': sorted(population), 'exclusions': exclusions, 'durations': durations}
    version = DIRECT_ASCII_LAX.sha256(version_body)
    reasons: dict[str, list[str]] = {}
    explained = set()
    affected = set(changed) | production_import_closure(index, changed)
    import_targets: dict[str, set[str]] = {}
    for path in affected:
        if not path.startswith('tests/'):
            for module in module_names(path):
                import_targets.setdefault(module, set()).add(path)
    for name, (imports, paths) in index.items():
        if not name.startswith('tests/'):
            continue
        why = []
        if triggers:
            why.append('whole population: ' + ', '.join(triggers))
        if name in changed:
            why.append('changed test file')
        imported = set()
        for dependency in imports:
            parts = dependency.split('.')
            for end in range(1, len(parts) + 1):
                imported.update(import_targets.get('.'.join(parts[:end]), ()))
        referenced = affected & paths
        patterns = {path for path in paths if '**' in path}
        if patterns:
            referenced.update(path for path in affected if gate_paths.gate_path_matches(path, patterns))
        for path in sorted(imported | referenced):
            if path in imported:
                why.append(('direct import: ' if path in changed else 'production import: ') + path)
            else:
                why.append('literal path reference: ' + path)
        if why:
            reasons[name] = why
            explained.update(imported | referenced)
    reverse = helper_importers(index, changed, existing=helper_graph)
    pending = sorted(set(reasons) |
                     {path for path in changed if path.startswith('tests/')})
    seen = set(pending)
    while pending:
        helper = pending.pop()
        for importer in sorted(reverse.get(helper, ())):
            reasons.setdefault(importer, []).append('helper dependency: ' + helper)
            if importer not in seen:
                seen.add(importer)
                pending.append(importer)
    unexplained = [path for path in changed if not triggers and
        not (path.startswith('docs/') and path.endswith(('.md', '.rst'))) and
        path not in {'README.md', 'CHANGELOG.md', 'LICENSE'} and
        path not in reasons and path not in explained]
    if unexplained:
        triggers.extend(unexplained)
        for name in population:
            reasons.setdefault(name, []).append('whole population: uncertain change ' + ', '.join(unexplained))
    shadow = [{'path': name, 'reasons': reasons[name],
               'estimated_seconds': durations['files'].get(name)}
              for name in sorted(population & reasons.keys())]
    selected = shadow if mode == 'selection' else [
        {'path': name, 'reasons': ['full queue population'],
         'estimated_seconds': durations['files'].get(name)} for name in sorted(population)]
    deferred = [{'path': row['target'], 'reason': row['reason'], 'issue': row['issue'],
                  'kind': row['kind'], **({'lane': row['lane']} if row['kind'] == 'environment_lanes' else {}), 'selected_by': reasons.get(row['target'].split('::')[0], [])}
                 for row in exclusions]
    deselect = sorted(row['target'] for row in exclusions if '::' in row['target'] and
                      row['target'].split('::')[0] in {item['path'] for item in selected})
    deselect = sorted(set(deselect) | {node for node in DEDICATED_NODEIDS
                      if node.split('::', 1)[0] in {item['path'] for item in selected}})
    files = [row['path'] for row in selected]
    return {'schema': 'prismaquant.gate_plan.v1', 'policy': POLICY,
            'population': {'size': len(population), 'version': version,
                           'estimate': estimate(sorted(population), durations)},
            'mode': mode, 'changed_files': sorted(set(changed)), 'selected': selected,
            'shadow_selected': shadow, 'deferred': deferred,
            'deselect': deselect, 'estimate': estimate(files, durations),
            'budget': {'target_wall_seconds': 300, 'maximum_wall_seconds': 600,
                       'includes_pickup': False, 'policy': 'never truncate relevant coverage'}}


def summary(plan: dict) -> str:
    """A bounded preview; plan.json is the complete machine-readable record."""
    preview_entries = 50
    def cell(value):
        text = str(value).replace('|', '\\|').replace('\n', ' ')
        return text[:500] + ('...' if len(text) > 500 else '')
    counts = Counter(reason.split(':', 1)[0] for row in plan['selected'] for reason in row['reasons'])
    lines = ['## Merge gate selection', '', f"Population `{cell(plan['population']['version'])}`",
             f"Selected {len(plan['selected'])} of {plan['population']['size']} files.",
             f"Conservative single-job test seconds at 4 workers: {plan['estimate']['wall_seconds']!s}.",
             'Runner pickup and setup are additional. Missing durations mean an unknown estimate.',
             'Complete selection, reasons and exclusions are in the plan.json artifact.', '',
             'Selection reasons: ' + '; '.join(f'{cell(reason)}: {count}' for reason, count in sorted(counts.items())[:50]), '',
             '| Test file | Reasons | Estimated seconds |', '| --- | --- | ---: |']
    lines.extend(f"| {cell(row['path'])} | {cell('; '.join(row['reasons']))} | {cell(row['estimated_seconds'])} |"
                 for row in plan['selected'][:preview_entries])
    if len(plan['selected']) > preview_entries:
        lines += ['', f"{len(plan['selected']) - preview_entries} more selected files in plan.json."]
    lines += ['', '<details><summary>Other lanes and exclusions</summary>', '',
              f"{len(plan['deferred'])} entries; complete record in plan.json.", '',
              '| Test | Reason | Owning issue |', '| --- | --- | --- |']
    lines.extend(f"| {cell(row['path'])} | {cell(row['reason'])} | {('#' + str(row['issue'])) if row['issue'] else 'separate lane'} |"
                 for row in plan['deferred'][:preview_entries])
    lines += ['', '</details>', '']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--mode', choices=['selection', 'full'], default='selection')
    parser.add_argument('--files-output', type=Path, help='newline-separated files for a CI splitter')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        changed = changed_files(root, args.base, args.head)
        quarantine = load_gate_json(root / QUARANTINE_PATH)
        quarantine_policy(quarantine, base_quarantine(root, args.base), changed)
        check_quarantine_owners(exclusion_entries(quarantine))
    except ValueError as error:
        print('GATE_POLICY_FAILURE: ' + str(error), file=sys.stderr)
        return 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('NO_VERDICT: ' + str(error), file=sys.stderr)
        return 75
    plan = create_plan(root, changed, durations=load_gate_json(root / 'tools/ci/gate_durations.json'),
                       quarantine=quarantine, mode=args.mode)
    plan.update(base_sha=args.base, head_sha=args.head)
    output = json.dumps(plan, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    if args.files_output:
        args.files_output.write_text(''.join(row['path'] + '\n' for row in plan['selected']))
    if args.summary:
        with args.summary.open('a') as handle:
            handle.write(summary(plan))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
