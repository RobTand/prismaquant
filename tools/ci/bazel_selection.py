"""Pure selection decisions for the Remote Bazel cutover gate."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ci.bazel_gate import GateError, partition_population
from tools.ci.bazel_artifacts import child_environment
from prismaquant.digests import bytes_sha256hex

GATE_INPUTS = {
    '.github/workflows/gate.yml', '.bazelrc', '.bazelversion', '.bazelignore',
    'MODULE.bazel', 'MODULE.bazel.lock', 'tools/ci/bazel_selection.py',
    'tools/ci/bazel_gate.py', 'tools/ci/remote_gate.py',
}
BAZEL_DIFF_URL = 'https://github.com/Tinder/bazel-diff/releases/download/v42.0.0/bazel-diff_deploy.jar'
BAZEL_DIFF_SHA256 = '0170b70fe2f24477ab056c11f5c3240d8b45a1dca5ab73ad252c096679c5500c'


class SelectionError(RuntimeError):
    """The gate cannot establish a valid selection request."""


def fallback_files(paths):
    return sorted(path for path in paths if path in GATE_INPUTS
                  or path.startswith('tools/bazel/')
                  or PurePosixPath(path).name in {'BUILD', 'BUILD.bazel'}
                  or path.endswith('.bzl'))


def everything(reason, tests):
    return {'selection': 'fallback', 'reason': sorted(reason), 'selected': sorted(tests),
            'build': ['//...'], 'scopes': [], 'all_scopes': True}


def gate_decision(changed, deleted, impacted, used_sources, tests, *, detection_error=None):
    """Missing impact evidence selects the full suite; unread files select none.

    Lock and project metadata use graph evidence, including fine-grained wheel
    hashes. Other workflows and documentation select their actual consumers.
    Unmapped Python files may be missing generated BUILD inputs, so fail closed.
    """
    if not changed and not deleted:
        raise SelectionError('empty diff: base/head changed-file evidence is unavailable')
    if detection_error:
        return everything(['selection error: ' + detection_error], tests)
    forced = fallback_files(changed | deleted)
    if forced:
        return everything(forced, tests)
    unmapped = {path for path in changed - deleted
                if path.endswith('.py') and path not in used_sources
                and not path.startswith('.github/')}
    if unmapped:
        return everything(['unmapped Python: ' + path for path in unmapped], tests)
    relevant = (changed & used_sources) | deleted
    if relevant and not impacted:
        return everything(['missing impact: ' + path for path in relevant], tests)
    internal = {label for label in impacted if label.startswith('//')}
    return {'selection': 'impacted' if impacted else 'irrelevant', 'reason': [],
            'selected': sorted(impacted & tests), 'build': sorted(internal - tests),
            'scopes': sorted({label.split(':', 1)[0] for label in internal}),
            'all_scopes': False}


def generate_hashes_command(java, jar, root, bazel, hashes, edges, wheel_repositories):
    """Explicit wheel repositories prevent treating the entire pip hub as opaque."""
    return [str(java), '-jar', str(jar), 'generate-hashes', '--workspacePath', str(root),
            '--bazelPath', str(bazel), '--bazelCommandOptions=--config=ci', '--excludeExternalTargets', '--includeTargetType',
            '--fineGrainedHashExternalReposFile', str(wheel_repositories),
            '--depEdgesFile', str(edges), str(hashes)]


def bazel_diff_jar(directory, *, fetch=None):
    """Only the pinned v42 bytes may enter the selection runner's Java process."""
    path = Path(directory) / 'bazel-diff-v42.0.0.jar'
    try:
        if path.is_file() and bytes_sha256hex(path.read_bytes()) == BAZEL_DIFF_SHA256:
            return path
        if fetch is None:
            with urllib.request.urlopen(BAZEL_DIFF_URL, timeout=120) as response:
                content = response.read()
        else:
            content = fetch(BAZEL_DIFF_URL)
        if bytes_sha256hex(content) != BAZEL_DIFF_SHA256:
            raise SelectionError('bazel-diff v42 checksum mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path
    except (OSError, ValueError, TypeError):
        raise SelectionError('bazel-diff v42 download is unavailable') from None


def wheel_repositories(labels):
    """Hash individual canonical wheel repos rather than only their shared hub."""
    repositories = set()
    for label in labels:
        match = re.match(r'^@@?(rules_python\+\+pip\+pq_pip_[A-Za-z0-9_.+-]+)//', label)
        if match:
            repositories.add(match[1])
    if not repositories:
        raise SelectionError('query produced no canonical wheel repositories')
    return sorted(repositories)


def query_population(rows, architecture):
    """Read Bazel query records once, preserving build rules and lane routes."""
    if architecture not in ('amd64', 'arm64'):
        raise SelectionError('query has an unsupported architecture')
    rules, sources, labels = {}, set(), []
    try:
        for row in rows:
            item = row.get('rule') or row.get('sourceFile') or row.get('generatedFile')
            label = item['name']
            if not isinstance(label, str):
                raise ValueError('invalid label')
            label = label.removeprefix('@') if label.startswith('@//') else label
            labels.append(label)
            if not label.startswith('//'):
                continue
            if not re.fullmatch(r'//[^:\s]*:[^:\s]+', label):
                raise ValueError('invalid internal label')
            if 'sourceFile' in row:
                path = label[2:].replace(':', '/').lstrip('/')
                if '\\' in path or any(part in ('.', '..') for part in path.split('/')):
                    raise ValueError('invalid source path')
                sources.add(path)
            elif 'rule' in row:
                kind = item['ruleClass']
                tags = next((attr['stringListValue'] for attr in item.get('attribute', [])
                             if attr['name'] == 'tags'), [])
                if (not isinstance(kind, str) or not kind or not isinstance(tags, list)
                        or any(not isinstance(tag, str) for tag in tags) or label in rules):
                    raise ValueError('invalid rule record')
                rules[label] = kind, set(tags)
        targets = {label for label, (_, tags) in rules.items()
                   if 'manual' not in tags and not (architecture == 'arm64'
                       and tags & {'runner', 'hosted-arm64'})}
        tests = {label for label in targets if rules[label][0].endswith('_test')}
        if not targets or not tests:
            raise ValueError('empty build or test population')
        return {'targets': sorted(targets), 'tests': sorted(tests),
                'hosted': sorted(label for label, (_, tags) in rules.items()
                                 if 'gate-lane-hosted' in tags),
                'hosted_arm64': sorted(label for label, (_, tags) in rules.items()
                                       if 'hosted-arm64' in tags),
                'used_sources': sorted(sources),
                'wheel_repositories': wheel_repositories(labels)}
    except (AttributeError, KeyError, TypeError, ValueError):
        raise SelectionError('query population is incomplete or malformed') from None


def impacted_targets_command(java, jar, base_hashes, head_hashes, output):
    return [str(java), '-jar', str(jar), 'get-impacted-targets',
            '--targetType', 'Rule',
            '--startingHashes', str(base_hashes), '--finalHashes', str(head_hashes),
            '--output', str(output)]


def compare_graphs(base_graph, head_graph, changed, deleted, directory, *,
                   java, jar, runner=subprocess.run):
    """Keep a queried head population even when base hashes or impact fail."""
    try:
        population = head_graph['population']
        tests, used_sources = set(population['tests']), set(population['used_sources'])
        if not tests:
            raise ValueError('empty head test population')
    except (KeyError, TypeError, ValueError):
        raise SelectionError('queried head population is unavailable') from None
    if fallback_files(changed | deleted):
        return gate_decision(changed, deleted, set(), used_sources, tests)
    try:
        if base_graph is None:
            raise SelectionError('base graph is unavailable')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'impacted.txt'
        # Stale evidence cannot turn a missing output into an empty selection.
        output.unlink(missing_ok=True)
        result = runner(impacted_targets_command(java, jar, base_graph['hashes'],
            head_graph['hashes'], output), cwd=directory, env=child_environment(),
            text=True, capture_output=True, check=False)
        if result.returncode:
            raise SelectionError('graph comparison failed')
        labels = output.read_text().split()
        if any(not re.fullmatch(r'@{0,2}[^\s/]*//[^:\s]*:[^:\s]+', label)
               for label in labels):
            raise SelectionError('graph comparison returned invalid labels')
        impacted = {label[1:] if label.startswith('@//') else label for label in labels}
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError, SelectionError):
        return gate_decision(changed, deleted, set(), used_sources, tests,
                             detection_error='graph comparison unavailable')
    return gate_decision(changed, deleted, impacted, used_sources, tests)


def collect_revision(root, directory, architecture, *, java, jar, bazel='bazel',
                     runner=subprocess.run):
    """Capture one revision inside Remote Bazel; callers keep this directory private."""
    directory = Path(directory)
    query, hashes, edges, wheels = [directory / name for name in
        ('query.jsonl', 'hashes.json', 'edges.json', 'wheel-repositories.txt')]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        environment = child_environment()
        queried = runner([bazel, 'query', 'deps(//...) union buildfiles(//...)',
                          '--output=streamed_jsonproto', '--keep_going=false',
                          '--config=ci'], cwd=root, env=environment, text=True,
                         capture_output=True, check=False)
        if queried.returncode:
            raise SelectionError('revision graph query failed')
        population = query_population((json.loads(line) for line in queried.stdout.splitlines()),
                                      architecture)
        query.write_text(queried.stdout)
        wheels.write_text('\n'.join(population['wheel_repositories']) + '\n')
        generated = runner(generate_hashes_command(java, jar, root, bazel, hashes, edges, wheels),
                           cwd=root, env=environment, text=True, capture_output=True, check=False)
        if generated.returncode:
            raise SelectionError('revision graph hashes failed')
        document = json.loads(hashes.read_text())
        hashed = document.get('hashes', document)
        if (not isinstance(hashed, dict) or not hashed
                or not isinstance(json.loads(edges.read_text()), dict)):
            raise SelectionError('revision graph hashes or edges are incomplete')
        return {'population': population, 'query': query, 'hashes': hashes, 'edges': edges}
    except (OSError, ValueError, TypeError, AttributeError, SelectionError):
        raise SelectionError('revision graph is incomplete or unavailable') from None


def verdict_of(selected, statuses):
    failed = sorted(label for label in selected if statuses.get(label) != 'PASSED')
    return ('fail' if failed else 'pass' if selected else 'skipped'), failed


def selection_manifest(decision, populations, *, hosted, base, head, mode,
                       hosted_arm64=frozenset(), shard_counts=None):
    """Freeze selection and lane routes before any test shard is submitted."""
    if (mode not in ('pr', 'main')
            or any(not re.fullmatch('[0-9a-f]{40}', revision) for revision in (base, head))):
        raise SelectionError('invalid manifest mode or source revisions')
    shard_counts = {'amd64': 2, 'arm64': 4} if shard_counts is None else shard_counts
    hosted, hosted_arm64 = set(hosted), set(hosted_arm64)
    full = mode == 'main' or decision['selection'] == 'fallback'
    selected = set(decision.get('selected', ()))
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': mode,
                'base': base, 'head': head, 'selection': decision['selection'],
                'scopes': sorted(set(decision.get('scopes', ()))),
                'all_scopes': mode == 'main' or decision.get('all_scopes', decision['selection'] == 'fallback'),
                'reason': list(decision.get('reason', ())),
                'hosted': sorted(hosted if full else hosted & selected),
                'hosted_arm64': sorted(hosted_arm64) if mode == 'main' else [],
                'architectures': {}}
    for architecture in ('amd64', 'arm64') if mode == 'main' else ('amd64',):
        if architecture not in populations:
            raise SelectionError('missing queried architecture population: ' + architecture)
        row = populations[architecture]
        population, tests = set(row['targets']), set(row['tests'])
        if (not population or len(population) != len(row['targets'])
                or not tests <= population or population & hosted
                or (architecture == 'arm64' and population & hosted_arm64)):
            raise SelectionError('invalid queried build population or hosted overlap')
        requested = (selected | set(decision.get('build', ()))) - hosted
        if not full and not requested <= population:
            raise SelectionError('selection contains targets outside the queried population')
        labels = population if full else requested
        try:
            shards = partition_population(labels, tests=tests & labels,
                shard_count=shard_counts[architecture]) if labels else []
        except (GateError, KeyError) as error:
            raise SelectionError('cannot partition queried population') from error
        document['architectures'][architecture] = {
            'population': sorted(labels), 'tests': sorted(tests & labels), 'shards': shards}
    return document
