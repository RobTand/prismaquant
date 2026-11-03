"""Pure selection decisions for the Remote Bazel cutover gate."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ci.bazel_gate import GateError, partition_population

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
            '--bazelPath', str(bazel), '--excludeExternalTargets', '--includeTargetType',
            '--fineGrainedHashExternalReposFile', str(wheel_repositories),
            '--depEdgesFile', str(edges), str(hashes)]


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
            '--startingHashes', str(base_hashes), '--finalHashes', str(head_hashes),
            '--output', str(output)]


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
