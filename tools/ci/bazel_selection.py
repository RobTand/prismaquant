"""Pure selection decisions for the Remote Bazel cutover gate."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ci.bazel_gate import GateError, partition_population
from tools.ci import bazel_artifacts as artifacts
from tools.ci.bazel_artifacts import child_environment
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex

GATE_INPUTS = {
    '.github/workflows/gate.yml', '.github/workflows/bazel-hosted.yml', '.bazelrc', '.bazelversion', '.bazelignore',
    'MODULE.bazel', 'MODULE.bazel.lock', 'tools/ci/bazel_selection.py',
    'tools/ci/bazel_gate.py', 'tools/ci/remote_gate.py',
    'tools/ci/bazel_artifacts.py', 'tools/ci/check_no_secret_leak.py',
}
BAZEL_DIFF_URL = 'https://github.com/Tinder/bazel-diff/releases/download/v42.0.0/bazel-diff_deploy.jar'
BAZEL_DIFF_SHA256 = '0170b70fe2f24477ab056c11f5c3240d8b45a1dca5ab73ad252c096679c5500c'
DURATIONS_PATH = 'tools/ci/gate_durations.json'


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
            item = (row.get('rule') or row.get('sourceFile') or row.get('generatedFile')
                    or row.get('packageGroup'))
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
                tags = next((attr.get('stringListValue',
                                     [] if attr.get('type') == 'STRING_LIST' else None)
                             for attr in item.get('attribute', []) if attr['name'] == 'tags'), [])
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
                'excluded': sorted(label for label, (_, tags) in rules.items()
                                   if 'manual' in tags and 'gate-lane-hosted' not in tags),
                'hosted': sorted(label for label, (_, tags) in rules.items()
                                 if 'gate-lane-hosted' in tags),
                'hosted_arm64': sorted(label for label, (_, tags) in rules.items()
                                       if 'hosted-arm64' in tags),
                'used_sources': sorted(sources),
                'wheel_repositories': wheel_repositories(labels)}
    except (AttributeError, KeyError, TypeError, ValueError):
        raise SelectionError('query population is incomplete or malformed') from None


def impacted_targets_command(java, jar, root, bazel, base_hashes, head_hashes, output):
    return [str(java), '-jar', str(jar), 'get-impacted-targets',
            '--workspacePath', str(root), '--bazelPath', str(bazel),
            '--targetType', 'Rule',
            '--startingHashes', str(base_hashes), '--finalHashes', str(head_hashes),
            '--output', str(output)]


def compare_graphs(base_graph, head_graph, changed, deleted, directory, *,
                   java, jar, root, bazel='bazel', runner=subprocess.run):
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
        result = runner(impacted_targets_command(java, jar, root, bazel, base_graph['hashes'],
            head_graph['hashes'], output), cwd=directory, env=child_environment(),
            text=True, capture_output=True, check=False)
        if result.returncode:
            raise SelectionError('graph comparison failed')
        labels = output.read_text().split()
        if any(not re.fullmatch(r'@{0,2}[^\s/]*//[^:\s]*:[^:\s]+', label)
               for label in labels):
            raise SelectionError('graph comparison returned invalid labels')
        impacted = {label[1:] if label.startswith('@//') else label for label in labels}
        impacted -= set(population['excluded'])
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError, SelectionError):
        return gate_decision(changed, deleted, set(), used_sources, tests,
                             detection_error='graph comparison unavailable')
    return gate_decision(changed, deleted, impacted, used_sources, tests)


def query_revision(root, directory, architecture, *, bazel='bazel', runner=subprocess.run):
    """Freeze the queried population before optional hash evidence can fail."""
    directory = Path(directory)
    phase = 'output-precondition'
    try:
        directory.mkdir(parents=True, exist_ok=True)
        phase = 'command'
        queried = runner([bazel, 'query', 'deps(//...) union buildfiles(//...)',
                          '--output=streamed_jsonproto', '--keep_going=false',
                          '--config=ci'], cwd=root, env=child_environment(), text=True,
                         capture_output=True, check=False)
        if queried.returncode:
            raise SelectionError('revision graph query failed:\n' + queried.stderr)
        def rows():
            nonlocal phase
            for line in queried.stdout.splitlines():
                phase = 'json'
                row = json.loads(line)
                phase = 'population'
                yield row
        phase = 'population'
        population = query_population(rows(), architecture)
        phase = 'query-record'
        query = directory / 'query.jsonl'
        query.write_text(queried.stdout)
        return {'population': population, 'query': query}
    except (OSError, ValueError, TypeError, AttributeError, SelectionError) as error:
        print('Revision query refused at ' + phase, file=sys.stderr)
        raise SelectionError('revision graph query is incomplete or unavailable') from error


def collect_revision(root, directory, architecture, *, java, jar, bazel='bazel',
                     runner=subprocess.run, queried=None):
    """Capture one revision inside Remote Bazel; callers keep this directory private."""
    directory = Path(directory)
    hashes, edges, wheels = [directory / name for name in
        ('hashes.json', 'edges.json', 'wheel-repositories.txt')]
    try:
        graph = queried or query_revision(root, directory, architecture,
                                          bazel=bazel, runner=runner)
        population = graph['population']
        wheels.write_text('\n'.join('@@' + name for name in population['wheel_repositories']) + '\n')
        generated = runner(generate_hashes_command(java, jar, root, bazel, hashes, edges, wheels),
                           cwd=root, env=child_environment(), text=True,
                           capture_output=True, check=False)
        if generated.returncode:
            raise SelectionError('revision graph hashes failed')
        document = json.loads(hashes.read_text())
        hashed = document.get('hashes', document)
        if (not isinstance(hashed, dict) or not hashed
                or not isinstance(json.loads(edges.read_text()), dict)):
            raise SelectionError('revision graph hashes or edges are incomplete')
        return {**graph, 'hashes': hashes, 'edges': edges}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, SelectionError):
        raise SelectionError('revision graph is incomplete or unavailable') from None


def plan_revisions(root, directory, *, base, head, mode, java=None, jar=None,
                   bazel='bazel', runner=subprocess.run):
    """Plan in the Remote Bazel runner from immutable Git and private graph inputs."""
    if (mode not in ('pr', 'main')
            or any(not re.fullmatch('[0-9a-f]{40}', revision) for revision in (base, head))):
        raise SelectionError('invalid planner mode or source revisions')
    root, directory = Path(root), Path(directory)

    def git(*arguments, binary=False):
        result = runner(['git', *arguments], cwd=root, env=child_environment(),
            text=not binary, capture_output=True, check=False)
        if result.returncode:
            raise SelectionError('Git revision evidence is unavailable')
        return result.stdout

    if git('rev-parse', 'HEAD').strip() != head:
        raise SelectionError('planner checkout has the wrong head')
    # Query failure is fatal. A missing population can never become empty success.
    head_query = query_revision(root, directory / 'head', 'amd64',
                                bazel=bazel, runner=runner)
    population = head_query['population']
    if mode == 'main':
        decision = everything(['post-merge full record'], population['tests'])
    else:
        try:
            try:
                git('cat-file', '-e', base + '^{commit}')
            except SelectionError:
                git('fetch', '--no-tags', 'origin', base)
            changed = set(filter(None, git('diff', '--no-renames', '--name-only', '-z',
                '--diff-filter=ACMRT', base, head, '--').split('\0')))
            deleted = set(filter(None, git('diff', '--no-renames', '--name-only', '-z',
                '--diff-filter=D', base, head, '--').split('\0')))
            if not changed and not deleted:
                raise SelectionError('empty source diff')
            if fallback_files(changed | deleted):
                decision = everything(fallback_files(changed | deleted), population['tests'])
            else:
                base_root = directory / 'base-source'
                base_root.mkdir()
                with tarfile.open(fileobj=io.BytesIO(git('archive', base, binary=True))) as archive:
                    archive.extractall(base_root, filter='data')
                if jar is None:
                    jar = bazel_diff_jar(directory)
                if java is None:
                    info = runner([bazel, 'info', 'java-home', '--config=ci'],
                        cwd=root, env=child_environment(), text=True,
                        capture_output=True, check=False)
                    if info.returncode or not Path(info.stdout.strip()).is_absolute():
                        raise SelectionError('Bazel Java runtime is unavailable')
                    java = Path(info.stdout.strip()) / 'bin/java'
                before = collect_revision(base_root, directory / 'base', 'amd64',
                    java=java, jar=jar, bazel=bazel, runner=runner)
                after = collect_revision(root, directory / 'head', 'amd64',
                    java=java, jar=jar, bazel=bazel, runner=runner, queried=head_query)
                decision = compare_graphs(before, after, changed, deleted, directory,
                                          java=java, jar=jar, root=root, bazel=bazel, runner=runner)
        except (SelectionError, OSError, ValueError, TypeError,
                tarfile.TarError, subprocess.SubprocessError):
            decision = everything(['source or graph evidence unavailable'], population['tests'])
    populations = {'amd64': population}
    if mode == 'main':
        populations['arm64'] = query_population(
            (json.loads(line) for line in head_query['query'].read_text().splitlines()), 'arm64')
    durations = {}
    timing_file = root / DURATIONS_PATH
    if mode == 'pr' and timing_file.is_file():
        files = json.loads(timing_file.read_text())['files']
        durations = {label: files[path] for label in population['tests']
                     if (path := label[2:].replace(':', '/') + '.py') in files}
    return selection_manifest(decision, populations, hosted=population['hosted'],
        hosted_arm64=population['hosted_arm64'], base=base, head=head, mode=mode, durations=durations)


def verdict_of(selected, statuses):
    failed = sorted(label for label in selected if statuses.get(label) != 'PASSED')
    return ('fail' if failed else 'pass' if selected else 'skipped'), failed


def selection_manifest(decision, populations, *, hosted, base, head, mode,
                       hosted_arm64=frozenset(), shard_counts=None, durations=None):
    """Freeze selection and lane routes before any test shard is submitted."""
    if (mode not in ('pr', 'main')
            or any(not re.fullmatch('[0-9a-f]{40}', revision) for revision in (base, head))):
        raise SelectionError('invalid manifest mode or source revisions')
    shard_counts = {'amd64': 6 if mode == 'pr' else 2, 'arm64': 4} if shard_counts is None else shard_counts
    hosted, hosted_arm64 = set(hosted), set(hosted_arm64)
    full = mode == 'main' or decision['selection'] == 'fallback'
    selected = set(decision.get('selected', ()))
    requested = selected | set(decision.get('build', ()))
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': mode,
                'base': base, 'head': head, 'selection': decision['selection'],
                'scopes': sorted(set(decision.get('scopes', ()))),
                'all_scopes': mode == 'main' or decision.get('all_scopes', decision['selection'] == 'fallback'),
                'reason': list(decision.get('reason', ())),
                'hosted': sorted(hosted if full else hosted & requested),
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
        action_requested = requested - hosted
        if not full and not action_requested <= population:
            raise SelectionError('selection contains targets outside the queried population')
        labels = population if full else action_requested
        try:
            shards = partition_population(labels, tests=tests & labels,
                shard_count=shard_counts[architecture], durations=durations) if labels else []
        except (GateError, KeyError) as error:
            raise SelectionError('cannot partition queried population') from error
        document['architectures'][architecture] = {
            'population': sorted(labels), 'tests': sorted(tests & labels), 'shards': shards}
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['plan'])
    parser.add_argument('--mode', choices=['pr', 'main'], required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args(argv)
    directory = os.environ.get('BUILDBUDDY_ARTIFACTS_DIRECTORY')
    if not directory:
        print('plan upload directory unavailable', file=sys.stderr)
        return 75
    upload = Path(directory)
    keys = ['BUILDBUDDY_API_KEY'] + [name for name in (
        'GITHUB_TOKEN', 'MERGIFY_TOKEN') if os.environ.get(name)]
    phase = 'upload-precondition'
    try:
        upload.mkdir(parents=True, exist_ok=True)
        if upload.is_symlink() or list(upload.iterdir()):
            raise SelectionError('plan upload directory is not empty')
        phase = 'scanner-runtime'
        artifacts.require_scan_python()
        phase = 'planning'
        with tempfile.TemporaryDirectory(prefix='pq-private-plan-') as private:
            manifest = plan_revisions(Path.cwd(), Path(private),
                base=args.base, head=args.head, mode=args.mode)
            raw = DIRECT_ASCII_SPACED_LAX.encoded(manifest) + b'\n'
            result = {key: manifest[key] for key in (
                'mode', 'base', 'head', 'selection', 'scopes', 'all_scopes',
                'reason', 'hosted', 'hosted_arm64')}
            result.update({'schema': 'prismaquant.bazel-gate.v1',
                'operation': 'plan', 'verdict': 'pass',
                'manifest_sha256': bytes_sha256hex(raw)})
            phase = 'publication'
            artifacts.publish_json(upload, {
                'shard-manifest.json': raw,
                'gate-result.json': DIRECT_ASCII_SPACED_LAX.encoded(result) + b'\n'}, keys)
        return 0
    except (SelectionError, ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        if upload.is_symlink():
            upload.unlink()
        elif upload.exists():
            shutil.rmtree(upload)
        upload.mkdir(parents=True, exist_ok=True)
        print('plan refused at ' + phase + '; upload refused', file=sys.stderr)
        if phase == 'planning':
            with contextlib.suppress(RuntimeError, OSError):
                artifacts.publish_json(upload, {artifacts.REFUSAL_DETAIL: json.dumps({'phase': phase,
                    'error': ''.join(traceback.format_exception(error))}).encode()}, keys)
                print('Private detail: invocation artifact ' + artifacts.REFUSAL_DETAIL, file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
