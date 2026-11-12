"""Collect an exact gate assignment with one traversal of each directory."""
from __future__ import annotations

from fnmatch import fnmatch
import json
import os
from pathlib import Path, PurePosixPath

import pytest

from gate_partition import node_owner


class GateCollection:
    def __init__(self, root, files):
        self.files = set(files)
        self.partition = json.loads(os.environ.get('GATE_NODE_PARTITION', 'null'))
        self.split_collected = {path: set() for path in self.partition['owners']} if self.partition else {}
        self.own_cgroup_files = set()
        self.seen, self.collected, self.selected = set(), set(), set()
        self.collection_failed = False
        self.ranks = {root / path: rank for rank, path in enumerate(files)}
        self.directories = {parent for path in self.ranks for parent in path.parents
                            if parent == root or root in parent.parents}

    def pytest_ignore_collect(self, collection_path):
        if collection_path not in self.ranks and collection_path not in self.directories:
            return True
        return None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        items.sort(key=lambda item: self.ranks[item.path])
        self.collected.update(item.nodeid for item in items)
        for item in items:
            item.user_properties.append(('gate_nodeid', item.nodeid))
            path = item.nodeid.split('::', 1)[0]
            if path in self.split_collected:
                self.split_collected[path].add(item.nodeid)
        if self.partition:
            removed = [item for item in items if item.nodeid.split('::', 1)[0] in self.split_collected
                       and node_owner(item.nodeid, self.partition) != self.partition['index']]
            removed_ids = {id(item) for item in removed}
            items[:] = [item for item in items if id(item) not in removed_ids]
            if removed:
                removed[0].config.hook.pytest_deselected(items=removed)
        yield
        self.selected.update(item.nodeid for item in items)

    def pytest_deselected(self, items):
        self.own_cgroup_files.update(item.nodeid.split('::', 1)[0] for item in items
                                     if item.get_closest_marker('own_cgroup') is not None and
                                     (item.nodeid.split('::', 1)[0] not in self.split_collected or
                                      node_owner(item.nodeid, self.partition) == self.partition['index']))

    def pytest_collectreport(self, report):
        path = report.nodeid.split('::', 1)[0]
        if path in self.files:
            self.seen.add(path)
        if report.failed:
            self.collection_failed = True

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.collected.update(ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        coverage = node.workeroutput.get('gate_coverage', {})
        self.seen.update(coverage.get('seen', []))
        self.own_cgroup_files.update(coverage.get('own_cgroup_files', []))
        self.collected.update(coverage.get('collected', []))
        self.selected.update(coverage.get('selected', []))
        for path, nodes in coverage.get('split_collected', {}).items():
            self.split_collected[path].update(nodes)
        self.collection_failed |= bool(error) or coverage.get('collection_failed', True)

    def pytest_sessionfinish(self, session, exitstatus):
        coverage = {'seen': sorted(self.seen), 'collected': sorted(self.collected),
                    'selected': sorted(self.selected),
                    'split_collected': {path: sorted(nodes) for path, nodes in self.split_collected.items()},
                    'own_cgroup_files': sorted(self.own_cgroup_files),
                    'collection_failed': self.collection_failed, 'exitstatus': int(exitstatus)}
        if hasattr(session.config, 'workerinput'):
            session.config.workeroutput['gate_coverage'] = coverage
        elif path := os.environ.get('GATE_COLLECTION_REPORT'):
            Path(path).write_text(json.dumps(coverage) + '\n')


@pytest.hookimpl(specname='pytest_configure')
def pytest_gate_configure(config):
    if 'GATE_ASSIGNED_FILES' not in os.environ:
        return  # Post-merge CI uses the exact exclusion hook without a gate assignment.
    try:
        files = json.loads(os.environ['GATE_ASSIGNED_FILES'])
        if (not isinstance(files, list) or not files or
                any(not isinstance(path, str) or not path or
                    PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                    for path in files) or len(files) != len(set(files))):
            raise ValueError('assignment must contain unique repository-relative files')
        root = Path(config.rootpath)
        if any(not (root / path).is_file() for path in files):
            raise ValueError('an assigned test file is missing')
    except (KeyError, ValueError, TypeError) as error:
        raise pytest.UsageError('invalid gate collection assignment: ' + str(error)) from error
    # Match explicit unusual filenames without assertion-rewriting every imported
    # production dependency, which would change Python's bytecode policy.
    for filename in sorted({Path(path).name for path in files}):
        if not any(fnmatch(filename, pattern) for pattern in config.getini('python_files')):
            config.addinivalue_line('python_files', filename)
    config.pluginmanager.register(GateCollection(root, files), 'gate-assigned-collection')


@pytest.hookimpl(specname='pytest_collection_modifyitems', tryfirst=True)
def pytest_gate_exact_exclusions(config, items):
    raw = os.environ.get('GATE_DESELECT_NODEIDS')
    if raw is None and 'GATE_ASSIGNED_FILES' not in os.environ:
        quarantine = Path(config.rootpath) / 'tools/ci/gate_quarantine.json'
        excluded = [row['target'] for row in json.loads(quarantine.read_text())['entries']
                    if '::' in row['target']]
    else:
        excluded = json.loads(raw or '[]')
    if not isinstance(excluded, list) or any(not isinstance(node, str) for node in excluded):
        raise pytest.UsageError('invalid gate exact exclusions')
    targets = set(excluded)
    deselected = [item for item in items if item.nodeid in targets]
    items[:] = [item for item in items if item.nodeid not in targets]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
