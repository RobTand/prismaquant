#!/usr/bin/env python3
"""Deterministic duration-balanced shards shared by gate and nightly CI."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import sys
import math
from pathlib import Path, PurePosixPath


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prismaquant.digests import text_sha256hex

def _cost(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('durations must be finite nonnegative numbers, not booleans')
    return float(value)


def default_unknown_seconds(durations: Mapping[str, float]) -> float:
    """Nearest-rank p90; empty or all-zero inventories use one second."""
    costs = sorted(_cost(value) for value in durations.values())
    return (costs[math.ceil(len(costs) * .9) - 1] or 1.0) if costs else 1.0


def node_owner(nodeid: str, partition: dict) -> int:
    """Use measured placement, or a process/architecture-independent new-node owner."""
    owners = partition['owners'][nodeid.split('::', 1)[0]]
    if nodeid in owners:
        return owners[nodeid]
    return int(text_sha256hex(nodeid)[:16], 16) % partition['count']


def assigned_nodes(collected: Mapping[str, Sequence[str]], partition: dict,
                   excluded: Sequence[str] = ()) -> list[str]:
    excluded = set(excluded)
    return sorted(node for nodes in collected.values() for node in nodes
                  if node not in excluded and node_owner(node, partition) == partition['index'])


def validate_node_partition(partition: dict, files: Sequence[str]) -> None:
    if (not isinstance(partition, dict) or type(partition.get('count')) is not int or
            partition['count'] < 2 or type(partition.get('index')) is not int or
            not 0 <= partition['index'] < partition['count'] or
            not isinstance(partition.get('owners'), dict) or not partition['owners'] or
            not partition['owners'].keys() <= set(files)):
        raise ValueError('invalid node partition')
    for path, owners in partition['owners'].items():
        if not isinstance(owners, dict) or any(
                not isinstance(node, str) or not node.startswith(path + '::') or
                type(index) is not int or not 0 <= index < partition['count']
                for node, index in owners.items()):
            raise ValueError('invalid recorded node owner')


def shards(files: Sequence[str], durations: Mapping[str, float], count: int,
           *, unknown_seconds: float, node_durations: Mapping[str, float] | None = None,
           runner_factors: Sequence[float] | None = None, excluded: Sequence[str] = ()) -> list[dict]:
    """Pack longest units first; split files larger than total/count by live node ID.

    Every shard collects split files. Recorded nodes use LPT placement and new
    nodes use a stable hash, so a stale inventory cannot drop or duplicate tests.
    Without node data this retains the original whole-file partition contract.
    Runner factors convert the recorded costs into expected runner time.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if len(files) != len(set(files)):
        raise ValueError('duplicate files')
    factors = [1.0] * count if runner_factors is None else [_cost(value) for value in runner_factors]
    if len(factors) != count or any(value == 0 for value in factors):
        raise ValueError('runner factors must give one positive cost multiplier per shard')
    costs = {path: _cost(value) for path, value in durations.items()}
    unknown = _cost(unknown_seconds)
    nodes = {} if node_durations is None else {node: _cost(value) for node, value in node_durations.items()}
    if any(not isinstance(node, str) or '::' not in node for node in nodes):
        raise ValueError('node durations must name exact pytest node IDs')
    excluded = set(excluded)
    for node in excluded:
        path = node.split('::', 1)[0]
        if node in nodes and path in costs:
            costs[path] = max(0.0, costs[path] - nodes[node])
        nodes.pop(node, None)
    target = sum(costs.get(path, unknown) for path in files) / count
    split = {path: {} for path in sorted(files)
             if count > 1 and node_durations is not None and costs.get(path, unknown) > target}
    units = []
    for path in files:
        cost = costs.get(path, unknown)
        if path not in split:
            units.append((cost, path, ''))
            continue
        measured = {node: value for node, value in nodes.items() if node.split('::', 1)[0] == path}
        total = sum(measured.values())
        if measured:
            units.extend((cost * value / total if total else cost / len(measured), path, node)
                         for node, value in measured.items())
    result = [{'index': index, 'files': sorted(split), 'estimated_seconds': 0.0, 'unknown_files': []}
              for index in range(count)]
    for cost, path, node in sorted(units, key=lambda unit: (-unit[0], unit[1], unit[2])):
        row = min(result, key=lambda row: (row['estimated_seconds'] + cost * factors[row['index']], row['index']))
        if node:
            split[path][node] = row['index']
        else:
            row['files'].append(path)
            if path not in costs:
                row['unknown_files'].append(path)
        row['estimated_seconds'] += cost * factors[row['index']]
    for row in result:
        # Without recorded nodes, live collection still splits the file by hash.
        row['estimated_seconds'] += sum(costs.get(path, unknown) * factors[row['index']] / count
                                        for path, owners in split.items() if not owners)
        row['files'].sort()
        row['unknown_files'].sort()
        if split:
            row['node_partition'] = {'index': row['index'], 'count': count, 'owners': split}
    return result



def selection_shard_count(files: Sequence[str], durations: Mapping[str, float], maximum: int,
                          *, workers: int, single_shard_seconds: float,
                          node_durations: Mapping[str, float] | None = None, excluded: Sequence[str] = ()) -> int:
    """Keep small PRs on one slot; larger selections share the full gate matrix."""
    unknown = default_unknown_seconds(durations)
    # Reuse partition validation so an unknown file or invalid cost cannot be free.
    packed = shards(files, durations, workers, unknown_seconds=unknown, node_durations=node_durations,
                    excluded=excluded)
    wall = max(row['estimated_seconds'] for row in packed)
    node_counts = Counter(node.split('::', 1)[0] for node in (node_durations or {})
                          if node not in set(excluded))
    useful = sum(max(1, node_counts[path]) for path in files if durations.get(path, unknown) > 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('maximum must be a positive integer')
    threshold = _cost(single_shard_seconds)
    return 1 if wall <= threshold else min(maximum, max(1, useful))


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--files-from', type=Path, required=True)
    common.add_argument('--durations', type=Path, required=True)
    common.add_argument('--plan', type=Path, help='exact node exclusions from the gate plan')
    common.add_argument('--shards', type=int, required=True)
    common.add_argument('--unknown-seconds', type=float)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    matrix = commands.add_parser('matrix', parents=[common])
    matrix.add_argument('--runners', required=True)
    matrix.add_argument('--single-shard-seconds', type=float, help='selection only: keep cheap work in one job')
    matrix.add_argument('--workers', type=int, default=4)
    matrix.add_argument('--output', type=Path, required=True)
    shard = commands.add_parser('shard', parents=[common])
    shard.add_argument('--matrix', type=Path, help='recorded runner factors from the planning job')
    shard.add_argument('--index', type=int, required=True)
    shard.add_argument('--output-files', type=Path, required=True)
    shard.add_argument('--output-assignment', type=Path)
    args = parser.parse_args(argv)
    try:
        files = [line.strip() for line in args.files_from.read_text().splitlines() if line.strip()]
        if any(PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts for path in files):
            raise ValueError('files must be repository-relative paths without traversal')
        excluded = json.loads(args.plan.read_text()).get('deselect', []) if args.plan else []
        if not isinstance(excluded, list) or any(not isinstance(node, str) for node in excluded):
            raise ValueError('plan exclusions must be a list of node IDs')
        recorded = json.loads(args.durations.read_text())
        durations = recorded['files']
        if not isinstance(durations, dict):
            raise ValueError('duration files must be a mapping')
        unknown = default_unknown_seconds(durations) if args.unknown_seconds is None else args.unknown_seconds
        if args.command == 'matrix' and args.single_shard_seconds is not None:
            args.shards = selection_shard_count(files, durations, args.shards, workers=args.workers,
                                                single_shard_seconds=args.single_shard_seconds,
                                                node_durations=recorded.get('nodes'), excluded=excluded)
        if args.command == 'matrix':
            runners = [name.strip() for name in args.runners.split(',')]
            if not all(runners):
                raise ValueError('runners must be nonempty comma-separated names')
            # Hosted paired-file median x64/ARM ratio is 1.716; use 1.6.
            factors = [1 / 1.6 if runners[index % len(runners)] == 'ubuntu-24.04-arm' else 1.0
                       for index in range(args.shards)]
        else:
            matrix = json.loads(args.matrix.read_text()) if args.matrix else {}
            if matrix and matrix['shards'] != args.shards:
                raise ValueError('matrix shard count does not match requested count')
            factors = matrix.get('runner_cost_factors')
        result = shards(files, durations, args.shards, unknown_seconds=unknown,
                        node_durations=recorded.get('nodes'), runner_factors=factors, excluded=excluded)
        if args.command == 'matrix':
            output = {'schema': 'prismaquant.gate_matrix.v1', 'shards': args.shards, 'runner_cost_factors': factors,
                      'total_estimated_seconds': sum(row['estimated_seconds'] for row in result),
                      'include': [{'index': row['index'], 'runner': runners[row['index'] % len(runners)],
                                   'files': len(row['files']), 'estimated_seconds': row['estimated_seconds']}
                                  for row in result]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + '\n')
        else:
            if not 0 <= args.index < args.shards:
                raise ValueError('index is outside the shard range')
            args.output_files.parent.mkdir(parents=True, exist_ok=True)
            args.output_files.write_text(''.join(path + '\n' for path in result[args.index]['files']))
            if args.output_assignment:
                args.output_assignment.parent.mkdir(parents=True, exist_ok=True)
                args.output_assignment.write_text(json.dumps(result[args.index]) + '\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
