"""Whole-target statistics windows, derived from the joint lease's own groups.

This plan owns names and immutable scalar metadata only. It does not schedule
replay, load candidates, install observers or change probe arithmetic. Source,
activation, cotangent, graph, PWC and transient memory need separate admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .joint_aura import _joint_projection_requirements, identity_sha256

SCHEMA = 'prismaquant.joint_statistics_target_windows.v1'


@dataclass(frozen=True)
class JointStatisticsGroup:
    formats: tuple[str, ...]
    activation_identity_json: str

    def as_dict(self):
        return dict(formats=list(self.formats), activation=json.loads(self.activation_identity_json))


@dataclass(frozen=True)
class JointStatisticsTarget:
    name: str
    shape: tuple[int, int]
    statistics_bytes: int
    groups: tuple[JointStatisticsGroup, ...]

    def as_dict(self):
        return dict(name=self.name, shape=list(self.shape), statistics_bytes=self.statistics_bytes,
                    groups=[dict(index=index, **group.as_dict()) for index, group in enumerate(self.groups)])


@dataclass(frozen=True)
class JointStatisticsTargetPlan:
    max_statistics_bytes: int
    targets: tuple[JointStatisticsTarget, ...]
    windows: tuple[tuple[str, ...], ...]
    window_statistics_bytes: tuple[int, ...]
    projection_backend_identity_json: str

    @property
    def total_statistics_bytes(self):
        return sum(target.statistics_bytes for target in self.targets)

    def as_dict(self):
        return dict(schema=SCHEMA, max_statistics_bytes=self.max_statistics_bytes,
            target_order='lexicographic', grouping_order='original_format_insertion_order',
            targets=[target.as_dict() for target in self.targets],
            windows=[dict(names=list(names), statistics_bytes=size)
                     for names, size in zip(self.windows, self.window_statistics_bytes)],
            projection_backend=json.loads(self.projection_backend_identity_json))

    @property
    def identity_sha256(self):
        return identity_sha256(self.as_dict())


def plan_joint_statistics_target_windows(modules, specs_by_qname, *, max_statistics_bytes,
                                         activation_max_abs=None, projection_backend=None):
    """Validate all targets and pack deterministic whole-target windows.

    Meta Linear modules are supported by the torch reference backend for
    geometry-only CPU planning. A fused backend still requires its actual
    prewarmed device. Neither mode certifies source residency or a full fit.
    Dynamic callable distinctions become separate ordered format groups; raw
    callable IDs never leave the shared in-process grouping calculation.
    """
    if type(max_statistics_bytes) is not int or max_statistics_bytes <= 0:
        raise ValueError('joint statistics planning requires a positive integer statistics budget')
    if not modules:
        raise ValueError('joint statistics planning requires nonempty target coverage')
    backend, requirements = _joint_projection_requirements(modules, specs_by_qname,
        activation_max_abs=activation_max_abs, projection_backend=projection_backend)
    targets = tuple(JointStatisticsTarget(name, row.shape, row.statistics_bytes,
        tuple(JointStatisticsGroup(group.formats, group.activation_identity_json) for group in row.groups))
        for name, row in sorted(requirements.items()))
    for target in targets:
        if target.statistics_bytes > max_statistics_bytes:
            raise RuntimeError(f'joint statistics target {target.name} requires {target.statistics_bytes} '
                               f'bytes, exceeding statistics budget {max_statistics_bytes}')
    windows, sizes, current, used = [], [], [], 0
    for target in targets:
        if current and used + target.statistics_bytes > max_statistics_bytes:
            windows.append(tuple(current))
            sizes.append(used)
            current, used = [], 0
        current.append(target.name)
        used += target.statistics_bytes
    windows.append(tuple(current))
    sizes.append(used)
    return JointStatisticsTargetPlan(max_statistics_bytes, targets, tuple(windows), tuple(sizes),
        json.dumps(backend.identity, sort_keys=True, separators=(',', ':'), allow_nan=False))
