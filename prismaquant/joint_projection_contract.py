"""Packaged qualification and serialized joint-projection arithmetic contract."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from .digests import bytes_sha256hex


SCHEMA = 'prismaquant.joint_projection_backend.v1'


FUSED_NAME = 'fused_fp32_v1'


QUALIFICATION_PATH = Path(__file__).with_name('kernels') / 'joint_projection_reduce_qualification.json'


REFERENCE_IDENTITY = {'schema': SCHEMA, 'name': 'torch', 'expression': '(left * right).sum()'}


@lru_cache(maxsize=1)
def projection_qualification():
    # Immutable package metadata, like the loaded code; row admission must not
    # reopen the qualification file for every format/probe result.
    data = QUALIFICATION_PATH.read_bytes()
    value = json.loads(data)
    if value.get('schema') != 'prismaquant.joint_projection_qualification.v1' or value.get('status') != 'qualified':
        raise RuntimeError('joint projection backend has no qualified packaged runtime')
    return value, bytes_sha256hex(data)


def validate_projection_backend_identity(identity):
    """Validate the serialized arithmetic contract without loading GPU code."""
    if identity == REFERENCE_IDENTITY:
        return
    qualification, digest = projection_qualification()
    expected = {'schema': SCHEMA, 'name': FUSED_NAME, 'qualification_sha256': digest,
                'build': qualification['build'], 'runtime': qualification['runtime'],
                'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    if identity != expected:
        raise ValueError('joint projection arithmetic has an unqualified backend identity')
