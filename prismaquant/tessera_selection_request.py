"""Persist the allocation request before selected-wire materialization."""
from __future__ import annotations

import copy
from pathlib import Path

from . import tessera_expert_projection as tep
from .cost_stage_checkpoint import atomic_write_bytes
from .digests import file_sha256hex, indent2_json_file_bytes

REQUEST_SCHEMA = "prismaquant.tessera_selected_wire_request.v1"
_sha = file_sha256hex



def write_selection_request(path, *, layer_config, assignment, cost_path,
                            cost_payload, output_path):
    """Persist decisions and existing evidence without claiming export readiness."""
    from .layer_config import canonicalize_assignment
    if canonicalize_assignment(layer_config) != dict(assignment):
        raise RuntimeError('selection layer config disagrees with resolved assignment')
    block = tep.selection_expert_projection_block(cost_payload, assignment)
    if not block.get(tep.PROJECTION_KEY):
        raise RuntimeError('selected-wire materialization requires a producer projection')
    config = copy.deepcopy(layer_config)
    config['__prismaquant__'].update(block)
    request = dict(schema=REQUEST_SCHEMA, status='selection_only_nonexportable',
                   layer_config=config, assignment=dict(assignment),
                   cost_path=str(Path(cost_path).resolve()), cost_sha256=_sha(cost_path),
                   output_path=str(Path(output_path).resolve()))
    atomic_write_bytes(Path(path), indent2_json_file_bytes(request))
    return request
