"""Create immutable seeded tiny-GLM inputs in the qualified derivative image.

CPU-only preparation: source binding and tensor finiteness are qualified here;
forward/backward GPU kernels are deliberately qualified by the later action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA, validate_streamed_model_identity
from prismaquant.glm_source_derivative import bind_source_derivative
from prismaquant.joint_aura import source_execution_identity
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from tests.test_glm5_next_streamed_forward_parity import _build_tiny_model


DERIVATIVE = {
    'schema': 'prismaquant.glm_source_derivative.v1',
    'version': 'glm_kda_causal_exp_v1',
    'image_build': {
        'path': '/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/derivative-image-build-03/result.json',
        'sha256': 'e8426d3554180219a0fff421149118a9e0a17ef7a2091c5a03e21834dd1744c5',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = _build_tiny_model()
    model.config._attn_implementation = 'eager'
    model.config.text_config._attn_implementation = 'eager'
    model.config._experts_implementation = 'grouped_mm'
    model.config.text_config._experts_implementation = 'grouped_mm'
    derivative = bind_source_derivative(model, Glm5NextProfile(), DERIVATIVE)
    ids = torch.randint(2, 128, (4, 32), generator=torch.Generator().manual_seed(1962))
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    assert all(torch.isfinite(value).all() for value in state.values())
    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.output / 'model.safetensors'
    config_path = args.output / 'config.json'
    ids_path = args.output / 'tokens.safetensors'
    save_file(state, str(model_path))
    save_file({'calibration_ids': ids}, str(ids_path))
    config = model.config.to_dict()
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    values = {'config': config, 'weight_map': {name: model_path.name for name in state},
              'shards': [{'path': str(model_path), 'size': model_path.stat().st_size,
                          'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}]}
    identity = {'schema': STREAMED_MODEL_IDENTITY_SCHEMA, 'source': str(args.output),
                'resolved_commit': None,
                'content_sha256': canonical_json_sha256(values, where='tiny GLM generated checkpoint'),
                **values}
    validate_streamed_model_identity(identity, where='tiny GLM preparation')
    metadata = {
        'schema': 'prismaquant.research.tiny_glm_source.v1',
        'scope': 'synthetic seeded two-layer GLM mechanism control; not original GLM weights',
        'construction_seed': 20260826, 'tokens_seed': 1962,
        'n_sequences': 4, 'sequence_length': 32, 'n_probes': 2,
        'probe_seed_base': 7000, 'global_token_count': 128,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'container_content_sha256': os.environ['PRISMAQUANT_CONTAINER_CONTENT_SHA256'],
        'source_derivative': derivative, 'derivative_policy': DERIVATIVE,
        'source_execution': source_execution_identity(model),
        'model_identity': identity,
        'inputs': {path.name: {'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (model_path, config_path, ids_path)},
        'tokens_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        'qualified': ['CPU construction', 'all tensors finite', 'exact derivative callable binding'],
        'unqualified': ['GPU forward/backward', 'grouped_mm dtype support', 'Stage A/direct parity'],
    }
    (args.output / 'source.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(f'PASS: tiny GLM source binding; {sum(v.numel() for v in state.values())} stored elements')
    print(f'Checkpoint SHA {values["shards"][0]["sha256"]}; tokens SHA {metadata["tokens_sha256"]}')


if __name__ == '__main__':
    main()
