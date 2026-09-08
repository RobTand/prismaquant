"""CPU/meta checks of the actual derived image and original decorated GLM graph."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-build', type=Path, required=True)
    parser.add_argument('--image-build-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    from prismaquant.glm_source_derivative import SCHEMA, VERSION, bind_source_derivative, sha256
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.model_profiles.default import DefaultProfile
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'this gate must remain CPU-only'
    path = Path('/mnt/shared/models/GLM-5.3-Flash-BF16/config.json')
    assert sha256(path) == '33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
    config = AutoConfig.from_pretrained(str(path.parent), trust_remote_code=False)
    with init_empty_weights():
        model = modeling.Glm5NextForConditionalGeneration._from_config(config, attn_implementation='eager')
    model.eval().requires_grad_(False)
    assert all(parameter.device.type == 'meta' for parameter in model.parameters())
    rejected = []
    def refuses(label, call, match):
        try:
            call()
        except ValueError as error:
            assert match in str(error), (label, str(error))
            rejected.append(dict(case=label, error=str(error)))
        else:
            raise AssertionError(label + ' was incorrectly accepted')
    refuses('omitted_explicit_policy', lambda: source_execution_identity(model), 'explicit derivative binding')
    policy = dict(schema=SCHEMA, version=VERSION,
                  image_build=dict(path=str(args.image_build), sha256=args.image_build_sha256))
    refuses('wrong_profile', lambda: bind_source_derivative(model, DefaultProfile(), policy), 'profile does not declare')
    derivative = bind_source_derivative(model, Glm5NextProfile(), policy)
    initial = source_execution_identity(model)
    assert initial['schema'] == 'prismaquant.joint_aura.source_execution.v2'
    assert initial['source_derivative'] == derivative and len(derivative['gates']) == 34
    first = next(module for module in model.modules() if type(module) is modeling.Glm5NextTextLinearAttention)
    original_bound = first.forget_gate.safe_gate_lower_bound
    try:
        first.forget_gate.safe_gate_lower_bound = 1.
        refuses('changed_live_gate', lambda: source_execution_identity(model), 'gate bound differs')
    finally:
        first.forget_gate.safe_gate_lower_bound = original_bound
    try:
        config.text_config.linear_lower_bound = -4.
        refuses('changed_gate_config', lambda: source_execution_identity(model), 'gate bound differs')
    finally:
        config.text_config.linear_lower_bound = original_bound
    actual = modeling.chunk_kimi_delta_attention
    try:
        modeling.chunk_kimi_delta_attention = lambda *args, **kwargs: None
        refuses('substituted_callable', lambda: source_execution_identity(model), 'callable code changed')
    finally:
        modeling.chunk_kimi_delta_attention = actual
    cells = dict(zip(actual.__code__.co_freevars, actual.__closure__))
    prior = cells['implementation'].cell_contents
    try:
        cells['implementation'].cell_contents = lambda *args, **kwargs: None
        refuses('selected_other_dispatch', lambda: source_execution_identity(model), 'not the decorated Torch fallback')
    finally:
        cells['implementation'].cell_contents = prior
    try:
        first.forward = lambda *args, **kwargs: None
        refuses('instance_forward_override', lambda: source_execution_identity(model), 'instance forward substitution')
    finally:
        del first.forward
    refuses('remove_binding', lambda: bind_source_derivative(model, Glm5NextProfile(), None), 'cannot remove')
    image = os.environ['PRISMAQUANT_CONTAINER_CONTENT_SHA256']
    try:
        os.environ['PRISMAQUANT_CONTAINER_CONTENT_SHA256'] = '0'*64
        refuses('changed_actual_image', lambda: source_execution_identity(model), 'image evidence changed')
    finally:
        os.environ['PRISMAQUANT_CONTAINER_CONTENT_SHA256'] = image
    assert source_execution_identity(model) == initial
    assert not torch.cuda.is_initialized()
    result = dict(status='complete', device='meta', cuda_initialized=False, native_or_gpu_work=False,
        torch_version=torch.__version__, source_derivative=derivative, execution_identity=initial,
        actual_kda_modules=34, refusals=rejected, parameter_devices=sorted({str(p.device) for p in model.parameters()}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete', actual_kda_modules=34, refusals=len(rejected), cuda_initialized=False)))


if __name__ == '__main__':
    main()
