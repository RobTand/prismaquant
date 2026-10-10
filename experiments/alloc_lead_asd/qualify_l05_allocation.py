"""Construct the original-config layer on meta and price its allocations only.

Use the known-good derivative container through PrismaBuild, without a GPU.
No original weight payload or retained activation tensor is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers.models.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration

from experiments.alloc_lead_asd.qualify_tiny_glm import DERIVATIVE
from prismaquant.glm_source_derivative import bind_source_derivative
from prismaquant.layer_streaming import _model_tensor_dtypes
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config_raw, protocol_raw = args.config.read_bytes(), args.protocol.read_bytes()
    assert hashlib.sha256(config_raw).hexdigest() == args.config_sha256
    assert hashlib.sha256(protocol_raw).hexdigest() == args.protocol_sha256
    config = Glm5NextConfig.from_dict(json.loads(config_raw))
    protocol = json.loads(protocol_raw)
    config._attn_implementation = config.text_config._attn_implementation = 'eager'
    config._experts_implementation = config.text_config._experts_implementation = 'grouped_mm'
    with torch.device('meta'):
        model = Glm5NextForConditionalGeneration(config)
    derivative = bind_source_derivative(model, Glm5NextProfile(), DERIVATIVE)
    dtypes = _model_tensor_dtypes(model, torch.bfloat16)
    prefix = f'model.language_model.layers.{args.layer}.'
    rows = []
    for kind, tensors in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in tensors:
            if name.startswith(prefix):
                assert value.is_meta
                dtype = dtypes.get(name, torch.bfloat16 if value.is_floating_point() else value.dtype)
                rows.append({'kind': kind, 'name': name, 'shape': list(value.shape),
                             'dtype': str(dtype), 'bytes': value.numel() * torch.empty((), dtype=dtype).element_size()})
    assert rows and protocol['source_layer_tensor_count'] == len(rows)
    # The existing packer gathers CUDA source projections, then allocates the
    # final packed parameter and consumes them. Reserve both complete sets;
    # do not assume freed allocator blocks return physical UMA memory.
    resident = sum(row['bytes'] for row in rows)
    packed = sum(row['bytes'] for row in rows if '.mlp.experts.' in row['name'])
    source = protocol['all_held_shard_bytes']
    captured = protocol['serialized_capture_file_bytes']
    decoded = protocol['decoded_capture_tensor_bytes']
    # Count client filesystem/pagecache independently of the held memfd. The
    # PB tier is not permission to omit the action's possible clean file charge.
    # The consumer fences load and empties only its free CUDA allocator blocks
    # before constructing a graph, so packing and graph transients are distinct
    # phases. Their contemporaneous maximum, rather than their sum, is reserved.
    graph = 8 << 30
    largest_source = max(row['bytes'] // (
        row['shape'][0] * (2 if row['name'].endswith('gate_up_proj') else 1))
        if '.mlp.experts.' in row['name'] else row['bytes'] for row in rows)
    common = {'possible_client_source_file_cache': source,
              'held_sealed_source_files': source,
              'possible_capture_file_cache': captured,
              'runtime_and_metadata_reserve': 2 << 30}
    load_phase = {'resident_layer': resident, 'conservative_extra_source_packing': packed,
                  'single_reader_pin_and_conversion_bound': largest_source * 2}
    graph_phase = {'resident_layer': resident, 'copied_capture_host_tensors': decoded,
                   'copied_capture_device_tensors': decoded,
                   'capture_concatenation_and_one_sealed_entry_bound': decoded,
                   'graph_qdq_component_and_trace_reserve': graph}
    peak_phase = max(sum(load_phase.values()), sum(graph_phase.values()))
    result = {'schema': f'prismaquant.research.l{args.layer:02d}_allocation_qualification.v1',
              'scope': 'meta construction and explicit prospective byte bounds only; no measured peak',
              'config_sha256': args.config_sha256, 'protocol_sha256': args.protocol_sha256,
              'layer': args.layer, 'live_tensors': rows, 'resident_layer_bytes': resident,
              'packed_expert_bytes': packed, 'common_reservation_components': common,
              'phase_reservation_components': {'load': load_phase, 'graph': graph_phase},
              'prospective_aggregate_bytes': sum(common.values()) + peak_phase,
              'cuda_initialized': torch.cuda.is_initialized(),
              'derivative': derivative,
              'missing': ['actual source consumer CPU qualification', 'synthetic profile supporting graph reserve',
                          'Astra final review', 'actual in-process and PB peak memory evidence']}
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS: meta-only layer{args.layer} resident bytes={resident}, packed expert bytes={packed}')
    print(f'Prospective aggregate bytes={result["prospective_aggregate_bytes"]}; not a measured peak')


if __name__ == '__main__':
    main()
