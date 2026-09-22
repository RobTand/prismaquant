"""Real registered-op comparison of whole-stage and routed-slice A4 QDQ."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from prismaquant import format_registry
from prismaquant.joint_aura import activation_identity
from prismaquant.joint_cost_quantum import bind_joint_served_quantizer
from prismaquant.joint_served_activation import FORMAT, verify_policy, activate_policy, joint_activation_maxima
from prismaquant.perturbed_x_cache import _activation_qdq

p = argparse.ArgumentParser()
p.add_argument('--policy', required=True)
p.add_argument('--policy-sha256', required=True)
a = p.parse_args()
binding = {'path': a.policy, 'sha256': a.policy_sha256}
policy = verify_policy(binding)
cache = SimpleNamespace(activation_max_abs=dict(policy['qualification_max_abs']),
                        weights={(n, FORMAT): 'not-loaded-in-this-activation-probe' for n in policy['effective_max_abs']})
before = dict(cache.activation_max_abs)
activate_policy(cache, binding)
maxima = joint_activation_maxima(cache)
identity = bind_joint_served_quantizer({'probe': [FORMAT]})
spec = format_registry.get_format(FORMAT)
fp8 = format_registry.get_format('TESSERA_E4M3_K1_R1024')
observations = []
for key, group in policy['executed_grouping']['groups'].items():
    if '.layers.8.' not in group['module']:
        continue
    names = sorted(group['members'], key=lambda n: before[n])
    first, second = names[0], names[-1]
    x = torch.linspace(-group['max_abs'], group['max_abs'], 4 * 4096,
                       device='cuda', dtype=torch.bfloat16).reshape(4, 4096)
    full = _activation_qdq(x, spec, maxima, first)
    split = torch.cat([_activation_qdq(x[:2], spec, maxima, first),
                       _activation_qdq(x[2:], spec, maxima, second)])
    original = _activation_qdq(x, spec, before, first)
    assert torch.equal(full, split), 'grouped row slicing changed actual QDQ bytes'
    assert torch.isfinite(full).all().item()
    assert torch.equal(_activation_qdq(x, fp8, maxima, first), _activation_qdq(x, fp8, before, first))
    assert activation_identity(fp8, maxima, first) == activation_identity(fp8, before, first)
    actual = activation_identity(spec, maxima, first)
    assert actual['input_global_scale'] == group['input_global_scale']
    observations.append({'group': key, 'input_global_scale': actual['input_global_scale'],
        'max_abs': actual['activation_max_abs'], 'whole_vs_slices_equal': True,
        'different_from_historical_per_unit': int((full != original).sum().item()),
        'original_minimum_max_abs': before[first], 'shape': list(x.shape)})
assert len(observations) == 2 and all(v['different_from_historical_per_unit'] > 0 for v in observations)
assert cache.activation_max_abs == before
torch.cuda.synchronize()
print(json.dumps({'status': 'passed', 'policy': binding, 'served_quantizer': identity,
                  'old_maxima_unchanged': True, 'a8_qdq_and_identity_unchanged': True,
                  'groups': observations}, sort_keys=True))
