"""Exercise the measured ceiling API with synthetic receipt inputs on CPU.

This smoke proves parser behavior only. It does not execute a GPU workload.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prismaquant.container_cache_peak import derive_cache_ceiling_from_receipt


def input_sha256(receipt):
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def main():
    fixtures = runpy.run_path(str(ROOT / 'tests/test_container_cache_ceiling_2608.py'))
    valid = fixtures['_evidence_fixture']()
    quantum = fixtures['_quantum_fixture']()
    results = []
    for name, receipt in [('compile_results', valid), ('quantum_results', quantum)]:
        value = derive_cache_ceiling_from_receipt(receipt, headroom_bytes=1073740801)
        assert value['peak_bytes'] == 1024
        assert value['ceiling_bytes'] == 2147483648
        results.append({'case': name, 'input_sha256': input_sha256(receipt), 'result': value})

    invalid = []
    receipt = copy.deepcopy(valid)
    receipt['device'] = 'cuda'
    receipt['workload'] = {'reservation_bytes': 2147483648}
    invalid.append(('device_only', receipt))
    receipt = copy.deepcopy(valid)
    del receipt['measurement']['samples'][1]['allocated_bytes']
    invalid.append(('missing_peak_sample_bytes', receipt))
    receipt = copy.deepcopy(valid)
    receipt['measurement']['peak_allocated_bytes'] = 2147483648
    invalid.append(('unsupported_aggregate', receipt))
    receipt = copy.deepcopy(valid)
    receipt['measurement']['row_finished_unix'] = 101.0
    invalid.append(('uncovered_row_end', receipt))
    receipt = copy.deepcopy(quantum)
    receipt['device'] = 'cpu'
    invalid.append(('cpu_quantum_with_cuda_compile', receipt))
    for name, receipt in invalid:
        try:
            derive_cache_ceiling_from_receipt(receipt, headroom_bytes=0)
        except ValueError as exc:
            results.append({'case': name, 'input_sha256': input_sha256(receipt), 'refusal': str(exc)})
        else:
            raise AssertionError(f'{name} did not refuse')
    print(json.dumps({'scope': 'CPU parser/API logic only; all receipt inputs are synthetic.',
                      'source': sys.modules[derive_cache_ceiling_from_receipt.__module__].__file__,
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'gpu_execution_claimed': False, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
