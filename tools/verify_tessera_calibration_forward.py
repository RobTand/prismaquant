"""Compare the calibration-owner move on fixed model and calibration bytes.

Execute only inside a PrismaBuild-admitted known-image GPU action. The baseline
is the actual pre-move campaign source, sealed separately in the input checkout.
This records numerical parity and instrument coverage, not a performance claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-campaign', type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from prismaquant.digests import file_sha256hex, indent2_json_file_bytes
    if file_sha256hex(args.baseline_campaign) != args.baseline_sha256:
        raise ValueError('baseline campaign bytes differ from the selected source')
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    import torch
    from transformers import AutoModelForCausalLM
    from prismaquant.tessera_calibration_forward import capture_activation_statistics
    from prismaquant.tensor_digests import tensor_identity
    from experiments.workspace_netdata import NetdataWriter, sample_netdata

    if not torch.cuda.is_available():
        raise RuntimeError('capture comparison requires an admitted CUDA device')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model_path = args.input_root / 'model'
    calibration_path = args.input_root / 'calibration.pt'
    tokens = torch.load(calibration_path, map_location='cpu', weights_only=True)
    if tuple(tokens.shape) != (4, 256):
        raise ValueError('comparison requires the sealed four-window calibration')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='eager').to('cuda').eval()
    targets = [name for name, module in model.named_modules()
               if name.startswith('model.layers.0.') and isinstance(module, torch.nn.Linear)]
    if len(targets) != 7:
        raise ValueError('comparison requires the seven first-layer Qwen projections')
    baseline_spec = importlib.util.spec_from_file_location(
        'prismaquant._capture_owner_baseline', args.baseline_campaign)
    baseline = importlib.util.module_from_spec(baseline_spec)
    sys.modules[baseline_spec.name] = baseline
    baseline_spec.loader.exec_module(baseline)
    collectors = [('before', baseline._collect_activations),
                  ('after', capture_activation_statistics)]
    args.output.mkdir()
    stop = threading.Event()
    errors, samples = [], []

    def sample():
        with (args.output / 'netdata-both-boxes.jsonl').open('w') as handle:
            writer = NetdataWriter(handle)
            while not stop.is_set():
                for host in ('sparky', 'sparklina'):
                    try:
                        record = sample_netdata(host)
                        writer.write(record)
                        samples.append((host, record['time']))
                    except Exception as exc:
                        errors.append(f'{host}: {type(exc).__name__}: {exc}')
                stop.wait(1)

    sampler = threading.Thread(target=sample, name='capture-netdata')
    sampler.start()
    evidence = []
    expected = None
    original_hooks = {name: len(model.get_submodule(name)._forward_pre_hooks)
                      for name in targets}
    try:
        for label, collect in collectors:
            started = time.time()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True) as profile:
                actual = collect(model, targets, tokens.split(1), 32, 'cuda', want_hessian=True)
                torch.cuda.synchronize()
            rows, hessians, counts, maxima = actual
            assert all(rows[name].shape[0] == 32 for name in targets)
            assert set(counts.values()) == {1024}
            assert all(torch.isfinite(hessians[name]).all() for name in targets)
            assert original_hooks == {name: len(model.get_submodule(name)._forward_pre_hooks)
                                      for name in targets}
            if expected is None:
                expected = actual
            else:
                assert expected[2:] == actual[2:]
                for old, new in zip(expected[:2], actual[:2]):
                    assert old.keys() == new.keys()
                    assert all(torch.equal(old[name], new[name]) for name in old)
            profile.export_chrome_trace(str(args.output / f'{label}-torch.json'))
            (args.output / f'{label}-torch.txt').write_text(
                profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=70))
            evidence.append(dict(
                arm=label, started=started, ended=time.time(),
                rows={name: tensor_identity(rows[name]) for name in targets},
                hessians={name: tensor_identity(hessians[name]) for name in targets},
                counts=counts, maxima=maxima,
                profile_sha256=file_sha256hex(args.output / f'{label}-torch.json')))
    finally:
        stop.set()
        sampler.join(20)
        if sampler.is_alive():
            errors.append('Netdata sampler did not finish')
        (args.output / 'instrument-status.json').write_bytes(indent2_json_file_bytes(
            dict(samples=samples, errors=errors)))
    if errors:
        raise RuntimeError(f'capture evidence is incomplete: {errors}')
    if any(sum(host == wanted for host, _ in samples) < 2
           for wanted in ('sparky', 'sparklina')):
        raise RuntimeError('capture comparison needs paired Netdata samples on both hosts')
    artifacts = {path.name: dict(sha256=file_sha256hex(path), bytes=path.stat().st_size)
                 for path in args.output.iterdir() if path.is_file()}
    result = dict(
        schema='prismaquant.calibration_owner_comparison.v1', numerical_parity=True,
        performance_claim=None, affinity=sorted(os.sched_getaffinity(0)),
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
        baseline_campaign_sha256=args.baseline_sha256,
        calibration_file_sha256=file_sha256hex(calibration_path),
        calibration=tensor_identity(tokens), targets=targets, score_row_cap=32,
        model_source={path.name: file_sha256hex(path)
                      for path in sorted(model_path.glob('*.safetensors'))},
        evidence=evidence, artifacts=artifacts,
        provenance_change='Current package source closure includes the new calibration owner; '
                          'receipt operation spellings and historical migration rules are unchanged.')
    (args.output / 'comparison.json').write_bytes(indent2_json_file_bytes(result))
    print(json.dumps(dict(numerical_parity=True, output=str(args.output),
                          targets=len(targets), netdata_samples=len(samples))), flush=True)


if __name__ == '__main__':
    main()
