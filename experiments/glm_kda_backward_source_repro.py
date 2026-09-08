"""CPU numerical reproduction of the pinned GLM KDA fallback, never a runtime patch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


PIN = '2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b'
ROOT = Path(__file__).parent / 'measurements/glm-kda-backward-source-repro-20260908'
OLD = '(g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()'
NEW = '(g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(mask.triu(diagonal=1).unsqueeze(-1), 0).exp().float()'


def load_functions():
    pinned = json.loads((ROOT / 'pinned-functions.json').read_text())
    assert pinned['modeling_source_sha256'] == PIN
    sources = pinned['functions']
    for row in sources.values():
        assert hashlib.sha256(row['source'].encode()).hexdigest() == row['sha256']
    source = sources['l2norm']['source'] + '\n' + sources['chunk_kimi_delta_attention']['source']
    assert source.count(OLD) == 1
    functions = {}
    for name, code in [('original', source), ('premask', source.replace(OLD, NEW)),
                       ('float64_oracle', source.replace('torch.float32', 'torch.float64').replace('.float()', '.double()'))]:
        scope = {'torch': torch, 'F': F}
        exec(compile(code, f'<pinned_glm_kda_{name}>', 'exec'), scope)
        functions[name] = scope['chunk_kimi_delta_attention']
    return functions, pinned


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(tensor):
    flat = tensor.detach().reshape(-1)
    finite = torch.isfinite(flat)
    return dict(elements=flat.numel(), finite=int(finite.sum()), nan=int(torch.isnan(flat).sum()),
                inf=int(torch.isinf(flat).sum()), finite_nonzero=int((finite & (flat != 0)).sum()))


def run_case(functions, *, length, gate, dtype, use_norm, state):
    generator = torch.Generator().manual_seed(711)
    shape = (1, length, 1, 2)
    values = [torch.randn(shape, generator=generator) * .1 for _ in range(3)]
    values += [torch.full(shape, gate), torch.full(shape[:-1], .25)]
    if state:
        values += [torch.randn((1, 1, 2, 2), generator=generator) * .01]
    # The oracle receives exactly representable copies of each low-precision input.
    values = [x.to(dtype) for x in values]
    stimulus = torch.randn(shape, generator=generator).to(dtype)
    outputs, gradients, rows = {}, {}, {}
    for name, function in functions.items():
        target_dtype = torch.float64 if name == 'float64_oracle' else dtype
        inputs = [x.to(target_dtype).detach().clone().requires_grad_() for x in values]
        output, final = function(*inputs[:5], initial_state=inputs[5] if state else None,
                                 output_final_state=state, use_qk_l2norm_in_kernel=use_norm)
        loss = (output * stimulus.to(target_dtype)).sum()
        if state:
            loss = loss + final.sum() * .125
        loss.backward()
        outputs[name], gradients[name] = output, [x.grad for x in inputs]
        rows[name] = dict(output=stats(output), output_sha256=digest(output),
                          gradients={key: stats(x.grad) for key, x in zip(('q', 'k', 'v', 'g', 'beta', 'state'), inputs)})
    assert torch.equal(outputs['original'], outputs['premask']), 'forward bytes changed'
    assert digest(outputs['original']) == digest(outputs['premask'])
    assert torch.isfinite(outputs['original']).all()
    assert all(torch.isfinite(x).all() for x in gradients['premask'])
    assert all(torch.isfinite(x).all() for x in gradients['float64_oracle'])
    errors = {}
    # BF16 backward rounds intermediate gradients; FP32 is the numerical oracle gate.
    for key, actual, oracle in zip(('q', 'k', 'v', 'g', 'beta', 'state'), gradients['premask'], gradients['float64_oracle']):
        difference = (actual.double() - oracle).abs()
        errors[key] = dict(max_abs=float(difference.max()), relative_l2=float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(oracle).clamp_min(1e-30)))
        if dtype == torch.float32:
            torch.testing.assert_close(actual.double(), oracle, rtol=3e-4, atol=2e-7)
    original_finite = all(torch.isfinite(x).all() for x in gradients['original'])
    if gate == -.1:
        assert original_finite, 'control unexpectedly failed'
        for a, b in zip(gradients['original'], gradients['premask']):
            assert torch.equal(a, b), 'non-overflow control derivative changed'
    else:
        assert not original_finite, 'strong-gate source regression did not reproduce'
    return dict(length=length, gate=gate, dtype=str(dtype), use_norm=use_norm, initial_state=state,
                original_all_gradients_finite=bool(original_finite), forward_byte_equal=True,
                variants=rows, premask_vs_float64_gradient_errors=errors)


def expression_repro():
    original = torch.full((64,), -5., requires_grad=True)
    stable = original.detach().clone().requires_grad_()
    rows = {}
    for name, g in [('original', original), ('premask', stable)]:
        cumulative = g.cumsum(0)
        delta = cumulative[:, None] - cumulative[None, :]
        mask = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=0)
        exponent = (delta.masked_fill(mask, 0) if name == 'premask' else delta).exp()
        used = exponent.masked_fill(mask, 0)
        used.sum().backward()
        rows[name] = dict(delta_min=float(delta.detach().min()), delta_max=float(delta.detach().max()),
                          exponent=stats(exponent), used=stats(used), used_sha256=digest(used), gradient=stats(g.grad))
    assert rows['original']['used_sha256'] == rows['premask']['used_sha256']
    assert rows['original']['exponent']['inf'] > 0 and rows['original']['gradient']['nan'] > 0
    assert rows['premask']['exponent']['inf'] == 0 and rows['premask']['gradient']['finite'] == 64
    # Independent analytic derivative: d exp(sum(g[j+1:i+1])) / d g[t].
    analytic = torch.zeros(64, dtype=torch.float64)
    for i in range(64):
        for j in range(i):
            analytic[j + 1:i + 1] += torch.exp(torch.tensor(-5. * (i - j), dtype=torch.float64))
    torch.testing.assert_close(stable.grad.double(), analytic, rtol=2e-5, atol=1e-8)
    rows['analytic_max_abs_error'] = float((stable.grad.double() - analytic).abs().max())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-original-finite', action='store_true')
    args = parser.parse_args()
    functions, pinned = load_functions()
    expression = expression_repro()
    cases = [run_case(functions, length=length, gate=gate, dtype=dtype, use_norm=norm, state=state)
             for length, gate, dtype, norm, state in [
                 (64, -5., torch.float32, False, False), (65, -5., torch.float32, True, True),
                 (64, -.1, torch.float32, True, False), (65, -.1, torch.float32, False, True),
                 (64, -5., torch.bfloat16, True, False), (65, -5., torch.bfloat16, True, True)]]
    result = dict(schema='prismaquant.glm_kda_source_cpu_reproduction.v1', torch_version=torch.__version__,
                  device='cpu', threads=torch.get_num_threads(), source=pinned, expression=expression, cases=cases,
                  proposed_expression=dict(before=OLD, after=NEW),
                  scope='Exact pinned fallback bodies with decorators omitted only for CPU source attribution. No native runtime, capture, image, checkpoint, profile or probe identity changed.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(cases=len(cases), expression=expression, all_forward_bytes_equal=True)))
    if args.require_original_finite:
        assert all(row['original_all_gradients_finite'] for row in cases), 'pinned original fallback has nonfinite gradients with finite complete inputs and outputs'


if __name__ == '__main__':
    main()
