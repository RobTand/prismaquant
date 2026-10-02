import torch
import pytest
import torch.nn.functional as F

from prismaquant.kl_fisher import fisher_probe_scalar, fisher_quadratic_form


def _forward_kl(teacher_logits, student_logits, *, temperature=1.0, token_scope="last"):
    if token_scope == "last":
        teacher_logits = teacher_logits[..., -1:, :]
        student_logits = student_logits[..., -1:, :]
    elif token_scope == "causal":
        teacher_logits = teacher_logits[..., :-1, :]
        student_logits = student_logits[..., :-1, :]
    elif token_scope != "all":
        raise ValueError(token_scope)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()


def test_fisher_quadratic_matches_forward_kl_second_order():
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 7)
    delta = 0.05 * torch.randn(2, 4, 7)

    actual = _forward_kl(
        logits,
        logits + delta,
        temperature=1.7,
        token_scope="all",
    )
    approx = fisher_quadratic_form(
        logits,
        delta,
        temperature=1.7,
        token_scope="all",
    )

    assert actual.item() > 0.0
    assert approx.item() > 0.0
    torch.testing.assert_close(
        actual,
        approx,
        rtol=0.08,
        atol=2e-5,
    )


def test_fisher_probe_gradient_is_centered_and_respects_last_scope():
    torch.manual_seed(13)
    logits = torch.randn(1, 3, 11, requires_grad=True)

    scalar = fisher_probe_scalar(
        logits,
        seed=5,
        token_scope="last",
        temperature=1.3,
        distribution="rademacher",
    )
    scalar.backward()

    grad = logits.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:, :-1, :]).item() == 0
    assert torch.count_nonzero(grad[:, -1:, :]).item() > 0
    torch.testing.assert_close(
        grad.sum(dim=-1),
        torch.zeros_like(grad.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_fisher_probe_token_count_override_rescales():
    """The token_count_override rescales the probe by sqrt(real/override).

    Guards the micro-batched AURA path: each micro-batch must normalize by
    the GLOBAL token count, or the gradient summed across M micro-batches is
    sqrt(M)-inflated (and the squared cost M-inflated). Deterministic, bit-
    level — same seed gives the same Rademacher draw.
    """
    import math
    import torch
    from prismaquant.kl_fisher import fisher_probe_scalar

    torch.manual_seed(0)
    logits = torch.randn(4, 8, 16)  # B=4, T=8, V=16; scope="all" -> N=32
    kw = dict(seed=5, token_scope="all", distribution="rademacher")
    base = fisher_probe_scalar(logits, **kw)
    # override == real count is an exact no-op
    same = fisher_probe_scalar(logits, token_count_override=32, **kw)
    assert torch.equal(same, base)
    # override == 4x real count scales the probe (hence the scalar) by 1/2
    quad = fisher_probe_scalar(logits, token_count_override=4 * 32, **kw)
    assert torch.allclose(quad, base * 0.5, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize('teacher_dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('student_dtype', [torch.bfloat16, torch.float32, torch.float64])
def test_shared_forward_kl_preserves_existing_dtype_broadcast_and_reductions(
        teacher_dtype, student_dtype):
    from prismaquant.build_rtn_cache import kl_divergence
    from prismaquant.kl_fisher import forward_kl_per_token

    # A strided teacher broadcasts over two student batches. The old API
    # preserves teacher precision while deliberately widening students to FP32.
    teacher = torch.log_softmax(torch.linspace(-16, 16, 42, dtype=teacher_dtype)
                               .reshape(3, 14)[:, ::2], dim=-1)
    student = torch.linspace(13, -11, 42, dtype=student_dtype).reshape(2, 3, 7)
    student_lp = torch.log_softmax(student.float(), dim=-1)
    old_tokens = (teacher.exp() * (teacher - student_lp)).sum(dim=-1)
    actual_tokens = forward_kl_per_token(student_lp, teacher)
    torch.testing.assert_close(actual_tokens, old_tokens, rtol=0, atol=0)
    torch.testing.assert_close(kl_divergence(student, teacher), old_tokens.mean(), rtol=0, atol=0)


@pytest.mark.parametrize('scope', ['all', 'last', 'causal'])
@pytest.mark.parametrize('temperature', [0.7, 1.0, 1.7])
def test_global_probe_second_moment_matches_production_kl_hessian(
        monkeypatch, scope, temperature):
    """Exact orthogonal noise moment, rather than a Monte Carlo variance screen.

    Only the random draw is replaced. The real row-indexed Fisher probe,
    temperature/scoping and global normalization remain in the measured path.
    The independent oracle differentiates the existing production KL metric.
    """
    from prismaquant.build_rtn_cache import kl_divergence
    from prismaquant.kl_fisher import select_token_scope, token_count_for_logits

    logits = torch.linspace(-1.7, 2.1, 24, dtype=torch.float64).reshape(2, 3, 4)
    selected = select_token_scope(logits, scope)
    token_count = token_count_for_logits(selected)
    dimensions = selected.numel()
    # Omit the constant column: every retained Rademacher coordinate has zero
    # mean and the complete draw's second moment is exactly the identity.
    hadamard = torch.ones((1, 1))
    while hadamard.shape[0] <= dimensions:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                              torch.cat((hadamard, -hadamard), 1)), 0)
    noise = hadamard[:, 1:dimensions + 1]
    assert torch.equal(noise.T @ noise, torch.eye(dimensions) * len(noise))
    assert torch.equal(noise.sum(0), torch.zeros(dimensions))
    current = {'noise': None, 'row': 0}

    def complete_draw(target, _probability=0.5, *, generator=None):
        row = current['row']
        target.copy_((current['noise'][row] + 1) / 2)
        current['row'] += 1
        return target

    monkeypatch.setattr(torch.Tensor, 'bernoulli_', complete_draw)
    gradients, wrong_normalizer_gradients = [], []
    for draw in noise:
        current.update(noise=draw.reshape(selected.shape), row=0)
        leaf = logits.detach().requires_grad_(True)
        probe = fisher_probe_scalar(leaf, seed=7000, token_scope=scope,
            temperature=temperature, distribution='rademacher',
            token_count_override=token_count, global_row_offset=0)
        gradient, = torch.autograd.grad(probe, leaf)
        assert current['row'] == len(logits)
        gradients.append(gradient.reshape(-1))
        current['row'] = 0
        wrong_probe = fisher_probe_scalar(leaf, seed=7000, token_scope=scope,
            temperature=temperature, distribution='rademacher',
            token_count_override=token_count * 4, global_row_offset=0)
        wrong_gradient, = torch.autograd.grad(wrong_probe, leaf)
        wrong_normalizer_gradients.append(wrong_gradient.reshape(-1))
    matrix = torch.stack(gradients)
    second_moment = matrix.T @ matrix / len(matrix)
    teacher_lp = torch.log_softmax(selected.float() / temperature, dim=-1).detach()

    def measured_loss(student):
        return kl_divergence(select_token_scope(student, scope).float() / temperature, teacher_lp)

    hessian = torch.autograd.functional.hessian(measured_loss, logits).reshape(24, 24)
    assert torch.isfinite(second_moment).all() and torch.isfinite(hessian).all()
    # The production metric/probe execute FP32; the Hessian and moment sums
    # are retained in FP64. This is their rounding bound, not sampling error.
    torch.testing.assert_close(second_moment, hessian, rtol=3e-6, atol=1e-8)
    # A fourfold wrong token normalizer quarters the squared price and must
    # fail this gate. Keep that discriminating control explicit.
    wrong_matrix = torch.stack(wrong_normalizer_gradients)
    wrong_second_moment = wrong_matrix.T @ wrong_matrix / len(wrong_matrix)
    torch.testing.assert_close(wrong_second_moment, second_moment / 4, rtol=3e-6, atol=1e-8)
    assert not torch.allclose(wrong_second_moment, hessian, rtol=3e-6, atol=1e-8)
    print({'scope': scope, 'temperature': temperature, 'tokens': token_count,
           'orthogonal_draws': len(matrix),
           'max_abs_second_moment_minus_kl_hessian': float((second_moment - hessian).abs().max()),
           'wrong_global_normalizer_refused': True})
