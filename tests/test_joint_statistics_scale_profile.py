"""Bounded shape planning and independent sampled-oracle checks."""
import pytest
import torch

from experiments.joint_statistics_scale_profile import (
    SAMPLES, check_samples, derive_plan, fp64_samples, sample_coordinates,
)
from test_joint_allocator_turnover_profile import ledger


def census():
    shapes = {}
    for expert in range(288):
        prefix = f'model.language_model.layers.3.mlp.experts.{expert}'
        shapes[prefix + '.down_proj'] = [4096, 2048]
        for projection in ('gate_proj', 'up_proj'):
            shapes[prefix + '.' + projection] = [2048, 4096]
    for projection, shape in [('down_proj', [4096, 2048]), ('gate_proj', [2048, 4096]), ('up_proj', [2048, 4096])]:
        shapes['model.language_model.layers.3.mlp.shared_experts.' + projection] = shape
    return dict(unit_shapes=shapes)


def scale_ledger():
    value = ledger()
    value['terms_bytes'].update(candidate_delta_cap=256 * 1024**2, replay_fork_cap=256 * 1024**2)
    return value


def test_real_shape_counts_plan_two_complete_near32gib_windows_without_native_allocation():
    shapes, specs, plan, geometry = derive_plan(census(), scale_ledger())
    assert len(shapes) == len(specs) == 867
    assert list(map(len, plan.windows)) == [341, 341, 185]
    assert plan.window_statistics_bytes[:2] == (34326183936,) * 2
    assert geometry['shape_source_bytes'] == 14545846272
    assert geometry['source_remainder_bytes'] == 18769917616
    assert geometry['candidate_storage_bytes'] == 32 * 1024**2
    assert all(len(target.groups) == 2 for target in plan.targets)


def test_partial_or_wrong_census_cannot_silently_shrink_the_native_gate():
    value = census()
    value['unit_shapes'].pop(next(iter(value['unit_shapes'])))
    with pytest.raises(ValueError, match='exact layer-three'):
        derive_plan(value, scale_ledger())


def test_sample_coordinates_cover_endpoints_in_both_real_orientations():
    for shape in [(2048, 4096), (4096, 2048)]:
        rows, cols = sample_coordinates(shape)
        assert len(set(zip(rows, cols))) == SAMPLES
        assert (rows[0], cols[0]) == (0, 0)
        assert (rows[1], cols[1]) == (shape[0] - 1, shape[1] - 1)
        assert min(rows) >= 0 and max(rows) < shape[0]
        assert min(cols) >= 0 and max(cols) < shape[1]


def test_fp64_oracle_reduces_only_selected_pairs_and_preserves_exact_bf16_operands():
    x = torch.arange(8 * SAMPLES, dtype=torch.float32).reshape(8, SAMPLES).to(torch.bfloat16)
    g = torch.full_like(x, 0.5)
    expected = [sum(float(x[row, col]) * 0.5 for row in range(8)) for col in range(SAMPLES)]
    assert fp64_samples(x, g) == expected
    assert check_samples(expected, expected)['max_absolute'] == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 2.])
def test_oracle_refuses_nonfinite_or_wrong_matrix_samples(bad):
    with pytest.raises(RuntimeError, match='FP64 sample differs'):
        check_samples([1.] * SAMPLES, [bad] * SAMPLES)


def test_oracle_refuses_zero_or_missing_operator_coverage():
    with pytest.raises(RuntimeError, match='entirely zero'):
        check_samples([0.] * SAMPLES, [0.] * SAMPLES)
    with pytest.raises(RuntimeError, match='coverage'):
        check_samples([1.] * SAMPLES, [1.])
