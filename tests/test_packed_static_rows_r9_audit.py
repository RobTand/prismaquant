"""Static CPU QDQ row grouping must preserve ties and underflow boundaries."""
import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_activation_contract as contract_owner
from prismaquant import perturbed_x_cache as owner


@pytest.mark.parametrize("name", ["TESSERA_E2M1_K1_R768", "TESSERA_E2M1_K2_R896"])
@pytest.mark.parametrize("width", [32, 64, 96])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_static_model_qdq_rows_preserve_numeric_boundaries(monkeypatch, name, width, dtype):
    contract_owner.bind_served_quantizer_identity(
        identity=contract_owner.ServedQuantizerIdentity(
            backend=contract_owner.SERVED_QUANTIZER_BACKEND_MODEL), require=False)
    monkeypatch.setenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", "0")
    spec = fr.get_format(name)
    assert spec.static_activation_contract.measured_as_served
    assert spec.static_activation_contract.input_global_scale_from_max_abs(6.0) == 1.0
    block = torch.tensor([6, -.25, .25, -.75, .75, -1.25, 1.25, -1.75,
                          1.75, -2.5, 2.5, -3.5, 3.5, -5, 5, -6], dtype=dtype)
    rows = block.repeat(width // 16).repeat(13, 1)
    rows.mul_(torch.tensor([2.0 ** exponent for exponent in range(-6, 7)], dtype=dtype)[:, None])
    rows[0].zero_()
    tie = torch.tensor(6 * 2.0 ** -10, dtype=dtype)
    rows[1].fill_(tie.item())
    rows[2].fill_(torch.nextafter(tie, torch.tensor(float("inf"), dtype=dtype)).item())
    rows[-1].mul_(1024)
    scales = {"projection": 6.0}
    actual = owner._activation_qdq(rows, spec, scales, "projection")
    reference = torch.cat([owner._activation_qdq(part, spec, scales, "projection")
                           for part in rows.split([3, 1, 9])])
    assert owner._packed_activation_qdq_is_row_local(spec)
    assert torch.equal(actual, reference)
    assert torch.count_nonzero(actual[:2]) == 0
    assert torch.count_nonzero(actual[2]) == width
    assert actual[-1].abs().max().item() == 2688
