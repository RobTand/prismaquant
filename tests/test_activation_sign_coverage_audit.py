"""Whole vectors from the pinned table cover every signed midpoint."""
import torch

from prismaquant.nvfp4_activation_contract import E2M1_MIDPOINTS
from prismaquant.tessera_runtime_contract import packaged_activation_quantizers, require_activation_quantizer_attested


def test_authentic_vector_subsets_cannot_hide_missing_negative_ties():
    digest, table = packaged_activation_quantizers()
    rows = 0
    for platform, images in table.items():
        for image, contracts in images.items():
            for name, row in contracts.items():
                require_activation_quantizer_attested(name, platform=platform,
                    executing_image=row.generated.image, table=table, contract_sha256=digest)
                coverage = []
                for vector in row.vectors:
                    stored = torch.tensor([vector.stored_scale_byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
                    raw_g = vector.global_scale_bits
                    global_scale = torch.tensor([raw_g - (1 << 32) if raw_g >> 31 else raw_g], dtype=torch.int32).view(torch.float32).item()
                    used = stored / global_scale
                    raw = [bits - (1 << 16) if bits >> 15 else bits for bits in vector.input_bits]
                    values = torch.tensor(raw, dtype=torch.int16).view(torch.bfloat16).float().tolist()
                    positive, negative = set(), set()
                    if used:
                        positive = {v / used for v in values if v > 0 and v / used in E2M1_MIDPOINTS}
                        negative = {-v / used for v in values if v < 0 and -v / used in E2M1_MIDPOINTS}
                    coverage.append((positive, negative))
                for subset in range(1 << len(coverage)):
                    positive, negative = set(), set()
                    for index, (pos, neg) in enumerate(coverage):
                        if subset & (1 << index):
                            positive.update(pos)
                            negative.update(neg)
                    if positive | negative == set(E2M1_MIDPOINTS):
                        assert positive == negative == set(E2M1_MIDPOINTS)
                rows += 1
    assert rows > 0
