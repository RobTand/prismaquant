import torch

from prismaquant.joint_aura import make_joint_probe_identity


def test_probe_identity_binds_exact_calibration_bytes_and_explicit_measurement_contract():
    calibration = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    probe = make_joint_probe_identity(
        calibration_ids=calibration, source_model={"model": "fixture"},
        n_probes=4, seed_base=237, token_scope="last", temperature=0.75,
        producer_source_sha256="a" * 64,
        source_execution={"attention": "eager"}, arithmetic={"projection": "fp32"})
    assert probe == {
        "schema": "prismaquant.joint_aura.probes.v2",
        "source_model": {"model": "fixture"},
        "calibration_sha256": "f190072c5052f4f440d4a607c25f5bced487c420806c9aab4ca5b0653e72da61",
        "calibration_shape": [2, 3], "calibration_dtype": "torch.int64",
        "n_probes": 4, "seed_base": 237,
        "token_scope": "last", "temperature": 0.75,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "producer_source_sha256": "a" * 64,
        "source_execution": {"attention": "eager"}, "arithmetic": {"projection": "fp32"},
    }


def test_probe_identity_hashes_logical_token_order_of_a_noncontiguous_draw():
    tokens = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int64).t()
    assert not tokens.is_contiguous()
    probe = make_joint_probe_identity(
        calibration_ids=tokens, source_model={}, n_probes=4, seed_base=237,
        token_scope="all", temperature=1.0, producer_source_sha256="a" * 64,
        source_execution={}, arithmetic={})
    assert probe["calibration_sha256"] == "f190072c5052f4f440d4a607c25f5bced487c420806c9aab4ca5b0653e72da61"
    assert probe["calibration_shape"] == [2, 3]
