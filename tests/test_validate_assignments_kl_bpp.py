from __future__ import annotations

from prismaquant import format_registry as fr
from prismaquant.validate_assignments_kl import (
    _assignment_bpp_details,
    _kl_repeat_summary,
    _profile_excludes_bpp_name,
)


class _Profile:
    def is_pinned_name(self, qname: str) -> bool:
        return qname == "lm_head"

    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return ("mtp.", "model.visual.")


def _stats(n_params: int = 256) -> dict:
    return {
        "n_params": n_params,
        "in_features": 16,
        "out_features": 16,
    }


def test_assignment_bpp_excludes_pinned_and_auxiliary_entries():
    stats = {
        "model.layers.0.mlp.down_proj": _stats(),
        "mtp.layers.0.mlp.down_proj": _stats(),
        "model.visual.blocks.0.mlp.fc1": _stats(),
        "lm_head": _stats(),
    }
    assignment = {
        "model.layers.0.mlp.down_proj": "NVFP4",
        "mtp.layers.0.mlp.down_proj": "BF16",
        "model.visual.blocks.0.mlp.fc1": "BF16",
        "lm_head": "BF16",
    }
    specs = {name: fr.get_format(name) for name in ("NVFP4", "BF16")}

    details = _assignment_bpp_details(
        stats,
        assignment,
        specs,
        profile=_Profile(),
    )

    expected = (
        8.0
        * (
            fr.get_format("NVFP4").memory_bytes_for_shape((16, 16))
            + 8  # fp32 weight_global_scale + input_global_scale
        )
        / 256.0
    )
    assert details["bpp"] == expected
    assert details["quantizable_entries"] == 1
    assert details["excluded_entries"] == 3


def test_assignment_bpp_excludes_auxiliary_entries_even_when_quantized():
    stats = {
        "model.layers.0.mlp.down_proj": _stats(),
        "model.visual.blocks.0.mlp.fc1": _stats(),
    }
    assignment = {
        "model.layers.0.mlp.down_proj": "NVFP4",
        "model.visual.blocks.0.mlp.fc1": "NVFP4",
    }
    specs = {name: fr.get_format(name) for name in ("NVFP4", "BF16")}

    details = _assignment_bpp_details(
        stats,
        assignment,
        specs,
        profile=_Profile(),
    )

    expected = (
        8.0
        * (
            fr.get_format("NVFP4").memory_bytes_for_shape((16, 16))
            + 8  # fp32 weight_global_scale + input_global_scale
        )
        / 256.0
    )
    assert details["bpp"] == expected
    assert details["quantizable_entries"] == 1
    assert details["excluded_entries"] == 1


def test_bpp_exclusion_is_profile_driven_not_bf16_default():
    profile = _Profile()

    assert not _profile_excludes_bpp_name(
        "model.layers.0.mlp.down_proj", "BF16", profile,
    )
    assert not _profile_excludes_bpp_name(
        "model.layers.0.mlp.down_proj", "NVFP4", profile,
    )
    assert _profile_excludes_bpp_name("lm_head", "BF16", profile)
    assert _profile_excludes_bpp_name(
        "model.visual.blocks.0.mlp.fc1", "NVFP4", profile,
    )


def test_kl_repeat_summary_reports_stderr_and_ucb():
    summary = _kl_repeat_summary([0.10, 0.20, 0.30], ucb_z=2.0)

    assert abs(summary["last_token_kl"] - 0.20) < 1e-12
    assert summary["kl_repeat_count"] == 3
    assert summary["kl_std"] > 0
    assert summary["kl_stderr"] > 0
    assert summary["kl_ucb"] > summary["last_token_kl"]

def test_kl_repeat_summary_emits_kl_mean_and_keeps_the_alias():
    """R28: kl_mean is canonical; last_token_kl stays an alias for one cycle."""
    summary = _kl_repeat_summary([0.10, 0.20, 0.30], ucb_z=2.0)

    assert abs(summary["kl_mean"] - 0.20) < 1e-12
    assert summary["last_token_kl"] == summary["kl_mean"]


def test_kl_repeat_summary_emits_gold_lane_tail_keys():
    """R9: per-sequence tail at zero extra forward cost, gold-lane key names."""
    summary = _kl_repeat_summary(
        [0.20],
        ucb_z=0.0,
        kl_per_sample=[0.10, 0.12, 0.14, 0.90],
        nll_per_sample=[2.0, 2.1, 2.2, 5.0],
    )

    assert summary["kl_per_sample"] == [0.10, 0.12, 0.14, 0.90]
    assert summary["kl_max"] == 0.90
    assert summary["kl_p99"] > summary["kl_p95"] > 0.14
    assert summary["kl_tail_domain"] == "sequence"
    assert abs(summary["nll_mean"] - 2.825) < 1e-12
    assert summary["nll_p99"] > 4.0
    # The repeat mean stays authoritative: pooling per-sequence values would
    # silently reweight repeats of unequal size.
    assert abs(summary["kl_mean"] - 0.20) < 1e-12


def test_kl_repeat_summary_without_per_sequence_data_omits_tail():
    summary = _kl_repeat_summary([0.10, 0.20], ucb_z=0.0)
    for key in ("kl_p95", "kl_p99", "kl_max", "kl_per_sample", "nll_p99"):
        assert key not in summary
    for key in ("kl_max_repeats", "kl_p99_repeats", "nll_p99_repeats"):
        assert key not in summary


def test_kl_repeat_summary_emits_per_repeat_tails_for_the_derived_slack():
    """The tail's between-seed spread is what `--tail-eta auto` derives from."""
    summary = _kl_repeat_summary(
        [0.20, 0.22],
        ucb_z=0.0,
        kl_per_sample_repeats=[[0.10, 0.12, 0.90], [0.11, 0.13, 0.95]],
        nll_per_sample_repeats=[[2.0, 2.1, 5.0], [2.0, 2.2, 5.2]],
    )

    assert summary["kl_max_repeats"] == [0.90, 0.95]
    assert len(summary["kl_p99_repeats"]) == 2
    assert len(summary["nll_p99_repeats"]) == 2
    # The pooled tail is still emitted, from the same values, flattened.
    assert summary["kl_max"] == 0.95
    assert summary["kl_per_sample"] == [0.10, 0.12, 0.90, 0.11, 0.13, 0.95]

    from prismaquant.select_validated_frontier import tail_eta_auto
    eta, source = tail_eta_auto(summary, "kl_max")
    assert source == "derived"
    assert eta > 0.0


def test_kl_repeat_summary_single_repeat_tail_list_is_one_element():
    """One repeat => a 1-element list, so the selector can say 'single seed'."""
    summary = _kl_repeat_summary(
        [0.20], ucb_z=0.0, kl_per_sample_repeats=[[0.10, 0.12, 0.90]])
    assert summary["kl_max_repeats"] == [0.90]

    from prismaquant.select_validated_frontier import tail_eta_auto
    assert tail_eta_auto(summary, "kl_max") == (0.0, "single_repeat")
