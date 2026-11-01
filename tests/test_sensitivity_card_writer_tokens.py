import pickle

from prismaquant.format_cost_protocol import weight_dloss_scalar
from prismaquant.sensitivity_card_build import card_from_probe


def test_card_retains_body_and_visual_writer_denominators(tmp_path):
    path = tmp_path / "probe.pkl"
    path.write_bytes(pickle.dumps({
        "meta": {"nsamples": 32, "seqlen": 1024},
        "stats": {
            "model.layers.0.mlp.up_proj": {
                "out_features": 2, "in_features": 3,
                "h_trace_raw": 98304.0, "h_trace": 3.0,
                "h_trace_norm_tokens": 32768, "n_tokens_seen": 32768,
            },
            "visual.linear": {
                "out_features": 2, "in_features": 3,
                "h_trace_raw": 5120.0, "h_trace": 5.0,
                "h_trace_norm_tokens": 1024, "n_tokens_seen": 1024,
            },
        },
    }))
    units = {unit.topology.name: unit for unit in card_from_probe(str(path)).units()}
    body, visual = units["model.layers.0.mlp.up_proj"], units["visual.linear"]
    assert body.h_trace == 3.0
    assert visual.n_tokens == 1024
    assert visual.h_trace == 5.0
    assert weight_dloss_scalar(visual, 0.25) == 0.625
