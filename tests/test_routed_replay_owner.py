"""Routed discovery and replay operate without the measurement application."""
from pathlib import Path
import subprocess
import sys
import textwrap


def test_routed_discovery_and_replay_do_not_load_measurement():
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
        from pathlib import Path
        import sys

        class RejectMeasurement(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'prismaquant.measure_quant_cost':
                    raise ImportError('measurement application is unavailable')

        sys.meta_path.insert(0, RejectMeasurement())
        sys.path.insert(0, str(Path.cwd() / 'tests'))
        import torch
        from prismaquant import routed_experts
        from test_tessera_campaign_packed import _RoutedModel, _routed_tokens

        torch.set_num_threads(1)
        model = _RoutedModel().eval()
        members = routed_experts.profile_declared_packed_expert_projections(model)
        assert len(members) == 6
        block = model.model.layers[2].feed_forward
        values = _routed_tokens()[0].reshape(-1, 4)
        indices, weights = block.route_tokens_to_experts(block.gate(values))
        with torch.no_grad():
            expected = block(values)
            actual = routed_experts.packed_experts_forward_with_weights(
                block.experts, values, indices, weights,
                block.experts.gate_up_proj, block.experts.down_proj)
            rows = routed_experts.derive_per_expert_activations(
                block.experts, values, block)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert rows['row_counts'] == [2, 2]
        assert 'prismaquant.measure_quant_cost' not in sys.modules
    """)
    result = subprocess.run([sys.executable, '-c', script], cwd=root,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
