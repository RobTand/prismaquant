"""Inject nonfinite observations into the actual profile crosscheck on CPU."""
import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.alloc_lead_asd import a_side_diag as diagnostic


class _ProfileFixture:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def key_averages(self):
        return self

    def table(self, **kwargs):
        return 'CPU regression fixture; profiler timings not measured'

    def export_chrome_trace(self, path):
        Path(path).write_text('{"traceEvents": []}\n')


class FiniteCrosscheck(unittest.TestCase):
    def control(self, fault=None):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=str(Path(directory) / 'control'), deterministic_backward=False)
            model = SimpleNamespace(config=SimpleNamespace(vocab_size=3))
            reference = torch.ones(3, 1, 2, 1, dtype=torch.float64)
            if fault == 'reference_nan':
                reference[0, 0, 0, 0] = float('nan')
            if fault == 'reference_rms_overflow':
                reference.fill_(1e200)

            def price(*args, **kwargs):
                components = args[10]
                components.copy_(reference)
                if fault == 'candidate_nan':
                    components[0, 0, 0, 0] = float('nan')

            def measure(*args, **kwargs):
                return dict(kl=torch.ones(1), q=torch.ones(1), s_real=torch.ones(2, 1))

            def arms(*args, **kwargs):
                result = args[11]['A_all']
                for name in ('kl', 'q', 's_real'):
                    result[name].fill_(1)
                if fault in ('kl_nan', 'q_nan', 'probe_nan'):
                    result[dict(kl_nan='kl', q_nan='q', probe_nan='s_real')[fault]].fill_(float('nan'))

            with patch.object(diagnostic, 'DEVICE', 'cpu'), \
                    patch('torch.profiler.profile', return_value=_ProfileFixture()), \
                    patch.object(diagnostic, 'price_v1', return_value=reference), \
                    patch.object(diagnostic, 'price_sequence', side_effect=price), \
                    patch.object(diagnostic, 'measure_v1', side_effect=measure), \
                    patch.object(diagnostic, 'arms_sequence', side_effect=arms):
                return diagnostic.profile_and_crosscheck(args, model, {'unit': torch.nn.Linear(2, 2)},
                    ['unit'], torch.tensor([[1, 2]]), [7000, 7001],
                    {name: None for name in diagnostic.SPECS}, 2, 'all', 1.0)

    def test_finite_matched_observations_pass(self):
        self.assertEqual(self.control()['arm_crosscheck']['kl_rel_diff'], 0)

    def test_nonfinite_observations_cannot_certify_crosscheck(self):
        for fault in ('reference_nan', 'candidate_nan', 'reference_rms_overflow',
                      'kl_nan', 'q_nan', 'probe_nan'):
            with self.subTest(fault=fault):
                with self.assertRaisesRegex(SystemExit, 'nonfinite|invalid reference'):
                    self.control(fault)


if __name__ == '__main__':
    unittest.main()
