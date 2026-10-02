"""Causal CPU regressions for unsupported cached-pricing reuse and missing inputs."""
import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.alloc_lead_asd import a_side_diag as diagnostic


class ReuseGuards(unittest.TestCase):
    def test_foreign_pricing_cannot_publish_decision_numbers(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        model.config = SimpleNamespace(vocab_size=4)
        ids = torch.tensor([[1, 2]], dtype=torch.int64)
        expected = dict(model='current-checkpoint', dtype='float32',
                        n_global=2, impl=diagnostic.IMPL)
        for field, foreign in (('model', 'foreign-checkpoint'), ('dtype', 'bfloat16'),
                               ('n_global', 512), ('impl', 'foreign-implementation')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                saved = dict(units=['model.layers.0'], seeds=[7000, 7001],
                    ids_sha256=diagnostic.hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    specs=list(diagnostic.SPECS), comps=torch.ones(3, 1, 2, 1),
                    n_global=expected['n_global'], impl=expected['impl'],
                    args=dict(model=expected['model'], dtype=expected['dtype'], deterministic_backward=False))
                if field in ('model', 'dtype'):
                    saved['args'][field] = foreign
                else:
                    saved[field] = foreign
                torch.save(saved, root / 'foreign.pt')
                args = argparse.Namespace(model=expected['model'], dtype=expected['dtype'],
                    text='fixture', seed_base=7000, n_probes=2, n_single=0, layer_arms=False,
                    profile=False, pricing_from=str(root / 'foreign.pt'), output=str(root / 'run'),
                    dz_dtype='float32', deterministic_backward=False)
                with patch.object(diagnostic, 'DEVICE', 'cpu'), \
                        patch.object(diagnostic, 'build_plan', return_value=[]), \
                        patch.object(diagnostic, 'arms_sequence', return_value=None):
                    with self.assertRaisesRegex(SystemExit, 'cached pricing reuse is unsupported'):
                        diagnostic.run(args, model, ids=ids)
                self.assertFalse((root / 'run.json').exists())

    def test_missing_token_artifact_names_the_requested_input(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'immutable-tokens.safetensors'
            with self.assertRaisesRegex(FileNotFoundError, 'calibration token artifact does not exist') as refusal:
                diagnostic.load_tokens(str(missing), 'fit', 1)
            self.assertIn(str(missing), str(refusal.exception))
            self.assertNotIn('prep_inputs.py', str(refusal.exception))


if __name__ == '__main__':
    unittest.main()
