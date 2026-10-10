"""CPU causal checks for retaining a local control before a later gate fails."""
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.alloc_lead_asd import l05_source_replay as owner


class LocalPublication(unittest.TestCase):
    def test_later_reference_refusal_retains_completed_local_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / 'reference.pt'
            row = dict(unit='test', probe=0, coordinates=torch.tensor([[0, 0, -1]]),
                       signed_rows=torch.tensor([1.0], dtype=torch.float64))
            expected = {**row, 'signed_rows': torch.tensor([2.0], dtype=torch.float64)}
            torch.save(dict(rows=[expected]), reference)
            binding = dict(scope='synthetic only', layer=0, layer_keys=['test'],
                           reference=owner.record(reference))
            result = dict(rows=[row], selected_units=['test'],
                          max_abs_fixed_g_operator_difference=0.0,
                          max_relative_to_operator_rms=0.0, coordinate_coverage_exact=True,
                          projection_backend={'name': 'test'}, activation_format='test',
                          activation_callable='test')
            transport = Mock(copy_fences=0, transient_staging_peak=0)
            transport.guard.snapshot.return_value = {'scope': 'test'}
            runner = Mock()
            progress = Mock()
            args = SimpleNamespace(prepare_synthetic=None, binding=root / 'binding.json',
                                   binding_sha256='test', data_manifest_sha256='test',
                                   projection_contract=None, device='cpu', output=root / 'output')
            buffer = SimpleNamespace(path=reference, close=Mock())
            with ExitStack() as stack:
                for name, value in {
                    'activate_staged_tier_policy': Mock(), 'bind_residency_manifest': Mock(),
                    'read_staged_whole_file': Mock(return_value=json.dumps(binding).encode()),
                    'ReplayDecoders': Mock(return_value=transport),
                    'make_context': Mock(return_value=(runner, {'scope': 'test'})),
                    'verify_installed_source_tensors': Mock(return_value={'scope': 'test', 'exact': True}),
                    'profiled': lambda *args: nullcontext(), 'replay': Mock(return_value=result),
                    'read_staged_sealed_file': Mock(return_value=buffer),
                    'residency_report': Mock(return_value={'scope': 'test'}), 'progress': progress,
                }.items():
                    stack.enter_context(patch.object(owner, name, value))
                with self.assertRaises(AssertionError):
                    owner.execute(args)
            self.assertTrue((args.output / 'rows.pt').is_file())
            saved = torch.load(args.output / 'rows.pt', weights_only=True)
            torch.testing.assert_close(saved['rows'][0]['signed_rows'], row['signed_rows'])
            local = json.loads((args.output / 'local-control.json').read_text())
            self.assertTrue(local['same_primal_operator_gate_passed'])
            self.assertFalse(local['resident_reference_gate_passed'])
            self.assertEqual(local['rows']['sha256'], owner.record(args.output / 'rows.pt')['sha256'])
            self.assertFalse((args.output / 'result.json').exists())
            self.assertEqual(progress.call_args_list[-1].args, (2, 'control'))
            transport.close.assert_called_once_with()
            buffer.close.assert_called_once_with()

    def test_installed_source_values_cover_profile_projection_and_buffer(self):
        source = {'source.weight': torch.tensor([[1.0, 2.0]]),
                  'source.buffer': torch.tensor([0.125], dtype=torch.float64)}
        buffer = source['source.buffer'].float()
        projected = source['source.weight'].clone()
        model = SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: [('body.buffer', buffer)])
        runner = SimpleNamespace(model=model, profile=None, context=SimpleNamespace(
            weight_ckpt={'body.expert.weight': 'source.weight', 'body.buffer': 'source.buffer'},
            weight_shard={'body.expert.weight': '/held/source', 'body.buffer': '/held/source'}))
        transport = SimpleNamespace(dtypes={}, dtype=torch.float32, guard=Mock(),
                                    binding={'layer_keys': ['source.weight', 'source.buffer']})
        reader = SimpleNamespace(get_tensor=source.__getitem__)
        with patch.object(owner._SealedDecoders, 'safe_open', return_value=nullcontext(reader)), \
             patch.object(owner, 'profile_declared_packed_expert_projections',
                          return_value=[SimpleNamespace(qname='body.expert', weight=projected)]):
            self.assertEqual(owner.verify_installed_source_tensors(runner, transport)['selected_keys'], 2)
            projected[0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, 'installed source value differs: body.expert.weight'):
                owner.verify_installed_source_tensors(runner, transport)

    def test_installed_mutation_refuses_entrypoint_before_local_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / 'reference.pt'
            row = dict(unit='test', probe=0, coordinates=torch.tensor([[0, 0, -1]]),
                       signed_rows=torch.tensor([1.0], dtype=torch.float64))
            result = dict(rows=[row], selected_units=['test'],
                          max_abs_fixed_g_operator_difference=0.0,
                          max_relative_to_operator_rms=0.0, coordinate_coverage_exact=True,
                          projection_backend={'name': 'test'}, activation_format='test', activation_callable='test')
            torch.save(result, reference)
            binding = dict(scope='synthetic only', layer=0, layer_keys=['source.weight'],
                           reference=owner.record(reference))
            transport = Mock(copy_fences=0, transient_staging_peak=0, binding=binding,
                             dtypes={}, dtype=torch.float32)
            transport.guard.snapshot.return_value = {'scope': 'test'}
            model = SimpleNamespace(named_parameters=lambda: [('body.weight', torch.tensor([[2.0]]))],
                                    named_buffers=lambda: [])
            runner = Mock(model=model, profile=None, context=SimpleNamespace(
                weight_ckpt={'body.weight': 'source.weight'}, weight_shard={'body.weight': '/held/source'},
                layer_cache=Mock()))
            args = SimpleNamespace(prepare_synthetic=None, binding=root / 'binding.json',
                                   binding_sha256='test', data_manifest_sha256='test',
                                   projection_contract=None, device='cpu', output=root / 'output')
            replay = Mock(return_value=result)
            with ExitStack() as stack:
                for name, value in {
                    'activate_staged_tier_policy': Mock(), 'bind_residency_manifest': Mock(),
                    'read_staged_whole_file': Mock(return_value=json.dumps(binding).encode()),
                    'ReplayDecoders': Mock(return_value=transport),
                    'make_context': Mock(return_value=(runner, {'scope': 'test'})),
                    'profile_declared_packed_expert_projections': Mock(return_value=[]),
                    'profiled': lambda *args: nullcontext(), 'replay': replay,
                    'read_staged_sealed_file': Mock(return_value=SimpleNamespace(path=reference, close=Mock())),
                    'residency_report': Mock(return_value={'scope': 'test'}), 'progress': Mock(),
                }.items():
                    stack.enter_context(patch.object(owner, name, value))
                stack.enter_context(patch.object(owner._SealedDecoders, 'safe_open',
                    return_value=nullcontext(SimpleNamespace(get_tensor=lambda name: torch.tensor([[1.0]])))))
                with self.assertRaisesRegex(RuntimeError, 'installed source value differs: body.weight'):
                    owner.execute(args)
            replay.assert_not_called()
            self.assertFalse((args.output / 'rows.pt').exists())
            transport.close.assert_called_once_with()

    def test_reference_context_preserves_cross_primal_difference_and_refuses_nan(self):
        row = dict(unit='test', probe=0, coordinates=torch.tensor([[0, 0, -1]]),
                   signed_rows=torch.tensor([1.0], dtype=torch.float64))
        foreign = {**row, 'signed_rows': torch.tensor([2.0], dtype=torch.float64)}
        context = owner.reference_comparison({'rows': [row]}, {'rows': [foreign]})
        self.assertFalse(context['observations'][0]['unchanged_cpu_tolerance_passed'])
        self.assertEqual(context['observations'][0]['max_abs_difference'], 1.0)
        foreign['signed_rows'][0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'nonfinite resident/staged contextual signed rows'):
            owner.reference_comparison({'rows': [row]}, {'rows': [foreign]})

    def test_allocator_envelope_restores_prior_fraction_after_failure(self):
        binding = {'allocator_limits': {'torch_allocator_bytes': 40 << 30}}
        with patch.object(owner, 'allocator_device', return_value=0), \
             patch.object(owner, 'enforce_device_envelope', return_value={'enforced': True}) as enforce, \
             patch.object(torch.cuda, 'get_per_process_memory_fraction', return_value=0.75), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction') as restore, \
             patch.object(torch.cuda, 'synchronize') as fence:
            with self.assertRaisesRegex(RuntimeError, 'later failure'):
                with owner.bounded_cuda_allocator(binding, torch.device('cuda')) as record:
                    self.assertTrue(record['enforced'])
                    raise RuntimeError('later failure')
            enforce.assert_called_once_with(torch.device('cuda'), 40 << 30, where='L05 local control')
            fence.assert_called_once_with(torch.device('cuda'))
            restore.assert_called_once_with(0.75, 0)

    def test_original_scope_without_opt_in_refuses_before_source_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = {'scope': 'original L05 local mechanism control'}
            args = SimpleNamespace(prepare_synthetic=None, binding=root / 'binding.json',
                                   binding_sha256='test', data_manifest_sha256='test',
                                   projection_contract=None, device='cpu', output=root / 'output',
                                   original_local_control=False)
            with patch.object(owner, 'activate_staged_tier_policy'), \
                 patch.object(owner, 'bind_residency_manifest'), \
                 patch.object(owner, 'read_staged_whole_file', return_value=json.dumps(binding).encode()), \
                 patch.object(owner, 'ReplayDecoders') as transport, \
                 patch.object(owner, 'read_staged_sealed_file') as source:
                with self.assertRaisesRegex(RuntimeError, 'exact accepted contract and explicit opt-in'):
                    owner.execute(args)
            transport.assert_not_called()
            source.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_historical_activation_policy_refuses_foreign_callable_or_scale_mode(self):
        # Exact descriptor recovered from sealed historical action986cb012.
        expected = dict(act_bits=8, act_dtype_name='fp8_e4m3', act_group_size=0,
            activation_max_abs=10.5, clip_enabled=False, input_global_scale=None,
            quantizer='prismaquant.format_registry._make_plain_fp8_activation_vllm_rtn.<locals>.f',
            quantizes_input=True, schema='prismaquant.joint_aura.activation.v1',
            served_scales_enabled=False, static_contract=None)
        contract = dict(format='TESSERA_E4M3_K1_R1024', activation_identities=[expected])
        with patch.dict(owner.os.environ, {'PRISMAQUANT_PROD_ACT_SCALES': '0',
                                          'PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES': '0'}):
            self.assertEqual(owner.qualify_historical_activation_policy(contract), expected)
            for key, value in (('quantizer', 'foreign.callable'), ('clip_enabled', True)):
                foreign = {**expected, key: value}
                with self.assertRaisesRegex(RuntimeError, 'historical activation callable/scale/clip policy differs'):
                    owner.qualify_historical_activation_policy({**contract, 'activation_identities': [foreign]})

    def test_split_concat_sources_match_existing_owner_views_and_refuse_mutation(self):
        prefix = 'model.language_model.layers.5.self_attn.'
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(6)])
        attention = torch.nn.Module()
        attention.conv1d = torch.nn.Conv1d(6, 6, 1, groups=6, bias=False)
        attention.conv1d.weight.data.copy_(torch.arange(1, 7, dtype=torch.float32).reshape(6, 1, 1))
        model.model.language_model.layers[5].self_attn = attention
        source = {prefix + 'q_conv1d.weight': torch.tensor([1.0]).reshape(1, 1, 1),
                  prefix + 'k_conv1d.weight': torch.tensor([2.0, 3.0]).reshape(2, 1, 1),
                  prefix + 'v_conv1d.weight': torch.tensor([4.0, 5.0, 6.0]).reshape(3, 1, 1)}
        profile = owner.Glm5NextProfile()
        keys = {name: name for name in source}
        with patch('prismaquant.model_profiles.profile_from_model', return_value=profile):
            merger = owner._build_concat_merger(model, keys)
        self.assertIsNotNone(merger)
        runner = SimpleNamespace(model=model, profile=profile, context=SimpleNamespace(
            weight_ckpt=keys, weight_shard={name: '/held/source' for name in source}, concat_merger=merger))
        transport = SimpleNamespace(dtypes={}, dtype=torch.float32, guard=Mock(),
                                    binding={'layer_keys': list(source)})
        reader = SimpleNamespace(get_tensor=source.__getitem__, get_slice=lambda name:
                                 SimpleNamespace(get_shape=lambda: list(source[name].shape)))
        with patch.object(owner._SealedDecoders, 'safe_open', side_effect=lambda *args, **kwargs: nullcontext(reader)):
            result = owner.verify_installed_source_tensors(runner, transport)
            self.assertEqual(result['selected_keys'], 3)
            self.assertTrue(result['exact'])
            source[prefix + 'k_conv1d.weight'][0, 0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, 'installed source value differs: .*k_conv1d.weight'):
                owner.verify_installed_source_tensors(runner, transport)
            source[prefix + 'k_conv1d.weight'] = torch.ones((2, 1, 2))
            with self.assertRaisesRegex(RuntimeError, 'concat source .*shape'):
                owner.verify_installed_source_tensors(runner, transport)


if __name__ == '__main__':
    unittest.main()
