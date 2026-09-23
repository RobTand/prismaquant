"""Recovered foreign boundaries feed a scratch-backed Stage B quantum exactly.

The recovery authority fixture is qualified separately; this seam exercises
its completed receipt, foreign-reference whitelist, checkpoint reader, local
cotangent owner and real cost/journal arithmetic together.
"""
from pathlib import Path


def test_recovered_capture_to_scratch_quantum_matches_single_run(tmp_path, monkeypatch):
    from prismaquant import joint_cost_stage_a as stage_a
    import test_joint_forward_resume as recovery
    import test_joint_cost_quantum_runtime as runtime
    from tools.compare_joint_layer_gate import compare_layer

    captured = {}
    original = stage_a.run_adjoint_capture_core

    def collect(*args, **kwargs):
        receipt = original(*args, **kwargs)
        if kwargs.get('forward_recovery') is not None:
            captured['receipt'] = receipt
            captured['output_root'] = kwargs['output_root']
        return receipt

    monkeypatch.setattr(stage_a, 'run_adjoint_capture_core', collect)
    recovery.test_recovered_tail_and_reverse_checkpoints_equal_uninterrupted(
        tmp_path / 'recovery', monkeypatch)
    single_root = tmp_path / 'single'
    single = runtime._single_run(single_root, monkeypatch,
                                 checkpoint=single_root / 'checkpoints')
    scratch = tmp_path / 'scratch'; scratch.mkdir()
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_ROOT', str(scratch))
    monkeypatch.setenv('PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES', str(1 << 20))
    assert captured['receipt']['boundary_storage']['forward_recovery']
    for layer in (1, 0):
        payload, record, _ = runtime._run_quantum(
            tmp_path, monkeypatch, single=single, layer=layer,
            receipt=captured['receipt'], output_root=captured['output_root'],
            plan_sha='b' * 64, prepared_sha='c' * 64)
        for name, formats in payload['costs'].items():
            for fmt, actual in formats.items():
                expected = single[0]['costs'][name][fmt]
                assert actual['signed_components_per_probe'] == expected['signed_components_per_probe']
                assert actual['x2_per_probe'] == expected['x2_per_probe']
        verdict = compare_layer(single_root / 'checkpoints',
            Path(record['output_space']['checkpoint_dir']), layer=layer, qname_filter=None)
        assert verdict['verdict'] == 'match', verdict
        assert not list(scratch.iterdir())
