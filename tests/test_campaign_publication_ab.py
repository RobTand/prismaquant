"""The comparison must preserve compatible work and report truthful spans."""
import pytest

import json
import time

from experiments.campaign_publication_ab import (OVERLAP_BUDGET, SCHEMA_V1, SCHEMA_V2, SCHEMA_V3,
    MemorySampler, arm_order, command_batch, plan_order, preferred_batches, with_batch, PhaseRecorder)


def test_shape_permutation_preserves_whole_batches_and_internal_order():
    down = [('a.down_proj', 'family', 832), ('b.down_proj', 'family', 832)]
    gate = [('a.gate_proj', 'family', 832), ('a.up_proj', 'family', 832)]
    later = [('b.gate_proj', 'family', 1088)]
    original = [down, gate, later]
    reordered = preferred_batches(original, 'gate_up')
    assert reordered == [gate, later, down]
    assert reordered[0] is gate and reordered[1] is later and reordered[2] is down
    assert original == [down, gate, later]


@pytest.mark.parametrize('names', [[], ['a.down_proj'], ['a.not_an_expert'], ['a.gate_proj', 'a.down_proj']])
def test_missing_or_mixed_projection_refuses(names):
    batches = [[(name, 'family', 832) for name in names]] if names else []
    with pytest.raises(ValueError):
        preferred_batches(batches, 'gate_up')


def test_profile_keeps_original_error_and_records_failed_span():
    recorder = PhaseRecorder()
    error = ValueError('original publication failure')
    def fail():
        raise error
    with pytest.raises(ValueError) as caught:
        recorder.wrap(fail, 'publication')()
    assert caught.value is error
    assert len(recorder.records) == 1
    assert recorder.records[0]['phase'] == 'publication'
    assert recorder.records[0]['seconds'] >= 0
    assert recorder.records[0]['thread_cpu_seconds'] >= 0


def test_a_repeat_order_replaces_the_default_matrix_instead_of_being_shadowed():
    B, I = OVERLAP_BUDGET, 268435456
    assert arm_order(I) == [(0, 0), (B, 0), (B, I), (B, I), (B, 0)]
    assert arm_order(0) == [(0, 0), (B, 0), (B, 0), (0, 0)]
    assert arm_order(I, [(B, I)]) == [(B, I)]
    assert arm_order(I, [[B, I], [B, 0]]) == [(B, I), (B, 0)]
    with pytest.raises(ValueError):
        arm_order(I, [(B,)])
    with pytest.raises(ValueError):
        arm_order(I, [(B, -1)])


def test_batch_width_arms_keep_the_recipe_width_unless_named():
    B, I = OVERLAP_BUDGET, 268435456
    command = ['--foo', '--anchor-batch-size', '8', '--bar']
    assert command_batch(command) == 8
    assert with_batch([(0, 0), (B, I), (B, I, 16), [B, I, 32]], 8) == [(0, 0, 8), (B, I, 8), (B, I, 16), (B, I, 32)]
    assert arm_order(I, [(B, I, 8), (B, I, 16)]) == [(B, I, 8), (B, I, 16)]
    for bad in ([(B, I, 0)], [(B, I, 8, 1)], [(B,)]):
        with pytest.raises(ValueError):
            arm_order(I, bad)
    with pytest.raises(ValueError):
        with_batch([(B, I, 0)], 8)
    assert plan_order(dict(schema=SCHEMA_V1, command=command, order=[0, B])) == [(0, 0, 8), (B, 0, 8)]
    assert plan_order(dict(schema=SCHEMA_V2, command=command, order=[[0, 0], [B, I]])) == [(0, 0, 8), (B, I, 8)]
    assert plan_order(dict(schema=SCHEMA_V3, command=command, order=[[B, I, 8], [B, I, 32]])) == [(B, I, 8), (B, I, 32)]
    with pytest.raises(ValueError):
        plan_order(dict(schema=SCHEMA_V3, command=command, order=[[B, I]]))
    with pytest.raises(ValueError):
        plan_order(dict(schema=SCHEMA_V2, command=command, order=[[B, I, 8]]))


def test_memory_sampler_records_a_continuous_series_with_peaks(tmp_path):
    sampler = MemorySampler(tmp_path/'memory-samples.json', interval=0.05)
    sampler.start()
    ballast = bytearray(64*1024**2)
    ballast[::4096] = b'x'*len(ballast[::4096])
    time.sleep(0.3)
    sampler.close()
    del ballast
    record = json.loads((tmp_path/'memory-samples.json').read_text())
    assert record['samples'] >= 4 and len(record['series']) == record['samples']
    assert record['peak']['vmhwm_bytes'] >= record['peak']['vmrss_bytes'] > 64*1024**2
    assert record['peak']['mem_available_bytes'] == min(s['mem_available_bytes'] for s in record['series'])
    assert all(b['unix'] >= a['unix'] for a, b in zip(record['series'], record['series'][1:]))


def test_with_headroom_replaces_the_floor_and_refuses_a_recipe_without_one():
    from experiments.campaign_publication_ab import with_headroom
    recipe = ['python3', '-m', 'prismaquant.tessera_campaign', '--streaming',
              '--streaming-cache-headroom-gb', '24', '--anchor-batch-size', '8']
    assert with_headroom(recipe, 12)[recipe.index('--streaming-cache-headroom-gb')+1] == '12'
    assert with_headroom(recipe, 12.5)[recipe.index('--streaming-cache-headroom-gb')+1] == '12.5'
    assert with_headroom(recipe, 12)[:4] == recipe[:4] and recipe[5] == '24'
    with pytest.raises(ValueError):
        with_headroom(['--anchor-batch-size', '8'], 12)
    with pytest.raises(ValueError):
        with_headroom(recipe, 0)
