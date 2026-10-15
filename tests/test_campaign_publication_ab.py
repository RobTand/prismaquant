"""The comparison must preserve compatible work and report truthful spans."""
import pytest

import json

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


def test_memory_sampler_records_a_continuous_series_with_peaks(monkeypatch, tmp_path):
    from threading import Event, Thread

    live = MemorySampler.read()
    assert live['vmhwm_bytes'] >= live['vmrss_bytes'] > 0
    assert live['mem_available_bytes'] >= 0
    readings = [
        dict(unix=1, vmrss_bytes=10, vmhwm_bytes=12, mem_available_bytes=90),
        dict(unix=2, vmrss_bytes=20, vmhwm_bytes=25, mem_available_bytes=80),
        dict(unix=3, vmrss_bytes=15, vmhwm_bytes=25, mem_available_bytes=70),
        dict(unix=4, vmrss_bytes=12, vmhwm_bytes=25, mem_available_bytes=85),
    ]
    source = iter(readings)
    last_started, release, stop_observed = Event(), Event(), Event()
    sampler = MemorySampler(tmp_path/'memory-samples.json', interval=0)

    def controlled_read():
        sample = next(source)
        if sample['unix'] == 3:
            last_started.set()
            if not release.wait(30):
                raise RuntimeError('sampler close did not release the pending read')
        return sample

    def release_after_close():
        if sampler.stopped.wait(30):
            stop_observed.set()
            release.set()

    monkeypatch.setattr(MemorySampler, 'read', staticmethod(controlled_read))
    releaser = Thread(target=release_after_close, daemon=True)
    sampler.start()
    try:
        assert last_started.wait(30)
        releaser.start()
        sampler.close()
        assert stop_observed.is_set()
    finally:
        sampler.stopped.set()
        release.set()
        sampler.join(30)
        if releaser.ident is not None:
            releaser.join(30)
    assert not sampler.is_alive() and not releaser.is_alive()
    record = json.loads((tmp_path/'memory-samples.json').read_text())
    assert record['samples'] == 4
    assert record['series'] == readings
    assert record['peak'] == dict(vmrss_bytes=20, vmhwm_bytes=25, mem_available_bytes=70)


def test_memory_sampler_converts_proc_kibibytes_to_bytes(monkeypatch):
    from types import SimpleNamespace
    import torch
    from experiments import campaign_publication_ab as publication

    proc = {
        '/proc/self/status': 'Name:\tfixture\nVmRSS:\t64 kB\nVmHWM:\t80 kB\n',
        '/proc/meminfo': 'MemTotal:\t256 kB\nMemAvailable:\t128 kB\n',
    }
    monkeypatch.setattr(publication, 'Path', lambda path: SimpleNamespace(
        read_text=lambda: proc[str(path)]))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    actual = MemorySampler.read()
    assert actual['vmrss_bytes'] == 65536
    assert actual['vmhwm_bytes'] == 81920
    assert actual['mem_available_bytes'] == 131072


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


def test_rebind_instruments_public_identity_with_historical_phase_and_cleanup(monkeypatch, tmp_path):
    import torch
    from prismaquant import tessera_campaign as campaign
    from experiments import campaign_seed_rebind_timing as rebind

    class ObservedBinding(RuntimeError):
        pass

    def identity():
        return 'bound'

    def stop_after_binding(argv):
        assert campaign.campaign_checkpoint_identity() == 'bound'
        raise ObservedBinding('instrumentation bound before model execution')

    monkeypatch.setattr(campaign, 'campaign_checkpoint_identity', identity)
    monkeypatch.setattr(campaign, 'main', stop_after_binding)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    out = tmp_path / 'result'
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(
        schema=rebind.SCHEMA, input_sha256={}, environment={}, out=str(out),
        command=['--out', 'old', '--cache-dir', 'old', '--checkpoint', 'old'],
        seed_checkpoint='unused', seed_wire_dir='unused', overlap_bytes=0,
        identity_bytes=0, row_id='binding', seed_wire_bytes=0, seed_unit_shards=0)))
    with pytest.raises(ObservedBinding):
        rebind.run(plan, rebind.digest(plan))
    assert campaign.campaign_checkpoint_identity is identity
    result = json.loads((out / 'rebind-timing.json').read_text())
    assert result['phases']['prismaquant.tessera_campaign._campaign_checkpoint_identity']['calls'] == 1


def test_profile_reader_keeps_historical_key_for_old_new_and_recursive_frames(tmp_path):
    from experiments.stream_row_head_640.profile_summary import profile

    (tmp_path / 'pyspy.raw.txt').write_text(
        'main;_campaign_checkpoint_identity (old.py:1) 3\n'
        'main;campaign_checkpoint_identity (new.py:1) 5\n'
        'main;campaign_checkpoint_identity (new.py:1);campaign_checkpoint_identity (new.py:2) 7\n')
    result = profile(tmp_path)
    assert result['samples'] == 15
    assert result['inclusive']['_campaign_checkpoint_identity'] == 15
    assert 'campaign_checkpoint_identity' not in result['inclusive']
