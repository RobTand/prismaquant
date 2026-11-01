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


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('failed', [False, True])
def test_profile_preserves_unit_attribution_when_a_callback_mutates_its_batch(batched, failed):
    recorder = PhaseRecorder()
    names = ['a.gate_proj', 'a.up_proj']
    inputs = dict(qnames=names) if batched else dict(qname=names[0])
    result = object()
    error = ValueError('original measurement failure')

    def measure(**kwargs):
        if batched:
            kwargs['qnames'].clear()
        if failed:
            raise error
        return result

    measured = recorder.wrap(measure, 'measurement')
    if failed:
        with pytest.raises(ValueError) as caught:
            measured(format_name='tessera', **inputs)
        assert caught.value is error
    else:
        assert measured(format_name='tessera', **inputs) is result
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record['format_name'] == 'tessera'
    if batched:
        assert record['qname'] is None
        assert record['qnames'] == ['a.gate_proj', 'a.up_proj']
        assert record['batch_size'] == 2
    else:
        assert record['qname'] == 'a.gate_proj'
        assert 'qnames' not in record and 'batch_size' not in record


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


#: Collapsed py-spy stacks of both code eras. The first two old lines are
#: frames of the 2026-09-15 row-0055 arm (stream-row-head-640/arms/after-1)
#: verbatim; the scalar line follows the 138069b799 call chain; the new lines
#: follow the tessera_cell_measurement owner. Counts are chosen so every sum
#: is distinct.
_RUNPY = ('_run_module_as_main (<frozen runpy>:198);_run_code (<frozen runpy>:88);'
          '<module> (prismaquant/tessera_campaign.py:7041);')
_OLD_MAIN = _RUNPY + 'main (prismaquant/tessera_campaign.py:5068);_main (prismaquant/tessera_campaign.py:6608);'
_NEW_MAIN = (_RUNPY + 'main (prismaquant/tessera_campaign.py:3829);_main (prismaquant/tessera_campaign.py:4836);'
             'run_loaded_campaign (prismaquant/tessera_campaign.py:6016);')
OLD_STACKS = (
    _OLD_MAIN + '_measure_anchor_batch (prismaquant/tessera_campaign.py:698);'
    'encode_tessera_units (prismaquant/tessera_render.py:1402);read_unit_artifact '
    '(tessera/unit_artifact.py:811);reconstruct_unit (tessera/decode.py:1139);'
    'dequantize (tessera/decode.py:646) 3',
    _OLD_MAIN + '_measure_anchor_batch (prismaquant/tessera_campaign.py:705);'
    '_finish_anchor (prismaquant/tessera_campaign.py:648);_publish '
    '(prismaquant/tessera_campaign.py:625);_store_rendered_weight_entry '
    '(prismaquant/production_weight_cache.py:1856);save (torch/serialization.py:1003) 5',
    _OLD_MAIN + '_measure_anchor (prismaquant/tessera_campaign.py:522);_encode_and_render '
    '(prismaquant/tessera_campaign.py:451);encode_tessera_unit (prismaquant/tessera_render.py:1287) 7',
)
NEW_STACKS = (
    _NEW_MAIN + 'measure_cells (prismaquant/tessera_cell_measurement.py:494);'
    'encode_tessera_units (prismaquant/tessera_render.py:1260) 11',
    _NEW_MAIN + 'measure_cells (prismaquant/tessera_cell_measurement.py:501);_score_and_publish '
    '(prismaquant/tessera_cell_measurement.py:425);_publish (prismaquant/tessera_cell_measurement.py:400) 13',
    _NEW_MAIN + 'measure_cells (prismaquant/tessera_cell_measurement.py:466);measure_cell '
    '(prismaquant/tessera_cell_measurement.py:103);encode_tessera_unit (prismaquant/tessera_render.py:1066) 17',
)


@pytest.mark.parametrize(('stacks', 'samples', 'expected'), [
    (OLD_STACKS, 15, {'_measure_anchor_batch': 8, '_encode_and_render': 7,
                      'measure_cells': 0, 'encode_tessera_unit': 10}),
    (NEW_STACKS, 41, {'_measure_anchor_batch': 0, '_encode_and_render': 0,
                      'measure_cells': 41, 'encode_tessera_unit': 28}),
], ids=['before-cell-owner', 'cell-owner'])
def test_profile_reader_counts_each_era_under_its_own_measurement_names(
        tmp_path, stacks, samples, expected):
    """A historical arm still reads its measurement frames, and a new arm's
    frames are never counted under the old names (C-PRF-02)."""
    from experiments.stream_row_head_640.profile_summary import profile

    (tmp_path / 'pyspy.raw.txt').write_text('\n'.join(stacks) + '\n')
    result = profile(tmp_path)
    assert result['samples'] == samples
    assert {name: result['inclusive'][name] for name in expected} == expected


def test_instrument_names_cell_spans_apart_and_keeps_the_memo_key(monkeypatch):
    """``measure_cells`` and ``_score_and_publish`` record under their own
    names, nothing records as ``_finish_anchor``, and the moved memo's
    counters keep their historical key (C-PRF-02)."""
    from experiments.campaign_publication_ab import instrument
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_cell_measurement as cell_measurement

    def score(**kwargs):
        return 'scored'

    def batch(**kwargs):
        return 'measured'

    monkeypatch.setattr(cell_measurement, '_score_and_publish', score)
    monkeypatch.setattr(cell_measurement, 'measure_cells', batch)
    recorder = PhaseRecorder()
    with instrument(campaign, recorder, 'gate_up'):
        assert cell_measurement._score_and_publish(qname='u', format_name='f') == 'scored'
        assert cell_measurement.measure_cells(qnames=['u', 'v'], format_name='f') == 'measured'
        cell_measurement.encoder_kwargs_memo(None, {}, 'cpu')
    assert cell_measurement._score_and_publish is score
    assert cell_measurement.measure_cells is batch
    assert [(record['phase'], record['qname'], record.get('qnames'))
            for record in recorder.records] == [
        ('prismaquant.tessera_cell_measurement._score_and_publish', 'u', None),
        ('prismaquant.tessera_cell_measurement.measure_cells', None, ['u', 'v']),
        ('tessera_campaign._activation_kwargs_memo.cache_info', None, None)]
