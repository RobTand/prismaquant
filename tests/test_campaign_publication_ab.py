"""The comparison must preserve compatible work and report truthful spans."""
import pytest

from experiments.campaign_publication_ab import preferred_batches, PhaseRecorder


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
