from __future__ import annotations

import contextlib
import json

import pytest

from prismaquant import stage_b_workspace_profile as profile
from test_stage_b_workspace_profile import _Guard, _fake_measure


@pytest.mark.parametrize("capture_batch,previous_peak,expected_last_reserve", [
    (3, 19, 32),
    (5, 31, 42),
    (4, 19, 41),
    (3, 1152921504606846977, 1729382256910270469),
], ids=["two_to_three_rounds_up", "four_to_five_rounds_up", "doubling_control",
        "large_byte_count_remains_exact"])
def test_capture_ladder_charges_the_full_measured_batch_ratio(
    monkeypatch, capture_batch, previous_peak, expected_last_reserve,
):
    peaks = {1: 10, 2: 19, 3: 27, 4: 31, 5: 37}
    peaks[profile.ladder(capture_batch)[-2][2]] = previous_peak
    monkeypatch.setattr(profile, "measure_group", _fake_measure(peaks))
    monkeypatch.setattr("prismaquant.aura_cost._release_streamed_anchor_allocator_cache",
                        lambda device: None)
    runs = []

    @contextlib.contextmanager
    def observed():
        yield "observer"

    guard = _Guard(refuse_label=None)
    groups = 16
    settings, complete = profile.run_ladder(
        [(index, None, None, None) for index in range(groups * capture_batch)],
        run_group=lambda group, observer: runs.append((len(group), observer)),
        observed=observed, guard=guard, declared_workspace_bytes=100,
        capture_reserve_bytes=3, capture_batch=capture_batch, groups=groups,
        device="cpu", host=None,
    )

    assert complete is True
    assert profile.ladder(capture_batch)[-1][2] == capture_batch
    assert settings[-1]["name"] == f"capture-b{capture_batch}"
    assert guard.charged[-1][1:] == (0, expected_last_reserve)
    assert settings[-1]["admission"]["reserve_device_bytes"] == expected_last_reserve
    assert runs.count((capture_batch, "observer")) == groups
    receipt = json.loads(json.dumps(settings, allow_nan=False))
    assert receipt[-1]["admission"]["reserve_device_bytes"] == expected_last_reserve
