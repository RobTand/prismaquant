"""Replay synthetic telemetry through the real summary CLI; no GPU measurement."""
import json
import sys

import pytest

from experiments.stream_row_head_640 import profile_summary


@pytest.mark.parametrize("sampled_encode", [False, True])
def test_encode_power_requires_samples_in_the_encode_window(tmp_path, monkeypatch, sampled_encode):
    arm = tmp_path / "arm"
    arm.mkdir()
    start = 1_000_000_000.0
    (arm / "host.txt").write_text(f"start={start} end={start + 24}\n")
    (arm / "timeline.tsv").write_text(
        "time\tunused\tnfs_bytes\tunused\tpower_w\n" + "".join(
            f"{start + 2 * i}\t0\t{i * 100}\t0\t80\n" for i in range(12)))
    offsets = (2.0, 18.0) if sampled_encode else (0.5, 1.5)
    (arm / "campaign.log").write_text("".join(
        f"{start + offset} [campaign] r1 {index}/2 batch=2\n"
        for index, offset in enumerate(offsets, 1)))
    (arm / "pyspy.raw.txt").write_text("synthetic_frame 1\n")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [str(profile_summary.__file__), str(arm), "--json", str(output)])
    assert profile_summary.main() == 0
    result = json.loads(output.read_bytes())[arm.name]["timeline"]
    assert result["samples"] == 12
    assert result["whole_mean_power_w"] == 80.0
    assert result["nfs_read_bytes_total"] == 1100
    if sampled_encode:
        assert result["encode_median_power_w"] == result["encode_mean_power_w"] == 80.0
        assert result["sustained_power_seconds"] == 0.0
        assert result["nfs_read_bytes_before_sustained"] == 0
    else:
        assert result["encode_median_power_w"] is None
        assert result["encode_median_envelope_fraction"] is None
        assert result["encode_mean_power_w"] is None
        assert result["sustained_power_seconds"] is None
        assert result["nfs_read_bytes_before_sustained"] is None



def _summary_for_samples(tmp_path, monkeypatch, powers, encode_offsets):
    arm = tmp_path / "arm"
    arm.mkdir()
    start = 1_000_000_000.0
    (arm / "host.txt").write_text(f"start={start} end={start + 24}\n")
    (arm / "timeline.tsv").write_text(
        "time\tunused\tnfs_bytes\tunused\tpower_w\n" + "".join(
            f"{start + 2 * index}\t0\t{100 * index}\t0\t{power}\n"
            for index, power in enumerate(powers)))
    (arm / "campaign.log").write_text("".join(
        f"{start + offset} [campaign] r1 {index}/2 batch=2\n"
        for index, offset in enumerate(encode_offsets, 1)))
    (arm / "pyspy.raw.txt").write_text("synthetic_frame 1\n")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        str(profile_summary.__file__), str(arm), "--json", str(output),
    ])
    assert profile_summary.main() == 0
    return json.loads(output.read_bytes())[arm.name]["timeline"]


def test_observed_zero_encode_power_stays_numeric(tmp_path, monkeypatch):
    result = _summary_for_samples(
        tmp_path, monkeypatch, [80, *([0] * 9), 80, 80], (2.0, 18.0),
    )
    assert result["samples"] == 12
    assert result["encode_median_power_w"] == 0.0
    assert result["encode_median_envelope_fraction"] == 0.0
    assert result["encode_mean_power_w"] == 0.0
    assert result["whole_mean_power_w"] == 20.0
    assert result["peak_power_w"] == 80.0
    # Preserve the existing threshold rule when its observed median is zero.
    assert result["sustained_power_seconds"] == 0.0
    assert result["nfs_read_bytes_before_sustained"] == 0
    assert result["nfs_read_bytes_total"] == 1100
    assert result["encode_seconds"] == 16.0
    assert result["batches"] == 2


def test_one_observed_encode_sample_is_not_missing(tmp_path, monkeypatch):
    result = _summary_for_samples(
        tmp_path, monkeypatch, [10, 10, 53, *([10] * 9)], (3.5, 4.5),
    )
    assert result["samples"] == 12
    assert result["encode_median_power_w"] == 53.0
    assert result["encode_median_envelope_fraction"] == 0.379
    assert result["encode_mean_power_w"] == 53.0
    assert result["whole_mean_power_w"] == 13.58
    assert result["peak_power_w"] == 53.0
    assert result["sustained_power_seconds"] is None
    assert result["nfs_read_bytes_before_sustained"] is None
    assert result["nfs_read_bytes_total"] == 1100
    assert result["encode_seconds"] == 1.0
    assert result["batches"] == 2
