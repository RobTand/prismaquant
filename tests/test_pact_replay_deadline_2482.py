"""The PACT replay CLI accepts the approved deadline and keeps its bounds."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "prismaquant" / "pact_replay"
sys.path.insert(0, str(REPLAY_ROOT))

from multi_stream_replay import cli_parser, resolve_cli_run  # noqa: E402


def _fixture():
    roster = {
        "schema": "pact.stream_roster.v1",
        "band": "3:9",
        "streams": [
            {
                "stream_id": "null",
                "class": "routed",
                "kind": "W_T8",
                "weight_source": "A8S",
                "amplitude": 1,
                "null_replay": True,
            }
        ],
    }
    receipt = {
        "layer_start": 3,
        "layer_stop": 9,
        "start_reference_owner": {"session": {"generation": "g"}},
        "session": {"generation": "g"},
    }
    return roster, receipt, b"fixture-receipt-bytes"


def _args(deadline):
    return SimpleNamespace(
        band="3:9",
        layer_range="3:9",
        input_contract="prefixed_514",
        weight_source=["A8S"],
        window_layers=21,
        window_start=3,
        window_stop=24,
        deadline_seconds=deadline,
    )


def _public_args(deadline=None):
    paths = [
        "--pq-root", "--tessera-src", "--g3-source", "--source-root",
        "--a8-root", "--t8r-root", "--a4-root", "--exl3-root",
        "--manifest", "--reference-manifest", "--capture-root", "--draw",
        "--output", "--run-manifest-out", "--stream-manifest", "--energy-rows",
        "--teacher-frontier", "--teacher-content", "--output-dir", "--checkpoint-dir",
    ]
    argv = [value for flag in paths for value in (flag, "fixture")]
    argv += [
        "--draw-sha256", "fixture-draw", "--teacher-content-sha256", "fixture-teacher",
        "--band", "3:9", "--layer-range", "3:9", "--input-contract", "prefixed_514",
        "--sample-range", "384:448", "--device", "cpu", "--window-layers", "21",
        "--window-start", "3", "--window-stop", "24", "--weight-source", "A8S",
    ]
    if deadline is not None:
        argv.append("--deadline-seconds=" + str(deadline))
    return cli_parser().parse_args(argv)


def test_default_deadline_stays_1700():
    args = _public_args()
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(args, receipt, raw, roster)
    assert args.deadline_seconds == 1700
    assert run["plan"] == [
        {"window_start": 3, "window_stop": 24},
        {"window_start": 24, "window_stop": 45},
    ]


def test_approved_deadline_passes():
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(_args(3500), receipt, raw, roster)
    assert (run["band_start"], run["band_stop"]) == (3, 9)
    assert (run["replay_start"], run["replay_stop"]) == (3, 45)


@pytest.mark.parametrize("deadline", [1, 100, 1699, 1700, 1701, 3499, 3500])
def test_valid_legacy_and_approved_deadlines_pass(deadline):
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(_args(deadline), receipt, raw, roster)
    assert run["window_start"] == 3
    assert run["window_stop"] == 24


@pytest.mark.parametrize(
    "deadline", [0, -1, -3500, 3501, 3600, 7200, float("nan"), float("inf")]
)
def test_invalid_deadlines_refuse(deadline):
    roster, receipt, raw = _fixture()
    with pytest.raises(ValueError, match="bounded deadline"):
        resolve_cli_run(_args(deadline), receipt, raw, roster)


def test_science_inputs_match_canonical_resolution():
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(_args(1700), receipt, raw, roster)
    assert run["plan"] == [
        {"window_start": 3, "window_stop": 24},
        {"window_start": 24, "window_stop": 45},
    ]
    assert run["generations"]["source_window"] == [3, 9]
    assert [s["stream_id"] for s in run["manifest_streams"]] == ["null"]


def test_science_refusals_stay():
    roster, receipt, raw = _fixture()
    bad_band = _args(3500)
    bad_band.band = "0:4"
    with pytest.raises(ValueError, match="band"):
        resolve_cli_run(bad_band, receipt, raw, roster)
    bad_contract = _args(3500)
    bad_contract.input_contract = "raw_512"
    with pytest.raises(ValueError, match="cohort"):
        resolve_cli_run(bad_contract, receipt, raw, roster)
    bad_window = _args(3500)
    bad_window.window_stop = 25
    with pytest.raises(ValueError, match="window grid"):
        resolve_cli_run(bad_window, receipt, raw, roster)


@pytest.mark.parametrize("deadline", [1, 100, 1700, 3500])
def test_public_cli_resolves_valid_deadlines(deadline):
    args = _public_args(deadline)
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(args, receipt, raw, roster)
    assert (run["band_start"], run["band_stop"], run["replay_stop"]) == (3, 9, 45)
    assert (run["window_start"], run["window_stop"]) == (3, 24)
    assert run["manifest_streams"][0]["stream_id"] == "null"


@pytest.mark.parametrize("deadline", ["nan", "inf", "-inf"])
def test_public_cli_refuses_nonfinite_deadlines(deadline, capsys):
    with pytest.raises(SystemExit) as refusal:
        _public_args(deadline)
    assert refusal.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


@pytest.mark.parametrize("deadline", [0, -1, 3501])
def test_public_cli_refuses_invalid_bounds(deadline):
    args = _public_args(deadline)
    roster, receipt, raw = _fixture()
    with pytest.raises(ValueError, match="bounded deadline"):
        resolve_cli_run(args, receipt, raw, roster)
