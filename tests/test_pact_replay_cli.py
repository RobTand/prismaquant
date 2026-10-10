"""The replay CLI keeps its science contract and admits a stall allowance."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "pact_replay"
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


def _args(allowance):
    return SimpleNamespace(
        band="3:9",
        layer_range="3:9",
        input_contract="prefixed_514",
        weight_source=["A8S"],
        window_layers=21,
        window_start=3,
        window_stop=24,
        stall_seconds=allowance,
    )


def _public_args(allowance=None):
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
    if allowance is not None:
        argv.append("--stall-seconds=" + str(allowance))
    return cli_parser().parse_args(argv)



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


@pytest.mark.parametrize("allowance", [0.01, 1, 1800, 3500, 3501, 7200])
def test_public_cli_resolves_positive_stall_allowances(allowance):
    args = _public_args(allowance)
    roster, receipt, raw = _fixture()
    run = resolve_cli_run(args, receipt, raw, roster)
    assert (run["band_start"], run["band_stop"], run["replay_stop"]) == (3, 9, 45)
    assert (run["window_start"], run["window_stop"]) == (3, 24)
    assert run["manifest_streams"][0]["stream_id"] == "null"


@pytest.mark.parametrize("allowance", [0, -1, "nan", "inf", "-inf"])
def test_public_cli_refuses_invalid_stall_allowances(allowance):
    args = _public_args(allowance)
    roster, receipt, raw = _fixture()
    with pytest.raises(ValueError, match="positive finite stall allowance"):
        resolve_cli_run(args, receipt, raw, roster)


def test_cli_uses_the_approved_provisional_stall_policy():
    args = _public_args()
    assert args.stall_seconds == 1800

