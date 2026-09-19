"""Every frontier point carries the dispersion its own medians resolve.

An ``attained_prefill_ms`` is a sum of per-row medians. Published alone it reads
as a resolved number, so two points 0.2 ms apart look ordered whether or not the
samples behind them can tell them apart. These tests hold the two halves of the
fix: the interval is drawn from the priced rows' OWN samples and from nothing
else, and it is published rather than applied -- no point is dropped, promoted
or reordered by it.

The refusals are demonstrated by mutating the driver: a verdict that does not
say which rows it summed, and a table missing the samples for a row the verdict
summed. Both are cases where the easy failure is a silently absent interval.

Synthetic CPU fixtures throughout; nothing here is GPU evidence.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from prismaquant import prefill_frontier
from prismaquant.measured_runtime_prices import RuntimePriceError, bootstrap_sum
from test_allocator_measured_runtime_cli import admit_synthetic_table
from test_prefill_frontier import _curve_fixture

#: Small enough to keep the suite fast, large enough that the 2.5/97.5 indices
#: are distinct draws rather than the same one.
DRAWS = 400


def _run(tmp_path, allocator_argv, *, own=("--slo-grid", "auto")):
    output = tmp_path / "frontier.json"
    code = prefill_frontier.main([*own, "--bootstrap-draws", str(DRAWS),
                                  "--output", str(output), "--", *allocator_argv])
    return code, json.loads(output.read_text())


def _retime_one_row(tmp_path, *, unit_suffix, fmt, samples_ms):
    """Give one priced row samples that disagree, leaving its median alone.

    The row's ``resources.prefill_ms`` must stay the row's median or the parser
    refuses the table, so the point estimate -- and therefore every assignment,
    every bound and the whole curve -- is unchanged by construction. Only the
    width the samples resolve moves.
    """
    path = tmp_path / "runtime.json"
    table = json.loads(path.read_text())
    hit = 0
    for row in table["rows"]:
        if row["unit"].endswith(unit_suffix) and row["format"] == fmt:
            assert sorted(samples_ms)[len(samples_ms) // 2] == row["resources"]["prefill_ms"]
            row["prefill"]["samples_ms"] = list(samples_ms)
            hit += 1
    assert hit == 1, f"expected exactly one {unit_suffix}@{fmt} row, found {hit}"
    path.write_text(json.dumps(table))


# --------------------------------------------------------------------------- #
# The interval is there, and it says what it covers
# --------------------------------------------------------------------------- #

def test_every_feasible_point_carries_an_interval_over_its_own_rows(tmp_path, monkeypatch):
    admit_synthetic_table(monkeypatch)
    _, allocator_argv = _curve_fixture(tmp_path)
    code, doc = _run(tmp_path, allocator_argv)
    assert code == 0 and doc["n_feasible"] == doc["n_points"] >= 3

    for point in doc["points"]:
        interval = point["attained_prefill_ms_bootstrap"]
        assert interval is not None
        assert interval["draws"] == DRAWS
        assert interval["p2.5"] <= interval["p50"] <= interval["p97.5"]
        # The fixture times every repeat identically, so the samples resolve
        # this sum exactly and the interval collapses onto the point. That is
        # the measurement's answer, not a missing interval.
        assert interval["p2.5"] == interval["p97.5"] == point["attained_prefill_ms"]
        assert interval["samples_per_row"] == [3, 3, 3]
        # Three units, each priced once by the chosen assignment.
        assert len(point["serve_constraints"]["coverage"]["priced_rows"]) == 3
        # The fixture prices no decode, so there is no decode sum to disperse.
        assert point["attained_decode_ms"] is None
        assert point["attained_decode_ms_bootstrap"] is None

    stamp = doc["provenance"]["bootstrap"]
    assert stamp["draws"] == DRAWS and stamp["seed"] == prefill_frontier.BOOTSTRAP_SEED
    assert stamp["function"] == "prismaquant.measured_runtime_prices.bootstrap_sum"
    assert stamp["applied_as_a_threshold"] is False


def test_the_interval_widens_when_the_rows_own_samples_disagree(tmp_path, monkeypatch):
    admit_synthetic_table(monkeypatch)
    """The check that bites: a narrow interval must be a fact, not a default.

    One row is re-timed so its three repeats disagree while its median -- the
    priced number -- is untouched. Nothing the solver reads changes, so the
    curve must be point-for-point identical, and the one thing that must move
    is the width.
    """
    _, allocator_argv = _curve_fixture(tmp_path)
    _, before = _run(tmp_path, allocator_argv)
    # FP8_E4M3 is the slow accurate rung every point at a loose budget takes.
    _retime_one_row(tmp_path, unit_suffix="layers.0.self_attn.o_proj",
                    fmt="FP8_E4M3", samples_ms=[2.0, 8.0, 14.0])
    _, after = _run(tmp_path, allocator_argv)

    unchanged = ("slo_ms", "feasible", "predicted_dloss", "achieved_bits", "payload_bytes",
                 "attained_prefill_ms", "device_memory_bytes", "assignment_sha256",
                 "nondominated")
    assert ([{key: point[key] for key in unchanged} for point in after["points"]]
            == [{key: point[key] for key in unchanged} for point in before["points"]])
    assert after["saturation"]["slo_ms"] == before["saturation"]["slo_ms"]
    assert after["n_nondominated"] == before["n_nondominated"]

    widened = [point for point in after["points"]
               if point["attained_prefill_ms_bootstrap"]["p2.5"]
               < point["attained_prefill_ms_bootstrap"]["p97.5"]]
    assert widened, "re-timing a priced row resolved the sum no less precisely"
    for point in widened:
        interval = point["attained_prefill_ms_bootstrap"]
        # The interval brackets the point it is the dispersion of.
        assert interval["p2.5"] <= point["attained_prefill_ms"] <= interval["p97.5"]
        # ... and the row that was re-timed is one this point actually summed.
        assert ["model.layers.0.self_attn.o_proj", "FP8_E4M3"] in \
            point["serve_constraints"]["coverage"]["priced_rows"]


def test_the_same_rows_and_seed_give_the_same_interval_twice(tmp_path, monkeypatch):
    admit_synthetic_table(monkeypatch)
    """A published interval is a number a reader can reproduce."""
    _, allocator_argv = _curve_fixture(tmp_path)
    _retime_one_row(tmp_path, unit_suffix="layers.1.self_attn.o_proj",
                    fmt="FP8_E5M2", samples_ms=[0.5, 2.0, 5.0])
    _, first = _run(tmp_path, allocator_argv)
    _, again = _run(tmp_path, allocator_argv)
    assert ([point["attained_prefill_ms_bootstrap"] for point in first["points"]]
            == [point["attained_prefill_ms_bootstrap"] for point in again["points"]])


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #

def _verdict(priced_rows, *, prefill=6.0, decode=None):
    return {"coverage": {"priced_rows": [list(row) for row in priced_rows]},
            "predicted": {"operator_sum_prefill_ms": prefill,
                          "operator_sum_decode_ms": decode}}


def test_a_verdict_that_hides_which_rows_it_summed_is_refused():
    with pytest.raises(prefill_frontier.PrefillFrontierError,
                       match="does not say which rows it summed"):
        prefill_frontier.point_dispersion(
            {"coverage": {}, "predicted": {"operator_sum_prefill_ms": 6.0}},
            prefill_samples_ms={}, decode_samples_ms={},
            fixed_prefill_ms=0.0, fixed_decode_ms=None, draws=DRAWS, seed=1)


def test_a_summed_row_with_no_samples_is_refused_not_omitted():
    """The failure mode is a point published with a silently absent interval."""
    with pytest.raises(prefill_frontier.PrefillFrontierError,
                       match="cannot be published without the dispersion"):
        prefill_frontier.point_dispersion(
            _verdict([("u", "A"), ("v", "B")]),
            prefill_samples_ms={("u", "A"): (3.0, 3.0, 3.0)}, decode_samples_ms={},
            fixed_prefill_ms=0.0, fixed_decode_ms=None, draws=DRAWS, seed=1)


def test_a_feasible_point_with_no_prefill_interval_is_refused(tmp_path):
    """Mutate the driver: a dispersion source that returns nothing for prefill."""
    record = {"slo_ms": 1.0, "target_bits": 4.0, "feasible": True, "assignment": {"u": "A"},
              "predicted_dloss": 1.0, "achieved_bits": 4.0, "payload_bytes": 8,
              "serve_constraints": {"predicted": {"operator_sum_prefill_ms": 1.0,
                                                  "operator_sum_decode_ms": None,
                                                  "device_memory_bytes": None}},
              "diagnostics": {}}
    with pytest.raises(prefill_frontier.PrefillFrontierError,
                       match="with no interval over the samples"):
        prefill_frontier._point_record(record, Path(tmp_path), provenance_stub={},
                                       dispersion=lambda verdict: {"prefill": None, "decode": None})


def test_zero_draws_is_not_a_way_to_skip_the_interval(tmp_path, capsys):
    _, allocator_argv = _curve_fixture(tmp_path)
    with pytest.raises(SystemExit):
        prefill_frontier.main(["--slo-grid", "auto", "--bootstrap-draws", "0",
                               "--output", str(tmp_path / "frontier.json"),
                               "--", *allocator_argv])
    assert "--bootstrap-draws must be at least 1" in capsys.readouterr().err
    assert not (tmp_path / "frontier.json").exists()
    with pytest.raises(RuntimePriceError, match="draws must be at least 1"):
        bootstrap_sum([(1.0, 2.0, 3.0)], draws=0, seed=1)


# --------------------------------------------------------------------------- #
# The two scopes stay separate inside the interval too
# --------------------------------------------------------------------------- #

def test_the_fixed_term_shifts_the_interval_and_adds_no_width():
    """The whole-engine charge is a constant, because nothing measured it.

    The report schema carries no samples for the fixed term, so it can only
    enter as an offset. Drawing a width for it would be dispersion with no
    measurement under it.
    """
    rows = [(1.0, 5.0, 9.0), (2.0, 2.0, 2.0)]
    plain = bootstrap_sum(rows, draws=DRAWS, seed=11)
    shifted = bootstrap_sum(rows, draws=DRAWS, seed=11, offset_ms=100.0)
    assert plain["p97.5"] > plain["p2.5"]          # the rows themselves disperse
    assert shifted["offset_ms"] == 100.0
    for key in ("p2.5", "p50", "p97.5"):
        assert shifted[key] == pytest.approx(plain[key] + 100.0)
    assert (shifted["p97.5"] - shifted["p2.5"]) == pytest.approx(plain["p97.5"] - plain["p2.5"])


def test_decode_gets_its_own_interval_only_where_the_table_prices_it():
    rows = [("u", "A"), ("v", "B")]
    samples = {("u", "A"): (1.0, 1.0, 1.0), ("v", "B"): (2.0, 2.0, 2.0)}
    both = prefill_frontier.point_dispersion(
        _verdict(rows, prefill=3.0, decode=3.0),
        prefill_samples_ms=samples, decode_samples_ms=samples,
        fixed_prefill_ms=0.0, fixed_decode_ms=0.0, draws=DRAWS, seed=1)
    assert both["prefill"]["p50"] == 3.0 and both["decode"]["p50"] == 3.0

    # The same rows, with decode unpriced: the attained decode is already null,
    # so there is nothing to disperse and nothing is invented.
    prefill_only = prefill_frontier.point_dispersion(
        _verdict(rows, prefill=3.0, decode=None),
        prefill_samples_ms=samples, decode_samples_ms={},
        fixed_prefill_ms=0.0, fixed_decode_ms=None, draws=DRAWS, seed=1)
    assert prefill_only["decode"] is None and prefill_only["prefill"]["p50"] == 3.0


# --------------------------------------------------------------------------- #
# One bootstrap, two tools
# --------------------------------------------------------------------------- #

def test_the_accuracy_curve_uses_this_very_function():
    """Not a second bootstrap that happens to agree today."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "pq_prefill_accuracy_curve.py"
    spec = importlib.util.spec_from_file_location("pq_prefill_accuracy_curve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.bootstrap_sum is bootstrap_sum
