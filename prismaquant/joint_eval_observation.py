"""What a diagnostic joint-evaluation panel observed, per unit.

Neutral home for the two facts the cost stage and the currency gate read off a
pilot panel's rows (decoupling step 6, PQ #1550): the panel status every
diagnostic row carries, and whether an expert was observed at all. The panel
itself (``tessera_joint_eval_panel``) selects rows and re-exports both.
"""
from __future__ import annotations

#: The status a diagnostic pilot panel stamps on its provenance and rows.
STATUS = 'diagnostic_pilot'


def observation_status(count):
    """An invocation with zero token rows does not observe that expert."""
    if (not isinstance(count, dict) or not {'tokens', 'calls'} <= set(count)
            or any(type(count[key]) is not int or count[key] < 0 for key in ('tokens', 'calls'))
            or (count['tokens'] > 0 and count['calls'] == 0)):
        raise ValueError("joint evaluation observation counts are invalid")
    return "observed" if count["tokens"] > 0 else "unknown_unobserved"


def new_observation_counts(names, n_probes):
    if type(n_probes) is not int or n_probes < 2:
        raise ValueError("joint evaluation needs at least two integer probes")
    return {name: {"tokens": 0, "calls": 0, "n_probes": n_probes,
                   "count_scope": "summed_over_probes",
                   "per_probe": [{"tokens": 0, "calls": 0} for _ in range(n_probes)]}
            for name in names}


def observe_probe(counts, name, probe_index, diagnostic):
    count = counts[name]
    if type(probe_index) is not int or not 0 <= probe_index < count["n_probes"]:
        raise ValueError("joint evaluation probe index differs")
    observed = {"tokens": diagnostic["observed_tokens"], "calls": diagnostic["observed_calls"]}
    observation_status(observed)
    for key, value in observed.items():
        count[key] += value
        count["per_probe"][probe_index][key] += value


def stamp_observations(payload, counts, panel):
    if set(counts) != set(payload["costs"]):
        raise RuntimeError("joint pilot observation roster differs")
    for name, count in counts.items():
        if (count["count_scope"] != "summed_over_probes"
                or len(count["per_probe"]) != count["n_probes"]
                or any(count[key] != sum(part[key] for part in count["per_probe"])
                       for key in ("tokens", "calls"))):
            raise RuntimeError(f"joint pilot invalid observation count for {name}")
        status = observation_status(count)
        payload["stats"][name]["joint_eval_observations"] = dict(count)
        payload["stats"][name]["joint_eval_status"] = status
        for row in payload["costs"][name].values():
            row["joint_eval_status"] = status
            row["joint_eval_observations"] = dict(count)
    payload["provenance"]["joint_eval"] = panel
