#!/usr/bin/env python3
"""REHOME-R1 mutants of prismaquant/tessera_resource_transfer.py.

usage: python3 tools_mutate_rehome.py <checkout> <name>
Each anchor must match exactly once, or the script refuses.
"""
import pathlib
import sys

root, name = sys.argv[1], sys.argv[2]
p = pathlib.Path(root) / "prismaquant" / "tessera_resource_transfer.py"
s = p.read_text()

GATE_K = "    K = 2 * k + 2\n"
GATE_K2K = "    K = 2 * k\n"

M = {
    "n2_k2k": (GATE_K, GATE_K2K),
    "n7_alpha": ("        if p_screen < ALPHA / k:\n", "        if p_screen < ALPHA:\n"),
    "n8_df": ("            df_i = (r - 1) * v ** 2 / den\n", "            df_i = (r - 1)\n"),
    "n14_mde": ("    mde80 = (t_crit + float(_st.norm.ppf(0.8))) * factor * s\n",
                "    mde80 = t_crit * factor * s\n"),
    "n15_interval": (
        "    hw = float(_st.t.ppf(0.995, k - 1)) * spread\n",
        "    hw = 1.96 * spread\n"),
    "n6_vsum": (
        "            v = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[key] ** 2\n"
        "                                   + sum(s_i[other] ** 2 for other in s_i\n"
        "                                         if other != key) / k ** 2)\n",
        "            v = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[key] ** 2)\n"),
    "n3_epslog_pooled": (
        "        else:\n            threshold = t_crit * factor * s + eps_log[key]\n",
        "        else:\n            threshold = t_crit * factor * s\n"),
    "n16_epslog_fb": (
        "            threshold += eps_log[key]\n",
        ""),
    "wire_double_prefix": (
        "        def wire(rate):\n"
        "            return f\"{family}_R{rate}\"\n",
        "        def wire(rate):\n"
        "            return f\"TESSERA_{family}_R{rate}\"\n"),
    "fresh_rate": (
        "        wire_rate = _wire_rate(runtime_identity, rate)\n"
        "        if set(report[\"pass_r\"]) != {wire_rate}:\n"
        "            raise ValueError(f\"fresh report for {rate} must key exactly {wire_rate}\")\n",
        "        wire_rate = _wire_rate(runtime_identity, rate)\n"),
    "b3_ordinal": (
        "            if row.get(\"time_in_process\") != record.get(\"time_in_process\"):\n"
        "                reasons.append(f\"band ordinal for {wire_rate} {phase} differs from \"\n"
        "                               f\"the run report\")\n",
        ""),
    "b3_order": (
        "        if [row[\"q256\"] for row in band_rows] != window_order:\n"
        "            reasons.append(f\"persistent timing order disagrees with the pass-R \"\n"
        "                           f\"windows: {phase}\")\n",
        ""),
    "distinct_trace": (
        "        if fresh_window.get(\"trace_sha256\") == persistent_trace:\n"
        "            failures.append(\"fresh and persistent windows derive from one trace\")\n",
        ""),
    "window_pid": (
        "        process = record.get(\"process\")\n"
        "        if not isinstance(process, dict) or window.get(\"process_id\") != process.get(\"pid\"):\n"
        "            raise ValueError(f\"pass-R record for {rate} window belongs to another process\")\n",
        ""),
    "one_process_r": (
        "    if len(resource_processes) != 1:\n"
        "        reasons.append(\"pass-R sampled rates ran in more than one process\")\n",
        ""),
    "band_fresh_disjoint": (
        "    gate_fresh = set(gate.get(\"fresh_processes\", []))\n"
        "    if gate_fresh & (set(fresh_processes) | persistent_processes):\n"
        "        reasons.append(\"band fresh repeats share a process with a report process\")\n",
        ""),
    "digest_bind": (
        "    if hashlib.sha256(raw).hexdigest() != report_sha256:\n"
        "        raise ValueError(f\"{label} file {str(path)!r} does not hash to its digest\")\n",
        ""),
    "m5a_fresh_reuse": (
        "    if len(set(fresh_processes)) != len(fresh_processes):\n"
        "        reasons.append(\"fresh reference process was reused across rates\")\n",
        ""),
}

if name not in M:
    raise SystemExit(f"unknown mutant {name}; known: {sorted(M)}")
anchor, replacement = M[name]
count = s.count(anchor)
if count != 1:
    raise SystemExit(f"anchor for {name} matched {count} times (need exactly 1)")
p.write_text(s.replace(anchor, replacement))
print(f"mutant {name} applied")
