"""Conditional byte/launch budgets, not measurements of a mixed decoder.

Reuse the retained NCU intake. This is a CPU PrismaBuild action. No kernels,
allocation, calibration capture, serving change, or speed-neutrality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-json", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    intake_bytes = args.intake_json.read_bytes()
    intake = json.loads(intake_bytes)
    table = json.loads(args.table.read_bytes())
    rows = [row for row in table["rungs"] if row["rung"] == 1024]
    if len(rows) != 1:
        raise ValueError("table must contain exactly one baseline row")
    baseline = rows[0]
    # Tessera b7e62b6 routed_fused.py: BN=128, BK=32, BDESC_INTS=12,
    # run_pair: four int32s per wire run, 512-row tile per packed column.
    # Hypothetical extensions retain those dimensions; they are NOT a wire.
    bn, bk, tile_rows, body_bits = 128, 32, 512, 4
    descriptor_bytes, wire_run_bytes = 12 * 4, 4 * 4
    issue_body_bytes = bn * bk * body_bits // 8
    tile_body_bytes = tile_rows * bk * body_bits // 8
    # Prior fixed-cost report: 21 setup launches total 0.16 ms and 0.09 ms
    # gaps. Average is a proxy, not a measured launch latency or an upper bound.
    setup_and_gap_ms, setup_launches, routed_pair_ms = 0.25, 21, 11.70
    launch_proxy_ms = setup_and_gap_ms / setup_launches
    estimates = []
    for classes in (2, 3):
        # Two matrices (concatenated gate/up and down); c compute launches each,
        # one reduction each. Uniform baseline has two compute launches.
        extra_launches = 2 * classes
        estimates.append({
            "format_classes": classes,
            "sorted_runs_per_tile_max": classes,
            "run_switches_per_tile_max": classes - 1,
            "hypothetical_per_tile_run_metadata_bytes": classes * wire_run_bytes,
            "extra_run_metadata_bytes_vs_one_run": (classes - 1) * wire_run_bytes,
            "extra_run_metadata_bpp_if_per_issue_tile":
                (classes - 1) * wire_run_bytes * 8 / (bn * bk),
            "split_class_extra_launches": extra_launches,
            "split_class_launch_proxy_ms": extra_launches * launch_proxy_ms,
            "split_class_launch_proxy_percent_of_M256_pair":
                100 * extra_launches * launch_proxy_ms / routed_pair_ms,
            "M1_M16_split_class_total_time": None,
            "M2048_M4096_split_class_total_time": None,
            "missing_costs": ["class-submatrix GEMMs", "partial-output traffic",
                              "reduction", "new packed reader", "occupancy changes"],
        })
    observed = []
    for cell in baseline["measurements"]:
        observed.append({key: cell.get(key) for key in (
            "cell_id", "kernel_kind", "shape_id", "M", "measurement_status",
            "kernel_time_us", "kernel_path")})
    for estimate in estimates:
        estimate["split_launch_proxy_by_M"] = []
        for m in (1, 16, 2048, 4096):
            cells = [cell for cell in observed
                     if cell["kernel_kind"] == "routed" and cell["M"] == m]
            if len(cells) != 2 or any(cell["kernel_time_us"] is None for cell in cells):
                raise ValueError(f"missing routed gate/up and down observations for M{m}")
            uniform_us = sum(cell["kernel_time_us"] for cell in cells)
            proxy_us = estimate["split_class_launch_proxy_ms"] * 1000
            estimate["split_launch_proxy_by_M"].append({
                "M": m, "stored_uniform_pair_us": uniform_us,
                "extra_launch_proxy_us": proxy_us,
                "extra_launch_proxy_percent": 100 * proxy_us / uniform_us,
                "total_mixed_decoder_time_measured": False,
                "M256_average_launch_proxy_assumed_to_transfer": True})
    result = {
        "schema": "prismaquant.block_decode_estimate.v1",
        "intake_path": str(args.intake_json),
        "intake_sha256": hashlib.sha256(intake_bytes).hexdigest(),
        "table_path": str(args.table), "table_version": table["table_version"],
        "baseline_rung_status": baseline["measurement_status"],
        "baseline_observations_not_allocation_admission": observed,
        "retained_ncu_uniform_kernels": intake["profile"]["kernels"],
        "geometry": {"BN": bn, "BK": bk, "packed_tile_rows": tile_rows,
                     "body_bits_per_weight": body_bits,
                     "body_bytes_per_issued_projection_tile": issue_body_bytes,
                     "body_bytes_per_packed_512_by_32_tile": tile_body_bytes},
        "existing_two_rate_descriptor": {
            "bytes_per_32_columns": descriptor_bytes,
            "percent_of_issued_body_bytes": 100 * descriptor_bytes / issue_body_bytes,
            "percent_of_stored_512_by_32_body_bytes": 100 * descriptor_bytes / tile_body_bytes,
            "resident_metadata_bpp": descriptor_bytes * 8 / (tile_rows * bk),
            "not_a_measured_runtime_penalty": True},
        "hypothetical_256_weight_restart": {
            "window_state_bits": 14, "state_bpp": 14 / 256,
            "percent_of_4bpp_body": 100 * 14 / (256 * body_bits),
            "condition": "one independently addressed column stream per 256 weights; a fresh zero-state encode instead changes the quantization"},
        "class_split_estimates": estimates,
        "decode_once_actual_scope": {
            "at_load_not_per_chunk": True, "resident_dense_shared_only": True,
            "eager_only": True, "MIN_M": 256,
            "extra_resident_bytes_per_weight": 1,
            "extra_scale_bytes_per_output_row": 4,
            "heterogeneous_decode_overhead_per_forward_after_load": 0,
            "condition": "all candidates use the same E4M3 bytes plus row-scale epilogue and are decoded during load",
            "routed_extension_measured": False},
        "same_instruction_stream_mix": {
            "ideal_added_inner_loop_instructions": 0,
            "condition": "equal stored width, one prepared instruction stream, table selection hoisted out of decode loop",
            "different_width_rungs_satisfy_condition": False,
            "table_bandwidth_and_smem_penalty_measured": False},
        "permutation": {
            "ideal_added_forward_launches_if_fully_folded": 0,
            "constraints": ["one consistent permutation at every consumer of a shared activation",
                            "gate and up output permutations agree before SwiGLU",
                            "down input columns carry the matching permutation",
                            "per-expert input permutations cannot all be folded into one shared norm output",
                            "two-dimensional block schedules need not admit a single channel permutation"],
            "real_arithmetic_exact": True,
            "floating_point_bit_exact_or_speed_neutral_measured": False},
        "warp_specialization": {
            "ideal_equal_work_overhead": 0,
            "required_producer_counts": "proportional to block_count[class] * measured_decode_cost[class]",
            "actual_decode_costs_by_mixed_class": None,
            "existing_consumer_wait_fraction_supplier_report": [0.59, 0.77],
            "warning": "SASS warp samples are not a wall-time fraction; assigning more producers has already failed in the prior experiment"},
        "quality_gate_measured": False,
        "speed_neutral_combination_qualified": False,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
