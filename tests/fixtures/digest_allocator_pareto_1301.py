# Independent original consumer block, verbatim from prismaquant/allocator.py
# source commit: 6d8288adc94b3c4b624c143ed372fb6513f32784
# Execute with the same CPU-only supplied namespace as the revised consumer.
if args.pareto_output_dir:
        out_dir = Path(args.pareto_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        seen_payloads: set[str] = set()
        for idx, record in enumerate(pareto_seed_records):
            assignment = record["assignment"]
            digest_src = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(digest_src.encode()).hexdigest()[:12]
            if digest in seen_payloads:
                continue
            seen_payloads.add(digest)
            label = (
                f"allocator_target_{record['target_bits']:.4f}"
                f"_achieved_{record['achieved_bits']:.4f}_{digest}"
            ).replace(".", "p")
            path = out_dir / f"{label}.json"
            record_budget_stamp = None
            if args.target_disk_gb is not None:
                record_budget_bytes = int(math.floor(
                    float(args.target_disk_gb) * 1_000_000_000.0
                ))
                upper = record.get("whole_artifact_upper_bound_bytes")
                payload_bytes = record.get("artifact_tensor_payload_bytes")
                if (
                    isinstance(upper, int)
                    and isinstance(payload_bytes, int)
                    and upper <= record_budget_bytes
                ):
                    record_budget_stamp = whole_artifact_budget_stamp(
                        budget_bytes=record_budget_bytes,
                        selection_tensor_payload_bytes=payload_bytes,
                        selection_non_tensor_reserve_bytes=int(
                            args.artifact_overhead_reserve_bytes
                        ),
                        selection_assignment=assignment,
                        excluded_source_prefixes=(
                            getattr(args, "exclude_source_prefix", None) or ()
                        ),
                    )
            payload = {
                "schema": "prismaquant.allocator.pareto_assignment.v2",
                "label": label,
                "source": "allocator_pareto",
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "bits_total_with_aux": float(record["bits_total_with_aux"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "total_predicted_dloss_with_aux": float(
                    record["total_predicted_dloss_with_aux"]
                ),
                "format_counts": record["format_counts"],
                "artifact_bytes": record.get("artifact_bytes"),
                "artifact_byte_scope": record.get("artifact_byte_scope"),
                "artifact_tensor_payload_bytes": record.get(
                    "artifact_tensor_payload_bytes"
                ),
                "artifact_tensor_payload_scope": record.get(
                    "artifact_tensor_payload_scope"
                ),
                "whole_artifact_upper_bound_bytes": record.get(
                    "whole_artifact_upper_bound_bytes"
                ),
                "target_profile": target_profile,
                "assignment": dict(sorted(assignment.items())),
                **({
                    "whole_artifact_budget": record_budget_stamp,
                } if record_budget_stamp is not None else {}),
                **({"serve_constraints": _serve_feasibility(assignment).as_dict()}
                   if measured_runtime_table is not None else {}),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            manifest_rows.append({
                "label": label,
                "path": str(path),
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "bits_total_with_aux": float(record["bits_total_with_aux"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "total_predicted_dloss_with_aux": float(
                    record["total_predicted_dloss_with_aux"]
                ),
                "variable_predicted_dloss": float(record["variable_predicted_dloss"]),
                "aux_fixed_predicted_dloss": float(
                    record["aux_fixed_predicted_dloss"]
                ),
                "fixed_predicted_dloss": float(record["fixed_predicted_dloss"]),
                "format_counts": record["format_counts"],
                "artifact_bytes": record.get("artifact_bytes"),
                "artifact_byte_scope": record.get("artifact_byte_scope"),
                "artifact_tensor_payload_bytes": record.get(
                    "artifact_tensor_payload_bytes"
                ),
                "whole_artifact_upper_bound_bytes": record.get(
                    "whole_artifact_upper_bound_bytes"
                ),
            })
        (out_dir / "manifest.json").write_text(json.dumps({
            "schema": "prismaquant.allocator.pareto_manifest.v1",
            "probe": str(args.probe),
            "costs": str(args.costs),
            "target_profile": target_profile,
            "target_disk_gb": (float(args.target_disk_gb)
                               if args.target_disk_gb is not None else None),
            "artifact_overhead_reserve_bytes": (
                int(args.artifact_overhead_reserve_bytes)
                if args.artifact_overhead_reserve_bytes is not None
                else None
            ),
            "formats": [s.name for s in specs_sorted],
            "target_bits": [float(x) for x in targets],
            "knees": knee_summary,
            "candidates": manifest_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(
            f"[alloc] Pareto seed assignments → {out_dir} "
            f"({len(manifest_rows)} unique)",
            flush=True,
        )
