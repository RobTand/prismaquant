    payload = layer_config_payload(result, meta=meta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for value in result.assignment.values():
        counts[value] = counts.get(value, 0) + 1
    reasons: dict[str, int] = {}
    for reason in result.excluded.values():
        key = reason.split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1

    print(f"[uniform] units={result.unit_count} "
          f"assigned={len(result.assignment)} omitted={len(result.excluded)}")
    for value, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {value:>14}: {n:>5} Linears")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  omitted [{reason}]: {n}")
    if result.demoted_units:
        print(f"[uniform] {len(result.demoted_units)} unit(s) demoted to the "
              f"fallback (sample: {list(result.demoted_units)[:4]})")
    print(f"[uniform] achieved_bits={achieved:.4f} bpp over "
          f"{sum(result.params_by_format.values()):,} quantizable params")
    print(f"[uniform] layer_config -> {out}")

    if args.report:
        report = {
            "meta": meta,
            "unit_count": result.unit_count,
            "assigned": len(result.assignment),
            "format_counts": counts,
            "params_by_format": result.params_by_format,
            "achieved_bits": achieved,
            "demoted_units": list(result.demoted_units),
            "excluded": result.excluded,
        }
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[uniform] report -> {rp}")
    return 0
