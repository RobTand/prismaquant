    manifest = {
        "base_assignment": str(args.base_assignment),
        "probe": str(args.probe),
        "costs": str(args.costs),
        "base_bpp": float(base_bpp),
        "categories": categories,
        "top_groups_per_category": int(args.top_groups_per_category),
        "variants": variants,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    for variant in variants:
        print(
            f"{variant['label']}: changed={variant['non_bf16_count']} "
            f"bpp_delta={variant['bpp_delta']:.6f} "
            f"pred_saved={variant['predicted_dloss_saved']:.6g}"
        )
    return 0
