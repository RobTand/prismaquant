def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write or validate PrismaQuant pipeline contracts."
    )
    ap.add_argument(
        "--write-default-production",
        metavar="PATH",
        help="Write the configured production PipelineSpec JSON to PATH.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated or loaded PipelineSpec and fail on errors.",
    )
    ap.add_argument(
        "--input",
        metavar="PATH",
        help="Validate an existing PipelineSpec JSON instead of generating one.",
    )
    ap.add_argument("--render-mechanisms", default="")
    ap.add_argument("--disable-render-mechanisms", default="")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--formats", default=None)
    ap.add_argument("--target-bits", type=float, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--calibration-modality", default=None)
    ap.add_argument("--selection-mode", default=None)
    ap.add_argument("--production-cache", default=None)
    ap.add_argument("--production-recache", default=None)
    ap.add_argument(
        "--check-frontier-materialization",
        metavar="MODEL_PATH",
        help=(
            "Fail closed when hooks materialization is requested for a MoE, "
            "a checkpoint with at least 35B parameters, or a model whose "
            "header-only classification cannot be proven."
        ),
    )
    ap.add_argument(
        "--frontier-materialization",
        metavar="MODE",
        help="Requested validated-frontier materializer: hooks or inplace.",
    )
    ap.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="K=V",
        help="A pipeline setting value. Consumed by --write-stage-settings "
             "(projected onto each artifact's declared key set) and by "
             "--check-stage-settings (late-computed overrides).",
    )
    ap.add_argument(
        "--write-stage-settings",
        metavar="PATH",
        help="Write the per-artifact settings-hash key sets, already "
             "projected onto --setting values, to PATH (re-vet R5).",
    )
    ap.add_argument(
        "--check-stage-settings",
        action="store_true",
        help="Guard one skip-if-exists artifact against its recorded "
             "settings. Requires --stage-settings/--artifact/--stage.",
    )
    ap.add_argument("--calibration-hash", nargs=5, metavar=("MODEL", "DATASET", "N", "SL", "SEED"))
    ap.add_argument("--assignment-kl")
    ap.add_argument("--assignment-path")
    ap.add_argument("--measurement-cache")
    ap.add_argument("--assignment-kl-mode", choices=("bind", "reuse"), default="reuse")
    ap.add_argument("--draw-digest")
    ap.add_argument("--held-out-calibration-skip", action="store_true")
    ap.add_argument("--calibration-draw", nargs=3, type=int, action="append", default=[])
    ap.add_argument("--stage-settings", metavar="PATH", default=None,
                    help="Path written by --write-stage-settings.")
    ap.add_argument("--artifact", metavar="PATH", default=None)
    ap.add_argument("--stage", metavar="ID", default=None)
    args = ap.parse_args(argv)

    if args.calibration_hash:
        model, source, samples, seqlen, seed = args.calibration_hash
        print(calibration_draw_digest(model, source, int(samples), int(seqlen), int(seed)))
        return 0

    if args.assignment_kl:
        try:
            document = json.loads(Path(args.stage_settings).read_text())
            measured = assignment_kl_measurement(
                args.assignment_kl, args.assignment_path, document,
                draw_digest=args.draw_digest, cache_path=args.measurement_cache,
                record=args.assignment_kl_mode == "bind")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"[pipeline] additivity measurement has no matching binding: {exc}", file=sys.stderr)
            return 0
        if measured is not None:
            print(*measured)
        return 0

    if args.held_out_calibration_skip:
        settings = parse_settings(args.setting)
        try:
            with redirect_stdout(sys.stderr):
                skip = held_out_calibration_skip(
                    source=settings["DATASET"],
                    validation_source=settings["VALIDATED_FRONTIER_DATASET"],
                    validation_seqlen=int(settings["VALIDATED_FRONTIER_SEQLEN"]),
                    draws=args.calibration_draw,
                    requested_skip=(int(settings["VALIDATED_FRONTIER_SKIP_CALIB"])
                                    if settings.get("VALIDATED_FRONTIER_SKIP_CALIB") else None),
                    enforce=settings["SELECTION_MODE"] == "validated-surrogate",
                    tokenizer=(_calibration_tokenizer(settings["MODEL_PATH"])
                               if (settings["SELECTION_MODE"] == "validated-surrogate"
                                   and settings.get("VALIDATED_FRONTIER_SKIP_CALIB") != "0") else None),
                    validation_samples=(int(settings["VALIDATED_FRONTIER_NSAMPLES"])
                                        * int(settings["VALIDATED_FRONTIER_CALIB_REPEATS"])),
                )
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
            return 2
        print(skip)
        return 0

    if args.check_frontier_materialization:
        if args.frontier_materialization is None:
            print(
                "[pipeline] ERROR: --check-frontier-materialization needs "
                "--frontier-materialization"
            )
            return 2
        code, message = check_frontier_materialization(
            args.check_frontier_materialization,
            args.frontier_materialization,
        )
        prefix = "[pipeline]" if code == 0 else "[pipeline] ERROR:"
        print(f"{prefix} {message}")
        return code

    if args.check_stage_settings:
        if not (args.stage_settings and args.artifact and args.stage):
            print("[pipeline] ERROR: --check-stage-settings needs "
                  "--stage-settings, --artifact and --stage")
            return 2
        document = json.loads(Path(args.stage_settings).read_text())
        code, messages = check_stage_settings(
            args.artifact,
            args.stage,
            document,
            overrides=parse_settings(args.setting),
        )
        for message in messages:
            print(message, flush=True)
        return code

    if args.write_stage_settings:
        document = stage_settings_document(parse_settings(args.setting))
        out = Path(args.write_stage_settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(f"[pipeline-spec] wrote {out}")
        return 0

    if args.input:
        spec = load_pipeline_spec(args.input)
    else:
        spec = production_pipeline_spec_from_config(
            render_mechanisms=args.render_mechanisms,
            disabled_render_mechanisms=args.disable_render_mechanisms,
            model_path=args.model_path,
            work_dir=args.work_dir,
            formats=args.formats,
            target_bits=args.target_bits,
            target_profile=args.target_profile,
            calibration_modality=args.calibration_modality,
            selection_mode=args.selection_mode,
            production_cache=args.production_cache,
            production_recache=args.production_recache,
        )

    validation = spec.validate()
    if args.validate and validation.errors:
        for error in validation.errors:
            print(f"[pipeline-spec] ERROR: {error}")
        return 2
    for warning in validation.warnings:
        print(f"[pipeline-spec] WARN: {warning}")

    if args.write_default_production:
        write_pipeline_spec(spec, args.write_default_production)
        print(f"[pipeline-spec] wrote {args.write_default_production}")
    elif not args.input:
        print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    elif validation.ok:
        print(f"[pipeline-spec] valid: {args.input}")
    return 0
