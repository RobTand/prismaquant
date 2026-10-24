def main(argv=None):
    args = _capture_arguments(sys.argv[1:] if argv is None else argv)
    summary = {"final-hidden": final_hidden_phase, "projection": projection_phase,
               "capture": capture_phase}[args.phase](args)
    raw = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = Path(args.out) / f"{args.phase}-run.json"
    report.write_text(raw)
    print(raw, flush=True)
    return 0
