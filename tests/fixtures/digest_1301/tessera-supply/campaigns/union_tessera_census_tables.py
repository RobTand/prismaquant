def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", action="append", required=True, metavar="NAME=DIR",
                        help="a merged census table directory (repeat; order sets anchor order)")
    parser.add_argument("--out", required=True, help="union table directory to create")
    parser.add_argument("--dry-run", action="store_true", help="run every check, print the plan, write nothing")
    parser.add_argument("--resume", action="store_true",
                        help="continue into an existing --out whose files verify against the plan")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if (out.exists() or out.is_symlink()) and not args.resume and not args.dry_run:
            raise UnionRefused(f"--out {out} exists; pass --resume to continue into it")
        plan = build_plan(args.table, out)
        summary = dict(plan.summary)
        if args.dry_run:
            summary["dry_run"] = True
            summary["writes"] = 0
        else:
            summary["written"] = write_plan(plan, resume=args.resume)
    except UnionRefused as exc:
        print(f"union refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0
