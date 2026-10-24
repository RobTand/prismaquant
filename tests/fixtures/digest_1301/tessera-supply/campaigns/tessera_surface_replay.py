def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", required=True, help="trusted campaign cost pickle")
    parser.add_argument("--checkpoint", required=True, help="existing campaign checkpoint manifest")
    parser.add_argument("--plan", required=True, help="explicit JSON replay plan")
    parser.add_argument("--out", required=True, help="research JSON report")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(Path(args.plan).read_text())
        report = replay_campaign(args.costs, args.checkpoint, plan)
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        output = Path(args.out)
        destination = output.parent.resolve(strict=False) / output.name
        created = publish_new_bytes(destination, encoded, nofollow=True)
        if not created and destination.read_bytes() != encoded:
            raise ValueError("output already contains a different replay; use another --out")
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"replay refused: {exc}\n")
    return 0
