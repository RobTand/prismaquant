def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shipcard")
    parser.add_argument("--window", help="U4 arm receipt; raw unsupported outputs remain refusals")
    parser.add_argument("--records-dir", help="producer slot records named <slot>.json")
    parser.add_argument("--apply", action="store_true", help="fill replayed records; default writes nothing")
    args = parser.parse_args(argv)
    try:
        report = ingest_receipts(args.shipcard, window=args.window,
                                 records_dir=args.records_dir, apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(DIRECT_ASCII_INDENT2_LAX.text(report))
    return 0 if report["status"] == "ready" else 1
