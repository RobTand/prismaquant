def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "_exec-request":
        return _worker_main()
    if args.command == "sealed-stage":
        return run_sealed_stage(
            receipt_path=args.receipt,
            token=args.token,
            child_argv=_nested_argv(args.child_argv),
        )
    if args.command == "sealed-stage-receipt-sha256":
        print(
            sealed_stage_receipt_sha256(
                args.token,
                _nested_argv(args.child_argv),
            )
        )
        return 0
    if args.command == "seal-manifest":
        body = _read_json_mapping(args.body, where="manifest body")
        sealed = seal_campaign_manifest_v2(body)
        _atomic_write_new_json(args.output, sealed)
        print(args.output)
        return 0
    manifest = load_campaign_manifest_v2(args.manifest)
    if args.command == "run":
        state = run_campaign_v2(manifest, args.state)
        print(json.dumps(state, sort_keys=True))
        return 0
    with _StateStore(args.state) as store:
        state = store.load(manifest)
    if state is None:
        raise CampaignContractError(f"campaign state does not exist: {args.state}")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0
