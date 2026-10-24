def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if os.environ.get(_INTERNAL) != "1":
            _outer_reexec(raw_argv)
            raise AssertionError("execve returned")
        args = _parser().parse_args(raw_argv)
        result = _inner_main(args)
    except (OSError, WaiterError, RuntimeError) as exc:
        print(f"dsv4-campaign-completion: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0
