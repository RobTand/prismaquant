def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-action-records", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--terminals", type=Path, default=DEFAULT_TERMINALS)
    args = parser.parse_args()
    print(json.dumps(derive(args.root_action_records, args.terminals), indent=2, sort_keys=True))
    return 0
