"""Run the source-owned bounded profiler without an inline full-run wrapper."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from bounded_profile import profile_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("entry")
    parser.add_argument("entry_args", help="Pass the unchanged entry arguments as a JSON list")
    parser.add_argument("expected_device", help="Pass cpu or cuda")
    args = parser.parse_args()
    profile_run(args.output, args.entry, json.loads(args.entry_args), args.expected_device)


if __name__ == "__main__":
    main()
