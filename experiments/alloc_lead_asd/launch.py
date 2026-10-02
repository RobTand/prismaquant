"""Launch a bounded diagnostic through the existing admitted container adapter.

This is a child of a PrismaBuild action, never a submission or scheduler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.tessera_campaign_container import main as container_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--memory-gb", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    recipe = Path(__file__).with_name("recipes") / "container.json"
    spec = json.loads(recipe.read_text())
    spec["cpu_memory_gb"] = args.memory_gb
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return container_main(["--spec", json.dumps(spec),
                           *(["--cpu-only"] if args.cpu_only else []), "--", *command])


if __name__ == "__main__":
    raise SystemExit(main())
