#!/usr/bin/env python3
"""Print PrismaQuant's reviewed PrismaBuild commit without importing PrismaQuant.

pbtest runs every ``tools/resolve_<module>_dev_pin.py`` inside each shard and
refuses the shard before pytest starts when the interpreter's installed
``prismabuild`` is not a non-editable Git install at this commit
(``pbtest_pins.verify_install``). Without this resolver a stale interpreter
ran the whole suite and failed 289 tests one by one at
``staged_lease.inject_installed_sdk_for_tests`` (PQ #1929, 2026-10-01).
"""
from __future__ import annotations

from pathlib import Path
import runpy

# Resolve the existing parser by its adjacent source file. Safe-path and
# isolated script execution intentionally omit this directory from sys.path.
resolve_literal_pin = runpy.run_path(
    str(Path(__file__).resolve().with_name("resolve_tessera_dev_pin.py"))
)["resolve_literal_pin"]

PIN_NAME = "PB_READER_LEASE_PIN_COMMIT"
PIN_SOURCE = Path(__file__).resolve().parents[1] / "prismaquant" / "staged_lease.py"


if __name__ == "__main__":
    print(resolve_literal_pin(PIN_SOURCE, PIN_NAME))
