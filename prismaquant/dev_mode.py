"""``PRISMAQUANT_DEV_MODE``: the owner's rapid-iteration switch (2026-09-19).

Rob's decision, verbatim: "a dev mode that turns all this shit off. It buys
me nothing... It's just a massive tax right now." Pre-enterprise, every
iteration paid provenance tax at the RUN gate -- source-transition receipts,
checkpoint-lineage identity refusals, prepared-record digest equality -- with
no consumer of the certification yet. The seal returns at the ARTIFACT gate,
not the run gate.

The contract, in one paragraph:

* The switch is the environment variable ``PRISMAQUANT_DEV_MODE`` and it is
  ON exactly when its value is the string ``1``. Any other value -- unset,
  empty, ``0``, ``true`` -- is certified mode.
* Certified mode is byte-identical to the behavior before this module
  existed. Every gate that reads this module refuses under exactly the same
  fixtures with the variable off; that is proven by tests, not by intention.
* Dev mode never weakens a proof. It bypasses AT THE GATE: the check still
  runs, its findings are still computed, and what it would have refused is
  RECORDED instead -- a loud, grep-able warning line and a
  ``dev_uncertified`` stamp carrying the actual digests of what executed.
  A dev result can therefore never masquerade as a certified one: the stamp
  is top-level in results.json and in every progress record.
* Recording is not optional in dev mode. Even a dev run records what ran:
  the executing package's actual tree digest is computed and stamped. A
  record, never a gate.

Nothing in this module decides policy. It answers "is dev mode on?", builds
the stamp, and prints the warning; the gates themselves stay in the modules
that own them.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

#: The one environment variable. Read at the gates, never cached, so a
#: subprocess or a container inherits it through the environment alone.
DEV_MODE_ENV = "PRISMAQUANT_DEV_MODE"

#: The stamp's fixed top-level marker. Present in results.json and in every
#: progress record written under dev mode, absent in certified mode.
DEV_UNCERTIFIED_KEY = "dev_uncertified"
DEV_MODE_KEY = "dev_mode"


def dev_mode_enabled(environ=None) -> bool:
    """Whether dev mode is ON: ``PRISMAQUANT_DEV_MODE`` is exactly ``1``."""
    environ = os.environ if environ is None else environ
    return environ.get(DEV_MODE_ENV, "") == "1"


def dev_stamp(producer_source_sha256: str | None = None, *, timestamped: bool = True) -> dict:
    """The unmistakable mark of a dev run.

    ``{"dev_uncertified": true, "dev_mode": {...}}`` at the top level of
    whatever record accepts it. ``dev_mode`` carries the executing package's
    ACTUAL tree digest when known -- a record of what ran, never a gate --
    and, only where equality is not compared across processes (results.json,
    progress records), a UTC timestamp. Callers that compare the stamp for
    equality across a resume (unit-checkpoint provenance) pass
    ``timestamped=False`` so a resumed run reproduces the identical record.
    """
    block: dict = {DEV_MODE_ENV: "1"}
    if producer_source_sha256 is not None:
        block["producer_source_sha256"] = str(producer_source_sha256)
    if timestamped:
        block["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {DEV_UNCERTIFIED_KEY: True, DEV_MODE_KEY: block}


def dev_warning(message: str) -> None:
    """The loud line every suspended gate prints instead of refusing.

    One stable ``[DEV-MODE]`` prefix so a dev run's suspended gates are as
    grep-able as its stamps: ``grep DEV-MODE`` finds every place a certified
    run would have stopped.
    """
    print(f"[DEV-MODE] {message}", flush=True)
