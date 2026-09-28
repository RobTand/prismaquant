"""Does a Tessera checkout's exporter take PrismaQuant's reuse authority?

Tessera contract v40 publishes, as data, which export drivers accept
``--producer-authority`` (``producer_interface.reuse_authority``, tessera#599).
This module reads that block from a checkout's packaged contract and answers
with the argv to add, which is empty for a pin that predates the option.

It imports only the standard library. ``run-pipeline.sh`` runs it by path
(``python3 -c 'import runpy ...'``), so one JSON read does not first import the
``prismaquant`` package, and with it torch and transformers.
:mod:`prismaquant.tessera_export_lane` re-exports every name here, and the
dispatcher reaches it through that module.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ProducerInterfaceError(RuntimeError):
    """A checkout's producer interface cannot be read.  Always actionable."""


#: The exporter every PrismaQuant export argv names, relative to its checkout.
EXPORTER_DRIVER = "experiments/export_tessera_serving.py"

#: The option the reuse-authority seam adds (tessera#599), as the contract
#: spells it. Read back from the contract below; never passed on this alone.
PRODUCER_AUTHORITY_OPTION = "--producer-authority"

#: Where a Tessera checkout packages its contract, under ``src/`` or flat --
#: the same two layouts ``tessera_export_lane.require_producer_repo_is_pinned``
#: accepts.
_CHECKOUT_CONTRACT = Path("tessera", "serving", "runtime_contract.json")


def checkout_contract_path(tessera_checkout) -> Path:
    """The ``runtime_contract.json`` a Tessera checkout packages; absent refuses."""
    root = Path(tessera_checkout)
    for candidate in (root / "src" / _CHECKOUT_CONTRACT, root / _CHECKOUT_CONTRACT):
        if candidate.is_file():
            return candidate
    raise ProducerInterfaceError(
        f"{root} packages no {_CHECKOUT_CONTRACT}, so whether its export "
        "drivers take --producer-authority cannot be read from it")


def advertises_producer_authority(contract: Mapping[str, Any],
                                  driver: str = EXPORTER_DRIVER) -> bool:
    """Does this Tessera contract attest that ``driver`` takes the option?

    Principle 14: the capability is READ from the pinned runtime's contract
    (``producer_interface.reuse_authority``, Tessera contract v40), never
    asserted here. A contract with no ``producer_interface`` block predates
    the option, and its drivers would refuse it as an unknown argument, so
    the answer is no. A block that is present but does not name the option in
    the shape v40 publishes refuses: it is not a contract this reader knows.
    """
    block = contract.get("producer_interface")
    if block is None:
        return False
    reuse = block.get("reuse_authority") if isinstance(block, Mapping) else None
    drivers = reuse.get("drivers") if isinstance(reuse, Mapping) else None
    if (not isinstance(reuse, Mapping)
            or reuse.get("option") != PRODUCER_AUTHORITY_OPTION
            or not isinstance(drivers, list)
            or not all(isinstance(item, str) for item in drivers)):
        raise ProducerInterfaceError(
            "the Tessera contract's producer_interface block does not publish "
            f"reuse_authority.option={PRODUCER_AUTHORITY_OPTION!r} with a "
            "drivers list; this reader cannot tell whether the exporter takes "
            "the producer authority")
    return driver in drivers


def producer_authority_argv(tessera_checkout, authority_path,
                            driver: str = EXPORTER_DRIVER) -> list[str]:
    """``[--producer-authority, <path>]`` when the checkout attests it, else ``[]``.

    Every PrismaQuant export argv goes through this, so a Tessera pin that
    predates the option is handed the argv it was always handed, byte for
    byte, and a pin that publishes it is handed PrismaQuant's reuse
    authority (``tessera_reuse_authority.py``).
    """
    contract = json.loads(checkout_contract_path(tessera_checkout).read_text())
    if not advertises_producer_authority(contract, driver):
        return []
    return [PRODUCER_AUTHORITY_OPTION, str(authority_path)]


def main(argv: list[str] | None = None) -> int:
    """Print the argv to add, one item per line; exit 2 when it cannot be read.

    ``argv`` is ``[tessera_checkout, authority_path]``.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: tessera_producer_interface.py TESSERA_CHECKOUT AUTHORITY_PATH",
              file=sys.stderr)
        return 2
    try:
        extra = producer_authority_argv(args[0], args[1])
    except (ProducerInterfaceError, OSError, ValueError) as error:
        print(f"[pipeline] ERROR: {error}", file=sys.stderr)
        return 2
    if extra:
        print("\n".join(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
