#!/usr/bin/env python3
"""Stock `vllm serve --headless` peer for an in-process gold driver on rank 0.

A multi-node gold measurement runs the measuring Python process as rank 0 and a
stock headless vLLM as every other rank. That peer cannot always be the stock
`/usr/local/bin/vllm` console script, for a reason that costs hours when it is
discovered live rather than read here.

vLLM **spawns** its workers, and a spawned worker re-imports the **launching
file** as `__mp_main__`. A driver that ships callables to its workers --
`llm.apply_model(...)`, `llm.collective_rpc(...)` -- pickles them as
`__main__.<name>`, because the driver itself is `__main__`. Under the stock
console script those names do not exist in the peer's `__mp_main__`, so rank 1
dies with `AttributeError: Can't get attribute 'install_capture'` and **rank 0
then hangs forever with no timeout** (2026-09-14, GLM-5.3 4-layer stub; the
first attempt was stopped by hand after 30 minutes). Launching the stock CLI
from a file that has first imported the driver module puts the driver's names
into `__main__`/`__mp_main__` and changes nothing else about the engine.

The same launch also needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on **both**
ranks, or the mp executor's msgpack refuses a `functools.partial` before any of
the above matters. This file does not set it: an environment variable that
changes how a serving process deserializes is the operator's declaration, and
silently exporting it here would hide it from the receipt.

`tools/measure_vllm_full_kl.py` and `tools/measure_vllm_wikitext_ppl.py` ship
no callables -- they only call `llm.generate()` -- so for those two the
namespace copy is inert and the peer is simply a stock serve. It is kept
uniform anyway: one peer launcher that is correct for every gold driver beats
two, one of which is silently wrong for the driver that grows an
`apply_model` call later.

Usage (the driver module is named, never guessed):

    PQ_GOLD_PEER_DRIVER=experiments.measure_glm_tr3_vllm \\
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
        python3 tools/gold_headless_peer.py serve MODEL --node-rank 1 \\
            --headless --tensor-parallel-size 2 ...

Everything after the script name is handed to the stock vLLM CLI unchanged, so
the peer's engine arguments are the stock ones. Derive them from rank 0's own
engine kwargs with `gold_engine_options.headless_peer_argv` rather than
retyping them: a peer that joins with different engine arguments than the
coordinator built is a measurement of neither configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable naming the driver module whose namespace the peer must
#: carry. Required: a default would silently produce a peer that is correct for
#: one driver and hangs rank 0 for another.
DRIVER_ENV = "PQ_GOLD_PEER_DRIVER"

#: Optional override for the PrismaQuant checkout the driver is imported from.
#: The default is this file's own repository root, so a peer launched from a
#: checkout imports that checkout -- the driver source must be present at the
#: same import path on the peer box.
ROOT_ENV = "PQ_GOLD_PEER_ROOT"


def repository_root() -> Path:
    """The checkout this peer imports its driver from."""
    override = os.environ.get(ROOT_ENV)
    if override:
        root = Path(override)
        if not root.is_absolute():
            raise ValueError(f"{ROOT_ENV} must be an absolute path: {override!r}")
        return root
    return Path(__file__).resolve().parents[1]


def load_driver_namespace(
    driver: str,
    *,
    root: Path,
    namespace: dict,
) -> list[str]:
    """Import `driver` and copy its public names into `namespace`.

    Returns the names copied, so a caller (and the test) can prove the copy
    happened rather than trusting that the import did something. Dunder names
    are excluded: overwriting the launching module's `__name__`/`__file__`
    would break the very `__mp_main__` identity this exists to establish.
    """
    import importlib

    for entry in (str(root), str(root / "tools")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    module = importlib.import_module(driver)
    copied = [name for name in vars(module) if not name.startswith("__")]
    namespace.update({name: vars(module)[name] for name in copied})
    return copied


#: Loaded at MODULE scope, and that placement is the whole mechanism.
#
# vLLM spawns its workers, and a spawned worker re-imports this file with
# `run_name="__mp_main__"`: top-level code runs, `main()` does NOT, and neither
# does anything under `if __name__ == "__main__"`. Loading the driver inside
# `main()` would put its names in the launcher's `__main__` and leave the
# worker's `__mp_main__` bare -- which is exactly the
# `AttributeError: Can't get attribute 'install_capture' on <module
# '__mp_main__'>` this file exists to prevent, with rank 0 then hanging on the
# dead rank forever. `tools/../tmp` prototype `tr3_headless_peer.py` has the
# import at top level for the same reason.
#
# Gated on the variable so that merely importing this module (a test, a doc
# tool) stays inert. The spawned child inherits the parent's environment, so
# the load fires again in `__mp_main__` exactly when it fired in `__main__`.
_DRIVER = os.environ.get(DRIVER_ENV, "").strip()
if _DRIVER:
    load_driver_namespace(_DRIVER, root=repository_root(), namespace=globals())


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    driver = os.environ.get(DRIVER_ENV, "").strip()
    if not driver:
        raise SystemExit(
            f"{DRIVER_ENV} must name the gold driver module whose callables "
            "this peer has to be able to unpickle (for example "
            "experiments.measure_glm_tr3_vllm). Without it the peer is a bare "
            "stock serve, and a driver that ships callables makes rank 1 die "
            "and rank 0 hang forever with no timeout."
        )
    if not arguments:
        raise SystemExit(
            "pass the stock vLLM serve arguments for this rank, for example: "
            "serve MODEL --node-rank 1 --headless --tensor-parallel-size 2"
        )
    load_driver_namespace(driver, root=repository_root(), namespace=globals())

    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", *arguments]
    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
