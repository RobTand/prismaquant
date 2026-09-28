"""Tessera export drivers that run through PrismaBuild's public commands.

These moved here from PrismaBuild's ``tools/fleet/`` and ``src/prismabuild/``
on 2026-09-28 (RobTand/prismabuild#1076). PrismaBuild stands alone and names
no client; PrismaQuant may depend on both Tessera and PrismaBuild's public
interface, so the drivers that join the two live here. They submit only
through ``pbcampaign.py`` and read endings only through ``pbwait.py``, both
run as commands; nothing here imports PrismaBuild.

Run each as a module from the repository root, for example
``python3 -m tools.tessera_fleet.dispatch_model --help``.
"""
