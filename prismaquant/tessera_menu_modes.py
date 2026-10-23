"""The Tessera menu-mode vocabulary, importable without Tessera.

Campaign planners and fleet coordinators parse ``--menu-mode`` before any
Tessera rung is named, so the spelling lives here rather than in
``tessera_menu``, which builds menus and needs the producer package.
"""
from __future__ import annotations

#: Only rungs the pinned serving release attests.  The production default.
MENU_ATTESTED = "attested"
#: Every writable, shape-legal rung, each stamped ``unattested``.  Opt-in, and
#: refused by the export gate exactly as any other unattested unit is.
MENU_RESEARCH = "research"
#: Only rungs the pinned contract's own decoder accepts: the family is
#: published and the rate lies inside its ``reader_rate_range_q256``.  Opt-in,
#: still stamped ``unattested``, and refused by the export gate exactly as any
#: other unattested unit is -- it narrows what a CAMPAIGN pays to encode, not
#: what an artifact may ship.
MENU_READABLE = "readable"
MENU_MODES = frozenset({MENU_ATTESTED, MENU_READABLE, MENU_RESEARCH})

#: The lever that selects the mode.  ONE name, read in ONE place
#: (``tessera_menu.menu_mode``), stamped into every allocation's provenance.
#: Unset is the production default, so a run that never mentions it cannot
#: silently allocate onto rungs no runtime attests.
MENU_MODE_ENV = "PRISMAQUANT_TESSERA_MENU"
