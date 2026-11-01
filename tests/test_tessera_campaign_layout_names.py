"""The campaign table's stage and checkpoint layout names have one owner.

``tessera_campaign_table`` names the stage its journals carry and the files a
campaign row publishes. A package module that spells one of them again can be
left behind by a rename and silently read or reseal a different layout.
"""
from tests.boundary_gate import code_strings, python_files

import ast

from prismaquant import tessera_campaign_table as table

OWNER = "prismaquant/tessera_campaign_table.py"
#: ``COST_NAME`` (``cost.pkl``) is also the pipeline's cost table, so only the
#: names no other artifact shares are held here.
NAMES = {table.STAGE, table.MANIFEST_NAME, table.PARTS_NAME}


def test_only_the_campaign_table_spells_its_layout_names():
    found = sorted(
        f"{rel}: {text!r}"
        for rel, source in python_files("prismaquant") if rel != OWNER
        for _node, text in code_strings(ast.parse(source)) if text in NAMES)
    assert found == [], "import the name from tessera_campaign_table"
