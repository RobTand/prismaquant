"""Activation-cache readers and schemas must not load the export cycle (#3215)."""
from __future__ import annotations

import pytest

from test_import_cycle_wave1_3215 import graph, _reach  # noqa: F401


@pytest.mark.parametrize("lower,higher", [
    ("prismaquant.activation_index", "prismaquant.measure_quant_cost"),
    ("prismaquant.activation_index", "prismaquant.export_batched_gptq"),
    ("prismaquant.schemas", "prismaquant.format_registry"),
])
def test_cache_and_schema_readers_do_not_load_export_code(graph, lower, higher):
    assert lower in graph and higher in graph
    assert higher not in _reach(graph, lower)


def test_the_graph_still_counts_the_registry_export_dependency(graph):
    assert "prismaquant.nvfp4_activation_contract" in graph["prismaquant.format_registry"]
