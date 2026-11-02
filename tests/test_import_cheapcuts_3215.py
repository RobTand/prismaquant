"""Ranked cheap cuts must remove imports even inside function bodies."""
from pathlib import Path

import pytest

from import_graph_support import _modules, _targets, _lazy_exports

ROOT = Path(__file__).resolve().parents[1]
CUTS = [
    ('prismaquant.shipcard', 'tools.serve_fingerprint'),
    ('prismaquant.tessera_lane', 'prismaquant.tessera_materialization'),
    ('tools.pq_row_profile_observer', 'tools.pq_admitted_profile'),
    ('prismaquant.prismabuild_progress', 'prismaquant.aura_provenance'),
    ('tools.tessera_fleet.dispatch_model', 'prismaquant.tessera_export_lane'),
    ('prismaquant.name_projection', 'prismaquant.decision_units'),
    ('prismaquant.stage_b_prep_io', 'prismaquant.joint_layer_quanta'),
    ('prismaquant.joint_head_walk_quanta', 'prismaquant.tessera_joint_aura'),
]


@pytest.mark.parametrize('source,target', CUTS)
def test_ranked_import_edge_is_gone(source, target):
    modules = _modules()
    for name in (source, target):
        modules[name] = ROOT / (name.replace('.', '/') + '.py')
    assert target not in _targets(source, modules[source], modules, _lazy_exports(modules))


def test_edge_oracle_counts_nested_and_package_imports(tmp_path):
    path = tmp_path / 'caller.py'
    path.write_text('def call():\n    from . import upstream\n')
    modules = {'package.caller': path, 'package.upstream': tmp_path / 'upstream.py'}
    assert 'package.upstream' in _targets('package.caller', path, modules, {})


def test_gold_source_roster_still_accepts_published_producer_identities():
    from tools.serve_fingerprint import _GOLD_PRODUCER_COMMON_FILES

    assert _GOLD_PRODUCER_COMMON_FILES == (
        'prismaquant/tessera_runtime/tessera_serving_runtime_pin.json',
        'tools/dsv4_wikitext_inputs.py',
        'tools/prepare_dsv4_wikitext_inputs.py',
        'tools/prismaquant_source_bootstrap.py',
        'tools/serve_fingerprint.py',
        'tools/spec_decode_guard.py',
    )
