"""The recorded graph baseline keeps main 9fb507d6's minima under the current analyzer."""
import json
from pathlib import Path

from tools.ci import graph_impact as impact
from tools.ci import graph_impact_pq as pq

ROOT = Path(__file__).resolve().parents[1]


def test_pra_baseline_keeps_main_minima_under_the_current_analyzer():
    main = pq.snapshot_at(ROOT, '9fb507d69a13dc058e731f039bf445e84e4b5d3a')
    recorded = json.loads((ROOT / 'tests/fixtures/graph_impact_baseline.json').read_text())
    expected = {**main['recorded_baseline'],
                'figures': {**main['recorded_baseline']['figures'], 'scc_max': main['scc_max']},
                'test_to_test_count': len(impact.test_to_test_edges(main)),
                'analyzer_sha256': pq.file_sha256hex(ROOT / 'tools/bazel/build_files.py')}
    assert impact.baseline_errors(recorded, expected) == []
    fixed = expected.keys() - {'figures', 'reach_budget', 'test_to_test_count', 'edge_groups', 'test_to_test'}
    assert {key: recorded[key] for key in fixed} == {key: expected[key] for key in fixed}
