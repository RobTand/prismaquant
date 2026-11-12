"""Changed files must select every known dependent or the full population."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_select', ROOT / 'tools/ci/gate_select.py')
select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select)


def project(root):
    sources = {
        'prismaquant/owner.py': 'VALUE = 1\n',
        'prismaquant/middle.py': 'from . import owner\n',
        'tests/test_direct.py': 'from prismaquant import owner\n',
        'tests/test_indirect.py': 'import prismaquant.middle\n',
        'tests/test_unrelated.py': 'def test_value(): pass\n',
        'tests/helper.py': 'from prismaquant import owner\n',
        'tests/test_helper.py': 'from tests import helper\n',
        'tests/test_path.py': 'from pathlib import Path\np = Path("tools") / "launch.py"\n',
        'tools/launch.py': 'VALUE = 1\n',
    }
    for path, source in sources.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return root


def plan(root, changed, **options):
    return select.create_plan(root, changed, durations={'files': {}},
                              quarantine={'entries': []}, **options)


def selected(result):
    return {row['path'] for row in result['selected']}


def test_reverse_imports_relative_imports_and_helper_closure(tmp_path):
    result = plan(project(tmp_path), ['prismaquant/owner.py'])
    assert selected(result) == {'tests/test_direct.py', 'tests/test_indirect.py', 'tests/test_helper.py'}


def test_changed_test_and_literal_path(tmp_path):
    assert selected(plan(project(tmp_path), ['tools/launch.py', 'tests/test_unrelated.py'])) == {
        'tests/test_path.py', 'tests/test_unrelated.py'}


@pytest.mark.parametrize('path', [
    'conftest.py', 'tests/nested/conftest.py', 'pytest.ini', 'pyproject.toml',
    '.github/ci-constraints.txt', '.github/workflows/ci.yml', '.github/workflows/gate.yml',
    'tools/ci/gate_run.py', 'tools/resolve_tessera_dev_pin.py',
    'prismaquant/tessera_runtime/tessera_serving_runtime_pin.json',
    'prismaquant/tessera_runtime_contract.py',
    'prismaquant/new_unknown.py', 'assets/new-model.json',
])
def test_uncertain_changes_select_full_population(tmp_path, path):
    root = project(tmp_path)
    assert selected(plan(root, [path])) == selected(plan(root, [], mode='full'))


def test_syntax_error_falls_back_to_full(tmp_path):
    root = project(tmp_path)
    (root / 'prismaquant/middle.py').write_text('not valid python !!!')
    assert selected(plan(root, ['prismaquant/owner.py'])) == selected(plan(root, [], mode='full'))


def test_quarantine_ignores_files_and_deselects_nodes(tmp_path):
    root = project(tmp_path)
    quarantine = {'entries': [
        {'target': 'tests/test_path.py', 'issue': 3200, 'reason': 'missing hosted dependency'},
        {'target': 'tests/test_direct.py::test_bad', 'issue': 3201, 'reason': 'known failure'},
    ]}
    result = select.create_plan(root, [], mode='full', durations={'files': {}}, quarantine=quarantine)
    assert 'tests/test_path.py' not in selected(result)
    assert result['deselect'] == ['tests/test_direct.py::test_bad']


@pytest.mark.parametrize('issue', [None, 0, True, 3180])
def test_quarantine_requires_a_specific_owner(tmp_path, issue):
    with pytest.raises(ValueError):
        select.create_plan(project(tmp_path), [], mode='full', durations={'files': {}},
            quarantine={'entries': [{'target': 'tests/test_direct.py', 'issue': issue, 'reason': 'failure'}]})


def test_docs_only_can_select_nothing(tmp_path):
    assert selected(plan(project(tmp_path), ['docs/guide.md'])) == set()


def test_bare_wildcard_does_not_explain_an_unknown_file(tmp_path):
    root = project(tmp_path)
    (root / 'tests/test_wildcard.py').write_text('from pathlib import Path\np = Path("**")\n')
    index = select.dependency_index(root)
    index['tests/test_wildcard.py'] = (set(), {'**'})
    assert selected(plan(root, ['totally/unknown/place/file.xyz'], index=index)) == selected(plan(root, [], mode='full'))


def test_model_profile_spec_selects_all_profile_references(tmp_path):
    root = project(tmp_path)
    (root / 'tests/test_profile.py').write_text('from prismaquant.model_profiles import registry\n')
    (root / 'tests/test_profile_path.py').write_text('from pathlib import Path\np = Path("prismaquant/model_profiles")\n')
    index = select.dependency_index(root)
    index['tests/test_wildcard.py'] = (set(), {'**'})
    assert {'tests/test_profile.py', 'tests/test_profile_path.py'} <= selected(
        plan(root, ['prismaquant/model_profiles/specs/qwen3.json'], index=index))


def test_nonexistent_literal_in_a_test_does_not_explain_a_repository_change(tmp_path):
    root = project(tmp_path)
    unknown = 'totally/unknown/place/file.xyz'
    index = select.dependency_index(root)
    index['tests/test_regression_example.py'] = (set(), {unknown})
    assert selected(plan(root, [unknown], index=index)) == selected(plan(root, [], mode='full', index=index))
