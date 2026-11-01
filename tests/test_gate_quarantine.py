"""Quarantine cannot hide regressions in the same change."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_select', ROOT / 'tools/ci/gate_select.py')
select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select)
QUARANTINE = 'tools/ci/gate_quarantine.json'


def entry(target='tests/test_one.py::test_bad', issue=3201):
    return {'target': target, 'issue': issue, 'reason': 'tracked failure'}


@pytest.mark.parametrize('current', [
    [entry(), entry('tests/test_other.py')],
    [entry('tests/test_one.py')],
    [entry(issue=3202)],
])
def test_quarantine_growth_cannot_accompany_code_changes(current):
    with pytest.raises(ValueError, match='quarantine-only'):
        select.quarantine_policy({'entries': current}, {'entries': [entry()]},
                                 [QUARANTINE, 'prismaquant/owner.py'])


@pytest.mark.parametrize('changed', [[QUARANTINE], [QUARANTINE, 'docs/ci_gate.md']])
def test_quarantine_only_change_can_add_entries(changed):
    select.quarantine_policy({'entries': [entry(), entry('tests/test_other.py')]},
                             {'entries': [entry()]}, changed)


def test_removal_and_file_to_exact_node_are_always_allowed():
    select.quarantine_policy({'entries': [entry()]},
        {'entries': [entry('tests/test_one.py'), entry('tests/test_other.py')]},
        [QUARANTINE, 'prismaquant/owner.py'])


def test_first_policy_bootstraps_when_base_has_no_policy():
    select.quarantine_policy({'entries': [entry()]}, None, [QUARANTINE, 'tools/ci/gate_select.py'])


@pytest.mark.parametrize('response', [{'state': 'closed'}, {'state': 'open', 'pull_request': {}}])
def test_owner_must_be_an_open_issue(response):
    with pytest.raises(ValueError, match='open issue'):
        select.check_quarantine_owners([entry()], fetch=lambda issue: response)


def test_owner_lookup_is_deduplicated():
    calls = []
    def fetch(issue):
        calls.append(issue)
        return {'state': 'open'}
    select.check_quarantine_owners([entry(), entry('tests/test_other.py')], fetch=fetch)
    assert calls == [3201]


def test_owner_api_failure_is_no_verdict():
    def unavailable(issue):
        raise TimeoutError('GitHub unavailable')
    with pytest.raises(RuntimeError, match='no verdict'):
        select.check_quarantine_owners([entry()], fetch=unavailable)
