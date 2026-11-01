"""One loaded-model resolver separates priced units from their whole scope."""
import json

import pytest
import torch

from prismaquant import tessera_campaign as campaign
from prismaquant.model_profiles import DefaultProfile
from prismaquant.tessera_campaign_cli import parse_campaign_arguments
from prismaquant.tessera_campaign_requests import campaign_scope_request


NAMES = ['model.layers.0.proj', 'model.layers.1.proj']


def _model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in model.model.layers:
        layer.proj = torch.nn.Linear(4, 4, bias=False)
    model.lm_head = torch.nn.Linear(4, 4, bias=False)
    return model


def _request(tmp_path, extra=()):
    args = parse_campaign_arguments(['--model', 'fixture', '--out', str(tmp_path/'cost.pkl'),
                                    '--cache-dir', str(tmp_path/'cache'), *extra])
    args.calibration_input_receipt = None
    return campaign_scope_request(args)


def _resolve(request):
    return campaign.resolve_campaign_scope(request, _model(), DefaultProfile(),
        mode='research', device=torch.device('cpu'), serving_target=None,
        selected_capture_census=None, capture_unit_names=None, acquisition=None)


def _selection(tmp_path, *, model='fixture'):
    path = tmp_path/'units.json'
    path.write_text(json.dumps({'schema': campaign.UNITS_SCHEMA, 'model': model,
        'layer_stride': 1, 'groups': [{'key': 'u:'+NAMES[1], 'members': [NAMES[1]]}]}))
    return path


def test_whole_scope_retains_original_roster_and_explicit_empty_expert_population(tmp_path):
    scope = _resolve(_request(tmp_path))
    assert scope.roster.dense == tuple(NAMES)
    assert scope.roster.pinned == ()
    assert scope.population.members == ()
    assert scope.dense_targets == scope.census_dense_targets == NAMES
    assert scope.expert_targets == scope.census_expert_targets == []
    assert scope.selection is None and scope.exact_member_scope is None
    assert scope.partition_menu_targets is None
    assert scope.selected_groups == sorted(scope.scope_groups)
    assert scope.context_by_unit is None and scope.structure_by_unit is None
    assert not (tmp_path/'cache').exists()


def test_selection_narrows_priced_units_while_preserving_whole_scope(tmp_path):
    path = _selection(tmp_path)
    scope = _resolve(_request(tmp_path, ['--units', str(path)]))
    assert scope.dense_targets == [NAMES[1]]
    assert scope.census_dense_targets == NAMES
    assert scope.selected_groups == ['u:'+NAMES[1]]
    assert scope.scope_groups == {'u:'+name: [name] for name in NAMES}
    assert scope.selection['groups'] == [{'key': 'u:'+NAMES[1], 'members': [NAMES[1]]}]
    assert scope.audit_units == set() and scope.inclusion_probability == {}
    assert not (tmp_path/'cache').exists()


def test_census_selection_refusal_precedes_reading_the_selection(tmp_path):
    missing = tmp_path/'never-read-units.json'
    request = _request(tmp_path, ['--census-out', str(tmp_path/'census.json'),
                                 '--units', str(missing)])
    with pytest.raises(RuntimeError, match='whole scope.*cannot be narrowed'):
        _resolve(request)
    assert not missing.exists()
    assert not (tmp_path/'census.json').exists()


def test_selection_model_disagreement_refuses_before_pricing_or_cache(tmp_path):
    path = _selection(tmp_path, model='another-source')
    with pytest.raises(campaign.StackSampleError, match='model/layer_stride disagrees'):
        _resolve(_request(tmp_path, ['--units', str(path)]))
    assert not (tmp_path/'cache').exists()
    assert not (tmp_path/'cost.pkl').exists()


def test_scope_operation_has_all_actual_global_bindings():
    from tools.verify_campaign_operation_globals import unbound_global_reads

    assert unbound_global_reads(campaign.resolve_campaign_scope) == ()


def _scope_dependency():
    return 1


def _deferred_operation():
    def later():
        return _scope_dependency()
    return later()


def test_global_binding_guard_detects_missing_deferred_dependency():
    import builtins
    from types import FunctionType
    from tools.verify_campaign_operation_globals import unbound_global_reads

    incomplete = FunctionType(_deferred_operation.__code__,
                              {'__builtins__': vars(builtins)})
    assert unbound_global_reads(incomplete) == (
        ('_deferred_operation.<locals>.later', '_scope_dependency'),)


def test_global_binding_guard_uses_the_functions_actual_builtins():
    from types import FunctionType
    from tools.verify_campaign_operation_globals import unbound_global_reads

    def needs_builtin():
        return len([])
    incomplete = FunctionType(needs_builtin.__code__, {'__builtins__': {}})
    assert unbound_global_reads(incomplete) == ((needs_builtin.__qualname__, 'len'),)
