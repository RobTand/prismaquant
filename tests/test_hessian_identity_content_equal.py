"""The #204 Hessian-identity gate over a table joined from two reference files.

``joint_catalog_extension.attach_candidate_overlay`` (PQ #985) admits overlay
rows whose ``capture_sha256`` is the seal of a different
``hessian_capture.references.json`` than the panel's, provided both files bind
one canonical capture and census and commit equal per-unit H. The joined
GLM-5.3 table (``joint-allocation.pkl`` d490a96d) carries exactly that shape:
197,990 rows under the panel's seal and 36,288 overlay rows under another.
``assert_uniform_hessian_identity`` compared the seals and refused it before the
allocator solved anything (RobTand/prismaquant#1270). A differing seal is now
accepted only when every other part of the identity is uniform and each
secondary row's unit carries the same H content in both reference files.
"""
import copy
import json
import pickle

import pytest

from prismaquant.tessera_menu import assert_uniform_hessian_identity
from tests.test_joint_catalog_extension import (_EXTRA_DENSE, _canonical_capture,
                                                _workspace_hessian)

PANEL_FMT = 'TESSERA_E4M3_K1_R1024'
ADDED_FMT = 'TESSERA_E2M1_K2_R896'
UNITS = ['model.layers.0.mlp.experts.0.down_proj', 'model.layers.0.mlp.experts.1.down_proj']
KWARG = 'ldl,ldl_block,refit_gauss_seidel,refit_metric,refit_reach_floor'


def _joined(tmp_path, *, override=None):
    """A panel over every unit and an overlay over the routed units, from one capture."""
    shared = _canonical_capture(tmp_path/'canonical', [*UNITS, _EXTRA_DENSE])
    panel_row, panel = _workspace_hessian(tmp_path/'panel', shared, [*UNITS, _EXTRA_DENSE])
    overlay_row, overlay = _workspace_hessian(tmp_path/'overlay', shared, UNITS, census_copy=True,
                                              override=override)
    assert panel_row['capture_sha256'] != overlay_row['capture_sha256']

    def row(identity):
        return {'hessian_identity': dict(copy.deepcopy(identity), kwarg=KWARG)}

    costs = {name: {PANEL_FMT: row(panel_row), ADDED_FMT: row(overlay_row)} for name in UNITS}
    costs[_EXTRA_DENSE] = {PANEL_FMT: row(panel_row)}
    references = {'primary': panel['capture_sha256'],
                  'captures': {panel['capture_sha256']: panel,
                               overlay['capture_sha256']: overlay}}
    return costs, references, panel, overlay


def test_two_captures_with_equal_unit_content_are_accepted(tmp_path):
    from prismaquant.tessera_menu import project_hessian_identity
    costs, references, panel, overlay = _joined(tmp_path)
    identity = assert_uniform_hessian_identity(costs, references=references)
    # Export binds the table's own capture; the overlay rows are named per row.
    assert identity['capture_sha256'] == panel['capture_sha256']
    assert identity['reference_binding'] == panel['reference_binding']
    assert identity['row_capture_sha256'] == {
        name: {ADDED_FMT: overlay['capture_sha256']} for name in UNITS}
    assert identity['captures'] == {panel['capture_sha256']: 3, overlay['capture_sha256']: 2}
    assert identity['stamped_rows'] == 5
    # Selection projects the row map onto the chosen units only.
    chosen = {UNITS[0]: ADDED_FMT, UNITS[1]: PANEL_FMT, _EXTRA_DENSE: PANEL_FMT, 'lm_head': 'BF16'}
    stamp = project_hessian_identity(identity, chosen)
    assert 'row_capture_sha256' not in stamp and 'captures' not in stamp
    assert stamp['unit_capture_sha256'] == {UNITS[0]: overlay['capture_sha256']}
    assert stamp['capture_sha256'] == panel['capture_sha256']


def test_two_captures_without_reference_files_are_refused(tmp_path):
    costs, _references, _panel, overlay = _joined(tmp_path)
    with pytest.raises(ValueError, match='mixes Hessian identities') as error:
        assert_uniform_hessian_identity(costs)
    assert 'no reference file' in str(error.value)
    references = copy.deepcopy(_references)
    del references['captures'][overlay['capture_sha256']]
    with pytest.raises(ValueError, match='no reference file'):
        assert_uniform_hessian_identity(costs, references=references)


def test_one_unit_with_different_content_is_refused_by_name(tmp_path):
    import torch
    shared_h = torch.eye(2)*3  # UNITS[0]'s committed H in the canonical fixture
    costs, references, _panel, _overlay = _joined(tmp_path, override={UNITS[0]: shared_h*7})
    with pytest.raises(ValueError, match='Hessian content differs') as error:
        assert_uniform_hessian_identity(costs, references=references)
    assert f'{UNITS[0]}[{ADDED_FMT}]' in str(error.value)
    assert UNITS[1] not in str(error.value)


@pytest.mark.parametrize('change', ['triple', 'kwarg', 'reference_binding'])
def test_a_different_draw_kwarg_or_binding_is_still_refused(tmp_path, change):
    costs, references, _panel, _overlay = _joined(tmp_path)
    ident = costs[UNITS[0]][ADDED_FMT]['hessian_identity']
    if change == 'triple':
        ident['fit_ids_sha256'] = 'f'*64
    elif change == 'kwarg':
        ident['kwarg'] = 'ldl'
    else:
        ident['reference_binding'] = dict(ident['reference_binding'], census_sha256='0'*64)
    with pytest.raises(ValueError, match='mixes Hessian identities'):
        assert_uniform_hessian_identity(costs, references=references)


def test_a_secondary_unit_absent_from_the_primary_file_is_refused(tmp_path):
    costs, references, _panel, overlay = _joined(tmp_path)
    costs['model.layers.9.mlp.experts.0.down_proj'] = {
        ADDED_FMT: copy.deepcopy(costs[UNITS[0]][ADDED_FMT])}
    with pytest.raises(ValueError, match='commits no Hessian'):
        assert_uniform_hessian_identity(costs, references=references)


def test_a_single_capture_return_is_unchanged(tmp_path):
    from prismaquant.tessera_menu import project_hessian_identity
    costs, references, _panel, _overlay = _joined(tmp_path)
    for rows in costs.values():
        rows.pop(ADDED_FMT, None)
    plain = assert_uniform_hessian_identity(costs)
    assert assert_uniform_hessian_identity(costs, references=references) == plain
    assert 'row_capture_sha256' not in plain and 'captures' not in plain
    assert project_hessian_identity(plain, {UNITS[0]: PANEL_FMT}) == plain


def _bind(path, payload):
    import hashlib
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def test_references_resolve_through_the_bound_catalog_extension_chain(tmp_path):
    """joint table -> catalog extension -> extended plan -> overlay catalog -> overlay cost."""
    from prismaquant.joint_catalog_extension import hessian_references
    _costs, _references, panel, overlay = _joined(tmp_path)
    cost = _bind(tmp_path/'overlay-cost.pkl', pickle.dumps({'provenance': {'hessian': overlay}}))
    catalog = _bind(tmp_path/'catalog.json', {'cost': cost})
    plan = _bind(tmp_path/'plan.json', {'inputs': {'candidate_overlay': catalog}})
    extension = _bind(tmp_path/'extension.json', {'inputs': {'extended_plan': plan}})
    payload = {'provenance': {'hessian': panel, 'catalog_extension': extension}}
    resolved = hessian_references(payload)
    assert resolved['primary'] == panel['capture_sha256']
    assert resolved['captures'] == {panel['capture_sha256']: panel,
                                    overlay['capture_sha256']: overlay}
    # A table with no extension names only its own capture.
    assert hessian_references({'provenance': {'hessian': panel}})['captures'] == {
        panel['capture_sha256']: panel}
    # The chain is hash-bound: a rewritten overlay cost is refused, never read.
    (tmp_path/'overlay-cost.pkl').write_bytes(pickle.dumps({'provenance': {'hessian': panel}}))
    with pytest.raises(ValueError, match='owned bytes'):
        hessian_references(payload)
