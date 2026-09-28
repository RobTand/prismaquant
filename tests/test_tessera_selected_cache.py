"""Selected dense/expert bundle keeps exact measured wire identities."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.tessera_export_lane import TesseraExportLaneError, selected_cached_units_manifest
from prismaquant.tessera_reuse_authority import PRODUCER_AUTHORITY
from test_tessera_expert_projection import (
    STACK, _declared, _projection, _record,
)

FMT = "TESSERA_E4M3_K1_R1024"
DENSE = "model.layers.2.mlp.down_proj"


def fixture(tmp_path: Path):
    projection = _projection(experts=(0,))
    carried = tep.carried_projection(
        projection, tep.bind_expert_projection(projection, declared=_declared(experts=(0,))),
        request=tep.stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")
    source, experts, _ = tep.carried_units(carried)
    names = {DENSE, *experts}
    records, cells, units, costs = {}, {}, {}, {}
    for name in names:
        expert = name in experts
        record = (_record(tmp_path, name, experts[name]) if expert else
                  _record(tmp_path, name, {"tensor": name + ".weight", "rows": 2, "cols": 2,
                                           "source_tensor": name + ".weight", "source_layout": "whole",
                                           "source_slice": {}, "expert": 0, "projection": "down_proj", "group": "w2"}))
        record["identity"]["schema"] = (
            "tessera.cached_unit_inputs.v1" if expert else "tessera.encoding_inputs.v1")
        record["identity"]["source"] = {"dtype": "torch.bfloat16", "shape": [2, 2],
                                          "sha256": "a" * 64}
        record["identity"]["calibration"] = {"hessian": {"sha256": "b" * 64}}
        record["identity"]["encoder_source_sha256"] = "c" * 64
        if not expert:
            record["identity"].pop("projection")
        records[name] = record
        cells[name, FMT] = {"record": record}
        units[name] = {"weight": copy.deepcopy(record["identity"]["source"]),
                       "hessian": copy.deepcopy(record["identity"]["calibration"]["hessian"])}
        costs[name] = {FMT: {"joint_operator_identity": {"source_weight": {"shape": [2, 2]}}}}
    handoff = {"costs": costs, "provenance": {
        "tessera_joint_allocation": {"status": "research_metadata_handoff"},
        tep.PROJECTION_KEY: carried, "wire_dir": str(tmp_path.resolve())}}
    metadata = {tep.PROJECTION_KEY: carried, tep.WIRE_DIR_KEY: str(tmp_path.resolve()),
                tep.EXPERT_WIRES_KEY: {name: records[name] for name in experts}}
    data = SimpleNamespace(unit_scope=None, census={"unit_shapes": {name: [2, 2] for name in names}},
        manifest={"identity": {"units": units, "encoder_source_sha256": "c" * 64}},
        payload={"provenance": {tep.PROJECTION_KEY: carried, "wire_dir": str(tmp_path.resolve())},
                 "costs": {name: {FMT: {"output_mse_measured": True,
                                      "hessian_identity": {"applied": True}}} for name in names}},
        cells=cells)
    return source, names, records, handoff, metadata, data


def test_full_selected_bundle_includes_dense_and_expert(tmp_path):
    source, names, _, handoff, metadata, data = fixture(tmp_path)
    manifest = selected_cached_units_manifest(
        {name: FMT for name in names}, metadata, handoff, data,
        schema="tessera.cached_units.v1")
    from tessera.cached_unit import CachedUnitBundle
    bundle = CachedUnitBundle(manifest, tmp_path, set(names), source)
    assert set(bundle.units) == names
    assert manifest["units"][DENSE]["identity"]["schema"] == "tessera.encoding_inputs.v1"


def test_sampled_research_bundle_keeps_same_source_hessian_encoder_and_wire_gates(tmp_path):
    from prismaquant.tessera_sampled_stack_proposal import (
        SCHEMA, selected_assignment_sha256)
    source, names, records, handoff, metadata, data = fixture(tmp_path)
    assignment = {name: FMT for name in names}
    panel = {'status': 'diagnostic_pilot'}
    handoff['provenance']['joint_eval'] = panel
    handoff['provenance']['tessera_joint_allocation'].update(
        status='research_sampled_joint_panel', plan_sha256='a'*64,
        prepared={'path': '/prepared', 'sha256': 'b'*64})
    proposal = {'schema': SCHEMA, 'status': 'research_proposal',
        'production_export_authority': False, 'validation_export_eligible': None,
        'research_validation_permitted': True, 'pilot': panel,
        'expanded_assignment': assignment,
        'selected_assignment_sha256': selected_assignment_sha256(assignment),
        'original_joint_plan_sha256': 'a'*64,
        'original_prepared': handoff['provenance']['tessera_joint_allocation']['prepared']}
    manifest = selected_cached_units_manifest(
        assignment, metadata, handoff, data, schema='tessera.cached_units.v1',
        research_proposal=proposal)
    assert set(manifest['units']) == names
    records[DENSE]['identity']['encoder_source_sha256'] = 'd'*64
    with pytest.raises(TesseraExportLaneError, match='encoder differs'):
        selected_cached_units_manifest(assignment, metadata, handoff, data,
            schema='tessera.cached_units.v1', research_proposal=proposal)


@pytest.mark.parametrize('kind', ['dense', 'expert'])
def test_same_size_wire_change_passes_the_bundle_and_refuses_at_export_intake(tmp_path, kind):
    # The builder publishes the receipts and locates and sizes each blob; the
    # exporter hashes the bytes it reads.  A same-size content change therefore
    # builds and refuses where the blob is consumed (PrismaQuant #641, #643).
    from tessera.cached_unit import CachedUnitBundle, verify_cached_unit
    source, names, records, handoff, metadata, data = fixture(tmp_path)
    name = DENSE if kind == 'dense' else next(name for name in names if name != DENSE)
    path = tmp_path / records[name]['file']
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    manifest = selected_cached_units_manifest(
        {name: FMT for name in names}, metadata, handoff, data,
        schema='tessera.cached_units.v1')
    bundle = CachedUnitBundle(manifest, tmp_path, set(names), source)
    blob, record = bundle.read(name)
    with pytest.raises(ValueError, match='blob size/sha256 mismatch'):
        verify_cached_unit(blob, record, record['identity'])


def test_manifest_builder_reads_no_wire_bytes(tmp_path, monkeypatch):
    """The builder locates and sizes each blob; hashing stays at intake (#643)."""
    _, names, records, handoff, metadata, data = fixture(tmp_path)
    reads = []
    real_read_bytes = Path.read_bytes

    def counting(self, *args, **kwargs):
        reads.append(str(self))
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting)
    selected_cached_units_manifest({name: FMT for name in names}, metadata, handoff,
                                   data, schema='tessera.cached_units.v1')
    assert reads == [], "the manifest build must not read any blob bytes"


@pytest.mark.parametrize("change,match", [
    ("interpolated", "no exact measured joint wire"),
    ("source", "source differs from checkpoint seal"),
    ("hessian", "Hessian differs from checkpoint seal"),
    ("encoder", "encoder differs from checkpoint seal"),
    ("wire", "does not match its receipt|escapes the campaign directory"),
    ("coverage", "does not cover the full source roster"),
])
def test_missing_or_changed_selected_evidence_refuses(tmp_path, change, match):
    _, names, records, handoff, metadata, data = fixture(tmp_path)
    selected = {name: FMT for name in names}
    if change == "interpolated":
        data.cells.pop((DENSE, FMT))
    elif change == "source":
        records[DENSE]["identity"]["source"]["sha256"] = "d" * 64
    elif change == "hessian":
        records[DENSE]["identity"]["calibration"]["hessian"]["sha256"] = "d" * 64
    elif change == "encoder":
        records[DENSE]["identity"]["encoder_source_sha256"] = "d" * 64
    elif change == "wire":
        (tmp_path / records[DENSE]["file"]).write_bytes(b"changed")
    elif change == "coverage":
        selected.pop(DENSE)
    with pytest.raises(TesseraExportLaneError, match=match):
        selected_cached_units_manifest(selected, metadata, handoff, data,
                                       schema="tessera.cached_units.v1")


def _rooted_case(tmp_path, monkeypatch, change=None):
    """The rooted (v2) selected-cache case: two seals, an adopted overlay and a served policy.

    ``change`` applies one of the bridge test's refusal edits. Returns every
    name the case builds; ``build()`` runs the manifest builder on it.
    """
    # Capture-reuse and full512 policy derivation have independent artifact-level
    # tests. This bridge supplies their accepted boundary, then runs the real
    # per-cell migration proof checker, builder and Tessera mixed-root reader.
    from prismaquant import joint_catalog_extension as bridge, joint_served_activation as served
    from test_joint_catalog_extension import _encoder_proof, _write
    source, names, records, handoff, metadata, data = fixture(tmp_path)
    proof = _encoder_proof(tmp_path)
    added = tmp_path / 'added'; added.mkdir()
    rows, groups, assignment = [], {}, {}
    data.manifest['identity']['encoder_source_sha256'] = '4'*64
    for name in sorted(names):
        record = records[name]
        record['identity'].update(encoder_source_sha256='4'*64, encoder_fixture_id='f'*64)
        assignment[name] = FMT
        if name == DENSE: continue
        reference = copy.deepcopy(record['identity'])
        candidate = copy.deepcopy(record)
        candidate['identity'].update(encoder_source_sha256='8'*64, recipe=bridge.added_format_recipe(bridge.R13_ADDED_FORMAT))
        blob = (tmp_path / record['file']).read_bytes()
        path = added / record['file']; path.write_bytes(blob)
        if change == 'accepted_wrong_size':
            # Sealed into the catalog as written, so its stat fence holds and
            # only the receipt's blob_bytes disagrees with the file.
            path.write_bytes(blob + b'\x00')
        render = added / (record['file'] + '.pt'); render.write_bytes(b'render')
        adoption = {'schema': bridge.ADOPTION_SCHEMA, 'reference_pair': [name, FMT],
                    'reference_encoding_identity': reference, 'candidate_encoding_identity': candidate['identity'],
                    'encoder_source_proof': None if change in ('no_proof', 'no_proof_dev') else proof}
        qualified = {'act_bits': 4, 'static_contract': {'measured_as_served': True},
                     'activation_max_abs': 12.0, 'input_global_scale': 0.5}
        row = {'qname': name, 'format': bridge.R13_ADDED_FORMAT, 'record': candidate,
               'wire': str(path), 'render': str(render), 'catalog_source_adoption': adoption,
               'activation': qualified}
        for key in ('wire', 'render'):
            stat = Path(row[key]).stat()
            row[key+'_stat'] = {'inode': stat.st_ino, 'bytes': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
        rows.append(row); data.cells[name, bridge.R13_ADDED_FORMAT] = copy.deepcopy(row)
        data.payload['costs'][name][bridge.R13_ADDED_FORMAT] = copy.deepcopy(data.payload['costs'][name][FMT])
        metadata[tep.EXPERT_WIRES_KEY][name] = candidate
        assignment[name] = bridge.R13_ADDED_FORMAT
        groups[name] = {'members': [name], 'max_abs': 24.0, 'input_global_scale': 0.25}
    catalog = _write(tmp_path, 'selected-overlay.json', {'schema': 'prismaquant.t4_adopted_catalog.v1', 'cells': rows})
    data.inputs = {'candidate_overlay': catalog}
    data.payload['provenance']['candidate_overlay'] = catalog
    policy = {'schema': served.SCHEMA, 'format': bridge.R13_ADDED_FORMAT,
              'effective_max_abs': {name: 24.0 for name in groups},
              'qualification_max_abs': {name: 12.0 for name in groups},
              'executed_grouping': {'groups': groups},
              **{key: _write(tmp_path, key+'.json', {}) for key in ('original_prepared', 'original_cache', 'census')}}
    policy_bound = _write(tmp_path, 'policy.json', policy)
    for row in rows:
        name = row['qname']; qualification = row['activation']
        operator = {'source_weight': {'shape': [2, 2]},
            'activation': {**qualification, 'activation_max_abs': 24.0, 'input_global_scale': 0.25},
            'served_activation_policy': served.operator_policy_record(policy_bound, policy, name, bridge.R13_ADDED_FORMAT, qualification)}
        handoff['costs'][name][bridge.R13_ADDED_FORMAT] = {'joint_operator_identity': operator, 'input_global_scale': 0.25}
    old_plan = _write(tmp_path, 'accepted-old-plan.json', {'inputs': {}})
    old_pwc = _write(tmp_path, 'accepted-old-pwc.json', {})
    old_prepared = _write(tmp_path, 'accepted-old-prepared.json', {'production_cache': old_pwc})
    resource = _write(tmp_path, 'accepted-resources.json', {'inputs': {'original_plan': old_plan,
        'original_prepared': old_prepared, 'candidate_overlay': catalog, 'served_activation_policy': policy_bound}})
    plan = _write(tmp_path, 'accepted-plan.json', {'inputs': data.inputs, 'served_activation_policy': policy_bound,
                                                 'stage_b_resource_policy': resource})
    new_pwc = _write(tmp_path, 'accepted-new-pwc.json', {})
    new_prepared = _write(tmp_path, 'accepted-new-prepared.json', {'production_cache': new_pwc,
        'served_activation_policy': policy_bound, 'stage_b_resource_policy': resource})
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.joint_adjoint_slices import stage_a_run_header
    from test_stage_b_band_binding import synthetic_receipt
    receipt = synthetic_receipt(plan_sha256=old_plan['sha256'], prepared_sha256=old_prepared['sha256'],
                                scope={'fixture': 'selected-cache'}, num_layers=2, stride=1)
    header = stage_a_run_header(receipt)
    capture = _write(tmp_path, 'accepted-capture.json', receipt)
    inputs = {'extended_plan': plan, 'original_plan': old_plan, 'original_prepared': old_prepared,
              'extended_prepared': new_prepared}
    if change == 'v2_extension':
        # PQ #993's header-bound extension, which the first band can create.
        # No capture file is a control dependency of it.
        document = {'schema': bridge.SCHEMA, 'adjoint_run_header': header,
                    'adjoint_run_header_sha256': canonical_json_sha256(header, where='header'),
                    'inputs': inputs}
    else:
        document = {'schema': bridge.SCHEMA_V1, 'adjoint_capture': capture,
                    'adjoint_receipt_sha256': canonical_json_sha256(receipt, where='capture'),
                    'inputs': inputs}
    extension = _write(tmp_path, 'accepted-extension.json', document)
    handoff['provenance']['catalog_extension'] = extension
    handoff['provenance']['tessera_joint_allocation'].update(plan_sha256=plan['sha256'], prepared={'sha256': '2'*64})
    seen = []
    monkeypatch.setattr(bridge, 'require_extension', lambda *a, **kw: seen.append((a, kw)))
    monkeypatch.setattr(served, 'verify_policy', lambda *a, **kw: policy)
    packages = {'4'*64: {'path': str(tmp_path/'old-producer'), 'sha256': '4'*64},
                '8'*64: {'path': str(tmp_path/'candidate-producer'), 'sha256': '8'*64}}
    if change == 'missing_extension': extension = None
    elif change == 'missing_proof': Path(proof['path']).write_text('{}')
    elif change == 'changed_wire': Path(rows[0]['wire']).write_bytes(b'changed')
    elif change == 'wrong_scale': handoff['costs'][rows[0]['qname']][bridge.R13_ADDED_FORMAT]['input_global_scale'] = 0.5
    elif change == 'no_proof_dev':
        # A cell no reseal proof covers is admitted unproven in dev mode only
        # (PQ #1147); certified mode ('no_proof') refuses it (PQ #1438).
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    def build():
        return selected_cached_units_manifest(assignment, metadata, handoff, data,
            schema='tessera.cached_units.v2', catalog_extension=extension, producer_packages=packages)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize('change', [None, 'missing_extension', 'missing_proof', 'changed_wire', 'wrong_scale',
                                    'v2_extension', 'no_proof', 'no_proof_dev'])
def test_rooted_builder_reader_bridge_binds_adoption_and_served_scale(tmp_path, monkeypatch, capsys, change):
    import hashlib
    from operator import attrgetter
    from prismaquant import joint_catalog_extension as bridge
    from tessera.cached_unit import CachedUnitBundle
    names_ = ('build packages names source groups seen header old_plan plan old_prepared new_prepared '
              'old_pwc new_pwc resource capture rows records proof policy_bound policy').split()
    (build, packages, names, source, groups, seen, header, old_plan, plan, old_prepared, new_prepared,
     old_pwc, new_pwc, resource, capture, rows, records, proof, policy_bound, policy) = attrgetter(*names_)(
        _rooted_case(tmp_path, monkeypatch, change))
    if change == 'no_proof':
        with pytest.raises(ValueError, match='no encoder source proof'): build()
        return
    if change and change not in ('v2_extension', 'no_proof_dev'):
        with pytest.raises((ValueError, RuntimeError)): build()
        return
    manifest = build()
    for package in packages.values():
        directory = Path(package['path']); directory.mkdir()
        (directory/'__init__.py').write_text('# source fixture')
    if change == 'no_proof_dev':
        assert manifest['reuse_authority']['encoder_source_proofs'] == []
        assert all(adoption['encoder_source_proof'] is None for adoption in manifest['encoder_adoptions'].values())
        assert manifest['producer_packages'] == packages
        # The #1441 producer's actual proof-less manifest crosses the #644
        # reader boundary only through the producer's explicit dev opt-in.
        from prismaquant.tessera_export_lane import read_cached_unit_bundle
        bundle = read_cached_unit_bundle(manifest, tmp_path, set(names), source)
        assert bundle.encoder_source_proof_mode == 'permissive'
        assert {stamp['unit'] for stamp in bundle.warnings} == set(manifest['encoder_adoptions'])
        assert '[cached-unit warning]' in capsys.readouterr().err
        bundle.require_served_scales({name+'.input_global_scale': 0.25 for name in groups})
        monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
        with pytest.raises(ValueError, match='encoder source proof'):
            read_cached_unit_bundle(manifest, tmp_path, set(names), source)
        return
    assert len(seen) == 1 and seen[0][1]['run_header'] == header
    reads = set(bridge.selected_cache_read_paths(manifest))
    assert {item['path'] for item in (old_plan, plan, old_prepared, new_prepared, old_pwc, new_pwc, resource)} <= reads
    assert (capture['path'] in reads) == (change != 'v2_extension')
    assert {row['wire'] for row in rows} <= reads
    assert str(tmp_path/records[DENSE]['file']) in reads
    assert {proof['path'], str(tmp_path/'source-proof-arm.json'), str(tmp_path/'source-fixture.json')} <= reads
    assert {str(Path(package['path'])/'__init__.py') for package in packages.values()} <= reads
    assert {policy_bound['path'], *(policy[key]['path'] for key in ('original_prepared', 'original_cache', 'census'))} <= reads
    # The v2 extension case used to stop here: the pinned Tessera read only a
    # v1 extension as rooted cached-unit authority. Tessera f46be81f7 (in the
    # af7a86d43 pin) reads v2 as well, so a band-created extension now binds
    # through the same reader and the full bundle checks below apply to it.
    bundle = CachedUnitBundle(manifest, tmp_path, set(names), source,
                              authority=PRODUCER_AUTHORITY)
    assert len(bundle.roots) == 2 and bundle.producer_packages == packages
    for name in names:
        blob, record = bundle.read(name)
        assert hashlib.sha256(blob).hexdigest() == record['blob_sha256']
        assert record == manifest['units'][name]
    assert bundle.served_activations == {name: {'group': name, 'input_global_scale': 0.25} for name in groups}
    bundle.require_served_scales({name+'.input_global_scale': 0.25 for name in groups})
    with pytest.raises(ValueError, match='served policy'):
        bundle.require_served_scales({name+'.input_global_scale': 0.5 for name in groups})


VISUAL = "model.visual.blocks.0.attn.qkv"


def test_export_lane_passes_a_bf16_sidecar_outside_the_census(tmp_path):
    """GLM-5.3 Flash's allocator config also names the 124 visual-tower
    Linears it keeps at BF16. The census holds no wire for them, so the
    export lane passes them through, as the census cache does (#1510)."""
    source, names, _, handoff, metadata, data = fixture(tmp_path)
    manifest = selected_cached_units_manifest(
        {**{name: FMT for name in names}, VISUAL: "BF16"}, metadata, handoff, data,
        schema="tessera.cached_units.v1")
    assert set(manifest["units"]) == names


def test_export_lane_refuses_a_wired_sidecar_outside_the_census(tmp_path):
    source, names, _, handoff, metadata, data = fixture(tmp_path)
    with pytest.raises(TesseraExportLaneError, match="outside the census roster are not BF16"):
        selected_cached_units_manifest(
            {**{name: FMT for name in names}, VISUAL: FMT}, metadata, handoff, data,
            schema="tessera.cached_units.v1")


def test_export_lane_refuses_a_census_unit_the_assignment_omits(tmp_path):
    source, names, _, handoff, metadata, data = fixture(tmp_path)
    with pytest.raises(TesseraExportLaneError, match="does not cover the full source roster"):
        selected_cached_units_manifest(
            {name: FMT for name in names if name != DENSE}, metadata, handoff, data,
            schema="tessera.cached_units.v1")


def _two_roots(tmp_path: Path):
    """The #1513 shape: receipts were priced under the loader payload root,
    the joint handoff (and the allocator's recorded root) name a second wire
    directory holding copies."""
    import shutil
    from prismaquant import tessera_expert_projection as tep
    source, names, records, handoff, metadata, data = fixture(tmp_path)
    home = tmp_path.parent / (tmp_path.name + "-handoff")
    home.mkdir()
    for record in records.values():
        shutil.copy2(tmp_path / record["file"], home / record["file"])
    handoff["provenance"]["wire_dir"] = str(home)
    metadata[tep.WIRE_DIR_KEY] = str(home)
    return source, names, records, handoff, metadata, data, home


def test_two_wire_roots_holding_identical_bytes_build_the_selected_cache(tmp_path):
    # Path equality between the manifest root and the loader's payload root is
    # context, never the gate: presence, placement and receipted size decide
    # at manifest time, content identity at intake (PrismaQuant #1513).
    from tessera.cached_unit import CachedUnitBundle
    source, names, records, handoff, metadata, data, home = _two_roots(tmp_path)
    manifest = selected_cached_units_manifest(
        {name: FMT for name in names}, metadata, handoff, data,
        schema="tessera.cached_units.v1")
    assert set(manifest["units"]) == names
    bundle = CachedUnitBundle(manifest, home, set(names), source)
    assert set(bundle.units) == names
    blob, record = bundle.read(DENSE)
    assert hashlib.sha256(blob).hexdigest() == record["blob_sha256"]


def test_handoff_root_copy_that_diverges_from_the_priced_receipt_refuses(tmp_path):
    # A same-size content change passes the manifest-time gate and refuses at
    # intake, where the bytes the export reads are hashed against the priced
    # receipt (PrismaQuant #1513, #641/#643).
    from tessera.cached_unit import CachedUnitBundle, verify_cached_unit
    source, names, records, handoff, metadata, data, home = _two_roots(tmp_path)
    path = home / records[DENSE]["file"]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    manifest = selected_cached_units_manifest(
        {name: FMT for name in names}, metadata, handoff, data,
        schema="tessera.cached_units.v1")
    bundle = CachedUnitBundle(manifest, home, set(names), source)
    blob, record = bundle.read(DENSE)
    with pytest.raises(ValueError, match="blob size/sha256 mismatch"):
        verify_cached_unit(blob, record, record["identity"])


def test_handoff_root_copy_of_the_wrong_size_refuses_at_manifest_time(tmp_path):
    source, names, records, handoff, metadata, data, home = _two_roots(tmp_path)
    path = home / records[DENSE]["file"]
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(TesseraExportLaneError,
                       match="does not match its receipt"):
        selected_cached_units_manifest(
            {name: FMT for name in names}, metadata, handoff, data,
            schema="tessera.cached_units.v1")


def test_handoff_root_missing_the_priced_wire_refuses(tmp_path):
    source, names, records, handoff, metadata, data, home = _two_roots(tmp_path)
    (home / records[DENSE]["file"]).unlink()
    with pytest.raises(TesseraExportLaneError, match="is not in the wire directory"):
        selected_cached_units_manifest(
            {name: FMT for name in names}, metadata, handoff, data,
            schema="tessera.cached_units.v1")


def test_a_catalog_adopted_wire_only_under_its_accepted_root_builds(tmp_path, monkeypatch):
    # PR #1515 r2: a catalog-adopted cell is read from its catalog row's own
    # directory, so the handoff root need not hold a copy of its wire.
    case = _rooted_case(tmp_path, monkeypatch)
    for row in case.rows:
        (tmp_path / row['record']['file']).unlink()
    manifest = case.build()
    for row in case.rows:
        root = manifest['wire_roots'][manifest['unit_roots'][row['qname']]]
        assert root == str(Path(row['wire']).resolve().parent)


def test_a_catalog_adopted_wire_of_the_wrong_size_refuses_at_manifest_time(tmp_path, monkeypatch):
    # The size gate runs on the root the export reads: the accepted catalog
    # root, against the receipt its bytes are priced by (PR #1515 r2).
    case = _rooted_case(tmp_path, monkeypatch, 'accepted_wrong_size')
    import re
    bad = Path(case.rows[0]['wire']).resolve()
    with pytest.raises(TesseraExportLaneError, match=re.escape(f'{bad} does not match its receipt')):
        case.build()
