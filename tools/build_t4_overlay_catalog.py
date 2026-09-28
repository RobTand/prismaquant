"""Freeze missing expert A4 cells by adopting exact already-qualified source/H commitments.

Each cell's ``anchor`` is the E2M1 extension journal's ``CampaignAnchor`` row
for ``(qname, format)``, read through the journal's own sealed unit reader
(``cost_stage_checkpoint._load_unit``). That is the row
``attach_candidate_overlay`` and ``tc.CampaignAnchor(**cell["anchor"])``
read. The first catalog stored the scalar cost row instead, and the loader
refused all 36,288 cells ("overlay measured output_mse differs"). The builder
now checks each journal row against the scalar cost row with the loader's own
field pairs, so a catalog the loader would refuse is never written.

Several formats (PQ #1432). One catalog may carry every cell of existing
catalogs (``--carry CATALOG SHA256``, cells copied byte for byte so their
qualification results and rebindings still bind them) plus cells from one or
more merged campaign workspaces (``--workspace COST_PKL PLAN_JSON``; the
journal is ``cost.anchors.json`` beside the cost, and the plan's rows give
each unit's ``dir``). A workspace contributes a cell for every priced
Tessera format a unit does not already offer (``--formats`` narrows that).
Each cell's recipe must be the pinned contract's
(``joint_catalog_extension.added_format_recipe``). Workspace cells bind
``--proof``/``--proof-sha256``, or, with ``--no-proof``, no reseal proof
(``encoder_source_proof: null``): the loader admits those only in dev mode
(PQ #1147). Reference cells are found through ``--reference-plan``. With no
``--workspace`` and no ``--carry``, the tool builds the R13 A4 catalog from
its original workspace and proof, and writes the v1 bytes it always did;
any other build writes ``prismaquant.t4_adopted_catalog.v2``.
"""
import argparse
import concurrent.futures
import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import _load_unit, canonical_json_sha256, unit_path
from prismaquant.joint_aura import activation_identity
from prismaquant import format_registry as fr
from prismaquant.digests import bytes_sha256hex
from prismaquant.joint_catalog_extension import (
    CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2, R13_ADDED_FORMAT, added_format_recipes, catalog_sources, catalog_stat_fence,
    catalog_view)

BASE = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911')
PREP = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json')
PROOF = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/rollout-inputs/proof-bundle-9753a5b7c5-c92826fa4.json')
PROOF_SHA256 = '15b373db8429240ef29c0641818d1f7e705d18154a8c2d841381a00213fd86c9'
FMT = R13_ADDED_FORMAT
#: The R13 A4 build's one workspace and the workspace that owns its reference cells.
LEGACY_WORKSPACE_COST = BASE / 'extension-e2m1-01/workspace/merged-c92826fa4/cost.pkl'
LEGACY_WORKSPACE_PLAN = BASE / 'extension-e2m1-01/workspace/plan.json'
REFERENCE_PLAN = BASE / 'extension-r1024-02/workspace/plan.json'
#: The prepared fields every catalog header copies, and a carried catalog must share.
SCIENCE_FIELDS = ('source_model_identity', 'calibration_input', 'source_execution', 'reader_identity',
                  'projection_backend')
#: The loader's measured-row pairs (``joint_catalog_extension.attach_candidate_overlay``):
#: scalar cost field -> journal anchor field.
ANCHOR_FIELDS = (("output_mse", "dloss"), ("tessera_family", "family"),
                 ("tessera_body_rate_q256", "body_rate_q256"),
                 ("activation_contract", "activation_contract"),
                 ("activation_quantized", "activation_quantized"),
                 ("input_global_scale", "input_global_scale"), ("wire_bytes", "wire_bytes"))
MEASURED_CURRENCY = "output_mse_under_route_activation_contract"


sha = bytes_sha256hex


def open_anchor_journal(manifest_path):
    """The journal manifest, its digest and ``qname -> unit envelope path``."""
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    parts = Path(str(manifest_path) + '.parts')
    units = {}
    for entry in manifest['units']:
        path = unit_path(parts, entry['qname'])
        assert parts / entry['file'] == path, entry
        assert entry['qname'] not in units, entry['qname']
        units[entry['qname']] = path
    return manifest, sha(raw), units


def journal_anchor(unit_file, *, stage, qname, identity_sha256, fmt):
    """The journal's ``CampaignAnchor`` row and wire record for ``(qname, fmt)``."""
    state = _load_unit(unit_file, stage=stage, qname=qname, identity_sha256=identity_sha256)
    rows = [row for row in state['anchors'] if row['format_name'] == fmt]
    assert len(rows) == 1, (qname, fmt, len(rows))
    anchor = dict(rows[0])
    assert anchor['qname'] == qname, (qname, anchor['qname'])
    return anchor, state['wire_records'][fmt]


def require_anchor_matches_scalar(anchor, scalar, *, qname):
    """Refuse the row ``attach_candidate_overlay`` would refuse."""
    assert (scalar.get('output_mse_measured') is True
            and scalar.get('cost_source') == 'tessera_campaign_measured'
            and scalar.get('tessera_provenance') == 'measured'
            and scalar.get('currency') == MEASURED_CURRENCY), (qname, 'not a measured row')
    for target, source in ANCHOR_FIELDS:
        assert scalar.get(target) == anchor.get(source), (qname, target, scalar.get(target), anchor.get(source))
    assert type(anchor['dloss']) in (int, float) and math.isfinite(anchor['dloss']) and anchor['dloss'] >= 0, qname


def plan_owners(plan_paths):
    """``qname -> row dir`` over one or more campaign plans; a unit owned twice refuses."""
    owners = {}
    for path in plan_paths:
        plan = json.loads(Path(path).read_text())
        for row in plan['rows']:
            for qname in row['members']:
                assert qname not in owners, (qname, 'owned by two plan rows')
                owners[qname] = Path(row['dir'])
    return owners


def workspace_cells(prepared, formats_filter):
    """The ``(qname, format)`` cells a workspace's cost contributes."""
    def cells(cost):
        rows = []
        for qname in sorted(cost['costs']):
            assert qname in prepared['formats_by_qname'], (qname, 'not in the original roster')
            offered = set(prepared['formats_by_qname'][qname])
            for fmt in sorted(cost['costs'][qname]):
                if not fmt.startswith('TESSERA_') or fmt in offered:
                    continue
                if formats_filter is None or fmt in formats_filter:
                    rows.append((qname, fmt))
        return rows
    return cells


def carried_cells(binding, prepared_binding, prepared):
    """Every cell of an existing catalog, with its sources, checked against this build's original."""
    raw = Path(binding['path']).read_bytes()
    assert sha(raw) == binding['sha256'], (binding['path'], 'SHA256 mismatch')
    catalog = json.loads(raw)
    view = catalog_view(catalog)
    assert catalog['old_prepared'] == prepared_binding, (binding['path'], 'carried catalog extends another prepared')
    assert catalog['old_pwc'] == prepared['production_cache'], (binding['path'], 'carried catalog original PWC')
    for key in SCIENCE_FIELDS:
        assert catalog[key] == prepared[key], (binding['path'], key)
    return catalog['cells'], view['sources'], view['cell_sources']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new catalog path; must not exist')
    parser.add_argument('--prepared', default=str(PREP))
    parser.add_argument('--census-base', default=str(BASE),
                        help='root of the default workspace and reference plan paths')
    parser.add_argument('--workspace', nargs=2, action='append', metavar=('COST_PKL', 'PLAN_JSON'),
                        help='a merged workspace cost.pkl and the plan whose rows own its units (repeatable)')
    parser.add_argument('--carry', nargs=2, action='append', metavar=('CATALOG', 'SHA256'),
                        help='carry every cell of an existing catalog unchanged (repeatable)')
    parser.add_argument('--reference-plan', action='append',
                        help='a plan whose rows own the original reference cells (repeatable; '
                             'default: the R13 extension-r1024-02 plan)')
    parser.add_argument('--formats', help='comma-separated added formats to take from the workspaces '
                                          '(default: every priced Tessera format a unit does not offer)')
    parser.add_argument('--proof', help='reseal proof bundle every workspace cell binds')
    parser.add_argument('--proof-sha256')
    parser.add_argument('--no-proof', action='store_true',
                        help='workspace cells bind no reseal proof (admitted in dev mode only)')
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists(), out
    base, prep = Path(args.census_base), Path(args.prepared)
    from prismaquant.tessera_joint_aura import STAGE

    legacy = not args.workspace and not args.carry
    if bool(args.proof) != bool(args.proof_sha256) or (args.proof and args.no_proof):
        parser.error('--proof and --proof-sha256 go together, and exclude --no-proof')
    if legacy:
        workspaces = [(str(base / LEGACY_WORKSPACE_COST.relative_to(BASE)),
                       str(base / LEGACY_WORKSPACE_PLAN.relative_to(BASE)))]
        formats_filter = {FMT} if args.formats is None else set(args.formats.split(','))
        proof_path, proof_expected = (PROOF, PROOF_SHA256) if not (args.proof or args.no_proof) else (
            (None, None) if args.no_proof else (Path(args.proof), args.proof_sha256))
    else:
        workspaces = list(args.workspace or ())
        formats_filter = None if args.formats is None else set(args.formats.split(','))
        if workspaces and not (args.proof or args.no_proof):
            parser.error('a --workspace build names its reseal proof (--proof) or declares none (--no-proof)')
        proof_path, proof_expected = (None, None) if not args.proof else (Path(args.proof), args.proof_sha256)
    reference_plans = args.reference_plan or [str(base / REFERENCE_PLAN.relative_to(BASE))]

    started = time.monotonic()
    prepared_raw = prep.read_bytes()
    prepared_binding = {'path': str(prep), 'sha256': sha(prepared_raw)}
    prepared = json.loads(prepared_raw)
    raw = Path(prepared['production_cache']['path']).read_bytes()
    assert sha(raw) == prepared['production_cache']['sha256']
    cache = pickle.loads(raw)
    del raw
    proof_binding = None
    if proof_path is not None:
        proof_sha = sha(proof_path.read_bytes())
        assert proof_sha == proof_expected, (str(proof_path), 'reseal proof SHA256')
        proof_binding = {'path': str(proof_path), 'sha256': proof_sha}
    recipe = added_format_recipes()
    spec_of = {}
    oldowners = plan_owners(reference_plans) if workspaces else {}

    cells, sources, cell_sources, carried_from, seen = [], [], [], [], set()
    for path, digest in args.carry or ():
        binding = {'path': str(Path(path)), 'sha256': digest}
        carried, carried_sources, carried_index = carried_cells(binding, prepared_binding, prepared)
        offset = len(sources)
        sources.extend(carried_sources)
        for cell, index in zip(carried, carried_index):
            pair = (cell['qname'], cell['format'])
            assert pair not in seen, (pair, 'carried twice')
            seen.add(pair)
            cells.append(cell)
            cell_sources.append(offset + index)
        carried_from.append(binding)

    for cost_arg, plan_arg in workspaces:
        cost_path = Path(cost_arg)
        raw = cost_path.read_bytes()
        cost_sha = sha(raw)
        cost = pickle.loads(raw)
        del raw
        journal_path = cost_path.parent / 'cost.anchors.json'
        journal, journal_sha, journal_units = open_anchor_journal(journal_path)
        assert journal['stage'] == STAGE, journal['stage']
        owners = plan_owners([plan_arg])
        wanted = workspace_cells(prepared, formats_filter)(cost)
        units = {qname for qname, _fmt in wanted}
        assert units <= set(owners) and units <= set(journal_units), 'workspace plan or journal misses a unit'
        if legacy:
            expert_names = {q for q in prepared['formats_by_qname'] if '.experts.' in q}
            assert expert_names == set(cost['costs']) == set(owners) == set(journal_units)
        prior = {}
        for key, verified in cache.metadata['verified_cells'].items():
            q, fmt = key
            if q in units and q not in prior:
                prior[q] = (fmt, verified)

        def cell(pair, cost=cost, owners=owners, journal=journal, journal_units=journal_units):
            q, fmt = pair
            if fmt not in spec_of:
                spec_of[fmt] = fr.get_format(fmt)
            oldfmt, verified = prior[q]
            oldpath = oldowners[q] / 'cost.anchors.json.parts/units' / (sha(q.encode()) + '.pkl')
            envelope_raw = oldpath.read_bytes()
            envelope = pickle.loads(envelope_raw)
            assert sha(envelope['payload']) == envelope['payload_sha256']
            unit = pickle.loads(envelope['payload'])
            old = unit['wire_records'][oldfmt]
            assert old['blob_sha256'] == verified['wire_sha256'], q
            normalized = copy.deepcopy(old['identity'])
            normalization = verified.get('encoder_source_reuse', {}).get('recorded_encoder_source_sha256')
            if normalization:
                normalized['encoder_source_sha256'] = normalization
            assert canonical_json_sha256(normalized, where='adopted old encoding identity') == verified['encoding_identity_sha256'], q
            anchor, record = journal_anchor(
                journal_units[q], stage=STAGE, qname=q,
                identity_sha256=journal['identity_sha256'], fmt=fmt)
            expert_record = cost.get('tessera_expert_wires', {}).get(q, {}).get(fmt)
            if expert_record is not None:
                assert record == expert_record, (q, 'journal wire record differs from the merged cost')
            for field in ('unit', 'source', 'projection', 'calibration', 'encoder_fixture_id'):
                assert record['identity'][field] == old['identity'][field], (q, field)
            assert record['identity'].get('recipe') == recipe(fmt), (q, fmt, 'recipe is not the pinned contract\'s')
            require_anchor_matches_scalar(anchor, cost['costs'][q][fmt], qname=q)
            activation = activation_identity(spec_of[fmt], cache.activation_max_abs, q)
            assert activation['input_global_scale'] == anchor['input_global_scale'], q
            render = owners[q] / 'cache' / (q.replace('/', '__').replace('.', '_') + '__' + fmt + '.pt')
            wire = Path(cost['provenance']['wire_dir']) / record['file']
            assert wire.stat().st_size == record['blob_bytes']
            return {'qname': q, 'format': fmt, 'render': str(render), 'render_stat': catalog_stat_fence(render.stat()),
                    'wire': str(wire), 'wire_stat': catalog_stat_fence(wire.stat()), 'record': record, 'anchor': anchor,
                    'source_weight': verified['source_weight'], 'activation': activation,
                    'encoding_identity_sha256': canonical_json_sha256(record['identity'], where='adopted encoding identity'),
                    'render_origin': 'encoded', 'render_comparison': 'independent_render_vs_wire',
                    'catalog_source_adoption': {
                        'schema': 'prismaquant.joint_catalog_source_adoption.v1', 'reference_pair': [q, oldfmt],
                        'reference_encoding_identity': normalized, 'candidate_encoding_identity': record['identity'],
                        'encoder_source_proof': proof_binding},
                    'adopted_source_hessian': {
                        'schema': 'prismaquant.adopted_source_hessian.v1', 'old_format': oldfmt,
                        'old_pwc_sha256': prepared['production_cache']['sha256'],
                        'old_verified_record_sha256': canonical_json_sha256(verified, where='old verified cell'),
                        'old_wire_sha256': old['blob_sha256'],
                        'old_encoding_identity_sha256': verified['encoding_identity_sha256'],
                        'old_unit_envelope': str(oldpath), 'old_unit_envelope_sha256': sha(envelope_raw),
                        'encoder_source_normalized_to': normalization,
                        'reseal_proof_sha256': None if proof_binding is None else proof_binding['sha256'],
                        'scope': 'source/H/projection/calibration commitments adopted by exact equality to an independently qualified old cell; no new source/H payload recomputation'}}

        for pair in wanted:
            assert pair not in seen, (pair, 'named by two sources')
            seen.add(pair)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            cells.extend(pool.map(cell, wanted))
        cell_sources.extend([len(sources)] * len(wanted))
        sources.append({'cost': {'path': str(cost_path), 'sha256': cost_sha},
                        'anchor_journal': {'path': str(journal_path), 'sha256': journal_sha,
                                           'identity_sha256': journal['identity_sha256']},
                        'reseal_proof': proof_binding})
    assert cells, 'the catalog adds no cell'
    formats = sorted({cell['format'] for cell in cells})
    header = {'old_prepared': prepared_binding, 'old_pwc': prepared['production_cache'],
              **{key: prepared[key] for key in SCIENCE_FIELDS}, 'cells': cells,
              'status': 'source_hessian_adopted_render_qualification_pending'}
    if legacy and len(sources) == 1 and formats == [FMT]:
        catalog = {'schema': CATALOG_SCHEMA_V1, **header, 'format': FMT, 'cost': sources[0]['cost'],
                   'anchor_journal': sources[0]['anchor_journal'], 'reseal_proof': sources[0]['reseal_proof']}
    else:
        catalog = {'schema': CATALOG_SCHEMA_V2, **header, 'formats': formats, 'sources': sources,
                   'cell_sources': cell_sources, 'carried_from': carried_from}
    catalog_view(catalog)
    assert len(catalog_sources(catalog)) == len(sources)
    raw = json.dumps(catalog, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(raw)
    counts = {fmt: sum(1 for cell in cells if cell['format'] == fmt) for fmt in formats}
    print(json.dumps({'path': str(out), 'sha256': sha(raw), 'schema': catalog['schema'], 'cells': len(cells),
                      'cells_by_format': counts, 'carried_from': carried_from,
                      'seconds': time.monotonic() - started, 'status': catalog['status']}), flush=True)


if __name__ == '__main__':
    main()
