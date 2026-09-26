"""Freeze missing expert A4 cells by adopting exact already-qualified source/H commitments.

Each cell's ``anchor`` is the E2M1 extension journal's ``CampaignAnchor`` row
for ``(qname, format)``, read through the journal's own sealed unit reader
(``cost_stage_checkpoint._load_unit``). That is the row
``attach_candidate_overlay`` and ``tc.CampaignAnchor(**cell["anchor"])``
read. The first catalog stored the scalar cost row instead, and the loader
refused all 36,288 cells ("overlay measured output_mse differs"). The builder
now checks each journal row against the scalar cost row with the loader's own
field pairs, so a catalog the loader would refuse is never written.
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

BASE = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911')
PREP = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/joint-panel/complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/prepare/prepared.json')
PROOF = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/rollout-inputs/proof-bundle-9753a5b7c5-c92826fa4.json')
PROOF_SHA256 = '15b373db8429240ef29c0641818d1f7e705d18154a8c2d841381a00213fd86c9'
FMT = 'TESSERA_E2M1_K2_R896'
#: The loader's measured-row pairs (``joint_catalog_extension.attach_candidate_overlay``):
#: scalar cost field -> journal anchor field.
ANCHOR_FIELDS = (("output_mse", "dloss"), ("tessera_family", "family"),
                 ("tessera_body_rate_q256", "body_rate_q256"),
                 ("activation_contract", "activation_contract"),
                 ("activation_quantized", "activation_quantized"),
                 ("input_global_scale", "input_global_scale"), ("wire_bytes", "wire_bytes"))
MEASURED_CURRENCY = "output_mse_under_route_activation_contract"


sha = bytes_sha256hex


def stamp(path):
    s = path.stat()
    return {'inode': s.st_ino, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new catalog path; must not exist')
    parser.add_argument('--prepared', default=str(PREP))
    parser.add_argument('--census-base', default=str(BASE))
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists(), out
    base, prep = Path(args.census_base), Path(args.prepared)
    from prismaquant.tessera_joint_aura import STAGE

    started = time.monotonic()
    prepared = json.loads(prep.read_bytes())
    raw = Path(prepared['production_cache']['path']).read_bytes()
    assert sha(raw) == prepared['production_cache']['sha256']
    cache = pickle.loads(raw)
    del raw
    cost_path = base / 'extension-e2m1-01/workspace/merged-c92826fa4/cost.pkl'
    raw = cost_path.read_bytes()
    cost_sha = sha(raw)
    cost = pickle.loads(raw)
    del raw
    journal_path = cost_path.parent / 'cost.anchors.json'
    journal, journal_sha, journal_units = open_anchor_journal(journal_path)
    assert journal['stage'] == STAGE, journal['stage']
    proof_sha = sha(PROOF.read_bytes())
    assert proof_sha == PROOF_SHA256
    oldplan = json.loads((base / 'extension-r1024-02/workspace/plan.json').read_text())
    a4plan = json.loads((base / 'extension-e2m1-01/workspace/plan.json').read_text())
    oldowners = {q: Path(row['dir']) for row in oldplan['rows'] for q in row['members']}
    a4owners = {q: Path(row['dir']) for row in a4plan['rows'] for q in row['members']}
    expert_names = sorted(q for q in prepared['formats_by_qname'] if '.experts.' in q)
    assert set(expert_names) == set(cost['costs']) == set(a4owners) == set(journal_units)
    prior = {}
    for key, verified in cache.metadata['verified_cells'].items():
        q, fmt = key
        if q in a4owners and q not in prior:
            prior[q] = (fmt, verified)
    spec = fr.get_format(FMT)

    def cell(q):
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
        record = cost['tessera_expert_wires'][q][FMT]
        for field in ('unit', 'source', 'projection', 'calibration', 'encoder_fixture_id'):
            assert record['identity'][field] == old['identity'][field], (q, field)
        anchor, journal_record = journal_anchor(
            journal_units[q], stage=STAGE, qname=q,
            identity_sha256=journal['identity_sha256'], fmt=FMT)
        assert journal_record == record, (q, 'journal wire record differs from the merged cost')
        require_anchor_matches_scalar(anchor, cost['costs'][q][FMT], qname=q)
        activation = activation_identity(spec, cache.activation_max_abs, q)
        assert activation['input_global_scale'] == anchor['input_global_scale'], q
        render = a4owners[q] / 'cache' / (q.replace('/', '__').replace('.', '_') + '__' + FMT + '.pt')
        wire = Path(cost['provenance']['wire_dir']) / record['file']
        assert wire.stat().st_size == record['blob_bytes']
        return {'qname': q, 'format': FMT, 'render': str(render), 'render_stat': stamp(render),
                'wire': str(wire), 'wire_stat': stamp(wire), 'record': record, 'anchor': anchor,
                'source_weight': verified['source_weight'], 'activation': activation,
                'encoding_identity_sha256': canonical_json_sha256(record['identity'], where='adopted A4 encoding identity'),
                'render_origin': 'encoded', 'render_comparison': 'independent_render_vs_wire',
                'catalog_source_adoption': {
                    'schema': 'prismaquant.joint_catalog_source_adoption.v1', 'reference_pair': [q, oldfmt],
                    'reference_encoding_identity': normalized, 'candidate_encoding_identity': record['identity'],
                    'encoder_source_proof': {'path': str(PROOF), 'sha256': proof_sha}},
                'adopted_source_hessian': {
                    'schema': 'prismaquant.adopted_source_hessian.v1', 'old_format': oldfmt,
                    'old_pwc_sha256': prepared['production_cache']['sha256'],
                    'old_verified_record_sha256': canonical_json_sha256(verified, where='old verified cell'),
                    'old_wire_sha256': old['blob_sha256'],
                    'old_encoding_identity_sha256': verified['encoding_identity_sha256'],
                    'old_unit_envelope': str(oldpath), 'old_unit_envelope_sha256': sha(envelope_raw),
                    'encoder_source_normalized_to': normalization, 'reseal_proof_sha256': proof_sha,
                    'scope': 'source/H/projection/calibration commitments adopted by exact equality to an independently qualified old cell; no new source/H payload recomputation'}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        cells = list(pool.map(cell, expert_names))
    catalog = {'schema': 'prismaquant.t4_adopted_catalog.v1',
               'old_prepared': {'path': str(prep), 'sha256': sha(prep.read_bytes())},
               'old_pwc': prepared['production_cache'], 'cost': {'path': str(cost_path), 'sha256': cost_sha},
               'anchor_journal': {'path': str(journal_path), 'sha256': journal_sha,
                                  'identity_sha256': journal['identity_sha256']},
               'reseal_proof': {'path': str(PROOF), 'sha256': proof_sha},
               'source_model_identity': prepared['source_model_identity'],
               'calibration_input': prepared['calibration_input'],
               'source_execution': prepared['source_execution'],
               'reader_identity': prepared['reader_identity'],
               'projection_backend': prepared['projection_backend'],
               'format': FMT, 'cells': cells,
               'status': 'source_hessian_adopted_render_qualification_pending'}
    raw = json.dumps(catalog, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(raw)
    print(json.dumps({'path': str(out), 'sha256': sha(raw), 'cells': len(cells),
                      'seconds': time.monotonic() - started, 'status': catalog['status']}), flush=True)


if __name__ == '__main__':
    main()
