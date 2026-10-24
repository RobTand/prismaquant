"""Publish factual closure/canary joins; never rewrite or promote retained prices."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path

QNAME = 'model.language_model.layers.15.mlp.experts.56.gate_proj'
OLD_SOURCE = '6b558df3a2746f2a79637fc1d6dc488dc69351c20388b2d8d7212bd3087837c5'
ALLOWED_PACKAGE_CHANGE = {'joint_catalog_extension.py', 'joint_served_activation.py', 'joint_stageb_resources.py'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.relative_to(root).parts and p.suffix not in {'.pyc', '.pyo'}}


def load_cost(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def binding(path):
    p = Path(path)
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def scalar_fields(row):
    return {k: v for k, v in row.items() if v is None or isinstance(v, (str, bool, int, float))}


def journal_cell(root, qname, fmt):
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    root = Path(root)
    manifest = json.loads((root / 'cost.anchors.json').read_text())
    state = _load_unit(unit_path(root / 'cost.anchors.json.parts', qname), stage=manifest['stage'],
                       qname=qname, identity_sha256=manifest['identity_sha256'])
    record = state['wire_records'][fmt]
    anchor = next(a for a in state['anchors'] if a['format_name'] == fmt)
    wire = root / 'cache' / 'wire' / record['file']
    return manifest, record, anchor, binding(wire)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-source', type=Path, required=True)
    ap.add_argument('--original-canary', type=Path, required=True)
    ap.add_argument('--bridge-canary', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    original = args.original_source / 'prismaquant'
    current = Path(__file__).resolve().parents[1] / 'prismaquant'
    old_hash = _production_cache_source_sha256(original)
    new_hash = _production_cache_source_sha256(current)
    if old_hash != OLD_SOURCE:
        raise ValueError(f'Original package is not the actual retained producer: {old_hash}')
    old_files, new_files = inventory(original), inventory(current)
    changed = sorted(k for k in set(old_files) | set(new_files) if old_files.get(k) != new_files.get(k))
    if set(changed) != ALLOWED_PACKAGE_CHANGE:
        raise ValueError(f'Non-admission package closure changed: {changed}')
    accepted_catalog = Path('/mnt/shared/tessera-measurements/pact-missing224-20261005/bridge-pricing-sourceb5221b25/prismaquant/joint_catalog_extension.py')
    if digest(accepted_catalog) != digest(current / 'joint_catalog_extension.py'):
        raise ValueError('Accepted b522 catalog/proof walls changed during companion backport')
    report = {'schema': 'pact.catalog_source_comparability.v2',
              'original_source': {'root': str(args.original_source), 'package_sha256': old_hash},
              'bridge_source': {'package_root': str(current), 'package_sha256': new_hash},
              'package_files': len(old_files), 'changed_package_paths': changed,
              'non_admission_package_files_byte_identical': True,
              'accepted_b522_catalog_owner': binding(accepted_catalog),
              'accepted_b522_catalog_owner_byte_identical': True,
              'file_identities': {k: {'original': old_files.get(k), 'bridge': new_files.get(k)} for k in sorted(set(old_files) | set(new_files))},
              'numerical_closure_statement': 'Every durable original PrismaQuant package input outside the three explicitly authorized metadata/admission owners remains byte-identical, including encoder invocation, loss/probe, calibration, model/profile and quantizer owners. The accepted b522 catalog/proof owner is itself unchanged. Changed resource/policy metadata and exact new head still require independent review; file hashes alone are not numerical or full-price qualification.',
              'source_hash_restamp': False, 'retained_rows_modified': False,
              'new_joint_prices_admitted': 0, 'review_required': True}
    prior_path = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace/merged/cost.pkl')
    observed_prior = binding(prior_path)
    if observed_prior['sha256'] != 'cd21541019cb670876fcd3501cba8c58e3473044090e54f16726ad26b0eb27e1':
        raise ValueError('Retained control cost changed')
    prior = load_cost(prior_path)
    fmt = 'TESSERA_E4M3_K1_R1024'
    reference = prior['costs'][QNAME][fmt]
    original_cost = load_cost(args.original_canary / 'cost.pkl')
    original_row = original_cost['costs'][QNAME][fmt]
    _manifest, record, anchor, wire = journal_cell(args.original_canary, QNAME, fmt)
    report['original_same_encoder_control'] = {'qname': QNAME, 'format': fmt,
        'fresh_cost': binding(args.original_canary / 'cost.pkl'), 'wire': wire,
        'wire_record': record, 'anchor': anchor,
        'fresh_row_scalar_fields': scalar_fields(original_row), 'retained_row_scalar_fields': scalar_fields(reference),
        'actual_wire_bytes_equal_retained': wire['bytes'] == reference['wire_bytes'],
        'actual_anchor_loss_equal_retained': original_row['output_mse'] == reference['output_mse'],
        'retained_cost_input': observed_prior}
    report['bridge_same_encoder_control'] = None
    report['canary_comparability_proved'] = False
    if args.bridge_canary:
        bridge_cost = load_cost(args.bridge_canary / 'cost.pkl')
        bridge_row = bridge_cost['costs'][QNAME][fmt]
        _bm, b_record, b_anchor, b_wire = journal_cell(args.bridge_canary, QNAME, fmt)
        volatile = {'seconds', 'encode_seconds', 'encode_seconds_accounting', 'encoding_batch_size'}
        left = {k: v for k, v in original_row.items() if k not in volatile}
        right = {k: v for k, v in bridge_row.items() if k not in volatile}
        report['bridge_same_encoder_control'] = {'cost': binding(args.bridge_canary / 'cost.pkl'), 'wire': b_wire,
            'wire_sha256_equal': b_wire['sha256'] == wire['sha256'], 'wire_bytes_equal': b_wire['bytes'] == wire['bytes'],
            'row_equal_except_observation_timing': left == right,
            'actual_anchor_loss_equal': bridge_row['output_mse'] == original_row['output_mse'],
            'encoder_source_sha256_equal': b_record['identity']['encoder_source_sha256'] == record['identity']['encoder_source_sha256'],
            'encoder_fixture_id_equal': b_record['identity']['encoder_fixture_id'] == record['identity']['encoder_fixture_id'],
            'anchor_equal_except_observation_timing': {k: v for k, v in b_anchor.items() if k not in volatile} == {k: v for k, v in anchor.items() if k not in volatile}}
        control = report['bridge_same_encoder_control']
        report['canary_comparability_proved'] = all(control[k] for k in ('wire_sha256_equal', 'wire_bytes_equal',
            'row_equal_except_observation_timing', 'actual_anchor_loss_equal', 'encoder_source_sha256_equal',
            'encoder_fixture_id_equal', 'anchor_equal_except_observation_timing'))
    # Independent acceptance is deliberately not inferred from this report.
    report['accepted_as_comparable'] = False
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(raw).hexdigest(),
                     'original_package_sha256': old_hash, 'bridge_package_sha256': new_hash,
                     'changed_package_paths': changed,
                     'original_control_equal_retained': report['original_same_encoder_control']['actual_anchor_loss_equal_retained'],
                     'canary_comparability_proved': report['canary_comparability_proved'], 'accepted_as_comparable': False}), flush=True)


if __name__ == '__main__':
    main()
