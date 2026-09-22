"""Recover the published campaign binding from its existing sealed quantum.

The interrupted producer's caller identity remains evidence. A new capture's
receipt uses the canonical roster and scope already sealed by the campaign.
No numeric measurement or old boundary identity is changed.
"""
from __future__ import annotations

import hashlib
import json


def resolve_forward_campaign(document, *, plan_sha256, prepared_sha256,
                             read_manifest_sha256, formats_by_qname,
                             calibration_shape):
    from .cost_stage_checkpoint import canonical_json_sha256
    from .joint_layer_quanta import check_quantum_for_campaign, roster_digest
    from .tessera_joint_aura import _bound

    bound = document['source_campaign_record']
    path = _bound(bound, 'forward recovery source campaign record')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bound['sha256']:
        raise RuntimeError('forward recovery campaign record changed during read')
    record = json.loads(raw)
    campaign = record['campaign']
    # Existing consumer validates the record's own canonical identity, schema,
    # exact campaign digests, scope and checkpoint binding.
    check_quantum_for_campaign(record, campaign)
    for field, expected in (('plan_sha256', plan_sha256),
                            ('prepared_sha256', prepared_sha256),
                            ('read_manifest_sha256', read_manifest_sha256)):
        if campaign.get(field) != expected:
            raise RuntimeError('forward recovery campaign differs in ' + field)
    names = sorted(formats_by_qname)
    canonical_roster = roster_digest(names)
    if campaign['unit_roster_sha256'] != canonical_roster:
        raise RuntimeError('forward recovery campaign unit roster differs')
    scope = campaign['campaign_scope']
    if (scope.get('source_unit_count') != len(names) or
            scope.get('source_roster_sha256') != canonical_json_sha256(
                names, where='forward recovery source roster') or
            list(calibration_shape) != [scope.get('window_count'),
                                        scope.get('calib_seqlen')] or
            scope.get('kind') != 'complete_campaign' or
            scope.get('window_count') != scope.get('campaign_window_count')):
        raise RuntimeError('forward recovery complete campaign geometry/roster differs')
    published = {field: campaign[field] for field in (
        'plan_sha256', 'prepared_sha256', 'read_manifest_sha256',
        'unit_roster_sha256', 'campaign_scope')}
    if document.get('published_campaign_identity') != published:
        raise RuntimeError('forward recovery published campaign identity differs')
    old = document['campaign_identity']
    for field in ('plan_sha256', 'prepared_sha256', 'read_manifest_sha256'):
        if old.get(field) != published[field]:
            raise RuntimeError('forward recovery original campaign differs in ' + field)
    # Precisely the historical caller bug: trailing newline and absent scope.
    # Also accept already-canonical records; no arbitrary rebinding exception.
    old_roster = hashlib.sha256(''.join(name + '\n' for name in names).encode()).hexdigest()
    if (old.get('unit_roster_sha256') not in (old_roster, canonical_roster) or
            old.get('campaign_scope') not in (None, scope)):
        raise RuntimeError('forward recovery original caller binding is unsupported')
    return published
