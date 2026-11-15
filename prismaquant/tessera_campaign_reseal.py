"""Read source-seal proof bundles for catalog source adoption."""
from __future__ import annotations

import json
from pathlib import Path

from .digests import file_sha256hex

BUNDLE_SCHEMA = 'prismaquant.reseal_proof_bundle.v1'
PIN_KEYS = ('prismaquant_source_sha256', 'encoder_source_sha256')


class Refused(RuntimeError):
    """The proof reader refuses invalid input."""


def unit_kind(qname):
    """Classify a routed expert projection or a dense unit."""
    return 'routed' if '.experts.' in str(qname) else 'dense'


def format_family(format_name):
    """``TESSERA_E2M1_K2_R896`` -> ``TESSERA_E2M1_K2``; any other spelling is refused."""
    family, sep, rate = str(format_name).rpartition('_R')
    if not (sep and family and rate.isdigit()):
        raise Refused(f'format {format_name!r} is not spelled FAMILY_R<rate>')
    return family


def strata_names(strata):
    return sorted(f'{kind}:{family}' for kind, family in strata)


def bundle_coverage(bundle, path):
    """The (kind, family) strata a bundle proves; its strata must match its own cells."""
    strata = bundle.get('strata')
    if not isinstance(strata, dict):
        raise Refused(f'{path}: the bundle carries no strata, so it cannot show which rows it covers')
    declared = set()
    for kind, families in strata.items():
        if kind not in ('dense', 'routed') or not isinstance(families, dict):
            raise Refused(f'{path}: strata[{kind!r}] is not a dense or routed family map')
        declared.update((kind, family) for family, rates in families.items() if rates)
    from_cells = set()
    for cell in bundle.get('cells') or ():
        kind, family = unit_kind(cell.get('qname')), format_family(cell.get('format_name'))
        if (cell.get('kind'), cell.get('family')) != (kind, family):
            raise Refused(f'{path}: cell {cell.get("qname")}@{cell.get("format_name")} is recorded as '
                          f'{cell.get("kind")}:{cell.get("family")}, not {kind}:{family}')
        from_cells.add((kind, family))
    if declared != from_cells:
        raise Refused(f'{path}: strata {strata_names(declared)} do not match the strata of its cells {strata_names(from_cells)}')
    return declared


def arm_pin_scope(arm_old, arm_new, pins, *, where):
    """``'same-tree'`` or ``'cross-tree'`` for an arm's pins against the pins file; refuse anything else.

    A same-tree arm carries exactly the pins file's old and new pins. A
    cross-tree arm re-encoded rows written under another PrismaQuant tree, so
    it proves the encoder transition and nothing about a PrismaQuant change.
    It is accepted only when all of these hold:

    * its encoder old and new pins are the pins file's;
    * the pins file holds its PrismaQuant pin fixed and moves the encoder pin;
    * the arm holds its own PrismaQuant pin fixed.
    """
    pq, encoder = PIN_KEYS
    for side, block in (('old', arm_old), ('new', arm_new)):
        if not isinstance(block, dict) or set(block) != set(PIN_KEYS):
            raise Refused(f'{where}: arm {side} pins must carry exactly {PIN_KEYS}')
    if (arm_old[encoder], arm_new[encoder]) != (pins['old'][encoder], pins['new'][encoder]):
        raise Refused(f'{where}: arm encoder transition {arm_old[encoder]} -> {arm_new[encoder]} is not the '
                      f'pins file\'s {pins["old"][encoder]} -> {pins["new"][encoder]}')
    if arm_old == pins['old'] and arm_new == pins['new']:
        return 'same-tree'
    if pins['old'][pq] != pins['new'][pq]:
        raise Refused(f'{where}: arm PrismaQuant pins {arm_old[pq]} -> {arm_new[pq]} are not the pins file\'s '
                      f'{pins["old"][pq]} -> {pins["new"][pq]}; the pins file moves the PrismaQuant pin, so '
                      'every arm must come from that tree')
    if pins['old'][encoder] == pins['new'][encoder]:
        raise Refused(f'{where}: arm PrismaQuant pin {arm_old[pq]} is not the pins file\'s {pins["old"][pq]}, and a '
                      'cross-tree arm proves only an encoder transition, which this pins file does not make')
    if arm_old[pq] != arm_new[pq]:
        raise Refused(f'{where}: arm moves its PrismaQuant pin {arm_old[pq]} -> {arm_new[pq]}; a cross-tree arm '
                      'must hold its PrismaQuant pin fixed')
    return 'cross-tree'


FUSED_STAGES = {'stage1': ('recipe_body', 'tcq'), 'stage2': ('recipe_plane', 'lut16')}


ENGAGEMENT_COUNTS = ('encode_windows', 'stage1_admitted', 'stage1_refused', 'stage1_fused',
                     'stage2_admitted', 'stage2_refused', 'stage2_fused', 'stage2_tripped', 'stage2_nonfinite')


def arm_engagement(result, comparison, *, where):
    """Check fused-stage counts for the cells in an arm.

    The caller requires this check when the encoder pin changes.
    Counts cover the interval from the first encode call to the last.
    Refused calls, fused LUT errors, and inconsistent counters cause refusal.
    Each stage must run if an arm's cells require it.
    The unchanged window and CHANNEL paths prove other cells.
    """
    cells = comparison.get('cells') or ()
    for cell in cells:
        if any(cell.get(field) is None for field, _ in FUSED_STAGES.values()):
            raise Refused(f'{where}: cell {cell.get("qname")}@{cell.get("format_name")} records no recipe body/plane, '
                          'so which fused stages the arm had to engage cannot be read (the arm predates the field)')
    expects = {stage: any(cell[field] == value for cell in cells) for stage, (field, value) in FUSED_STAGES.items()}
    summary = (result.get('fused_engagement') or {}).get('summary')
    if not isinstance(summary, dict) or any(type(summary.get(name)) is not int for name in ENGAGEMENT_COUNTS):
        raise Refused(f'{where}: the arm carries no fused-engagement counts scoped to its encode window '
                      '(it predates the field), so it reads as unengaged')
    if summary['encode_windows'] <= 0:
        raise Refused(f'{where}: no encode window was recorded, so the fused-stage counts are not this arm\'s encodes')
    for stage in FUSED_STAGES:
        if summary[f'{stage}_refused']:
            raise Refused(f'{where}: the {stage} gate refused {summary[f"{stage}_refused"]} call(s) in the encode window: '
                          f'{summary.get(f"{stage}_refusals") or {}}')
    if summary['stage2_tripped']:
        raise Refused(f'{where}: the fused LUT tripwire fired {summary["stage2_tripped"]} time(s) in the encode window; '
                      'its cost disagreed with the reference')
    if summary['stage2_nonfinite']:
        raise Refused(f'{where}: {summary["stage2_nonfinite"]} fused LUT fit(s) were nonfinite in the encode window '
                      'and ran the reference')
    if summary['stage1_admitted'] != summary['stage1_fused'] or summary.get('stage2_counts_agree') is False:
        raise Refused(f'{where}: the fused-stage counters disagree (stage 1 admitted {summary["stage1_admitted"]}, '
                      f'fused {summary["stage1_fused"]}; stage 2 probe and STATS agree: {summary.get("stage2_counts_agree")})')
    for stage, needed in expects.items():
        if needed and (summary[f'{stage}_admitted'] <= 0 or summary[f'{stage}_fused'] <= 0):
            field, value = FUSED_STAGES[stage]
            raise Refused(f'{where}: cells with {field}={value} need fused {stage}, and it never ran in the encode window '
                          f'(admitted {summary[f"{stage}_admitted"]}, fused {summary[f"{stage}_fused"]})')
    return dict(expects=expects, **{name: summary[name] for name in ENGAGEMENT_COUNTS})


def _arm_comparison(result):
    return result.get('comparison', result)


def _require_requested_setting_drops(comparison, pins, *, where):
    """Match the arm's removed settings to the requested settings."""
    declared = tuple(sorted(comparison.get('dropped_settings') or ()))
    if declared != tuple(sorted(pins['drop_settings'])):
        raise Refused(f'{where}: arm dropped settings {list(declared)}, the pins file '
                      f'drops {sorted(pins["drop_settings"])}')


def load_bundle(path, pins):
    bundle = json.loads(Path(path).read_text())
    if bundle.get('schema') != BUNDLE_SCHEMA:
        raise Refused(f'{path}: not a {BUNDLE_SCHEMA} proof bundle')
    if not bundle.get('ok'):
        raise Refused(f'{path}: the bundle does not certify the migration (ok is false)')
    for side in ('old', 'new'):
        if bundle['pins'][side] != pins[side]:
            raise Refused(f'{path}: bundle {side} pins {bundle["pins"][side]} differ from the pins file {pins[side]}')
    if not bundle.get('encoder_fixture_id_equal'):
        raise Refused(f'{path}: encoder_fixture_id is not shown equal between the producers')
    encoder_moves = bundle['pins']['old']['encoder_source_sha256'] != bundle['pins']['new']['encoder_source_sha256']
    if encoder_moves and not (bundle.get('fixture_id') or {}).get('result'):
        raise Refused(f'{path}: the encoder pin moves but the bundle carries no fixture-id arm')
    for arm in bundle['arms']:
        actual = file_sha256hex(arm['result'])
        if actual != arm['result_sha256']:
            raise Refused(f'{path}: arm result {arm["result"]} changed since the bundle was assembled')
        result = json.loads(Path(arm['result']).read_text())
        comparison = _arm_comparison(result)
        scope = arm_pin_scope(comparison.get('old_pins'), comparison.get('new_pins'), pins, where=f'{path}: {arm["result"]}')
        if arm.get('pin_scope') not in (None, scope):
            raise Refused(f'{path}: arm {arm["result"]} is recorded as {arm["pin_scope"]}, but its pins make it {scope}')
        # Pin-only source adoption declares no setting-removal request.
        if 'drop_settings' in pins:
            _require_requested_setting_drops(comparison, pins, where=f'{path}: {arm["result"]}')
        if encoder_moves:
            arm_engagement(result, comparison, where=f'{path}: {arm["result"]}')
    bundle['covered'] = bundle_coverage(bundle, path)
    bundle['bundle_sha256'] = file_sha256hex(path)
    bundle['path'] = str(Path(path).resolve())
    return bundle
