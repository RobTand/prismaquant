"""Proof-gated source-seal migration of plain campaign checkpoint metadata.

Row scope, proof coverage, source pin transitions, wire identities and scored
content retain their independent checks. Numerical model loading is not needed.
"""
from __future__ import annotations

import getpass
import io
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PINS_SCHEMA = 'prismaquant.reseal_pins.v1'
BUNDLE_SCHEMA = 'prismaquant.reseal_proof_bundle.v1'
RECORD_SCHEMA = 'prismaquant.identity_migration.v1'
JOURNAL_SCHEMA = 'prismaquant.reseal_row_journal.v1'
MANIFEST_SCHEMA = 'prismaquant.cost_stage_checkpoint.manifest.v1'
UNIT_SCHEMA = 'prismaquant.cost_stage_checkpoint.unit.v1'
from .tessera_campaign_table import COST_NAME, MANIFEST_NAME, PARTS_NAME, STAGE
from .digests import (
    DIRECT_UTF8_STRICT, DIRECT_UTF8_INDENT2_STRICT,
    DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR, LengthFramedSourceSha256,
    SOURCE_TREE_V1, source_tree_profiles, bytes_sha256hex, file_sha256hex,
)
from .cost_stage_checkpoint import atomic_write_bytes, unit_path as checkpoint_unit_path
PIN_KEYS = ('prismaquant_source_sha256', 'encoder_source_sha256')
LIVE_WORKSPACE_MARKER = 'first-proof-anchor-preparation-05/workspace/rows'
# The floor every assembled bundle must cover, whatever rows it is later used
# on. It does not decide which rows a bundle may rewrite: require_coverage
# derives that from the rows themselves. E2M1 is priced at one rate in this
# campaign (dense rows carry TESSERA_E2M1_K2_R896 only), so it is required at
# any rate rather than at each band rate.
REQUIRED_STRATA = {
    'dense': {'TESSERA_BF16_K1': {832, 960, 1088}, 'TESSERA_E4M3_K1': {832, 960, 1088}, 'TESSERA_E2M1_K2': set()},
    'routed': {'TESSERA_E4M3_K1': {832, 960, 1088}},
}
MIN_CELLS = 24


class Refused(RuntimeError):
    """The tool will not proceed; the message says why."""


# ---------------------------------------------------------------------------
# strata: the (kind, family) cells a proof covers and a row prices
# ---------------------------------------------------------------------------

def unit_kind(qname):
    """'routed' for a routed expert projection, otherwise 'dense'.

    The one classifier for both sides of the coverage check: proof cells in
    ``assemble_bundle`` and ``bundle_coverage``, and row units in
    ``identity_strata``.  If the two sides classified differently, a covered
    stratum would not be the stratum the row prices.
    """
    return 'routed' if '.experts.' in str(qname) else 'dense'


def format_family(format_name):
    """``TESSERA_E2M1_K2_R896`` -> ``TESSERA_E2M1_K2``; any other spelling is refused."""
    family, sep, rate = str(format_name).rpartition('_R')
    if not (sep and family and rate.isdigit()):
        raise Refused(f'format {format_name!r} is not spelled FAMILY_R<rate>')
    return family


def strata_names(strata):
    return sorted(f'{kind}:{family}' for kind, family in strata)


def identity_strata(identity, *, where):
    """(kind, family) for every menu format of every unit in a sealed identity."""
    units = identity.get('units')
    if not isinstance(units, dict) or not units:
        raise Refused(f'{where}: the identity carries no units to derive strata from')
    strata = set()
    for qname, unit in units.items():
        menu = unit.get('menu') if isinstance(unit, dict) else None
        if not isinstance(menu, (list, tuple)) or not menu:
            raise Refused(f'{where}: identity unit {qname} has no menu')
        strata.update((unit_kind(qname), format_family(fmt)) for fmt in menu)
    return strata


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


# ---------------------------------------------------------------------------
# source pins and checkpoint byte encodings
# ---------------------------------------------------------------------------

def prismaquant_tree_sha256(package_dir):
    """== production_weight_cache._production_cache_source_sha256(package_dir)."""
    root = Path(package_dir)
    paths = sorted(p for p in root.rglob('*') if p.is_file()
                   and '__pycache__' not in p.relative_to(root).parts and p.suffix not in {'.pyc', '.pyo'})
    digest = LengthFramedSourceSha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix(), path.read_bytes())
    return digest.hexdigest(), len(paths)


def encoder_tree_profiles(src_tessera_dir):
    """Label both profiles; preserve the original encoder suffixes and order."""
    root = Path(src_tessera_dir)
    paths = sorted(p for p in root.rglob('*') if p.suffix in {'.py', '.cu', '.cuh', '.cpp', '.h'})
    profiles = source_tree_profiles(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in paths)
    return profiles, len(paths)


def encoder_tree_sha256(src_tessera_dir):
    """The unchanged Tessera legacy scalar and count over src/tessera."""
    profiles, count = encoder_tree_profiles(src_tessera_dir)
    return profiles[SOURCE_TREE_V1], count


def canonical_bytes(value):
    """The byte form cost_stage_checkpoint.canonical_json_sha256 digests."""
    canonical = json.loads(DIRECT_UTF8_STRICT.text(value))
    return DIRECT_UTF8_STRICT.encoded(canonical)


def identity_sha256(identity):
    return bytes_sha256hex(canonical_bytes(identity))


def manifest_bytes(manifest):
    """The byte form the campaign and the dispatcher merge write."""
    return DIRECT_UTF8_INDENT2_STRICT.encoded(manifest)


def pickle_protocol(data):
    if data[:1] != b'\x80':
        raise Refused('pickle without a PROTO opcode; refusing to guess its protocol')
    return data[1]


def dumps_like(value, original_bytes):
    return pickle.dumps(value, protocol=pickle_protocol(original_bytes))


class StdlibUnpickler(pickle.Unpickler):
    """Refuse any global: campaign checkpoints hold only builtin containers."""

    def find_class(self, module, name):
        raise Refused(f'checkpoint pickle references {module}.{name}; the tool only rewrites plain data')


def loads(data):
    return StdlibUnpickler(io.BytesIO(data)).load()


def write_json(path, value):
    atomic_write_bytes(Path(path), DIRECT_ASCII_INDENT1_LAX_DEFAULT_STR.encoded(value) + b'\n')


# ---------------------------------------------------------------------------
# inputs: pins file, proof bundle
# ---------------------------------------------------------------------------

def load_pins(path):
    pins = json.loads(Path(path).read_text())
    if pins.get('schema') != PINS_SCHEMA:
        raise Refused(f'{path}: not a {PINS_SCHEMA} pins file')
    for side in ('old', 'new'):
        block = pins.get(side)
        if not isinstance(block, dict) or set(block) != set(PIN_KEYS):
            raise Refused(f'{path}: {side} must carry exactly {PIN_KEYS}')
        for key, value in block.items():
            if not (isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)):
                raise Refused(f'{path}: {side}.{key} is not a lowercase sha256')
    drop = pins.get('drop_settings') or []
    if not isinstance(drop, list) or any(not (isinstance(k, str) and k) for k in drop) or len(set(drop)) != len(drop):
        raise Refused(f'{path}: drop_settings must be a list of distinct, non-empty setting names')
    pins['drop_settings'] = tuple(drop)
    if pins['old'] == pins['new'] and not drop:
        raise Refused(f'{path}: old and new pins are identical and no setting is dropped; nothing to migrate')
    pins['source_checks'] = {}
    for name, hasher, sub in (('prismaquant', prismaquant_tree_sha256, 'prismaquant'), ('encoder', encoder_tree_sha256, 'src/tessera')):
        source = (pins.get('sources') or {}).get(name)
        if source and source.get('tree'):
            tree = Path(source['tree'])/sub
            if not tree.is_dir():
                raise Refused(f'{path}: sources.{name}.tree has no {sub}/ directory')
            actual, count = hasher(tree)
            key = 'prismaquant_source_sha256' if name == 'prismaquant' else 'encoder_source_sha256'
            if actual != pins['new'][key]:
                raise Refused(f'{path}: sources.{name}.tree hashes to {actual}, not the declared new {key} {pins["new"][key]}')
            pins['source_checks'][name] = dict(tree=str(tree), files=count, sha256=actual, commit=source.get('commit'))
    return pins


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


# Which tessera#486 fused stage a cell's encode can reach, read from the recipe
# its own wire receipt records (``tessera.export.WireRecipe.to_config``): the
# fused TCQ trellis sits behind ``viterbi_columns``, which only the TCQ body
# calls (a window body runs ``viterbi_window``); the fused LUT swap passes sit
# behind ``_pack_scales_lut``, which only the LUT16 scale plane calls. BF16 and
# E4M3 are window over CHANNEL at every rung, so their cells reach neither.
FUSED_STAGES = {'stage1': ('recipe_body', 'tcq'), 'stage2': ('recipe_plane', 'lut16')}


ENGAGEMENT_COUNTS = ('encode_windows', 'stage1_admitted', 'stage1_refused', 'stage1_fused',
                     'stage2_admitted', 'stage2_refused', 'stage2_fused', 'stage2_tripped', 'stage2_nonfinite')


def arm_engagement(result, comparison, *, where):
    """What the arm's cells needed from the fused encoder stages and what its encode window shows; refuse a gap.

    Required whenever the pins file moves the encoder pin. An arm must carry the
    fused-engagement counts ``reseal_identity_proof`` scopes to its encode
    window (first encode call to last); one that predates them reads as
    unengaged. In that window no gate may have refused a call, the fused LUT
    tripwire (its cost disagreeing with torch's) may not have fired, no fused
    LUT fit may have been nonfinite, and every stage-1 admission must have run
    the fused trellis. A stage is required of an arm only when one of its cells
    is encoded through it: then the gate must have admitted calls and the fused
    stage must have run. An arm whose cells reach neither stage (BF16, E4M3)
    still proves its bytes through the unchanged window/CHANNEL paths;
    ``assemble_bundle`` requires each stage from some arm.
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
    """An arm must prove the setting removal the migration requests."""
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


def assemble_bundle(pins_path, *, arm_paths, fixture_id=None, action_keys=()):
    pins = load_pins(pins_path)
    encoder_moves = pins['old']['encoder_source_sha256'] != pins['new']['encoder_source_sha256']
    if encoder_moves:
        if not fixture_id:
            raise Refused('the encoder pin moves; a fixture-id arm result is required (--fixture-id)')
        fixture = json.loads(Path(fixture_id).read_text())
        if fixture.get('kind') != 'fixture_id' or not fixture.get('ok'):
            raise Refused('fixture-id result is not an ok fixture_id arm')
        seals = fixture['encoder_source_sha256']
        if set(seals.values()) != {pins['old']['encoder_source_sha256'], pins['new']['encoder_source_sha256']}:
            raise Refused(f'fixture-id arm compared producers {seals}, not the pins file old/new encoder pins')
        fixture_record = dict(result=str(Path(fixture_id).resolve()), result_sha256=file_sha256hex(fixture_id),
                              ids=fixture['encoder_fixture_ids'], seals=seals,
                              fixture_id_equal=bool(fixture.get('fixture_id_equal')))
    else:
        if fixture_id:
            raise Refused('the encoder pin does not move (old == new); do not pass --fixture-id, the producer is unchanged')
        # The producer is the one that wrote the rows: its fixture id is the stored one by definition.
        fixture_record = dict(result=None, result_sha256=None, ids=None, seals={'unchanged': pins['old']['encoder_source_sha256']},
                              fixture_id_equal=True, encoder_pin_unchanged=True)
    cells, arms, strata = [], [], {}
    for path in arm_paths:
        result = json.loads(Path(path).read_text())
        comparison = _arm_comparison(result)
        if comparison.get('kind') == 'comparison':
            pass
        elif result.get('kind') not in ('prefix', 'dense'):
            raise Refused(f'{path}: not a prefix/dense proof arm result')
        if not comparison.get('ok'):
            raise Refused(f'{path}: arm did not pass ({comparison.get("failures")})')
        scope = arm_pin_scope(comparison.get('old_pins'), comparison.get('new_pins'), pins, where=path)
        # A GPU arm records the pins it actually ran under; they must be the
        # new pins it declares, or the arm is labelled with another tree.
        running = result.get('environment') or {}
        for key in PIN_KEYS:
            if key in running and running[key] != comparison['new_pins'][key]:
                raise Refused(f'{path}: arm ran with {key}={running[key]}, not its declared new pin '
                              f'{comparison["new_pins"][key]}')
        # The arm has to have compared against the identity ``migrate`` will
        # write, dropped settings and all.  An arm that dropped nothing proves
        # a different migration than the one this pins file describes, and an
        # older result that predates the field reads as "dropped nothing".
        _require_requested_setting_drops(comparison, pins, where=path)
        if not comparison.get('identity_matches_with_pins_substituted'):
            raise Refused(f'{path}: produced identity does not equal the stored identity with pins substituted')
        engagement = arm_engagement(result, comparison, where=path) if encoder_moves else None
        for cell in comparison['cells']:
            if not (cell.get('ok') and cell.get('byte_identical') and cell.get('dloss') == cell.get('stored_dloss')):
                raise Refused(f'{path}: cell {cell.get("qname")}@{cell.get("format_name")} is not a passing cell')
            kind = unit_kind(cell['qname'])
            if format_family(cell['format_name']) != cell['family']:
                raise Refused(f'{path}: cell {cell["qname"]}@{cell["format_name"]} records family {cell["family"]}')
            strata.setdefault(kind, {}).setdefault(cell['family'], set()).add(int(cell['body_rate_q256']))
            cells.append(dict(qname=cell['qname'], format_name=cell['format_name'], family=cell['family'], kind=kind,
                              body_rate_q256=cell['body_rate_q256'], blob_sha256=cell['blob_sha256'], blob_bytes=cell['blob_bytes'],
                              dloss=cell['dloss'], encoder_fixture_id=cell.get('encoder_fixture_id')))
        arms.append(dict(result=str(Path(path).resolve()), result_sha256=file_sha256hex(path), kind=result.get('kind', 'comparison'),
                         cells=len(comparison['cells']), strata=comparison.get('strata'), environment=result.get('environment'),
                         pin_scope=scope, old_pins=dict(comparison['old_pins']), new_pins=dict(comparison['new_pins']),
                         fused_engagement=(result.get('fused_engagement') or {}).get('summary'),
                         engagement=engagement, action_key=None))
    missing = []
    for kind, families in REQUIRED_STRATA.items():
        for family, rates in families.items():
            have = strata.get(kind, {}).get(family, set())
            if not have or not rates <= have:
                missing.append(f'{kind}:{family}@{sorted(rates) or "any"} (have {sorted(have)})')
    # A moving encoder is proven through its fused stages only by an arm whose
    # cells reached them; window/CHANNEL arms prove bytes on the paths around them.
    engagement_missing = []
    if encoder_moves:
        for stage, (field, value) in FUSED_STAGES.items():
            if not any(arm['engagement']['expects'][stage] and arm['engagement'][f'{stage}_fused'] > 0 for arm in arms):
                engagement_missing.append(f'{stage}: no arm has a {field}={value} cell encoded with the fused {stage} running')
    # Two arms may re-encode the same cell (two prefixes of one row start
    # at the same experts); both are evidence, but a cell is counted once.
    unique, seen = [], set()
    for cell in cells:
        key = (cell['qname'], cell['format_name'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(cell)
    duplicate_cells = len(cells) - len(unique)
    cells = unique
    ok = not missing and not engagement_missing and len(cells) >= MIN_CELLS
    bundle = dict(schema=BUNDLE_SCHEMA, assembled_unix=time.time(), assembled_by=getpass.getuser(), host=platform.node(),
                  pins={'old': pins['old'], 'new': pins['new']}, sources=pins.get('sources'), source_checks=pins['source_checks'],
                  fixture_id=fixture_record,
                  encoder_fixture_id_equal=fixture_record['fixture_id_equal'], arms=arms,
                  arm_prismaquant_pins=sorted({arm['new_pins']['prismaquant_source_sha256'] for arm in arms}),
                  cross_tree_arms=sum(arm['pin_scope'] == 'cross-tree' for arm in arms),
                  pb_actions=list(action_keys or []), cells=cells, cell_count=len(cells), duplicate_cells=duplicate_cells,
                  strata={k: {f: sorted(r) for f, r in fam.items()} for k, fam in strata.items()},
                  strata_missing=missing, min_cells=MIN_CELLS, engagement_required=encoder_moves,
                  engagement_missing=engagement_missing, ok=ok and fixture_record['fixture_id_equal'])
    return bundle


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

AUDIT_SCHEMA_PREFIX = 'prismaquant.tessera_campaign.checkpoint_audit'


def load_checkpoint_audit(path):
    """Row journal states from the campaign's checkpoint audit: row_id -> done | withdrawn.

    A row's completeness is a fact about its journal (did the campaign commit
    it, or withdraw it mid-way), not about which files happen to exist: a
    withdrawn row can carry a cost.pkl from an earlier round.
    """
    audit = json.loads(Path(path).read_text())
    if not str(audit.get('schema', '')).startswith(AUDIT_SCHEMA_PREFIX) and 'rows' not in audit:
        raise Refused(f'{path}: not a checkpoint audit')
    states = {}
    for entry in audit['rows']:
        state = entry.get('state')
        if state not in ('done', 'withdrawn'):
            raise Refused(f'{path}: row {entry.get("row_id")} has journal state {state!r}, not done/withdrawn')
        states[entry['row_id']] = state
    return states


def discover_rows(row_paths=(), *, workspace=None, checkpoint_audit=None, allow_live=False):
    rows = [Path(r) for r in (row_paths or [])]
    if workspace:
        if not checkpoint_audit:
            raise Refused('--workspace walks every row; pass --checkpoint-audit so partial journals are selected by state, not by which files exist')
        rows += sorted(p for p in (Path(workspace)/'rows').iterdir() if p.is_dir() and (p/MANIFEST_NAME).is_file())
    if not rows:
        raise Refused('no rows: pass --row DIR and/or --workspace WS')
    for row in rows:
        if LIVE_WORKSPACE_MARKER in str(row.resolve()) and not allow_live:
            raise Refused(f'{row} is inside the live campaign workspace; pass --allow-live to touch it')
    return rows


def classify_pins(identity, pins):
    """'pending' rows carry the old pins and every setting the pins file drops;
    'migrated' rows carry the new pins and none of them; anything else is foreign."""
    current = {key: identity.get(key) for key in PIN_KEYS}
    settings = identity.get('settings')
    drop = tuple(pins.get('drop_settings') or ())
    present = tuple(k for k in drop if isinstance(settings, dict) and k in settings)
    if current == pins['new'] and not present:
        return 'migrated', current
    if current == pins['old'] and present == drop:
        return 'pending', current
    return 'foreign', current


def require_coverage(rows, pins, bundle):
    """Refuse when a row to be rewritten prices a stratum the bundle does not prove.

    Runs over every row before the first one is planned or rewritten, so a
    refusal never leaves a workspace half migrated.  Reads manifests only.
    Rows already at the new pins are not rewritten, so they are not checked.
    """
    covered = bundle['covered']
    checked, uncovered = {}, []
    for row in rows:
        identity = json.loads((Path(row)/MANIFEST_NAME).read_bytes()).get('identity') or {}
        if classify_pins(identity, pins)[0] == 'migrated':
            continue
        strata = identity_strata(identity, where=row)
        checked[str(row)] = strata_names(strata)
        missing = strata - covered
        if missing:
            uncovered.append(f'{row} [{", ".join(strata_names(missing))}]')
    if uncovered:
        raise Refused(f'the proof bundle covers {strata_names(covered)}; {len(uncovered)} row(s) price strata it '
                      f'does not: ' + '; '.join(uncovered))
    return checked


def tool_commit():
    try:
        out = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent.parent), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent.parent), 'status', '--porcelain'],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ('+dirty' if dirty else '')
    except Exception:  # noqa: BLE001 - a PB checkout may not be a git tree
        return None


def row_kind(row, cost_path, audit_states):
    """complete | partial, from the audit when the row is listed there.

    Listed withdrawn: partial (its shards may stop short, and a cost.pkl from
    an earlier round is resealed if present). Listed done, or unlisted: the
    row must be complete, and a missing cost.pkl or shard is refused rather
    than silently treated as a partial journal. Without an audit the old
    file-presence rule applies (cost.pkl present -> complete).
    """
    if audit_states is None:
        return 'complete' if cost_path.is_file() else 'partial'
    state = audit_states.get(row.name)
    if state == 'withdrawn':
        return 'partial'
    if not cost_path.is_file():
        raise Refused(f'{row}: no cost.pkl and the checkpoint audit does not record the row as withdrawn')
    return 'complete'


def plan_row(row, pins, *, run_id, audit_states=None):
    """Everything the rewrite would do to one row, computed without writing."""
    row = Path(row)
    manifest_path = row/MANIFEST_NAME
    parts = row/PARTS_NAME
    cost_path = row/COST_NAME
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('stage') != STAGE:
        raise Refused(f'{row}: not a {STAGE} checkpoint manifest')
    identity = manifest['identity']
    old_sha = identity_sha256(identity)
    if old_sha != manifest['identity_sha256']:
        raise Refused(f'{row}: stored identity_sha256 {manifest["identity_sha256"]} != recomputed {old_sha}; the digest reimplementation or the row is wrong')
    state, current = classify_pins(identity, pins)
    strata = identity_strata(identity, where=row)
    plan = dict(row=str(row), kind=row_kind(row, cost_path, audit_states), journal_state=(audit_states or {}).get(row.name),
                state=state, current_pins=current, strata=strata_names(strata),
                old_identity_sha256=old_sha, run_id=run_id, manifest_bytes=len(manifest_raw),
                manifest_reserialization_identical=(manifest_bytes(manifest) in (manifest_raw, manifest_raw.rstrip(b'\n'))),
                edits=[], shard_count=0, shard_bytes=0, receipt_seals=0, cost_seals=0, bytes_to_write=0)
    if state == 'migrated':
        plan['note'] = 'row already carries the new pins'
        plan['migration_records'] = manifest.get('identity_migration', [])
        return plan, None
    if state == 'foreign':
        raise Refused(f'{row}: pins {current} with settings {sorted(identity.get("settings") or {})} are neither the old nor the new '
                      f'(pins, dropped settings {list(pins.get("drop_settings") or ())}) of the pins file')
    encoder_moves = pins['old']['encoder_source_sha256'] != pins['new']['encoder_source_sha256']
    new_identity = json.loads(json.dumps(identity))
    for key in PIN_KEYS:
        new_identity[key] = pins['new'][key]
    # Settings the campaign no longer binds (scheduling knobs that never touched
    # a receipt) leave the identity; their values are kept in the migration record.
    dropped = {k: new_identity['settings'].pop(k) for k in pins.get('drop_settings') or ()}
    plan['dropped_settings'] = dropped
    new_sha = identity_sha256(new_identity)
    plan['new_identity_sha256'] = new_sha
    plan['edits'].append(dict(file=str(manifest_path), fields=[f'identity.{k}' for k in PIN_KEYS if pins['old'][k] != pins['new'][k]]
                              + [f'identity.settings.{k}' for k in dropped] + ['identity_sha256', 'identity_migration[+1]'],
                              before=dict(identity_sha256=old_sha, **current, **({'settings': dropped} if dropped else {})),
                              after=dict(identity_sha256=new_sha, **pins['new'])))
    shards = []
    for entry in manifest['units']:
        path = parts/entry['file']
        if path != checkpoint_unit_path(parts, entry['qname']):
            raise Refused(f'{row}: {entry["qname"]} names a noncanonical shard {entry["file"]}')
        if not path.is_file():
            if plan['kind'] == 'complete':
                raise Refused(f'{row}: complete row lacks shard {path}')
            continue
        shards.append((entry['qname'], path))
    plan['shard_count'] = len(shards)
    shard_plans = []
    for qname, path in shards:
        raw = path.read_bytes()
        envelope = loads(raw)
        for field, expected in (('schema', UNIT_SCHEMA), ('stage', STAGE), ('qname', qname), ('identity_sha256', old_sha)):
            if envelope.get(field) != expected:
                raise Refused(f'{row}: shard {path.name} {field}={envelope.get(field)!r}, expected {expected!r}')
        payload = envelope['payload']
        if bytes_sha256hex(payload) != envelope['payload_sha256']:
            raise Refused(f'{row}: shard {path.name} payload_sha256 does not describe its payload')
        state_obj = loads(payload)
        # The strata come from the sealed menu; a shard that priced anything
        # else would be rewritten under a stratum no proof was checked against.
        menu = set((identity['units'].get(qname) or {}).get('menu') or ())
        for anchor in state_obj.get('anchors') or ():
            fmt = anchor.get('format_name')
            if fmt not in menu or anchor.get('family') != format_family(fmt):
                raise Refused(f'{row}: shard {path.name} anchor {qname}@{fmt} (family {anchor.get("family")}) '
                              f"is not a format of the unit's sealed menu")
        for fmt in state_obj.get('wire_records') or {}:
            if fmt not in menu:
                raise Refused(f"{row}: shard {path.name} receipt {qname}@{fmt} is not a format of the unit's sealed menu")
        seals = 0
        for fmt, record in (state_obj.get('wire_records') or {}).items():
            seal = record.get('identity', {}).get('encoder_source_sha256')
            if seal != pins['old']['encoder_source_sha256']:
                raise Refused(f'{row}: {qname}@{fmt} receipt seal {seal} is not the old encoder pin')
            seals += 1
        if encoder_moves:
            repickled = dumps_like(state_obj, payload)
            if repickled != payload:
                raise Refused(f'{row}: shard {path.name} does not re-pickle to its own bytes; refusing to rewrite a payload the tool cannot reproduce')
            for record in state_obj['wire_records'].values():
                record['identity']['encoder_source_sha256'] = pins['new']['encoder_source_sha256']
            new_payload = dumps_like(state_obj, payload)
        else:
            new_payload = payload
        new_envelope = dict(envelope, identity_sha256=new_sha, payload_sha256=bytes_sha256hex(new_payload), payload=new_payload)
        if dumps_like(envelope, raw) != raw:
            raise Refused(f'{row}: shard {path.name} envelope does not re-pickle to its own bytes')
        new_raw = dumps_like(new_envelope, raw)
        shard_plans.append(dict(qname=qname, path=path, raw_bytes=len(raw), new_raw=new_raw, seals=seals,
                                before=dict(identity_sha256=old_sha, payload_sha256=envelope['payload_sha256']),
                                after=dict(identity_sha256=new_sha, payload_sha256=new_envelope['payload_sha256'])))
        plan['shard_bytes'] += len(raw)
        plan['receipt_seals'] += seals
    plan['edits'].append(dict(file=str(parts/'units'), shards=len(shard_plans), fields=['identity_sha256'] + (['payload_sha256', 'payload.wire_records[*].identity.encoder_source_sha256'] if encoder_moves else []),
                              sample=[dict(qname=s['qname'], before=s['before'], after=s['after']) for s in shard_plans[:3]]))
    cost_plan = None
    if cost_path.is_file():
        raw = cost_path.read_bytes()
        cost = loads(raw)
        if dumps_like(cost, raw) != raw:
            raise Refused(f'{row}: cost.pkl does not re-pickle to its own bytes; refusing to rewrite it')
        wires = cost.get('tessera_expert_wires') or {}
        for unit, records in wires.items():
            for fmt, record in records.items():
                seal = record.get('identity', {}).get('encoder_source_sha256')
                if seal != pins['old']['encoder_source_sha256']:
                    raise Refused(f'{row}: cost.pkl tessera_expert_wires[{unit}][{fmt}] seal {seal} is not the old encoder pin')
                plan['cost_seals'] += 1
                if encoder_moves:
                    record['identity']['encoder_source_sha256'] = pins['new']['encoder_source_sha256']
        provenance = cost['provenance']
        if not isinstance(provenance, dict):
            raise Refused(f'{row}: cost.pkl provenance is not a mapping')
        cost_plan = dict(path=cost_path, raw_bytes=len(raw), cost=cost, raw=raw)
        plan['edits'].append(dict(file=str(cost_path), fields=(['tessera_expert_wires[*][*].identity.encoder_source_sha256'] if encoder_moves and wires else []) + ['provenance.identity_migration[+1]'],
                                  seals=plan['cost_seals'], bytes=len(raw)))
    plan['bytes_to_write'] = len(manifest_raw) + sum(len(s['new_raw']) for s in shard_plans) + (cost_plan['raw_bytes'] if cost_plan else 0)
    work = dict(manifest=manifest, manifest_raw=manifest_raw, new_identity=new_identity, new_sha=new_sha, old_sha=old_sha,
                shards=shard_plans, cost=cost_plan, encoder_moves=encoder_moves)
    return plan, work


def migration_record(pins, bundle, plan, *, operator, run_id, when):
    return dict(schema=RECORD_SCHEMA, run_id=run_id, migrated_unix=when, operator=operator, host=platform.node(),
                tool='tools/reseal_campaign_identity.py', tool_commit=tool_commit(),
                old_pins=dict(pins['old']), new_pins=dict(pins['new']), sources=pins.get('sources'),
                dropped_settings=dict(plan.get('dropped_settings') or {}),
                old_identity_sha256=plan['old_identity_sha256'], new_identity_sha256=plan['new_identity_sha256'],
                proof_bundle=bundle['path'], proof_bundle_sha256=bundle['bundle_sha256'],
                proof_cells=bundle['cell_count'], proof_pb_actions=bundle.get('pb_actions'),
                encoder_fixture_id=sorted(set((bundle['fixture_id'].get('ids') or {}).values())) or None,
                shards=plan['shard_count'], receipt_seals=plan['receipt_seals'], cost_seals=plan['cost_seals'],
                row_strata=plan['strata'], proof_strata=strata_names(bundle['covered']))


def migrate_row(row, plan, work, record, *, keep_previous):
    """Stage, verify, swap. Crash-safe: a journal in the row says which phase."""
    row = Path(row)
    run_id = plan['run_id']
    stage = row/f'.reseal-stage-{run_id}'
    previous = row/f'.reseal-previous-{run_id}'
    journal = row/'identity_migration.json'
    if stage.exists() or previous.exists():
        raise Refused(f'{row}: leftover {stage.name} or {previous.name}; inspect and remove before migrating')
    started = time.time()
    entry = dict(schema=JOURNAL_SCHEMA, run_id=run_id, phase='staging', started_unix=started, record=record)
    write_json(journal, entry)
    manifest = dict(work['manifest'])
    manifest['identity'] = json.loads(canonical_bytes(work['new_identity']).decode('utf-8'))
    manifest['identity_sha256'] = work['new_sha']
    manifest['identity_migration'] = list(manifest.get('identity_migration') or []) + [record]
    new_manifest_raw = manifest_bytes(manifest)
    atomic_write_bytes(stage/MANIFEST_NAME, new_manifest_raw)
    written = len(new_manifest_raw)
    for shard in work['shards']:
        atomic_write_bytes(stage/PARTS_NAME/'units'/shard['path'].name, shard['new_raw'])
        written += len(shard['new_raw'])
    cost_raw_new = None
    if work['cost']:
        cost = work['cost']['cost']
        cost['provenance'] = dict(cost['provenance'])
        cost['provenance']['identity_migration'] = list(cost['provenance'].get('identity_migration') or []) + [record]
        cost_raw_new = dumps_like(cost, work['cost']['raw'])
        atomic_write_bytes(stage/COST_NAME, cost_raw_new)
        written += len(cost_raw_new)
    # verify the staged row exactly as a consumer would read it
    check = verify_row(stage, pins={'new': record['new_pins'], 'drop_settings': tuple(record.get('dropped_settings') or ())},
                       wire_root=row, wire_bytes=False, expect_record=record)
    if not check['ok']:
        raise Refused(f'{row}: staged row failed verification: {check["failures"][:3]}')
    if work['cost']:
        content = content_equality(work['cost']['raw'], cost_raw_new, encoder_moves=work['encoder_moves'])
        if not content['identical']:
            raise Refused(f'{row}: cost.pkl content is not byte-identical apart from provenance and seals: {content}')
        entry['cost_content'] = content
    entry.update(phase='swapping', staged_unix=time.time(), bytes_written=written)
    write_json(journal, entry)
    previous.mkdir()
    os.rename(row/MANIFEST_NAME, previous/MANIFEST_NAME)
    os.rename(row/PARTS_NAME, previous/PARTS_NAME)
    if work['cost']:
        os.rename(row/COST_NAME, previous/COST_NAME)
    # the staged parts dir holds only the shards this row has; the previous
    # parts dir may hold nothing else (units/ only), which the plan checked
    os.rename(stage/MANIFEST_NAME, row/MANIFEST_NAME)
    os.rename(stage/PARTS_NAME, row/PARTS_NAME)
    if work['cost']:
        os.rename(stage/COST_NAME, row/COST_NAME)
    stage.rmdir()
    entry.update(phase='swapped', swapped_unix=time.time(), previous=str(previous))
    write_json(journal, entry)
    if not keep_previous:
        shutil.rmtree(previous)
        entry.update(previous=None, previous_discarded_unix=time.time())
    entry.update(phase='done', finished_unix=time.time(), seconds=time.time()-started)
    write_json(journal, entry)
    return entry


def content_equality(old_raw, new_raw, *, encoder_moves):
    """Bit-identity of cost.pkl apart from provenance and the receipt seals."""
    def strip(raw):
        value = loads(raw)
        value.pop('provenance', None)
        for records in (value.get('tessera_expert_wires') or {}).values():
            for record in records.values():
                record.get('identity', {}).pop('encoder_source_sha256', None)
        return dumps_like(value, raw)
    a, b = strip(old_raw), strip(new_raw)
    return dict(identical=(a == b), compared_bytes=len(a), old_bytes=len(old_raw), new_bytes=len(new_raw),
                masked=['provenance'] + (['tessera_expert_wires[*][*].identity.encoder_source_sha256'] if encoder_moves else []))


# ---------------------------------------------------------------------------
# verify: read a row the way its consumers do
# ---------------------------------------------------------------------------

def verify_row(row, *, pins, wire_root=None, wire_bytes=True, expect_record=None, partial=None):
    row = Path(row)
    wire_root = Path(wire_root) if wire_root else row
    result = dict(row=str(row), failures=[], shards=0, missing_shards=0, wires=0, wire_bytes_checked=0, partial=partial)
    fail = result['failures'].append
    manifest = json.loads((row/MANIFEST_NAME).read_text())
    identity = manifest.get('identity', {})
    sha = identity_sha256(identity)
    if sha != manifest.get('identity_sha256'):
        fail(dict(what='identity_sha256', stored=manifest.get('identity_sha256'), recomputed=sha))
    for key in PIN_KEYS:
        if identity.get(key) != pins['new'][key]:
            fail(dict(what='pin', field=key, stored=identity.get(key), expected=pins['new'][key]))
    for key in pins.get('drop_settings') or ():
        if key in (identity.get('settings') or {}):
            fail(dict(what='dropped_setting_still_bound', field=key, stored=identity['settings'][key]))
    records = manifest.get('identity_migration') or []
    if expect_record is not None and (not records or records[-1] != expect_record):
        fail(dict(what='migration_record', detail='manifest lacks the expected identity_migration record'))
    parts = row/PARTS_NAME
    for entry in manifest['units']:
        path = parts/entry['file']
        if not path.is_file():
            # A withdrawn journal stops short of its unit list; a done row may not.
            result['missing_shards'] += 1
            if partial is False:
                fail(dict(what='shard_missing', file=path.name, qname=entry['qname']))
            continue
        raw = path.read_bytes()
        try:
            envelope = loads(raw)
        except Exception as error:  # noqa: BLE001
            fail(dict(what='shard', file=path.name, detail=repr(error)))
            continue
        for field, expected in (('schema', UNIT_SCHEMA), ('stage', STAGE), ('qname', entry['qname']), ('identity_sha256', manifest['identity_sha256'])):
            if envelope.get(field) != expected:
                fail(dict(what='envelope', file=path.name, field=field, stored=envelope.get(field), expected=expected))
        payload = envelope.get('payload', b'')
        if bytes_sha256hex(payload) != envelope.get('payload_sha256'):
            fail(dict(what='payload_sha256', file=path.name))
        state = loads(payload)
        result['shards'] += 1
        for fmt, record in (state.get('wire_records') or {}).items():
            result['wires'] += 1
            if record['identity'].get('encoder_source_sha256') != pins['new']['encoder_source_sha256']:
                fail(dict(what='receipt_seal', unit=entry['qname'], format=fmt, stored=record['identity'].get('encoder_source_sha256')))
            if wire_bytes:
                wire = wire_root/'cache'/'wire'/record['file']
                if not wire.is_file():
                    fail(dict(what='wire_missing', file=str(wire)))
                    continue
                size = wire.stat().st_size
                if size != record['blob_bytes'] or file_sha256hex(wire) != record['blob_sha256']:
                    fail(dict(what='wire_bytes', file=str(wire), size=size, expected=record['blob_bytes']))
                result['wire_bytes_checked'] += size
        anchors = {a['format_name'] for a in state.get('anchors', [])}
        if anchors != set(state.get('wire_records') or {}) and not state.get('unservable'):
            fail(dict(what='anchor_receipt_coverage', unit=entry['qname']))
    cost_path = row/COST_NAME
    if cost_path.is_file():
        cost = loads(cost_path.read_bytes())
        prov = cost.get('provenance', {})
        recs = prov.get('identity_migration') or []
        if expect_record is not None and (not recs or recs[-1] != expect_record):
            fail(dict(what='cost_migration_record'))
        if records and recs != records:
            fail(dict(what='migration_records_differ', detail='manifest and cost.pkl carry different identity_migration lists'))
        for unit, wires in (cost.get('tessera_expert_wires') or {}).items():
            for fmt, record in wires.items():
                if record['identity'].get('encoder_source_sha256') != pins['new']['encoder_source_sha256']:
                    fail(dict(what='cost_seal', unit=unit, format=fmt))
        result['cost_pkl'] = True
    result['ok'] = not result['failures']
    return result


def consumer_merge(rows, out_dir):
    """Merge verified row checkpoints through the campaign table owner."""
    from prismaquant.tessera_campaign_table import (merge_checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = merge_checkpoint({Path(r).name: Path(r) for r in rows}, out/MANIFEST_NAME)
    return dict(merged_identity_sha256=manifest['identity_sha256'], units=len(manifest['units']),
                identity_migration_carried=('identity_migration' in manifest))


def hash_trees(*, prismaquant_root=None, producer_root=None):
    out = {}
    if prismaquant_root:
        out['prismaquant_source_sha256'], out['prismaquant_files'] = prismaquant_tree_sha256(Path(prismaquant_root)/'prismaquant')
    if producer_root:
        profiles, count = encoder_tree_profiles(Path(producer_root)/'src'/'tessera')
        out['encoder_source_sha256'] = profiles[SOURCE_TREE_V1]
        out['encoder_files'] = count
        out['encoder_source_profiles'] = profiles
    return out


def dry_run(pins_path, *, row_paths=(), workspace=None, allow_live=False, checkpoint_audit=None, proof_path=None, on_row=None):
    pins = load_pins(pins_path)
    bundle = load_bundle(proof_path, pins) if proof_path else None
    audit_states = load_checkpoint_audit(checkpoint_audit) if checkpoint_audit else None
    rows = discover_rows(row_paths, workspace=workspace, checkpoint_audit=checkpoint_audit, allow_live=allow_live)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    report = dict(mode='dry-run', pins={'old': pins['old'], 'new': pins['new']}, drop_settings=list(pins['drop_settings']), proof=(bundle or {}).get('path'),
                  checkpoint_audit=checkpoint_audit, rows=[])
    if bundle is not None:
        report['coverage'] = require_coverage(rows, pins, bundle)
    for row in rows:
        started = time.time()
        plan, _ = plan_row(row, pins, run_id=run_id, audit_states=audit_states)
        plan['plan_seconds'] = time.time()-started
        report['rows'].append(plan)
        if on_row is not None:
            on_row(plan)
    return report


def migrate_rows(pins_path, *, proof_path, row_paths=(), workspace=None, allow_live=False, checkpoint_audit=None, operator=None, keep_previous=True, on_row=None):
    pins = load_pins(pins_path)
    if not proof_path:
        raise Refused('migrate requires --proof BUNDLE.json')
    bundle = load_bundle(proof_path, pins)
    audit_states = load_checkpoint_audit(checkpoint_audit) if checkpoint_audit else None
    rows = discover_rows(row_paths, workspace=workspace, checkpoint_audit=checkpoint_audit, allow_live=allow_live)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    operator = operator or getpass.getuser()
    report = dict(mode='migrate', run_id=run_id, pins={'old': pins['old'], 'new': pins['new']}, drop_settings=list(pins['drop_settings']), proof=bundle['path'],
                  proof_bundle_sha256=bundle['bundle_sha256'], checkpoint_audit=checkpoint_audit, rows=[])
    report['coverage'] = require_coverage(rows, pins, bundle)
    for row in rows:
        started = time.time()
        plan, work = plan_row(row, pins, run_id=run_id, audit_states=audit_states)
        if work is None:
            plan['seconds'] = time.time()-started
            report['rows'].append(plan)
            if on_row is not None:
                on_row(plan)
            continue
        record = migration_record(pins, bundle, plan, operator=operator, run_id=run_id, when=time.time())
        entry = migrate_row(row, plan, work, record, keep_previous=keep_previous)
        plan.update(migrated=True, seconds=time.time()-started, bytes_written=entry['bytes_written'], previous=entry.get('previous'),
                    cost_content=entry.get('cost_content'))
        report['rows'].append(plan)
        if on_row is not None:
            on_row(plan)
    return report


def verify_rows(pins_path, *, row_paths=(), workspace=None, allow_live=False, checkpoint_audit=None, wire_root=None, wire_bytes=True, consumer_merge_dir=None, on_row=None):
    pins = load_pins(pins_path)
    audit_states = load_checkpoint_audit(checkpoint_audit) if checkpoint_audit else None
    rows = discover_rows(row_paths, workspace=workspace, checkpoint_audit=checkpoint_audit, allow_live=allow_live)
    report = dict(mode='verify', checkpoint_audit=checkpoint_audit, rows=[])
    ok = True
    for row in rows:
        started = time.time()
        partial = None if audit_states is None else (row_kind(row, row/COST_NAME, audit_states) == 'partial')
        result = verify_row(row, pins=pins, wire_root=wire_root or row, wire_bytes=wire_bytes, partial=partial)
        result['seconds'] = time.time()-started
        ok &= result['ok']
        report['rows'].append(result)
        if on_row is not None:
            on_row(result)
    if consumer_merge_dir:
        try:
            report['consumer_merge'] = consumer_merge(rows, consumer_merge_dir)
        except Exception as error:  # noqa: BLE001
            ok = False
            report['consumer_merge'] = dict(error=repr(error))
    report['ok'] = ok
    return report
