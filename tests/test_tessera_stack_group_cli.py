"""A selected-source campaign CLI row over a sampled ``s:`` stack group.

RobTand/prismaquant#389: every CLI test in this tree drives dense or fused
anchor groups, so the completed-anchor page release in
``tessera_campaign._finish_anchor`` had no end-to-end cover.  That branch is
guarded by the production cache's ``release_completed_anchor_file_pages``
metadata, which only a selected-source row sets. The branch requests page
release for the render shard and the ``.tessera`` wire of each completed rung;
this test observes the advice call and file identity, not physical eviction.

This module runs the real census, the real capture and then the real
selected-source row of ``prismaquant.tessera_campaign.main`` on the tiny GLM
checkpoint, with a selection the real planner drew from a packed probe, so the
stack path reaches the release branch on the bytes the row actually wrote.
"""
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from test_glm_campaign_streaming import (  # noqa: E402,F401
    glm_checkpoint, write_original_layout_checkpoint,
)
from test_tessera_stack_sample_cost import _packed_probe_row  # noqa: E402

#: The producer the census, the capture and every encode in this module run
#: against, resolved the way ``test_glm_campaign_streaming`` resolves it.
PINNED_PRODUCER = ('/mnt/shared/tessera-measurements/first-model-20260907'
                   '/inputs/tessera-382a1a97')


def _campaign_source(tmp_path):
    """The bumped tiny GLM checkpoint the streamed campaign fixture uses."""
    from test_glm5_next_streamed_forward_parity import _build_model, _tiny_config

    config = _tiny_config()
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.moe_intermediate_size = 256
    config.vision_config.out_hidden_size = 256
    config = type(config).from_dict(config.to_dict())
    source = tmp_path / 'campaign-source'
    model = _build_model(config).to(torch.bfloat16)
    write_original_layout_checkpoint(model, source)
    return model, source


def _stack_selection(tmp_path, census_payload, model, source):
    """The planner's own sampled ``s:`` row, drawn from a packed probe.

    The draw goes through ``dispatch.cmd_plan`` rather than a hand written
    selection so the file under test is the one the dispatcher writes, with
    the schema, the full frame inclusion probabilities and the replayable
    receipt a campaign refuses without.
    """
    import dispatch_tessera_campaign as dispatch
    from prismaquant import tessera_campaign as campaign
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    profile = Glm5NextProfile()
    population = campaign._require_campaign_population(model, profile, 1)
    assert population.packed_in_scope, 'the GLM fixture must declare a packed stack'
    module_of = {member.packed_qname: member.module_qname
                 for member in population.members}
    stats = {}
    for name, shape in sorted(population.packed_in_scope.items()):
        row = _packed_probe_row(shape[0], [1.0] * shape[0],
                                packed_param=name.rsplit('.', 1)[-1],
                                out_features=shape[1], in_features=shape[2])
        row['_packed_experts_module'] = module_of[name]
        stats[name] = row

    workspace = tmp_path / 'plan'
    workspace.mkdir()
    (workspace / 'census.json').write_text(json.dumps(census_payload))
    probe = workspace / 'probe.pkl'
    probe.write_bytes(pickle.dumps({'stats': stats, 'meta': {'model': str(source)}}))
    spec = workspace / 'spec.json'
    spec.write_text(json.dumps({'model': str(source), 'cwd': str(workspace),
                                'python': 'python', 'env': {}, 'campaign_argv': []}))
    args = SimpleNamespace(spec=spec, workspace=workspace, stack_sample=2,
                           probe=probe, stack_sample_seed=5, audit_rate=10,
                           groups_per_row=1, seed_checkpoint=None,
                           seed_wire_dir=None, rows_per_box=1, timeout_s=300)
    assert dispatch.cmd_plan(args) == 0

    rows = []
    for path in sorted((workspace / 'units').glob('row-*.json')):
        selection = json.loads(path.read_text())
        if any(str(entry['key']).startswith('s:') for entry in selection['groups']):
            rows.append((path, selection))
    assert len(rows) == 1, 'the fixture must plan exactly one stack row'
    path, selection = rows[0]
    assert selection['schema'] == campaign.UNITS_SCHEMA_V2
    entry, = selection['groups']
    assert entry['sampled'] and set(entry['sampled']) < set(entry['members'])
    return path, selection


def test_selected_source_row_prices_a_sampled_stack_and_releases_each_anchor(
        glm_checkpoint, tmp_path, monkeypatch):
    """The stack row completes, and every completed anchor is released.

    One test rather than two because the release evidence is a property of a
    row that ran: splitting it would pay for a second census, capture and
    encode to assert half of the same run.
    """
    from prismaquant import tessera_calibration_cache as capture
    from prismaquant import perturbed_x_cache
    from prismaquant import tessera_campaign as campaign
    from prismaquant.tessera_expert_projection import EXPERT_WIRES_KEY

    producer = Path(os.environ.get('TESSERA_REPO') or PINNED_PRODUCER)
    if not producer.is_dir():
        pytest.skip('TESSERA_REPO must name the pinned producer checkout '
                    f'(unset, and {PINNED_PRODUCER} is absent)')
    monkeypatch.setenv('TESSERA_REPO', str(producer))

    model, source = _campaign_source(tmp_path)
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens',
                        lambda *_: (tokens, 'tiny GLM frozen draw'))
    common = ['--model', str(source), '--out', str(tmp_path / 'unused.pkl'),
              '--menu-mode', 'research', '--nsamples', '1', '--seqlen', '257',
              '--max-act-rows', '7', '--attention-implementation', 'eager',
              '--streaming', '--streaming-cache-headroom-gb', '0']

    census = tmp_path / 'census.json'
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'census-cache'),
                          '--census-out', str(census)]) == 0
    capture_root = tmp_path / 'capture'
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'capture-cache'),
                          '--calibration-census', str(census),
                          '--capture-calibration-out', str(capture_root)]) == 0
    manifest = capture_root / 'capture_manifest.json'
    capture.require_capture_contract(manifest)
    assert json.loads(manifest.read_text())['status'] == 'complete'

    census_payload = json.loads(census.read_text())
    units_path, selection = _stack_selection(tmp_path, census_payload, model, source)
    priced = sorted(selection['groups'][0]['sampled'])

    # Two rungs, not one: with a single rung the row cannot say whether it
    # walks its pending list unit major or rung major, which is the diagnostic
    # the follow-up reorder work needs. They come from one family so the
    # group's realisable set is exactly these two and no audit unit earns an
    # interior third anchor.
    chosen: dict = {}
    original_menus = campaign.expand_menus_for_targets

    def two_rungs(weights, targets, **kwargs):
        from prismaquant import format_registry as fr
        from prismaquant.tessera_formats import parse_tessera_format_name

        menus = original_menus(weights, targets, **kwargs)
        assert set(menus) == set(priced), 'the selection must narrow the scope'
        if not chosen:
            shared = set.intersection(*[{row.format_name for row in rows}
                                        for rows in menus.values()])
            by_family: dict = {}
            for name in sorted(shared):
                if fr.get_format(name).static_activation_contract is not None:
                    # A static contract rung needs a calibrated input scale to
                    # be priced honestly; the dynamic families need nothing
                    # this fixture does not already have.
                    continue
                family, rung = parse_tessera_format_name(name)
                by_family.setdefault(family.name, []).append((int(rung), name))
            assert by_family, 'the fixture must admit a dynamic activation family'
            widest = max(sorted(by_family), key=lambda key: len(by_family[key]))
            chosen['formats'] = [name for _, name in sorted(by_family[widest])[:2]]
        keep = set(chosen['formats'])
        narrowed = {name: [row for row in rows if row.format_name in keep]
                    for name, rows in menus.items()}
        assert all(len(rows) == len(keep) for rows in narrowed.values())
        return narrowed

    def no_forward(*args, **kwargs):
        pytest.fail('selected anchor reuse repeated calibration')

    monkeypatch.setattr(campaign, 'expand_menus_for_targets', two_rungs)
    monkeypatch.setattr(campaign, '_collect_activations', no_forward)

    # The release path, recorded and called through. The wrapper checks the
    # caller's ``expected_stat`` against the file itself, so a call that names
    # a stale identity is visible here and not only inside the helper.
    selected_cache = tmp_path / 'selected-cache'
    released: list = []
    original_release = perturbed_x_cache.release_activation_cache_file_pages

    def record_release(path, *, expected_stat):
        target = Path(path)
        if target.is_relative_to(selected_cache):
            actual = os.stat(target)
            same = all(getattr(actual, field) == getattr(expected_stat, field)
                       for field in ('st_dev', 'st_ino', 'st_size',
                                     'st_mtime_ns', 'st_ctime_ns'))
            released.append((target, same))
        return original_release(path, expected_stat=expected_stat)

    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages',
                        record_release)

    # Two recorders that carry the reorder diagnostic and assert nothing.
    memo_holder: dict = {}
    original_memo = campaign._activation_kwargs_memo

    def memo(*args, **kwargs):
        for_unit = original_memo(*args, **kwargs)
        memo_holder['for_unit'] = for_unit
        return for_unit

    order: list = []
    original_measure = campaign._measure_anchor

    def measure(*, qname, format_name, **kwargs):
        order.append((qname, format_name))
        return original_measure(qname=qname, format_name=format_name, **kwargs)

    monkeypatch.setattr(campaign, '_activation_kwargs_memo', memo)
    monkeypatch.setattr(campaign, '_measure_anchor', measure)

    selected_out = tmp_path / 'selected-cost.pkl'
    assert campaign.main([*common, '--out', str(selected_out),
                          '--cache-dir', str(selected_cache),
                          '--units', str(units_path),
                          '--calibration-census', str(census),
                          '--calibration-cache', str(manifest),
                          '--calibration-cache-sha256', capture.sha256(manifest),
                          '--max-rounds', '1']) == 0

    # Reported for RobTand/prismaquant#389's reorder half, never asserted: the
    # encoder memo holds one entry per selected-source row, so a factorization
    # per unit means the row walked its pending list unit major.
    info = memo_holder['for_unit'].cache_info()
    print(f"[389] pending order {order}")
    print(f"[389] encoder factorizations {info.misses} over "
          f"{len({name for name, _ in order})} distinct units, memo {info}")

    with selected_out.open('rb') as handle:
        payload = pickle.load(handle)
    receipt = payload['provenance']['selected_source_preparation']
    assert receipt['source_forward_count'] == 0

    # (a) The row carries the packed population for the stack, the way the
    # persisted-draw driver test states it: the priced rows are the stack's
    # packed parameters, never its members.
    stacks = sorted(selection['groups'][0]['stack_samples'])
    assert set(payload['costs']) == set(stacks) and stacks
    assert payload['provenance']['population']['priced']['routed_experts'] == stacks
    assert payload['provenance']['unit_selection']['groups'] == selection['groups']
    checkpoint = json.loads(selected_out.with_suffix('.anchors.json').read_text())
    assert set(checkpoint['identity']['stack_sampling_identity']) == set(stacks)
    for stack in stacks:
        assert set(payload['costs'][stack]) == set(chosen['formats'])

    # (d) The anchors the row was asked for, derived from the selection's own
    # sampled members and the rungs the menu admitted, not from a count.
    expected = {(name, fmt) for name in priced for fmt in chosen['formats']}
    assert {(name, fmt) for name, rows in payload[EXPERT_WIRES_KEY].items()
            for fmt in rows} == expected
    assert set(order) == expected

    # (b) and (c): both files of every completed anchor exist beside each
    # other and were released once, each against its own identity.
    wire_dir = selected_cache / 'wire'
    from prismaquant import format_registry as fr
    from prismaquant.production_weight_cache import _cache_weight_filename

    want = set()
    for name, fmt in sorted(expected):
        wire = campaign._wire_path(wire_dir, name, fmt)
        rendered = selected_cache / _cache_weight_filename(
            name, fr.canonical_format_name(fmt.strip().upper()))
        assert wire.is_file() and rendered.is_file()
        want.update({wire.resolve(), rendered.resolve()})
    seen = Counter(path.resolve() for path, _ in released)
    assert {path: seen[path] for path in want} == {path: 1 for path in want}
    assert all(same for _, same in released), released
    # The only other advised files in this row belong to the export-input
    # Hessian sidecar writer, which releases from its own site; anything else
    # would be an unaccounted owner of the same page cache.
    others = sorted(path for path in seen if path not in want)
    assert all(path.name.startswith('hessian_capture') for path in others), others
    print(f"[389] release calls {sum(seen.values())} over "
          f"{len(want)} anchor files, other advised files {others}")
