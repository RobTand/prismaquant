"""experiments/reseal_identity_proof.py: where each import comes from, and the fused-path counters.

A GPU proof arm runs from a driver checkout (the PB snapshot, the container's
working directory) against a PrismaQuant tree under test (--prismaquant-root).
``prismaquant`` must come only from the tree under test; the ``experiments``
helpers that drive the arm must come only from the driver. ``experiments`` is a
namespace package, so binding the tree under test silently moved the helpers
too, and a tree without ``campaign_prefix_profile.py`` failed both prefix arms.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROOF = ROOT/'experiments'/'reseal_identity_proof.py'

# Run as ``python -m experiments.arm_probe`` from the driver checkout, the way
# the container runs ``python -m experiments.reseal_identity_proof``.
PROBE = textwrap.dedent('''
    import json, sys
    import experiments.reseal_identity_proof as proof
    root, mode = sys.argv[1], sys.argv[2]
    prismaquant = proof.bind_prismaquant(root)
    modules = None
    if mode == 'unpinned':
        import experiments.campaign_prefix_profile as helper
    else:
        if mode == 'stray':
            import experiments.prefix_dependency
        helper, modules = proof.import_prefix_helper()
    import experiments.prefix_dependency as dependency
    print(json.dumps(dict(prismaquant=prismaquant.__file__, helper=helper.__file__, dependency=dependency.__file__,
                          helper_saw=helper.DEPENDENCY, modules=modules)))
''')


def _layout(tmp_path):
    driver, root = tmp_path/'driver', tmp_path/'pq-root'
    (driver/'experiments').mkdir(parents=True)
    shutil.copy(PROOF, driver/'experiments'/PROOF.name)
    (driver/'experiments'/'campaign_prefix_profile.py').write_text(
        'from experiments.prefix_dependency import WHERE as DEPENDENCY\nimport prismaquant\n')
    (driver/'experiments'/'prefix_dependency.py').write_text("WHERE = 'driver'\n")
    (driver/'experiments'/'arm_probe.py').write_text(PROBE)
    # The driver's own prismaquant is never the one an arm measures.
    (driver/'prismaquant').mkdir()
    (driver/'prismaquant'/'__init__.py').write_text("raise ImportError('imported the driver checkout prismaquant')\n")
    (root/'prismaquant').mkdir(parents=True)
    (root/'prismaquant'/'__init__.py').write_text('')
    # The tree under test has an experiments/ of its own: no helper, and a
    # module of the same name as one the helper imports.
    (root/'experiments').mkdir()
    (root/'experiments'/'prefix_dependency.py').write_text("WHERE = 'pq-root'\n")
    return driver.resolve(), root.resolve()


def _probe(driver, root, mode):
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-m', 'experiments.arm_probe', str(root), mode],
                          cwd=driver, env=env, capture_output=True, text=True, timeout=120)


def test_binding_the_tree_under_test_moves_the_namespace_helpers_with_it(tmp_path):
    """The failure the prefix arms hit on PQ 9753a5b7c5 (PB b34af4c30baf, 29ea6471539e)."""
    driver, root = _layout(tmp_path)
    proc = _probe(driver, root, 'unpinned')
    assert proc.returncode != 0
    assert "No module named 'experiments.campaign_prefix_profile'" in proc.stderr


def test_the_prefix_helper_comes_from_the_driver_and_prismaquant_from_the_root(tmp_path):
    driver, root = _layout(tmp_path)
    proc = _probe(driver, root, 'pinned')
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(seen['prismaquant']) == root/'prismaquant'/'__init__.py'
    assert Path(seen['helper']) == driver/'experiments'/'campaign_prefix_profile.py'
    assert Path(seen['dependency']) == driver/'experiments'/'prefix_dependency.py'
    assert seen['helper_saw'] == 'driver'
    assert set(seen['modules']) >= {'experiments.campaign_prefix_profile', 'experiments.prefix_dependency',
                                   'experiments.reseal_identity_proof'}
    assert all(Path(m['file']).parent == driver/'experiments' for m in seen['modules'].values())


def test_a_helper_module_already_loaded_from_the_root_is_refused(tmp_path):
    driver, root = _layout(tmp_path)
    proc = _probe(driver, root, 'stray')
    assert proc.returncode != 0
    assert 'loaded from outside the driver checkout' in proc.stderr
    assert "experiments.prefix_dependency" in proc.stderr


def test_fused_engagement_counts_each_outcome_and_restores_the_module(monkeypatch):
    proof = importlib.import_module('experiments.reseal_identity_proof')
    module = types.ModuleType('fake_fused_paths')
    module.refusal = lambda where: None if where == 'cuda' else f'targets are on {where}'
    module.fused = lambda value: None if value == 'tripwire' else value
    module.reference = lambda: 'reference'
    monkeypatch.setitem(sys.modules, 'fake_fused_paths', module)
    originals = (module.refusal, module.fused, module.reference)
    engagement = proof.FusedEngagement(stats=None, encode=None, probes=(
        ('fake_fused_paths', 'refusal', 'refusal'), ('fake_fused_paths', 'fused', 'fused'),
        ('fake_fused_paths', 'reference', 'reference'), ('fake_fused_paths', 'missing', 'fused'),
        ('no_such_fused_module_anywhere', 'refusal', 'refusal'))).install()
    assert module.refusal('cuda') is None and module.refusal('cpu') == 'targets are on cpu'
    # One reason, two ranks: one class, both spellings kept verbatim.
    assert module.refusal('rank 3') == 'targets are on rank 3' and module.refusal('rank 12') == 'targets are on rank 12'
    assert module.fused('tripwire') is None and module.fused('bytes') == 'bytes'
    assert module.reference() == 'reference' and module.reference() == 'reference'
    record = engagement.record()
    assert record['counts'] == {'fake_fused_paths.refusal': {'admitted': 1, 'targets are on cpu': 1,
                                                             'targets are on rank <n>': 2},
                                'fake_fused_paths.fused': {'fell_back': 1, 'ran': 1},
                                'fake_fused_paths.reference': {'calls': 2}}
    assert record['refusals_verbatim']['fake_fused_paths.refusal'] == {
        'targets are on cpu': 1, 'targets are on rank 3': 1, 'targets are on rank 12': 1}
    assert record['probes']['fake_fused_paths.missing'].startswith('absent')
    assert record['probes']['no_such_fused_module_anywhere.refusal'].startswith('absent')
    engagement.uninstall()
    assert (module.refusal, module.fused, module.reference) == originals


def _fake_tessera(monkeypatch):
    """Stand-ins for tessera.tcq_fused, tessera.lut_fused and tessera.export under the names the summary reads.

    ``encode_linear`` does per fit what tessera.encode does: ask each gate on
    its module at call time and run the fused function when admitted.
    """
    tcq, lut, export = (types.ModuleType(name) for name in ('tessera.tcq_fused', 'tessera.lut_fused', 'tessera.export'))
    lut.STATS = {'fused': 0, 'tripped': 0, 'nonfinite': 0}
    tcq.tcq_fused_refusal = lambda device: None if device == 'cuda' else f'targets are on {device}'
    tcq.viterbi_columns_fused = lambda device: 'trellis'
    lut.lut_swap_refusal = lambda device: None if device == 'cuda' else 'torch 2.13.0+cu130: checked on 2.11.x only'

    def swap_passes_fused(ok):
        lut.STATS['fused' if ok else 'tripped'] += 1
        return 'table' if ok else None
    lut.swap_passes_fused = swap_passes_fused

    def encode_linear(device):
        if tcq.tcq_fused_refusal(device) is None:
            tcq.viterbi_columns_fused(device)
        if lut.lut_swap_refusal(device) is None:
            lut.swap_passes_fused(True)
        return device
    export.encode_linear = encode_linear
    export.encode_linears = lambda devices: [export.encode_linear(device) for device in devices]
    for module in (tcq, lut, export):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return tcq, lut, export


def test_the_fused_stages_are_counted_inside_the_encode_window(monkeypatch):
    """Gate answers and STATS before the first encode or after the last are not the arm's."""
    proof = importlib.import_module('experiments.reseal_identity_proof')
    tcq, lut, export = _fake_tessera(monkeypatch)
    originals = (tcq.tcq_fused_refusal, lut.swap_passes_fused, export.encode_linear, export.encode_linears)
    probes = tuple(probe for probe in proof.FUSED_PROBES if probe[0] != 'tessera.encode')
    lut.STATS['fused'] = 7  # fits from before this arm
    engagement = proof.FusedEngagement(probes=probes).install()
    assert engagement.installed['tessera.export.encode_linear_planes'].startswith('absent')
    # Outside any encode: a CPU fixture matrix refused by both gates, and a stray fused fit.
    tcq.tcq_fused_refusal('cpu'), lut.lut_swap_refusal('cpu'), lut.swap_passes_fused(True)
    assert export.encode_linears(['cuda', 'cuda', 'cuda']) == ['cuda'] * 3
    assert export.encode_linear('cuda') == 'cuda'
    # After the last encode: a refusal and a tripped fit the arm did not encode.
    tcq.tcq_fused_refusal('cpu'), lut.swap_passes_fused(False)
    record = engagement.record()
    summary = record['summary']
    assert [(w['entry'], w['units']) for w in record['encode_windows']] == [
        ('tessera.export.encode_linears', 3), ('tessera.export.encode_linear', 1)]
    assert (summary['encode_windows'], summary['encode_units']) == (2, 4)
    assert (summary['stage1_admitted'], summary['stage1_refused'], summary['stage1_fused']) == (4, 0, 4)
    assert (summary['stage2_admitted'], summary['stage2_refused'], summary['stage2_fused']) == (4, 0, 4)
    assert (summary['stage2_tripped'], summary['stage2_nonfinite']) == (0, 0)
    assert summary['stage1_counts_agree'] and summary['stage2_counts_agree'] and summary['both_stages']
    assert record['stats']['before_first_encode']['fused'] == 8 and record['stats']['after_last_encode']['fused'] == 12
    # The process saw both CPU refusals and the trip; the window saw none of them.
    assert record['counts']['tessera.tcq_fused.tcq_fused_refusal']['targets are on cpu'] == 2 and lut.STATS['tripped'] == 1
    engagement.uninstall()
    assert (tcq.tcq_fused_refusal, lut.swap_passes_fused, export.encode_linear, export.encode_linears) == originals
    # A refusal inside the window is counted, with its reason.
    engagement = proof.FusedEngagement(probes=probes).install()
    export.encode_linears(['cuda', 'cpu'])
    summary = engagement.record()['summary']
    engagement.uninstall()
    assert (summary['stage1_admitted'], summary['stage1_refused']) == (1, 1)
    assert summary['stage2_refusals'] == {'torch <n>.<n>.<n>+cu<n>: checked on <n>.<n>.x only': 1}
    # No encode at all: nothing is scoped, and stage 2 cannot be read.
    idle = proof.FusedEngagement(probes=probes).install()
    summary = idle.record()['summary']
    idle.uninstall()
    assert summary['encode_windows'] == 0 and summary['stage2_fused'] is None and not summary['engaged']
