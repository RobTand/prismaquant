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
    engagement = proof.FusedEngagement(probes=(
        ('fake_fused_paths', 'refusal', 'refusal'), ('fake_fused_paths', 'fused', 'fused'),
        ('fake_fused_paths', 'reference', 'reference'), ('fake_fused_paths', 'missing', 'fused'),
        ('no_such_fused_module_anywhere', 'refusal', 'refusal'))).install()
    assert module.refusal('cuda') is None and module.refusal('cpu') == 'targets are on cpu'
    assert module.fused('tripwire') is None and module.fused('bytes') == 'bytes'
    assert module.reference() == 'reference' and module.reference() == 'reference'
    record = engagement.record()
    assert record['counts'] == {'fake_fused_paths.refusal': {'admitted': 1, 'targets are on cpu': 1},
                                'fake_fused_paths.fused': {'fell_back': 1, 'ran': 1},
                                'fake_fused_paths.reference': {'calls': 2}}
    assert record['probes']['fake_fused_paths.missing'].startswith('absent')
    assert record['probes']['no_such_fused_module_anywhere.refusal'].startswith('absent')
    engagement.uninstall()
    assert (module.refusal, module.fused, module.reference) == originals


def test_the_default_probes_summarize_the_tessera_paths(monkeypatch):
    proof = importlib.import_module('experiments.reseal_identity_proof')
    engagement = proof.FusedEngagement(probes=())
    engagement.counts = {'tessera.lut_fused.swap_passes_fused': {'ran': 3, 'fell_back': 1},
                         'tessera.lut_fused.lut_swap_refusal': {'admitted': 4, 'torch 2.13.0 is verified on 2.11.x only': 2},
                         'tessera.encode._lut_swap_passes_reference': {'calls': 3}}
    summary = engagement.record()['summary']
    assert summary['lut_fused_ran'] == 3 and summary['tcq_fused_ran'] == 0 and summary['engaged']
    assert summary['lut_reference_calls'] == 3
    assert summary['lut_refusals'] == {'torch 2.13.0 is verified on 2.11.x only': 2}
    assert not proof.FusedEngagement(probes=()).record()['summary']['engaged']
