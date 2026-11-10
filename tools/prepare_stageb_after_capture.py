#!/usr/bin/env python3
"""Publish the GLM Stage B metadata generation after Stage A's capture completes, as one native Flux job.

Writes the preparation submission (the read manifest and strict-read flags, through
``tools.stage_b_preparation_submission prepare``), then submits ``tools.prepare_extended_joint_quanta``
as one CPU job through ``flux_jobs.submit_job`` with the caller's explicit staged launch binding and
produced-output declaration. It publishes metadata only (catalog extension, extended parent manifest,
layer records and a launch recipe under FRESH_METADATA_ROOT); it submits no Stage B quantum.

``--dry-run`` validates the whole stage (source identity, resources, staged binding and its exact read
coverage, declaration) with ``flux_jobs.validate_job`` and submits nothing.

Arguments: PAIR_JSON PAIR_SHA CAPTURE_JSON CAPTURE_SHA FRESH_METADATA_ROOT, plus
``--staged-binding FILE`` (the versioned staged launch binding whose read contract is the submission's
read manifest). The produced-output declaration is the submission's own template, which
``stage_b_preparation_submission`` writes with its ``native_argv``.
Environment overrides: PQ_CHECKOUT, PQ_PYTHON, STAGEB_SPEC, STAGEB_SPEC_SHA256.
The deadline is the job's finite walltime. Immutable staged coverage is checked once;
there is no staged-range wait or wait-vs-progress-grace coupling.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant import flux_jobs  # noqa: E402
from prismaquant.digests import DIRECT_ASCII_SPACED_LAX, bytes_sha256hex  # noqa: E402

CAMPAIGN = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913')
PANEL = CAMPAIGN / 'allocation/joint-panel'
DEFAULT_SPEC = CAMPAIGN / 'campaign-final-union-20260922/stage-b-spec-local-scratch.json'
DEFAULT_SPEC_SHA = 'a52b53594cae071dd0f38616674acd9072b4b85f3ebd447f8fbdfd3aa1f8e2b9'
PARENT_MANIFEST = PANEL / 'complete-512-seed237.executed-group.r607.a2v4.encoder-reuse-02/data-manifests/prismaquant.tessera_joint_aura.run.json.gz'
PARENT_MANIFEST_SHA = 'd6481239715c44c8f3c9972708f5fcc8f533d175f49a2163deba9cdb6e0a3f55'
DERIVATION = PANEL / 'stage-a-recovery-20260920/records/derivation.json'
DERIVATION_SHA = '50bc8899938b1f6e61bc34efdd7752ab8c8a016889a3994481de3f3848ea79e3'
RESOURCES = flux_jobs.FluxResources(cores=4, memory_gib=16, gpus=0, walltime_s=1800, placement='portable')
THREADS = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


def resolve_pin(checkout: Path) -> str:
    return subprocess.run(['python3', str(checkout / 'tools/resolve_tessera_dev_pin.py')], check=True,
                          capture_output=True, text=True).stdout.strip()


def preparation_arguments(args, spec: str, spec_sha: str) -> list[str]:
    return ['--pair-inputs', args.pair_json, '--pair-inputs-sha256', args.pair_sha,
            '--adjoint-receipt', args.capture_json, '--adjoint-receipt-sha256', args.capture_sha,
            '--metadata-root', str(args.metadata_root),
            '--parent-manifest', str(PARENT_MANIFEST), '--parent-manifest-sha256', PARENT_MANIFEST_SHA,
            '--derivation', str(DERIVATION), '--derivation-sha256', DERIVATION_SHA,
            '--spec', spec, '--spec-sha256', spec_sha]


def plan_job(args, environ=os.environ, run=subprocess.run) -> dict:
    """Everything up to the submission: the submission directory, the staged binding check and the job argv."""
    checkout = Path(environ.get('PQ_CHECKOUT') or ROOT).resolve()
    python = environ.get('PQ_PYTHON') or f'/home/rob/venvs/pq-tessera-{resolve_pin(checkout)[:8]}/bin/python'
    spec = environ.get('STAGEB_SPEC') or str(DEFAULT_SPEC)
    spec_sha = environ.get('STAGEB_SPEC_SHA256') or DEFAULT_SPEC_SHA
    prep = preparation_arguments(args, spec, spec_sha)
    submission = Path(str(args.metadata_root).rstrip('/') + '.submission')
    run([python, '-m', 'tools.stage_b_preparation_submission', 'prepare', *prep, '--out', str(submission)],
        cwd=checkout, check=True, stdout=subprocess.DEVNULL)
    record = json.loads((submission / 'submission.json').read_text())
    strict, declared = record['strict_read_flags'], record['native_argv']
    if not strict or not declared:
        raise ValueError('the submission declares no strict read flags or no produced-output declaration')
    binding = json.loads(Path(args.staged_binding).read_text())
    if binding.get('read_contract_path') != str(submission / 'read-manifest.json.gz'):
        raise ValueError('the staged binding does not name this submission\'s read manifest')
    argv = [python, '-m', 'tools.prepare_extended_joint_quanta', *prep, '--produced-output', *declared, *strict]
    return {'checkout': checkout, 'argv': argv, 'binding': binding, 'submission': submission}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for name in ('pair_json', 'pair_sha', 'capture_json', 'capture_sha'):
        parser.add_argument(name)
    parser.add_argument('metadata_root', type=Path)
    parser.add_argument('--staged-binding', required=True, type=Path)
    parser.add_argument('--record-dir', type=Path, help='shared directory for the native submission record')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    plan = plan_job(args)
    flux_jobs.validate_job(checkout=plan['checkout'], resources=RESOURCES, environment=THREADS, staged_input=plan['binding'])
    if args.dry_run:
        print(DIRECT_ASCII_SPACED_LAX.text({'status': 'validated', 'argv_sha256': bytes_sha256hex(json.dumps(plan['argv']).encode()),
                                            'resources': RESOURCES.__dict__}))
        return 0
    if args.record_dir is None:
        parser.error('--record-dir is required to submit')
    submission = flux_jobs.submit_job(plan['argv'], checkout=plan['checkout'], resources=RESOURCES,
                                      record_dir=args.record_dir, environment=THREADS, staged_input=plan['binding'])
    print(DIRECT_ASCII_SPACED_LAX.text({'status': 'submitted', 'job_id': submission.job_id, 'record_dir': str(submission.record_dir)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
