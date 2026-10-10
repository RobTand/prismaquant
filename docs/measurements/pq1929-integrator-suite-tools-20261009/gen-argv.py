"""D29 argv: x86 + /tmp for most shards; known spill-guard files on dl380g10's btrfs root (tmpfs refused)."""
import glob, json, os, sys
inv = '/home/rob/fleet/inventory/'
name, checkout = sys.argv[1], sys.argv[2]
u = json.load(open(inv + 'pq-integrator-batch25-untagged-argv-20261005.json'))
p = json.load(open(inv + 'pq-integrator-batch25-pinned-argv-20261005.json'))
ut = [a for a in u if a.startswith('tests/')]; pt = [a for a in p if a.startswith('tests/')]
os.chdir('/home/rob/tmp/pq-next-batch25'); b25 = set(glob.glob('tests/test_*.py')); excluded = b25 - set(ut) - set(pt)
spill = ['test_band_serial_batched_regime','test_band_serial_spill','test_checkpoint_staged_readset_1366','test_compute_tail_grace_1190','test_digest_prepare_io_1301','test_glm_tr3_dual_teacher','test_spill_readplan_metadata_2115','test_stage_b_spill_ceiling_sealed','test_stageb_checkpoint_incoming_1802','test_stageb_host_staging_1246','test_stageb_one_pass_spill','test_stageb_scatter_reads_1794','test_stageb_spill_records_1086','test_staged_wait_phase_1166']
spill = ['tests/%s.py' % s for s in spill]
# 15 further files that failed on tmpfs in batch26b (guards: perturbed_x_cache 1510/1164, PrismaBuild produced_spool) plus test_ci_temp_root_1454
spill += ['tests/test_band_serial_handoff_disposal_real_pb.py','tests/test_band_serial_handoff_spool_real_pb.py','tests/test_band_serial_scratch_plane.py','tests/test_campaign_recovery_scratch_seam.py','tests/test_ci_temp_root_1454.py','tests/test_compute_phase_grace_1165.py','tests/test_pb905_real_produced_lifecycle.py','tests/test_produced_output_spool_real_pb.py','tests/test_resume_capture_phase_1172.py','tests/test_stage_b_kernel_profile_scope.py','tests/test_stage_b_streamed_handoff.py','tests/test_stage_b_streamed_incoming_1143.py','tests/test_stageb_cotangent_scratch.py','tests/test_stageb_spill_integrity_1369.py','tests/test_strict_reader_tier_enforcement.py']
os.chdir(checkout); top = sorted(glob.glob('tests/test_*.py'))
pinned = [t for t in top if t in set(pt)]
spillf = [t for t in top if t in set(spill) and t not in set(pt)]
untag = [t for t in top if t not in set(pt) and t not in excluded and t not in set(spill)]
print(name, len(untag), len(pinned), len(spillf), 'new', sorted(set(top) - b25))
MEM = {'untagged': 8, 'pinned': 5, 'spill': 5}  # 1.5 x measured peak RSS (6.4G, 2.8G, 2.9G) rounded up; CEO right-sizing rule
def build(src, files, out, shards, tag, tmpdir, mem):
    a = list(src); i = next(k for k, x in enumerate(a) if x.startswith('tests/')); a = a[:i]
    def setv(f, v): a[a.index(f) + 1] = v
    setv('--python', os.environ.get('PQ_PYTHON', '/home/rob/venvs/pq-d13-candidate-2dbac191/bin/python'));  # PQ_PYTHON overrides D13 (pin PR 2429 uses the pq-pin-fca4c6ce0 overlay)  # D13: the batch25 base argv names an older venv (batch74 wrong-interpreter incident)
    setv('--checkout', checkout); setv('--json', out); setv('--shards', str(shards)); setv('--max-clients', str(min(shards, 2 if shards > 2 else shards)))  # celestia 30 GB: suite total <=4 clients (2 untagged + pinned + spill), 14:30Z OOM
    setv('--history', inv + 'pq-integrator-batch25-untagged-20261005.json'); setv('--timeout-s', '3600'); setv('--wait-s', '9000'); setv("--mem-gb", str(mem))
    pa = json.loads(a[a.index('--pytest-args') + 1])
    if os.environ.get('PQ_RETENTION_FLAG') == '1' and 'tmp_path_retention_policy=failed' not in pa:  # pbtest refuses -o (PYTEST_VALUES allowlist)
        pa += ['-o', 'tmp_path_retention_policy=failed']  # D29: free passing tmp dirs at once (inodes)
    setv('--pytest-args', json.dumps(pa))
    if '--tag' in a: setv('--tag', tag)
    else: a += ['--tag', tag]
    if tmpdir: a += ['--tmpdir', tmpdir]
    return a + files
jobs = [('untagged', u, untag, 12, 'x86', '/tmp'), ('pinned', p, pinned, 1, 'dl380g10', '/tmp'), ('spill', p, spillf, 1, 'dl380g10', None)]
for kind, src, files, sh, tag, td in jobs:
    out = f'{inv}pq-integrator-{name}-{kind}-20261005.json'
    json.dump(build(src, files, out, sh, tag, td, MEM[kind]), open(f'{inv}pq-integrator-{name}-{kind}-argv-20261005.json', 'w'))
