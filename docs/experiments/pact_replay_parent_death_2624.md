# PACT replay parent-death cleanup (#2624 / PR #2632)

## Contract and repair

`supervise` captures its own PID before forking and passes it to `_worker`.
The worker arms `PR_SET_PDEATHSIG=SIGKILL` immediately after `setsid`, then
compares `getppid()` against that pre-fork identity. Sampling the expected
identity in the child was insufficient: a supervisor killed before that sample
could already have been replaced by an adopter, including a subreaper.

The regression driver runs in a fresh interpreter and adopts/reaps orphaned
workers itself, so PID/group disappearance does not depend on PID 1's reaping
behavior. The startup case stops the child before the original `_worker`, kills
the supervisor, verifies adoption by the driver, then resumes the child. Work
must never start. The initialized native-block and steady-progress cases require
an actual SIGKILL wait status as well as removal of the PID and process group;
a pipe error or normal return cannot satisfy them. Native-blocking cases require
visible Linux `/proc` wait channels and explicitly skip if those are hidden;
such a skip, like a CUDA skip, is not native-blocking/GPU evidence. An outer
timeout also kills the supervisor group and the separately sessioned worker.

## GPU preflight

Decision: does a replay worker with a live CUDA allocation also disappear after
its supervisor receives SIGKILL while the worker is inside a native read?
CPU tests cannot establish that a real GPU-holding process is cleaned up.

Smallest case: one worker allocates and synchronizes a 1 MiB CUDA tensor, keeps
it live across `os.read`, and reports positive allocated CUDA bytes. The driver
observes the native pipe-read wait channel before killing only the supervisor.
The worker retains both pipe ends, so parent death cannot cause EOF. The driver
must reap a SIGKILLed worker and observe that its PID and process group are gone
before failure-cleanup signals are permitted. This proves process removal, not
an independent measurement of driver memory-reclamation latency or cleanup of
arbitrary worker descendants.

Planned targeted run: `run_tests(arm="gpu")` with
`python -m pytest tests/test_pact_replay_pdeathsig_2624.py::test_sigkill_parent_leaves_no_gpu_holding_worker -q -ra --basetemp="$TMPDIR/pytest"`.
Expected GPU usage: under one minute, one allocation, no model/checkpoint,
serving runtime, campaign, sweep, or performance claim. Checks use the supplied
admitted PrismaBuild runner directly; no nested submission tools are used.

The initial targeted GPU run passed in 2.72 seconds. After tightening the test
harness's unavailable-wait-channel handling and outer-timeout cleanup, repeat
only this same targeted case to verify the final harness. The decision,
allocation, and CPU limitation are unchanged; budget under one additional GPU
minute, with no sweep.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9bdba00cb0ea3eeb22eb4533f7d9df4ecb3c92cff29e623285666de1c90b1687","size":764}},"operation":"c46e35a7a6733862a70fdcb717baf7e251c2c13f158329b7c005722d1f6faa1a","receipt":"52d94f43afcfd8512e4f03703f7a7faaf805ef2abdcc7c7f93fbebec4e00c40f"}]

## Verification

Before the repair, the deterministic startup regression failed with
`the worker survived its parent's SIGKILL` and readiness `"phase": "work"`:
the orphan entered native-blocked work after its original supervisor died.
Command: `python -m pytest tests/test_pact_replay_pdeathsig_2624.py::test_sigkill_parent_before_worker_startup -q --basetemp="$TMPDIR/pytest"`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0a98044958a8e2f4c2313541bf501ae1fad78e43eb5bbb2f53516941cbb45302","size":3270}},"operation":"444fd0ea4784984e35549f92ba36e7996bec86ea2d7a2cc6168a0c488326c11f","receipt":"5067248e1f3b1af059a9a40178e1b0ca75c339e0bb1fdae17c5d26b771283995"}]

After the repair, the focused CPU suite passed: **3 passed, 1 skipped**. The skip
is explicitly CUDA-gated and is not GPU evidence.
Command: `python -m pytest tests/test_pact_replay_pdeathsig_2624.py -q -ra --basetemp="$TMPDIR/pytest"`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8fc33cd2e2d15cdbeae9254b2c00a84337baa767f73a76cce3f312c12095a0af","size":953}},"operation":"635ffb06e6e94fb9066d62ed6d52b705349c62ba155b906fecbbf3cc95e774e2","receipt":"2ce2bfced4dd875dc434f1600c2cb30ca9310b20895da00a93982d6dc5a1221a"}]

Final targeted GPU verification: **1 passed in 2.59 seconds** using CUDA Torch
2.11.0+cu130 on the supplied GB10 GPU arm. No availability/wait-channel skip;
the real allocation, native-block, SIGKILL-status, PID, and group checks passed.
The command is the targeted GPU command recorded above.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"df3e315247607fd8538b9a05c6ae8dd458eebffb0a88405a9260cda196d8429a","size":764}},"operation":"3bd90c492cec63b6969d9afc6e001c468ee48a9015ec7e440e0013e3fd2a7a4b","receipt":"50a20c82e421eb2cdabeef7a9f19dd2be632bbd8c7d801d606d889ea238dfaa7"}]

Final CPU checks: compileall succeeded; replay regressions plus duplication,
architecture/documentation, and PrismaBuild-boundary guards reported
**31 passed, 1 skipped**. Only the real CUDA test skipped on the CPU arm.
Command: `PYTHONPYCACHEPREFIX="$TMPDIR/pycache" python -m compileall -q experiments/pact_replay/replay_progress.py tests/test_pact_replay_pdeathsig_2624.py && python -m pytest tests/test_pact_replay_pdeathsig_2624.py tests/test_duplication_baseline.py tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_prismabuild_boundary.py -q -ra --basetemp="$TMPDIR/pytest"`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1ef7c2e6ae402c81fe28c7c2a0b6e3909a89ae11d871af18d28e2eea830c4939","size":954}},"operation":"6e561363c58e9fffdf72fdcc1d6c923932dbe052c34c37075414e1e11243dcca","receipt":"1716fc8e00d42cdea841d7e896e2fa80d4ce7166cf6af8ca77547626136a239c"}]

All observed warnings were the existing Python 3.14 Torch JIT deprecation
warnings. The full repository suite and full PACT/model replay were not run;
this repair only changes supervisor identity capture and worker cleanup tests.
