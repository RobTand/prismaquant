# Admitted row profiling

Run `tools.pq_admitted_profile` only inside a PB-admitted action. This wrapper
owns host telemetry and one original row command, not placement or admission.
Profiling outputs never enter cost, anchor, wire or runtime identities.

## Preserve the admission boundary

PB's installed resource payload deliberately calls `prctl(38, 1)` before exec:
`NoNewPrivileges=1` is part of its contract. In incident #1942, the observer's
`sudo -n test -w` failed only inside that action. The same unscoped host command
succeeded. The root resource-broker service also declares
`NoNewPrivileges=yes`; the ordinary fleet coordinator process is not the
privilege boundary. Do not change sudoers, this policy, or the Yama setting.

The observer now uses its own UID for directory checks and `/proc/PID/io`.
The profiler runs **inside the existing workload container**, as the same UID
and in the same namespace. `tools.pq_profile_child` launches py-spy, which
parents a status worker and the unchanged original Python argv. This satisfies
the parent relationship without a privileged sibling attach. It records until
the workload ends; the existing PB row timeout is the bound, not a profiler
cutoff that could terminate a live row.

## Explicit dependencies and outputs

Host startup needs only the standard library. `tools.pq_profile_source` loads
only the existing digest and IO-span source owners from the same snapshot;
it does not execute `prismaquant.__init__` or install a replacement package
namespace. The observer uses the authoritative `io_spans.PeriodicSampler`,
including its native identity and joined shutdown, rather than a copied sampler.
Torch and compressed-tensors remain workload-container dependencies, not host
observer readiness prerequisites. A fresh `-S` interpreter executes the actual
observer import chain in `tests/test_profile_observer_bootstrap_1942.py`.

Pass `--profiler-executable` as a pinned executable available at the same path
on the admitted host and in the existing container. Declare it in the action's
read set. Do not depend on an unrelated image's or host's ambient py-spy version.
The owner must acquire the appropriate architecture binary before submission.
No direct Docker invocation or implicit dependency installation occurs here.

The owned observation directory must be writable by that UID in both contexts.
The host-local `--profile-local` directory is metadata-only; no root-written
artifact is required. The child writes `child-profile.speedscope` and a typed
workload PID and return-code record. The parent atomically adds its separate
profiler return code; it cannot replace the original failed workload's code.
The observer validates workload-attributed samples and publishes the exact bytes
atomically after both-box telemetry completes. Publication requires both return
codes to be zero; a profiler shutdown error cannot qualify a successful row.
A py-spy zero exit cannot mask a failed workload. Missing status/profile,
telemetry gaps, missing target, nonzero row, or collector failure refuses
qualification. The observer never signals or retries the row.

## Qualification limits

The PB x86 regression applies the real inherited no-new-privileges setting in
an isolated child and exercises the observer's actual preflight statements.
Native CPU smoke must additionally produce real sampled frames under the same
setting and preserve an actual failed-child return code. Neither is a GPU or
throughput result. For an authorized GPU comparison, record each leg's UTC
window and both-box Netdata CPU and power, label exclusive GPU with ambient CPU,
and inspect terminal/CAS/profile/output evidence before accepting a speed or
artifact-equality claim. Respect the coordinator's window and retry rulings.
