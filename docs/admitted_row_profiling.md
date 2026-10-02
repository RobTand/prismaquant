# Admission-bound row profiling

The #1899 instrumentation owns one observer and one unchanged row inside an
ordinary admitted PrismaBuild action. It does not submit work or schedule it.

The entrypoint is `python -m tools.pq_admitted_profile --launch ROW.json
--observations NEW_DIR --profile-local NEW_HOST_LOCAL_DIR`. `ROW.json` stores
an exact container-wrapper argv. The action must separately declare that row's
environment, image, input readset, aggregate CPU/memory/GPU and timeout.

The observer starts after admission, validates both-box Netdata power charts,
same-UID writability of the owned observation directory, the declared py-spy
version and exact target identity, then publishes readiness. The owner launches no workload until that barrier succeeds.
Old observation directories and mismatched targets refuse. Observation continues
through the actual row's exit; the owner joins it before returning, and observer
failure cannot be hidden by a successful row. Workload failure cannot be hidden
by successful observation either. The owner signals only its own observer during
cleanup; it never signals/retries the workload.

The observer recognizes the actual Python campaign PID, not Docker wrappers.
It retains PB affinity and records both hosts correctly on either GB10 worker.
The same-UID py-spy parent runs inside the workload's existing container/PID
namespace and profiles its owned original Python child. The child writes a new
speedscope file and workload status in the owned observation directory, writable
by that UID on the host and in the container; the host-local `--profile-local`
directory is metadata-only. After telemetry completes, the observer requires
successful workload and profiler outcomes and valid workload-PID-attributed
samples before publishing identical speedscope bytes atomically. No privilege
escalation or workload signalling is added. See the current dependency and
publication contract in [Admitted row profiling](campaign_row_profiling.md).
No wire, price, journal or encoder identity is changed by telemetry output.

Historical #1750 preparation described a one-A/one-B replacement using
`--tag gb10`, priority -20, `pbrun --detach` from dl380g10 and no host pin,
isolation or measurement mode. Those directions and permissions are superseded,
not current instructions or evidence that a matching A/B measurement occurred.
#1750 remains held pending accepted #1942 merge and renewed Astra assignment
with explicit window/inputs and valid PB admission. Retain PB-assigned affinity
and the newly authorized aggregate resource/thread/deadline contract; this
runbook authorizes no run. Any future authorized comparison must inspect canonical
terminal/log/CAS, sampled profile, complete both-box Netdata series and artifacts
before B. An observer failure stops the experiment: no retry or extra repeat
without renewed authorization. GPU work remains inside PrismaBuild; this
entrypoint is not a local GPU bypass.

CPU mocked lifecycle tests qualify readiness/ownership/refusal, not actual GPU
profile accessibility or throughput. Any speed claim needs valid before/after
in-process profiles plus both-box Netdata, co-tenancy inspection, power against
the ~140 W envelope/work per joule, and identical non-timing priced artifacts.
