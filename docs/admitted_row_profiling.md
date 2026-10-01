# Admission-bound row profiling

The #1899 instrumentation owns one observer and one unchanged row inside an
ordinary admitted PrismaBuild action. It does not submit work or schedule it.

The entrypoint is `python -m tools.pq_admitted_profile --launch ROW.json
--observations NEW_DIR --profile-local NEW_HOST_LOCAL_DIR`. `ROW.json` stores
an exact container-wrapper argv. The action must separately declare that row's
environment, image, input readset, aggregate CPU/memory/GPU and timeout.

The observer starts after admission, validates both-box Netdata power charts,
local privileged py-spy writability/version and exact target identity, then
publishes readiness. The owner launches no workload until that barrier succeeds.
Old observation directories and mismatched targets refuse. Observation continues
through the actual row's exit; the owner joins it before returning, and observer
failure cannot be hidden by a successful row. Workload failure cannot be hidden
by successful observation either. The owner signals only its own observer during
cleanup; it never signals/retries the workload.

The observer recognizes the actual Python campaign PID, not Docker wrappers.
It retains PB affinity and records both hosts correctly on either GB10 worker.
The privileged profiler writes a new host-local file, then the unprivileged
observer validates and publishes identical speedscope bytes atomically. No wire,
price, journal or encoder identity is changed by telemetry output.

For the coordinator-authorized #1750 replacement, use one ordinary `--tag gb10`
row at priority -20, explicit timeout, `pbrun --detach` from dl380g10, no host pin,
isolation or measurement mode. Inspect canonical terminal/log/CAS, sampled
profile, complete Netdata series and artifacts before B. A observer failure
stops the experiment: no retry. One A and one B are authorized, not extra repeats.
GPU work remains inside PrismaBuild; this entrypoint is not a local GPU bypass.

CPU mocked lifecycle tests qualify readiness/ownership/refusal, not actual GPU
profile accessibility or throughput. Any speed claim needs valid before/after
in-process profiles plus both-box Netdata, co-tenancy inspection, power against
the ~140 W envelope/work per joule, and identical non-timing priced artifacts.
