# wsl-gpu: what a PrismaBuild box window can see there

2026-09-13 · RobTand/prismaquant#542 gap 3 · box `wsl-gpu`
(`DESKTOP-P5UOGNJ`, Ryzen 9 9800X3D under WSL2, Radeon RX 9070 XT / gfx1201)

PrismaBuild admitted GPU actions on this box before this change, but every one
of them recorded `"source": "unavailable"` for the box around it, so the
machine's own view of a GPU action there was blind. This records what was
provisioned, what the window carries now, and what it still cannot see.

## Before

Action `7e5561c153f951b7ac057377a0031cd1f983c8b5f2691062eed19c528da99d05`
(`--tag wsl-gpu --demand gpu=1,mem_gb=10`, done, rc 0):

```json
{"schema": "prismabuild.box_window.v1", "host": "DESKTOP-P5UOGNJ",
 "source": "unavailable",
 "reason": "no pqteld CSV for DESKTOP-P5UOGNJ covering the window; netdata system.cpu unavailable"}
```

`box_window.read_window` reads two recorders on the box that finishes an
action: `pqteld`'s CSV under `~/pqtel/csv`, and Netdata at
`http://127.0.0.1:19999`. Neither was running here.

## What was installed

Netdata, which is what every other box in the fleet publishes CPU busy and CPU
pressure from, and therefore the source `box_window` already knows how to read.
Installed 2026-09-13 with root on the box:

* `netdata` 2.11.0, native Debian package from netdata's stable apt repository
  (`get.netdata.cloud/kickstart.sh --dont-wait --disable-telemetry
  --stable-channel --non-interactive`). The kickstart added that repository to
  apt, created the `netdata` user and group, added `netdata` to the `docker`
  group for its container collector, and enabled `netdata.service`.
* The unit is `enabled` under systemd, and `/etc/wsl.conf` has `[boot]
  systemd=true`, so it comes back with the distribution. That has not been
  tested against an actual WSL restart: restarting WSL stops the PrismaBuild
  worker loops on the box and is Rob's call, not an agent's.
* Netdata listens on `0.0.0.0:19999`, the same as the other boxes.

`pqteld` was deliberately **not** installed. It is a GB10 recorder: its GPU
columns come from one long-lived `nvidia-smi -l 1`, and its memory group is
named `unified_*` because a GB10's CPU and GPU share one physical pool. This
box has neither. Running it here would produce a series whose name is a claim
about hardware that is not there.

## After

Action `c6563be4124bc3bd87880c592454dedde3f3c438a3b9829c99067c8c6d21943f`
(`--tag wsl-gpu --priority -10 --cpus 4 --demand gpu=1,mem_gb=4`, done, rc 0,
32.4 s, a 30-second bf16 matmul loop on the card):

```json
{"cpu": {"busy_percent_mean": 24.176354851428574,
         "busy_percent_peak": 29.0976038,
         "psi_some_avg10_max": 0.0, "psi_some_avg10_samples": 34,
         "psi_some_avg10_update_every_s": 1,
         "samples": 35, "source": "netdata", "time_group": "average",
         "update_every_s": 1},
 "errors": ["no pqteld CSV for DESKTOP-P5UOGNJ covering the window"],
 "host": "DESKTOP-P5UOGNJ", "schema": "prismabuild.box_window.v1",
 "source": "netdata",
 "start_unix": 1789308293.8777802, "end_unix": 1789308326.4291651}
```

The CPU figure is a real reading and cross-checks: the action's own resource
telemetry reports 121.8 CPU-seconds over 32.4 s of wall clock, which is 3.76
cores of the box's 16, and 24.18% of 16 cores is 3.87.

## What the window still cannot see, and why

* **No GPU group.** `box_window` sources GPU power, utilization and
  temperature from `pqteld` columns only, and there is nothing on this platform
  to fill them: WSL2 has no amdgpu driver. `rocm-smi` answers
  `ERROR:root:Driver not initialized (amdgpu not found in modules)`,
  `/sys/class/drm` holds only `version`, and `/sys/class/hwmon` is empty. The
  fleet's own box record already states this as `telemetry_class memory_only`.
  There is no power reading on this box to rank work per joule against, so
  principle 15's power-against-envelope method does not apply here — wall
  clock and the action's own device-memory readings are what it has.
* **No memory group.** That group is also `pqteld`'s, and its `unified_*`
  fields describe a GB10's single pool. This card's 16 GiB is discrete.
* **VRAM is measurable but unreported.** PrismaBuild's AMD capacity reader
  already takes total and free VRAM from a HIP probe at admission
  (`gpu_admission.gpu_memory_budget_bytes` is recorded on every action here);
  it is simply not summarised into the action's window. Filed as
  RobTand/prismabuild#548.

## Measured cost of one action on this box

From the action above and from the #542 spike, for sizing a `mem_gb` demand
rather than habituating one:

| reading | probe (30 s matmul) | spike (weights-only `BF16_K1` encode) |
|---|---|---|
| device reserved peak | 0.21 GB | 1.69 GB |
| host maxrss | 0.92 GB | 1.76 GB |
| PrismaBuild cgroup peak | 1.64 GB | 1.25 GB |

## The ROCm container attachment, measured

A campaign row that runs in a container needs different flags here: WSL2 has
no NVIDIA container runtime, so `--gpus all` attaches nothing, and the device
is the paravirtualization node `/dev/dxg` with the userspace driver on the
host under `/usr/lib/wsl/lib` (`libdxcore.so`, `libd3d12.so`,
`libd3d12core.so`). `/dev/dxg` is `crw-rw-rw-`, so a container user reaches it
without extra privilege.

Action `7e767098e6daff62481b6c33ed776b8b9754a69178e11a0eb4392fc36e8c9c7b`
(`--tag wsl-gpu --demand gpu=1,mem_gb=6`, done, rc 0) ran exactly the flags
`tools/tessera_campaign_container.py` emits for `gpu_runtime: "rocm-wsl"`
against the pinned gfx1201 image
(`prismaquant/vllm-rocm:0.30.0.dev0-rocm714-gfx1201`, image id `0461258dfe25`):

```
docker run --rm --device /dev/dxg \
  --mount type=bind,src=/usr/lib/wsl/lib,dst=/usr/lib/wsl/lib,readonly \
  --ipc=host --entrypoint '' <image> python3 -c "…"
cuda True AMD Radeon RX 9070 XT hip 7.2.53211
```

So the attachment is measured, not assumed. What is *not* measured is a real
campaign row through the launcher: nothing emits a ROCm row yet.

## One inconsistency left standing

PB's scope cleanup records
`settle_error: PoolContractError: docker ownership query failed (1): permission
denied while trying to connect to the docker API at unix:///var/run/docker.sock`
on some actions here and not others — present on
`c6563be4124b…` (claimed by worker `631116`) and on the #542 spike's
`7e5561c153f9…`, absent on `7e767098e6da…` (worker `631115`), which itself ran
`docker` successfully. Both worker loops carry no `docker` gid in
`/proc/PID/status`, so the cause is not simply the group, and the cleanup
still reports `complete: true` either way. It is recorded here rather than
guessed at: the fix would be restarting the worker loops, which drops any
lease they hold, and a lease loss in PB is terminal. Left for Rob.
