#!/usr/bin/env python3
"""Host side of one G3 action: run g3_offline_decoded_kl.py in teacher-04's producer container.

Numerical runner arguments and pinned /pq files remain teacher-04's (PQ7882eda3).
The harness calls the current central D32 policy owner at its own seams; only
the source-read seam replaces a pinned hook. PB ranges use the admitted helper
mounts and a writable queue for attempt-bound reader pins. Host keys are
translated from /source, exports and teachers; /workspace is this pbjob checkout.

usage: g3_launch.py (--pilot | --arm ARM) [--run-tag TAG]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PQ = "/mnt/shared/tessera-measurements/surrogate-diag-20260929/src/pq-7882eda3"
PQ_COMMIT = "7882eda3a87fd6a45a1142de140aa17c49a8021e"
PIN = "/mnt/shared/tessera-pins/b40c93cb73745097e57a1ba4cf5b9eee166c759a"
CENSUS = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908"
G3 = "/mnt/shared/tessera-measurements/surrogate-diag-20260929/g3"
T8R = "/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported"
A8 = "/mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2"
# The census allocation run's per-(Linear, format) unit artifacts: the law-fit pick's cells that
# neither export carries.  Opened by name only; never list this directory (it stalls NFS).
WIRE = ("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/"
        "extension-r1024-02/workspace/merged/cache/wire")
# The EXL3 reference artifact (exl3_w rows, manifest v3 root "exl3"); its shards are read by the
# byte ranges the pre-pass recorded (exl3_prepass.json), never listed.
EXL3 = "/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw"
IMAGE = "prismaquant-glm-producer:content-qualified-20260908"
IMAGE_CONTENT = "eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd"

p = argparse.ArgumentParser(description=__doc__)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--arm")
g.add_argument("--pilot", action="store_true")
g.add_argument("--arms", help="comma-separated arms for one multi-arm source pass (null first)")
g.add_argument('--reader-smoke', help='JSON for one staged-range CUDA correctness smoke; no quality arm')
p.add_argument("--run-tag", default="cold1")
p.add_argument('--manifest-sha256')
p.add_argument("--manifest-name", default="unit_manifest_v2.json")
p.add_argument("--manifest-root", default=G3)
p.add_argument("--output-root", default=str(Path(G3) / "runs"))
p.add_argument("--artifact-export", default=T8R)
p.add_argument("--a8-export", default=A8)
p.add_argument("--preflight", action="store_true")
p.add_argument("--qualify-then-score", action="store_true")
p.add_argument("--source-preparation")
p.add_argument("--source-preparation-sha256")
args = p.parse_args()
if not args.reader_smoke and not args.manifest_sha256:
    p.error('a scorer requires --manifest-sha256')

HOST = os.uname().nodename
source_proof = None
if not args.pilot and not args.reader_smoke and args.source_preparation:
    from g3_prepared_source import read_prepared_source_json
    source_proof = read_prepared_source_json(args.source_preparation, args.source_preparation_sha256)
    source_proof_root = Path(args.source_preparation).resolve().parent

name = ('reader-smoke-' + args.run_tag if args.reader_smoke else
        'pilot-' + args.run_tag if args.pilot else
        f'multi-{args.run_tag}' if args.arms else f'{args.arm}-{args.run_tag}')
runs = Path(args.output_root)
for d in ("tmp", "cache", "hf-cache", "torch-cache"):
    (runs / d).mkdir(parents=True, exist_ok=True)
if (runs / name).exists():
    raise SystemExit(f"{runs / name} exists; refusing to overwrite a run")
for a in (args.arms.split(",") if args.arms else []):
    if (runs / f"{a}-{args.run_tag}").exists():
        raise SystemExit(f"{runs / (a + '-' + args.run_tag)} exists; refusing to overwrite a run")
if not args.pilot and not args.reader_smoke and not args.preflight:
    (runs / f"staging-{name}").mkdir(exist_ok=False)

mounts = [
    ("/mnt/shared/models/GLM-5.3-Flash-BF16", "/source"),
    (f"{CENSUS}/tr3-teacher-inputs-01", "/panel"),
    (f"{CENSUS}/exl3-first-artifact-01", "/binding"),
    (PQ, "/pq"),
    (PIN, "/tessera"),
    (f"{CENSUS}/tr3-teacher-04/artifact", "/teacher1"),
    (f"{CENSUS}/tr3-teacher-exl3ref-01/artifact", "/teacher2"),
    (args.artifact_export, "/t8r"),
    (args.a8_export, "/a8"),
    (args.manifest_root, "/g3in"),
    (WIRE, "/wirecache"),
    (EXL3, "/exl3"),
] + ([] if source_proof is None else [(str(source_proof_root), str(source_proof_root))])
spec = {"container": {"image": IMAGE, "content_sha256": IMAGE_CONTENT,
                      "mounts": [{"source": s, "target": t, "readonly": True} for s, t in mounts]
                      + [{"source": str(runs), "target": "/out"}]},
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PRISMAQUANT_LAYER_READ_THREADS": "4", "PRISMAQUANT_RELEASE_SOURCE_PAGES": "1",
                "PRISMAQUANT_DETERMINISTIC": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "TRITON_CACHE_DIR": "/tmp/triton", "PRISMAQUANT_TMPDIR": "/out/tmp", "TMPDIR": "/out/tmp",
                "XDG_CACHE_HOME": "/out/cache", "HF_HOME": "/out/hf-cache", "TORCH_HOME": "/out/torch-cache",
                "PRISMAQUANT_IDENTITY_GIT_COMMIT": PQ_COMMIT, "PRISMAQUANT_IDENTITY_GIT_DIRTY": "false",
                "TESSERA_SRC": "/tessera/src", "G3_PQ_ROOT": "/pq"}}
from g3_residency import container_contract
pb_mounts, pb_env = container_contract()
spec["container"]["mounts"].extend(pb_mounts)
spec["env"].update(pb_env)
spec["env"]["G3_HOST_MOUNTS"] = json.dumps({target: source for source, target in mounts})
if "PRISMAQUANT_DEV_MODE" in os.environ:
    spec["env"]["PRISMAQUANT_DEV_MODE"] = os.environ["PRISMAQUANT_DEV_MODE"]
if args.reader_smoke:
    cmd = ['python3', '/workspace/tools/g3job/g3_residency_smoke.py', '--spec', args.reader_smoke,
           '--out', f'/out/{name}.json']
else:
    cmd = ["python3", "/workspace/tools/g3job/g3_offline_decoded_kl.py",
           "--run-tag", args.run_tag, "--out", f"/out/{name}",
           "--manifest", "/g3in/" + args.manifest_name, "--manifest-sha256", args.manifest_sha256,
           "--t8r-export", "/t8r", "--a8-export", "/a8", "--wire-cache", "/wirecache", "--exl3-dir", "/exl3"]
    if args.pilot:
        cmd += ["--pilot"]
    else:
        cmd += (["--arms", args.arms, "--decode-threads", "3"] if args.arms else ["--arm", args.arm]) + [
                "--model", "/source",
                "--panel", "/panel/final_panel_handoff.json", "--arrays-root", "/panel/arrays",
                "--reference-binding", "/binding/root-bf16-upstream-source-binding.json",
                "--reference-binding-sha256", "23370e25f6b42e316f72d6cbc8f3f5747a65705b388c910da93545f487edc0fa",
                "--source-derivative-json", "/panel/original-execution.json",
                "--source-derivative-sha256", "38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed",
                "--offload-folder", f"/out/staging-{name}", "--cache-headroom-gb", "12",
                "--teacher", "/teacher1",
                "--teacher-sha256", "1cc798a32a3457f996e859f778fe61fd987561b91490fe2953b698457ea747ae",
                "--teacher2", "/teacher2",
                "--teacher2-sha256", "da595505a3e9a2bcced69bde1f156f71b62f43571b1797cede21d12f4b9c423b"]
        if args.source_preparation:
            cmd += ["--source-preparation", args.source_preparation,
                    "--source-preparation-sha256", args.source_preparation_sha256]

launch = {"g3_launch": name, "host": HOST, "spec": spec, "cmd": cmd}
launch_path = runs / f"{name}{'.preflight' if args.preflight else ''}.launch.json"
launch_path.write_text(json.dumps(launch, indent=1))
print(json.dumps(launch), flush=True)
if args.preflight or args.qualify_then_score:
    cmd = ["python3", "/workspace/tools/g3job/g3_v1_preflight.py", json.dumps(cmd)] + (["--qualify-then-score"] if args.qualify_then_score else [])
sys.path.insert(0, PQ)
from tools import tessera_campaign_container as adapter
from g3_pq_policy.dev_mode import dev_mode_enabled, seal_check
if dev_mode_enabled():
    # The adapter still observes/publishes the actual image digest. Only its
    # recorded-versus-running image seal is converted to the central stamp.
    expected_image = spec["container"].pop("content_sha256")
    original_image_digest = adapter.image_content_sha256
    def image_metadata(inspected):
        actual = original_image_digest(inspected)
        seal_check("producer image content", expected_image, actual, where="G3 launch",
                   refusal=RuntimeError(f"Docker image content differs for {IMAGE!r}: "
                                        f"expected {expected_image}, observed {actual}"))
        return actual
    adapter.image_content_sha256 = image_metadata
raise SystemExit(adapter.main(["--spec", json.dumps(spec), "--"] + cmd))
