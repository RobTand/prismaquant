# RECONSTRUCTION, 2026-10-10. The original was deleted with /home/rob/tmp/pb-2188-job (commit 1599f27 in a repo that is gone).
# This is the script text as I wrote it for PB action c736f9b072a0681454387a8a6203880060f73546543755ac954ab5dfaebc60b9.
# The exact bytes of the original are in that action's checkout snapshot in the PrismaBuild CAS; I did not extract them.
# PQ #2188, pb-integrator: import the new pure-Python artifact inside the GPU container image, under that image's own interpreter.
# CPU only; reads /art read-only; extracts to a temporary directory under fixed bounds.
import base64, hashlib, json, platform, subprocess, sys, tarfile, tempfile
from pathlib import Path
WANT = "9923cb74caa5e5fa1c41c9aef809941bf21c4687e53fd0b30ed781a0ce49023c"
MAX_FILES, MAX_BYTES, MAX_MEMBER = 20000, 64 * 1024 * 1024, 16 * 1024 * 1024
tar_path = Path("/art/pure-python-installs.tar")
print("PYTHON", sys.version.split()[0], "ARCH", platform.machine(), "IMPL", platform.python_implementation())
digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
print("TAR_SHA256", digest, "MATCH" if digest == WANT else "MISMATCH")
assert digest == WANT, "artifact digest differs"
with tarfile.open(tar_path) as t:
    members = t.getmembers()
    n = len(members); total = sum(m.size for m in members)
    assert n <= MAX_FILES and total <= MAX_BYTES, (n, total)
    for m in members:
        assert m.isfile() and not m.name.startswith("/") and ".." not in Path(m.name).parts and m.size <= MAX_MEMBER, m.name
    tmp = Path(tempfile.mkdtemp(prefix="pb2188img."))
    t.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else t.extractall(tmp)
print("EXTRACTED members", n, "bytes", total, "limits", MAX_FILES, MAX_BYTES, MAX_MEMBER)
checked = bad = 0
for rec in sorted(tmp.glob("*.dist-info/RECORD")):
    for line in rec.read_text().splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3 or not parts[1].startswith("sha256="):
            continue
        f = tmp / parts[0]
        if not f.is_file():
            continue
        want = base64.urlsafe_b64decode(parts[1][7:] + "=" * (-len(parts[1][7:]) % 4))
        checked += 1
        bad += hashlib.sha256(f.read_bytes()).digest() != want
print("RECORD_HASHES checked", checked, "bad", bad)
for d in sorted(tmp.glob("prismabuild-*.dist-info")) + sorted(tmp.glob("tessera_quant-*.dist-info")):
    u = json.loads((d / "direct_url.json").read_text())
    print("DIRECT_URL", d.name, u["url"], u["vcs_info"]["commit_id"], "editable" if u.get("dir_info", {}).get("editable") else "non-editable")
code = ("import sys; sys.path.insert(0, %r); import prismabuild, tessera, pytest, xdist, pluggy, packaging; "
        "print('IMPORT_OK', [m.__file__.startswith(%r) for m in (prismabuild, tessera, pytest, xdist, pluggy, packaging)], pytest.__version__)") % ((str(tmp),) * 2)
r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=120)
print(r.stdout.strip()); print("IMPORT_RC", r.returncode, r.stderr.strip()[-300:])
ok = bad == 0 and r.returncode == 0 and "IMPORT_OK [True, True, True, True, True, True]" in r.stdout
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
