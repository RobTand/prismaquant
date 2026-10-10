#!/bin/bash
# PQ #2188, pb-integrator, 2026-10-09: build a NEW Python 3.12 install of prismabuild 027103d + the tessera pin + pytest, pack it with the existing
# packer (experiments/original_cuda_pack_dependencies.py, unmodified), and verify a bounded extraction under Python 3.12.  CPU only.
# Creates only new paths; refuses to overwrite.  The old artifact and every existing venv are untouched.
set -euo pipefail
PB=027103d9a8417e06c7f13356e58779a313cd7088
PQ_COMMIT=${PQ_COMMIT:?}
PY312=/home/rob/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
STAMP=20261009
VENV=/home/rob/venvs/pq-cpu312-pb027103d9-pure-$STAMP
OUT=/mnt/shared/astra-pq-2188-evidence-$STAMP/original-cuda-test-deps-02
WORK=$(mktemp -d /tmp/pb2188.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
[ -x "$PY312" ] || { say "no Python 3.12 at $PY312"; exit 10; }
[ ! -e "$VENV" ] || { say "venv exists, refusing: $VENV"; exit 11; }
[ ! -e "$OUT" ] || { say "output exists, refusing: $OUT"; exit 12; }
say "python: $($PY312 --version)"
say "clone prismaquant at $PQ_COMMIT"
git clone -q https://github.com/RobTand/prismaquant.git "$WORK/pq"
git -C "$WORK/pq" checkout -q "$PQ_COMMIT"
TESS=$(python3 "$WORK/pq/tools/resolve_tessera_dev_pin.py")
PBPIN=$(python3 "$WORK/pq/tools/resolve_prismabuild_dev_pin.py")
say "tessera pin $TESS; prismabuild pin from staged_lease.py $PBPIN"
[ "$PBPIN" = "$PB" ] || { say "the consumer pin is $PBPIN, not $PB: STOP"; exit 13; }
say "create venv $VENV"
"$PY312" -m venv "$VENV"
"$VENV/bin/python" -m pip --version
say "install prismabuild $PB (non-editable, from git, no deps)"
"$VENV/bin/python" -m pip install -q --no-deps --no-cache-dir "prismabuild @ git+https://github.com/RobTand/prismabuild.git@$PB"
say "install tessera $TESS (non-editable, from git, no deps)"
"$VENV/bin/python" -m pip install -q --no-deps --no-cache-dir "tessera-quant @ git+https://github.com/RobTand/tessera.git@$TESS"
say "install pure-Python test dependencies at the versions of the old artifact"
"$VENV/bin/python" -m pip install -q --no-deps --no-cache-dir --only-binary :all: \
  pytest==9.1.1 pytest-timeout==2.4.0 pytest-xdist==3.8.0 execnet==2.1.2 pluggy==1.6.0 packaging==26.3 iniconfig pygments
say "pack with the existing packer"
mkdir -p "$(dirname "$OUT")"
( cd "$WORK/pq" && PYTHONPATH="$WORK/pq" "$VENV/bin/python" experiments/original_cuda_pack_dependencies.py --out "$OUT" )
say "bounded extraction and import check under Python 3.12"
"$VENV/bin/python" - "$OUT" <<'PYEOF'
import sys, tarfile, tempfile, hashlib, json, base64, subprocess, importlib.metadata as md
from pathlib import Path
out = Path(sys.argv[1]); tar_path = out / "pure-python-installs.tar"
MAX_FILES, MAX_BYTES, MAX_MEMBER = 20000, 64 * 1024 * 1024, 16 * 1024 * 1024
with tarfile.open(tar_path) as t:
    members = t.getmembers()
    n = len(members); total = sum(m.size for m in members)
    assert n <= MAX_FILES, f"too many members {n}"
    assert total <= MAX_BYTES, f"too many bytes {total}"
    for m in members:
        assert m.isfile(), f"not a regular file: {m.name}"
        assert not m.name.startswith("/") and ".." not in Path(m.name).parts, f"unsafe name {m.name}"
        assert m.size <= MAX_MEMBER, f"member too large {m.name}"
    tmp = Path(tempfile.mkdtemp(prefix="pb2188x."))
    t.extractall(tmp, filter="data")
bad = 0; checked = 0
for rec in sorted(tmp.glob("*.dist-info/RECORD")):
    for line in rec.read_text().splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3 or not parts[1].startswith("sha256="): continue
        f = tmp / parts[0]
        if not f.is_file(): continue
        want = base64.urlsafe_b64decode(parts[1][7:] + "=" * (-len(parts[1][7:]) % 4))
        checked += 1
        if hashlib.sha256(f.read_bytes()).digest() != want: bad += 1
code = ("import sys; sys.path.insert(0, %r); import prismabuild, tessera, pytest, importlib.metadata as m; "
        "print(sys.version.split()[0], prismabuild.__file__.startswith(%r), tessera.__file__.startswith(%r), pytest.__file__.startswith(%r))") % ((str(tmp),) * 4)
r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=120)
res = dict(members=n, bytes=total, limits=dict(files=MAX_FILES, bytes=MAX_BYTES, member=MAX_MEMBER), record_hashes_checked=checked, record_hashes_bad=bad,
           import_rc=r.returncode, import_out=r.stdout.strip(), import_err=r.stderr.strip()[-400:], python=sys.version.split()[0])
(out / "extraction-check.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
assert bad == 0 and r.returncode == 0 and r.stdout.split()[1:] == ["True", "True", "True"], "extraction check failed"
PYEOF
say "pip freeze of the new venv"
"$VENV/bin/python" -m pip freeze --all > "$OUT/pip-freeze.txt"
sha256sum "$OUT/pure-python-installs.tar" "$OUT/receipt.json" "$OUT/extraction-check.json" "$OUT/pip-freeze.txt" | tee "$OUT/SHA256SUMS"
say "DONE: venv $VENV; artifact $OUT"
