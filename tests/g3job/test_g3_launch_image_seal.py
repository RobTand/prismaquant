"""The launcher's producer-image seal, run on the real pinned container adapter.

g3_launch.py runs teacher-04's pinned tools/tessera_campaign_container.py (PQ 7882eda3).
In default dev mode (D32) it turns the adapter's declared-versus-running image check into
the central stamp. In certified mode (PRISMAQUANT_DEV_MODE=0) the adapter keeps its own
refusal.

The launcher is a script and runs on import, so each test starts it as a child process.
The child loads the pinned adapter from the fleet mount. Only the docker executable is
replaced. The observed digest comes from the pinned adapter's own function. A launcher
that reports the expected digest in place of the observed one fails here.

The checkout's own tools/tessera_campaign_container.py is a different file. It binds
_runtime_identity, and the launcher never loads it.
"""
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.fleet_data

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "g3job" / "g3_launch.py"


def launcher_constant(name):
    """A module-level literal of the launcher. The launcher cannot be imported."""
    for node in ast.parse(LAUNCHER.read_text()).body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == [name]:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{LAUNCHER.name} has no literal {name}")


PQ = Path(launcher_constant("PQ"))
ADAPTER = PQ / "tools" / "tessera_campaign_container.py"
IMAGE = launcher_constant("IMAGE")
IMAGE_CONTENT = launcher_constant("IMAGE_CONTENT")

# One image inspection row. Its content digest differs from IMAGE_CONTENT.
INSPECTION = {
    "Id": "sha256:" + "a1" * 32,
    "Os": "linux",
    "Architecture": "arm64",
    "RootFS": {"Type": "layers", "Layers": ["sha256:" + "b2" * 32, "sha256:" + "c3" * 32]},
    "Config": {"Env": ["PATH=/usr/local/bin:/usr/bin"], "Cmd": ["/bin/bash"]},
}

# `docker image inspect` answers with INSPECTION. `docker run` records its arguments and exits.
FAKE_DOCKER = textwrap.dedent("""\
    #!PYTHON
    import json, sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    argv = sys.argv[1:]
    if argv[:2] == ["image", "inspect"]:
        sys.stdout.write((here / "inspect.json").read_text())
    elif argv[:1] == ["run"]:
        (here / "run.json").write_text(json.dumps(argv))
    else:
        sys.exit("unexpected docker call: " + " ".join(argv))
    """)


def pinned_image_content_sha256(image):
    """The pinned adapter's digest of `image`, from its own source, without touching the pinned tree."""
    path = PQ / "tools" / "container_runtime_identity.py"
    namespace = {"__name__": "pinned_container_runtime_identity"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["image_content_sha256"](image)


def run_launcher(tmp_path, **env):
    """Start the real launcher for one arm. Return the finished child and the `docker run` record."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "inspect.json").write_text(json.dumps([INSPECTION]))
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.replace("PYTHON", sys.executable))
    docker.chmod(0o755)
    child = subprocess.run(
        [sys.executable, str(LAUNCHER), "--arm", "a8_w", "--manifest-sha256", "0" * 64,
         "--run-tag", "seal", "--output-root", str(tmp_path / "runs")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", **env})
    return child, bin_dir / "run.json"


def test_pinned_adapter_binds_the_digest_function_the_launcher_wraps(tmp_path):
    probe = ("import json, runpy, sys; sys.path.insert(0, sys.argv[1]); "
             "namespace = runpy.run_path(sys.argv[2])['main'].__globals__; "
             "print(json.dumps(callable(namespace.get('image_content_sha256'))))")
    child = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(PQ), str(ADAPTER)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert child.stdout.strip() == "true", (
        f"{ADAPTER} does not bind image_content_sha256 in the namespace of main(); "
        f"the launcher wraps that name.\n{child.stderr[-1500:]}")


def test_default_dev_mode_stamps_a_differing_image_and_starts_the_container(tmp_path):
    child, ran = run_launcher(tmp_path)
    observed = pinned_image_content_sha256(INSPECTION)
    assert observed != IMAGE_CONTENT
    assert child.returncode == 0, child.stderr[-2000:]
    lines = child.stdout.splitlines()
    stamps = [line for line in lines if line.startswith("[DEV-MODE]") and "producer image content" in line]
    assert len(stamps) == 1
    assert IMAGE_CONTENT in stamps[0] and observed in stamps[0]
    published = [json.loads(line) for line in lines if line.startswith("{")]
    adapter = next(row for row in published if row.get("schema") == "prismaquant.tessera_campaign_container.v1")
    assert adapter["image_content_sha256"] == observed
    assert adapter["declared_content_sha256"] is None
    argv = json.loads(ran.read_text())
    assert f"PRISMAQUANT_CONTAINER_CONTENT_SHA256={observed}" in argv
    assert INSPECTION["Id"] in argv


def test_certified_mode_refuses_a_differing_image_before_any_container_starts(tmp_path):
    child, ran = run_launcher(tmp_path, PRISMAQUANT_DEV_MODE="0")
    observed = pinned_image_content_sha256(INSPECTION)
    assert child.returncode != 0
    assert (f"RuntimeError: Docker image content differs for {IMAGE!r}: "
            f"expected {IMAGE_CONTENT}, observed {observed}") in child.stderr
    assert not ran.exists()
    assert "[DEV-MODE]" not in child.stdout
    assert "prismaquant.tessera_campaign_container.v1" not in child.stdout
