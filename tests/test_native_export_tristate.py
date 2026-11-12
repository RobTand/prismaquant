"""Compare native argument assembly with the frozen pre-PRI-226 shell path."""

import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OLD = (ROOT / "tests/fixtures/pre226_native_export_args.sh").read_text()
FALSE = ("0", "false", "False", "FALSE", "no", "No", "NO")
TRUE = ("1", "true", "True", "TRUE", "yes", "Yes", "YES")
INVALID = ("AUTO", "False ", " yes", "tRuE", "2", "-1", "\n", "1 0", "$(false)")
SETTINGS = (("EXPORT_GPTQ", "gptq"), ("EXPORT_SCALE_SWEEP", "scale-sweep"))
BASE_ARGS = (
    b"python3", b"-m", b"prismaquant.export_native_compressed",
    b"--model", b"model with spaces", b"--layer-config", b"work\nspace/artifacts/layer_config.json",
    b"--output", b"work\nspace/exported", b"--device", b"cuda:1",
    b"--shard-bytes", b"12345", b"--activation-cache-dir", b"work\nspace/act",
    b"--pin-example", b"literal * $(false)",
)


def _current_block():
    script = (ROOT / "prismaquant/run-pipeline.sh").read_text()
    start = script.index('echo "[pipeline] [4/4] exporting to compressed-tensors ..."')
    end = script.index('if [[ -n "$PRODUCTION_CACHE_PATH" ]]; then', start)
    defaults = []
    for setting, _ in SETTINGS:
        matches = re.findall(rf'^: "\$\{{{setting}:=[^}}]*\}}"$', script, re.MULTILINE)
        assert len(matches) == 1
        defaults.extend(matches)
    return "\n".join(defaults) + "\n" + script[start:end]


def _run(block, values):
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "MODEL_PATH": "model with spaces",
           "WORK_DIR": "work\nspace", "EXPORT_DEVICE": "cuda:1", "EXPORT_SHARD_BYTES": "12345"}
    env.update({key: value for key, value in values.items() if value is not None})
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail\nEXPORT_PIN_ARGS=(--pin-example \'literal * $(false)\')\n'
         + block + '\nprintf "%s\\0" "${EXPORT_ARGS[@]}"'],
        env=env, capture_output=True, timeout=10,
    )


def _assert_equivalent(values, flags=(), error=None):
    old, current = (_run(block, values) for block in (OLD, _current_block()))
    assert (current.returncode, current.stdout, current.stderr) == (
        old.returncode, old.stdout, old.stderr,
    )
    prefix = b"[pipeline] [4/4] exporting to compressed-tensors ...\n"
    assert current.stdout == prefix + (
        b"" if error else b"\0".join((*BASE_ARGS, *flags)) + b"\0"
    )
    assert current.returncode == (2 if error else 0)
    assert current.stderr == (
        f"[pipeline] ERROR: {error} must be auto, 0, or 1\n".encode() if error else b""
    )


@pytest.mark.parametrize("setting,suffix", SETTINGS)
@pytest.mark.parametrize("value", (None, "", "auto", *FALSE, *TRUE, *INVALID))
def test_each_native_export_setting_retains_its_contract(setting, suffix, value):
    flags = (f"--no-{suffix}".encode(),) if value in FALSE else (
        (f"--{suffix}".encode(),) if value in TRUE else ()
    )
    _assert_equivalent({setting: value}, flags, setting if value in INVALID else None)


@pytest.mark.parametrize("gptq,sweep,flags,error", (
    ("0", "1", (b"--no-gptq", b"--scale-sweep"), None),
    ("1", "0", (b"--gptq", b"--no-scale-sweep"), None),
    ("false", "false", (b"--no-gptq", b"--no-scale-sweep"), None),
    ("true", "true", (b"--gptq", b"--scale-sweep"), None),
    ("invalid", "invalid", (), "EXPORT_GPTQ"),
    ("yes", "invalid", (), "EXPORT_SCALE_SWEEP"),
))
def test_combined_settings_preserve_order_and_first_refusal(gptq, sweep, flags, error):
    _assert_equivalent({"EXPORT_GPTQ": gptq, "EXPORT_SCALE_SWEEP": sweep}, flags, error)


def test_both_native_settings_use_the_shared_parser():
    spy = '''append_export_tristate_arg() {
  printf '%s\\0' "$@" >&2
  EXPORT_ARGS+=("observed:$1:$2:$3")
}'''
    block, count = re.subn(
        r"^append_export_tristate_arg\(\) \{\n.*?^\}",
        lambda _: spy, _current_block(), flags=re.MULTILINE | re.DOTALL,
    )
    assert count == 1, "native export needs one shared tri-state parser"
    result = _run(block, {"EXPORT_GPTQ": "raw gptq", "EXPORT_SCALE_SWEEP": "raw sweep"})
    assert result.returncode == 0
    assert result.stderr.split(b"\0") == [
        b"EXPORT_GPTQ", b"raw gptq", b"gptq",
        b"EXPORT_SCALE_SWEEP", b"raw sweep", b"scale-sweep", b"",
    ]
    assert result.stdout.split(b"\0")[-3:] == [
        b"observed:EXPORT_GPTQ:raw gptq:gptq",
        b"observed:EXPORT_SCALE_SWEEP:raw sweep:scale-sweep", b"",
    ]
