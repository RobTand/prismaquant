"""Returned pin metadata cannot alter the cached reproducibility contract."""
import subprocess
import sys


def test_mutating_a_pin_view_cannot_change_later_native_extension_verification():
    result = subprocess.run([sys.executable, "-c", """
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.tessera_serving_runtime_pin import load_tessera_serving_runtime_pin
snapshot = runtime.installed_runtime_snapshot()
pin = load_tessera_serving_runtime_pin()
extension = pin.serving_native_extensions[0]
mode = next(iter(extension.when_unavailable))
extension.when_unavailable[mode]["decoder"] = "caller-edited"
contract = runtime.read_tessera_contract(snapshot.path, snapshot.sha256, pin.producer_commit)
runtime.require_pin_native_extensions_match_contract(contract)
snapshot.require_exact_pin()
print("native extension identity preserved")
"""], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "native extension identity preserved"
