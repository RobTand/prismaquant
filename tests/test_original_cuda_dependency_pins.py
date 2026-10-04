"""Stdlib fixture pin ownership only; no archive, image or CUDA qualification."""
from pathlib import Path
import subprocess
import sys

from experiments import original_cuda_inner, original_cuda_pack_dependencies
from tools.resolve_prismabuild_dev_pin import PIN_NAME, PIN_SOURCE, resolve_literal_pin
from tools.resolve_tessera_dev_pin import resolve_tessera_dev_pin


def test_packer_pins_are_the_existing_source_owned_installed_contracts():
    assert original_cuda_pack_dependencies.PINS == {
        'prismabuild': resolve_literal_pin(PIN_SOURCE, PIN_NAME),
        'tessera-quant': resolve_tessera_dev_pin(),
    }


def test_inner_reuses_the_exact_packer_pin_mapping():
    assert original_cuda_inner.PINS is original_cuda_pack_dependencies.PINS


def test_owned_pin_imports_need_no_numeric_or_consumer_package():
    program = (
        'import sys; from experiments import original_cuda_inner as inner, '
        'original_cuda_pack_dependencies as pack; '
        'from tools.resolve_prismabuild_dev_pin import PIN_SOURCE, PIN_NAME, resolve_literal_pin; '
        'from tools.resolve_tessera_dev_pin import resolve_tessera_dev_pin; '
        'assert inner.PINS is pack.PINS; '
        'assert pack.PINS == {"prismabuild": resolve_literal_pin(PIN_SOURCE, PIN_NAME), '
        '"tessera-quant": resolve_tessera_dev_pin()}; '
        'assert "prismaquant" not in sys.modules and "torch" not in sys.modules')
    subprocess.run([sys.executable, '-S', '-c', program],
                   cwd=Path(__file__).resolve().parents[1], check=True)
