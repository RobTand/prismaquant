"""PrismaQuant: mixed-native quantization policy engine for LLMs.

Current production path:
    1. incremental_probe.py              stream Fisher and activation shards
    2. incremental_measure_quant_cost.py build per-Linear format costs
    3. allocator.py                      choose Pareto layer-format assignments
    4. validate_assignments_kl.py        measure held-out KL for candidates
    5. export_native_compressed.py       write compressed-tensors artifacts
    6. validate_native_export.py         run vLLM load/generation smoke
    7. validate_quantized_model.py       run downstream quality checks

Older cross-layer allocators are archived under archive/cross_layer_2026-05-09
for artifact replay and comparison."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prismaquant")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

_LAZY_EXPORTS = {
    "FormatSpec": ".format_registry", "REGISTRY": ".format_registry",
    "register_format": ".format_registry",
    "initialize_model_runtime": ".model_runtime",
    "genuine_weight_initialization": ".model_runtime",
    "pretrained_initialization_contract": ".streaming_initialization",
    "validate_pretrained_initialization_contract": ".streaming_initialization",
    "validate_source_initialization_contract": ".streaming_initialization",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
