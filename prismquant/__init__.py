"""PrismQuant: mixed-native quantization policy engine for LLMs.

Canonical pipeline:
    1. sensitivity_probe.py         measure per-Linear Fisher sensitivity
    2. measure_quant_cost.py        measure per-Linear × per-format functional cost
    3. allocator.py                 solve additive mixed-format frontier
    4. measure_interactions.py      probe sparse pairwise interactions near the knee
    5. quadratic_refine_allocator.py refine the knee neighborhood under interactions
    6. calibrate_allocator.py       validate predicted frontier points against real KL

The additive allocator stays as the fast global backbone. Interaction and
calibration stages refine the deployment point without turning the whole
optimization into an intractable dense quadratic problem.
"""
from .format_registry import FormatSpec, REGISTRY, register_format
