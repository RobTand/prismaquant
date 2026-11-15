"""Check the merge settings when no acquisition request is present."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from prismaquant import tessera_campaign_table as campaign_table

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.model_profiles import DefaultProfile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.tessera_full_domain_acquisition import (
    joint_acquisition_from_cost_data, load_joint_campaign_acquisition,

)
from prismaquant.tessera_legal_domain import live_pins, tessera_source_state
from test_streamed_cost_checkpoints import (
    _DenseTinyLM, _FakeStreamingContext, _model_identity,
)

FAMILY = "TESSERA_E4M3_K1"
DEFERRED = "TESSERA_BF16_K1"
FORMATS = (FAMILY + "_R256", FAMILY + "_R2048")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def real_joint_run():
    import prismaquant.aura_cost as aura
    pytest.importorskip("tessera")
    torch.manual_seed(85)
    model = _DenseTinyLM(width=32, layers=3).eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _FakeStreamingContext(model)
    runner = StreamedCausalLM(context, DefaultProfile())
    names = sorted(name for name, _ in model.named_modules() if name.endswith(".proj"))
    source_weights = {name: model.get_submodule(name).weight.detach().clone() for name in names}
    captures, handles = {}, []
    for name in names:
        def capture(_module, inputs, _output, name=name):
            if name not in captures:
                captures[name] = inputs[0].detach().clone().reshape(-1, inputs[0].shape[-1])
        handles.append(model.get_submodule(name).register_forward_hook(capture))
    cache = ProductionWeightCache(weights={(name, fmt): weight + delta
        for name, weight in source_weights.items() for fmt, delta in zip(FORMATS, (0.25, 0.03125))},
        levers={}, activation_max_abs={name: 1.0 for name in names})
    tokens = torch.tensor([[1, 2, 3, 4]])
    try:
        payload = aura.compute_aura_cost_streamed(runner, tokens, [*FORMATS, "BF16"],
            n_probes=3, min_free_gib=0, production_cache=cache, joint_activation=True,
            model_identity=_model_identity("joint-acquisition-pb-handoff"))
    finally:
        for handle in handles:
            handle.remove()
    payload = pickle.loads(pickle.dumps(payload))
    shapes = {name: list(weight.shape) for name, weight in source_weights.items()}
    active = joint_acquisition_from_cost_data(payload, shapes, [FAMILY], max_new_points=1)
    deferred = joint_acquisition_from_cost_data(payload, shapes, [DEFERRED],
        max_new_points=0, boundary_policy="defer")
    idle_name = next((r["unit_name"] for r in active["reports"] if not r["proposed_q256"]),
                     names[-1])
    idle = joint_acquisition_from_cost_data(payload, {idle_name: shapes[idle_name]}, [FAMILY],
        max_new_points=0, boundary_policy="defer")["reports"][0]
    active["reports"] = [idle if r["unit_name"] == idle_name else r for r in active["reports"]]
    active["reports"].extend(deferred["reports"])
    document = dict(schema="prismaquant.tessera_full_domain_campaign_acquisition.v1",
        source_tensor_inventory_sha256=sha(json.dumps([
            {"name": name + ".weight", "shape": shape} for name, shape in sorted(shapes.items())]).encode()),
        journal_bindings={}, active_encoder_source_sha256=None,
        domain_pins=live_pins().as_dict(), producer_source_state=tessera_source_state(),
        atomic_serving_group_expansion_required=True, allocator_payload=False,
        production_qualified=False, **active)
    document["total_requested_quality_measurements"] = sum(len(r["proposed_q256"]) for r in document["reports"])
    active_names = sorted(r["unit_name"] for r in document["reports"]
        if r["proposed_q256"])
    assert len(active_names) >= 2, "the two-atomic-row scenario needs two proposing units"
    assert all(not r["proposed_q256"] for r in document["reports"] if r["unit_name"] == idle_name)
    return SimpleNamespace(payload=payload, document=document, names=names, shapes=shapes,
        weights=source_weights, inputs=captures, tokens=tokens, idle_name=idle_name,
        active_names=active_names)


def test_no_acquisition_keeps_legacy_settings_and_manifest_shape():
    assert campaign_table._merge_acquisition_settings({"row": {"settings": {"legacy": True}, "units": {}}},
        {"row": {}}, acquisition=None, scope_groups=None) is None
