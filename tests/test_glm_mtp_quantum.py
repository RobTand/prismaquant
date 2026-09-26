"""GLM MTP layer 45 is priced on the MTP head's self-KL (PQ #1353, M5 of #1271).

``glm_mtp_quantum.compute_mtp_cost`` turns qualified renders into joint-AURA
rows with the body's statistics lease, seeded by ``glm_mtp.mtp_probe_scalar``
on the draft logits. These tests run a tiny random MTP layer on CPU:

* the rows equal an independent computation of ``<dL_k/dW, dW>`` from
  autograd parameter gradients, for the routed experts and the shared expert;
* a render, source or activation contract that differs from the prepared one
  refuses, as does a roster that differs from the priced units;
* the payload is what the allocator's MTP path (``glm_mtp_selection``) reads.

The rows are synthetic measurements on random weights; they make no claim
about GLM-5.3's costs.
"""
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from prismaquant import format_registry as fr  # noqa: E402
from prismaquant import genuine_weight_initialization  # noqa: E402
from prismaquant import glm_mtp, glm_mtp_quantum as quantum, glm_mtp_selection  # noqa: E402
from prismaquant import joint_aura as joint  # noqa: E402
from prismaquant.cost_currency import probe_identity_walls_differ  # noqa: E402
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA  # noqa: E402
from prismaquant.export_native_compressed import _split_packed_expert_tensor  # noqa: E402
from prismaquant.joint_statistics_replay import SCHEMA as WINDOWS_SCHEMA  # noqa: E402
from prismaquant.model_profiles.glm5_next import Glm5NextProfile  # noqa: E402
from prismaquant.production_weight_cache import (  # noqa: E402
    ProductionWeightCache, _cb_cache_tensor_identity,
)
from tests.test_glm_mtp_layer import BACKBONE, VOCAB, _randomize, _text_config  # noqa: E402

PROFILE = Glm5NextProfile()
PREFIX = f"model.language_model.layers.{BACKBONE}.mlp."
EXPERTS = 6
BF16_RUNG, E4M3_RUNG = "TESSERA_BF16_K1_R1024", "TESSERA_E4M3_K1_R1024"
RUNGS = (BF16_RUNG, E4M3_RUNG)
N_SEQUENCES, LENGTH, N_PROBES, SEED_BASE = 3, 9, 3, 7000


def _source_model():
    content = {"config": {"fixture": "mtp quantum"},
               "weight_map": {"fixture.weight": "fixture.safetensors"},
               "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "a" * 64}]}
    return {"schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic",
            "resolved_commit": None, **content, "content_sha256": joint.identity_sha256(content)}


def _layer(seed=20260927):
    with genuine_weight_initialization():
        built = glm_mtp.Glm5NextMtpLayer(_text_config())
    _randomize(built, seed)
    with torch.no_grad():
        # Keep every expert reachable on a few dozen rows (as the capture
        # tests do): a router bias of the scores' own scale starves some.
        built.mlp.gate.e_score_correction_bias.mul_(0.05)
    built = built.to(torch.float32).eval()
    for parameter in built.parameters():
        parameter.requires_grad_(False)
    return built


def _inputs(hidden_size):
    generator = torch.Generator().manual_seed(11)
    ids = torch.randint(0, VOCAB, (N_SEQUENCES, LENGTH), generator=generator)
    hidden = torch.randn(N_SEQUENCES, LENGTH, hidden_size, generator=generator)
    embed = torch.nn.Embedding.from_pretrained(
        torch.randn(VOCAB, hidden_size, generator=generator), freeze=True)
    head = torch.nn.Linear(hidden_size, VOCAB, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB, hidden_size, generator=generator) / 8.0)
    head.weight.requires_grad_(False)
    return ids, hidden, embed, head


class _Env:
    """A tiny MTP layer, its inputs, and a qualified cache of rendered rungs."""

    def __init__(self, tmp_path):
        self.layer = _layer()
        self.model = glm_mtp.MtpCheckpointModel(self.layer)
        self.ids, self.hidden, self.embed, self.head = _inputs(self.layer.config.hidden_size)
        self.modules = quantum.mtp_priced_modules(self.model, PROFILE)
        self.maxima = {name: 3.0 for name in self.modules}
        generator = torch.Generator().manual_seed(29)
        paths, self.deltas, verified = {}, {}, {}
        for name, module in self.modules.items():
            source = module.weight.detach()
            for index, fmt in enumerate(RUNGS):
                delta = torch.randn(source.shape, generator=generator) * (0.01 * (index + 1))
                rendered = (source + delta).contiguous()
                path = tmp_path / f"{name}.{fmt}.pt"
                torch.save(rendered, path)
                paths[(name, fmt)] = str(path)
                self.deltas[(name, fmt)] = rendered - source
                verified[(name, fmt)] = {
                    "source_weight": _cb_cache_tensor_identity(source),
                    "rendered_weight": _cb_cache_tensor_identity(rendered),
                    "activation": joint.activation_identity(fr.get_format(fmt), self.maxima, name),
                }
        self.cache = ProductionWeightCache(paths, {}, activation_max_abs=dict(self.maxima),
                                           metadata={"verified_cells": verified})
        largest = max(path.stat().st_size for path in tmp_path.glob("*.pt"))
        self.cache.enable_lru(8 * largest)
        self.windows = dict(
            schema=WINDOWS_SCHEMA, max_statistics_bytes=1 << 26, max_candidate_bytes=1 << 22,
            max_render_resident_bytes=8 * largest, max_load_buffer_bytes=largest,
            workspace_reserve_bytes=1 << 22, max_replay_cotangent_bytes=1 << 20,
            prefetch_workers=1)
        self.formats = {name: ["BF16", *RUNGS] for name in self.modules}
        self.wire = {name: {fmt: 100 + index for index, fmt in enumerate(RUNGS)}
                     for name in self.modules}
        groups = {f"s:{PREFIX}experts": sorted(n for n in self.modules if ".experts." in n)}
        groups[f"g:{PREFIX}shared_experts.gate_up_proj"] = [
            f"{PREFIX}shared_experts.gate_proj", f"{PREFIX}shared_experts.up_proj"]
        groups[f"u:{PREFIX}shared_experts.down_proj"] = [f"{PREFIX}shared_experts.down_proj"]
        self.census = {"anchor_groups": groups}

    def run(self, **overrides):
        hidden = self.hidden
        options = dict(
            production_cache=self.cache, formats_by_qname=self.formats, wire_bytes=self.wire,
            census=self.census, profile=PROFILE, n_probes=N_PROBES, seed_base=SEED_BASE,
            source_model=_source_model(), projection_backend=None,
            operator_windows=self.windows, device=torch.device("cpu"))
        options.update(overrides)
        return quantum.compute_mtp_cost(
            self.model, self.embed, self.head, self.ids,
            quantum.ResidentHiddenStates(lambda index: hidden[index:index + 1], N_SEQUENCES),
            **options)


@pytest.fixture
def env(tmp_path):
    return _Env(tmp_path)


def _reference_signed(env):
    """``sum_i <dL_{k,i}/dW, dW>`` per probe, from autograd parameter grads.

    Independent of the lease: a copy of the layer with trainable weights, the
    MTP seed per sequence, and the exporter's own split of each packed
    gradient into per-expert slices, contracted in float64.
    """
    layer = copy.deepcopy(env.layer)
    experts, shared = layer.mlp.experts, layer.mlp.shared_experts
    parameters = {"gate_up_proj": experts.gate_up_proj, "down_proj": experts.down_proj}
    linears = {name: module for name, module in shared.named_children()
               if isinstance(module, torch.nn.Linear)}
    for parameter in [*parameters.values(), *(m.weight for m in linears.values())]:
        parameter.requires_grad_(True)
    signed = {key: [] for key in env.deltas if key[1] == BF16_RUNG}
    for probe in range(N_PROBES):
        layer.zero_grad(set_to_none=True)
        for index in range(N_SEQUENCES):
            logits = glm_mtp.mtp_logits(layer, env.embed, env.head, env.ids[index:index + 1],
                                        env.hidden[index:index + 1])
            glm_mtp.mtp_probe_scalar(logits, seed=SEED_BASE + probe, global_row_offset=index,
                                     n_sequences=N_SEQUENCES).backward()
        grads = {}
        for param_name, parameter in parameters.items():
            for projection, per_expert in _split_packed_expert_tensor(
                    parameter.grad, param_name, PROFILE):
                for expert in range(per_expert.shape[0]):
                    grads[f"{PREFIX}experts.{expert}.{projection}"] = per_expert[expert]
        for name, module in linears.items():
            grads[f"{PREFIX}shared_experts.{name}"] = module.weight.grad
        for name, fmt in signed:
            value = (grads[name].double() * env.deltas[(name, fmt)].double()).sum()
            signed[(name, fmt)].append(float(value))
    return signed


def test_rows_equal_the_autograd_projection_of_the_mtp_seed(env):
    """Weight-only rows: each probe's signed term is <dL_k/dW, dW>, summed
    over sequences, for every routed expert and the shared expert."""
    payload = env.run()
    reference = _reference_signed(env)
    assert len(reference) == len(env.modules)
    reached = 0
    for (name, fmt), expected in reference.items():
        row = payload["costs"][name][fmt]
        assert row["signed_per_probe"] == pytest.approx(expected, rel=2e-4, abs=1e-9), name
        reached += any(abs(value) > 1e-9 for value in expected)
        components = row["signed_components_per_probe"]
        assert all(part["activation"] == 0.0 and part["mixed"] == 0.0 for part in components)
    # Most units see gradient; an all-zero reference would make the match vacuous.
    assert reached >= len(env.modules) - 2


def test_activation_rungs_carry_their_quantized_input_terms(env):
    """E4M3 quantizes its input, so its rows carry activation and mixed terms
    and differ from the weight-only projection of the same dW."""
    payload = env.run()
    shared = f"{PREFIX}shared_experts.gate_proj"
    parts = payload["costs"][shared][E4M3_RUNG]["signed_components_per_probe"]
    assert any(part["activation"] != 0.0 for part in parts)
    assert any(part["mixed"] != 0.0 for part in parts)
    for rows in payload["costs"].values():
        for row in rows.values():
            assert joint.validate_joint_aura_entry(row)


def test_rows_carry_the_mtp_objective_and_one_probe_identity(env):
    payload = env.run()
    probes = {row["probe_identity_sha256"] for rows in payload["costs"].values()
              for row in rows.values()}
    assert probes == {payload["provenance"]["probe_identity_sha256"]}
    probe = next(iter(next(iter(payload["costs"].values())).values()))["probe_identity"]
    assert probe["objective"]["objective"] == glm_mtp.MTP_OBJECTIVE
    assert probe["objective"]["mtp_layer"] == BACKBONE
    assert probe["noise_layout"]["global_token_count"] == N_SEQUENCES * (LENGTH - 1)
    assert probe["noise_layout"]["selected_tokens_per_row"] == LENGTH - 1
    body = {key: value for key, value in probe.items() if key != "objective"}
    assert probe_identity_walls_differ(body, probe)


def test_payload_is_what_the_allocator_mtp_path_reads(env, tmp_path):
    """BF16 passthrough is not a priced row; the selection adds it."""
    payload = env.run()
    assert all("BF16" not in rows for rows in payload["costs"].values())
    assert payload["params"][f"{PREFIX}shared_experts.down_proj"] == (
        env.modules[f"{PREFIX}shared_experts.down_proj"].weight.numel())
    path = tmp_path / "mtp-cost.pkl"
    path.write_bytes(pickle.dumps(payload))
    loaded = glm_mtp_selection.load_mtp_cost(path)
    record = glm_mtp_selection.select_mtp_rungs(
        loaded, byte_budget=10 ** 9,
        constants={"t_ms": 20.0, "d0_ms": 1.0, "c_ms_per_bit": 0.2, "source": "fixture"})
    assert set(record["assignment"]) == set(env.modules)
    assert set(record["rung_by_group"]) == set(env.census["anchor_groups"])


def test_a_render_the_preparation_did_not_qualify_refuses(env):
    name = f"{PREFIX}experts.3.down_proj"
    del env.cache.metadata["verified_cells"][(name, E4M3_RUNG)]
    with pytest.raises(RuntimeError, match="not qualified"):
        env.run()


def test_a_render_that_differs_from_the_prepared_one_refuses(env):
    name = f"{PREFIX}shared_experts.up_proj"
    cell = env.cache.metadata["verified_cells"][(name, BF16_RUNG)]
    cell["rendered_weight"] = dict(cell["rendered_weight"], content_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="render differs"):
        env.run()


def test_a_source_that_differs_from_the_prepared_one_refuses(env):
    name = f"{PREFIX}experts.0.gate_proj"
    for fmt in RUNGS:
        cell = env.cache.metadata["verified_cells"][(name, fmt)]
        cell["source_weight"] = dict(cell["source_weight"], content_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="source weight differs"):
        env.run()


def test_a_unit_outside_the_mtp_roster_refuses(env):
    env.formats["model.language_model.layers.1.mlp.down_proj"] = list(RUNGS)
    with pytest.raises(RuntimeError, match="roster differs"):
        env.run()


def test_a_served_activation_policy_refuses(env):
    env.cache._joint_served_activation = object()
    with pytest.raises(RuntimeError, match="served activation policy"):
        env.run()


def test_hidden_states_stream_once_then_replay():
    reads = []

    def read(index):
        reads.append(index)
        return torch.full((1, 2, 3), float(index))

    hidden = quantum.ResidentHiddenStates(read, 2)
    with pytest.raises(RuntimeError, match="in order"):
        hidden(1)
    assert float(hidden(0)[0, 0, 0]) == 0.0 and float(hidden(1)[0, 0, 0]) == 1.0
    hidden.require_complete()
    assert float(hidden(0)[0, 0, 0]) == 0.0
    assert reads == [0, 1]


def _scoped_call(census, *, source_layers=(BACKBONE,), joint_eval=None, windows=True):
    from types import SimpleNamespace

    runner = SimpleNamespace(source_layers=source_layers,
                             context=SimpleNamespace(install=lambda *a, **k: pytest.fail(
                                 "installed a layer before the inputs were admitted")))
    config = {"joint_eval": joint_eval, "execution": {}, "min_free_gib": 0}
    return quantum.run_mtp_scope(
        runner, config=config, data=SimpleNamespace(census=census), production_cache=None,
        calibration_ids=torch.zeros(1, 2, dtype=torch.int64),
        calibration={"calibration_sha256": "c" * 64}, source_model=None,
        projection_backend=None, operator_windows={"schema": WINDOWS_SCHEMA} if windows else None,
        device_bytes=None)


def test_the_run_refuses_a_census_that_is_not_the_mtp_capture_s():
    with pytest.raises(RuntimeError, match="not an MTP census"):
        _scoped_call({"anchor_groups": {}})


def test_the_run_refuses_a_source_that_does_not_hold_the_mtp_layer():
    census = {"mtp_extension": {"layer": BACKBONE,
                                "final_hidden": {"path": "/nonexistent", "sha256": "0" * 64}}}
    with pytest.raises(RuntimeError, match="not the MTP layer"):
        _scoped_call(census, source_layers=(0, 1))


def test_the_run_refuses_an_evaluation_panel_and_missing_windows():
    with pytest.raises(RuntimeError, match="full draw"):
        _scoped_call({}, joint_eval={"panel": 1})
    with pytest.raises(RuntimeError, match="operator windows"):
        _scoped_call({}, windows=False)
