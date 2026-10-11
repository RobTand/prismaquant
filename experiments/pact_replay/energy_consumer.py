"""Synchronous live-row collector using the existing packed input/derivation owners.

No activation cache or reservoir. Prefix positions are removed from clean
module input BEFORE routed derivation; never from an expert's slot-major
row order. The clean forward still includes both prefix tokens. Each
sample is consumed before the next forward. Only energy sums persist.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import json


class LiveRows(AbstractContextManager):
    def __init__(self, model, profile, rows, consumer, *, prefix_rows=0):
        self.model, self.profile, self.rows = model, profile, list(rows)
        self.consumer, self.prefix_rows = consumer, prefix_rows
        self.handles, self.packed = [], None
        self.calls = {r["qname"]: 0 for r in self.rows}

    def _consume(self, names, value):
        for name in names:
            self.calls[name] += 1
        self.consumer(tuple(names), value)

    def __enter__(self):
        import torch
        from prismaquant.routed_experts import profile_declared_packed_expert_projections, packed_activation_input_kind
        from prismaquant.production_weight_cache import _PackedExpertActivationCollector
        from prismaquant.measure_quant_cost import derive_per_expert_activations, _packed_experts_parent_module
        modules = dict(self.model.named_modules())
        from menu_contract import executed_module
        routed = [r for r in self.rows if r["kind"] == "routed"]
        projected = profile_declared_packed_expert_projections(self.model, self.profile) if routed else []
        by_name = {m.qname: m for m in projected}
        selected = {}
        try:
            for row in self.rows:
                name = row["qname"]
                if name in modules:
                    module = modules[name]
                    if not isinstance(module, torch.nn.Linear):
                        raise ValueError("Declared dense input is not an actual Linear: " + name)
                    def tap(_module, args, name=name):
                        x = args[0].detach().reshape(-1, args[0].shape[-1])
                        self._consume((name,), x[self.prefix_rows:])
                    self.handles.append(module.register_forward_pre_hook(tap))
                elif name in by_name:
                    member = by_name[name]
                    selected.setdefault(member.module_qname, []).append((row, member))
                else:
                    module = executed_module(self.model, self.profile, row)
                    def tap(_module, args, name=name):
                        x = args[0].detach().reshape(-1, args[0].shape[-1])
                        self._consume((name,), x[self.prefix_rows:])
                    self.handles.append(module.register_forward_pre_hook(tap))
            parents = {q: _packed_experts_parent_module(self.model, q) for q in selected}
            def consume_packed(module_qname, x):
                members = selected[module_qname]
                clean_original = x.detach().reshape(-1, x.shape[-1])[self.prefix_rows:]
                # Existing router, bias correction, slot order and SwiGLU owner.
                derived = derive_per_expert_activations(members[0][1].module,
                    clean_original, parents[module_qname],
                    capture_down=any(r["role"] == "down_proj" for r, _ in members),
                    max_rows_per_expert=None)
                grouped = {}
                for row, member in members:
                    kind = packed_activation_input_kind(member.param_name)
                    grouped.setdefault((member.expert_id, kind), []).append(row["qname"])
                for (expert, kind), names in grouped.items():
                    self._consume(names, derived[kind][expert])
            if selected:
                self.packed = _PackedExpertActivationCollector(self.model, set(selected),
                    module_token_budget=0, store_device=next(self.model.parameters()).device,
                    store_qnames=set(), profile=self.profile, row_consumer=consume_packed)
                self.packed.install()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.packed is not None:
            self.packed.remove()
            self.packed = None
        return False


class SequenceEnergy:
    """Same actual consumer for CPU proof and resident GPU layer forwards."""
    def __init__(self, rows, weights, *, cohort, device, chunk_rows, guard, stream):
        self.rows = {r["qname"]: r for r in rows}
        self.weights, self.cohort = weights, cohort
        import torch
        requested = torch.device(device)
        actual = next(iter(weights.values()))["source"].device
        if requested.type != actual.type or (requested.index is not None
                and actual.index is not None and requested.index != actual.index):
            raise ValueError("Resident source and declared measurement devices differ")
        self.device, self.chunk_rows, self.guard, self.stream = actual, chunk_rows, guard, stream
        self.sample_id = None
        self.seen = set()
        self.counts = {}
        self.totals = {}

    def begin(self, sample_id):
        if self.sample_id is not None:
            raise RuntimeError("Previous sequence consumer is still open")
        self.sample_id = sample_id
        self.counts = {name: 0 for name in self.rows}

    def consume(self, names, x):
        from energy_math import batched_option_sums
        if self.sample_id is None or x.device != self.device:
            raise ValueError("Live row consumer lacks sample or measurement device")
        self.guard("live rows")
        for name in names:
            row = self.rows[name]
            entry = self.weights[name]
            self.counts[name] += len(x)
            specs = [(option["name"], decoded, option["contract"],
                (option.get("scale") or {}).get("effective"),
                option.get("tp_splits", 2 if row["role"] == "down_proj" else 1))
                for option, decoded in entry["options"]]
            for option, _decoded in entry["options"]:
                key = (name, option["name"], self.sample_id)
                if key in self.seen:
                    raise ValueError("Repeated unit/option/sample hook would double energy")
            results = batched_option_sums(x, entry["source"], specs,
                chunk_rows=self.chunk_rows, resource_check=self.guard)
            for option, _decoded in entry["options"]:
                sums = results[option["name"]]
                doc = {"schema": "pact.stage1a_per_sequence_energy.v1",
                    "qname": name, "layer": row["layer"], "kind": row["kind"], "role": row["role"],
                    "expert": row["expert"], "family": option["family"], "q256": option["q256"],
                    "weight_source": option["name"], "activation_contract": option["contract"],
                    "weight_reference_definition": option["weight_reference_definition"],
                    "body": option.get("body"), "outer_scheme": option.get("outer_scheme"),
                    "anchor_source": option.get("anchor_source"),
                    "wire_facts": option.get("wire_facts"),
                    "canonical_T16_priceable": option.get("canonical_T16_priceable"),
                    "sample_id": self.sample_id, "input_contract": self.cohort["input_contract"],
                    "original_tokens": 512, "global_original_tokens": self.cohort["global_original_tokens"],
                    "local_prefix_rows": self.cohort["local_prefix_rows"],
                    "prefix_rows_enter_local_energies": False,
                    "prefix_ids": self.cohort["prefix_ids"], "input_token_sha256": self.cohort["token_sha256"],
                    "dry_run_cpu": self.device.type == "cpu", "diagnostic_only": option.get("diagnostic_only", False),
                    "quality_chord_check_only": option.get("quality_chord_check_only", False),
                    "passthrough": option.get("passthrough", False),
                    "tp_splits": option.get("tp_splits",2 if row["role"] == "down_proj" else 1),
                    "scale": ({key:value for key,value in option["scale"].items() if key != "members"} | {"member_count":len(option["scale"].get("members",()))}) if option.get("scale") else None,
                    "wire": option.get("wire", option.get("location")),
                    **sums}
                self.stream.write(json.dumps(doc, allow_nan=False) + "\n")
                self.seen.add((name, option["name"], self.sample_id))
                aggregate_key = (name, option["name"])
                aggregate = self.totals.setdefault(aggregate_key, {"routed_rows": 0,
                    **{k: 0.0 for k in ("E_W_sum", "E_A_sum", "E_WA_sum", "E_W_sum_amp2", "E_A_sum_amp2", "E_WA_sum_amp2", "E_W_sum_amp4", "E_A_sum_amp4", "E_WA_sum_amp4")}})
                for k in aggregate:
                    aggregate[k] += sums[k]

    def end(self):
        for name, entry in self.weights.items():
            for option, _ in entry["options"]:
                if (name, option["name"], self.sample_id) not in self.seen:
                    raise ValueError("Missing unit/sample input observation: " + name)
        self.sample_id = None

    def finish(self, samples):
        expected = {(name, option["name"], sample) for name, entry in self.weights.items()
                    for option, _ in entry["options"] for sample in samples}
        if self.seen != expected:
            raise ValueError("Exact unique unit/option/sample coverage differs")
        return [{"qname": name, "weight_source": option, **sums,
                 **{k.replace("_sum", "_global"): v / self.cohort["global_original_tokens"]
                    for k, v in sums.items() if k != "routed_rows"}}
                for (name, option), sums in sorted(self.totals.items())]
