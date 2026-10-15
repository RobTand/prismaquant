"""Measured-unit accumulation, joint row completion and resume policy."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re

import torch

from .cost_currency import probe_identity_seals, probe_identity_walls_differ
from .dev_mode import seal_check
from .joint_eval_observation import observe_probe


class ProbeResumePolicy(Enum):
    EXACT = "exact"
    RUN_SEALS = "run-seals"


@dataclass(frozen=True)
class RequiredAttribution:
    keys: set[tuple[str, str]] | None
    refusal: Callable[[str, str], Exception]

    def require(self, name, fmt, row, *, zero_cost_formats):
        if ((self.keys is None or (name, fmt) in self.keys)
                and "sequence_attribution" not in row and fmt not in zero_cost_formats):
            raise self.refusal(name, fmt)


@dataclass(frozen=True)
class JointResumeContract:
    probe: ProbeResumePolicy
    activation_identities: Mapping | None = None
    attribution: RequiredAttribution | None = None

    def require_probe(self, expected_identity, stored_identity, *, name, fmt):
        if self.probe is ProbeResumePolicy.EXACT:
            if stored_identity != expected_identity:
                raise ValueError("probe/operator alignment mismatch")
        else:
            if probe_identity_walls_differ(expected_identity, stored_identity):
                raise ValueError("probe/operator alignment mismatch")
            seal_check("joint probe identity", probe_identity_seals(expected_identity),
                       probe_identity_seals(stored_identity),
                       where=f"joint AURA checkpoint {name}@{fmt}",
                       same=stored_identity == expected_identity,
                       refusal=lambda: ValueError("probe/operator alignment mismatch"))

    def require_activation(self, name, fmt, operator):
        if self.activation_identities is not None:
            expected_identity = self.activation_identities[name][fmt]
            if operator["activation"] != expected_identity:
                raise ValueError("activation identity mismatch")


class AuraUnitMeasurements:
    """The accumulated measurement vectors and their checkpoint state."""

    def __init__(self, *, n_probes, render_formats, collect_col_energy=False,
                 diagnostic_weight_mse_pairs=None, require_source_weight_identity=False,
                 observation_counts=None):
        self.n_probes = n_probes
        self.render_formats = render_formats
        self.collect_col_energy = collect_col_energy
        self.diagnostic_weight_mse_pairs = diagnostic_weight_mse_pairs
        self.require_source_weight_identity = require_source_weight_identity
        self.observation_counts = observation_counts
        self.s2 = {}
        self.s4 = {}
        self.x2_probe = {}
        self.dw_src = {}
        self.g_trace = {}
        self.col_energy = {}
        self.weight_mse_diagnostic = {}
        self.source_weight_identity = {}

    def restore_unit(self, name, state):
        _restore_unit_state(
            name, state, nonzero_formats=self.render_formats[name],
            n_probes=self.n_probes, collect_col_energy=self.collect_col_energy,
            s2=self.s2, s4=self.s4, x2_probe=self.x2_probe, dw_src=self.dw_src,
            g_trace=self.g_trace, col_energy=self.col_energy,
            diagnostic_weight_mse_pairs=self.diagnostic_weight_mse_pairs,
            weight_mse_diagnostic=self.weight_mse_diagnostic,
            require_source_weight_identity=self.require_source_weight_identity,
            source_weight_identity=self.source_weight_identity,
            observation_counts=self.observation_counts)

    def captured_state(self, name):
        return _capture_unit_state(
            name, self.render_formats[name], s2=self.s2, s4=self.s4,
            x2_probe=self.x2_probe, dw_src=self.dw_src, g_trace=self.g_trace,
            col_energy=self.col_energy, weight_mse_diagnostic=self.weight_mse_diagnostic,
            source_weight_identity=self.source_weight_identity,
            observation_counts=self.observation_counts)


class JointPriceStep:
    """Finish measured units; replay and incoming cotangent adapters supply probes."""

    def __init__(self, measurements, *, probe_identity, formats_by_qname,
                 zero_cost_formats, rendered_identities, resume_contract,
                 validated_probe=None):
        self.measurements = measurements
        self.probe_identity = probe_identity
        self.formats_by_qname = formats_by_qname
        self.zero_cost_formats = zero_cost_formats
        self.rendered_identities = rendered_identities
        self.resume_contract = resume_contract
        self._checked_probe = probe_identity if validated_probe is None else validated_probe
        self._validated_probe = validated_probe
        self.rows = {}
        self.components = {}
        self.operators = {}

    def restore_unit(self, name, state):
        from prismaquant.joint_aura_rows import validate_joint_aura_entry
        self.measurements.restore_unit(name, state)
        rows = state.get("joint_aura_rows")
        if not isinstance(rows, Mapping) or set(rows) != set(self.formats_by_qname[name]):
            raise RuntimeError(f"joint AURA checkpoint row scope mismatch for {name}")
        for fmt, row in rows.items():
            try:
                if not validate_joint_aura_entry(row):
                    raise ValueError("not a joint row")
                if self.resume_contract.attribution is not None:
                    self.resume_contract.attribution.require(
                        name, fmt, row, zero_cost_formats=self.zero_cost_formats)
                operator = row["joint_operator_identity"]
                if operator["qname"] != name or operator["format"] != fmt:
                    raise ValueError("probe/operator alignment mismatch")
                self.resume_contract.require_probe(
                    self.probe_identity, row["probe_identity"], name=name, fmt=fmt)
                self.resume_contract.require_activation(name, fmt, operator)
                if fmt in self.measurements.render_formats[name]:
                    if self.rendered_identities is not None:
                        expected_rendered_identity = self.rendered_identities[name][fmt]
                        if operator["rendered_weight"] != expected_rendered_identity:
                            raise ValueError("actual rendered-weight identity mismatch")
                    if row["x2_per_probe"] != self.measurements.x2_probe[(name, fmt)]:
                        raise ValueError("legacy/joint sample alignment mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"joint AURA checkpoint identity mismatch for {name}@{fmt}: {exc}") from exc
        self.rows[name] = dict(rows)

    def consume_probe(self, probe_index, terms, diagnostics):
        from prismaquant.joint_aura_rows import squared_signed
        values = self.measurements
        for name, diagnostic in diagnostics.items():
            values.g_trace[name] += diagnostic["g_trace"]
            if values.observation_counts is not None:
                observe_probe(values.observation_counts, name, probe_index, diagnostic)
            if values.collect_col_energy:
                previous = values.col_energy.get(name)
                values.col_energy[name] = (diagnostic["col_energy"] if previous is None
                                           else previous + diagnostic["col_energy"])
        for key, components in terms.items():
            self.components[key].append(components)
            value = squared_signed(components["total"])
            values.s2[key] += value
            values.s4[key] += value * value
            values.x2_probe[key].append(value)

    def prepare_rows(self, names, *, sidecar=None):
        from prismaquant.joint_aura_rows import make_joint_aura_entry
        for name in names:
            self.rows[name] = {}
            for fmt in self.formats_by_qname[name]:
                if fmt in self.zero_cost_formats:
                    components = [{"weight": 0.0, "activation": 0.0,
                                   "mixed": 0.0, "total": 0.0}
                                  for _ in range(self.measurements.n_probes)]
                else:
                    components = self.components[(name, fmt)]
                attribution = None if sidecar is None else sidecar(name, fmt, components)
                row = make_joint_aura_entry(operator_identity=self.operators[(name, fmt)],
                    probe_identity=self._checked_probe, signed_components=components,
                    sequence_attribution=attribution)
                if self._validated_probe is not None:
                    row["probe_identity"] = self.probe_identity
                self.rows[name][fmt] = row

    def captured_state(self, name):
        return {**self.measurements.captured_state(name), "joint_aura_rows": self.rows[name]}

    def complete_unit(self, name, *, sidecar=None):
        self.prepare_rows([name], sidecar=sidecar)
        return self.captured_state(name)

    def construction_references(self, name):
        values = self.measurements
        references = [self.probe_identity, values.g_trace[name]]
        for fmt in self.formats_by_qname[name]:
            references.extend((fmt, self.operators[(name, fmt)]))
            if fmt not in self.zero_cost_formats:
                references.append(self.components[(name, fmt)])
        for fmt in values.render_formats[name]:
            key = (name, fmt)
            if key in values.s2:
                references.extend((values.x2_probe[key], values.dw_src[key]))
        return references


def _capture_unit_state(
    name: str,
    nonzero_formats: Sequence[str],
    *,
    s2: Mapping[tuple[str, str], float],
    s4: Mapping[tuple[str, str], float],
    x2_probe: Mapping[tuple[str, str], list[float]],
    dw_src: Mapping[tuple[str, str], str],
    g_trace: Mapping[str, float],
    col_energy: Mapping[str, torch.Tensor],
    weight_mse_diagnostic: Mapping[tuple[str, str], float] | None = None,
    source_weight_identity: Mapping[str, Mapping[str, object]] | None = None,
    observation_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for fmt in nonzero_formats:
        key = (name, fmt)
        if key not in s2:
            continue
        rows[fmt] = {
            "s2": float(s2[key]),
            "s4": float(s4[key]),
            "x2_probe": list(x2_probe[key]),
            "dw_src": str(dw_src[key]),
        }
        if (
            weight_mse_diagnostic is not None
            and key in weight_mse_diagnostic
        ):
            rows[fmt]["weight_mse_diagnostic"] = float(
                weight_mse_diagnostic[key]
            )
    state = {
        "g_trace": float(g_trace[name]),
        "rows": rows,
        "col_energy": (
            col_energy[name].detach().to(device="cpu", dtype=torch.float32)
            if name in col_energy
            else None
        ),
    }
    if source_weight_identity is not None and name in source_weight_identity:
        state["source_weight_identity"] = dict(
            source_weight_identity[name]
        )
    if observation_counts is not None:
        state['joint_eval_observations'] = dict(observation_counts[name])
    return state


def _restore_unit_state(
    name: str,
    state: Mapping[str, object],
    *,
    nonzero_formats: Sequence[str],
    n_probes: int,
    collect_col_energy: bool,
    s2: dict[tuple[str, str], float],
    s4: dict[tuple[str, str], float],
    x2_probe: dict[tuple[str, str], list[float]],
    dw_src: dict[tuple[str, str], str],
    g_trace: dict[str, float],
    col_energy: dict[str, torch.Tensor],
    diagnostic_weight_mse_pairs: set[tuple[str, str]] | None = None,
    weight_mse_diagnostic: dict[tuple[str, str], float] | None = None,
    require_source_weight_identity: bool = False,
    source_weight_identity: dict[str, dict[str, object]] | None = None,
    observation_counts: dict[str, dict[str, int]] | None = None,
) -> None:
    if observation_counts is not None:
        observation = state.get('joint_eval_observations')
        if (not isinstance(observation, Mapping)
                or set(observation) != {'tokens', 'calls', 'per_probe', 'count_scope', 'n_probes'}
                or observation.get('count_scope') != 'summed_over_probes'
                or observation.get('n_probes') != n_probes
                or not isinstance(observation.get('per_probe'), list)
                or len(observation['per_probe']) != n_probes
                or any(not isinstance(item, Mapping) or set(item) != {'tokens', 'calls'}
                       or any(type(value) is not int or value < 0 for value in item.values())
                       for item in observation['per_probe'])
                or any(type(observation.get(key)) is not int
                       or observation[key] != sum(item[key] for item in observation['per_probe'])
                       for key in ('tokens', 'calls'))):
            raise RuntimeError(f'joint pilot checkpoint observation counts missing for {name}')
        observation_counts[name] = {**observation, 'per_probe': [dict(item) for item in observation['per_probe']]}
    try:
        g_trace[name] = float(state["g_trace"])
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid g_trace; "
            "refusing reuse or recompute"
        ) from exc
    rows = state.get("rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid rows; "
            "refusing reuse or recompute"
        )
    unexpected = sorted(set(str(fmt) for fmt in rows) - set(nonzero_formats))
    if unexpected:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has unexpected formats "
            f"{unexpected}; refusing reuse or recompute"
        )
    for raw_fmt, raw_row in rows.items():
        fmt = str(raw_fmt)
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is invalid; "
                "refusing reuse or recompute"
            )
        samples = raw_row.get("x2_probe")
        if (
            not isinstance(samples, Sequence)
            or isinstance(samples, (str, bytes))
            or len(samples) != int(n_probes)
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} has "
                f"x2_probe length {len(samples) if isinstance(samples, Sequence) else '<invalid>'}; "
                f"expected {n_probes}; refusing reuse or recompute"
            )
        key = (name, fmt)
        try:
            s2[key] = float(raw_row["s2"])
            s4[key] = float(raw_row["s4"])
            x2_probe[key] = [float(value) for value in samples]
            dw_src[key] = str(raw_row["dw_src"])
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is malformed; "
                "refusing reuse or recompute"
            ) from exc
        expected_diagnostic = (
            diagnostic_weight_mse_pairs is not None
            and key in diagnostic_weight_mse_pairs
        )
        observed_diagnostic = "weight_mse_diagnostic" in raw_row
        if observed_diagnostic != expected_diagnostic:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} diagnostic "
                "weight-MSE scope differs; refusing reuse or recompute"
            )
        if expected_diagnostic:
            if weight_mse_diagnostic is None:
                raise RuntimeError(
                    "AURA diagnostic weight-MSE restore has no destination"
                )
            try:
                weight_mse_diagnostic[key] = float(
                    raw_row["weight_mse_diagnostic"]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"AURA unit checkpoint state for {name}@{fmt} has an "
                    "invalid diagnostic weight-MSE"
                ) from exc
    stored_col_energy = state.get("col_energy")
    if collect_col_energy:
        if stored_col_energy is not None and not isinstance(
            stored_col_energy, torch.Tensor
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "col_energy; refusing reuse or recompute"
            )
        if isinstance(stored_col_energy, torch.Tensor):
            col_energy[name] = stored_col_energy.detach().to(
                device="cpu", dtype=torch.float32
            )
    elif stored_col_energy is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries "
            "col_energy; refusing reuse or recompute"
        )
    raw_source = state.get("source_weight_identity")
    if require_source_weight_identity:
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has no production "
                "anchor source-weight identity; refusing reuse or recompute"
            )
        try:
            shape = [int(dim) for dim in raw_source["shape"]]
            digest = str(raw_source["sha256"]).lower()
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has malformed "
                "source-weight identity"
            ) from exc
        if (
            len(shape) != 2
            or any(dim <= 0 for dim in shape)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "source-weight shape/hash"
            )
        if source_weight_identity is None:
            raise RuntimeError(
                "AURA source-weight identity restore has no destination"
            )
        source_weight_identity[name] = {
            "shape": shape,
            "sha256": digest,
        }
    elif raw_source is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries a "
            "production-anchor source identity"
        )


