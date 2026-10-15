"""Immutable native numerical options, resolved once at a render boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Mapping

SCHEMA = "prismaquant.native_render_levers.v1"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _int_value(raw, default):
    try:
        return int(raw)
    except Exception:
        return default


def _positive_float(raw, default):
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0.0 else default


def _roles(raw) -> tuple[tuple[str, float], ...]:
    table = {}
    for part in str(raw).split(","):
        role, sep, value = part.strip().partition("=")
        role = role.strip()
        if not sep or not role:
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        if value > 0.0:
            table[role] = value
    return tuple(sorted(table.items()))


def _int_set(raw) -> tuple[int, ...]:
    values = []
    for part in str(raw).split(","):
        try:
            values.append(int(part.strip()))
        except ValueError:
            continue
    values.append(0)
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class NativeRenderLevers:
    gptq: bool = False
    scale_sweep: bool = False
    static_act_order: bool = False
    joint_scale_opt: bool = False
    fisher_gptq: bool = False
    weighted_vq: bool = True
    render_progressive_gates: bool = True
    render_gate_min_gain: float = 0.0
    production_activation_scales: bool = True
    gptq_damp_sweep: bool = False
    gptq_fixed_damp: float = 1.0
    gptq_damp_roles: tuple[tuple[str, float], ...] = ()
    gptq_block_size: int = 128
    nvfp4_scale_rule: str = "static_6"
    nvfp4_joint_scale_levels: tuple[float, ...] = (6.0, 4.0)
    nvfp4_joint_scale_global_grid: int = 5
    nvfp4_joint_scale_global_span_lo: float = 0.75
    nvfp4_joint_scale_global_span_hi: float = 1.25
    nvfp4_snapped_scale_scoring: bool = False
    nvfp4_input_global_scale_policy: str = "legacy_6_over_calibration_amax.v1"
    act_clip_quantile: float | None = 0.999
    fisher_row_weight_clip: float = 64.0
    do_no_harm: bool = True
    mxfp8_joint_scale_shifts: tuple[int, ...] = (-1, 0)
    mxfp8_scale_sweep_shifts: tuple[int, ...] = (0,)
    fp8_scale_sweep_factors: tuple[float, ...] = (
        0.25, 0.3535533906, 0.5, 0.7071067812, 0.8408964153,
        1.0, 1.189207115, 1.414213562, 2.0)
    gptq_damp_analytical: bool = False
    gptq_damp_analytical_c: float = 1.784e-5
    batched_nvfp4_export: bool = True
    nvfp4_export_match_render_scale: bool = True

    def record(self) -> dict:
        value = asdict(self)
        value["schema"] = SCHEMA
        value["gptq_damp_roles"] = dict(self.gptq_damp_roles)
        for key in ("nvfp4_joint_scale_levels", "mxfp8_joint_scale_shifts",
                    "mxfp8_scale_sweep_shifts", "fp8_scale_sweep_factors"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_record(cls, record: Mapping[str, object]):
        """Parse a complete stored recipe without reading the environment."""
        if not isinstance(record, Mapping):
            raise ValueError("native render options must be a mapping")
        missing = set(cls.__dataclass_fields__) - set(record)
        if record.get("schema") != SCHEMA or missing:
            raise ValueError(f"incomplete native render options: {sorted(missing)}; re-render this cache")
        values = {key: record[key] for key in cls.__dataclass_fields__}

        def refuse(key, requirement):
            raise ValueError(f"native render option {key} {requirement}")

        def finite_number(value):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False

        for key, field in cls.__dataclass_fields__.items():
            value = values[key]
            if isinstance(field.default, bool) and type(value) is not bool:
                refuse(key, "must be a boolean")
            if type(field.default) is int and type(value) is not int:
                refuse(key, "must be an integer")
            if (isinstance(field.default, float)
                    and not (key == "act_clip_quantile" and value is None)
                    and not finite_number(value)):
                refuse(key, "must be finite numeric data")
            if isinstance(field.default, str) and not isinstance(value, str):
                refuse(key, "must be a string")

        roles = values["gptq_damp_roles"]
        if not isinstance(roles, Mapping) or any(
            not isinstance(role, str) or not role
            or not finite_number(damp) or damp <= 0
            for role, damp in roles.items()
        ):
            refuse("gptq_damp_roles", "must map roles to finite positive damping")
        values["gptq_damp_roles"] = tuple(sorted(roles.items()))

        for key in ("nvfp4_joint_scale_levels", "mxfp8_joint_scale_shifts",
                    "mxfp8_scale_sweep_shifts", "fp8_scale_sweep_factors"):
            sequence = values[key]
            if not isinstance(sequence, (list, tuple)) or not sequence:
                refuse(key, "must be a nonempty numerical sequence")
            if key in ("mxfp8_joint_scale_shifts", "mxfp8_scale_sweep_shifts"):
                valid = all(type(item) is int for item in sequence)
            else:
                valid = all(finite_number(item) and item > 0 for item in sequence)
            if not valid:
                refuse(key, "contains invalid numerical entries")
            values[key] = tuple(sequence)

        quantile = values["act_clip_quantile"]
        if quantile is not None and not (
            finite_number(quantile) and 0 < quantile < 1
        ):
            refuse("act_clip_quantile", "must be None or a finite quantile in (0, 1)")
        if values["gptq_fixed_damp"] <= 0:
            refuse("gptq_fixed_damp", "must be positive")
        if values["gptq_block_size"] < 1:
            refuse("gptq_block_size", "must be positive")
        if not 1 <= values["nvfp4_joint_scale_global_grid"] <= 33:
            refuse("nvfp4_joint_scale_global_grid", "must be in [1, 33]")
        if (values["nvfp4_joint_scale_global_span_lo"] <= 0
                or values["nvfp4_joint_scale_global_span_hi"]
                < values["nvfp4_joint_scale_global_span_lo"]):
            refuse("nvfp4_joint_scale_global_span_hi", "must bound a positive ordered span")
        if not -1 <= values["render_gate_min_gain"] <= 1:
            refuse("render_gate_min_gain", "must be in [-1, 1]")

        from .native_render_codec import resolve_nvfp4_scale_rule
        from .nvfp4_activation_contract import resolve_input_global_scale_policy
        for key, resolver in (
            ("nvfp4_scale_rule", resolve_nvfp4_scale_rule),
            ("nvfp4_input_global_scale_policy", resolve_input_global_scale_policy),
        ):
            try:
                canonical = resolver(values[key])
            except ValueError as error:
                raise ValueError(f"native render option {key} is unsupported") from error
            if values[key] != canonical:
                refuse(key, "must use its canonical policy name")
        return cls(**values)

    @classmethod
    def resolve(cls, overrides: Mapping[str, object] | None = None, *,
                gptq_default: bool = False, optional_flags_from_env: bool = False):
        """Bind existing defaults and every numerical environment control."""
        from .native_render_codec import resolve_nvfp4_scale_rule
        from .nvfp4_activation_contract import resolve_input_global_scale_policy
        from .render_score import resolve_fisher_row_weight_clip
        if isinstance(overrides, cls):
            return overrides
        if overrides is not None and overrides.get("schema") == SCHEMA:
            return cls.from_record(overrides)
        values = dict(overrides or {})
        optional_defaults = not bool(values.pop("none", False))
        for name, env in (("scale_sweep", None),
                          ("static_act_order", "PRISMAQUANT_GPTQ_STATIC_ACT_ORDER"),
                          ("joint_scale_opt", "PRISMAQUANT_NVFP4_JOINT_SCALE_OPT"),
                          ("fisher_gptq", "PRISMAQUANT_FISHER_WEIGHTED_GPTQ")):
            values.setdefault(name, bool(optional_defaults and optional_flags_from_env and env and _flag(env)))
        values.setdefault("gptq", bool(optional_defaults and gptq_default))
        values.setdefault("render_progressive_gates", _flag("PRISMAQUANT_RENDER_PROGRESSIVE_GATES", True))
        try:
            minimum_gain = float(os.environ.get("PRISMAQUANT_RENDER_GATE_MIN_GAIN", "0"))
        except Exception:
            minimum_gain = 0.0
        values.setdefault("render_gate_min_gain", max(-1.0, min(1.0, minimum_gain)))
        values.setdefault("production_activation_scales", _flag("PRISMAQUANT_PROD_ACT_SCALES", True))
        values["static_act_order"] = bool(values["gptq"] and values["static_act_order"])
        values["joint_scale_opt"] = bool(values["gptq"] and values["joint_scale_opt"])
        values.setdefault("gptq_damp_sweep", bool(values["gptq"]) and os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0") != "0")
        values.setdefault("gptq_fixed_damp", _positive_float(os.environ.get("PRISMAQUANT_GPTQ_DAMP", ""), 1.0))
        values.setdefault("gptq_damp_roles", _roles(os.environ.get("PRISMAQUANT_GPTQ_DAMP_ROLES", "")))
        if isinstance(values["gptq_damp_roles"], Mapping):
            values["gptq_damp_roles"] = tuple(sorted(values["gptq_damp_roles"].items()))
        else:
            values["gptq_damp_roles"] = tuple(tuple(row) for row in values["gptq_damp_roles"])
        values.setdefault("gptq_block_size", max(1, _int_value(os.environ.get("PRISMAQUANT_GPTQ_BLOCK_SIZE", os.environ.get("PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE", "128")), 128)))
        if values["joint_scale_opt"] and "nvfp4_scale_rule" not in values and "PRISMAQUANT_NVFP4_SCALE_RULE" not in os.environ:
            values["nvfp4_scale_rule"] = "joint_mse"
        values["nvfp4_scale_rule"] = resolve_nvfp4_scale_rule(values.get("nvfp4_scale_rule"))
        raw_levels = os.environ.get("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS")
        try:
            levels = tuple(float(x) for x in raw_levels.replace(",", " ").split()) if raw_levels else ()
        except ValueError:
            levels = ()
        values.setdefault("nvfp4_joint_scale_levels", levels or (6.0, 4.0))
        values["nvfp4_joint_scale_levels"] = tuple(values["nvfp4_joint_scale_levels"])
        values.setdefault("nvfp4_joint_scale_global_grid", max(1, min(33, _int_value(os.environ.get("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID", "5"), 5))))
        lo = float(os.environ.get("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO", "0.75"))
        hi = float(os.environ.get("PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI", "1.25"))
        if not math.isfinite(lo) or lo <= 0.0:
            lo = 0.75
        if not math.isfinite(hi) or hi < lo:
            hi = max(lo, 1.25)
        values.setdefault("nvfp4_joint_scale_global_span_lo", lo)
        values.setdefault("nvfp4_joint_scale_global_span_hi", hi)
        values.setdefault("nvfp4_snapped_scale_scoring", os.environ.get("PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING", "0") != "0")
        values["nvfp4_input_global_scale_policy"] = resolve_input_global_scale_policy(values.get("nvfp4_input_global_scale_policy"))
        raw = os.environ.get("PRISMAQUANT_ACT_CLIP_QUANTILE", "0.999")
        try:
            quantile = float(raw) if raw else None
        except ValueError:
            quantile = None
        values.setdefault("act_clip_quantile", quantile if quantile is not None and 0.0 < quantile < 1.0 else None)
        values.setdefault("fisher_row_weight_clip", resolve_fisher_row_weight_clip())
        values.setdefault("do_no_harm", os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0")
        values.setdefault("mxfp8_joint_scale_shifts", _int_set(os.environ.get("PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS", "-1,0")))
        try:
            shifts = tuple(int(x.strip()) for x in os.environ.get("PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS", "0").split(",") if x.strip())
        except Exception:
            shifts = (0,)
        values.setdefault("mxfp8_scale_sweep_shifts", shifts or (0,))
        factors = []
        for part in os.environ.get("PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS", "0.25,0.3535533906,0.5,0.7071067812,0.8408964153,1.0,1.189207115,1.414213562,2.0").split(","):
            try:
                value = float(part.strip())
            except ValueError:
                continue
            if math.isfinite(value) and value > 0.0:
                factors.append(value)
        factors.append(1.0)
        values.setdefault("fp8_scale_sweep_factors", tuple(sorted(set(factors))))
        for key in ("mxfp8_joint_scale_shifts", "mxfp8_scale_sweep_shifts", "fp8_scale_sweep_factors"):
            values[key] = tuple(values[key])
        values.setdefault("gptq_damp_analytical", os.environ.get("PRISMAQUANT_DAMP_ANALYTICAL", "").lower() in {"1", "true", "yes", "on", "kappa_target"})
        try:
            coefficient = float(os.environ.get("PRISMAQUANT_DAMP_ANALYTICAL_C", "1.784e-5"))
        except ValueError:
            coefficient = 1.784e-5
            values["gptq_damp_analytical"] = False
        values.setdefault("gptq_damp_analytical_c", coefficient)
        values.setdefault("batched_nvfp4_export", os.environ.get("PRISMAQUANT_BATCHED_NVFP4_EXPORT", "1") not in ("0", "", "false", "False", "FALSE", "no", "NO"))
        values.setdefault("nvfp4_export_match_render_scale", os.environ.get("PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE", "1") != "0")
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})
