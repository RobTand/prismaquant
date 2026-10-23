"""Joint AURA scalar rows, sample identity custody and coordinate validation.

This owner validates recorded scientific data without importing a tensor
runtime, loading a model or allocating a cache.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import re
from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex
from .dev_mode import seal_check


JOINT_CURRENCY = "joint_aura_predicted_dloss"


JOINT_AURA_COST_CURRENCY = JOINT_CURRENCY


JOINT_AURA_COST_SOURCE = "joint_aura"


PROBE_UNCERTAINTY_SCOPE = "probe_sampling_conditional_on_fixed_calibration"


# The descriptive sidecar never changes the authoritative whole-draw fields.
# Its reconstruction residual is reported and gated, never added to the price.
SEQUENCE_ATTRIBUTION_SCHEMA = "prismaquant.joint_aura.sequence_attribution.v1"


SEQUENCE_ATTRIBUTION_GATE_METHOD = (
    "residual_per_probe_leq_gate_relative_times_norm_per_probe"
    "_norm_is_fsum_abs_weight_activation_mixed_over_blocks"
    "_zero_norm_requires_exact_zero_residual")


SEQUENCE_LEAVEOUT_SCOPE = "descriptive_conditional_fixed_probes"


_ROW_UNCERTAINTY_SCOPES = frozenset({PROBE_UNCERTAINTY_SCOPE})


@dataclass(frozen=True, slots=True, init=False)
class _ValidatedProbeIdentity(Mapping):
    """Owned immutable JSON fields; mutable values are returned as fresh copies.

    Only this constructor can issue the fast-path type, after checking the
    source model. Serialization deliberately restores an ordinary dict so a
    validation result never crosses a persisted artifact boundary.
    """

    _fields: tuple
    _sha256: str

    def __init__(self, probe):
        encoded = DIRECT_ASCII_STRICT.encoded(probe)
        snapshot = json.loads(encoded)
        from prismaquant.streamed_model_identity import validate_streamed_model_identity
        validate_streamed_model_identity(probe["source_model"], where="joint AURA row")
        object.__setattr__(self, "_fields", tuple(
            (key, json.dumps(value, separators=(",", ":"), allow_nan=False))
            for key, value in snapshot.items()))
        object.__setattr__(self, "_sha256", bytes_sha256hex(encoded))

    def __getitem__(self, key):
        for name, encoded in self._fields:
            if name == key:
                return json.loads(encoded)
        raise KeyError(key)

    def __iter__(self):
        return (name for name, _ in self._fields)

    def __len__(self):
        return len(self._fields)

    def __reduce__(self):
        return dict, (dict(self),)


def prepare_joint_aura_identities(cost_data):
    """Own shared probe identities for one allocator table, without trusting rows.

    Pickle preserves shared object references. Deduplicate by that reference
    only while retaining it here, then discard the temporary index. No global
    cache, digest-only trust, or caller promise of immutability is involved.
    Row/operator/digest/sample checks still run at every consumption boundary.
    """
    prepared = {}
    for per_unit in cost_data.get("costs", {}).values():
        if not isinstance(per_unit, Mapping):
            continue
        for row in per_unit.values():
            if not isinstance(row, dict):
                continue
            probe = row.get("probe_identity")
            if not isinstance(probe, dict) or probe.get("schema") != "prismaquant.joint_aura.probes.v2":
                continue
            key = id(probe)
            if key not in prepared:
                prepared[key] = (probe, _ValidatedProbeIdentity(probe))
            row["probe_identity"] = prepared[key][1]


def validated_identity_memo(cost_data) -> dict:
    """A ``copy.deepcopy`` memo that keeps ``cost_data``'s validated probe
    identities, which are immutable, as themselves in the copy.

    Without it a deep copy turns each one back into a plain dict and every
    row of the copy re-validates its source model (PQ #1256). Pair it with
    :func:`release_joint_aura_identities` before the copy leaves its owner.
    """
    memo = {}
    for per_unit in cost_data.get("costs", {}).values():
        if not isinstance(per_unit, Mapping):
            continue
        for row in per_unit.values():
            if isinstance(row, Mapping):
                probe = row.get("probe_identity")
                if type(probe) is _ValidatedProbeIdentity:
                    memo[id(probe)] = probe
    return memo


def release_joint_aura_identities(cost_data):
    """Give every row an ordinary dict again, built once per validated
    identity from its validated snapshot, so rows that shared an identity
    still share one dict, as they do after unpickling."""
    released = {}
    for per_unit in cost_data.get("costs", {}).values():
        if not isinstance(per_unit, Mapping):
            continue
        for row in per_unit.values():
            if not isinstance(row, dict):
                continue
            probe = row.get("probe_identity")
            if type(probe) is _ValidatedProbeIdentity:
                key = id(probe)
                if key not in released:
                    released[key] = (probe, dict(probe))
                row["probe_identity"] = released[key][1]


def validated_probe_identity(probe) -> Mapping:
    """``probe`` validated and hashed once, for a producer that publishes many rows.

    The result is the type :func:`identity_sha256` and
    :func:`validate_joint_aura_entry` recognize: its digest was computed and
    its source model identity validated once, here, not again for every row
    checked through it. It is not JSON data and compares equal only to its
    own type, so a published row carries the ordinary ``probe`` instead. The
    producer then owns the proof that ``probe`` still hashes to this digest.
    """
    if type(probe) is _ValidatedProbeIdentity:
        return probe
    return _ValidatedProbeIdentity(probe)


def joint_identity_sha256(value) -> str:
    if type(value) is _ValidatedProbeIdentity:
        return value._sha256
    return DIRECT_ASCII_STRICT.sha256(value)


def validate_joint_aura_entry(entry: Mapping) -> bool:
    """Recognize joint claims and fail closed before any scalar cost branch."""
    claims = (entry.get("cost_source") == "joint_aura" or
              entry.get("cost_currency") == JOINT_CURRENCY or
              "joint_operator_identity" in entry or "joint_operator_identity_sha256" in entry)
    if not claims:
        return False
    # A single draw provides no estimate of sampling variance. Refuse it
    # instead of publishing a zero standard error that looks like certainty.
    if (not isinstance(entry.get("probe_identity"), Mapping)
            or type(entry["probe_identity"].get("n_probes")) is not int
            or entry["probe_identity"]["n_probes"] < 2):
        raise ValueError("joint AURA requires at least two probes for sampling uncertainty")
    expected = {"cost_source": "joint_aura", "cost_currency": JOINT_CURRENCY,
                "fisher_application_count": 1, "activation_quantization_included": True,
                "output_mse_measured": False}
    for key, value in expected.items():
        if entry.get(key) != value or type(entry.get(key)) is not type(value):
            raise ValueError(f"joint AURA invalid {key}")
    scope = entry.get("uncertainty_scope")
    if scope is not None and scope not in _ROW_UNCERTAINTY_SCOPES:
        # A row published under any other label claims an uncertainty the
        # probe stderr cannot support. Legacy rows without the field keep
        # their historical conditional reading; present foreign or
        # generalization claims refuse.
        raise ValueError(
            f"joint AURA refuses foreign uncertainty_scope {scope!r}: the published "
            "stderr is probe sampling conditional on fixed calibration, never a "
            "draw-level or generalization error")
    if any(key in entry for key in ("act_dloss", "act_dloss_applied", "aqua_activation_dloss", "activation_pricing_applied", "output_mse")):
        raise ValueError("joint AURA refuses a second activation/Fisher application")
    operator = entry.get("joint_operator_identity")
    if not isinstance(operator, Mapping) or operator.get("schema") != "prismaquant.joint_aura.operator.v2":
        raise ValueError("joint AURA requires v2 operator identity; legacy artifacts require fresh prepare and recompute")
    if joint_identity_sha256(operator) != entry.get("joint_operator_identity_sha256"):
        raise ValueError("joint AURA operator identity digest mismatch")
    probe = entry.get("probe_identity")
    digest = entry.get("probe_identity_sha256")
    if not isinstance(probe, Mapping) or probe.get("schema") != "prismaquant.joint_aura.probes.v2" or joint_identity_sha256(probe) != digest or operator.get("probe_identity_sha256") != digest:
        raise ValueError("joint AURA probe identity mismatch")
    from prismaquant.streamed_model_identity import validate_streamed_model_identity
    try:
        if type(probe) is not _ValidatedProbeIdentity:
            validate_streamed_model_identity(probe["source_model"], where="joint AURA row")
        for field in ("calibration_sha256", "producer_source_sha256"):
            if re.fullmatch(r"[a-f0-9]{64}", probe[field]) is None:
                raise ValueError(f"invalid {field}")
        if type(probe["n_probes"]) is not int or probe["n_probes"] < 1 or type(probe["seed_base"]) is not int:
            raise ValueError("invalid probe indices")
        if probe['distribution'] != 'rademacher' or probe['normalization'] != 'global_kl_fisher':
            raise ValueError('invalid probe distribution/normalization')
        if not math.isfinite(float(probe["temperature"])) or probe["temperature"] <= 0:
            raise ValueError("invalid probe temperature")
        if not isinstance(operator["qname"], str) or not operator["qname"] or not isinstance(operator["format"], str) or not operator["format"]:
            raise ValueError("invalid operator coordinate")
        for field in ("source_weight", "rendered_weight"):
            tensor = operator[field]
            if len(tensor["shape"]) != 2 or any(type(dim) is not int or dim <= 0 for dim in tensor["shape"]):
                raise ValueError("invalid tensor geometry")
            if re.fullmatch(r"[a-f0-9]{64}", tensor["content_sha256"]) is None:
                raise ValueError("invalid tensor content hash")
            if tensor["dtype"] not in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64") or type(tensor["logical_bytes"]) is not int or tensor["logical_bytes"] <= 0:
                raise ValueError("invalid tensor storage identity")
        source_shape = operator["source_weight"]["shape"]
        rendered_shape = operator["rendered_weight"]["shape"]
        if source_shape != rendered_shape:
            raise ValueError("render/source geometry differs")
        activation = operator["activation"]
        if activation["schema"] != "prismaquant.joint_aura.activation.v1" or type(activation["quantizes_input"]) is not bool:
            raise ValueError("invalid activation identity")
        for field in ("activation_max_abs", "input_global_scale"):
            if activation[field] is not None and not math.isfinite(float(activation[field])):
                raise ValueError("nonfinite activation scale")
        arithmetic = operator["arithmetic"]
        from prismaquant.joint_projection_contract import validate_projection_backend_identity
        validate_projection_backend_identity(arithmetic["projection_backend"])
        if arithmetic != probe["arithmetic"] or arithmetic["projection_dtype"] != "torch.float32" or arithmetic["delta_dtype"] != "torch.float32" or arithmetic["aggregation"] != "sum_signed_invocations_then_square":
            raise ValueError("invalid projection arithmetic")
        from .joint_replay_regime import replay_regime_of
        replay_regime_of(arithmetic)
        from .matmul_arithmetic import bf16_reduction_of
        bf16_reduction_of(arithmetic)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(f"joint AURA incomplete identity: {exc}") from exc
    ids = entry.get("probe_ids")
    expected_ids = [int(probe["seed_base"]) + k for k in range(int(probe["n_probes"]))]
    signed, squared = entry.get("signed_per_probe"), entry.get("x2_per_probe")
    if not expected_ids or ids != expected_ids or not isinstance(signed, list) or not isinstance(squared, list) or len(signed) != len(ids) or len(squared) != len(ids):
        raise ValueError("joint AURA probe alignment mismatch")
    if any(not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(b), float(a)**2, rel_tol=1e-12, abs_tol=1e-30) for a, b in zip(signed, squared)):
        raise ValueError("joint AURA signed/squared sample mismatch")
    mean = 0.5 * sum(squared) / len(squared)
    if not math.isclose(float(entry["predicted_dloss"]), mean, rel_tol=1e-12, abs_tol=1e-30):
        raise ValueError("joint AURA predicted loss differs from aligned samples")
    stderr = float(entry.get("predicted_dloss_stderr", float("nan")))
    sample_mean = 2 * mean
    variance = sum((x - sample_mean)**2 for x in squared) / (len(squared) - 1) if len(squared) > 1 else 0.0
    expected_stderr = 0.5 * math.sqrt(variance / len(squared))
    if not math.isfinite(stderr) or stderr < 0 or not math.isclose(stderr, expected_stderr, rel_tol=1e-12, abs_tol=1e-30):
        raise ValueError("joint AURA invalid probe standard error")
    components = entry.get("signed_components_per_probe")
    if not isinstance(components, list) or len(components) != len(signed):
        raise ValueError("joint AURA incomplete signed components")
    for value, total in zip(components, signed):
        if not isinstance(value, Mapping) or set(value) != {"weight", "activation", "mixed", "total"}:
            raise ValueError("joint AURA invalid signed components")
        if not all(math.isfinite(float(x)) for x in value.values()) or value["total"] != total or not math.isclose(value["weight"] + value["activation"] + value["mixed"], total, rel_tol=1e-12, abs_tol=1e-30):
            raise ValueError("joint AURA component/signed sample mismatch")
    sidecar = entry.get("sequence_attribution")
    if sidecar is not None:
        _validate_sequence_attribution(sidecar, entry)
    return True


def _validate_sequence_attribution(sidecar, entry):
    """Rebuild the sidecar from its own parts and refuse any divergence.

    The reader recomputes residual, norms, attribution, leaveout and scope
    through the builder's one code path; a hand-edited field — including a
    foreign uncertainty label, an unproven per-sequence scope, a zero
    standard error that was not recomputed, or a hidden gate denominator —
    stops matching its own recomputation and refuses.
    """
    if not isinstance(sidecar, Mapping):
        raise ValueError("joint AURA sequence attribution must be a mapping")
    if sidecar.get("schema") != SEQUENCE_ATTRIBUTION_SCHEMA:
        raise ValueError("joint AURA sequence attribution schema mismatch")
    geometry = sidecar.get("block_geometry")
    reconciliation = sidecar.get("reconciliation")
    if not isinstance(geometry, Mapping) or not isinstance(reconciliation, Mapping):
        raise ValueError("joint AURA sequence attribution needs geometry and reconciliation")
    gate_relative = reconciliation.get("gate_relative", 1e-3)
    recalculated = sequence_attribution_sidecar(
        blocks=geometry.get("blocks"),
        components_per_probe=sidecar.get("components_per_probe"),
        authoritative_totals=entry["signed_per_probe"],
        gate_relative=gate_relative,
        arithmetic_scope=sidecar.get("arithmetic_scope"),
        sequence_length=geometry.get("sequence_length"),
        selected_tokens_per_row=geometry.get("selected_tokens_per_row"),
        n_sequences=geometry.get("n_sequences"),
        calibration_sha256=geometry.get("calibration_sha256"),
        with_leaveout=("leaveout" in sidecar),
    )
    if recalculated != sidecar:
        raise ValueError(
            "joint AURA sequence attribution differs from its own recomputation")
    binding = geometry.get("calibration_sha256")
    if binding is not None and binding != entry["probe_identity"]["calibration_sha256"]:
        raise ValueError("joint AURA sequence attribution binds a foreign calibration")
    blocks = geometry["blocks"]
    next_sequence = 0
    for block in blocks:
        if block["first_sequence"] != next_sequence:
            raise ValueError("joint AURA sequence attribution must cover the complete calibration draw")
        next_sequence += block["sequences"]
    if next_sequence != geometry["n_sequences"]:
        raise ValueError("joint AURA sequence attribution must cover the complete calibration draw")
    probe = entry["probe_identity"]
    shape = probe.get("calibration_shape")
    if shape is not None:
        if (len(shape) != 2 or geometry["n_sequences"] != shape[0]
                or geometry["sequence_length"] != shape[1]):
            raise ValueError("joint AURA sequence attribution geometry differs from priced calibration")
        selected_tokens = {"all": shape[1], "last": 1, "causal": shape[1] - 1}.get(
            probe.get("token_scope"))
        if geometry["selected_tokens_per_row"] != selected_tokens:
            raise ValueError("joint AURA sequence attribution token scope differs from priced calibration")
    for blocks_per_probe, authority in zip(
            sidecar["components_per_probe"], entry["signed_components_per_probe"]):
        for component in ("weight", "activation", "mixed"):
            reconstructed = math.fsum(block[component] for block in blocks_per_probe)
            norm = math.fsum(abs(block[component]) for block in blocks_per_probe)
            residual = reconstructed - authority[component]
            if (residual != 0.0 if norm == 0.0
                    else abs(residual) > gate_relative * norm):
                raise ValueError(f"joint AURA sequence attribution {component} component does not reconcile")


def squared_signed(total) -> float:
    """The one squaring of a signed joint projection.

    ``x ** 2`` goes through C ``pow`` and ``x * x`` is a correctly rounded
    multiply; on glibc they differ by one ulp for some inputs. The streamed
    accumulator and this row must square identically, because checkpoint
    reload compares the two sample lists exactly (PQ #261, run-02: 9 of 128
    samples of ``model.layers.0.mlp.down_proj@TESSERA_E4M3_K1_R896`` differed).
    """
    total = float(total)
    return total * total


def _attribution_reconciliation_residuals(components, authoritative_totals):
    """One home for the sidecar's reconstruction residual, builder and reader."""
    return [math.fsum(value["total"] for value in probe) - float(total)
            for probe, total in zip(components, authoritative_totals)]


def _attribution_reconciliation_norms(components):
    """The cancellation-aware denominator: fsum of absolute component mass."""
    return [math.fsum(abs(value["weight"]) + abs(value["activation"]) + abs(value["mixed"])
                      for value in probe) for probe in components]


def _attribution_value(components_per_probe, block_index, authoritative_totals):
    """c_i = .5 mean_p a_{p,i} * total_p against the AUTHORITATIVE totals.

    The sum over i reconciles the signed components algebraically: it equals
    the price when the reconstructed totals equal the authoritative ones, and
    otherwise carries the published residual — the row never claims the sum
    is the price.
    """
    n = len(authoritative_totals)
    return 0.5 * math.fsum(
        components_per_probe[p][block_index]["total"] * float(authoritative_totals[p])
        for p in range(n)) / n


def _leaveout_value(components_per_probe, block_index, authoritative_totals):
    """One delete-one price 0.5 mean_p (total_p - a_{p,i})**2; descriptive."""
    n = len(authoritative_totals)
    return 0.5 * math.fsum(
        squared_signed(float(authoritative_totals[p])
                       - components_per_probe[p][block_index]["total"])
        for p in range(n)) / n


def sequence_attribution_sidecar(*, blocks, components_per_probe,
                                 authoritative_totals, gate_relative=1e-3,
                                 arithmetic_scope, sequence_length=None,
                                 selected_tokens_per_row=None,
                                 n_sequences=None, calibration_sha256=None,
                                 with_leaveout=None):
    """Build the #1962 descriptive per-block signed sidecar for one cost row.

    ``blocks`` are whole caller-owned units of the calibration draw (capture
    batches or invocations; per-sequence only when every block is exactly one
    complete sequence). The authoritative whole-draw fields are not touched:
    the sidecar reports its own per-block components, the residual against
    the authoritative totals with the gate it must satisfy and the method
    that states its denominator, the attribution c_i = .5 mean_p a_pi*total_p
    computed against those authoritative totals, and — for genuine equal
    whole single-sequence blocks — the delete-one leaveout prices with their
    jackknife standard error. All of it is conditional on the fixed
    calibration: no generalization or draw-sampling error is claimed.
    """
    if (type(gate_relative) is not int and type(gate_relative) is not float) \
            or not 0.0 < float(gate_relative) < 1.0:
        raise ValueError("sequence attribution gate must be a fraction in (0, 1)")
    gate_relative = float(gate_relative)
    blocks = [dict(block) for block in blocks]
    if not blocks:
        raise ValueError("sequence attribution needs at least one caller-noted block")
    expected = None
    for index, block in enumerate(blocks):
        if (not isinstance(block, Mapping)
                or set(block) != {"block_index", "first_sequence", "sequences"}
                or type(block["block_index"]) is not int
                or block["block_index"] != index
                or type(block["first_sequence"]) is not int or block["first_sequence"] < 0
                or type(block["sequences"]) is not int or block["sequences"] < 1):
            raise ValueError(
                "sequence attribution blocks must be noted once, in order, nonempty")
        if expected is not None and block["first_sequence"] < expected:
            raise ValueError("sequence attribution blocks must not overlap or reorder")
        expected = block["first_sequence"] + block["sequences"]
    noted = sum(block["sequences"] for block in blocks)
    if n_sequences is None:
        if blocks[0]["first_sequence"] != 0:
            raise ValueError(
                "sequence attribution needs n_sequences when blocks do not start the draw")
        n_sequences = noted
    if type(n_sequences) is not int or n_sequences < noted:
        raise ValueError("sequence attribution n_sequences must cover the noted blocks")
    if expected > n_sequences:
        raise ValueError("sequence attribution block extends outside the calibration draw")
    for label, value in (("sequence_length", sequence_length),
                         ("selected_tokens_per_row", selected_tokens_per_row)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"sequence attribution {label} must be a positive integer")
    if (sequence_length is not None and selected_tokens_per_row is not None
            and selected_tokens_per_row > sequence_length):
        raise ValueError("sequence attribution selected tokens exceed the sequence length")
    per_sequence = all(block["sequences"] == 1 for block in blocks)
    if arithmetic_scope is None or not isinstance(arithmetic_scope, str) \
            or not arithmetic_scope:
        raise ValueError("sequence attribution needs the collector's arithmetic scope")
    components = [
        [{key: float(value[key]) for key in ("weight", "activation", "mixed", "total")}
         for value in probe]
        for probe in components_per_probe]
    totals = [float(value) for value in authoritative_totals]
    if len(components) != len(totals) \
            or any(len(probe) != len(blocks) for probe in components):
        raise ValueError("sequence attribution components must cover every probe and block")
    for probe in components:
        for value in probe:
            if not all(math.isfinite(x) for x in value.values()):
                raise ValueError("sequence attribution components must be finite")
            if not math.isclose(value["weight"] + value["activation"] + value["mixed"],
                                value["total"], rel_tol=1e-12, abs_tol=1e-30):
                raise ValueError(
                    "sequence attribution block components must sum to the block total")
    residual = _attribution_reconciliation_residuals(components, totals)
    norms = _attribution_reconciliation_norms(components)
    for value, norm in zip(residual, norms):
        if not math.isfinite(value) or not math.isfinite(norm):
            raise ValueError("sequence attribution reconciliation must be finite")
        if norm == 0.0:
            if value != 0.0:
                raise ValueError(
                    "sequence attribution zero-scale blocks must reconcile exactly")
        elif abs(value) > gate_relative * norm:
            raise ValueError(
                f"sequence attribution residual {value!r} exceeds the reported gate "
                f"{gate_relative!r} on this probe's block scale {norm!r}")
    attribution = [_attribution_value(components, index, totals)
                   for index in range(len(blocks))]
    if calibration_sha256 is not None \
            and (not isinstance(calibration_sha256, str)
                 or re.fullmatch(r"[a-f0-9]{64}", calibration_sha256) is None):
        raise ValueError("sequence attribution calibration binding must be a sha256")
    sidecar = {
        "schema": SEQUENCE_ATTRIBUTION_SCHEMA,
        "scope": "per_sequence" if per_sequence else "capture_batch_block",
        "block_geometry": {
            "sequence_length": (int(sequence_length)
                                if sequence_length is not None else None),
            "selected_tokens_per_row": (int(selected_tokens_per_row)
                                        if selected_tokens_per_row is not None else None),
            "n_sequences": int(n_sequences),
            "blocks": blocks,
            "calibration_sha256": calibration_sha256,
        },
        "arithmetic_scope": arithmetic_scope,
        "components_per_probe": components,
        "reconciliation": {
            "residual_per_probe": residual, "norm_per_probe": norms,
            "gate_relative": gate_relative,
            "gate_method": SEQUENCE_ATTRIBUTION_GATE_METHOD,
        },
        "attribution_per_block": attribution,
    }
    if with_leaveout is True and not (len(blocks) >= 2
                                      and len({block["sequences"] for block in blocks}) == 1
                                      and blocks[0]["first_sequence"] == 0
                                      and noted == n_sequences):
        raise ValueError(
            "sequence leaveout needs at least two equal whole blocks covering the draw")
    if (with_leaveout is True
            or (with_leaveout is None and len(blocks) >= 2
                and len({block["sequences"] for block in blocks}) == 1
                and blocks[0]["first_sequence"] == 0 and noted == n_sequences)):
        # Delete-one over equal whole blocks, descriptive and conditional on
        # the fixed draw: leaveblock_i = Nseq/(Nseq-k) * .5 mean_p(t_p-a_pi)^2
        # for uniform k-sequence blocks (per-sequence is k=1). A one-block or
        # unequal-block scope publishes no standard error rather than a fake
        # one; the exchangeability assumption is stated, never implied.
        k = blocks[0]["sequences"]
        scale = int(n_sequences) / (int(n_sequences) - k)
        count = len(blocks)
        prices = [scale * _leaveout_value(components, index, totals)
                  for index in range(count)]
        mean = math.fsum(prices) / count
        standard_error = math.sqrt(
            (count - 1) / count * math.fsum((price - mean) ** 2 for price in prices))
        if not math.isfinite(standard_error):
            raise ValueError("sequence leaveout standard error must be finite")
        sidecar["leaveout"] = {
            "price_per_block": prices, "jackknife_standard_error": standard_error,
            "delete_one_scale": scale, "sequences_per_block": k,
            "assumption": ("exchangeable equal whole blocks of the fixed "
                           "calibration draw; conditional fixed probes, no "
                           "generalization claim"),
            "uncertainty_scope": SEQUENCE_LEAVEOUT_SCOPE,
        }
    return sidecar


def sequence_cohort_summary(sidecar, *, first_block_index, block_count):
    """Descriptive restricted-price summary over WHOLE noted blocks.

    A cohort can never cut a coarse block. The N/m renormalization is stated
    with the selected-token geometry of the row's token scope so the scaled
    estimate is not read as a different position window. This is a
    conditional fixed-probe diagnostic; it publishes no cost row and no
    generalization claim.
    """
    if not isinstance(sidecar, Mapping) \
            or sidecar.get("schema") != SEQUENCE_ATTRIBUTION_SCHEMA:
        raise ValueError("sequence cohort needs a sequence attribution sidecar")
    geometry = sidecar["block_geometry"]
    blocks = geometry["blocks"]
    if (type(first_block_index) is not int or type(block_count) is not int
            or first_block_index < 0 or block_count < 1
            or first_block_index + block_count > len(blocks)):
        raise ValueError("sequence cohort must select whole noted blocks in range")
    selected = blocks[first_block_index:first_block_index + block_count]
    components = sidecar["components_per_probe"]
    cohort_sequences = sum(block["sequences"] for block in selected)
    restricted = 0.5 * math.fsum(
        squared_signed(math.fsum(value["total"] for value in probe[first_block_index:
                                                               first_block_index + block_count]))
        for probe in components) / len(components)
    scale = geometry["n_sequences"] / cohort_sequences
    return {
        "schema": "prismaquant.joint_aura.sequence_cohort.v1",
        "first_block_index": first_block_index, "block_count": block_count,
        "first_sequence": selected[0]["first_sequence"],
        "sequences": cohort_sequences,
        "restricted_price": restricted,
        "renormalized_scale": scale,
        "renormalized_full_draw_estimate": restricted * scale,
        "selected_tokens_per_row": geometry.get("selected_tokens_per_row"),
        "binding": (f"whole blocks [{first_block_index}, "
                    f"{first_block_index + block_count}) of calibration "
                    f"{geometry.get('calibration_sha256')}"),
    }


def make_joint_aura_entry(*, operator_identity, probe_identity, signed_components,
                          sequence_attribution=None):
    """Publish one complete, aligned cost row, also used by checkpoint replay."""
    signed = [float(value["total"]) for value in signed_components]
    squared = [squared_signed(value) for value in signed]
    n = len(squared)
    if not n:
        raise ValueError("joint AURA needs at least one signed probe")
    mean = sum(squared) / n
    variance = sum((value - mean)**2 for value in squared) / (n - 1) if n > 1 else 0.0
    row = {
        "cost_currency": JOINT_CURRENCY, "cost_source": "joint_aura",
        "fisher_application_count": 1, "activation_quantization_included": True,
        "output_mse_measured": False, "predicted_dloss": 0.5 * mean,
        "predicted_dloss_stderr": 0.5 * math.sqrt(variance / n),
        "signed_per_probe": signed, "signed_components_per_probe": signed_components,
        "x2_per_probe": squared,
        "probe_ids": [probe_identity["seed_base"] + k for k in range(n)],
        "probe_identity": probe_identity,
        "probe_identity_sha256": joint_identity_sha256(probe_identity),
        "joint_operator_identity": operator_identity,
        "joint_operator_identity_sha256": joint_identity_sha256(operator_identity),
        "measurement_status": "research",
        "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration",
    }
    if sequence_attribution is not None:
        row["sequence_attribution"] = sequence_attribution
    validate_joint_aura_entry(row)
    return row


def require_probe_alignment(left: Mapping, right: Mapping, *, where: str, message: str) -> None:
    """Separate usable sample dimensions from recorded probe metadata (D32)."""
    from .cost_currency import probe_identity_seals, probe_identity_walls_differ
    if (joint_identity_sha256(left['probe_ids']) != joint_identity_sha256(right['probe_ids'])
            or probe_identity_walls_differ(left['probe_identity'], right['probe_identity'])):
        raise ValueError(message)
    seal_check('probe identity', probe_identity_seals(left['probe_identity']),
               probe_identity_seals(right['probe_identity']), where=where,
               same=left['probe_identity_sha256'] == right['probe_identity_sha256'],
               refusal=lambda: ValueError(message))


def cost_entry_is_joint_aura(cost_entry: dict) -> bool:
    """Validate any joint claim before interpreting generic cost fields."""
    from .rung_allowability import CANONICAL_CHORD_SOURCE, CANONICAL_SUM_SOURCE, complete_scientific_price
    if (cost_entry.get("cost_source") in (CANONICAL_CHORD_SOURCE, CANONICAL_SUM_SOURCE)
            or "canonical_quality" in cost_entry or "canonical_members" in cost_entry):
        complete_scientific_price(cost_entry)
        return False
    return validate_joint_aura_entry(cost_entry)


class JointCellCoordinateError(ValueError):
    """A joint row found under a key its own operator identity does not name.

    Distinct from the ``ValueError`` ``validate_joint_aura_entry`` raises for a
    malformed row, so a caller can turn the coordinate mismatch into its own
    refusal while leaving malformed-evidence failures exactly as they were.
    """


def joint_row_binds_cell(cost_entry: dict, name: str, fmt: str, *, where: str) -> bool:
    """Whether a joint A-side found at ``name@fmt`` was produced *for* that cell.

    ``cost_entry_is_joint_aura`` validates a row *internally*: its currency, its
    Fisher application count, and that its operator and probe digests match the
    objects they name. Every one of those is true of a row correctly produced
    for one Linear when it is read under another: the row is self-consistent,
    and nothing in it was ever compared with the key it was found under. The
    operator identity carries the coordinate the row was produced for, so a
    coverage decision -- which reads keys -- has to compare the two. Without
    that comparison a row copied, mis-merged or re-keyed onto another cell
    reports an A-side nobody measured, and the cell it landed on keeps a
    weight-only cost that reads as covered.

    Returns False when the entry is not a joint row at all, so a caller can ask
    about every row it holds. Raises ``JointCellCoordinateError`` on a joint row
    whose operator identity does not name this cell: a known coordinate
    mismatch is a refusal, never a row to skip quietly.
    """
    if not cost_entry_is_joint_aura(cost_entry):
        return False
    operator = cost_entry.get("joint_operator_identity")
    produced = (f"{operator.get('qname')!r}@{operator.get('format')!r}"
                if isinstance(operator, Mapping) else "no operator identity")
    if (not isinstance(operator, Mapping) or operator.get("qname") != name
            or operator.get("format") != fmt):
        raise JointCellCoordinateError(
            f"{where}: the joint A-side at {name}@{fmt} was produced for "
            f"{produced}. A joint row carries the activation term of the "
            "operator it measured, so it is only its own cell; a row copied, "
            "mis-merged or re-keyed onto another (unit, format) is a refusal "
            "rather than coverage.")
    return True
