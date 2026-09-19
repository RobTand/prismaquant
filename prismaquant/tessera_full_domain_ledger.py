"""Model-wide ledger: which full-legal-domain rates are priced, and how.

PQ #581.  The completed GLM cost table holds tens of thousands of quality
units, but no interface says, per ``(unit, family)``, which of the family's
complete legal roster is measured, which is predicted, and which still needs
acquiring.  ``tessera_full_domain_acquisition`` proposes bounded next steps
for one quality member; this module is the model-wide bookkeeping around it:
derive the complete legal roster with :mod:`prismaquant.tessera_legal_domain`,
classify every legal rate from the cost rows' own ``cost_source`` spelling,
refuse an incomplete full-domain request by name, and emit the missing
endpoint/audit work as data.

What it does NOT do
-------------------
* No extrapolation, ever.  A predicted row outside its family's measured
  envelope is not a price; it is listed missing with reason
  ``predicted_outside_measured_envelope``.  Unknown prices remain unknown,
  and raw measured anchors are never altered.
* No prices are copied.  The ledger cites ``(unit, family, rate,
  cost_source, currency)`` and never restates the value: two copies of one
  claim drift, and the cost table owns the numbers.
* No currency mixing.  A row whose ``currency`` is not the campaign's is
  refused, not classified: a DP that mixes objectives is meaningless, and a
  ledger that mixed them would license one.
* No new runtime, default, menu, or gate.  Opt-in import; the allocator's
  measured-only expansion and the joint-AURA measured-wire requirement are
  untouched.  Research evidence until the ordinary promotion gates pass.

Row contract
------------
Each cost row is a mapping with ``unit``, ``tessera_family``,
``tessera_body_rate_q256``, ``cost_source`` and ``currency``.  Measured means
an exact measured wire (``tessera_campaign_measured`` -- the spelling
``tessera_joint_aura.load_measured_anchor_input`` consumes -- or
``tessera_campaign_measured_stack_sample`` for Horvitz-Thompson stack
estimates).  Predicted means ``tessera_campaign_interpolated`` strictly
inside the measured envelope, sub-kind ``transfer_law`` when the row carries
its ``transfer_law`` block and ``interpolated`` otherwise.  Anything else in
``cost_source`` is refused: an unknown spelling classifies nothing.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA = "prismaquant.tessera_full_domain_ledger.v1"

#: The cost_source spellings that mean an exact measured wire.  The first is
#: what `tessera_joint_aura.load_measured_anchor_input` consumes; the second
#: is a Horvitz-Thompson stack estimate, measured on the draw rather than a
#: census but still a measurement, never a model prediction.
MEASURED_COST_SOURCES = (
    "tessera_campaign_measured",
    "tessera_campaign_measured_stack_sample",
)

MISSING_ENDPOINT = "missing_domain_endpoint"
MISSING_BOUNDARY = "missing_recipe_boundary"
MISSING_INTERIOR = "missing_interior"
MISSING_EXTRAPOLATED = "predicted_outside_measured_envelope"
MISSING_UNSUPPORTED = "predicted_without_measured_support"


class FullDomainLedgerError(ValueError):
    """An incomplete full-domain request, or a row this ledger cannot read."""


def _currency() -> str:
    from .tessera_campaign import CURRENCY

    return CURRENCY


def _interpolated_cost_source() -> str:
    from .allocator_candidates import TESSERA_INTERPOLATED_COST_SOURCE

    return TESSERA_INTERPOLATED_COST_SOURCE


def _legal_rates(family: str, shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    from .tessera_legal_domain import legal_rate_domain

    return legal_rate_domain(family, shapes).rates


_LEGAL_CACHE: dict[tuple[str, tuple[tuple[int, ...], ...]], tuple[int, ...]] = {}


def _cached_legal_rates(family: str, shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    key = (family, tuple(tuple(shape) for shape in shapes))
    if key not in _LEGAL_CACHE:
        _LEGAL_CACHE[key] = _legal_rates(family, shapes)
    return _LEGAL_CACHE[key]


def _transition_rates(family: str, rates: Sequence[int],
                      shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    from .tessera_legal_domain import resolver_transitions

    return resolver_transitions(family, rates, shapes)


#: Transition rates depend on the family and the shapes only, never on the
#: unit being ledgered, so they are derived once per roster rather than once
#: per cost row owner.  A pure derivation over frozen inputs; the cache is
#: keyed by the exact inputs.
_TRANSITION_CACHE: dict[tuple[str, tuple[int, ...], tuple[tuple[int, ...], ...]], tuple[int, ...]] = {}


def _cached_transition_rates(family: str, rates: Sequence[int],
                             shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    key = (family, tuple(rates), tuple(tuple(shape) for shape in shapes))
    if key not in _TRANSITION_CACHE:
        _TRANSITION_CACHE[key] = _transition_rates(family, rates, shapes)
    return _TRANSITION_CACHE[key]


def _default_shapes() -> tuple[tuple[int, int], ...]:
    from .tessera_legal_domain import GLM53_LINEAR_SHAPES

    return GLM53_LINEAR_SHAPES


def build_full_domain_ledger(
    cost_rows: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str],
    shapes: Sequence[Sequence[int]] | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Classify every legal rate of every ``(unit, family)`` the rows name.

    ``families`` is the explicit roster to ledger -- no model-wide default.
    ``shapes`` go to the legal-domain derivation (default: the GLM-5.3-Flash
    Linear shapes); ``currency`` defaults to the campaign's.  Units are read
    off the rows; a ``(unit, family)`` with no row at all is simply absent,
    never invented.
    """
    names = tuple(families)
    if not names or any(not isinstance(name, str) for name in names):
        raise FullDomainLedgerError("families must be a nonempty roster of family names")
    if len(set(names)) != len(names):
        raise FullDomainLedgerError("families names a family twice")
    resolved_shapes = tuple(tuple(shape) for shape in shapes) if shapes is not None else _default_shapes()
    want_currency = currency if currency is not None else _currency()
    interpolated_source = _interpolated_cost_source()

    by_unit_family: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(cost_rows):
        if not isinstance(row, Mapping):
            raise FullDomainLedgerError(f"cost row {index} is not a mapping")
        try:
            unit = row["unit"]
            family = row["tessera_family"]
            rate = row["tessera_body_rate_q256"]
            source = row["cost_source"]
            row_currency = row["currency"]
        except KeyError as exc:
            raise FullDomainLedgerError(
                f"cost row {index} names no {exc.args[0]!r}") from exc
        if family not in names:
            continue
        if row_currency != want_currency:
            raise FullDomainLedgerError(
                f"cost row {index} ({unit}, {family} R{rate}) carries currency "
                f"{row_currency!r}, not {want_currency!r}; a ledger that mixed "
                "objectives would license a DP that mixes them")
        if type(rate) is not int:
            raise FullDomainLedgerError(f"cost row {index} carries no integer rate")
        if source not in MEASURED_COST_SOURCES and source != interpolated_source:
            raise FullDomainLedgerError(
                f"cost row {index} ({unit}, {family} R{rate}) carries unknown "
                f"cost_source {source!r}; an unknown spelling classifies nothing")
        owner = by_unit_family.setdefault((unit, family), {})
        if rate in owner:
            raise FullDomainLedgerError(
                f"cost row {index} reprices ({unit}, {family} R{rate}); one "
                "price per rate -- a second row is a second claim about the "
                "same price, and the ledger does not pick between them")
        owner[rate] = row

    entries: dict[str, dict[str, Any]] = {}
    for (unit, family), rows in sorted(by_unit_family.items()):
        legal = _cached_legal_rates(family, resolved_shapes)
        legal_set = set(legal)
        for rate in rows:
            if rate not in legal_set:
                raise FullDomainLedgerError(
                    f"({unit}, {family} R{rate}) is priced outside the legal "
                    "domain; the roster is derived, and a price outside it is "
                    "a misbound row, not wider coverage")
        measured = sorted(rate for rate, row in rows.items()
                          if row["cost_source"] in MEASURED_COST_SOURCES)
        envelope = (min(measured), max(measured)) if measured else None
        predicted: dict[int, str] = {}
        missing_reasons: dict[int, str] = {}
        for rate, row in rows.items():
            if row["cost_source"] != interpolated_source:
                continue
            if envelope is None:
                missing_reasons[rate] = MISSING_UNSUPPORTED
            elif not envelope[0] <= rate <= envelope[1]:
                missing_reasons[rate] = MISSING_EXTRAPOLATED
            else:
                predicted[rate] = ("transfer_law" if "transfer_law" in row
                                   else "interpolated")
        priced = set(measured) | set(predicted)
        transitions = set(_cached_transition_rates(family, legal, resolved_shapes))
        missing: dict[int, str] = {}
        for rate in legal:
            if rate in priced:
                continue
            if rate in missing_reasons:
                missing[rate] = missing_reasons[rate]
            elif rate == legal[0] or rate == legal[-1]:
                missing[rate] = MISSING_ENDPOINT
            elif rate in transitions:
                missing[rate] = MISSING_BOUNDARY
            else:
                missing[rate] = MISSING_INTERIOR
        # A predicted row outside the envelope (or with no envelope at all)
        # is not coverage: name it missing too, under its own reason.
        for rate, reason in missing_reasons.items():
            if rate not in priced:
                missing[rate] = reason
        entries[f"{unit}|{family}"] = {
            "unit": unit,
            "family": family,
            "currency": want_currency,
            "legal_q256": list(legal),
            "legal_rate_count": len(legal),
            "measured_q256": measured,
            "measured_envelope_q256": list(envelope) if envelope else None,
            "predicted_q256": {str(rate): predicted[rate] for rate in sorted(predicted)},
            "missing_q256": {str(rate): missing[rate] for rate in sorted(missing)},
            "missing_rate_count": len(missing),
            "complete": not missing,
        }
    return {
        "schema": SCHEMA,
        "currency": want_currency,
        "families": list(names),
        "entries": entries,
        "complete_units": sorted(key for key, entry in entries.items() if entry["complete"]),
        "incomplete_units": sorted(key for key, entry in entries.items() if not entry["complete"]),
    }


def _selected(ledger: Mapping[str, Any], *, units: Sequence[str] | None,
              families: Sequence[str] | None) -> dict[str, Any]:
    entries = ledger["entries"]
    keys = sorted(entries)
    if units is not None:
        wanted = set(units)
        keys = [key for key in keys if entries[key]["unit"] in wanted]
    if families is not None:
        wanted_families = set(families)
        keys = [key for key in keys if entries[key]["family"] in wanted_families]
    return {key: entries[key] for key in keys}


def require_full_domain(ledger: Mapping[str, Any], *, units: Sequence[str] | None = None,
                        families: Sequence[str] | None = None) -> None:
    """Refuse an incomplete full-domain request, naming what is missing.

    The default selection is the whole ledger; ``units``/``families`` narrow
    it.  Unknown prices remain unknown: a missing rate is a refusal, never an
    interpolation.
    """
    if ledger.get("schema") != SCHEMA:
        raise FullDomainLedgerError(f"not a {SCHEMA} ledger")
    incomplete = {key: entry for key, entry in
                  _selected(ledger, units=units, families=families).items()
                  if not entry["complete"]}
    if not incomplete:
        return
    details = []
    for key in sorted(incomplete):
        entry = incomplete[key]
        missing = entry["missing_q256"]
        shown = ", ".join(sorted(missing)[:5])
        tail = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        details.append(f"{key}: {len(missing)} missing ({shown}{tail})")
    raise FullDomainLedgerError(
        "incomplete full-domain coverage: " + "; ".join(details))


def missing_acquisition_work(ledger: Mapping[str, Any], *,
                             units: Sequence[str] | None = None,
                             families: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The missing endpoint/audit work as data, one row per unpriced rate.

    Reasons are ``missing_domain_endpoint`` (a legal endpoint: interpolation
    has no support without it), ``missing_recipe_boundary`` (a resolver
    transition: the wire on either side differs), ``missing_interior``, or
    the predicted-but-unadmitted reasons.  Acquisition order is the caller's;
    this list is sorted for determinism, not priority.
    """
    if ledger.get("schema") != SCHEMA:
        raise FullDomainLedgerError(f"not a {SCHEMA} ledger")
    work = []
    for key in sorted(_selected(ledger, units=units, families=families)):
        entry = ledger["entries"][key]
        for rate, reason in sorted(entry["missing_q256"].items(), key=lambda item: int(item[0])):
            work.append({"unit": entry["unit"], "family": entry["family"],
                         "rate_q256": int(rate), "reason": reason})
    return work
