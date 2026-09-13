"""Audit an existing assignment at a tensor-parallel world size.  Read-only.

    python -m prismaquant.tessera_tp_audit \\
        --layer-config artifacts/layer_config.json \\
        --target-profile glm_packed_research_sm121 --tp 2 \\
        --out artifacts/tp2_audit.json

The question this answers is narrow and worth stating exactly: **for every
Tessera unit this assignment already names, will the pinned runtime's loader
cut it on the axis the target profile says tensor parallelism cuts it on?**

It is not an allocator and it never writes an assignment. A refusal here is a
measured platform fact about the pinned runtime, and the answer to one is to
re-run the allocation with the offending rung excluded by that fact -- never
to edit the assignment by hand, which would put a choice the measurement did
not make into the artifact.

What it audits, and what it does not:

* **Loader axis** -- audited. The contract publishes, per unit and per axis,
  whether this build's loader accepts a shard (``tensor_parallel.units[]
  .loader_axes``, which Tessera validates equal to the ``ROUTE_TP_AXES`` its
  routes gate on). ``TESSERA_E2M1_K2`` refuses the ``row`` axis, which is the
  axis a **column**-parallel Linear cuts, so every column-parallel K2 unit is
  refused at ``--tp 2`` and above.
* **Packed-expert cut kind** -- audited. Expert parallelism cuts the stack and
  leaves each expert's 2-D unit whole, so a packed expert resolves to a
  ``none`` cut whatever the profile's name rules say. The receipt records the
  rule's answer beside the effective one so a disagreement is visible without
  being a refusal.
* **Shard geometry** -- NOT audited, and the receipt says so. The geometry leg
  (``tessera.layout.can_shard`` on the half shape) needs each unit's shape and
  ``layer_config.json`` carries no shapes: it is a qname-to-format recipe.
  Auditing it means running the allocator's own legality gate, which reads the
  model.
* **Attested world size** -- NOT audited, by construction. The audit runs with
  ``require_attested_world=False`` because it asks a geometry-side question
  ("can the loader cut it"), not an attestation one ("has a served receipt
  covered this world size"). ``max_world_size`` is recorded per family so the
  reader can see what the contract does attest.

Exit codes: ``0`` no refusals, ``1`` at least one refusal, ``2`` the audit
could not be run (no contract, unknown profile, unreadable input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

#: Receipt schema. Bump when a consumer-visible field changes meaning.
AUDIT_SCHEMA = "prismaquant.tessera_tp_audit.v1"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_CANNOT_RUN = 2


class TesseraTpAuditError(Exception):
    """The audit cannot be run, so it certifies nothing."""


def _resolve_axes(contract_path: str | None) -> dict[str, Any]:
    """The loader-axis table to audit against, and where it came from.

    Two sources, one reader. ``--contract`` names a contract file directly --
    what a worker container holds when the plugin is not importable -- and
    otherwise the development pin's contract is read through
    ``load_tessera_contract``. There is deliberately no third case: an audit
    with no table would certify an assignment from silence.
    """
    from .tessera_runtime_contract import (
        TesseraContractError, load_tessera_contract,
        published_tensor_parallel_axes, published_tensor_parallel_limits,
        TESSERA_DEV_PIN_ENV,
    )

    if contract_path:
        path = Path(contract_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TesseraTpAuditError(f"--contract {contract_path}: {exc}") from exc
        sha = hashlib.sha256(raw).hexdigest()
        try:
            axes = published_tensor_parallel_axes(str(path), sha)
            world = published_tensor_parallel_limits(str(path), sha)
        except TesseraContractError as exc:
            raise TesseraTpAuditError(str(exc)) from exc
        return {"source": "--contract", "path": str(path), "sha256": sha,
                "commit": None, "loader_axes": dict(axes),
                "max_world_size": dict(world)}

    try:
        contract = load_tessera_contract()
    except TesseraContractError as exc:
        raise TesseraTpAuditError(str(exc)) from exc
    if contract is None:
        raise TesseraTpAuditError(
            "no Tessera runtime contract is pinned, so there is no loader-axis "
            f"table to audit against. Set {TESSERA_DEV_PIN_ENV} to the reviewed "
            "commit, or pass --contract with the contract file to read."
        )
    return {"source": "dev_pin", "path": contract.path,
            "sha256": contract.sha256, "commit": contract.commit,
            "loader_axes": dict(contract.loader_axes),
            "max_world_size": dict(contract.max_world_size)}


def audit_assignment(
    *,
    layer_config: str,
    target_profile: str,
    tp_degree: int,
    contract_path: str | None = None,
) -> dict[str, Any]:
    """Build the audit receipt.  Reads only; returns the payload."""
    from .layer_config import load_assignment
    from .name_projection import is_packed_expert_qname
    from .serving_profiles import load_serving_profile
    from .tessera_formats import TesseraFormatError, parse_tessera_format_name
    from .tessera_menu import (
        PARALLEL_NONE, tessera_tp_axis_legal, tp_cut_axis,
    )

    tp = int(tp_degree)
    if tp < 1:
        raise TesseraTpAuditError(f"--tp must be >= 1, got {tp_degree}")

    try:
        assignment = load_assignment(layer_config)
    except Exception as exc:
        raise TesseraTpAuditError(
            f"--layer-config {layer_config}: {exc}") from exc
    try:
        profile = load_serving_profile(target_profile)
    except FileNotFoundError as exc:
        raise TesseraTpAuditError(
            f"--target-profile {target_profile!r} names no serving profile"
        ) from exc

    table = _resolve_axes(contract_path)
    axes = table["loader_axes"]

    units: list[dict[str, Any]] = []
    refusals = 0
    by_reason: dict[str, int] = {}
    by_family: dict[str, int] = {}
    non_tessera = 0
    packed_units = 0

    for qname in sorted(assignment):
        fmt = str(assignment[qname])
        try:
            parsed = parse_tessera_format_name(fmt)
        except TesseraFormatError as exc:
            raise TesseraTpAuditError(f"{qname}: {exc}") from exc
        if parsed is None:
            non_tessera += 1
            continue
        family, body_rate_q256 = parsed
        by_family[family.name] = by_family.get(family.name, 0) + 1

        packed = is_packed_expert_qname(qname)
        packed_units += int(packed)
        rule_kind = profile.tensor_parallel.kind_for(qname)
        cut_kind = PARALLEL_NONE if packed else rule_kind
        cut_source = ("expert_parallel_construction" if packed
                      else "serving_profile_rule")
        axis = tp_cut_axis(cut_kind) if tp > 1 else None

        verdict, reason = "pass", ""
        if packed and cut_kind != PARALLEL_NONE:
            # Unreachable while the rule above stands; kept because the
            # assertion the issue asks for is the one worth keeping.
            verdict = "refused"
            reason = f"tp_packed_expert_cut:{family.name}:{qname}:{cut_kind}"
        elif axis is not None:
            if family.name not in axes:
                verdict = "refused"
                reason = f"tp_axis_unpublished:{family.name}:{qname}:{axis}"
            else:
                loads, why = tessera_tp_axis_legal(
                    family, tp_degree=tp, parallel_kind=cut_kind,
                    unit=qname, loader_axes=axes)
                if not loads:
                    verdict, reason = "refused", why
        if verdict == "refused":
            refusals += 1
            key = reason.split(":", 1)[0]
            by_reason[key] = by_reason.get(key, 0) + 1

        units.append({
            "unit": qname,
            "format": fmt,
            "family": family.name,
            "body_rate_q256": int(body_rate_q256),
            "packed_expert": packed,
            "cut_kind": cut_kind,
            "cut_kind_source": cut_source,
            "profile_rule_kind": rule_kind,
            "cut_axis": axis,
            "axis_status": (axes.get(family.name, {}).get(axis)
                            if axis is not None else None),
            "max_world_size": table["max_world_size"].get(family.name),
            "verdict": verdict,
            "reason": reason,
        })

    return {
        "schema": AUDIT_SCHEMA,
        "layer_config": str(layer_config),
        "target_profile": profile.id,
        "profile_world_size": int(profile.tensor_parallel.world_size),
        "tp_degree": tp,
        "require_attested_world": False,
        "contract": {k: table[k] for k in ("source", "path", "sha256", "commit")},
        "loader_axes": {family: dict(sorted(status.items()))
                        for family, status in sorted(axes.items())},
        "legs": {
            "loader_axis": "audited",
            "packed_expert_cut_kind": "audited",
            "shard_geometry": (
                "not audited: layer_config.json is a qname-to-format recipe "
                "and carries no shapes, and the geometry leg asks about the "
                "shard a rank holds"),
            "attested_world": (
                "not audited: this audit asks whether the loader can cut the "
                "unit, not whether a served receipt covers this world size; "
                "each unit records the contract's max_world_size instead"),
        },
        "summary": {
            "units_audited": len(units),
            "non_tessera_units": non_tessera,
            "packed_expert_units": packed_units,
            "tessera_units_by_family": dict(sorted(by_family.items())),
            "refusals": refusals,
            "refusals_by_reason": dict(sorted(by_reason.items())),
        },
        "verdict": "refused" if refusals else "pass",
        "units": units,
    }


def _write(receipt: dict[str, Any], out: str) -> None:
    text = json.dumps(receipt, indent=2, sort_keys=False)
    if out == "-":
        sys.stdout.write(text + "\n")
        return
    path = Path(out)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prismaquant.tessera_tp_audit",
        description=(
            "Audit an existing layer_config assignment at a tensor-parallel "
            "world size against the pinned runtime's loader-axis table. "
            "Read-only: it never writes the assignment."),
    )
    parser.add_argument("--layer-config", required=True,
                        help="the assignment to audit (read, never written)")
    parser.add_argument("--target-profile", required=True,
                        help="serving profile whose rules say how TP cuts "
                             "each Linear")
    parser.add_argument("--tp", required=True, type=int,
                        help="world size to audit at")
    parser.add_argument("--contract", default=None,
                        help="contract file to read the loader-axis table "
                             "from; the development pin's contract by default")
    parser.add_argument("--out", required=True,
                        help="receipt path, or - for stdout")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = audit_assignment(
            layer_config=args.layer_config,
            target_profile=args.target_profile,
            tp_degree=args.tp,
            contract_path=args.contract,
        )
    except TesseraTpAuditError as exc:
        print(f"tessera_tp_audit: {exc}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    _write(receipt, args.out)
    summary = receipt["summary"]
    print(
        f"tessera_tp_audit: tp={receipt['tp_degree']} "
        f"{summary['units_audited']} Tessera unit(s), "
        f"{summary['refusals']} refusal(s) -> {receipt['verdict']}",
        file=sys.stderr,
    )
    return EXIT_REFUSED if summary["refusals"] else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
