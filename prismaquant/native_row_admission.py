"""Load a measured runtime table and admit its native rows, outside runtime_provenance to cut a cycle (#4412)."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .digests import file_digest_sha256hex
from .measured_runtime_prices import (MeasuredRuntimeTable, RuntimeContext, RuntimePriceError, RuntimeRankResources,
                                      _json as _price_json, _object, parse_measured_runtime_table)
from .native_panel_kind import panel_kind, resolve_native_receipt_view
from .runtime_provenance import (ArtifactReader, _compare_sources, _equal, _require_served_route_family,
                                 _served_artifact_families, _source_claim, admit_fixed_resources, load_runtime_relation,
                                 routed_owner_rank_resources, routed_slowest_rank)


def admit_native_rows(table, relation):
    """Reuse exact same-panel producer gates before accepting v2 table rows."""
    reader = ArtifactReader(Path(table.source_path).parent)
    bindings = table.native_receipt_bindings
    if not isinstance(bindings, (list, tuple)):
        raise RuntimePriceError("native receipt bindings must be an explicit list")
    by_key = {}
    for item in bindings:
        # A dense or single-device row prices one process, so it has no peer
        # roster and keeps the binding shape it has always had. A ranked row
        # carries every other rank's receipt, and a ranked row without one is
        # refused rather than priced from the one rank that happened to write.
        fields = ("unit", "format", "run_id", "panel", "receipt", "memory_trace")
        if "peer_receipts" in item:
            fields += ("peer_receipts",)
        _object(item, fields, "native receipt binding")
        key = item["unit"], item["format"]
        if key in by_key:
            raise RuntimePriceError("duplicate native row receipt binding")
        by_key[key] = item
    _equal(set(by_key), {row.key for row in table.rows}, "native row receipt coverage")
    cohort_panels, source_framing = [], []
    for row in table.rows:
        binding = by_key[row.key]
        run_id = binding["run_id"]
        if run_id not in relation["runs"] or run_id == relation["full_engine_run_id"]:
            raise RuntimePriceError("native row requires its original native runtime")
        _, panel = reader.json(binding["panel"], "independent native panel")
        cohort_panels.append(panel)
        receipt_path, receipt = reader.json(binding["receipt"], "native receipt")
        receipt = resolve_native_receipt_view(receipt, panel)
        trace_path, _ = reader.bytes(binding["memory_trace"], "native memory trace")
        run = relation["runs"][run_id]
        _equal(panel["runtime"], run["raw"], "original native panel runtime")
        _equal(panel["cost_sha256"], table.cost_sha256, "native panel cost payload")
        _equal(panel["source_sha256"], table.context.source_sha256, "native source model")
        _equal(panel["calibration_sha256"], table.context.calibration_sha256, "native calibration")
        # The priced world is a coordinate of the row, not a label: member
        # shapes, per-rank bytes and the timing that includes the collectives
        # all belong to the world the panel was measured in.
        kind = panel_kind(panel)
        _equal(kind.world_size(panel), table.context.tensor_parallel, "native panel world size")
        if table.context.batch_size != 1:
            raise RuntimePriceError("native panel admission currently requires batch size one")
        expected_binding = kind.row_binding(panel)
        consume = kind.consume
        wire_records = kind.wire_records(panel)
        _equal(run["raw"]["schema"], kind.runtime_schema, "native runtime scope")
        if kind.routed:
            _equal(panel["serving_config_sha256"], relation["configuration_sha256"], "native panel configuration")
            # A world above one is priced from every rank's own receipt, so the
            # binding roster is read as a roster: each peer entry's receipt is
            # rehashed, must carry the same frozen panel, and must declare the
            # rank the roster places it at.
            roster = [(receipt["resources"]["rank"], receipt_path, receipt, trace_path,
                       binding["receipt"]["sha256"])]
            for peer in binding.get("peer_receipts", ()):
                _object(peer, ("rank", "receipt", "memory_trace"), "native peer receipt binding")
                peer_path, peer_receipt = reader.json(peer["receipt"], "native peer receipt")
                peer_receipt = resolve_native_receipt_view(peer_receipt, panel)
                peer_trace, _ = reader.bytes(peer["memory_trace"], "native peer memory trace")
                _equal(peer_receipt["panel"], panel, "native peer receipt panel")
                _equal(peer_receipt["resources"]["rank"], peer["rank"], "native peer receipt rank")
                roster.append((peer["rank"], peer_path, peer_receipt, peer_trace,
                               peer["receipt"]["sha256"]))
            roster.sort(key=lambda entry: entry[0])
        else:
            roster = [(0, receipt_path, receipt, trace_path, binding["receipt"]["sha256"])]
        cited_paths = {rank: path for rank, path, _receipt, _trace, _sha in roster}
        cited_paths_sha256 = {rank: sha for rank, _path, _receipt, _trace, sha in roster}
        # Legacy callers retain only the original common scalar. Companion
        # maps are optional, but a supplied map must still bind that scalar.
        producer_claim = {"source_sha256": run["common"]["producer_source_tree_sha256"]}
        if "producer_source_profiles" in run:
            producer_claim["source_profiles"] = run["producer_source_profiles"]
        producer_profiles = _source_claim(producer_claim, "source_sha256", "source_profiles", "native producer")
        wire_framing = []
        for record in wire_records:
            profiles = _source_claim(record["identity"], "encoder_source_sha256", "source_profiles", "wire producer")
            wire_framing.append(_compare_sources(producer_profiles, profiles,
                                                "original wire producer source-tree seal"))
        # This derived status is separate from the retained wire/panel/runtime
        # identities; old wire records remain explicitly legacy-framed.
        source_framing.append({"unit": row.unit, "format": row.fmt, "comparisons": wire_framing})
        observations = {}
        for rank, rank_path, _rank_receipt, rank_trace, rank_sha256 in roster:
            try:
                observations[rank] = consume(rank_path, expected_sha256=rank_sha256,
                                             expected_panel=panel, memory_trace_path=rank_trace)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimePriceError(f"native producer admission refused: {exc}") from exc
        observation = observations[roster[0][0]]
        _equal(observation["unit"], row.unit, "native row unit")
        _equal(observation["format"], row.fmt, "native row format")
        _equal(expected_binding, row.binding.as_dict(), "native row operator binding")
        ranked = isinstance(row.resources, RuntimeRankResources)
        if ranked:
            # The row's per-rank vector is re-derived here from every rank's own
            # receipt, not re-read from the row: the emitter used this same
            # function to fill it, so what this compares is two emissions of one
            # implementation rather than a number that travelled and was trusted.
            _equal(row.resources.as_dict(), routed_owner_rank_resources(
                panel, [(rank, rank_receipt, observations[rank]) for rank, _path, rank_receipt, _trace, _sha in roster],
                world_size=kind.world_size(panel), where=f"native row {row.unit}"), "native row per-rank resources")
        else:
            _equal(observation["serialized_unit_bytes"], row.resources.serialized_bytes, "native serialized bytes")
            _equal(observation["resident_bytes"], row.resources.resident_bytes, "native resident bytes")
        scratch, activation = [], []
        for phase, measurement in (("prefill", row.prefill), ("decode", row.decode)):
            if measurement is None:
                raise RuntimePriceError("native v2 row lacks a complete measured phase")
            # The row cites the receipt its own samples came from: at a world
            # above one that is the slowest rank's, because that is the median
            # the row priced.
            cited = roster[0][0] if not ranked else routed_slowest_rank(
                row.resources.as_dict(), phase)
            actual = observations[cited]["phases"][phase]
            if actual["peak_scratch_bytes"] is None:
                raise RuntimePriceError("native row has an incomplete resource ledger")
            expected_path, _ = reader.bytes({"path": measurement.receipt_path, "sha256": measurement.receipt_sha256}, "table native phase receipt")
            _equal(expected_path.resolve(), cited_paths[cited].resolve(), "native phase receipt path")
            _equal(measurement.receipt_sha256, cited_paths_sha256[cited], "native phase receipt bytes")
            for key in ("method", "samples_ms", "warmup_iterations"):
                _equal(actual["measurement"][key], measurement.as_dict()[key], "native phase samples")
            _equal(panel["phases"][phase]["m"], table.context.prompt_tokens if phase == "prefill" else 1, "native phase token scope")
            scratch.append(actual["peak_scratch_bytes"]); activation.append(actual["input_bytes"])
            if not ranked and row.resources.output_bytes is not None:
                # A row that names its returned output names the observation's,
                # phase by phase; the escape check reads it as a witness.
                _equal(row.resources.output_bytes.get(phase), actual["output_bytes"],
                       f"native {phase} returned output bytes")
        if not ranked:
            _equal(row.resources.peak_scratch_bytes, max(scratch), "native maximum phase scratch")
            _equal(row.resources.activation_bytes, max(activation), "native maximum phase input residency")
    if getattr(table.context, "native_cohort", None) is not None:
        from .native_runtime_cohort import bind_cohort
        _equal(bind_cohort(cohort_panels), table.context.native_cohort,
               "actual native operator contexts and shared runtime cohort")
    # D39 leg (b) residual (#570): byte coverage cannot see a route class that
    # loads no library, so the served artifact's manifest must carry every
    # priced route's family. Read-only -- this adds refusals, never admission.
    served = _served_artifact_families(relation)
    for row in table.rows:
        _require_served_route_family(row, served)
    relation["wire_source_framing"] = source_framing


def admit_runtime_provenance(table):
    """Admit native rows and return the fixed-resource refusal, if any.

    Native admission failures raise because the table cannot price its rows.
    Fixed-resource failures return their reason for budget consumers to enforce.
    """
    try:
        relation = load_runtime_relation(table.runtime_provenance, context=table.context,
                                         root=Path(table.source_path).parent)
        admit_native_rows(table, relation)
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimePriceError(f"runtime producer evidence is missing or malformed: {exc}") from exc
    try:
        admit_fixed_resources(table, relation)
    except (KeyError, TypeError, IndexError) as exc:
        return f"runtime producer evidence is missing or malformed: {exc}"
    except RuntimePriceError as exc:
        return str(exc)
    return None


def load_measured_runtime_table(path: str | Path, *, expected_context: RuntimeContext,
                                expected_cost_sha256: str, now: datetime | None = None) -> MeasuredRuntimeTable:
    table = parse_measured_runtime_table(_price_json(path), expected_context=expected_context,
                                        expected_cost_sha256=expected_cost_sha256, now=now, source_path=str(path))
    receipts = {(table.fixed_resources_receipt_path, table.fixed_resources_receipt_sha256)}
    for row in table.rows:
        for measurement in (row.prefill, row.decode):
            if measurement:
                receipts.add((measurement.receipt_path, measurement.receipt_sha256))
    for receipt, expected in sorted(receipts):
        receipt_path = Path(receipt) if Path(receipt).is_absolute() else Path(path).parent / receipt
        try:
            with receipt_path.open("rb") as stream:
                actual = file_digest_sha256hex(stream)
        except OSError as exc:
            raise RuntimePriceError(f"cannot read measurement receipt {receipt_path}: {exc}") from exc
        if actual != expected:
            raise RuntimePriceError(f"measurement receipt SHA-256 mismatch: {receipt_path}")
    if table.runtime_provenance is not None:
        refusal = admit_runtime_provenance(table)
        table = replace(table, native_rows_admitted=True, fixed_resources_admitted=refusal is None,
                        fixed_resources_refusal=refusal)
    return table
