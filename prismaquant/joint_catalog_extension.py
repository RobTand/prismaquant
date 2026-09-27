"""An additive candidate catalog over an unchanged BF16 adjoint capture.

The capture is never rewritten. A new record binds both the new candidate
catalog and this independently checked bridge back to the original capture.
No render is created, qualified, relabelled or priced here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import stat as stat_module
from pathlib import Path

from .cost_stage_checkpoint import canonical_json_sha256, publish_new_bytes
from .dev_mode import dev_mode_enabled, dev_warning, seal_check
from .tessera_joint_allocation import _read_bound, _bound_stat_fence
from .schemas import Contract

SCHEMA = "prismaquant.joint_catalog_extension.v2"
#: v1 bound one whole completed receipt; v2 (PQ #993) binds the Stage A run
#: header, which every band and the complete receipt of one run share, so an
#: extension can be created from the first sealed band. v1 documents still
#: verify: their receipt's header must equal the header a slice carries.
SCHEMA_V1 = "prismaquant.joint_catalog_extension.v1"
#: v3 (PQ #1126) is v2 plus ``original_campaign_scope``, written only when the
#: bound run header seals ``campaign_scope: null`` (R13 ran without a
#: forward-recovery capsule under a plan that declares no scope). The scope is
#: derived once, at creation, from the original plan by the one scope builder
#: (``tools.dispatch_tessera_campaign.joint_campaign_scope``, which re-hashes
#: every artifact the plan binds) against the frozen campaign identity file.
#: A header that seals its scope keeps producing v2 bytes: the sealed scope
#: rules, and nothing is embedded over it.
SCHEMA_V3 = "prismaquant.joint_catalog_extension.v3"
DERIVED_SCOPE_SCHEMA = "prismaquant.joint_catalog_extension.derived_scope.v1"
DERIVED_SCOPE_RULE = "sealed_null_derived_from_original_plan"
#: The artifacts a campaign scope names, where a joint plan binds them by
#: path and digest. Consumers compare these stated bindings; only creation
#: re-hashes the bytes.
SCOPE_ARTIFACT_BINDINGS = (("inputs", "census"), ("inputs", "campaign_plan"),
                           ("inputs", "merged_checkpoint"), ("calibration_input",),
                           ("canonical_capture",))
INPUTS = ("original_plan", "original_prepared", "extended_plan", "extended_prepared")
# These select candidate artifacts or their output namespace; every other
# plan field, including the entire execution/derivative policy, stays exact.
CANDIDATE_PLAN_FIELDS = frozenset(("inputs", "output_root", "historical_encoder_reuse", "served_activation_policy",
                                  "stage_b_resource_policy", "execution", "max_gpu_bytes"))
PREPARED_SCIENCE = ("source_model_identity", "source_execution", "calibration_input",
                    "projection_backend", "reader_identity")
QUALIFIED_CELL_FIELDS = ("source_weight", "rendered_weight", "activation",
                         "encoding_identity_sha256", "wire_sha256", "render_file_sha256",
                         "render_origin", "render_comparison")
_VERIFIED_PAIR = {}
_VERIFIED_ENCODER_PROOFS = {}
_SELECTED_CATALOG = {}
ADOPTION_SCHEMA = "prismaquant.joint_catalog_source_adoption.v1"
#: The adopted candidate catalog (the overlay). v1 names one added ``format``
#: and one ``cost``/``reseal_proof``. v2 (PQ #1432) names a sorted list of
#: added ``formats`` and a list of ``sources`` (each a measured ``cost``, its
#: ``anchor_journal`` and a ``reseal_proof`` or null); ``cell_sources`` maps
#: each cell, by position, to its source, so a cell carried from a v1 catalog
#: keeps its exact bytes. ``carried_from`` names the catalogs whose cells were
#: carried. Both schemas stay readable.
CATALOG_SCHEMA_V1 = "prismaquant.t4_adopted_catalog.v1"
CATALOG_SCHEMA_V2 = "prismaquant.t4_adopted_catalog.v2"
CATALOG_SCHEMAS = (CATALOG_SCHEMA_V1, CATALOG_SCHEMA_V2)
#: The one added format of every catalog before PQ #1432 (the R13 A4 build).
R13_ADDED_FORMAT = "TESSERA_E2M1_K2_R896"


_require = Contract(ValueError, "joint catalog extension: ").require


def _same(a, b, message):
    _require(a == b, message + " differs")


#: Files whose stat fence drifted but whose content re-hashed to the recorded
#: digest, by fence message. Read it for the run's report; it never admits
#: anything by itself.
FENCE_REHASHED = {}


def _artifact_fence(path, stat, recorded, content_sha256, message):
    """Admit a catalog artifact by its stat fence, or by its content when only metadata moved.

    The stat fence (inode, bytes, mtime_ns, ctime_ns) is the cheap proof that
    the bytes did not change since the catalog was sealed. It is not the
    identity: ctime moves under metadata-only operations (link/unlink of a
    hard link, chmod, an atime-restoring utime, an NFS delegation recall)
    that no unprivileged tool can undo (PQ #1495: 4,320 of 36,288 overlay
    wires, every one ctime-only, every one byte-identical to its recorded
    blob_sha256). So a stat mismatch is "unproven", not "changed": re-hash
    the file and admit it only if the digest equals the recorded one. A size
    change is a content change and still refuses without hashing, and an
    artifact with no recorded digest keeps the strict fence.
    """
    current = {"inode": stat.st_ino, "bytes": stat.st_size,
               "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    if current == recorded:
        return
    _require(content_sha256 is not None and current["bytes"] == recorded.get("bytes"), message + " differs")
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _same(digest, content_sha256, message + " (content re-hash after stat drift)")
    FENCE_REHASHED[message] = FENCE_REHASHED.get(message, 0) + 1


def _seal(expected, actual, message, *, same=None):
    """A run-gate seal (PQ #1147): certified mode refuses exactly as ``_same``
    or ``_require`` would; dev mode prints both values and continues."""
    return seal_check(message, expected, actual, where="joint catalog extension",
                      same=same, refusal=lambda: ValueError(
                          "joint catalog extension: " + message + (" differs" if same is None else "")))


def _json(bound, label):
    return json.loads(_read_bound(bound, label))


def added_format_recipe(fmt, *, contract=None):
    """The wire recipe an added ``TESSERA_<FAMILY>_R<q256>`` candidate must carry.

    Derived from the pinned Tessera contract (principle 14), never restated
    here: the family's ``formats[]`` row in the packaged
    ``runtime_contract.json`` (read through
    ``tessera_legal_domain.packaged_contract_payload``, the one reader of that
    table) publishes an ``attested_wire`` entry per attested rung. Every entry
    must agree on every field but ``q256``; that shared template, the row's
    ``grid`` and the requested ``q256`` are the recipe. A ``q256`` outside the
    row's ``reader_rate_range_q256``, or off its ``reader_rate_step_q256``
    from the range's low end, refuses. ``contract`` is the parsed payload, for
    a caller that has already read it.
    """
    family, sep, rate = str(fmt).rpartition("_R")
    _require(bool(sep and family and rate.isdigit()), f"added format {fmt!r} is not spelled FAMILY_R<q256>")
    q256 = int(rate)
    if contract is None:
        from .tessera_legal_domain import packaged_contract_payload
        contract = packaged_contract_payload()
    rows = [row for row in contract.get("formats") or () if isinstance(row, dict) and row.get("family") == family]
    _require(len(rows) == 1, f"the pinned Tessera contract publishes {len(rows)} format rows for {family}")
    row = rows[0]
    _require(row.get("kind") == "tessera_wire", f"{family} is not a Tessera wire family")
    pattern = row.get("name_pattern")
    if pattern is not None:
        _same(pattern.format(k=q256), fmt, f"{fmt}: the contract's name pattern")
    span = row.get("reader_rate_range_q256")
    step = row.get("reader_rate_step_q256")
    _require(isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span),
             f"{family} publishes no reader_rate_range_q256")
    _require(type(step) is int and step > 0, f"{family} publishes no reader_rate_step_q256")
    low, high = span
    _require(low <= q256 <= high, f"{fmt}: q256 {q256} is outside the reader range [{low}, {high}]")
    _require((q256 - low) % step == 0, f"{fmt}: q256 {q256} is off the reader step {step} from {low}")
    entries = row.get("attested_wire")
    _require(isinstance(entries, list) and entries and all(isinstance(e, dict) for e in entries),
             f"{family} publishes no attested_wire entries")
    templates = [{key: value for key, value in entry.items() if key != "q256"} for entry in entries]
    _require(all(template == templates[0] for template in templates),
             f"{family}: attested_wire entries disagree on a field other than q256")
    grid = row.get("grid")
    _require(isinstance(grid, str) and templates[0].get("grid", grid) == grid,
             f"{family}: attested_wire grid differs from the format row")
    return {**templates[0], "grid": grid, "q256": q256}


def added_format_recipes(contract=None):
    """A memoized ``fmt -> recipe`` lookup over one read of the pinned contract.

    A catalog pair or overlay names a few formats over many cells; the table
    is read once and each format's recipe derived once.
    """
    if contract is None:
        from .tessera_legal_domain import packaged_contract_payload
        contract = packaged_contract_payload()
    memo = {}

    def recipe(fmt):
        if fmt not in memo:
            memo[fmt] = added_format_recipe(fmt, contract=contract)
        return memo[fmt]
    return recipe


def catalog_sources(catalog):
    """The measured sources a v1 or v2 catalog names, tolerant of partial documents.

    A v1 catalog's one ``cost`` and ``reseal_proof`` form its one source.
    """
    if catalog.get("schema") == CATALOG_SCHEMA_V2:
        return list(catalog.get("sources") or ())
    return [{"cost": catalog.get("cost"), "anchor_journal": catalog.get("anchor_journal"),
             "reseal_proof": catalog.get("reseal_proof")}]


def catalog_view(catalog):
    """Check a v1 or v2 catalog's structure and return what intake reads.

    Returns ``{"formats", "sources", "cell_sources", "carried_from"}``:
    the sorted added formats, the measured sources, each cell's source index
    (by position) and the catalogs whose cells were carried. The cells
    declare the coverage: every cell names its own ``format``, and the
    declared formats are exactly the formats the cells name. No file is read.
    """
    _require(isinstance(catalog, dict), "candidate catalog is not a mapping")
    schema = catalog.get("schema")
    _require(schema in CATALOG_SCHEMAS, f"candidate overlay schema {schema!r} is not a known catalog schema")
    cells = catalog.get("cells")
    _require(isinstance(cells, list) and cells, "overlay cells are missing")
    if schema == CATALOG_SCHEMA_V1:
        _require(isinstance(catalog.get("format"), str), "v1 catalog names no added format")
        formats = (catalog["format"],)
        cell_sources = [0] * len(cells)
    else:
        formats = catalog.get("formats")
        _require(isinstance(formats, list) and formats and all(isinstance(f, str) for f in formats)
                 and formats == sorted(set(formats)), "v2 catalog formats must be a sorted unique list")
        formats = tuple(formats)
        cell_sources = catalog.get("cell_sources")
        _require(isinstance(cell_sources, list) and len(cell_sources) == len(cells),
                 "v2 catalog cell_sources must name one source per cell")
    sources = catalog_sources(catalog)
    _require(bool(sources) and all(isinstance(source, dict) and isinstance(source.get("cost"), dict)
                                   for source in sources), "catalog names no measured cost source")
    _require(all(type(index) is int and 0 <= index < len(sources) for index in cell_sources),
             "catalog cell names no declared source")
    for cell in cells:
        _require(isinstance(cell, dict) and cell.get("format") in formats,
                 "catalog cell format is not a declared added format")
    _same(sorted({cell["format"] for cell in cells}), list(formats), "catalog declared formats")
    carried = catalog.get("carried_from", [])
    _require(isinstance(carried, list) and all(isinstance(b, dict) and set(b) == {"path", "sha256"}
                                               for b in carried), "catalog carried_from must be bindings")
    return {"formats": formats, "sources": sources, "cell_sources": cell_sources, "carried_from": carried}


def catalog_control_bindings(catalog):
    """Every file a catalog binds for intake: original prepared and PWC, each cost, each proof.

    Proofs are deduplicated by digest; a source with no proof names none.
    """
    bindings = [catalog[key] for key in ("old_prepared", "old_pwc") if catalog.get(key) is not None]
    proofs = {}
    for source in catalog_sources(catalog):
        if source.get("cost") is not None:
            bindings.append(source["cost"])
        if source.get("reseal_proof") is not None:
            proofs.setdefault(source["reseal_proof"]["sha256"], source["reseal_proof"])
    return bindings + [proofs[key] for key in sorted(proofs)]


class EncoderAdoptionValidation:
    """One operation's retained proof bytes, fenced on entry and completion.

    Results are borrowed read-only metadata. Repeating a cell does no file
    metadata work; every distinct dependency is rechecked before successful
    completion, so a changed proof invalidates the whole operation.
    """
    def __init__(self):
        self._proofs, self._catalogs, self._fences = {}, {}, {}
        self._active = False

    def __enter__(self):
        _require(not self._active, "adoption validation cannot nest")
        self._active = True
        return self

    def verify(self, adoption, *, fmt):
        _require(self._active, "adoption validation is outside its operation")
        from tools.reseal_campaign_identity import unit_kind, format_family
        _same(adoption.get("schema"), ADOPTION_SCHEMA, "encoder adoption schema")
        reference, candidate = adoption["reference_encoding_identity"], adoption["candidate_encoding_identity"]
        bound = adoption.get("encoder_source_proof")
        # A cell with no proof keys on None, so a stratum's dev-mode stamp is
        # printed once per operation, not once per cell.
        key = (None if bound is None else bound["path"], None if bound is None else bound["sha256"],
               reference.get("encoder_source_sha256"),
               candidate.get("encoder_source_sha256"), reference.get("encoder_fixture_id"),
               candidate.get("encoder_fixture_id"), unit_kind(candidate.get("unit", '')), format_family(fmt))
        if key not in self._proofs:
            result = validated_encoder_adoption(adoption, fmt=fmt)
            self._proofs[key] = result
            for path, digest, fence in result["fences"]:
                self._fences[path, digest] = fence
        return self._proofs[key]

    def catalog(self, bound):
        _require(self._active, "adoption validation is outside its operation")
        key = (bound["path"], bound["sha256"])
        if key not in self._catalogs:
            document = _json(bound, "selected candidate catalog")
            self._catalogs[key] = document
            self._fences[key] = _bound_stat_fence(Path(bound["path"]))
        return self._catalogs[key]

    def __exit__(self, kind, error, traceback):
        _require(self._active, "adoption validation has already ended")
        self._active = False
        if kind is None:
            for (path, digest), fence in self._fences.items():
                _same(_bound_stat_fence(Path(path)), fence, "encoder proof dependency changed during operation")
        return False


def _unproven_adoption(stratum):
    """The result of an adoption no reseal proof covers, admitted in dev mode only."""
    return {"proof": None, "producer_package": None, "dependencies": [], "fences": (),
            "encoder_source_proof_covered": False, "stratum": list(stratum)}


def validated_encoder_adoption(adoption, *, fmt):
    """Authenticate the existing migration's semantics and all proof dependencies.

    This never substitutes an encoder seal. It binds the exact old/new source
    pair and fixture, reuses the owning reseal proof validator, and retains
    inode/size/mtime/ctime fences for every dependency before memoized reuse.
    The producer package is a declaration; Tessera independently hashes that
    complete package before using its historical identity factory.

    ``fmt`` is the added candidate's format; the proof must cover its
    ``(unit kind, format family)`` stratum. Whether a proof covers the
    candidate's encoder-source migration is a seal (PQ #1147, PQ #1432): no
    proof, a proof for another source pair and a proof that does not cover
    the stratum each go through ``seal_check``. Certified mode refuses; dev
    mode prints one ``[DEV-MODE]`` line and admits the cell with
    ``encoder_source_proof_covered: False`` and no producer package. The exact
    source-seal and fixture shapes, the fixture equality with the reference
    cell, and a named proof's bytes and internal consistency stay walls in
    both modes.
    """
    from tools.reseal_campaign_identity import load_bundle, unit_kind, format_family, Refused

    _same(adoption.get("schema"), ADOPTION_SCHEMA, "encoder adoption schema")
    reference, candidate = adoption["reference_encoding_identity"], adoption["candidate_encoding_identity"]
    old, new = reference.get("encoder_source_sha256"), candidate.get("encoder_source_sha256")
    for value in (old, new):
        _require(isinstance(value, str) and len(value) == 64
                 and all(c in '0123456789abcdef' for c in value), "encoder adoption needs exact source seals")
    fixture = reference.get("encoder_fixture_id")
    _require(isinstance(fixture, str) and len(fixture) == 64
             and all(c in '0123456789abcdef' for c in fixture), "encoder adoption needs an exact fixture identity")
    _same(fixture, candidate.get("encoder_fixture_id"), "adopted encoder fixture")
    proof = adoption.get("encoder_source_proof")
    stratum = (unit_kind(candidate["unit"]), format_family(fmt))
    migration = f"{stratum[0]}:{stratum[1]} encoder source {old} -> {new}"
    if proof is None:
        _seal("a reseal proof covering " + migration, None,
              "added candidate names no encoder source proof for " + migration, same=False)
        return _unproven_adoption(stratum)
    key = (proof["path"], proof["sha256"], _bound_stat_fence(Path(proof["path"])), old, new, fixture, stratum)
    cached = _VERIFIED_ENCODER_PROOFS.get(key)
    if cached is not None and cached["fences"] == tuple(
            (b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in cached["dependencies"]):
        return copy.deepcopy(cached)
    document = _json(proof, "encoder source proof")
    if not (_seal(old, document.get("pins", {}).get("old", {}).get("encoder_source_sha256"),
                  "encoder proof old source")
            and _seal(new, document.get("pins", {}).get("new", {}).get("encoder_source_sha256"),
                      "encoder proof new source")):
        return _unproven_adoption(stratum)
    _require(document.get("arms"), "encoder proof has no measured comparison arms")
    try:
        checked = load_bundle(proof["path"], document["pins"])
    except Refused as error:
        raise ValueError("joint catalog extension: encoder proof refused: " + str(error)) from error
    covered = stratum in checked["covered"]
    if not _seal(list(stratum), sorted(list(s) for s in checked["covered"]),
                 "encoder proof does not cover the added candidate stratum", same=covered):
        return _unproven_adoption(stratum)
    fixture_record = document.get("fixture_id", {})
    fixture_bound = {"path": fixture_record.get("result"), "sha256": fixture_record.get("result_sha256")}
    observed = _json(fixture_bound, "encoder fixture proof")
    _require(observed.get("kind") == "fixture_id" and observed.get("ok") is True
             and observed.get("fixture_id_equal") is True, "encoder fixture proof did not pass")
    _same(set((observed.get("encoder_source_sha256") or {}).values()), {old, new}, "fixture proof source pair")
    _same(set((observed.get("encoder_fixture_ids") or {}).values()), {fixture}, "fixture proof identity")
    _same(set((fixture_record.get("ids") or {}).values()), {fixture}, "bundled fixture identity")
    dependencies = [dict(proof), fixture_bound] + [
        {"path": arm["result"], "sha256": arm["result_sha256"]} for arm in document["arms"]]
    for bound in dependencies:
        _read_bound(bound, "encoder proof dependency")
    source = document.get("source_checks", {}).get("encoder", {})
    _same(source.get("sha256"), new, "declared candidate producer source")
    _require(isinstance(source.get("tree"), str) and Path(source["tree"]).is_absolute(),
             "encoder proof has no bound candidate producer package")
    result = {"proof": dict(proof), "producer_package": {"path": source["tree"], "sha256": new},
              "dependencies": dependencies,
              "fences": tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in dependencies),
              "encoder_source_proof_covered": True, "stratum": list(stratum)}
    _VERIFIED_ENCODER_PROOFS.clear()
    _VERIFIED_ENCODER_PROOFS[key] = result
    return copy.deepcopy(result)


def require_selected_catalog_cell(data, name, fmt, *, validation=None):
    """Rebind an added selected cell to its exact catalog row and current files."""
    bound = data.inputs.get("candidate_overlay")
    _same(data.payload.get("provenance", {}).get("candidate_overlay"), bound, "selected overlay provenance")
    _require(isinstance(bound, dict), "selected candidate has no explicit catalog overlay")
    if validation is not None:
        catalog = validation.catalog(bound)
        key = (bound["path"], bound["sha256"], id(catalog))
    else:
        key = (bound["path"], bound["sha256"], _bound_stat_fence(Path(bound["path"])))
        catalog = None
    if key not in _SELECTED_CATALOG:
        catalog = catalog if catalog is not None else _json(bound, "selected candidate catalog")
        _require(catalog.get("schema") in CATALOG_SCHEMAS, "selected candidate catalog schema differs")
        rows = {(row["qname"], row["format"]): row for row in catalog["cells"]}
        _same(len(rows), len(catalog["cells"]), "selected catalog unique cells")
        _SELECTED_CATALOG.clear()
        _SELECTED_CATALOG[key] = rows
    row = _SELECTED_CATALOG[key].get((name, fmt))
    _require(row is not None, "selected cell absent from bound overlay")
    cell = data.cells[name, fmt]
    for field in ("record", "wire", "render", "catalog_source_adoption"):
        _same(cell.get(field), row.get(field), "selected catalog " + field)
    adoption = row["catalog_source_adoption"]
    proof = (validation.verify(adoption, fmt=fmt) if validation is not None
             else validated_encoder_adoption(adoption, fmt=fmt))
    _same(adoption["candidate_encoding_identity"], cell["record"]["identity"], "selected candidate identity")
    reference_pair = tuple(adoption["reference_pair"])
    _require(reference_pair in data.cells and reference_pair[0] == name, "selected adoption reference absent")
    reference = data.cells[reference_pair]["record"]["identity"]
    for field in ("unit", "source", "projection", "calibration", "encoder_fixture_id"):
        _same(reference.get(field), adoption["reference_encoding_identity"].get(field), "selected reference " + field)
        _same(reference.get(field), cell["record"]["identity"].get(field), "selected candidate " + field)
    for field in ("wire", "render"):
        path = Path(row[field])
        _require(path.is_absolute() and not path.is_symlink() and path.is_file(), "selected artifact is not regular")
        _artifact_fence(path, path.stat(), row[field + "_stat"],
                        cell["record"]["blob_sha256"] if field == "wire" else None,
                        "selected current " + field + " fence")
    return {"qualification_activation": copy.deepcopy(row["activation"]),
            "adoption": copy.deepcopy(adoption), "wire_root": str(Path(row["wire"]).resolve().parent),
            "catalog": dict(bound), **proof}


def extended_roster(formats, added):
    """One unit's candidate roster with the ``added`` formats inserted before the terminal BF16.

    Every sealed prepared roster is its sorted formats with BF16 appended.
    The added formats a unit does not already offer go, sorted, between the
    base formats (which keep their order) and that BF16:
    ``(*base[:-1], *sorted(added), "BF16")``. ``added`` is one format or an
    iterable of them; with the single R13 format this is the order the T4
    overlay was assembled in. The loader, the overlay assembler and the pair
    check all use this one order, because Stage B compares the prepared
    roster with the loaded roster in order (RobTand/prismaquant#990).
    """
    formats = tuple(formats)
    added = (added,) if isinstance(added, str) else tuple(added)
    _require(bool(formats) and formats[-1] == "BF16", "base candidate roster must retain terminal BF16")
    _require(len(set(added)) == len(added) and "BF16" not in added, "added candidates must be distinct Tessera formats")
    _require(not set(added) & set(formats), "added candidate is already in the base roster")
    return (*formats[:-1], *sorted(added), "BF16")


def _pairs(prepared):
    formats = prepared.get("formats_by_qname")
    _require(isinstance(formats, dict) and formats, "prepared candidate roster is missing")
    pairs = set()
    for name, values in formats.items():
        _require(isinstance(name, str) and isinstance(values, (list, dict)) and values,
                 "malformed prepared candidate roster")
        for fmt in values:
            _require(isinstance(fmt, str), "candidate format is not a string")
            if fmt != "BF16":
                pairs.add((name, fmt))
    return pairs


def verify_catalog_pair(inputs):
    """Check a proposed overlay without issuing any capture-reuse authority."""
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_aura import PREPARED_SCHEMA, RENDER_COMPARISON_BY_ORIGIN

    _require(isinstance(inputs, dict) and set(inputs) == set(INPUTS), "four exact input bindings required")
    documents = {name: _json(inputs[name], name) for name in INPUTS}
    old_plan, new_plan = documents["original_plan"], documents["extended_plan"]
    old, new = documents["original_prepared"], documents["extended_prepared"]
    bindings = list(inputs.values()) + [old["production_cache"], new["production_cache"]]
    # The catalog declares the added coverage (PQ #1432): every added cell of
    # the extended PWC is exactly one of its cells, whatever its format.
    overlay = (new_plan.get("inputs") or {}).get("candidate_overlay")
    _require(isinstance(overlay, dict) and set(overlay) == {"path", "sha256"},
             "extended plan binds no candidate overlay catalog")
    bindings.append(overlay)
    from .joint_stageb_resources import require_plan_resources
    resources = require_plan_resources(old_plan, new_plan, inputs["original_plan"], inputs["original_prepared"])
    _same(new.get("stage_b_resource_policy"), new_plan.get("stage_b_resource_policy"), "extended resource policy")
    if resources is not None:
        bindings += [new_plan["stage_b_resource_policy"], *resources["inputs"].values()]
    _same(old_plan.get("served_activation_policy"), old.get("served_activation_policy"), "original served policy")
    _same(new_plan.get("served_activation_policy"), new.get("served_activation_policy"), "extended served policy")
    if new_plan.get("served_activation_policy") is not None:
        from .joint_served_activation import verify_policy
        _require(old_plan.get("served_activation_policy") is None, "cannot replace an original served policy")
        policy = verify_policy(new_plan["served_activation_policy"], original_prepared=inputs["original_prepared"])
        _same(policy["calibration_input"], old["calibration_input"], "served-group calibration")
        bindings += [new_plan["served_activation_policy"], policy["census"]]
    fences = tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in bindings)
    if fences in _VERIFIED_PAIR:
        cached = _VERIFIED_PAIR[fences]
        if cached["proof_fences"] == tuple((path, digest, _bound_stat_fence(Path(path)))
                                            for path, digest, _fence in cached["proof_fences"]):
            return dict(cached["evidence"])
    _same({k: v for k, v in old_plan.items() if k not in CANDIDATE_PLAN_FIELDS},
          {k: v for k, v in new_plan.items() if k not in CANDIDATE_PLAN_FIELDS},
          "BF16 source/calibration/derivative/execution plan")
    for name, prepared, plan in (("original", old, old_plan), ("extended", new, new_plan)):
        _same(prepared.get("schema"), PREPARED_SCHEMA, name + " prepared schema")
        _same(prepared.get("status"), "complete", name + " preparation completion")
        _same(prepared.get("plan_sha256"), inputs[name + "_plan"]["sha256"], name + " prepared plan")
        _same(prepared.get("calibration_input", {}).get("artifact_sha256"),
              plan.get("calibration_input", {}).get("sha256"), name + " calibration artifact")
    for key in PREPARED_SCIENCE:
        _require(key in old and key in new, "missing prepared scientific identity " + key)
        _same(old[key], new[key], "prepared " + key)
    _same(set(old["formats_by_qname"]), set(new["formats_by_qname"]), "source qname roster")
    old_pairs, new_pairs = _pairs(old), _pairs(new)
    _require(old_pairs < new_pairs, "catalog must strictly add candidates without removing any")
    catalog = _json(overlay, "candidate overlay catalog")
    catalog_view(catalog)
    declared = [(cell.get("qname"), cell.get("format")) for cell in catalog["cells"]]
    _same(len(set(declared)), len(declared), "unique catalog-declared added cells")
    _same(sorted(new_pairs - old_pairs), sorted(declared), "catalog-declared added cells")
    for name, formats in old["formats_by_qname"].items():
        extended = new["formats_by_qname"][name]
        _require(set(formats) <= set(extended), "original candidate removed for " + name)
        added = [fmt for fmt in extended if fmt not in formats]
        expected = tuple(formats) if not added else extended_roster(formats, added)
        _same(tuple(extended), expected,
              "extended candidate order (added formats, sorted, before terminal BF16) for " + name)

    caches = {}
    for name, prepared, plan, pairs in (("original", old, old_plan, old_pairs),
                                        ("extended", new, new_plan, new_pairs)):
        cache = pickle.loads(_read_bound(prepared["production_cache"], name + " production cache"))
        _require(isinstance(cache, ProductionWeightCache), name + " cache owner is not ProductionWeightCache")
        metadata = cache.metadata or {}
        _same(metadata.get("inputs"), plan["inputs"], name + " cache input catalog")
        _same(metadata.get("schema"), PREPARED_SCHEMA, name + " cache schema")
        _same(set(cache.weights), pairs, name + " render roster")
        _same(set(metadata.get("verified_cells", {})), pairs, name + " verified cell roster")
        _same(prepared.get("measured_cells"), len(pairs), name + " measured cell count")
        for field in ("source_model_identity", "reader_identity", "projection_backend"):
            _same(metadata.get(field), prepared[field], name + " cache " + field)
        caches[name] = cache
    previous, extended = caches["original"], caches["extended"]
    if resources is not None:
        _same({pair: str(Path(extended._path_for_value(path)).absolute()) for pair, path in extended.weights.items()},
              {tuple(row["member"]): row["path"] for row in resources["candidate_files"]},
              "resource-policy actual candidate file roster")
    _same(previous.levers, extended.levers, "original render levers")
    _same(previous.activation_max_abs, extended.activation_max_abs, "original activation maxima mapping")
    old_verified, new_verified = previous.metadata["verified_cells"], extended.metadata["verified_cells"]
    for pair in sorted(old_pairs):
        _same(previous.weights[pair], extended.weights[pair], "original render path " + repr(pair))
        _same(old_verified[pair], new_verified[pair], "original qualified identity " + repr(pair))
    source_by_name = {}
    for (name, _fmt), cell in old_verified.items():
        source = cell["source_weight"]
        if name in source_by_name:
            _same(source_by_name[name], source, "original source identity " + name)
        source_by_name[name] = source
    additions = sorted(new_pairs - old_pairs)
    proof_fences = set()
    recipe = added_format_recipes()
    with EncoderAdoptionValidation() as proof_checks:
        for pair in additions:
            cell = new_verified[pair]
            _require(isinstance(cell, dict) and all(k in cell for k in QUALIFIED_CELL_FIELDS),
                     "added candidate lacks actual render qualification " + repr(pair))
            _same(cell["render_origin"], "encoded", "historical candidate render origin")
            _same(cell["render_comparison"], RENDER_COMPARISON_BY_ORIGIN["encoded"],
                  "actual render/wire comparison")
            _same(cell["source_weight"], source_by_name.get(pair[0]), "added candidate source " + repr(pair))
            for key in ("encoding_identity_sha256", "wire_sha256", "render_file_sha256"):
                value = cell[key]
                _require(isinstance(value, str) and len(value) == 64
                         and all(c in "0123456789abcdef" for c in value), "added candidate lacks " + key)
            _require(isinstance(extended.weights[pair], str), "added render must name its existing file")
            rendered = cell["rendered_weight"]
            digest = rendered.get("content_sha256") if isinstance(rendered, dict) else None
            _require(isinstance(digest, str) and len(digest) == 64
                     and all(c in "0123456789abcdef" for c in digest),
                     "added candidate lacks measured rendered tensor identity")
            _same(rendered.get("shape"), cell["source_weight"].get("shape"), "added render shape")
            adoption = cell.get("catalog_source_adoption")
            _require(isinstance(adoption, dict) and adoption.get("schema") == ADOPTION_SCHEMA,
                     "added candidate lacks explicit source/H adoption evidence")
            reference_pair = tuple(adoption.get("reference_pair", []))
            _require(reference_pair in old_pairs and reference_pair[0] == pair[0],
                     "adoption reference is not an original candidate of the same unit")
            reference = adoption.get("reference_encoding_identity")
            candidate = adoption.get("candidate_encoding_identity")
            _require(isinstance(reference, dict) and isinstance(candidate, dict), "adoption identities are missing")
            _same(canonical_json_sha256(reference, where="adopted reference encoding"),
                  old_verified[reference_pair]["encoding_identity_sha256"], "authenticated original encoding identity")
            _same(canonical_json_sha256(candidate, where="adopted candidate encoding"),
                  cell["encoding_identity_sha256"], "authenticated added encoding identity")
            for field in ("unit", "source", "projection", "calibration", "encoder_fixture_id"):
                _require(field in reference and field in candidate, "adoption identity lacks " + field)
                _same(reference[field], candidate[field], "adopted source/H " + field)
            _same(candidate["unit"], pair[0], "adopted unit")
            _same(candidate.get("recipe"), recipe(pair[1]), "added candidate recipe for " + pair[1])
            # Hash the existing source-migration proof once; do not manufacture a
            # new proof or silently normalize another encoding field here.
            verified_proof = proof_checks.verify(adoption, fmt=pair[1])
            proof_fences.update(verified_proof["fences"])
    science = {"plan": {k: v for k, v in old_plan.items() if k not in CANDIDATE_PLAN_FIELDS},
               "original_execution": old_plan["execution"], "original_max_gpu_bytes": old_plan.get("max_gpu_bytes"),
               "prepared": {key: old[key] for key in PREPARED_SCIENCE},
               "qnames": sorted(old["formats_by_qname"])}
    evidence = {"original_cells": len(old_pairs), "extended_cells": len(new_pairs),
            "added_cells": len(additions),
            "added_pairs_sha256": canonical_json_sha256(additions, where="added candidate pairs"),
            "scientific_identity_sha256": canonical_json_sha256(science, where="adjoint science"),
            "source_qnames": len(old["formats_by_qname"])}
    _same(fences, tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"])))
                        for b in bindings), "catalog metadata changed during verification")
    # One validated pair per process. Rebinding 45 records does not deserialize
    # both large PWCs 45 times; every dependency retains the existing bound
    # reader's inode/size/mtime/ctime fence, and drift revalidates from bytes.
    _VERIFIED_PAIR.clear()
    _VERIFIED_PAIR[fences] = {"evidence": evidence, "proof_fences": tuple(sorted(proof_fences))}
    return dict(evidence)


def _run_header(capture):
    """The Stage A run header of a completed receipt or a sealed band."""
    from .joint_adjoint_slices import AdjointSliceRefused, stage_a_run_header
    try:
        return stage_a_run_header(capture)
    except AdjointSliceRefused as exc:
        raise ValueError("joint catalog extension: actual completed Stage A capture "
                         f"or sealed checkpoint band required ({exc})") from exc


def _check_capture(header, inputs, original):
    """Check a Stage A run header answers for the original campaign."""
    _require(isinstance(header, dict) and isinstance(header.get("run_identity"), dict),
             "a Stage A run header is required")
    identity = header["run_identity"]
    _same(identity.get("plan_sha256"), inputs["original_plan"]["sha256"], "capture original plan")
    _same(identity.get("prepared_sha256"), inputs["original_prepared"]["sha256"], "capture original prepared")
    calibration = original["calibration_input"]
    _same(identity.get("calibration_sha256"), calibration["calibration_sha256"], "capture calibration")
    _same(identity.get("calibration_shape"), calibration["shape"], "capture calibration shape")
    # Stage A seals one of two roster spellings, and the run header says
    # which. A fresh run hashes one terminating newline per qname. A run
    # resumed from a forward-recovery capsule seals the campaign's canonical
    # spelling, the quantum roster (``resolve_forward_campaign``). Never
    # compare unlike hashes.
    names = sorted(original["formats_by_qname"])
    if (header.get("boundary_storage") or {}).get("forward_recovery") is not None:
        from .joint_layer_quanta import roster_digest
        expected = roster_digest(names)
    else:
        expected = hashlib.sha256("".join(name + "\n" for name in names).encode()).hexdigest()
    _same(identity.get("unit_roster_sha256"), expected, "capture qname roster")
    plan = _json(inputs["original_plan"], "original plan")
    for key in ("n_probes", "seed_base"):
        _same(identity.get(key), plan["execution"][key], "capture " + key)
    return plan


def _plan_scope_bindings(plan):
    """The five artifacts a scope names, exactly as the original plan states them."""
    bound = {}
    for keys in SCOPE_ARTIFACT_BINDINGS:
        value = plan
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        _require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                 "original plan binds no " + ".".join(keys))
        bound[keys[-1]] = dict(value)
    return bound


def derive_original_campaign_scope(original_plan, campaign_identity):
    """The original campaign's scope, derived once from the original plan (PQ #1126).

    The one scope builder, ``joint_campaign_scope``, reads the plan's bound
    census and campaign plan and re-hashes every artifact the scope names,
    the merged checkpoint included; ``verify_joint_campaign_scope`` then holds
    it to the frozen campaign identity, field for field, and to the
    ``complete_campaign`` kind. The stamp is the one the dispatcher puts on
    every campaign-scoped submission: the scope plus the identity file's
    digest. This is the whole cost, paid at creation; consumers recheck the
    embedded block against the plan's stated digests and the identity file.
    """
    from tools.dispatch_tessera_campaign import (
        COMPLETE_CAMPAIGN_SCOPE, ScopeRefused, verify_joint_campaign_scope)
    _require(isinstance(campaign_identity, dict) and set(campaign_identity) == {"path", "sha256"},
             "the frozen campaign identity needs a bound path and SHA256")
    plan = _json(original_plan, "original plan")
    campaign = _json(campaign_identity, "frozen campaign identity")
    try:
        scope = verify_joint_campaign_scope(
            plan, require_scope=COMPLETE_CAMPAIGN_SCOPE, campaign=campaign,
            label="catalog extension: original plan " + original_plan["sha256"][:12])
    except ScopeRefused as exc:
        raise ValueError("joint catalog extension: the original plan's campaign scope "
                         f"refused: {exc}") from exc
    scope = {**scope, "campaign_identity_sha256": campaign_identity["sha256"]}
    return {"schema": DERIVED_SCOPE_SCHEMA, "rule": DERIVED_SCOPE_RULE, "scope": scope,
            "derived_from": {"original_plan": dict(original_plan),
                             "campaign_identity": dict(campaign_identity),
                             "require_scope": COMPLETE_CAMPAIGN_SCOPE,
                             "bound_artifacts": _plan_scope_bindings(plan)}}


def _check_derived_scope(block, inputs, plan):
    """Recheck an embedded derived scope the cheap way, and return it.

    The bytes it cost to derive are not re-read here. What is rechecked: the
    block names this extension's original plan and the ``complete_campaign``
    kind; its artifact bindings are the plan's stated ones; the identity file
    it binds (a few hundred bytes, re-hashed) carries every frozen identity
    field the scope carries, and its digest is the scope's stamp; the kind,
    window and sequence-length fields are the plan's; the checkpoint digest
    and the declared roster counts are the plan's. The parent's own scope,
    sealed in every record, is what the effective scope is then compared
    with, byte for byte, at every site.
    """
    from tools.dispatch_tessera_campaign import (
        CAMPAIGN_IDENTITY_FIELDS, CAMPAIGN_IDENTITY_SCHEMA, CAMPAIGN_SCOPE_SCHEMA,
        COMPLETE_CAMPAIGN_SCOPE)
    _require(isinstance(block, dict), "derived campaign scope block is not a mapping")
    _same(block.get("schema"), DERIVED_SCOPE_SCHEMA, "derived scope schema")
    _same(block.get("rule"), DERIVED_SCOPE_RULE, "derived scope rule")
    derived = block.get("derived_from")
    _require(isinstance(derived, dict), "derived scope names no derivation")
    _same(derived.get("original_plan"), inputs["original_plan"], "derived scope original plan")
    _same(derived.get("require_scope"), COMPLETE_CAMPAIGN_SCOPE, "derived scope required kind")
    _same(derived.get("bound_artifacts"), _plan_scope_bindings(plan), "derived scope artifact bindings")
    scope = block.get("scope")
    _require(isinstance(scope, dict) and scope, "derived scope is empty")
    identity_bound = derived.get("campaign_identity")
    campaign = _json(identity_bound, "frozen campaign identity")
    _same(campaign.get("schema"), CAMPAIGN_IDENTITY_SCHEMA, "frozen campaign identity schema")
    for field in CAMPAIGN_IDENTITY_FIELDS:
        _same(scope.get(field), campaign.get(field), "derived scope " + field)
    _same(scope.get("campaign_identity_sha256"), identity_bound["sha256"], "derived scope identity stamp")
    _same(scope.get("schema"), CAMPAIGN_SCOPE_SCHEMA, "derived scope schema")
    _same(scope.get("kind"), COMPLETE_CAMPAIGN_SCOPE, "derived scope kind")
    _same(scope.get("selection_sha256"), None, "derived scope selection")
    _same(scope.get("window_count"), scope.get("campaign_window_count"), "derived scope window count")
    execution = plan.get("execution") or {}
    _same(scope.get("window_count"), execution.get("n_calib_samples"), "derived scope evaluated windows")
    _same(scope.get("calib_seqlen"), execution.get("calib_seqlen"), "derived scope sequence length")
    _require(plan.get("joint_eval") is None, "original plan evaluates a diagnostic panel")
    plan_inputs = plan.get("inputs") or {}
    _same(scope.get("campaign_checkpoint_sha256"), plan_inputs["merged_checkpoint"]["sha256"],
          "derived scope campaign checkpoint")
    _same((scope.get("source_unit_count"), scope.get("campaign_group_count")),
          (plan_inputs.get("required_source_units"), plan_inputs.get("required_campaign_groups")),
          "derived scope declared roster counts")
    return scope


def _effective_run_identity(document, header, inputs, plan):
    """The run identity consumers compare with the campaign's.

    The sealed one when the run sealed a scope; otherwise the sealed identity
    with the extension's derived scope in place of the null, admitted only
    because the extension binds exactly this run header and the plan it
    sealed (``_check_capture``). A null with no derivation stays a null, which
    no campaign scope equals.
    """
    identity = header["run_identity"]
    block = document.get("original_campaign_scope")
    if identity.get("campaign_scope") is not None:
        _require(block is None, "the original run sealed its campaign scope; "
                 "a derived scope over a sealed one is refused")
        return identity
    if not _seal(None, identity.get("campaign_scope"),
                 "the original run sealed no campaign scope and the extension derives none; "
                 "create it with the frozen campaign identity (--campaign-identity)",
                 same=document.get("schema") == SCHEMA_V3 and block is not None):
        return identity
    return {**identity, "campaign_scope": _check_derived_scope(block, inputs, plan)}


def create_extension(*, inputs, adjoint_capture, output, publish=None, campaign_identity=None):
    """Publish an extension binding the original Stage A run header.

    ``adjoint_capture`` names the completed receipt or any sealed checkpoint
    band of the original run; only its run header enters the document, so
    the extension is the same bytes whichever of them created it.
    ``publish(path, raw)`` writes the file and returns whether it created it;
    the default is ``publish_new_bytes``. The Stage B preparation passes its
    produced-output writer (PQ #1070).

    ``campaign_identity`` binds the operator's frozen campaign identity file
    by path and SHA256. It is required, and used, only when the run header
    seals ``campaign_scope: null`` (PQ #1126): the document is then v3 and
    carries the scope :func:`derive_original_campaign_scope` derives from the
    original plan. A header that seals its scope creates the v2 bytes it
    always did, whether or not an identity is given.
    """
    evidence = verify_catalog_pair(inputs)
    header = _run_header(_json(adjoint_capture, "original adjoint capture"))
    plan = _check_capture(header, inputs, _json(inputs["original_prepared"], "original prepared"))
    document = {"schema": SCHEMA, "inputs": inputs, "adjoint_run_header": header,
        "adjoint_run_header_sha256": canonical_json_sha256(header, where="original Stage A run header"),
        "evidence": evidence}
    if header["run_identity"].get("campaign_scope") is None and dev_mode_enabled():
        # Deriving the scope re-hashes every artifact it names, the merged
        # checkpoint included, only to seal it: dev mode does no seal-only
        # work (PQ #1147) and publishes the v2 bytes with the null scope.
        dev_warning("the original run sealed no campaign scope (PQ #1126); the scope is "
                    "not derived (sealing is off, PQ #1147) and the extension keeps "
                    "the null scope")
    elif header["run_identity"].get("campaign_scope") is None:
        _require(campaign_identity is not None,
                 "the original run sealed no campaign scope (PQ #1126); bind the frozen campaign "
                 "identity (--campaign-identity, --campaign-identity-sha256) so the scope is "
                 "derived from the original plan")
        block = derive_original_campaign_scope(inputs["original_plan"], campaign_identity)
        # The creator holds its own block to the consumers' check before publishing it.
        _check_derived_scope(block, inputs, plan)
        document.update(schema=SCHEMA_V3, original_campaign_scope=block)
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    writer = publish_new_bytes if publish is None else publish
    _require(writer(Path(output), raw), "extension output already exists; refusing overwrite")
    return {"path": str(Path(output).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def extension_run_header(bound):
    """The Stage A run header an extension binds (embedded in v2, derived in v1).

    For callers that check an extension against itself, such as export and
    allocation, where no slice is in hand; :func:`require_extension` still
    verifies it independently.
    """
    document = _json(bound, "catalog extension")
    if document.get("schema") in (SCHEMA, SCHEMA_V3):
        return document["adjoint_run_header"]
    _same(document.get("schema"), SCHEMA_V1, "extension schema")
    return _run_header(_json(document["adjoint_capture"], "original adjoint capture"))


def require_extension(bound, *, run_header, plan_sha256, prepared_sha256):
    """Independently check the pair and return the effective original run identity.

    ``run_header`` is the Stage A run header a slice, band or receipt
    carries; the extension must bind exactly it. The identity returned is
    the sealed one, or, for a v3 extension over a run that sealed
    ``campaign_scope: null``, the sealed identity with the derived scope in
    its place (:func:`_effective_run_identity`); consumers compare its
    ``campaign_scope`` with the campaign's.
    """
    document = _json(bound, "catalog extension")
    _require(document.get("schema") in (SCHEMA, SCHEMA_V1, SCHEMA_V3), "extension schema differs")
    inputs = document["inputs"]
    _seal(plan_sha256, inputs["extended_plan"]["sha256"], "extended plan binding")
    _seal(prepared_sha256, inputs["extended_prepared"]["sha256"], "extended prepared binding")
    if document["schema"] in (SCHEMA, SCHEMA_V3):
        header = document["adjoint_run_header"]
        _same(document["adjoint_run_header_sha256"],
              canonical_json_sha256(header, where="original Stage A run header"),
              "original Stage A run header digest")
    else:
        capture = _json(document["adjoint_capture"], "original adjoint capture")
        _same(document["adjoint_receipt_sha256"], canonical_json_sha256(capture, where="original capture"),
              "original capture digest")
        header = _run_header(capture)
    _same(canonical_json_sha256(header, where="original Stage A run header"),
          canonical_json_sha256(run_header, where="Stage A run header"),
          "original Stage A run header")
    plan = _check_capture(header, inputs, _json(inputs["original_prepared"], "original prepared"))
    identity = _effective_run_identity(document, header, inputs, plan)
    if dev_mode_enabled():
        # Recomputing the evidence opens both production caches only to
        # re-seal what the creator recorded: seal-only work (PQ #1147).
        dev_warning("catalog extension evidence is not recomputed (sealing is off, "
                    "PQ #1147); the recorded evidence is used")
    else:
        _same(document["evidence"], verify_catalog_pair(inputs),
              "independently recomputed extension evidence")
    return identity


def extension_campaign_identity(document):
    """The frozen campaign identity binding a v3 extension derives its scope from, or None."""
    block = document.get("original_campaign_scope") if document.get("schema") == SCHEMA_V3 else None
    if block is None:
        return None
    return dict(block["derived_from"]["campaign_identity"])


def _hessian_reference_commitments(hessian, where):
    """One workspace's authenticated canonical binding and per-unit H digests.

    The reference file is opened through the producer's own reader, which
    recomputes the capture seal from the per-unit commitments and checks the
    canonical capture and census it names. Requiring that seal to equal the
    ``capture_sha256`` the workspace's cost rows carry binds the file to the
    prices, so the digests returned are the ones those rows were priced under.
    """
    from .tessera_calibration_cache import open_hessian_reference
    path = hessian.get("capture_path") if isinstance(hessian, dict) else None
    _require(isinstance(path, str) and bool(path), where + " Hessian reference path is missing")
    with open_hessian_reference(path) as owner:
        _same(owner.descriptor["capture_sha256"], hessian.get("capture_sha256"), where + " Hessian reference seal")
        _same(owner.binding(), hessian.get("reference_binding"), where + " Hessian reference binding")
        return owner.binding(), owner.committed_units()


def _overlay_hessian_commitments(overlay_hessian, panel_hessian):
    """Per-unit H digests of the overlay's cost run and of the panel, from one capture.

    A reference file's capture seal covers the unit roster it commits, so two
    workspaces that reference one canonical capture under different rosters
    or census paths carry different seals for identical Hessians. The identity
    of a measured H is its content: the canonical capture digest, the census
    digest and each unit's tensor digest. Without a reference binding on both
    sides there is nothing finer than the seal, and a differing seal refuses.
    """
    _require(isinstance(overlay_hessian, dict) and overlay_hessian.get("reference_binding") is not None
             and panel_hessian.get("reference_binding") is not None,
             "overlay measured H capture_sha256 differs")
    overlay_binding, overlay_units = _hessian_reference_commitments(overlay_hessian, "overlay measured")
    panel_binding, panel_units = _hessian_reference_commitments(panel_hessian, "panel")
    _same(overlay_binding, panel_binding, "overlay measured H canonical capture and census")
    return overlay_units, panel_units


def hessian_references(payload):
    """Every Hessian reference a cost table's rows can name, found through hash-bound inputs.

    A joined table keeps the panel's H provenance in ``provenance.hessian``;
    the rows :func:`attach_candidate_overlay` added keep the overlay cost
    run's ``capture_sha256`` (PQ #985). The overlay's own H provenance, the
    one that names its ``hessian_capture.references.json``, is reached through
    the chain the table already binds: ``provenance.catalog_extension`` ->
    ``inputs.extended_plan`` -> ``inputs.candidate_overlay`` -> ``cost`` ->
    ``provenance.hessian`` (or directly from ``provenance.candidate_overlay``
    on a table built with the overlay attached). Every hop is read by path and
    SHA-256, so no reference is taken on a name alone.

    Returns ``{"primary": <the table's own capture_sha256>, "captures":
    {capture_sha256: hessian provenance}}`` for
    ``tessera_menu.assert_uniform_hessian_identity`` (RobTand/prismaquant#1270).
    Nothing here opens a reference file; the gate does, for the rows that
    need it.
    """
    provenance = (payload or {}).get("provenance") or {}
    primary = provenance.get("hessian")
    captures = {}

    def add(hessian):
        if isinstance(hessian, dict) and hessian.get("capture_sha256"):
            captures.setdefault(hessian["capture_sha256"], dict(hessian))

    add(primary)
    overlays = []
    if provenance.get("catalog_extension") is not None:
        document = _json(provenance["catalog_extension"], "catalog extension")
        plan = _json(document["inputs"]["extended_plan"], "extended plan")
        overlays.append(plan.get("inputs", {}).get("candidate_overlay"))
    overlays.append(provenance.get("candidate_overlay"))
    for bound in overlays:
        if bound is None:
            continue
        catalog = _json(bound, "candidate overlay")
        for source in catalog_sources(catalog):
            costs = pickle.loads(_read_bound(source["cost"], "overlay measured scalar costs"))
            add((costs.get("provenance") or {}).get("hessian"))
    return {"primary": primary.get("capture_sha256") if isinstance(primary, dict) else None,
            "captures": captures}


def attach_candidate_overlay(data, bound, *, verify_payloads=False):
    """Attach an authenticated historical catalog without rewriting its base.

    This is intake, not qualification. The new PWC must separately carry the
    actual per-cell decoder comparison and render digests checked above.
    Historical scalar scores remain scalar scores; Stage B measures every
    added joint cost using the unchanged source adjoints.
    """
    from .tessera_expert_projection import EXPERT_WIRES_KEY
    catalog = _json(bound, "candidate overlay")
    view = catalog_view(catalog)
    original = _json(catalog["old_prepared"], "overlay original prepared")
    _same(catalog.get("old_pwc"), original["production_cache"], "overlay original PWC")
    for key in PREPARED_SCIENCE:
        _same(catalog.get(key), original.get(key), "overlay scientific identity " + key)
    base_inputs = {key: value for key, value in data.inputs.items() if key != "candidate_overlay"}
    # The base catalog is authenticated by its own complete loader. Matching
    # the prepared PWC's input identities ties the overlay's reference cells
    # to those exact base artifacts, rather than merely to matching names.
    old_cache = pickle.loads(_read_bound(original["production_cache"], "overlay original PWC"))
    _same(old_cache.metadata.get("inputs"), base_inputs, "overlay base catalog")
    sources = view["sources"]
    for source in sources:
        if source.get("reseal_proof") is not None:
            _read_bound(source["reseal_proof"], "overlay encoder source proof")
    # The cells declare the coverage (PQ #1432): each names an original unit
    # and a format that unit does not already offer, once.
    rows = catalog["cells"]
    pairs = [(row.get("qname"), row.get("format")) for row in rows]
    _same(len(set(pairs)), len(pairs), "unique added candidate roster")
    for name, fmt in pairs:
        _require(name in original["formats_by_qname"], "overlay unit is not in the original roster")
        _require(fmt not in original["formats_by_qname"][name], "overlay re-adds an original candidate of " + name)
    if catalog["schema"] == CATALOG_SCHEMA_V1:
        # A v1 catalog's one format covers every original unit that lacks it.
        (fmt,) = view["formats"]
        _same(set(pairs), {(name, fmt) for name, formats in original["formats_by_qname"].items()
                           if fmt not in formats}, "complete added candidate roster")
    selected_names = set(data.formats_by_qname)
    panel_hessian = data.payload["provenance"]["hessian"]
    recipe = added_format_recipes()
    # Each source's measured costs are read once, when its first selected
    # cell is met, and only for sources that answer a selected unit.
    source_costs, reference_units, added = {}, {}, {}
    with EncoderAdoptionValidation() as proof_checks:
        for row, index in zip(rows, view["cell_sources"]):
            name, fmt = row["qname"], row["format"]
            if name not in selected_names:
                continue
            _require((name, fmt) not in data.cells, "overlay attempts to replace an original cell")
            source = sources[index]
            if index not in source_costs:
                source_costs[index] = pickle.loads(_read_bound(source["cost"], "overlay measured scalar costs"))
            costs = source_costs[index]
            overlay_hessian = (costs.get("provenance") or {}).get("hessian")
            adoption = row["catalog_source_adoption"]
            _same(adoption.get("schema"), ADOPTION_SCHEMA, "overlay adoption schema")
            _same(adoption.get("encoder_source_proof"), source.get("reseal_proof"), "overlay encoder source proof")
            proof_checks.verify(adoption, fmt=fmt)
            reference_pair = tuple(adoption["reference_pair"])
            _require(reference_pair in data.cells and reference_pair[0] == name,
                     "overlay reference candidate is absent from the base catalog")
            reference = data.cells[reference_pair]["record"]["identity"]
            candidate = row["record"]["identity"]
            _same(candidate, adoption["candidate_encoding_identity"], "overlay recorded candidate identity")
            _same(canonical_json_sha256(adoption["reference_encoding_identity"], where="overlay original encoding"),
                  old_cache.metadata["verified_cells"][reference_pair]["encoding_identity_sha256"],
                  "overlay authenticated original encoding")
            _same(row["source_weight"], old_cache.metadata["verified_cells"][reference_pair]["source_weight"],
                  "overlay authenticated original source")
            for key in ("unit", "source", "projection", "calibration", "encoder_fixture_id"):
                _require(key in candidate and key in reference, "overlay source/H identity lacks " + key)
                _same(candidate.get(key), reference.get(key), "overlay source/H " + key)
                _same(candidate.get(key), adoption["reference_encoding_identity"].get(key),
                      "overlay adopted source/H " + key)
            _same(candidate.get("unit"), name, "overlay candidate unit")
            _same(candidate.get("recipe"), recipe(fmt), "overlay candidate recipe")
            _same(canonical_json_sha256(candidate, where="overlay candidate encoding"),
                  row["encoding_identity_sha256"], "overlay candidate encoding digest")
            anchor = row["anchor"]
            scalar = costs["costs"][name][fmt]
            _require(scalar.get("output_mse_measured") is True
                     and scalar.get("cost_source") == "tessera_campaign_measured"
                     and scalar.get("tessera_provenance") == "measured"
                     and scalar.get("currency") == "output_mse_under_route_activation_contract",
                     "overlay candidate is not a measured historical row")
            for target, source in (("output_mse", "dloss"), ("tessera_family", "family"),
                                   ("tessera_body_rate_q256", "body_rate_q256"),
                                   ("activation_contract", "activation_contract"),
                                   ("activation_quantized", "activation_quantized"),
                                   ("input_global_scale", "input_global_scale"), ("wire_bytes", "wire_bytes")):
                _same(scalar.get(target), anchor.get(source), "overlay measured " + target)
            row_hessian = scalar.get("hessian_identity", {})
            for key in ("supplied", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _same(row_hessian.get(key), panel_hessian.get(key), "overlay measured H " + key)
            if row_hessian.get("capture_sha256") != panel_hessian.get("capture_sha256"):
                if index not in reference_units:
                    reference_units[index] = _overlay_hessian_commitments(overlay_hessian, panel_hessian)
                overlay_units, panel_units = reference_units[index]
                _same(row_hessian.get("capture_sha256"), overlay_hessian.get("capture_sha256"),
                      "overlay measured H capture seal")
                _same(row_hessian.get("reference_binding"), overlay_hessian.get("reference_binding"),
                      "overlay measured H reference binding")
                _require(name in overlay_units, "overlay measured H commits no Hessian for " + name)
                _require(name in panel_units, "overlay unit has no Hessian in the panel capture: " + name)
                _same(overlay_units[name], panel_units[name], "overlay measured unit Hessian " + name)
            _same(row["activation"].get("input_global_scale"), anchor.get("input_global_scale"),
                  "overlay activation scale")
            for field in ("wire", "render"):
                path = Path(row[field])
                _require(path.is_absolute() and not path.is_symlink(), "overlay path must be an existing regular artifact")
                stat = path.stat()
                _require(stat_module.S_ISREG(stat.st_mode), "overlay artifact is not a regular file")
                _artifact_fence(path, stat, row[field + "_stat"],
                                row["record"]["blob_sha256"] if field == "wire" else None,
                                "overlay current " + field + " fence")
            cell = {key: copy.deepcopy(row[key]) for key in
                    ("anchor", "record", "wire", "render", "render_origin", "render_comparison",
                     "catalog_source_adoption", "adopted_source_hessian")}
            if verify_payloads:
                _same(hashlib.sha256(Path(cell["wire"]).read_bytes()).hexdigest(),
                      cell["record"]["blob_sha256"], "overlay wire bytes")
                cell["render_file_sha256"] = hashlib.sha256(Path(cell["render"]).read_bytes()).hexdigest()
            data.cells[name, fmt] = cell
            added.setdefault(name, []).append(fmt)
            data.payload["costs"][name][fmt] = copy.deepcopy(scalar)
            if name in data.payload.get(EXPERT_WIRES_KEY, {}):
                data.payload[EXPERT_WIRES_KEY][name][fmt] = copy.deepcopy(cell["record"])
    # One insertion per unit, so several added formats land in the one
    # deterministic order the assembler and the pair check use.
    for name, formats in added.items():
        data.formats_by_qname[name] = extended_roster(data.formats_by_qname[name], formats)
    data.payload["provenance"]["candidate_overlay"] = dict(bound)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="JSON with four path/sha256 input bindings")
    parser.add_argument("--check-catalog-only", action="store_true",
                        help="validate proposed metadata, without creating capture-reuse authority")
    parser.add_argument("--adjoint-capture")
    parser.add_argument("--adjoint-capture-sha256")
    parser.add_argument("--campaign-identity", default=None,
                        help="the frozen campaign identity file; required when the original "
                             "run sealed campaign_scope null (PQ #1126)")
    parser.add_argument("--campaign-identity-sha256", default=None)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    inputs = json.loads(Path(args.inputs).read_bytes())
    if bool(args.campaign_identity) != bool(args.campaign_identity_sha256):
        parser.error("--campaign-identity and --campaign-identity-sha256 go together")
    if args.check_catalog_only:
        if args.adjoint_capture or args.adjoint_capture_sha256 or args.out:
            parser.error("catalog-only validation takes no capture or publication path")
        print(json.dumps({"status": "proposed_catalog_validated", "capture_reuse_authority": False,
                          "evidence": verify_catalog_pair(inputs)}, sort_keys=True))
        return 0
    if not args.adjoint_capture or not args.adjoint_capture_sha256 or not args.out:
        parser.error("capture path, SHA256 and fresh output path are required to create an extension")
    identity = (None if args.campaign_identity is None else
                {"path": str(Path(args.campaign_identity).resolve()), "sha256": args.campaign_identity_sha256})
    result = create_extension(inputs=inputs,
        adjoint_capture={"path": args.adjoint_capture, "sha256": args.adjoint_capture_sha256},
        output=args.out, campaign_identity=identity)
    print(json.dumps(result, sort_keys=True))
    return 0


def selected_cache_read_paths(manifest):
    """Enumerate immutable mixed-cache export inputs for a PB read manifest.

    This returns paths, not placement or authorization. The caller binds their
    current bytes using PB's normal input-manifest machinery. Original wires
    remain at their original roots; every historical package and proof arm is
    explicit, including the served-policy source documents.
    """
    _same(manifest.get('schema'), 'tessera.cached_units.v2', 'rooted selected cache schema')
    paths = {str(Path(manifest['wire_roots'][manifest['unit_roots'][name]]) / row['file'])
             for name, row in manifest['units'].items()}
    for package in manifest['producer_packages'].values():
        root = Path(package['path'])
        _require(root.is_dir() and root.resolve() == root, "selected producer root must be canonical and present")
        paths.update(str(path) for path in root.rglob('*') if path.is_file()
                     and path.suffix in {'.py', '.cu', '.cuh', '.cpp', '.h'})
    authority = manifest['reuse_authority']
    documents = set()
    proof_bindings = {}
    def add(bound):
        _require(isinstance(bound, dict) and set(bound) == {'path', 'sha256'},
                 'selected control dependency needs an exact binding')
        paths.add(bound['path'])
    def control(bound, kind):
        add(bound)
        key = (bound['path'], bound['sha256'])
        if key in documents:
            return
        documents.add(key)
        document = _json(bound, 'selected ' + kind + ' dependencies')
        if kind == 'extension':
            if document.get('schema') == SCHEMA_V1:
                add(document['adjoint_capture'])
            identity = extension_campaign_identity(document)
            if identity is not None:
                add(identity)
            for name, value in document['inputs'].items():
                control(value, 'plan' if name.endswith('_plan') else 'prepared')
        elif kind == 'plan':
            for name, value in document.get('inputs', {}).items():
                if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
                    if name == 'candidate_overlay': control(value, 'catalog')
                    else: add(value)
        elif kind == 'prepared':
            add(document['production_cache'])
        elif kind == 'catalog':
            if 'old_pwc' in document: add(document['old_pwc'])
            for source in catalog_sources(document):
                if source.get('cost') is not None: add(source['cost'])
                if source.get('reseal_proof') is not None:
                    proof_bindings[source['reseal_proof']['sha256']] = source['reseal_proof']
            if 'old_prepared' in document: control(document['old_prepared'], 'prepared')
        elif kind == 'activation':
            for name in ('original_prepared', 'original_cache', 'census'): add(document[name])
        elif kind == 'resources':
            for name, value in document['inputs'].items():
                control(value, {'original_plan': 'plan', 'original_prepared': 'prepared',
                    'candidate_overlay': 'catalog', 'served_activation_policy': 'activation'}[name])
        for name, policy_kind in (('served_activation_policy', 'activation'),
                                  ('stage_b_resource_policy', 'resources')):
            if document.get(name) is not None: control(document[name], policy_kind)
    control(authority['catalog_extension'], 'extension')
    control(authority['candidate_overlay'], 'catalog')
    for bound in authority['encoder_source_proofs']:
        proof_bindings[bound['sha256']] = bound
    for bound in proof_bindings.values():
        document = _json(bound, 'selected encoder proof dependencies')
        paths.add(bound['path'])
        paths.add(document['fixture_id']['result'])
        paths.update(arm['result'] for arm in document['arms'])
    bound = manifest.get('served_activation_policy')
    if bound:
        policy = _json(bound, 'selected served policy dependencies')
        paths.add(bound['path'])
        paths.update(policy[key]['path'] for key in ('original_prepared', 'original_cache', 'census'))
    return sorted(paths)


if __name__ == "__main__":
    raise SystemExit(main())
