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
from .tessera_joint_allocation import _read_bound, _bound_stat_fence

SCHEMA = "prismaquant.joint_catalog_extension.v1"
INPUTS = ("original_plan", "original_prepared", "extended_plan", "extended_prepared")
# These select candidate artifacts or their output namespace; every other
# plan field, including the entire execution/derivative policy, stays exact.
CANDIDATE_PLAN_FIELDS = frozenset(("inputs", "output_root", "historical_encoder_reuse"))
PREPARED_SCIENCE = ("source_model_identity", "source_execution", "calibration_input",
                    "projection_backend", "reader_identity")
QUALIFIED_CELL_FIELDS = ("source_weight", "rendered_weight", "activation",
                         "encoding_identity_sha256", "wire_sha256", "render_file_sha256",
                         "render_origin", "render_comparison")
_VERIFIED_PAIR = {}
_VERIFIED_ENCODER_PROOFS = {}
_SELECTED_CATALOG = {}
ADOPTION_SCHEMA = "prismaquant.joint_catalog_source_adoption.v1"
ADDED_FORMAT = "TESSERA_E2M1_K2_R896"
ADDED_RECIPE = {"body": "tcq", "channel_sigma": None, "grid": "E2M1x2",
                "plane": "lut16", "q256": 896, "seed": 0, "sigma": None,
                "span": 2, "window_bits": 0}


def _require(value, message):
    if not value:
        raise ValueError("joint catalog extension: " + message)


def _same(a, b, message):
    _require(a == b, message + " differs")


def _json(bound, label):
    return json.loads(_read_bound(bound, label))


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

    def verify(self, adoption):
        _require(self._active, "adoption validation is outside its operation")
        from tools.reseal_campaign_identity import unit_kind
        _same(adoption.get("schema"), ADOPTION_SCHEMA, "encoder adoption schema")
        reference, candidate = adoption["reference_encoding_identity"], adoption["candidate_encoding_identity"]
        bound = adoption["encoder_source_proof"]
        key = (bound["path"], bound["sha256"], reference.get("encoder_source_sha256"),
               candidate.get("encoder_source_sha256"), reference.get("encoder_fixture_id"),
               candidate.get("encoder_fixture_id"), unit_kind(candidate.get("unit", '')))
        if key not in self._proofs:
            result = validated_encoder_adoption(adoption)
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


def validated_encoder_adoption(adoption):
    """Authenticate the existing migration's semantics and all proof dependencies.

    This never substitutes an encoder seal. It binds the exact old/new source
    pair and fixture, reuses the owning reseal proof validator, and retains
    inode/size/mtime/ctime fences for every dependency before memoized reuse.
    The producer package is a declaration; Tessera independently hashes that
    complete package before using its historical identity factory.
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
    proof = adoption["encoder_source_proof"]
    stratum = (unit_kind(candidate["unit"]), format_family(ADDED_FORMAT))
    key = (proof["path"], proof["sha256"], _bound_stat_fence(Path(proof["path"])), old, new, fixture, stratum)
    cached = _VERIFIED_ENCODER_PROOFS.get(key)
    if cached is not None and cached["fences"] == tuple(
            (b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in cached["dependencies"]):
        return copy.deepcopy(cached)
    document = _json(proof, "encoder source proof")
    _same(document.get("pins", {}).get("old", {}).get("encoder_source_sha256"), old,
          "encoder proof old source")
    _same(document.get("pins", {}).get("new", {}).get("encoder_source_sha256"), new,
          "encoder proof new source")
    _require(document.get("arms"), "encoder proof has no measured comparison arms")
    try:
        checked = load_bundle(proof["path"], document["pins"])
    except Refused as error:
        raise ValueError("joint catalog extension: encoder proof refused: " + str(error)) from error
    _require(stratum in checked["covered"],
             "encoder proof does not cover the added candidate stratum")
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
              "fences": tuple((b["path"], b["sha256"], _bound_stat_fence(Path(b["path"]))) for b in dependencies)}
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
        _same(catalog.get("schema"), "prismaquant.t4_adopted_catalog.v1", "selected candidate catalog schema")
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
    proof = (validation.verify(adoption) if validation is not None
             else validated_encoder_adoption(adoption))
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
        observed = path.stat()
        _same({"inode": observed.st_ino, "bytes": observed.st_size,
               "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns},
              row[field + "_stat"], "selected current " + field + " fence")
    return {"qualification_activation": copy.deepcopy(row["activation"]),
            "adoption": copy.deepcopy(adoption), "wire_root": str(Path(row["wire"]).resolve().parent),
            "catalog": dict(bound), **proof}


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
    _require(all(ADDED_FORMAT in formats for formats in new["formats_by_qname"].values()),
             "extended E2M1 candidate must cover the complete original qname roster")
    for name, formats in old["formats_by_qname"].items():
        _require(set(formats) <= set(new["formats_by_qname"][name]), "original candidate removed for " + name)

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
    _same(previous.levers, extended.levers, "original render levers")
    for name, value in (previous.activation_max_abs or {}).items():
        _same((extended.activation_max_abs or {}).get(name), value, "original activation scale " + name)
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
    with EncoderAdoptionValidation() as proof_checks:
        for pair in additions:
            _same(pair[1], ADDED_FORMAT, "this extension's added candidate format")
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
            _same(candidate.get("recipe"), ADDED_RECIPE, "added E2M1 recipe")
            # Hash the existing source-migration proof once; do not manufacture a
            # new proof or silently normalize another encoding field here.
            proof = adoption.get("encoder_source_proof")
            verified_proof = proof_checks.verify(adoption)
            proof_fences.update(verified_proof["fences"])
    science = {"plan": {k: v for k, v in old_plan.items() if k not in CANDIDATE_PLAN_FIELDS},
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


def _check_capture(receipt, inputs, original):
    _require(receipt.get("schema") == "prismaquant.joint_adjoint_capture.v1"
             and receipt.get("status") == "complete", "actual completed Stage A capture required")
    identity = receipt.get("run_identity", {})
    _same(identity.get("plan_sha256"), inputs["original_plan"]["sha256"], "capture original plan")
    _same(identity.get("prepared_sha256"), inputs["original_prepared"]["sha256"], "capture original prepared")
    calibration = original["calibration_input"]
    _same(identity.get("calibration_sha256"), calibration["calibration_sha256"], "capture calibration")
    _same(identity.get("calibration_shape"), calibration["shape"], "capture calibration shape")
    # Stage A's roster spelling includes one terminating newline per qname;
    # the quantum roster uses another spelling. Never compare unlike hashes.
    roster = "".join(name + "\n" for name in sorted(original["formats_by_qname"]))
    _same(identity.get("unit_roster_sha256"), hashlib.sha256(roster.encode()).hexdigest(), "capture qname roster")
    plan = _json(inputs["original_plan"], "original plan")
    for key in ("n_probes", "seed_base"):
        _same(identity.get(key), plan["execution"][key], "capture " + key)


def create_extension(*, inputs, adjoint_capture, output):
    evidence = verify_catalog_pair(inputs)
    capture = _json(adjoint_capture, "original adjoint capture")
    _check_capture(capture, inputs, _json(inputs["original_prepared"], "original prepared"))
    document = {"schema": SCHEMA, "inputs": inputs, "adjoint_capture": adjoint_capture,
        "adjoint_receipt_sha256": canonical_json_sha256(capture, where="original adjoint receipt"),
        "evidence": evidence}
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _require(publish_new_bytes(Path(output), raw), "extension output already exists; refusing overwrite")
    return {"path": str(Path(output).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def require_extension(bound, *, receipt, plan_sha256, prepared_sha256):
    """Independently check the pair and return the original receipt identity."""
    document = _json(bound, "catalog extension")
    _same(document.get("schema"), SCHEMA, "extension schema")
    inputs = document["inputs"]
    _same(inputs["extended_plan"]["sha256"], plan_sha256, "extended plan binding")
    _same(inputs["extended_prepared"]["sha256"], prepared_sha256, "extended prepared binding")
    capture = _json(document["adjoint_capture"], "original adjoint capture")
    _same(capture, receipt, "original capture bytes")
    _same(document["adjoint_receipt_sha256"], canonical_json_sha256(receipt, where="original capture"),
          "original capture digest")
    _check_capture(capture, inputs, _json(inputs["original_prepared"], "original prepared"))
    _same(document["evidence"], verify_catalog_pair(inputs), "independently recomputed extension evidence")
    return capture["run_identity"]


def attach_candidate_overlay(data, bound, *, verify_payloads=False):
    """Attach an authenticated historical catalog without rewriting its base.

    This is intake, not qualification. The new PWC must separately carry the
    actual per-cell decoder comparison and render digests checked above.
    Historical scalar scores remain scalar scores; Stage B measures every
    added joint cost using the unchanged source adjoints.
    """
    from .tessera_expert_projection import EXPERT_WIRES_KEY
    catalog = _json(bound, "candidate overlay")
    _same(catalog.get("schema"), "prismaquant.t4_adopted_catalog.v1", "candidate overlay schema")
    _same(catalog.get("format"), ADDED_FORMAT, "candidate overlay format")
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
    costs = pickle.loads(_read_bound(catalog["cost"], "overlay measured scalar costs"))
    _read_bound(catalog["reseal_proof"], "overlay encoder source proof")
    expected = {(name, ADDED_FORMAT) for name in original["formats_by_qname"]
                if ADDED_FORMAT not in original["formats_by_qname"][name]}
    rows = catalog.get("cells")
    _require(isinstance(rows, list), "overlay cells are missing")
    pairs = [(row.get("qname"), row.get("format")) for row in rows]
    _same(set(pairs), expected, "complete added candidate roster")
    _same(len(pairs), len(expected), "unique added candidate roster")
    selected_names = set(data.formats_by_qname)
    with EncoderAdoptionValidation() as proof_checks:
        for row in rows:
            name, fmt = row["qname"], row["format"]
            if name not in selected_names:
                continue
            _require((name, fmt) not in data.cells, "overlay attempts to replace an original cell")
            adoption = row["catalog_source_adoption"]
            _same(adoption.get("schema"), ADOPTION_SCHEMA, "overlay adoption schema")
            _same(adoption.get("encoder_source_proof"), catalog["reseal_proof"], "overlay encoder source proof")
            proof_checks.verify(adoption)
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
            _same(candidate.get("recipe"), ADDED_RECIPE, "overlay candidate recipe")
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
            for key in ("supplied", "capture_sha256", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _same(scalar.get("hessian_identity", {}).get(key),
                  data.payload["provenance"]["hessian"].get(key), "overlay measured H " + key)
            _same(row["activation"].get("input_global_scale"), anchor.get("input_global_scale"),
                  "overlay activation scale")
            for field in ("wire", "render"):
                path = Path(row[field])
                _require(path.is_absolute() and not path.is_symlink(), "overlay path must be an existing regular artifact")
                stat = path.stat()
                _require(stat_module.S_ISREG(stat.st_mode), "overlay artifact is not a regular file")
                _same({"inode": stat.st_ino, "bytes": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns},
                      row[field + "_stat"], "overlay current " + field + " fence")
            cell = {key: copy.deepcopy(row[key]) for key in
                    ("anchor", "record", "wire", "render", "render_origin", "render_comparison",
                     "catalog_source_adoption", "adopted_source_hessian")}
            if verify_payloads:
                _same(hashlib.sha256(Path(cell["wire"]).read_bytes()).hexdigest(),
                      cell["record"]["blob_sha256"], "overlay wire bytes")
                cell["render_file_sha256"] = hashlib.sha256(Path(cell["render"]).read_bytes()).hexdigest()
            data.cells[name, fmt] = cell
            data.formats_by_qname[name] = (*data.formats_by_qname[name], fmt)
            data.payload["costs"][name][fmt] = copy.deepcopy(scalar)
            if name in data.payload.get(EXPERT_WIRES_KEY, {}):
                data.payload[EXPERT_WIRES_KEY][name][fmt] = copy.deepcopy(cell["record"])
    data.payload["provenance"]["candidate_overlay"] = dict(bound)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="JSON with four path/sha256 input bindings")
    parser.add_argument("--check-catalog-only", action="store_true",
                        help="validate proposed metadata, without creating capture-reuse authority")
    parser.add_argument("--adjoint-capture")
    parser.add_argument("--adjoint-capture-sha256")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    inputs = json.loads(Path(args.inputs).read_bytes())
    if args.check_catalog_only:
        if args.adjoint_capture or args.adjoint_capture_sha256 or args.out:
            parser.error("catalog-only validation takes no capture or publication path")
        print(json.dumps({"status": "proposed_catalog_validated", "capture_reuse_authority": False,
                          "evidence": verify_catalog_pair(inputs)}, sort_keys=True))
        return 0
    if not args.adjoint_capture or not args.adjoint_capture_sha256 or not args.out:
        parser.error("capture path, SHA256 and fresh output path are required to create an extension")
    result = create_extension(inputs=inputs,
        adjoint_capture={"path": args.adjoint_capture, "sha256": args.adjoint_capture_sha256}, output=args.out)
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
    for key in ('catalog_extension', 'candidate_overlay'):
        paths.add(authority[key]['path'])
    for bound in authority['encoder_source_proofs']:
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
