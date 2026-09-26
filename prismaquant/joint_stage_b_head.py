"""The Stage B quantum head, read once at prepare (PQ #1010).

Every Stage B layer quantum used to repeat the whole campaign's metadata
intake before its first GPU allocation: ``load_measured_anchor_input`` over
the merged checkpoint (7.2 GB), the merged cost pickle, all 36,423 journal
shards and a stat of every wire and render, then the resource and served
activation policy re-derivations. None of those reads were declared, and all
of them answer the same question for every layer.

This module moves that intake to the metadata producer. The producer runs it
once (:func:`build_head_slices`) and publishes one **head slice** per layer:
a canonical JSON document holding exactly what a quantum's head checks and
uses -- the layer's candidate roster, the campaign facts the intake checked
(source model, attention implementation, calibration draw, encoder seal),
the render read bound, the device limits of a verified resource policy, and
the digest-bound head files the quantum still opens (the prepared completion,
the production pickle, the source-identity cache, the served activation
policy). The slice is bound to the prepared completion by digest, as the
Stage A slice (PQ #993) is bound to its receipt, and the record binds the
slice by digest.

The quantum's executable manifest declares the slice and its head files in
its ``head`` phase, and :func:`load_quantum_head` reads them through the
staged whole-file reader. Nothing else under the campaign inputs is opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .cost_stage_checkpoint import canonical_json_bytes, canonical_json_sha256
from .dev_mode import seal_check

HEAD_SLICE_SCHEMA = "prismaquant.joint_stage_b_head_slice.v1"

#: The head files a slice may declare, in the order a quantum reads them.
#: ``prepared`` and ``production_cache`` are always present; the other two
#: exist exactly when the plan binds them.
HEAD_FILE_ROLES = ("prepared", "production_cache", "served_activation_policy",
                   "source_identity_cache")

#: The calibration draw fields the quantum compares with the original full
#: draw the campaign's Hessians were fitted on (``run_layer_quantum``).
CALIBRATION_IDENTITY_FIELDS = ("fit_ids_sha256", "text_sha256", "nsamples",
                               "seqlen", "seed")

#: Formats that carry no render (``aura_cost._ZERO_COST_FORMATS`` is the
#: quantum's own list; the intake roster appends exactly this one).
PASSTHROUGH_FORMAT = "BF16"


class HeadSliceRefused(ValueError):
    """A head slice that is foreign, stale or tampered: refuse before any work."""


def _refuse(condition, message):
    if not condition:
        raise HeadSliceRefused(message)


def _same(actual, expected, label):
    _refuse(actual == expected, f"{label}: {actual!r} != {expected!r}")


def _seal(actual, expected, label, *, message=None):
    """A run seal (PQ #1147): certified mode refuses exactly as ``_same`` (or
    with ``message``); dev mode prints a ``[DEV-MODE]`` line and continues."""
    seal_check(label, expected, actual, where="Stage B head slice",
               refusal=lambda: HeadSliceRefused(
                   message if message is not None
                   else f"{label}: {actual!r} != {expected!r}"))


def _hex64(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def head_slice_bytes(head_slice) -> bytes:
    """The exact bytes a head-slice file carries: its canonical JSON."""
    return canonical_json_bytes(head_slice, where="Stage B head slice")


def head_slice_sha256(head_slice) -> str:
    return hashlib.sha256(head_slice_bytes(head_slice)).hexdigest()


def head_slice_directory(output_root: str, *, metadata_root: str | None = None) -> str:
    """Where head slices land: the control-metadata namespace (PQ #884)."""
    from .joint_layer_quanta import control_directory

    return control_directory(output_root, "head-slices", metadata_root=metadata_root)


def head_slice_path(output_root: str, layer: int, *,
                    metadata_root: str | None = None) -> str:
    return (head_slice_directory(output_root, metadata_root=metadata_root)
            + f"/layer-{int(layer):03d}.json")


def roster_sha256(formats_by_qname) -> str:
    """One digest of the complete candidate roster, in canonical JSON."""
    return canonical_json_sha256(
        {name: list(formats) for name, formats in formats_by_qname.items()},
        where="Stage B candidate roster")


def _file_binding(role, binding, *, where):
    """``{role, path, sha256, bytes}`` for one digest-bound head file."""
    _refuse(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
            f"{where}: {role} is not a bound path/sha256 pair")
    path = Path(binding["path"])
    _refuse(path.is_absolute() and _hex64(binding["sha256"]),
            f"{where}: {role} binding is not an absolute path with a sha256")
    size = path.stat().st_size
    _refuse(size > 0, f"{where}: {role} at {path} is empty")
    return {"role": role, "path": str(path), "sha256": binding["sha256"],
            "bytes": int(size)}


def build_head_slices(*, config, plan_sha256, prepared, completion,
                      production_cache, data, layers, layer_of,
                      implementation_sha256):
    """Run the campaign-level head gates once and cut one slice per layer.

    ``data`` is the completed ``load_measured_anchor_input`` over the plan's
    inputs, read with the options a quantum used (existing renders required,
    payloads unverified, the plan's historical encoder allowance).
    ``prepared`` is the prepared completion's path/sha256 binding,
    ``completion`` its parsed document and ``production_cache`` its loaded
    PWC. Every check the quantum ran against the whole campaign runs here,
    once; the per-layer facts it used go into the slice. Returns
    ``{layer: slice}``.
    """
    from .tessera_joint_aura import (
        PREPARED_SCHEMA, _prepare_file_read_bound, _prepared_digest_recorded)

    where = "Stage B head slice producer"
    _same(completion.get("schema"), PREPARED_SCHEMA, f"{where}: prepared schema")
    _same(completion.get("status"), "complete", f"{where}: prepared completion")
    # Dev mode records a plan drift exactly as the quantum's own prepared
    # check does; certified mode refuses it.
    if not _prepared_digest_recorded("plan_sha256", completion.get("plan_sha256"),
                                     plan_sha256):
        _same(completion.get("plan_sha256"), plan_sha256, f"{where}: prepared plan")
    _same(config["model"], data.census["model"], f"{where}: requested source model")
    _same(data.census.get("attention_implementation"), "eager",
          f"{where}: qualified source attention")
    _refuse(data.unit_scope is None,
            f"{where}: a scoped read is not the campaign input")
    formats = {name: list(fmts) for name, fmts in data.formats_by_qname.items()}
    _same(completion["formats_by_qname"], formats,
          f"{where}: prepared exact candidate roster")
    _same(production_cache.metadata["inputs"], data.inputs,
          f"{where}: prepared source bindings")
    _same(dict(data.inputs), dict(config["inputs"]), f"{where}: plan inputs")
    # The quantum binds every render the PWC holds (``require_file_load_sha256``
    # demands the complete roster); the intake's measured cells are that set.
    cells = set(data.cells)
    _same(cells, set(production_cache.weights), f"{where}: PWC render roster")
    verified = production_cache.metadata["verified_cells"]
    for pair in cells:
        _refuse(_hex64(verified[pair].get("render_file_sha256")),
                f"{where}: {pair} has no prepared render digest")
    max_render = _prepare_file_read_bound(
        data, max_render_bytes=config["max_render_bytes"])
    draw = data.payload["provenance"]["hessian"]["calibration_identity"]
    calibration_identity = {name: draw.get(name)
                            for name in CALIBRATION_IDENTITY_FIELDS}
    encoder_seal = data.manifest["identity"]["encoder_source_sha256"]
    _refuse(_hex64(encoder_seal), f"{where}: checkpoint encoder seal")

    resource = None
    if config.get("stage_b_resource_policy") is not None:
        from .joint_stageb_resources import verify_policy as verify_resources
        policy = verify_resources(config["stage_b_resource_policy"])
        _same(config["max_gpu_bytes"], policy["limits"]["gpu_bytes"],
              f"{where}: device limit differs from policy")
        resource = {"binding": dict(config["stage_b_resource_policy"]),
                    "limits": dict(policy["limits"])}
    served = None
    if config.get("served_activation_policy") is not None:
        from .joint_served_activation import verify_policy as verify_served
        verify_served(config["served_activation_policy"])
        served = {"binding": dict(config["served_activation_policy"])}

    head_files = [_file_binding("prepared", prepared, where=where),
                  _file_binding("production_cache", completion["production_cache"],
                                where=where)]
    if served is not None:
        head_files.append(_file_binding("served_activation_policy",
                                        served["binding"], where=where))
    if config.get("source_identity_cache") is not None:
        head_files.append(_file_binding("source_identity_cache",
                                        config["source_identity_cache"], where=where))

    campaign = {
        "plan_sha256": plan_sha256,
        "prepared_sha256": prepared["sha256"],
        "production_pkl_sha256": completion["production_cache"]["sha256"],
        "inputs_sha256": canonical_json_sha256(dict(config["inputs"]),
                                               where="Stage B plan inputs"),
        "roster_sha256": roster_sha256(formats),
        "producer_implementation_sha256": implementation_sha256,
    }
    intake = {
        "source_model": data.census["model"],
        "attention_implementation": data.census["attention_implementation"],
        "calibration_identity": calibration_identity,
        "encoder_source_sha256": encoder_seal,
        "units": len(formats),
        "measured_cells": len(cells),
        # The walk reports one unit per roster unit (``progress_committed``);
        # the slice-mode head reports the same cumulative count once.
        "progress_units": len(formats),
        "max_render_file_bytes": int(max_render),
    }
    by_layer: dict[int, dict] = {}
    for name in formats:
        owner = layer_of(name)
        if owner is not None:
            by_layer.setdefault(int(owner), {})[name] = formats[name]
    slices = {}
    for layer in sorted({int(layer) for layer in layers}):
        members = by_layer.get(layer)
        _refuse(bool(members), f"{where}: the roster holds no unit for layer {layer}")
        slices[layer] = {
            "schema": HEAD_SLICE_SCHEMA,
            "layer": layer,
            "campaign": dict(campaign),
            "intake": {**intake, "layer_formats": {name: members[name]
                                                   for name in sorted(members)}},
            "resource_policy": resource,
            "served_activation_policy": served,
            "head_files": [dict(row) for row in head_files],
        }
    return slices


def verify_head_slice(head_slice, *, layer, prepared_sha256, config,
                      plan_sha256=None):
    """Refuse a slice for another layer, another preparation or another plan.

    The byte digest is the caller's (the record binds it); this checks what
    the bytes say against the quantum's own sealed inputs.
    """
    where = "Stage B head slice"
    _refuse(isinstance(head_slice, dict), f"{where} is not an object")
    _same(head_slice.get("schema"), HEAD_SLICE_SCHEMA, f"{where} schema")
    _refuse(head_slice.get("layer") == int(layer),
            f"{where} is for layer {head_slice.get('layer')!r}, not layer {layer}")
    campaign = head_slice.get("campaign") or {}
    # The slice's campaign bindings are run seals (PQ #1147): dev mode prints
    # each difference and reads the slice's own declared head files.
    _seal(campaign.get("prepared_sha256"), prepared_sha256, f"{where} prepared",
          message=f"{where} is bound to prepared {campaign.get('prepared_sha256')!r}, "
                  f"not the quantum's prepared {prepared_sha256}")
    if plan_sha256 is not None:
        _seal(campaign.get("plan_sha256"), plan_sha256, f"{where} plan",
              message=f"{where} is bound to plan {campaign.get('plan_sha256')!r}, "
                      f"not the quantum's plan {plan_sha256}")
    _seal(campaign.get("inputs_sha256"),
          canonical_json_sha256(dict(config["inputs"]), where="Stage B plan inputs"),
          f"{where} campaign inputs")
    files = {row.get("role"): row for row in head_slice.get("head_files") or []}
    _refuse(len(files) == len(head_slice.get("head_files") or [])
            and set(files) <= set(HEAD_FILE_ROLES),
            f"{where} head files repeat or name an unknown role")
    _seal(files.get("prepared", {}).get("sha256"), prepared_sha256,
          f"{where} prepared head file")
    _same(files.get("production_cache", {}).get("sha256"),
          campaign.get("production_pkl_sha256"), f"{where} production pickle")
    for role, key in (("served_activation_policy", "served_activation_policy"),
                      ("source_identity_cache", "source_identity_cache")):
        bound = config.get(key)
        row = files.get(role)
        _seal(None if row is None else {"path": row["path"], "sha256": row["sha256"]},
              None if bound is None else {"path": str(Path(bound["path"])),
                                          "sha256": bound["sha256"]},
              f"{where} {role}")
    resource = head_slice.get("resource_policy")
    _seal(None if resource is None else resource.get("binding"),
          config.get("stage_b_resource_policy"), f"{where} resource policy")
    served = head_slice.get("served_activation_policy")
    _seal(None if served is None else served.get("binding"),
          config.get("served_activation_policy"), f"{where} served activation policy")
    intake = head_slice.get("intake") or {}
    _seal(intake.get("source_model"), config["model"], f"{where} source model")
    _same(intake.get("attention_implementation"), "eager", f"{where} source attention")
    _refuse(isinstance(intake.get("layer_formats"), dict) and intake["layer_formats"],
            f"{where} names no layer roster")
    return files


def read_head_file(row, *, label):
    """One declared head file's bytes: off the stage under the tier policy.

    Without an active staged-tier policy (a direct run, a test) it is the
    same digest-checked pool read the quantum made before.
    """
    from .staged_tier_policy import policy_is_active
    from .staged_whole_file import read_staged_whole_file

    path = Path(row["path"])
    if policy_is_active():
        raw = read_staged_whole_file(path, row["sha256"], label=label)
    else:
        raw = path.read_bytes()
    _refuse(hashlib.sha256(raw).hexdigest() == row["sha256"],
            f"{label} at {path} does not hash to its bound digest")
    return raw


def read_quantum_head_slice(config, *, record, prepared, plan_sha256):
    """Read and check the record's bound head slice: the first head read.

    Runs before the device envelope is applied, because the envelope's
    verified limits come from here. Returns ``(slice, binding, files)``.
    """
    block = (record.get("executable_readset") or {}).get("head_slice")
    _refuse(isinstance(block, dict) and _hex64(block.get("sha256"))
            and isinstance(block.get("path"), str),
            f"quantum {record.get('quantum_id')!r} binds no head slice")
    _same(block.get("schema"), HEAD_SLICE_SCHEMA, "record head slice schema")
    raw = read_head_file(block, label="head-slice")
    head_slice = json.loads(raw)
    _refuse(head_slice_bytes(head_slice) == raw,
            "head slice file is not its canonical JSON")
    files = verify_head_slice(head_slice, layer=int(record["layer"]),
                              prepared_sha256=prepared["sha256"], config=config,
                              plan_sha256=plan_sha256)
    _same(files["prepared"]["path"], str(Path(prepared["path"])),
          "prepared head file path")
    return head_slice, dict(block), files


def head_slice_limits(head_slice):
    resource = head_slice.get("resource_policy")
    return None if resource is None else dict(resource["limits"])


def read_prepared_head(files):
    """The prepared completion, read once from its declared head entry."""
    return json.loads(read_head_file(files["prepared"], label="prepared"))


def load_quantum_head(config, *, record, head_slice, files, completion,
                      plan_sha256, implementation_sha256, reader_identity,
                      projection_backend, progress_phase=None):
    """The quantum's head intake from its sealed slice (PQ #1010).

    Replaces ``load_measured_anchor_input`` and the per-quantum policy
    re-derivations in ``run_layer_quantum``. ``completion`` is the parsed
    prepared completion (:func:`read_prepared_head`). Reads, in order: the
    calibration draw, the production pickle and -- when the
    plan binds them -- the served activation policy and the source-identity
    cache. Each is a declared head entry. Returns a namespace with the fields
    ``run_layer_quantum`` consumed from the old intake.
    """
    import pickle

    from .calibration_data import load_calibration_input
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_aura import (
        _pb_commit, _prepared_digest_recorded, check_prepared_completion,
        normalize_historical_encoder_reuse, require_prepared_digests,
        resolve_encoder_source_reuse)
    from . import tessera_campaign as tc

    intake = head_slice["intake"]
    # The policy re-derivations ran under the producer's package, so the
    # slice binds to this executing package exactly as the prepared
    # completion does: certified mode refuses a different implementation,
    # dev mode records it.
    producer = head_slice["campaign"].get("producer_implementation_sha256")
    if not _prepared_digest_recorded("implementation_sha256", producer,
                                     implementation_sha256):
        _same(producer, implementation_sha256, "head slice producer implementation")
    completion = check_prepared_completion(
        completion, plan_sha256=plan_sha256, implementation_sha256=implementation_sha256,
        reader_identity=reader_identity, projection_backend=projection_backend)
    _same(roster_sha256(completion["formats_by_qname"]),
          head_slice["campaign"]["roster_sha256"], "prepared candidate roster digest")
    formats_by_qname = {name: tuple(fmts)
                        for name, fmts in completion["formats_by_qname"].items()}
    _same({name: list(formats_by_qname.get(name, ()))
           for name in intake["layer_formats"]},
          intake["layer_formats"], "head slice layer roster")

    # The encoder seal is a fact about this executing package: compared
    # here, under the plan's allowance, exactly as the walk compared it.
    resolve_encoder_source_reuse(
        intake["encoder_source_sha256"],
        tc._checkpoint_identity_api().encoder_source_sha256(),
        normalize_historical_encoder_reuse(config.get("historical_encoder_reuse")),
        where="joint anchor checkpoint encoder source")

    execution = config["execution"]
    ids, calibration = load_calibration_input(
        config["calibration_input"]["path"],
        expected_sha256=config["calibration_input"]["sha256"],
        n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
    for name in CALIBRATION_IDENTITY_FIELDS:
        _same(calibration["provenance"].get(name),
              intake["calibration_identity"].get(name), f"original full draw {name}")

    _same(completion.get("schema"), "prismaquant.tessera_joint_aura.prepared.v3",
          "prepared schema")
    require_prepared_digests(completion, plan_sha256=plan_sha256,
                             implementation_sha256=implementation_sha256)
    _same(completion.get("calibration_input"), calibration, "prepared calibration")
    _same(completion.get("production_cache", {}).get("sha256"),
          files["production_cache"]["sha256"], "prepared production pickle")
    cache = pickle.loads(read_head_file(files["production_cache"],
                                        label="production-cache"))
    _refuse(isinstance(cache, ProductionWeightCache),
            "prepared cache is not ProductionWeightCache")
    # Run seals (PQ #1147): dev mode prints each and runs under the plan's
    # own policies.
    _seal(cache.metadata["inputs"], dict(config["inputs"]), "prepared source bindings")
    _seal(completion.get("stage_b_resource_policy"),
          config.get("stage_b_resource_policy"), "prepared Stage B resource policy")
    if config.get("stage_b_resource_policy") is not None:
        cache._joint_stage_b_resource_policy = dict(config["stage_b_resource_policy"])
    _seal(completion.get("served_activation_policy"),
          config.get("served_activation_policy"), "prepared served activation policy")
    if config.get("served_activation_policy") is not None:
        _refuse(record.get("catalog_extension") is not None,
                "served activation policy requires an explicit catalog extension")
        from .joint_served_activation import activate_verified_policy
        activate_verified_policy(
            cache, config["served_activation_policy"],
            json.loads(read_head_file(files["served_activation_policy"],
                                      label="served-activation-policy")))
    verified = cache.metadata["verified_cells"]
    expected = {pair: verified[pair]["render_file_sha256"] for pair in cache.weights}
    _same(len(expected), intake["measured_cells"], "prepared PWC render roster size")
    cache.require_file_load_sha256(
        expected, max_file_bytes=int(intake["max_render_file_bytes"]))
    identity_cache = None
    if "source_identity_cache" in files:
        identity_cache = read_head_file(files["source_identity_cache"],
                                        label="source-identity-cache")
    progress_units = int(intake["progress_units"])
    if progress_phase is not None:
        # The slice attests the units the prepare-time walk verified; the
        # count is reported once, at its final cumulative value, exactly as
        # a resumed walk reports its reverified prefix (#822).
        _pb_commit(progress_units, progress_phase)
    return SimpleNamespace(
        producer_implementation_sha256=producer, completion=completion, formats_by_qname=formats_by_qname,
        calibration_ids=ids, calibration=calibration, cache=cache,
        identity_cache_bytes=identity_cache,
        units=int(intake["units"]), measured_cells=int(intake["measured_cells"]),
        progress_units=progress_units,
        max_render_file_bytes=int(intake["max_render_file_bytes"]))
