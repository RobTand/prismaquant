#!/usr/bin/env python3
"""GLM MTP layer calibration capture, in two PrismaBuild actions (PQ #1290).

``--phase final-hidden`` reads the last backbone layer's input for every
calibration sequence (Stage A's boundary entries, through a hash-bound
manifest), runs that layer on the plan's streamed BF16 source, then the
model's collapse and final norm, and writes each sequence's post-final-norm
hidden state as an exact entry under ``--out``.

``--phase capture`` loads the MTP layer from the checkpoint and runs the body
campaign's collector over it on those hidden states. It writes the MTP census
to ``--census-out`` and the capture, in the canonical format, under ``--out``.

Both phases read every source shard through the canonical capture's source
owner, so each shard is authenticated before a tensor from it is used. See
``prismaquant/glm_mtp_capture.py`` for what each phase binds.

Run inside the campaign container (the plan's source derivative needs it)::

    python3 -m tools.glm_mtp_capture --phase final-hidden \\
        --plan PLAN --plan-sha256 SHA --boundaries MANIFEST --boundaries-sha256 SHA \\
        --out DIR --offload-folder DIR
    python3 -m tools.glm_mtp_capture --phase capture \\
        --plan PLAN --plan-sha256 SHA --prepared PREPARED --prepared-sha256 SHA \\
        --final-hidden MANIFEST --final-hidden-sha256 SHA \\
        --out DIR --census-out PATH
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import closing
from pathlib import Path


def _capture_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", required=True, choices=("final-hidden", "capture"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--boundaries")
    parser.add_argument("--boundaries-sha256")
    parser.add_argument("--prepared")
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--final-hidden")
    parser.add_argument("--final-hidden-sha256")
    parser.add_argument("--out", required=True)
    parser.add_argument("--census-out")
    parser.add_argument("--offload-folder")
    parser.add_argument("--read-ahead-mb", type=int, default=256)
    parser.add_argument("--device", default="cuda",
                        help="the capture phase's device; phase 1 runs on the plan's runner")
    args = parser.parse_args(argv)
    needed = {"final-hidden": ("boundaries", "boundaries_sha256", "offload_folder"),
              "capture": ("prepared", "prepared_sha256", "final_hidden",
                          "final_hidden_sha256", "census_out")}[args.phase]
    missing = [f"--{name.replace('_', '-')}" for name in needed if getattr(args, name) is None]
    if missing:
        parser.error(f"--phase {args.phase} needs {' '.join(missing)}")
    return args


def _runtime():
    import importlib.metadata

    import torch

    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                transformers=importlib.metadata.version("transformers"))


def _peak_memory():
    import torch

    if not torch.cuda.is_available():
        return None
    return {"max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved())}


def _plan_inputs(args):
    """The plan, the calibration draw and the canonical capture's source owner."""
    from prismaquant import glm_mtp_capture as cap
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.calibration_data import load_calibration_input

    plan, plan_sha256 = cap.read_bound_json(args.plan, args.plan_sha256)
    execution = plan["execution"]
    ids, calibration = load_calibration_input(
        plan["calibration_input"]["path"], expected_sha256=plan["calibration_input"]["sha256"],
        n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
    canonical = plan["canonical_capture"]
    manifest = cc.require_capture_contract(canonical["path"], expected_sha256=canonical["sha256"])
    owner = cc.authenticate_selected_capture_source(
        plan["inputs"]["census"]["path"], canonical["path"], expected_sha256=canonical["sha256"],
        model=plan["model"], max_act_rows=manifest["identity"]["max_act_rows"],
        attention_implementation=manifest["identity"]["attention_implementation"],
        release_read_pages=True)
    plan_ref = {"path": str(Path(args.plan).resolve()), "sha256": plan_sha256}
    return plan, plan_ref, ids, calibration, manifest, owner


def final_hidden_phase(args):
    import torch

    from prismaquant import glm_mtp_capture as cap
    from prismaquant.joint_cost_quantum import build_quantum_source_runner

    started = time.monotonic()
    plan, plan_ref, ids, calibration, _manifest, owner = _plan_inputs(args)
    try:
        boundaries, boundaries_sha256 = cap.read_bound_json(args.boundaries, args.boundaries_sha256)
        if boundaries.get("calibration_sha256") != calibration["calibration_sha256"]:
            raise RuntimeError("the boundary entries were computed on another calibration draw")
        run_identity = {
            "plan": plan_ref,
            "boundaries": {"path": str(Path(args.boundaries).resolve()),
                           "sha256": boundaries_sha256},
            "calibration_input": plan["calibration_input"],
            "calibration_sha256": calibration["calibration_sha256"],
            "canonical_capture": plan["canonical_capture"],
            "census": plan["inputs"]["census"],
        }
        runner = build_quantum_source_runner(plan, offload_folder=args.offload_folder,
                                             source_authentication=owner)
        try:
            layer = runner.num_layers - 1
            run_identity["layer"] = layer
            runner.context.begin_source_initialization_audit()
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=runner.require_prefetched_residency,
                                   prefetch_following=False)
            installed = time.monotonic()
            session = cap.final_hidden_session(run_identity)
            records, head_check = cap.write_final_hidden(
                runner, ids,
                cap.ordered_boundary_records(boundaries, int(ids.shape[0]), layer=layer),
                boundary_session=boundaries["session"], layer=layer, out_dir=args.out,
                session=session, read_ahead_bytes=int(args.read_ahead_mb) << 20,
                head=runner._head())
            witness = runner.context.source_selected_initialization_witness([layer])
        finally:
            runner.shutdown()
        reference = cap.publish_final_hidden(
            args.out, session=session, records=records, layer=layer, inputs=run_identity,
            source_witness=witness, source_authentication=owner.receipt(), head_check=head_check)
    finally:
        owner.close()
    census, _ = cap.read_bound_json(plan["inputs"]["census"]["path"],
                                    plan["inputs"]["census"]["sha256"])
    n = len(head_check)
    return {
        "phase": "final-hidden", "manifest": reference, "sequences": n,
        "seconds": {"install": installed - started, "total": time.monotonic() - started},
        "cuda_peak": _peak_memory(),
        "head_check": {"mean_top1": sum(r["top1"] for r in head_check) / n,
                       "mean_nll": sum(r["nll"] for r in head_check) / n,
                       "min_top1": min(r["top1"] for r in head_check)},
        # The witness names the same checkpoint source map as the census's
        # complete traversal; reported, not required.
        "source_map_matches_census": (
            witness["source_map_sha256"] == census["model_load_contract"]["source_map_sha256"]),
        "torch": torch.__version__,
    }


def capture_phase(args):
    import torch
    from transformers import AutoConfig

    from prismaquant import glm_mtp, glm_mtp_capture as cap
    from prismaquant.model_profiles import detect_profile

    started = time.monotonic()
    plan, plan_ref, ids, calibration, manifest, owner = _plan_inputs(args)
    try:
        final, _ = cap.read_bound_json(args.final_hidden, args.final_hidden_sha256)
        if final["inputs"]["calibration_sha256"] != calibration["calibration_sha256"]:
            raise RuntimeError("the final hidden states were computed on another calibration draw")
        final_ref = {"schema": cap.FINAL_HIDDEN_SCHEMA,
                     "path": str(Path(args.final_hidden).resolve()),
                     "sha256": args.final_hidden_sha256}
        prepared, _ = cap.read_bound_json(args.prepared, args.prepared_sha256)
        body_layer = int(final["layer"])
        # The dispatch the body's last MoE layer ran (``source_execution``).
        dispatch = prepared["source_execution"]["modules"][
            f"model.language_model.layers.{body_layer}.mlp.experts"]["experts"]
        base, _ = cap.read_bound_json(plan["inputs"]["census"]["path"],
                                      plan["inputs"]["census"]["sha256"])
        config = AutoConfig.from_pretrained(plan["model"])
        text_config = getattr(config, "text_config", config)
        text_config._attn_implementation = "eager"
        profile = detect_profile(plan["model"])
        device = torch.device(args.device)
        layer, receipt = glm_mtp.load_mtp_layer(
            plan["model"], text_config, profile=profile, dtype=torch.bfloat16, device=device,
            experts_implementation=dispatch, source_authentication=owner)
        weight, _shard = glm_mtp.read_checkpoint_tensor(
            plan["model"], "model.language_model.embed_tokens.weight",
            source_authentication=owner)
        embed = torch.nn.Embedding.from_pretrained(
            weight.to(device=device, dtype=torch.bfloat16), freeze=True)
        del weight
        loaded = time.monotonic()
        contract = glm_mtp.mtp_layer_initialization_contract(layer, receipt,
                                                             input_manifest=final_ref)
        wrapper = glm_mtp.MtpCheckpointModel(layer)
        units = glm_mtp.mtp_priced_units(wrapper, profile)
        read, stream = cap.final_hidden_stream(final, int(ids.shape[0]),
                                               read_ahead_bytes=int(args.read_ahead_mb) << 20)
        with closing(stream):
            rows, hessians, counts, maxima = cap.capture_mtp_layer(
                wrapper, embed, ids, read, units=units, profile=profile, device=device,
                max_act_rows=manifest["identity"]["max_act_rows"])
        captured = time.monotonic()
        census = cap.mtp_census(
            base_census=base, base_census_ref=plan["inputs"]["census"],
            canonical_capture_ref=plan["canonical_capture"], final_hidden_ref=final_ref,
            layer=layer.layer_idx, units=units, counts=counts, max_abs=maxima,
            groups=cap.mtp_anchor_groups(wrapper, units, profile),
            model_load_contract=contract,
            attention_implementation=layer.config._attn_implementation,
            capture_runtime=_runtime())
        identity, census_sha256, sealed = cap.publish_mtp_capture(
            args.out, census=census, census_path=args.census_out, source_authentication=owner,
            calibration=manifest["identity"]["calibration"],
            max_act_rows=manifest["identity"]["max_act_rows"], rows=rows, hessians=hessians,
            counts=counts, max_abs=maxima,
            completed_contract=glm_mtp.mtp_layer_initialization_contract(
                layer, receipt, input_manifest=final_ref))
        authentication = owner.receipt()
    finally:
        owner.close()
    routed = [n for n in units if ".mlp.experts." in n]
    return {
        "phase": "capture", "capture": sealed,
        "census": {"path": str(Path(args.census_out).resolve()), "sha256": census_sha256},
        "units": len(units), "routed_units": len(routed),
        "experts_implementation": dispatch,
        "seconds": {"load": loaded - started, "capture": captured - loaded,
                    "total": time.monotonic() - started},
        "cuda_peak": _peak_memory(),
        "routed_rows": {"min": min(counts[n] for n in routed),
                        "max": max(counts[n] for n in routed)},
        "source_authentication": authentication,
        "plan": plan_ref, "identity_units": len(identity["units"]),
    }


def main(argv=None):
    args = _capture_arguments(sys.argv[1:] if argv is None else argv)
    summary = (final_hidden_phase if args.phase == "final-hidden" else capture_phase)(args)
    raw = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report = Path(args.out) / f"{args.phase}-run.json"
    report.write_text(raw)
    print(raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
