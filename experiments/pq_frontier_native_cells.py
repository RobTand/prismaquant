#!/usr/bin/env python3
"""Qwen3-0.6B layer-0 native cells for the first prefill frontier (PQ #237).

One "cell" is one ``(unit, format)`` a served Qwen3-0.6B Tessera artifact
assigns at layer 0. This driver produces, for a declared set of cells, the
three things a ``prismaquant.measured_runtime_prices.v2`` row needs from the
PrismaQuant side, in one admitted GPU action:

``prepare``
    Render every cell through the production renderer (``pq237_joint_aura_streamed
    ._capture_and_render``: BF16 activation capture on the resident model, Tessera
    encode, wire retention), price every cell as a joint AURA row on the streamed
    model (``aura_cost.compute_aura_cost_streamed(joint_activation=True)``), and
    write per-cell native preparation inputs (``inputs.json``, ``request.json``,
    ``weight.tessera``, ``wire-record.json``, ``tensors.safetensors``) through
    ``native_operator_panel.prepare_native_inputs`` for Tessera's
    ``experiments/bench_native_operator.py``.
``freeze``
    Freeze every prepared cell's panel from its inputs and Tessera preflight against
    the exact ``joint.pkl`` bytes (``native_operator_panel.freeze_native_panel``).
``manifest``
    Write the receipt manifest ``prismaquant.native_receipt_table`` consumes.

The cost identity this produces is real (a fresh joint AURA probe on the real
model and the real rendered bytes), but the rendered bytes are NOT the served
artifacts' layer-0 wires: those were rendered by the campaign under its own
calibration. Timing does not read weight values, so a same-shape same-rung
operator prices the artifact's operator; the doc that cites these rows says so.

The native FP8 operator executes Tessera's unclipped ``fp8_per_token_dynamic``
contract; the joint rows and the reference QDQ must be priced under the same
policy, so this driver refuses unless ``PRISMAQUANT_PROD_ACT_SCALES=0`` is set
explicitly, as the routed MoE protocol does.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cell_dir(root, unit, fmt):
    return Path(root) / "cells" / f"{unit}__{fmt}"


def tensor_digest(tensor):
    """Exact bytes of one weight, with its shape and dtype, as one digest."""
    import torch

    flat = tensor.detach().to("cpu").contiguous()
    raw = flat.view(torch.uint8) if flat.dtype is not torch.uint8 else flat
    digest = hashlib.sha256(f"{tuple(flat.shape)}|{flat.dtype}|".encode())
    digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def compare_body_weights(runner, resident, *, max_resident_bytes):
    """Exact per-Linear equality between the resident renderer and the streamed producer.

    The joint rows are computed on the streamed model while the rendered bytes
    and the activation capture come from the resident one, so the two have to
    be the same model. *Weight equality is the exact statement of that.* A
    logit comparison is not: a whole-model forward and a layer-major streamed
    forward select different matmul kernels, and two BF16 reduction orders over
    28 layers do not agree bit for bit. Thresholding that difference would be a
    constant chosen rather than derived, so the difference is recorded as a
    diagnostic (``streamed_forward_profile``) and this is the gate.

    Units are compared in byte-bounded batches through the runner's own
    ``snapshot_selected_weights``, so a model far larger than the device still
    checks its whole body. A Linear the streamed runner does not carry as a body
    unit -- ``lm_head`` is one -- is named in ``outside_streamed_body`` rather
    than dropped, because a silently uncompared unit is the hole this check
    exists to close; the caller refuses if a unit it prices lands there.
    """
    import torch

    resolved, outside, sizes = [], [], {}
    modules = dict(runner.model.named_modules())
    for name in sorted(resident):
        try:
            runner.layer_index_for_qname(name)
        except Exception:
            outside.append(name)
            continue
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            outside.append(name)
            continue
        sizes[name] = module.weight.numel() * torch.empty((), dtype=runner.dtype).element_size()
        resolved.append(name)
    mismatched, compared, batch, used = [], 0, [], 0
    for name in resolved + [None]:
        if batch and (name is None or used + sizes[name] > max_resident_bytes):
            weights, _ = runner.snapshot_selected_weights(batch, max_resident_bytes=max_resident_bytes)
            for unit, value in weights.items():
                compared += 1
                if tensor_digest(value) != resident[unit]:
                    mismatched.append(unit)
            del weights
            batch, used = [], 0
        if name is not None:
            batch.append(name)
            used += sizes[name]
    return {"schema": "prismaquant.streamed_body_parity.v1", "units_compared": compared,
            "units_declared": len(resident), "mismatched_units": sorted(mismatched),
            "outside_streamed_body": sorted(outside), "max_resident_bytes": int(max_resident_bytes),
            "policy": "exact per-Linear source-weight equality, resident renderer vs streamed producer"}


def _require_unclipped_policy():
    from prismaquant.memory_management import env_truthy
    if env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        raise ValueError("native dense cells require explicit PRISMAQUANT_PROD_ACT_SCALES=0: "
                         "the native FP8 operator executes the unclipped per-token dynamic contract")


def prepare(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from experiments.pq237_joint_aura_streamed import _capture_and_render, expected_currency_bindings
    from prismaquant.aura_cost import compute_aura_cost_streamed
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from prismaquant.joint_aura import arithmetic_identity, identity_sha256, prefetch_joint_cache
    from prismaquant.model_profiles import detect_profile
    from prismaquant.native_operator_panel import EXECUTION, prepare_native_inputs
    from prismaquant.perturbed_x_cache import activation_cache_filename

    _require_unclipped_policy()
    if not torch.cuda.is_available():
        raise RuntimeError("native cell preparation requires an admitted GPU action")
    plan = json.loads(Path(args.cells).read_text())
    if not isinstance(plan, dict) or not plan or not all(isinstance(v, list) and v for v in plan.values()):
        raise ValueError("--cells must map unit -> nonempty list of formats")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["PRISMAQUANT_COST_UCB_Z"] = "0"
    os.environ["PRISMAQUANT_TESSERA_MENU"] = "research"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed_base)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.time()

    ids, calibration_receipt = load_calibration_input(args.calibration, expected_sha256=args.calibration_sha256,
                                                      n_samples=args.n_samples, seqlen=args.seqlen)
    provenance = calibration_receipt["provenance"]
    hessian_identity = {key: provenance[key] for key in ("text_sha256", "fit_tokens", "fit_ids_sha256")}
    hessian_identity.update({key: provenance[key] for key in ("source", "seed", "nsamples", "seqlen") if key in provenance})
    calibration = ids.to("cuda")
    identity = {"schema": "prismaquant.frontier_native_cells.v1", "model": args.model, "cells": plan,
                "calibration": calibration_receipt, "n_probes": args.n_probes, "seed_base": args.seed_base,
                "runtime_image": args.runtime_image, "numerics": {"atol": args.atol, "rtol": args.rtol},
                "prefill_rows": args.prefill_rows, "decode_rows": args.decode_rows,
                "activation_scale_policy": {"PRISMAQUANT_PROD_ACT_SCALES": os.environ.get("PRISMAQUANT_PROD_ACT_SCALES")},
                "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                "arithmetic": arithmetic_identity(torch.bfloat16), "start_unix": start,
                "scope": "layer-0 cells of served Qwen3-0.6B artifacts; fresh renders, real joint AURA rows; "
                         "no served timing, no promotion"}
    dump(out / "identity.json", identity)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    source_weights = {name: model.get_submodule(name).weight.detach().clone() for name in plan}
    # Every body Linear the renderer saw, by exact bytes. This is the property
    # the streamed producer has to share with the resident renderer, and it is
    # checked below once the streamed runner exists (see `streamed_body_parity`).
    resident_body = {name: tensor_digest(module.weight)
                     for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear)}
    cache, renders, capture, reference_logits = _capture_and_render(model, calibration, plan, out,
                                                                    hessian_identity=hessian_identity)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    identity["activation_capture"] = capture
    identity["resident_cache"] = prefetch_joint_cache(cache, plan, plan,
        max_resident_bytes=int((torch.cuda.mem_get_info()[0] / 1024**3 - args.min_free_gib) * 1024**3))
    formats = sorted({fmt for rows in plan.values() for fmt in rows})
    runner = build_streamed_causal_lm(args.model, device=torch.device("cuda"), dtype=torch.bfloat16,
        offload_folder=str(out / "streamed-offload"), profile=detect_profile(args.model),
        prefetch_workers=2, prefetch_lookahead=2, require_prefetched_residency=True)
    runner.model.config._attn_implementation = "eager"
    runner.model.config.use_cache = False
    try:
        model_identity = build_streamed_model_identity(runner, args.model,
                                                       identity_cache_path=out / "streamed-model-identity.json")
        protocol = {"plan": plan, "n_probes": args.n_probes, "seed_base": args.seed_base}
        expected_probe, expected_probe_sha256, expected = expected_currency_bindings(
            runner, calibration, protocol, model_identity, cache, renders)
        dump(out / "expected-currency-bindings.json", {"probe": expected_probe,
                                                        "operator_identity_sha256_by_candidate": expected})
        parity = compare_body_weights(runner, resident_body, max_resident_bytes=args.max_resident_bytes)
        parity["priced_units_outside_body"] = sorted(set(plan) & set(parity["outside_streamed_body"]))
        identity["streamed_body_parity"] = parity
        dump(out / "streamed-body-parity.json", parity)
        print("BODY-PARITY", json.dumps({k: v for k, v in parity.items() if k != "policy"},
                                        sort_keys=True), flush=True)
        if parity["mismatched_units"] or parity["priced_units_outside_body"]:
            raise RuntimeError(
                "streamed producer installs different source weights than the resident renderer "
                f"(mismatched {parity['mismatched_units']}, "
                f"priced-but-uncompared {parity['priced_units_outside_body']}); "
                "the joint rows would price another model")
        with torch.no_grad():
            actual_logits = runner(calibration).logits.detach().cpu()
        difference = (actual_logits.float() - reference_logits.float()).abs()
        forward = {"bit_exact": bool(torch.equal(actual_logits, reference_logits)),
                   "max_absolute_logit_difference": float(difference.max()),
                   "mean_absolute_logit_difference": float(difference.mean()),
                   "max_reference_logit_magnitude": float(reference_logits.float().abs().max()),
                   "top1_agreement": float((actual_logits.argmax(-1) == reference_logits.argmax(-1))
                                           .to(torch.float64).mean()),
                   "positions": int(reference_logits.shape[0] * reference_logits.shape[1]),
                   "scope": "DIAGNOSTIC, not a gate. A resident whole-model forward and a streamed "
                            "layer-major forward select different matmul kernels, so their BF16 "
                            "reduction orders differ and their logits cannot be bit-equal. The "
                            "refusal is the exact per-Linear weight equality in streamed_body_parity."}
        identity["streamed_forward_profile"] = forward
        dump(out / "teacher-parity.json", forward)
        print("FORWARD-PROFILE", json.dumps({k: v for k, v in forward.items() if k != "scope"},
                                            sort_keys=True), flush=True)
        del reference_logits, actual_logits, difference
        payload = compute_aura_cost_streamed(runner, calibration, formats,
            n_probes=args.n_probes, token_scope="causal", temperature=1.0, production_cache=cache,
            min_free_gib=args.min_free_gib, seed_base=args.seed_base, require_production_cache=True,
            joint_activation=True, formats_by_qname=plan, checkpoint_dir=out / "checkpoints",
            model_identity=model_identity)
        if payload["provenance"]["probe_identity_sha256"] != expected_probe_sha256:
            raise ValueError("streamed producer probe identity differs from frozen inputs")
        for name, rows in payload["costs"].items():
            for fmt, row in rows.items():
                if identity_sha256(row["joint_operator_identity"]) != expected[name][fmt]:
                    raise ValueError("streamed operator differs from independently recorded production render")
    finally:
        runner.shutdown()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    joint_path = out / "joint.pkl"
    with joint_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    joint_sha256 = sha256(joint_path)
    identity["joint_pkl_sha256"] = joint_sha256

    probe_request = {"n_probes": args.n_probes, "seed_base": args.seed_base, "token_scope": "causal",
                     "temperature": 1.0, "distribution": "rademacher", "normalization": "global_kl_fisher",
                     "source_model": model_identity["source"],
                     "source_shards": {Path(item["path"]).name: item["sha256"] for item in model_identity["shards"]}}
    prepared = {}
    for name, fmts in plan.items():
        rows = torch.load(out / "activations" / activation_cache_filename(name), weights_only=True)["inputs"]
        rows = rows.to(device="cuda", dtype=torch.bfloat16).contiguous()
        weight = source_weights[name].to("cuda")
        for fmt in fmts:
            render = renders[name][fmt]
            blob = (out / render["wire_path"]).read_bytes()
            inputs, tensors = prepare_native_inputs(cache, weight, rows, unit=name, format_name=fmt,
                calibration_receipt=calibration_receipt, wire_blob=blob, wire_record=render["wire_record"],
                encoding_identity=render["encoding_identity"], numerics={"atol": args.atol, "rtol": args.rtol},
                prefill_rows=args.prefill_rows, decode_rows=args.decode_rows,
                max_resident_bytes=args.max_resident_bytes)
            cell = cell_dir(out, name, fmt)
            cell.mkdir(parents=True, exist_ok=False)
            (cell / "weight.tessera").write_bytes(blob)
            dump(cell / "wire-record.json", render["wire_record"])
            save_file({key: tensor.detach().cpu().contiguous().clone() for key, tensor in tensors.items()},
                      str(cell / "tensors.safetensors"))
            inputs["artifacts"] = {n: sha256(cell / n) for n in ("weight.tessera", "wire-record.json", "tensors.safetensors")}
            inputs["runtime_image"] = args.runtime_image
            inputs["probe_request"] = probe_request
            inputs["joint_pkl_sha256"] = joint_sha256
            inputs["joint_operator_identity_sha256_expected"] = expected[name][fmt]
            dump(cell / "inputs.json", inputs)
            dump(cell / "request.json", {"schema": "tessera.native_dense_request.v1", "unit": name,
                "format": fmt, "wire_path": "weight.tessera", "wire_record_path": "wire-record.json",
                "tensors_path": "tensors.safetensors", "runtime_image": args.runtime_image,
                "input_global_scale": inputs["activation"]["input_global_scale"], "execution": dict(EXECUTION)})
            prepared[f"{name}@{fmt}"] = {"dir": str(cell.relative_to(out)), "shape": inputs["shape"],
                                         "inputs_sha256": sha256(cell / "inputs.json"),
                                         "request_sha256": sha256(cell / "request.json"),
                                         "blob_bytes": render["blob_bytes"]}
            print("PREPARED", name, fmt, inputs["shape"], render["blob_bytes"], flush=True)
            del tensors
        del rows, weight
    identity.update(prepared_cells=prepared, elapsed_seconds=time.time() - start, completed=True,
                    peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    dump(out / "identity.json", identity)
    dump(out / "receipt.json", {"completed": True, "joint_pkl_sha256": joint_sha256,
        "artifacts": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                      if p.is_file() and p.name != "receipt.json" and "streamed-offload" not in p.parts
                      and "checkpoints" not in p.parts and "activations" not in p.parts}})
    print("COMPLETE", out, "cells", len(prepared), "elapsed_s", round(time.time() - start, 1), flush=True)


def freeze(args):
    from prismaquant.native_operator_panel import freeze_native_panel
    root = Path(args.out)
    joint_path = Path(args.joint)
    cost_sha256 = sha256(joint_path)
    with joint_path.open("rb") as stream:
        cost = pickle.load(stream)
    frozen, skipped = {}, {}
    for cell in sorted((root / "cells").iterdir()):
        inputs_path, preflight_path, panel_path = cell / "inputs.json", cell / "preflight.json", cell / "panel.json"
        if not preflight_path.is_file():
            skipped[cell.name] = "no preflight"
            continue
        if panel_path.exists():
            skipped[cell.name] = "panel already frozen"
            continue
        inputs = json.loads(inputs_path.read_text())
        if inputs.get("joint_pkl_sha256") != cost_sha256:
            raise ValueError(f"{cell.name}: inputs were prepared against another joint.pkl")
        preflight = json.loads(preflight_path.read_text())
        row = cost["costs"][inputs["unit"]][inputs["format"]]
        panel = freeze_native_panel(inputs, preflight, row, cost_sha256=cost_sha256)
        dump(panel_path, panel)
        frozen[cell.name] = sha256(panel_path)
        print("FROZEN", cell.name, frozen[cell.name], flush=True)
    dump(root / "freeze-summary.json", {"joint_pkl_sha256": cost_sha256, "frozen": frozen, "skipped": skipped})
    if not frozen and not skipped:
        raise ValueError("no prepared cells found")


def manifest(args):
    root = Path(args.out)
    bindings, missing = [], {}
    for cell in sorted((root / "cells").iterdir()):
        inputs = json.loads((cell / "inputs.json").read_text())
        receipt, trace, panel = cell / "receipt.json", cell / "receipt.json.memory.json", cell / "panel.json"
        absent = [p.name for p in (receipt, trace, panel) if not p.is_file()]
        if absent:
            missing[cell.name] = absent
            continue
        # One run per measured cell by default: the loaded-package identity and
        # the post-run core audit are facts about the process that produced this
        # receipt, and a shared run id would bind one process's evidence to
        # receipts other processes produced.
        bindings.append({"unit": inputs["unit"], "format": inputs["format"],
                         "run_id": cell.name if args.run_id is None else args.run_id,
                         "panel": str(panel.resolve()), "receipt": str(receipt.resolve()),
                         "memory_trace": str(trace.resolve())})
    dump(Path(args.manifest), bindings)
    dump(Path(args.manifest).with_suffix(".missing.json"), missing)
    print(json.dumps({"bindings": len(bindings), "missing": missing}, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--calibration-sha256", required=True)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--cells", required=True, help="JSON: unit -> [format, ...]")
    p.add_argument("--out", required=True)
    p.add_argument("--n-probes", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=267000)
    p.add_argument("--runtime-image", required=True)
    p.add_argument("--min-free-gib", type=float, default=4.0)
    p.add_argument("--prefill-rows", type=int, default=512)
    p.add_argument("--decode-rows", type=int, default=1)
    p.add_argument("--atol", type=float, default=0.015625)
    p.add_argument("--rtol", type=float, default=0.015625)
    p.add_argument("--max-resident-bytes", type=int, default=1 << 30)
    p.set_defaults(func=prepare)
    f = sub.add_parser("freeze")
    f.add_argument("--out", required=True)
    f.add_argument("--joint", required=True)
    f.set_defaults(func=freeze)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    m.add_argument("--run-id", default=None,
                   help="one shared runtime run id; omit to name each cell's own run")
    m.add_argument("--manifest", required=True)
    m.set_defaults(func=manifest)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
